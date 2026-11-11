//! `Linear`: an EXL3 linear served straight from the trellis (decode GEMV, 1..16+ rows).
//!
//! forward = exl3_rot_in (x -> xh fp16, rotated) + exl3_gemv_bB_wW (trellis decode + mma + output rotation).
//! No host syncs, fixed workspaces: graph-capturable.

use fe_cuda::{DevPtr, DeviceBuffer, Function, Stream};

use crate::gpu::{DeviceExl3, Exl3View, Kernels};
use crate::st::Checkpoint;
use crate::{Error, Result};

/// Activation element type. The `*Add` forms are outputs only: y += x @ W (fused residual add; bf16 adds in
/// f32 and rounds once).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    F32,
    Bf16Add,
    F32Add,
}

impl Dtype {
    fn flag(self) -> u32 {
        match self {
            Dtype::Bf16 => 0,
            Dtype::F32 => 1,
            Dtype::Bf16Add => 2,
            Dtype::F32Add => 3,
        }
    }
}

/// A launch plan: K splits (programs a 128-column block) and warps a program fix the summation order;
/// `ver` picks the kernel form (1: x as the mma A operand, 2: the same with two k steps prefetched; 8: W^T as A,
/// 8-row groups; 9: v8 with two k steps prefetched; 1/2 and 8/9 give identical bits) and `occ` the programs
/// an SM the registers are bounded for (0 = unbounded).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub sk: u32,
    pub wk: u32,
    pub ver: u32,
    pub occ: u32,
}

impl Plan {
    /// Parse "sk,wk[,ver[,occ]]".
    pub fn parse(s: &str) -> Option<Plan> {
        let v: Vec<u32> = s.split(',').map(|x| x.trim().parse().ok()).collect::<Option<_>>()?;
        Some(Plan {
            sk: *v.first()?,
            wk: *v.get(1)?,
            ver: v.get(2).copied().unwrap_or(1),
            occ: v.get(3).copied().unwrap_or(0),
        })
    }

    /// The kernel for `rows` (ver 8 has 8-, 16-, 32- and 48-row-pass forms, bit-identical a row).
    pub fn kernel_name(&self, bits: u32, rows: u32) -> String {
        let o = if self.occ > 0 { format!("_o{}", self.occ) } else { String::new() };
        match self.ver {
            // 40- / 48-row passes (groups 4.. accumulate in shared memory), 24-row passes: never launch-bounded,
            // depth 1
            8 | 9 if matches!(Self::groups(rows), 3 | 5 | 6) => {
                format!("exl3_gemv8_b{bits}_w{}_g{}", self.wk, Self::groups(rows))
            }
            // the 32-row form needs its registers (bounding it spills): never launch-bounded, depth 1
            8 | 9 if Self::groups(rows) == 4 => format!("exl3_gemv8_b{bits}_w{}_g4", self.wk),
            8 => format!("exl3_gemv8_b{bits}_w{}_g{}{o}", self.wk, Self::groups(rows)),
            // v8 with two k steps prefetched (same bits as v8)
            9 => format!("exl3_gemv8d_b{bits}_w{}_g{}{o}", self.wk, Self::groups(rows)),
            2 => format!("exl3_gemv_d2_b{bits}_w{}{o}", self.wk),
            _ => format!("exl3_gemv_b{bits}_w{}{o}", self.wk),
        }
    }

    /// 8-row groups a pass of the v8 kernels.
    pub fn groups(rows: u32) -> u32 {
        match rows {
            0..=8 => 1,
            9..=16 => 2,
            17..=24 => 3,
            25..=32 => 4,
            33..=40 => 5,
            _ => 6,
        }
    }

    /// Dynamic shared memory of one launch: the warps' partial sums, plus for g5 / g6 the shared accumulators of
    /// groups 4.. (WK x 4 KB a group).
    pub fn smem(&self, _bits: u32, rows: u32) -> u32 {
        let g = Self::groups(rows);
        if matches!(self.ver, 8 | 9) && g >= 5 {
            self.wk * 8 * 128 * 4 + self.wk * (g - 4) * 8 * 32 * 16
        } else {
            self.wk * rows.min(8) * 128 * 4
        }
    }
}

/// TensorFold's rule: wide projections (>= 64 column blocks) run 8 warps and no split; narrower ones split K
/// until there are `blocks` programs (at most 64 splits); then back off while K does not divide or a warp
/// would get fewer than `min_tiles` k tiles.
pub fn plan(k: u32, n: u32, blocks: u32, min_tiles: u32) -> Plan {
    let (sk, wk) = tf_plan(k, n, blocks, min_tiles);
    Plan { sk, wk, ver: 1, occ: 0 }
}

/// TensorFold's (sk, wk).
pub fn tf_plan(k: u32, n: u32, blocks: u32, min_tiles: u32) -> (u32, u32) {
    let kt = k / 16;
    let nb = n / 128;
    let mut sk = 1;
    let mut wk = if nb >= 64 { 8 } else { 4 };
    if nb < 64 {
        while nb * sk < blocks && sk < 64 {
            sk *= 2;
        }
    }
    while (wk > 4 || sk > 1) && (kt % (sk * wk) != 0 || kt / (sk * wk) < min_tiles) {
        if wk > 4 {
            wk = 4;
        } else if sk > 1 {
            sk /= 2;
        } else {
            break;
        }
    }
    (sk, wk)
}

/// The default program target: 4 programs an SM (GB10: 48 SMs -> 192).
pub fn default_blocks(sm_count: u32) -> u32 {
    4 * sm_count
}

/// Scratch shared by every Linear on one stream (calls on one stream run one at a time).
pub struct Workspace {
    pub max_rows: u32,
    /// fp16 rotated inputs [max_rows][max_k]
    pub xh: DeviceBuffer<u16>,
    /// fp32 split partials [sk][rows][n]
    pub z: DeviceBuffer<f32>,
    /// split arrival counters [passes (8-row)][n / 128], zero at rest (each call re-arms them)
    pub counters: DeviceBuffer<u32>,
}

impl Workspace {
    /// Sized for `linears` at up to `max_rows` rows.
    pub fn new(linears: &[&Linear], max_rows: u32) -> Result<Self> {
        let max_k = linears.iter().map(|l| l.k).max().unwrap_or(128) as usize;
        let max_z = linears.iter().map(|l| if l.plan.sk > 1 { l.plan.sk * l.n } else { 0 }).max().unwrap_or(0) as usize;
        let max_nb = linears.iter().map(|l| l.n / 128).max().unwrap_or(1) as usize;
        let passes = (max_rows as usize).div_ceil(8);
        Ok(Self {
            max_rows,
            xh: DeviceBuffer::alloc(max_rows as usize * max_k)?,
            z: DeviceBuffer::alloc((max_z * max_rows as usize).max(4))?,
            counters: DeviceBuffer::zeroed(passes * max_nb)?,
        })
    }
}

pub struct Linear {
    pub name: String,
    pub k: u32,
    pub n: u32,
    pub bits: u32,
    pub plan: Plan,
    /// Device pointers of the weight (owned by `owned`, or by the caller for `from_view`).
    pub view: Exl3View,
    pub owned: Option<DeviceExl3>,
    /// the GEMV for 1..8 rows, 9..16, 17..24, 25..32 (and 49+ in 32-row passes), 33..40, 41..48 (v1: one kernel)
    gemv: [Function; 6],
    /// dynamic smem of each form
    smem: [u32; 6],
}

impl Linear {
    /// Load `base` from the checkpoint (strips layout); `blocks` is the program target (see `default_blocks`).
    pub fn load(kern: &Kernels, ck: &Checkpoint, base: &str, blocks: u32) -> Result<Self> {
        Self::new(kern, DeviceExl3::upload_strips(&crate::Exl3Tensor::load(ck, base)?)?, blocks)
    }

    /// The tuned plan for the shape (crates/fe-exl3/src/plans.rs, GB10) or the default rule.
    /// FE_EXL3_UNTUNED=1 forces the default rule.
    /// Tensor-parallel column slice (see `Exl3Tensor::slice_cols`): y[rows][sum of range widths].
    pub fn load_cols(kern: &Kernels, ck: &Checkpoint, base: &str, ranges: &[(usize, usize)], blocks: u32) -> Result<Self> {
        let t = crate::Exl3Tensor::load(ck, base)?.slice_cols(ranges)?;
        Self::new(kern, DeviceExl3::upload_strips(&t)?, blocks)
    }

    /// Tensor-parallel row slice (see `Exl3Tensor::slice_rows`): x is [rows][k1 - k0] (this rank's input
    /// columns), y is the rank's partial [rows][n]; sum the ranks' partials (f32 out, or let one rank add the
    /// residual with `Dtype::*Add` before the reduction).
    pub fn load_rows(kern: &Kernels, ck: &Checkpoint, base: &str, k0: usize, k1: usize, blocks: u32) -> Result<Self> {
        let t = crate::Exl3Tensor::load(ck, base)?.slice_rows(k0, k1)?;
        Self::new(kern, DeviceExl3::upload_strips(&t)?, blocks)
    }

    /// This rank's part of `base` under the canonical Qwen3.5 split (`crate::tp::qwen35_split`).
    pub fn load_tp(kern: &Kernels, ck: &Checkpoint, base: &str, rank: usize, world: usize, blocks: u32) -> Result<(Self, crate::tp::Split)> {
        let t = crate::Exl3Tensor::load(ck, base)?;
        let split = crate::tp::qwen35_split(base, t.k, t.n, rank, world);
        let t = match &split {
            crate::tp::Split::Cols(r) => t.slice_cols(r)?,
            crate::tp::Split::Rows(a, b) => t.slice_rows(*a, *b)?,
            crate::tp::Split::Replicate => t,
        };
        Ok((Self::new(kern, DeviceExl3::upload_strips(&t)?, blocks)?, split))
    }

    pub fn new(kern: &Kernels, w: DeviceExl3, blocks: u32) -> Result<Self> {
        let p = Self::plan_for(w.view.k, w.view.n, w.view.bits, blocks);
        Self::with_plan(kern, w, p)
    }

    pub fn plan_for(k: u32, n: u32, bits: u32, blocks: u32) -> Plan {
        let tuned = if std::env::var("FE_EXL3_UNTUNED").is_ok() { None } else { {
            // the device's table by its SM count (blocks = 4 an SM); FE_EXL3_TABLE=gb10 forces GB10's (A/B runs)
            let sms = if std::env::var("FE_EXL3_TABLE").is_ok_and(|v| v == "gb10") { 48 } else { blocks / 4 };
            crate::plans::tuned(k, n, bits, sms)
        } };
        let mut p = tuned.unwrap_or_else(|| plan(k, n, blocks, 8));
        // FE_EXL3_NOSPLIT=1 (or FE_EXL3_SPLIT=0) forces one K split (no counters / partial reduction; debugging).
        // The tuned splits are the default again: the suspected split-K race did not reproduce (exl3-stress 1.28M
        // calls, hybrid repro identical in 5 pairs) and one split costs ~4% TP1 / 11% TP2 decode.
        let nosplit = std::env::var("FE_EXL3_NOSPLIT").is_ok_and(|v| v == "1")
            || std::env::var("FE_EXL3_SPLIT").is_ok_and(|v| v == "0");
        if nosplit {
            p.sk = 1;
        }
        p
    }

    pub fn with_plan(kern: &Kernels, w: DeviceExl3, p: Plan) -> Result<Self> {
        let mut l = Self::from_view(kern, &w.name, w.view, p)?;
        l.owned = Some(w);
        Ok(l)
    }

    /// A Linear over device buffers the caller owns (and keeps alive).
    pub fn from_view(kern: &Kernels, name: &str, view: Exl3View, p: Plan) -> Result<Self> {
        let (k, n, bits) = (view.k, view.n, view.bits);
        if k % 128 != 0 || n % 128 != 0 || (k / 16) % (p.sk * p.wk) != 0 {
            return Err(Error::msg(format!("{name}: {k}x{n} does not fit plan {p:?}")));
        }
        // the row count each form is built for; FE_EXL3_G3=0 runs 17..24 rows as 32-row passes, FE_EXL3_G6=0 runs
        // 33..48 rows as 32-row passes (as before those forms)
        let g3 = !std::env::var("FE_EXL3_G3").is_ok_and(|v| v == "0");
        let g6 = !std::env::var("FE_EXL3_G6").is_ok_and(|v| v == "0");
        let rows = [1, 16, if g3 { 17 } else { 25 }, 25, if g6 { 33 } else { 25 }, if g6 { 41 } else { 25 }];
        let mut gemv = Vec::with_capacity(6);
        let mut smem = [0u32; 6];
        for (i, &r) in rows.iter().enumerate() {
            let f = kern.module.function(&p.kernel_name(bits, r))?;
            smem[i] = p.smem(bits, r);
            // the g5 / g6 forms: always raise the limit (g6 at 4 warps needs exactly 48 KB dynamic, over the default with
            // its static smem); g1-g4 stay within the default (8 rows of partial sums at most)
            if i >= 4 {
                f.set_max_dynamic_smem(smem[i])?;
            }
            gemv.push(f);
        }
        let gemv: [Function; 6] = gemv.try_into().map_err(|_| Error::msg("gemv forms"))?;
        Ok(Self { name: name.to_string(), k, n, bits, plan: p, view, owned: None, gemv, smem })
    }

    /// Trellis bytes streamed per call.
    pub fn weight_bytes(&self) -> usize {
        self.k as usize * self.n as usize * self.bits as usize / 8
    }

    /// y[rows][n] = x[rows][k] @ W. x bf16 or f32, y bf16 or f32 (row-major, contiguous).
    pub fn forward(
        &self,
        kern: &Kernels,
        s: &Stream,
        x: DevPtr,
        x_dtype: Dtype,
        y: DevPtr,
        y_dtype: Dtype,
        rows: u32,
        ws: &Workspace,
    ) -> Result<()> {
        self.rotate_input(kern, s, x, x_dtype, rows, ws)?;
        self.gemv_rotated(s, ws.xh.ptr(), y, y_dtype, rows, ws)
    }

    /// xh = fp16(H(x * suh) / sqrt(128)) into the workspace.
    pub fn rotate_input(&self, kern: &Kernels, s: &Stream, x: DevPtr, x_dtype: Dtype, rows: u32, ws: &Workspace) -> Result<()> {
        if rows == 0 || rows > ws.max_rows {
            return Err(Error::msg(format!("{}: rows {rows} (workspace {})", self.name, ws.max_rows)));
        }
        unsafe {
            crate::gpu::pdl::launch(s, &kern.rot_in, (self.k.div_ceil(512), rows), 128, 0, (x, x_dtype.flag() & 1, self.view.suh, ws.xh.ptr(), self.k))?;
        }
        Ok(())
    }

    /// The trellis GEMV on already rotated fp16 inputs `xh` [rows][k].
    pub fn gemv_rotated(&self, s: &Stream, xh: DevPtr, y: DevPtr, y_dtype: Dtype, rows: u32, ws: &Workspace) -> Result<()> {
        if rows == 0 || rows > ws.max_rows || (self.plan.sk > 1 && (ws.z.len() as u64) < self.plan.sk as u64 * rows as u64 * self.n as u64) {
            return Err(Error::msg(format!("{}: rows {rows} do not fit the workspace (max {}, z {})", self.name, ws.max_rows, ws.z.len())));
        }
        if (ws.counters.len() as u32) < rows.div_ceil(8) * (self.n / 128) {
            return Err(Error::msg(format!("{}: workspace counters too small for {rows} rows", self.name)));
        }
        let p = self.plan;
        // one pass up to 48 rows (g1 / g2 / g3 / g4 / g5 / g6 by 8-row groups); 49+ rows: 32-row passes (g4). Per TP2
        // rank over all linears g6 takes 48 rows in 43.8 ms vs 56.7 for g4's 32 + 16, but 64 rows in 64.8 (48 + 16)
        // vs 62.1 for g4's 32 + 32
        let form = if rows > 48 { 3 } else { (Plan::groups(rows) as usize - 1).min(5) };
        // g1-g4: the partial sums of min(rows, 8) rows (1-row decode keeps its small footprint); g5 / g6: fixed
        let smem = if form >= 4 { self.smem[form] } else { p.smem(self.bits, rows.min(32)) };
        let (stride_k, stride_nb) = self.view.strides();
        unsafe {
            crate::gpu::pdl::launch(
                s,
                &self.gemv[form],
                (self.n / 128 * p.sk, 1),
                p.wk * 32,
                smem,
                (
                    xh,
                    self.view.trellis,
                    self.view.svh,
                    y,
                    ws.z.ptr(),
                    ws.counters.ptr(),
                    self.n,
                    self.k,
                    rows,
                    p.sk,
                    y_dtype.flag(),
                    stride_k,
                    stride_nb,
                ),
            )?;
        }
        Ok(())
    }
}
