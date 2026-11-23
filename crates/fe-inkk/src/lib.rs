//! Inkling-2-Mini kernel API (kernels/k-ink). Device pointers in, device pointers out; no allocation inside the
//! calls except `MoeScratch::new`. Everything is stream-ordered and graph-capturable.
//!
//! Routed experts, decode path (1-16 rows): [`Kern::moe_decode`] runs gate|up + softcapped SwiGLU, down and the
//! fixed-order weighted sum straight from the release bytes, [`ExpertLayer`] holds the per-layer pointers.
use anyhow::{ensure, Context, Result};
use fe_cuda::{DevPtr, DeviceBuffer, Function, Module, Stream};

pub mod cpu;
pub mod pdl;
pub mod slots;
pub static K_INK_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/k_ink.cubin"));

pub const HID: usize = 2304;
pub const FFN: usize = 1152;
pub const TOPK: usize = 8;
pub const N_EXPERTS: usize = 128;
/// dynamic smem of the grouped tensor-core GEMMs
pub const GEMM_SMEM: u32 = 4 * (2 * 8192 + 2 * 1024);
/// dynamic smem of the decode-size (<= 16 rows) swapped GEMMs
pub const GEMM_SMEM_SW: u32 = 4 * (64 * 64 + 64 * 8 + 1024 + 256);
/// smem of the decode-size down GEMMs with the quantization fused in (NVFP4 / MXFP8)
pub const GEMM_SMEM_SWQ_FP4: u32 = 4 * 4608 + 9 * 1024 + 9 * 128 + 64;
pub const GEMM_SMEM_SWQ_FP8: u32 = 3 * 4608 + 18 * 1024 + 18 * 32 + 64;
pub const NONE: u32 = 0xffff_ffff;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExpertFmt {
    /// layers 0-19: u8 code pairs + e4m3 scale per 16 + per-expert f32 factor
    Nvfp4,
    /// layers 20-23: e4m3 values + ue8m0 scale per 32
    Mxfp8,
}

/// One layer's routed experts, exactly as stored in the release (tensor names under `model.llm.layers.N.mlp.experts.`):
/// `w13_weight` / `w13_weight_scale`, `w2_weight` / `w2_weight_scale`, and for NVFP4 the per-expert factors
/// (`w13_weight_amax / w13_weight_outer_scale_denominator`, the same for w2: use [`expert_factors`]).
#[derive(Clone, Copy)]
pub struct ExpertLayer {
    pub fmt: ExpertFmt,
    pub w13: DevPtr,
    pub s13: DevPtr,
    /// f32 [128]; null for MXFP8
    pub f13: DevPtr,
    pub w2: DevPtr,
    pub s2: DevPtr,
    pub f2: DevPtr,
}

/// `amax[e] / denominator` in f32 (what the reference dequantizer multiplies every value of expert e with).
pub fn expert_factors(amax: &[f32], denominator: f32) -> Vec<f32> {
    amax.iter().map(|a| a / denominator).collect()
}

/// Scratch for the decode MoE: pre-down activations and per-pick outputs of up to `max_rows` rows.
pub struct MoeScratch {
    pub h: DeviceBuffer<u16>,
    pub y: DeviceBuffer<u16>,
    pub max_rows: usize,
}

impl MoeScratch {
    pub fn new(max_rows: usize) -> Result<MoeScratch> {
        Ok(MoeScratch {
            h: DeviceBuffer::alloc(max_rows * TOPK * FFN)?,
            y: DeviceBuffer::alloc(max_rows * TOPK * HID)?,
            max_rows,
        })
    }
}

pub struct Kern {
    pub module: Module,
    gu_fp4: Function,
    down_fp4: Function,
    gu_fp8: Function,
    down_fp8: Function,
    combine: Function,
    plan_count: Function,
    plan_build: Function,
    plan_perm: Function,
    plan_small: Function,
    combine_pos: Function,
    moe_tail: Function,
    router_fused: Function,
    sconv_resid: Function,
    headclip2: Function,
    kv_conv_norm: Function,
    kv_conv_dyn_norm: Function,
    comb_gate: Function,
    blockres2: Function,
    gemm_z: Function,
    quant4: Function,
    gu_tc: Function,
    down_tc: Function,
    repack_w13: Function,
    retile_sf: Function,
    fp8_quant: Function,
    fp8_dequant: Function,
    fp8_gemv: [Function; 4],
    quant8: Function,
    gu_tc8: Function,
    down_tc8: Function,
    repack_w13_8: Function,
    retile_sf8: Function,
    rq_amax_part: Function,
    rq_amax_fin: Function,
    rq_quant: Function,
    retile_codes: Function,
    gu_sw: Function,
    gu8_sw: Function,
    down_swq: Function,
    down8_swq: Function,
}

impl Kern {
    /// After the CUDA context exists; one per process (a Module is bound to the context it was loaded in).
    pub fn load() -> Result<Kern> {
        let module = Module::load(K_INK_CUBIN)?;
        let gu_tc = module.function("ink_gemm_gu_nvfp4")?;
        let down_tc = module.function("ink_gemm_down_nvfp4")?;
        gu_tc.set_max_dynamic_smem(GEMM_SMEM)?;
        down_tc.set_max_dynamic_smem(GEMM_SMEM)?;
        let gu_tc8 = module.function("ink_gemm_gu_mxfp8")?;
        let down_tc8 = module.function("ink_gemm_down_mxfp8")?;
        gu_tc8.set_max_dynamic_smem(GEMM_SMEM)?;
        let down_swq = module.function("ink_gemm_down_nvfp4_swq")?;
        let down8_swq = module.function("ink_gemm_down_mxfp8_swq")?;
        down_swq.set_max_dynamic_smem(GEMM_SMEM_SWQ_FP4)?;
        down8_swq.set_max_dynamic_smem(GEMM_SMEM_SWQ_FP8)?;
        down_tc8.set_max_dynamic_smem(GEMM_SMEM)?;
        Ok(Kern {
            plan_count: module.function("ink_plan_count")?,
            plan_build: module.function("ink_plan_build")?,
            plan_perm: module.function("ink_plan_perm")?,
            plan_small: module.function("ink_plan_small")?,
            combine_pos: module.function("ink_moe_combine_pos")?,
            moe_tail: module.function("ink_moe_tail")?,
            router_fused: module.function("ink_router_fused")?,
            sconv_resid: module.function("ink_sconv_resid")?,
            headclip2: module.function("ink_headclip2")?,
            kv_conv_norm: module.function("ink_kv_conv_norm")?,
            kv_conv_dyn_norm: module.function("ink_kv_conv_dyn_norm")?,
            comb_gate: module.function("ink_attn_comb_gate")?,
            blockres2: module.function("ink_blockres2")?,
            gemm_z: module.function("ink_gemm_z")?,
            quant4: module.function("ink_quant_nvfp4")?,
            repack_w13: module.function("ink_repack_w13_nvfp4")?,
            retile_sf: module.function("ink_retile_sf")?,
            fp8_quant: module.function("ink_fp8_quant_rows")?,
            fp8_dequant: module.function("ink_dequant_fp8")?,
            fp8_gemv: [module.function("ink_gemv_fp8_r1")?, module.function("ink_gemv_fp8_r4")?,
                module.function("ink_gemv_fp8_r8")?, module.function("ink_gemv_fp8_r16")?],
            gu_tc,
            down_tc,
            gu_tc8,
            down_tc8,
            quant8: module.function("ink_quant_mxfp8")?,
            repack_w13_8: module.function("ink_repack_w13_mxfp8")?,
            retile_sf8: module.function("ink_retile_sf8")?,
            rq_amax_part: module.function("ink_rq_amax_part")?,
            rq_amax_fin: module.function("ink_rq_amax_fin")?,
            rq_quant: module.function("ink_rq_quant")?,
            retile_codes: module.function("ink_retile_codes")?,
            gu_sw: module.function("ink_gemm_gu_nvfp4_sw")?,
            gu8_sw: module.function("ink_gemm_gu_mxfp8_sw")?,
            down_swq,
            down8_swq,
            gu_fp4: module.function("ink_moe_gu_nvfp4")?,
            down_fp4: module.function("ink_moe_down_nvfp4")?,
            gu_fp8: module.function("ink_moe_gu_mxfp8")?,
            down_fp8: module.function("ink_moe_down_mxfp8")?,
            combine: module.function("ink_moe_combine")?,
            module,
        })
    }

    /// Routed-expert MoE for `rows` <= `scr.max_rows` rows (decode / short prompts):
    /// out[r][:] (bf16, row stride `ldo`) = bf16( sum_k w[r][k] * down_{e[r][k]}(softcapped_swiglu(gate_up_{e[r][k]}(x[r][:]))) ),
    /// the sum in ascending expert-id order, f32. `x` = bf16 [rows][ldx] (ldx % 8 == 0, 16-byte aligned rows),
    /// `ids` = i32 [rows * 8] (an id < 0 skips the pick), `wts` = f32 [rows * 8] (routing weights x routed_scaling_factor).
    /// Each (row, pick) is computed independently with a fixed summation order: a row's bits never depend on `rows`.
    pub fn moe_decode(&self, s: &Stream, l: &ExpertLayer, scr: &MoeScratch, x: DevPtr, ldx: usize, ids: DevPtr,
        wts: DevPtr, rows: usize, out: DevPtr, ldo: usize) -> Result<()> {
        ensure!(rows >= 1 && rows <= scr.max_rows, "moe_decode: rows {rows} > scratch {}", scr.max_rows);
        ensure!(ldx % 8 == 0, "moe_decode: ldx {ldx} % 8");
        let pairs = (rows * TOPK) as u32;
        unsafe {
            match l.fmt {
                ExpertFmt::Nvfp4 => {
                    pdl::launch(s, &self.gu_fp4, ((FFN / 64) as u32, pairs), 256, 0,
                        (scr.h.ptr(), x, ldx as u32, ids, l.w13, l.s13, l.f13))?;
                    pdl::launch(s, &self.down_fp4, ((HID / 64) as u32, pairs), 256, 0,
                        (scr.y.ptr(), scr.h.ptr(), ids, l.w2, l.s2, l.f2))?;
                }
                ExpertFmt::Mxfp8 => {
                    pdl::launch(s, &self.gu_fp8, ((FFN / 64) as u32, pairs), 256, 0,
                        (scr.h.ptr(), x, ldx as u32, ids, l.w13, l.s13))?;
                    pdl::launch(s, &self.down_fp8, ((HID / 64) as u32, pairs), 256, 0,
                        (scr.y.ptr(), scr.h.ptr(), ids, l.w2, l.s2))?;
                }
            }
            pdl::launch(s, &self.combine, ((HID as u32).div_ceil(256), rows as u32), 256, 0,
                (out, ldo as u32, scr.y.ptr(), ids, wts))?;
        }
        Ok(())
    }
}


/// NVFP4 experts of one layer in the tensor-core layout: w13 rows interleaved by 8 (gate f..f+7, up f..f+7, ...) with
/// tiled scales, w2 codes as stored with tiled scales. Built once at load from the release bytes ([`Kern::repack_nvfp4`]).
pub struct PrefillLayer {
    pub fmt: ExpertFmt,
    /// the device buffers the tables point into after the repack: w13 codes, w13 scales, w2 codes, w2 scales (kept alive here)
    pub keep: Vec<DeviceBuffer<u8>>,
    /// per-expert pointer tables the GEMM kernels read (u64 [128] each): w13 codes, w13 scales, w2 codes, w2 scales. After the repack
    /// they point into the contiguous buffers above; a slot-cache engine (G16: coldest experts in host RAM) can re-point an entry at
    /// a slot buffer holding the same four tensors of that expert (sizes: [`PrefillLayer::expert_bytes`]) before the layer runs.
    pub tw13: DeviceBuffer<u64>,
    pub ts13: DeviceBuffer<u64>,
    pub tw2: DeviceBuffer<u64>,
    pub ts2: DeviceBuffer<u64>,
    /// `w13_weight_amax` / `w2_weight_amax` (f32 [128], raw) and 1 / (1536 * outer_scale_denominator) for each
    pub amax13: DevPtr,
    pub amax2: DevPtr,
    pub rcp13: f32,
    pub rcp2: f32,
    /// per-expert use counts (u32 [128]) accumulated by every moe call (the slot cache's adaptation input)
    pub use_cnt: DeviceBuffer<u32>,
}

/// Scratch of [`Kern::moe_prefill`] for chunks of up to `max_rows` rows.
impl PrefillLayer {
    /// An empty shell with the same scalars (a slot layer is built around it before the real layer is moved in).
    pub(crate) fn shell(pl: &PrefillLayer) -> PrefillLayer {
        let empty = || DeviceBuffer::<u64>::zeroed(1).unwrap();
        PrefillLayer { fmt: pl.fmt, keep: vec![], tw13: empty(), ts13: empty(), tw2: empty(), ts2: empty(),
            amax13: pl.amax13, amax2: pl.amax2, rcp13: pl.rcp13, rcp2: pl.rcp2, use_cnt: DeviceBuffer::zeroed(1).unwrap() }
    }
    /// (w13 codes, w13 scales, w2 codes, w2 scales) bytes of ONE expert in the tile-contiguous layout.
    pub fn expert_bytes(fmt: ExpertFmt) -> [usize; 4] {
        match fmt {
            ExpertFmt::Nvfp4 => [2304 * 1152, 331776, 2304 * 576, 165888],
            ExpertFmt::Mxfp8 => [2304 * 2304, 165888, 2304 * 1152, 82944],
        }
    }
    /// Point expert `e` at four device buffers (host-side update: not capturable; engines that promote experts inside a graph write
    /// the tables from a kernel instead, `self.tw13.ptr()` etc.).
    pub fn set_expert(&mut self, e: usize, w13: DevPtr, s13: DevPtr, w2: DevPtr, s2: DevPtr) -> Result<()> {
        ensure!(e < N_EXPERTS, "expert {e}");
        for (t, p) in [(&mut self.tw13, w13), (&mut self.ts13, s13), (&mut self.tw2, w2), (&mut self.ts2, s2)] {
            fe_cuda::memcpy_htod(t.at(e), &[p.0])?;
        }
        Ok(())
    }
    fn tables(fmt: ExpertFmt, w13: &DeviceBuffer<u8>, s13: &DeviceBuffer<u8>, w2: &DeviceBuffer<u8>, s2: &DeviceBuffer<u8>)
        -> Result<[DeviceBuffer<u64>; 4]> {
        let sz = Self::expert_bytes(fmt);
        let mk = |b: &DeviceBuffer<u8>, n: usize| -> Result<DeviceBuffer<u64>> {
            let v: Vec<u64> = (0..N_EXPERTS).map(|e| b.ptr().0 + (e * n) as u64).collect();
            Ok(DeviceBuffer::from_slice(&v)?)
        };
        Ok([mk(w13, sz[0])?, mk(s13, sz[1])?, mk(w2, sz[2])?, mk(s2, sz[3])?])
    }
}

pub struct PrefillScratch {
    pub max_rows: usize,
    pub maxpos: usize,
    pub maxtiles: usize,
    counts: DeviceBuffer<u32>,
    goff: DeviceBuffer<u32>,
    pub tab: DeviceBuffer<u32>,
    pub pair_of_pos: DeviceBuffer<u32>,
    pub pos_of_pair: DeviceBuffer<u32>,
    pub a_codes: DeviceBuffer<u8>,
    pub a_sf: DeviceBuffer<u8>,
    pub a_row: DeviceBuffer<f32>,
    pub h: DeviceBuffer<u16>,
    pub a2_codes: DeviceBuffer<u8>,
    pub a2_sf: DeviceBuffer<u8>,
    pub a2_row: DeviceBuffer<f32>,
    pub y: DeviceBuffer<u16>,
}

impl PrefillScratch {
    pub fn new(max_rows: usize) -> Result<PrefillScratch> {
        let npairs = max_rows * TOPK;
        let maxpos = (npairs + 128 * 127).div_ceil(128) * 128;
        let maxtiles = npairs.div_ceil(128) + 128;
        Ok(PrefillScratch {
            max_rows,
            maxpos,
            maxtiles,
            counts: DeviceBuffer::zeroed(N_EXPERTS)?,
            goff: DeviceBuffer::zeroed(N_EXPERTS + 1)?,
            tab: DeviceBuffer::zeroed(1 + 3 * maxtiles)?,
            pair_of_pos: DeviceBuffer::zeroed(maxpos)?,
            pos_of_pair: DeviceBuffer::zeroed(npairs)?,
            a_codes: DeviceBuffer::zeroed(maxpos * HID)?,
            a_sf: DeviceBuffer::zeroed(maxpos * HID / 16)?,
            a_row: DeviceBuffer::zeroed(maxpos)?,
            h: DeviceBuffer::zeroed(maxpos * FFN)?,
            a2_codes: DeviceBuffer::zeroed(maxpos * FFN)?,
            a2_sf: DeviceBuffer::zeroed(maxpos * FFN / 16)?,
            a2_row: DeviceBuffer::zeroed(maxpos)?,
            y: DeviceBuffer::zeroed(maxpos * HID)?,
        })
    }
}

impl Kern {
    /// Repack one NVFP4 layer (release layout, device) into the tensor-core layout. Needs ~0.4 GB for the new w13 +
    /// scales; the release w13 / s13 buffers can be freed afterwards (the scalar decode path needs them: it reads the
    /// release layout).
    pub fn repack_nvfp4(&self, s: &Stream, l: &ExpertLayer, amax13: DevPtr, amax2: DevPtr, denom13: f32, denom2: f32)
        -> Result<PrefillLayer> {
        ensure!(l.fmt == ExpertFmt::Nvfp4, "repack_nvfp4: not an NVFP4 layer");
        let w13 = DeviceBuffer::<u8>::alloc(N_EXPERTS * 2 * FFN * (HID / 2))?;
        let s13 = DeviceBuffer::<u8>::alloc(N_EXPERTS * 2 * FFN * (HID / 16))?;
        let s2 = DeviceBuffer::<u8>::alloc(N_EXPERTS * HID * (FFN / 16))?;
        let w2 = DeviceBuffer::<u8>::alloc(N_EXPERTS * HID * (FFN / 2))?;
        unsafe {
            s.launch(&self.retile_codes, (HID as u32, N_EXPERTS as u32), 128, 0, (w2.ptr(), l.w2, HID as u32, (FFN / 2) as u32))?;
            s.launch(&self.repack_w13, (2 * FFN as u32, N_EXPERTS as u32), 128, 0, (w13.ptr(), s13.ptr(), l.w13, l.s13))?;
            s.launch(&self.retile_sf, (HID as u32, N_EXPERTS as u32), 128, 0,
                (s2.ptr(), l.s2, HID as u32, (FFN / 16) as u32))?;
        }
        let [tw13, ts13, tw2, ts2] = PrefillLayer::tables(ExpertFmt::Nvfp4, &w13, &s13, &w2, &s2)?;
        Ok(PrefillLayer { fmt: ExpertFmt::Nvfp4, tw13, ts13, tw2, ts2, keep: vec![w13, s13, w2, s2], use_cnt: DeviceBuffer::zeroed(N_EXPERTS)?, amax13, amax2, rcp13: 1.0 / (1536.0 * denom13), rcp2: 1.0 / (1536.0 * denom2) })
    }

    /// MXFP8 twin of [`Kern::repack_nvfp4`] (layers 20-23): w13 values rows interleaved by 8 + tiled scales (0.7 GB new
    /// buffer a layer), w2 scales tiled, w2 values shared with the release buffer.
    pub fn repack_mxfp8(&self, s: &Stream, l: &ExpertLayer) -> Result<PrefillLayer> {
        ensure!(l.fmt == ExpertFmt::Mxfp8, "repack_mxfp8: not an MXFP8 layer");
        let w13 = DeviceBuffer::<u8>::alloc(N_EXPERTS * 2 * FFN * HID)?;
        let s13 = DeviceBuffer::<u8>::alloc(N_EXPERTS * 2 * FFN * (HID / 32))?;
        let s2 = DeviceBuffer::<u8>::alloc(N_EXPERTS * HID * (FFN / 32))?;
        let w2 = DeviceBuffer::<u8>::alloc(N_EXPERTS * HID * FFN)?;
        unsafe {
            s.launch(&self.retile_codes, (HID as u32, N_EXPERTS as u32), 128, 0, (w2.ptr(), l.w2, HID as u32, FFN as u32))?;
            s.launch(&self.repack_w13_8, (2 * FFN as u32, N_EXPERTS as u32), 128, 0, (w13.ptr(), s13.ptr(), l.w13, l.s13))?;
            s.launch(&self.retile_sf8, (HID as u32, N_EXPERTS as u32), 128, 0,
                (s2.ptr(), l.s2, HID as u32, (FFN / 32) as u32))?;
        }
        let null = DevPtr(0);
        let [tw13, ts13, tw2, ts2] = PrefillLayer::tables(ExpertFmt::Mxfp8, &w13, &s13, &w2, &s2)?;
        Ok(PrefillLayer { fmt: ExpertFmt::Mxfp8, tw13, ts13, tw2, ts2, keep: vec![w13, s13, w2, s2], use_cnt: DeviceBuffer::zeroed(N_EXPERTS)?, amax13: null, amax2: null, rcp13: 0.0, rcp2: 0.0 })
    }

    /// Routed-expert MoE for a chunk of `rows` <= `scr.max_rows` rows on tensor cores (NVFP4 and MXFP8 layers), the release's
    /// native rule: x is quantized once per token to NVFP4 (per-token amax, per-16 e4m3
    /// scale chosen by `fouroversix_mae`), gate|up accumulates exactly and gets both outer scales in f32, the softcapped
    /// SwiGLU is multiplied by the routing weight `wts` BEFORE its output is quantized the same way for the W2 GEMM, and
    /// the 8 slots are summed in slot order (f32, from 0) with no further weighting. `wts` = routing weight times the
    /// routed scaling factor. Picks with id < 0 are skipped.
    pub fn moe_prefill(&self, s: &Stream, l: &PrefillLayer, scr: &PrefillScratch, x: DevPtr, ldx: usize, ids: DevPtr,
        wts: DevPtr, rows: usize, out: DevPtr, ldo: usize) -> Result<()> {
        self.moe_prefill_timed(s, l, scr, x, ldx, ids, wts, rows, out, ldo, None, None)
    }

    /// Decode form of the layer tail: as [`Kern::moe_prefill`] (rows <= 16) but the slot sum, the MLP short convolution
    /// (`conv_w` bf16 [2304][4], `state` f32 [3][2304], advanced by `rows` steps) and the residual add run in ONE kernel:
    /// `tail.out` (bf16 [rows][2304]) = bf16(tail.resid (f32 [rows][2304]) + conv(moe sum)). Bits equal to combine + sconv + add.
    pub fn moe_prefill_tail(&self, s: &Stream, l: &PrefillLayer, scr: &PrefillScratch, x: DevPtr, ldx: usize, ids: DevPtr,
        wts: DevPtr, rows: usize, tail: &MoeTail) -> Result<()> {
        ensure!(rows <= 16, "moe_prefill_tail: rows {rows} > 16");
        self.moe_prefill_timed(s, l, scr, x, ldx, ids, wts, rows, DevPtr(0), 0, None, Some(tail))
    }

    /// Attention-side tail of a decode layer application (rows <= 16): static short convolution of `x` (bf16 [rows][2304], conv
    /// weights `w` bf16 [2304][4], f32 `state` [3][2304]) + residual: `resid` (f32) = conv + (f32(hs) if `add_hs`), and when
    /// `copy_dst` is non-null also `copy_dst[row * ldd + c] = hs[row][c]`. Bits of ink_sconv + ink_resid (+ ink_copy_rows).
    pub fn sconv_resid(&self, s: &Stream, resid: DevPtr, x: DevPtr, w: DevPtr, state: DevPtr, hs: DevPtr, add_hs: bool,
        copy_dst: DevPtr, ldd: usize, rows: usize, spec: bool) -> Result<()> {
        unsafe {
            pdl::launch(s, &self.sconv_resid, (HID as u32).div_ceil(256), 256, 0,
                (resid, x, w, state, hs, add_hs as u32, copy_dst, ldd as u32, rows as u32, spec as u32))?;
        }
        Ok(())
    }

    /// Router front for decode rows: post-attention RMS norm (f32 `src`, bf16 weight `w`) written to `xm` (bf16) and the router
    /// logits `xm . wg[e]` (f32 [rows][128], wg bf16 [128][2304]) in one launch, bits of `ink_rms` + `ink_router_logits`.
    pub fn router_front(&self, s: &Stream, xm: DevPtr, logits: DevPtr, src: DevPtr, w: DevPtr, wg: DevPtr, rows: usize) -> Result<()> {
        unsafe { pdl::launch(s, &self.router_fused, (rows as u32, 16u32), 256, 0, (xm, logits, src, w, wg))? };
        Ok(())
    }

    /// As [`Kern::moe_prefill`]; with `times` set, records an event after each stage and fills (stage, ms) pairs
    /// (synchronises at the end).
    pub fn moe_prefill_timed(&self, s: &Stream, l: &PrefillLayer, scr: &PrefillScratch, x: DevPtr, ldx: usize,
        ids: DevPtr, wts: DevPtr, rows: usize, out: DevPtr, ldo: usize,
        mut times: Option<&mut Vec<(&'static str, f32)>>, tail: Option<&MoeTail>) -> Result<()> {
        let mut ev: Vec<(&'static str, fe_cuda::Event)> = vec![];
        macro_rules! mark {
            ($n:expr) => {
                if times.is_some() {
                    let e = fe_cuda::Event::new()?;
                    s.record(&e)?;
                    ev.push(($n, e));
                }
            };
        }
        mark!("start");
        ensure!(rows >= 1 && rows <= scr.max_rows, "moe_prefill: rows {rows} > scratch {}", scr.max_rows);
        ensure!(ldx % 8 == 0, "moe_prefill: ldx {ldx} % 8");
        let npairs = (rows * TOPK) as u32;
        let maxtiles = (npairs as usize).min(N_EXPERTS + (npairs as usize).div_ceil(128));
        unsafe {
            if npairs <= 1024 {
                pdl::launch(s, &self.plan_small, 1u32, 1024, 0,
                    (scr.tab.ptr(), scr.goff.ptr(), scr.pair_of_pos.ptr(), scr.pos_of_pair.ptr(), ids, npairs, l.use_cnt.ptr()))?;
            } else {
                pdl::launch(s, &self.plan_count, 128u32, 256, 0, (scr.counts.ptr(), ids, npairs, l.use_cnt.ptr()))?;
                pdl::launch(s, &self.plan_build, 1u32, 128, 0, (scr.tab.ptr(), scr.goff.ptr(), scr.counts.ptr()))?;
                pdl::launch(s, &self.plan_perm, 128u32, 256, 0,
                    (scr.pair_of_pos.ptr(), scr.pos_of_pair.ptr(), ids, scr.goff.ptr(), npairs))?;
            }
            mark!("plan");
            let grid_q = npairs.div_ceil(8);
            let grid_g = ((HID / 128) as u32, maxtiles as u32);
            // decode-size kernels: 64 weight rows (4 warps) a CTA
            let grid_sw = ((HID / 64) as u32, maxtiles as u32);
            if l.fmt == ExpertFmt::Nvfp4 {
                pdl::launch(s, &self.quant4, grid_q, 256, 0,
                    (scr.a_codes.ptr(), scr.a_sf.ptr(), scr.a_row.ptr(), x, ldx as u32, scr.pos_of_pair.ptr(), 1u32,
                     npairs, HID as u32))?;
                mark!("quant x");
                let sw = rows <= 16;
                pdl::launch(s, if sw { &self.gu_sw } else { &self.gu_tc }, if sw { grid_sw } else { grid_g }, if sw { 128 } else { 256 }, if sw { GEMM_SMEM_SW } else { GEMM_SMEM },
                    (scr.h.ptr(), scr.a_codes.ptr(), scr.a_sf.ptr(), scr.a_row.ptr(), l.tw13.ptr(), l.ts13.ptr(), l.amax13,
                     l.rcp13, wts, scr.pair_of_pos.ptr(), scr.tab.ptr()))?;
                mark!("gate|up gemm");
                if sw {
                    mark!("quant h (fused)");
                    pdl::launch(s, &self.down_swq, grid_sw, 128, GEMM_SMEM_SWQ_FP4,
                        (scr.y.ptr(), scr.h.ptr(), l.tw2.ptr(), l.ts2.ptr(), l.amax2, l.rcp2, scr.tab.ptr()))?;
                } else {
                    pdl::launch(s, &self.quant4, grid_q, 256, 0,
                        (scr.a2_codes.ptr(), scr.a2_sf.ptr(), scr.a2_row.ptr(), scr.h.ptr(), FFN as u32,
                         scr.pos_of_pair.ptr(), 0u32, npairs, FFN as u32))?;
                    mark!("quant h");
                    pdl::launch(s, &self.down_tc, grid_g, 256, GEMM_SMEM,
                        (scr.y.ptr(), scr.a2_codes.ptr(), scr.a2_sf.ptr(), scr.a2_row.ptr(), l.tw2.ptr(), l.ts2.ptr(), l.amax2,
                         l.rcp2, scr.tab.ptr()))?;
                }
                mark!("down gemm");
            } else {
                pdl::launch(s, &self.quant8, grid_q, 256, 0,
                    (scr.a_codes.ptr(), scr.a_sf.ptr(), x, ldx as u32, scr.pos_of_pair.ptr(), 1u32, npairs, HID as u32))?;
                mark!("quant x");
                let sw = rows <= 16;
                pdl::launch(s, if sw { &self.gu8_sw } else { &self.gu_tc8 }, if sw { grid_sw } else { grid_g }, if sw { 128 } else { 256 }, if sw { GEMM_SMEM_SW } else { GEMM_SMEM },
                    (scr.h.ptr(), scr.a_codes.ptr(), scr.a_sf.ptr(), l.tw13.ptr(), l.ts13.ptr(), wts, scr.pair_of_pos.ptr(),
                     scr.tab.ptr()))?;
                mark!("gate|up gemm");
                if sw {
                    mark!("quant h (fused)");
                    pdl::launch(s, &self.down8_swq, grid_sw, 128, GEMM_SMEM_SWQ_FP8,
                        (scr.y.ptr(), scr.h.ptr(), l.tw2.ptr(), l.ts2.ptr(), scr.tab.ptr()))?;
                } else {
                    pdl::launch(s, &self.quant8, grid_q, 256, 0,
                        (scr.a2_codes.ptr(), scr.a2_sf.ptr(), scr.h.ptr(), FFN as u32, scr.pos_of_pair.ptr(), 0u32, npairs,
                         FFN as u32))?;
                    mark!("quant h");
                    pdl::launch(s, &self.down_tc8, grid_g, 256, GEMM_SMEM,
                        (scr.y.ptr(), scr.a2_codes.ptr(), scr.a2_sf.ptr(), l.tw2.ptr(), l.ts2.ptr(), scr.tab.ptr()))?;
                }
                mark!("down gemm");
            }
            if let Some(t) = tail {
                pdl::launch(s, &self.moe_tail, (HID as u32).div_ceil(256), 256, 0,
                    (t.out, t.resid, scr.y.ptr(), scr.pos_of_pair.ptr(), ids, t.conv_w, t.state, rows as u32, t.spec as u32))?;
            } else {
                pdl::launch(s, &self.combine_pos, ((HID as u32).div_ceil(256), rows as u32), 256, 0,
                    (out, ldo as u32, scr.y.ptr(), scr.pos_of_pair.ptr(), ids, wts, 1u32))?;
            }
        }
        mark!("combine");
        if let Some(t) = times.as_mut() {
            ev.last().unwrap().1.sync()?;
            t.clear();
            for w in ev.windows(2) {
                t.push((w[1].0, w[1].1.elapsed_since(&w[0].1)?));
            }
        }
        Ok(())
    }
}

/// FP8 dense linear: E4M3 weights [n][k] with one f32 scale per output row (the P24 format of the attention projections
/// and the unembedding), bf16 activations; effective weight = bf16(q * scale[n]).
pub struct Fp8Linear {
    pub w8: DeviceBuffer<u8>,
    pub scale: DeviceBuffer<f32>,
    pub n: usize,
    pub k: usize,
}

impl Kern {
    /// Quantize bf16 weights [n][k] (device) into an `Fp8Linear` (per-row amax / 448, RNE, the format study's rule).
    pub fn fp8_from_bf16(&self, s: &Stream, w: DevPtr, n: usize, k: usize) -> Result<Fp8Linear> {
        let w8 = DeviceBuffer::<u8>::alloc(n * k)?;
        let scale = DeviceBuffer::<f32>::alloc(n)?;
        unsafe { s.launch(&self.fp8_quant, n as u32, 256, 0, (w8.ptr(), scale.ptr(), w, n as u32, k as u32))? };
        Ok(Fp8Linear { w8, scale, n, k })
    }

    /// y (stride ldy) = x (stride lda) . W^T [+ res (stride ldy)]: y = bf16(bf16(x.W) + res) like core's linears.
    /// m <= 16 rows: the FP8 GEMV (row bits independent of m). More rows: `scratch` (bf16, n * k elements) gets the
    /// dequantized weights and `gemm` = core's `fe_gemm_bf16` runs on the tensor cores (grid ceil(m/128) x ceil(n/128),
    /// 256 threads, 48 KB smem, args (y, x, w, res, m, n, k, lda, ldy)).
    pub fn fp8_forward(&self, s: &Stream, l: &Fp8Linear, gemm: &Function, scratch: DevPtr, y: DevPtr, ldy: usize,
        x: DevPtr, lda: usize, m: usize, res: Option<DevPtr>) -> Result<()> {
        ensure!(l.k % 256 == 0, "fp8_forward: k {} % 256", l.k);
        let null = DevPtr(0);
        unsafe {
            if m <= 16 {
                let f = &self.fp8_gemv[match m { 1 => 0, 2..=4 => 1, 5..=8 => 2, _ => 3 }];
                pdl::launch(s, f, (l.n as u32).div_ceil(8), 256, 0,
                    (y, x, l.w8.ptr(), l.scale.ptr(), res.unwrap_or(null), m as u32, l.n as u32, l.k as u32, lda as u32,
                     ldy as u32))?;
            } else {
                ensure!(!scratch.is_null(), "fp8_forward: {m} rows need bf16 scratch");
                pdl::launch(s, &self.fp8_dequant, l.n as u32, 256, 0, (scratch, l.w8.ptr(), l.scale.ptr(), l.n as u32, l.k as u32))?;
                s.launch(gemm, ((m as u32).div_ceil(128), (l.n as u32).div_ceil(128)), 256, 3 * 2 * 128 * 32 * 2,
                    (y, x, scratch, res.unwrap_or(null), m as u32, l.n as u32, l.k as u32, lda as u32, ldy as u32))?;
            }
        }
        Ok(())
    }
}

/// An MXFP8 layer re-quantized to NVFP4 in the release's NVFP4 layout (G16 build): feed it to [`Kern::repack_nvfp4`]
/// (`amax13` / `amax2` with denominators 2688) or to the plain `moe_decode` through [`Requant::layer`].
pub struct Requant {
    pub w13: DeviceBuffer<u8>,
    pub s13: DeviceBuffer<u8>,
    pub w2: DeviceBuffer<u8>,
    pub s2: DeviceBuffer<u8>,
    pub amax13: DeviceBuffer<f32>,
    pub amax2: DeviceBuffer<f32>,
    /// amax / 2688 (what `moe_decode` multiplies the values with)
    pub f13: DeviceBuffer<f32>,
    pub f2: DeviceBuffer<f32>,
}

impl Requant {
    pub const DENOMINATOR: f32 = 2688.0;
    pub fn layer(&self) -> ExpertLayer {
        ExpertLayer { fmt: ExpertFmt::Nvfp4, w13: self.w13.ptr(), s13: self.s13.ptr(), f13: self.f13.ptr(), w2: self.w2.ptr(),
            s2: self.s2.ptr(), f2: self.f2.ptr() }
    }
}

impl Kern {
    /// Re-quantize an MXFP8 layer (release layout, device) to NVFP4 (the expert rule: per-expert amax over the whole
    /// matrix, denominator 2688, block scale max/6 vs max/4 by absolute error). Allocates the NVFP4 copy (~0.57 GB); the
    /// MXFP8 buffers can be freed afterwards.
    pub fn requant_mxfp8_to_nvfp4(&self, s: &Stream, l: &ExpertLayer) -> Result<Requant> {
        ensure!(l.fmt == ExpertFmt::Mxfp8, "requant: not an MXFP8 layer");
        let mut rq = Requant {
            w13: DeviceBuffer::alloc(N_EXPERTS * 2 * FFN * (HID / 2))?,
            s13: DeviceBuffer::alloc(N_EXPERTS * 2 * FFN * (HID / 16))?,
            w2: DeviceBuffer::alloc(N_EXPERTS * HID * (FFN / 2))?,
            s2: DeviceBuffer::alloc(N_EXPERTS * HID * (FFN / 16))?,
            amax13: DeviceBuffer::alloc(N_EXPERTS)?,
            amax2: DeviceBuffer::alloc(N_EXPERTS)?,
            f13: DeviceBuffer::alloc(N_EXPERTS)?,
            f2: DeviceBuffer::alloc(N_EXPERTS)?,
        };
        let part = DeviceBuffer::<f32>::alloc(N_EXPERTS * 64)?;
        for (w8, sc, codes, sf, amax, rows, k) in [
            (l.w13, l.s13, rq.w13.ptr(), rq.s13.ptr(), rq.amax13.ptr(), 2 * FFN, HID),
            (l.w2, l.s2, rq.w2.ptr(), rq.s2.ptr(), rq.amax2.ptr(), HID, FFN),
        ] {
            unsafe {
                s.launch(&self.rq_amax_part, (N_EXPERTS as u32, 64u32), 256, 0, (part.ptr(), w8, sc, rows as u32, k as u32))?;
                s.launch(&self.rq_amax_fin, N_EXPERTS as u32, 64, 0, (amax, part.ptr()))?;
                s.launch(&self.rq_quant, (((rows * k / 16) as u32).div_ceil(256), N_EXPERTS as u32), 256, 0,
                    (codes, sf, w8, sc, amax, rows as u32, k as u32))?;
            }
        }
        // f = amax / 2688 for the plain decode path
        s.sync()?;
        let mut h = vec![0f32; N_EXPERTS];
        rq.amax13.copy_to(&mut h)?;
        rq.f13.copy_from(&h.iter().map(|a| a / Requant::DENOMINATOR).collect::<Vec<_>>())?;
        rq.amax2.copy_to(&mut h)?;
        rq.f2.copy_from(&h.iter().map(|a| a / Requant::DENOMINATOR).collect::<Vec<_>>())?;
        Ok(rq)
    }
}

/// smem of the small-M FP8 GEMM variants `ink_s8_u<U>_st<S>_w<W>` (see kernels/k-ink/src/dense_s8.rs): S stages, W warps (8 weight
/// rows a warp), U x 64 k a stage
pub const fn s8_smem(st: u32, nw: u32, u: u32) -> u32 {
    let bn = 8 * nw;
    st * (bn * (64 * u + 16) + 16 * (128 * u + 16)) + 2 * bn * (128 * u + 16)
}

/// Operands of the fused decode tail ([`Kern::moe_prefill_tail`]).
pub struct MoeTail {
    pub resid: DevPtr,
    pub out: DevPtr,
    pub conv_w: DevPtr,
    pub state: DevPtr,
    /// speculative chunk: the conv rows are kept after the 3 state rows and the state is not advanced (the engine's speculative mode)
    pub spec: bool,
}

/// Fused decode glue of the attention block (each call replaces the named k-inkm launches with bit-identical arithmetic).
impl Kern {
    /// `ink_headclip` for q (dh 128) and rel (dh 16) in one launch. `x` / `xr` = the two slices of the projection output (stride `ld`).
    pub fn headclip2(&self, s: &Stream, out_q: DevPtr, out_r: DevPtr, x: DevPtr, xr: DevPtr, ld: usize, scale_q: DevPtr, scale_r: DevPtr,
        log: bool, pos: DevPtr, ntok: usize) -> Result<()> {
        let hb = (ntok as u32 * 24 + 7) / 8;
        unsafe { pdl::launch(s, &self.headclip2, 2 * hb, 256, 0, (out_q, out_r, x, xr, ld as u32, scale_q, scale_r, log as u32, pos, ntok as u32, hb))? };
        Ok(())
    }
    /// `ink_sconv` + `ink_headnorm` (static conv taps) for K (mode 0, thr) or V (mode 1): rows <= 16.
    pub fn kv_conv_norm(&self, s: &Stream, cache: DevPtr, x: DevPtr, w: DevPtr, state: DevPtr, pos: DevPtr, nt: usize, ring: u32,
        mode: u32, thr: f32, spec: bool) -> Result<()> {
        ensure!(nt <= 16, "kv_conv_norm: rows {nt}");
        unsafe { pdl::launch(s, &self.kv_conv_norm, 8u32, 128, 0, (cache, x, w, state, pos, nt as u32, ring, mode, thr, spec as u32))? };
        Ok(())
    }
    /// `ink_sconv_dyn` + `ink_headnorm` (full-attention layers).
    pub fn kv_conv_dyn_norm(&self, s: &Stream, cache: DevPtr, x: DevPtr, stat: DevPtr, dynw: DevPtr, state: DevPtr, pos: DevPtr, nt: usize,
        ring: u32, mode: u32, thr: f32, spec: bool) -> Result<()> {
        ensure!(nt <= 16, "kv_conv_dyn_norm: rows {nt}");
        unsafe { pdl::launch(s, &self.kv_conv_dyn_norm, 8u32, 128, 0, (cache, x, stat, dynw, state, pos, nt as u32, ring, mode, thr, spec as u32))? };
        Ok(())
    }
    /// `ink_blockres` (attention residual over the block states) with a cheaper reduction, same bits. Arguments as the k-inkm kernel.
    pub fn blockres(&self, s: &Stream, out: DevPtr, ldo: usize, blocks: DevPtr, bstride: usize, nb: usize, hs: DevPtr, hs_f32: bool,
        ldh: usize, query: DevPtr, nq: usize, t: usize) -> Result<()> {
        unsafe {
            pdl::launch(s, &self.blockres2, t as u32, 256, 0,
                (out, ldo as u32, blocks, bstride as u32, nb as u32, hs, hs_f32 as u32, ldh as u32, query, nq as u32))?;
        }
        Ok(())
    }
    /// Message gate z = bf16(g . Wd^T): g bf16 [m][4608] (stride lda), Wd bf16 [64][4608], z bf16 [m][64] (stride ldc); bits of
    /// k-inkm's `ink_gemm_s` with a short chain (one warp per 8 outputs, fragments straight from global memory).
    pub fn gate_z(&self, s: &Stream, z: DevPtr, g: DevPtr, wd: DevPtr, m: usize, lda: usize, ldc: usize) -> Result<()> {
        unsafe { pdl::launch(s, &self.gemm_z, (8u32, (m as u32).div_ceil(16)), 32, 0, (z, g, wd, m as u32, lda as u32, ldc as u32))? };
        Ok(())
    }
    /// `ink_attn_tc_comb` + `ink_gate_mul` in one launch.
    pub fn attn_comb_gate(&self, s: &Stream, out: DevPtr, part: DevPtr, gate: DevPtr, ldg: usize, nsup: u32, ntok: usize, qt_off: u32,
        qc: u32) -> Result<()> {
        unsafe { pdl::launch(s, &self.comb_gate, (24u32, qc), 128, 0, (out, part, gate, ldg as u32, nsup, ntok as u32, qt_off))? };
        Ok(())
    }
}

/// Boolean INK_* switch: set and not "0" / "false" / "off" / "no" / empty (so INK_SLOT_LOG=0 is OFF).
pub fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "" | "0" | "false" | "off" | "no"),
        Err(_) => false,
    }
}
