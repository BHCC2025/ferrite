//! Host side of `kernels-xe/kx-core`: the Qwen3.8 non-GEMV kernels of the B70 forward (norms, Gated DeltaNet window
//! kernels, attention prep, KV q8 append, matrix-engine decode attention + merge, embedding, SwiGLU, argmax).
//!
//! [`KxCore`] holds the kernels and launches them with exactly arc-engine's grids, work-group sizes and arguments
//! (`qwen_win.zig`, `qwen_blocks.zig`, `qwen_attn_long.zig`), so fe-xe's forward calls one method per launch.
//! [`KxCore::rust`] uses the Rust modules (the engine path); [`KxCore::reference`] loads arc-engine's OpenCL `.spv`
//! files with the same names, for the bitwise checks and timing baselines only (never on the engine path; NON-RUST.md).
use anyhow::{anyhow, Context, Result};
use fe_ze::{DevPtr, Event, EventPool, Kernel, KernelArg, Module, Queue, Ze};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;

pub mod kd;

pub static KX_CORE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kx_core.spv"));
pub static KX_CORE_DEC: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kx_core_dec.spv"));

/// Build flags of the decode attention module (and arc-engine's long-attention modules): 256 GRF a thread.
pub const DEC_FLAGS: &str = "-cl-intel-256-GRF-per-thread";

// Qwen3.8-27B shapes the kernels are written for.
pub const HIDDEN: u32 = 5120;
pub const GDN_QKV: u32 = 10240;
pub const GDN_HEADS: u32 = 48;
pub const GDN_V: u32 = 6144;
pub const HEADS: u32 = 24;
pub const KV_HEADS: u32 = 4;
pub const HEAD_DIM: u32 = 256;
pub const Q_DIM: u32 = HEADS * HEAD_DIM; // 6144
pub const KV_DIM: u32 = KV_HEADS * HEAD_DIM; // 1024
pub const KV_REC: u32 = 272; // q8 record bytes of one (position, kv head)
pub const DEC_CHUNK: u32 = 1024; // keys a split-K chunk of the decode attention
pub const MAX_CHUNKS: u32 = 128; // chunks a row in the po / pm / pl scratch (131072 keys)
pub const ARGMAX_PARTS: u32 = 64;
pub const ATTN_SCALE: f32 = 0.0625;
/// keys of a tree row tail window (kx-core tree.rs TAIL_KEYS)
pub const TREE_TAIL_KEYS: u32 = 64;
/// bytes of one tree row tail window of one cache
pub const TREE_TAIL_BYTES: usize = TREE_TAIL_KEYS as usize * 4 * KV_REC as usize;

/// Directory of arc-engine's committed OpenCL `.spv` files (oracles / baselines only). ARC_SPV_DIR overrides.
pub fn arc_spv_dir() -> String {
    std::env::var("ARC_SPV_DIR").expect("ARC_SPV_DIR is not set: the folder of the reference OpenCL .spv modules (test oracles, not in this repository)")
}

fn arc_module(ze: &Arc<Ze>, name: &str, flags: &str) -> Result<Module> {
    let p = format!("{}/{name}", arc_spv_dir());
    let b = std::fs::read(&p).with_context(|| p.clone())?;
    Module::from_spirv(ze, &b, flags).map_err(|e| anyhow!("{p}: {e}"))
}

/// A SPIR-V module from an env override path (experiments) or the embedded bytes.
fn rust_module(ze: &Arc<Ze>, env: &str, embedded: &[u8], flags: &str) -> Result<Module> {
    let bytes = match std::env::var(env) {
        Ok(p) => std::fs::read(&p).with_context(|| format!("{env}={p}"))?,
        Err(_) => embedded.to_vec(),
    };
    Module::from_spirv(ze, &bytes, flags).map_err(|e| anyhow!("{env}: {e}"))
}

/// (kernel name, module slot, work-group size). Slots (= the reference modules): 0 qwen_rows, 1 qwen_basic, 2 qwen_kvq,
/// 3 qwen_attn_long_q8 (merge), 4 qwen_attn_dec_q8, 5 qwen_gdn, 6 qwen_attn, 7 spec_head, 8 qwen_attn_pfxi_q8 (prompt
/// attention, int8 scores). The Rust set has them all in kx_core.spv
/// except slot 4 (kx_core_dec.spv).
const KERNELS: &[(&str, usize, [u32; 3])] = &[
    ("add_rmsnorm_r", 0, [512, 1, 1]),
    ("rmsnorm_r", 0, [512, 1, 1]),
    ("gdn_conv_prep_r", 0, [16, 1, 1]),
    ("gdn_vconv_r", 0, [64, 1, 1]),
    ("gdn_gates_r", 0, [48, 1, 1]),
    ("gdn_step_r2x4", 0, [64, 1, 1]),
    ("gdn_step_r2", 0, [64, 1, 1]),
    ("gdn_gate_norm_r", 0, [16, 1, 1]),
    ("gdn_conv_commit", 0, [64, 1, 1]),
    ("attn_prep_r", 0, [256, 1, 1]),
    ("embed_bf16", 1, [64, 1, 1]),
    ("swiglu", 1, [64, 1, 1]),
    ("round_bf16_f32", 1, [64, 1, 1]),
    ("argmax_partial", 1, [256, 1, 1]),
    ("argmax_final", 1, [256, 1, 1]),
    ("kvq_quant8", 2, [32, 1, 1]),
    ("attn_long_merge", 3, [256, 1, 1]),
    ("attn_dec_partial", 4, [128, 1, 1]),
    ("attn_dec_pack", 4, [128, 1, 1]),
    ("attn_dec_pack2", 4, [256, 1, 1]),
    ("attn_dec_pack4", 4, [512, 1, 1]),
    // one-token forms (qwen_gdn.cl, qwen_attn.cl attn_prep, qwen_basic.cl one-row norms)
    ("gdn_conv_prep", 5, [16, 1, 1]),
    ("gdn_step", 5, [64, 1, 1]),
    ("gdn_gate_norm", 5, [16, 1, 1]),
    ("attn_prep", 6, [256, 1, 1]),
    ("add_rmsnorm", 1, [512, 1, 1]),
    ("rmsnorm512", 1, [512, 1, 1]),
    ("add_bf16", 1, [64, 1, 1]),
    // MTP draft head (spec_head.cl)
    ("dhead_q8b", 7, [64, 1, 1]),
    ("dhead_final", 7, [256, 1, 1]),
    // prompt-window attention (qwen_attn_pfx.cl -DKVQ=8 -DI8 -DHAD -DNO_HALF2)
    ("attn_pfs_prep_i8", 8, [128, 1, 1]),
    ("kvq_expand_k8", 8, [64, 1, 1]),
    ("kvq_expand_vt", 8, [64, 1, 1]),
    ("attn_prefill_s", 8, [128, 1, 1]),
    // tree verify windows (Rust only: arc-engine has no tree kernels; RUST_ONLY + module slot)
    ("gdn_conv_prep_tr", RUST_ONLY, [16, 1, 1]),
    ("gdn_vconv_tr", RUST_ONLY, [64, 1, 1]),
    ("gdn_step_tr", RUST_ONLY, [64, 1, 1]),
    ("kv_tree_tail", RUST_ONLY, [272, 1, 1]),
    ("kv_tree_commit", RUST_ONLY, [256, 1, 1]),
    ("rows_gather_u16", RUST_ONLY, [256, 1, 1]),
    ("attn_long_merge_tr", RUST_ONLY, [256, 1, 1]),
    ("attn_dec_tree", RUST_ONLY + 4, [128, 1, 1]),
    ("attn_dec_tree2", RUST_ONLY + 4, [256, 1, 1]),
    ("attn_dec_tree4", RUST_ONLY + 4, [512, 1, 1]),
    // fused GDN window chain (kx-core gdnpre.rs)
    ("gdn_pre_r", RUST_ONLY, [64, 1, 1]),
    ("gdn_pre_tr", RUST_ONLY, [64, 1, 1]),
];
/// Slot offset of the kernels only the Rust set has (the slot is RUST_ONLY + the module slot).
const RUST_ONLY: usize = 100;

/// The kx-core kernels with their launch geometry. Not thread-safe (kernel arguments are per-object state).
pub struct KxCore {
    pub label: &'static str,
    k: HashMap<&'static str, Kernel>,
    pub modules: Vec<Module>,
    // timing: when on, every launch signals a fresh timestamp event
    pool: RefCell<Option<EventPool>>,
    evs: RefCell<Vec<Event>>,
    next_ev: Cell<u32>,
}

const EV_CAP: u32 = 4096;

impl KxCore {
    /// The Rust kernels (kx_core.spv, kx_core_dec.spv). Env KX_CORE_SPV / KX_CORE_DEC_SPV load other builds.
    pub fn rust(ze: &Arc<Ze>) -> Result<KxCore> {
        let core = rust_module(ze, "KX_CORE_SPV", KX_CORE, "")?;
        let dec = rust_module(ze, "KX_CORE_DEC_SPV", KX_CORE_DEC, DEC_FLAGS)?;
        // every slot but the decode attention lives in kx_core.spv
        Self::with(ze, "rust", vec![core.clone(), core.clone(), core.clone(), core.clone(), dec.clone(), core.clone(), core.clone(), core, dec])
    }

    /// arc-engine's OpenCL modules (qwen_rows, qwen_basic, qwen_kvq, qwen_attn_long_q8, qwen_attn_dec_q8) with the build
    /// flags arc-engine uses: test oracle and timing baseline only.
    pub fn reference(ze: &Arc<Ze>) -> Result<KxCore> {
        let ms = vec![
            arc_module(ze, "qwen_rows.spv", "")?,
            arc_module(ze, "qwen_basic.spv", "")?,
            arc_module(ze, "qwen_kvq.spv", "")?,
            arc_module(ze, "qwen_attn_long_q8.spv", DEC_FLAGS)?,
            arc_module(ze, "qwen_attn_dec_q8.spv", DEC_FLAGS)?,
            arc_module(ze, "qwen_gdn.spv", "")?,
            arc_module(ze, "qwen_attn.spv", "")?,
            arc_module(ze, "spec_head.spv", "")?,
            arc_module(ze, "qwen_attn_pfxi_q8.spv", DEC_FLAGS)?,
        ];
        Self::with(ze, "opencl", ms)
    }

    fn with(_ze: &Arc<Ze>, label: &'static str, modules: Vec<Module>) -> Result<KxCore> {
        let mut k = HashMap::new();
        for &(name, slot, group) in KERNELS {
            if slot >= RUST_ONLY && label != "rust" {
                continue;
            }
            k.insert(name, modules[slot % RUST_ONLY].kernel(name, group).map_err(|e| anyhow!("{label} {name}: {e}"))?);
        }
        Ok(KxCore { label, k, modules, pool: RefCell::new(None), evs: RefCell::new(Vec::new()), next_ev: Cell::new(0) })
    }

    pub fn kernel(&self, name: &str) -> &Kernel {
        self.k.get(name).unwrap_or_else(|| panic!("no kernel {name}"))
    }

    /// Whether this set has a kernel of that name.
    pub fn has(name: &str) -> bool {
        KERNELS.iter().any(|k| k.0 == name)
    }

    /// A fresh kernel object `name` from its module with work-group size `group` (dump replays).
    pub fn kernel_for(&self, name: &str, group: [u32; 3]) -> Result<Kernel> {
        let slot = KERNELS.iter().find(|k| k.0 == name).map(|k| k.1).ok_or_else(|| anyhow!("no kernel {name}"))?;
        self.modules[slot % RUST_ONLY].kernel(name, group).map_err(|e| anyhow!("{} {name}: {e}", self.label))
    }

    /// Timing mode: every launch from now on records its device time (read with [`KxCore::take_times`]).
    pub fn set_timing(&self, ze: &Arc<Ze>, on: bool) -> Result<()> {
        if on && self.pool.borrow().is_none() {
            *self.pool.borrow_mut() = Some(EventPool::new(ze, EV_CAP)?);
        }
        if !on {
            *self.pool.borrow_mut() = None;
        }
        self.evs.borrow_mut().clear();
        self.next_ev.set(0);
        Ok(())
    }

    /// Device times (ns) of the launches since the last call, in order.
    pub fn take_times(&self) -> Result<Vec<f64>> {
        let v: Result<Vec<f64>> = self.evs.borrow().iter().map(|e| e.kernel_ns().map_err(|e| anyhow!("{e}"))).collect();
        self.evs.borrow_mut().clear();
        self.next_ev.set(0);
        v
    }

    fn go(&self, q: &Queue, name: &str, args: &[&dyn KernelArg], groups: [u32; 3]) -> Result<()> {
        let k = self.kernel(name);
        k.args(args).map_err(|e| anyhow!("{} {name}: {e}", self.label))?;
        let pool = self.pool.borrow();
        if let Some(p) = pool.as_ref() {
            let i = self.next_ev.get();
            if i >= EV_CAP {
                return Err(anyhow!("timing: more than {EV_CAP} launches without take_times"));
            }
            let ev = p.event(i)?;
            q.launch_ev(k, groups, Some(&ev))?;
            self.next_ev.set(i + 1);
            self.evs.borrow_mut().push(ev);
        } else {
            q.launch(k, groups)?;
        }
        Ok(())
    }

    /// Generic launch by name (dump replays): arguments as given, `groups` work-groups.
    pub fn launch_raw(&self, q: &Queue, name: &str, args: &[&dyn KernelArg], groups: [u32; 3]) -> Result<()> {
        self.go(q, name, args, groups)
    }

    // ---- norms -----------------------------------------------------------------------------------------------------

    /// x = bf16(x + d); y = bf16(rmsnorm(x) * w) for `rows` rows of n (<= 5120) (qwen_win normIn, pending add).
    pub fn add_rmsnorm_r(&self, q: &Queue, x: DevPtr, d: DevPtr, w: DevPtr, y: DevPtr, rows: u32, n: u32, eps: f32) -> Result<()> {
        self.go(q, "add_rmsnorm_r", &[&x, &d, &w, &y, &n, &eps], [rows, 1, 1])
    }
    /// y = bf16(rmsnorm(x) * w) for `rows` rows of n (<= 5120).
    pub fn rmsnorm_r(&self, q: &Queue, x: DevPtr, w: DevPtr, y: DevPtr, rows: u32, n: u32, eps: f32) -> Result<()> {
        self.go(q, "rmsnorm_r", &[&x, &w, &y, &n, &eps], [rows, 1, 1])
    }

    // ---- Gated DeltaNet window (qwen_win gdnCore) ------------------------------------------------------------------

    /// q / k channels: conv + silu + per-head RMS scaling, rows of qkv [rows][cd] against the conv state [3][cd].
    pub fn gdn_conv_prep_r(&self, q: &Queue, qkv: DevPtr, cstate: DevPtr, conv: DevPtr, qn: DevPtr, kn: DevPtr, rows: u32) -> Result<()> {
        self.go(q, "gdn_conv_prep_r", &[&qkv, &cstate, &conv, &qn, &kn, &GDN_QKV], [16, rows, 1])
    }
    /// v channels: conv + silu into vconv [rows][6144].
    pub fn gdn_vconv_r(&self, q: &Queue, qkv: DevPtr, cstate: DevPtr, conv: DevPtr, vconv: DevPtr, rows: u32) -> Result<()> {
        self.go(q, "gdn_vconv_r", &[&qkv, &cstate, &conv, &vconv, &GDN_QKV], [rows, 96, 1])
    }
    /// beta, g [rows][48] from b, a [rows][48].
    pub fn gdn_gates_r(&self, q: &Queue, b: DevPtr, a: DevPtr, a_log: DevPtr, dt_bias: DevPtr, beta: DevPtr, g: DevPtr, rows: u32) -> Result<()> {
        self.go(q, "gdn_gates_r", &[&b, &a, &a_log, &dt_bias, &beta, &g], [rows, 1, 1])
    }
    /// Delta rule over `rows` rows from s_in into s_out (== s_in in place); y [rows][6144] bf16 (gdn_step_r2x4).
    pub fn gdn_step(&self, q: &Queue, qn: DevPtr, kn: DevPtr, vconv: DevPtr, beta: DevPtr, g: DevPtr, s_in: DevPtr, s_out: DevPtr, y: DevPtr, tiled: bool, rows: u32) -> Result<()> {
        self.go(q, "gdn_step_r2x4", &[&qn, &kn, &vconv, &beta, &g, &s_in, &s_out, &y, &(tiled as u32), &rows], [GDN_HEADS, 8, 1])
    }
    /// The 1-row-a-sub-group form (bit-identical to gdn_step; reference cross-check).
    pub fn gdn_step_r2(&self, q: &Queue, qn: DevPtr, kn: DevPtr, vconv: DevPtr, beta: DevPtr, g: DevPtr, s_in: DevPtr, s_out: DevPtr, y: DevPtr, tiled: bool, rows: u32) -> Result<()> {
        self.go(q, "gdn_step_r2", &[&qn, &kn, &vconv, &beta, &g, &s_in, &s_out, &y, &(tiled as u32), &rows], [GDN_HEADS, 32, 1])
    }
    /// out = bf16(silu(z) * rmsnorm(y) * w) per value head, rows of [6144].
    pub fn gdn_gate_norm_r(&self, q: &Queue, y: DevPtr, z: DevPtr, w: DevPtr, out: DevPtr, rows: u32, eps: f32) -> Result<()> {
        self.go(q, "gdn_gate_norm_r", &[&y, &z, &w, &out, &eps], [GDN_HEADS, rows, 1])
    }
    /// Conv state after `c` kept rows of qkv (all 10240 channels).
    pub fn gdn_conv_commit(&self, q: &Queue, qkv: DevPtr, cstate: DevPtr, c: u32) -> Result<()> {
        self.go(q, "gdn_conv_commit", &[&qkv, &cstate, &GDN_QKV, &c, &0u32, &GDN_QKV], [(GDN_QKV + 63) / 64, 1, 1])
    }

    // ---- attention ------------------------------------------------------------------------------------------------

    /// q / k RMSNorm + RoPE: q rows to qo [rows][6144], key rows to kc[k_off + row * 1024 ..] (bf16 elements).
    pub fn attn_prep_r(&self, q: &Queue, qg: DevPtr, kraw: DevPtr, qw: DevPtr, kw: DevPtr, rope: DevPtr, qo: DevPtr, kc: DevPtr, k_off: u64, rows: u32, eps: f32) -> Result<()> {
        self.go(q, "attn_prep_r", &[&qg, &kraw, &qw, &kw, &rope, &qo, &kc, &k_off, &eps], [HEADS + KV_HEADS, rows, 1])
    }
    /// bf16 rows [rows][1024] -> q8 records at positions pos0 .. (one cache, K or V).
    pub fn kvq_quant8(&self, q: &Queue, src: DevPtr, cache: DevPtr, pos0: u32, rows: u32) -> Result<()> {
        self.go(q, "kvq_quant8", &[&src, &cache, &pos0], [rows, 1, 1])
    }
    /// Decode attention partials for `rows` rows, row z over len0 + z keys of the q8 caches (qwen_attn_long.zig
    /// Long.runOnce with ARC_DEC=1: attn_dec_partial for one row, else the packed tiles, 4 / 2 / 1 tiles a group).
    pub fn attn_dec(&self, q: &Queue, qv: DevPtr, kc: DevPtr, vc: DevPtr, po: DevPtr, pm: DevPtr, pl: DevPtr, len0: u32, rows: u32) -> Result<()> {
        let nch = (len0 + rows - 1 + DEC_CHUNK - 1) / DEC_CHUNK;
        let maxc = MAX_CHUNKS;
        if rows == 1 {
            return self.go(q, "attn_dec_partial", &[&qv, &kc, &vc, &po, &pm, &pl, &len0, &ATTN_SCALE, &maxc], [4, nch, 1]);
        }
        let tiles = (6 * rows + 7) / 8;
        let tg = if tiles >= 4 { 4 } else if tiles >= 2 { 2 } else { 1 };
        let name = match tg {
            4 => "attn_dec_pack4",
            2 => "attn_dec_pack2",
            _ => "attn_dec_pack",
        };
        self.go(q, name, &[&qv, &kc, &vc, &po, &pm, &pl, &len0, &ATTN_SCALE, &maxc, &rows], [4 * tiles.div_ceil(tg), nch, 1])
    }
    /// One specific decode-attention kernel (checks): `attn_dec_partial` (rows must be 1... launched per row group) or
    /// `attn_dec_pack{,2,4}`.
    pub fn attn_dec_with(&self, q: &Queue, name: &str, qv: DevPtr, kc: DevPtr, vc: DevPtr, po: DevPtr, pm: DevPtr, pl: DevPtr, len0: u32, rows: u32) -> Result<()> {
        let nch = (len0 + rows - 1 + DEC_CHUNK - 1) / DEC_CHUNK;
        let maxc = MAX_CHUNKS;
        if name == "attn_dec_partial" {
            return self.go(q, name, &[&qv, &kc, &vc, &po, &pm, &pl, &len0, &ATTN_SCALE, &maxc], [4 * rows, nch, 1]);
        }
        let tiles = (6 * rows + 7) / 8;
        let tg = match name {
            "attn_dec_pack4" => 4,
            "attn_dec_pack2" => 2,
            _ => 1,
        };
        self.go(q, name, &[&qv, &kc, &vc, &po, &pm, &pl, &len0, &ATTN_SCALE, &maxc, &rows], [4 * tiles.div_ceil(tg), nch, 1])
    }
    /// Merge of the chunk partials + output gate: out [rows][6144] bf16, qg [rows][24][512].
    pub fn attn_long_merge(&self, q: &Queue, po: DevPtr, pm: DevPtr, pl: DevPtr, qg: DevPtr, out: DevPtr, len0: u32, rows: u32) -> Result<()> {
        self.go(q, "attn_long_merge", &[&po, &pm, &pl, &qg, &out, &len0, &MAX_CHUNKS], [HEADS, rows, 1])
    }

    // ---- one-token forms (qwen_blocks.zig: plain decode steps, MTP) --------------------------------------------------

    /// One row: x = bf16(x + d); y = bf16(rmsnorm(x) * w).
    pub fn add_rmsnorm(&self, q: &Queue, x: DevPtr, d: DevPtr, w: DevPtr, y: DevPtr, n: u32, eps: f32) -> Result<()> {
        self.go(q, "add_rmsnorm", &[&x, &d, &w, &y, &n, &eps], [1, 1, 1])
    }
    /// One row: y = bf16(rmsnorm(x) * w).
    pub fn rmsnorm512(&self, q: &Queue, x: DevPtr, w: DevPtr, y: DevPtr, n: u32, eps: f32) -> Result<()> {
        self.go(q, "rmsnorm512", &[&x, &w, &y, &n, &eps], [1, 1, 1])
    }
    /// x = bf16(x + d) over n elements (n a multiple of 64).
    pub fn add_bf16(&self, q: &Queue, x: DevPtr, d: DevPtr, n: u32) -> Result<()> {
        self.go(q, "add_bf16", &[&x, &d, &n], [n / 64, 1, 1])
    }
    /// q / k channels of one token (shifts their conv state).
    pub fn gdn_conv_prep(&self, q: &Queue, qkv: DevPtr, cstate: DevPtr, conv: DevPtr, qn: DevPtr, kn: DevPtr) -> Result<()> {
        self.go(q, "gdn_conv_prep", &[&qkv, &cstate, &conv, &qn, &kn, &GDN_QKV], [16, 1, 1])
    }
    /// Delta-rule step of one token, gates and v-channel conv fused (shifts the v channels' conv state), state in place.
    pub fn gdn_step1(&self, q: &Queue, qn: DevPtr, kn: DevPtr, qkv: DevPtr, cstate: DevPtr, conv: DevPtr, b: DevPtr, a: DevPtr, a_log: DevPtr, dt_bias: DevPtr, sstate: DevPtr, y: DevPtr, tiled: bool) -> Result<()> {
        self.go(q, "gdn_step", &[&qn, &kn, &qkv, &cstate, &conv, &b, &a, &a_log, &dt_bias, &sstate, &y, &(tiled as u32), &GDN_QKV], [GDN_HEADS, 32, 1])
    }
    /// Gated RMSNorm of one token.
    pub fn gdn_gate_norm(&self, q: &Queue, y: DevPtr, z: DevPtr, w: DevPtr, out: DevPtr, eps: f32) -> Result<()> {
        self.go(q, "gdn_gate_norm", &[&y, &z, &w, &out, &eps], [GDN_HEADS, 1, 1])
    }
    /// q / k RMSNorm + RoPE of one token; keys to kc[k_off ..] (bf16 elements).
    pub fn attn_prep(&self, q: &Queue, qg: DevPtr, kraw: DevPtr, qw: DevPtr, kw: DevPtr, rope: DevPtr, qo: DevPtr, kc: DevPtr, k_off: u32, eps: f32) -> Result<()> {
        self.go(q, "attn_prep", &[&qg, &kraw, &qw, &kw, &rope, &qo, &kc, &k_off, &eps], [HEADS + KV_HEADS, 1, 1])
    }

    /// Draft head over the n selected int8 rows [n][k] (fp32 scale a row) of the MTP-normed hidden x (bf16 [k]): logits
    /// [n] fp32, argmax_partial into part [64], dhead_final maps the row to ids[row] -> out[0] (qwen_mtp.zig).
    pub fn dhead(&self, q: &Queue, rows: DevPtr, scale: DevPtr, x: DevPtr, logits: DevPtr, part: DevPtr, out: DevPtr, ids: DevPtr, n: u32, k: u32) -> Result<()> {
        self.go(q, "dhead_q8b", &[&rows, &scale, &x, &logits, &n, &k], [n.div_ceil(32), 1, 1])?;
        let per = n.div_ceil(ARGMAX_PARTS);
        self.go(q, "argmax_partial", &[&logits, &part, &n, &per], [ARGMAX_PARTS, 1, 1])?;
        self.go(q, "dhead_final", &[&part, &out, &ARGMAX_PARTS, &ids], [1, 1, 1])
    }

    // ---- prompt-window attention (qwen_attn_long.zig Pfs.run, ARC_PFX=i8, q8 cache) -------------------------------------

    /// Causal attention of `rows` window rows at positions pos0 .. (their own K / V already in the caches) over all
    /// pos0 + rows keys, key ranges of `s.range` keys expanded at a time; q [rows][24][256] bf16 after RoPE, qg [rows][24][512],
    /// out [rows][24][256] bf16.
    pub fn attn_prefill(&self, q: &Queue, s: &PfScratch, qv: DevPtr, kc: DevPtr, vc: DevPtr, qg: DevPtr, out: DevPtr, pos0: u32, rows: u32) -> Result<()> {
        let tiles = rows.div_ceil(8);
        assert!(rows <= s.max_rows, "attn_prefill: {rows} rows > scratch {}", s.max_rows);
        self.go(q, "attn_pfs_prep_i8", &[&qv, &s.qb, &s.qs, &rows], [tiles * 24, 1, 1])?;
        let kall = pos0 + rows;
        let mut k0 = 0;
        while k0 < kall {
            let k1 = (k0 + s.range).min(kall);
            self.go(q, "kvq_expand_k8", &[&kc, &s.kx, &s.ks, &s.mu, &k0], [k1 - k0, 1, 1])?;
            self.go(q, "kvq_expand_vt", &[&vc, &s.vx, &k0, &(k1 - k0), &s.range], [(k1 - k0).div_ceil(64), 128, 1])?;
            let flags: u32 = (k0 == 0) as u32 | (((k1 == kall) as u32) << 1);
            let st = if flags != 3 { s.st } else { s.qb };
            self.go(q, "attn_prefill_s", &[&s.qb, &s.kx, &s.vx, &qg, &out, &st, &pos0, &rows, &k0, &k0, &k1, &flags, &s.range, &s.ks, &s.qs], [tiles, 4, 1])?;
            k0 = k1;
        }
        Ok(())
    }

    /// gdn_conv_prep_r + gdn_vconv_r + gdn_gates_r in one launch (same bits); gates = false leaves beta / g alone.
    pub fn gdn_pre_r(&self, q: &Queue, qkv: DevPtr, cstate: DevPtr, conv: DevPtr, qn: DevPtr, kn: DevPtr, vconv: DevPtr, b: DevPtr, a: DevPtr, a_log: DevPtr, dt_bias: DevPtr, beta: DevPtr, g: DevPtr, rows: u32, gates: bool) -> Result<()> {
        self.go(q, "gdn_pre_r", &[&qkv, &cstate, &conv, &qn, &kn, &vconv, &GDN_QKV, &b, &a, &a_log, &dt_bias, &beta, &g], [if gates { 101 } else { 100 }, rows, 1])
    }
    /// gdn_pre_r for a tree window.
    pub fn gdn_pre_tr(&self, q: &Queue, qkv: DevPtr, cstate: DevPtr, conv: DevPtr, qn: DevPtr, kn: DevPtr, vconv: DevPtr, b: DevPtr, a: DevPtr, a_log: DevPtr, dt_bias: DevPtr, beta: DevPtr, g: DevPtr, rows: u32, gates: bool, tree: DevPtr) -> Result<()> {
        self.go(q, "gdn_pre_tr", &[&qkv, &cstate, &conv, &qn, &kn, &vconv, &GDN_QKV, &b, &a, &a_log, &dt_bias, &beta, &g, &tree], [if gates { 101 } else { 100 }, rows, 1])
    }

    // ---- tree verify windows (draft trees; kx-core tree.rs, dec.rs dec_tree) -------------------------------------------
    // Rows topological (par[t] < t, row 0 the root at pos0, row t at position pos0 + depth[t], its K / V at cache slot
    // pos0 + t); every output of row t == decoding its root -> t path alone with the linear kernels, bitwise.

    /// q / k channels of a tree window (conv over each row's predecessor chain).
    pub fn gdn_conv_prep_tr(&self, q: &Queue, qkv: DevPtr, cstate: DevPtr, conv: DevPtr, qn: DevPtr, kn: DevPtr, rows: u32, tree: DevPtr) -> Result<()> {
        self.go(q, "gdn_conv_prep_tr", &[&qkv, &cstate, &conv, &qn, &kn, &GDN_QKV, &tree], [16, rows, 1])
    }
    /// v channels of a tree window.
    pub fn gdn_vconv_tr(&self, q: &Queue, qkv: DevPtr, cstate: DevPtr, conv: DevPtr, vconv: DevPtr, rows: u32, tree: DevPtr) -> Result<()> {
        self.go(q, "gdn_vconv_tr", &[&qkv, &cstate, &conv, &vconv, &GDN_QKV, &tree], [rows, 96, 1])
    }
    /// Delta rule over a tree window from s_in (unchanged); y [rows][6144]; sscr: TreeTable::slots x 3 MB of scratch.
    pub fn gdn_step_tr(&self, q: &Queue, qn: DevPtr, kn: DevPtr, vconv: DevPtr, beta: DevPtr, g: DevPtr, s_in: DevPtr, sscr: DevPtr, y: DevPtr, tiled: bool, rows: u32, tree: DevPtr) -> Result<()> {
        self.go(q, "gdn_step_tr", &[&qn, &kn, &vconv, &beta, &g, &s_in, &sscr, &y, &(tiled as u32), &rows, &tree], [GDN_HEADS, 8, 1])
    }
    /// The per-row tail windows tk / tv [rows][64][4][272] of the q8 caches (keys pos0 & !31 .. + 63 along each path).
    pub fn kv_tree_tail(&self, q: &Queue, kc: DevPtr, vc: DevPtr, tk: DevPtr, tv: DevPtr, tree: DevPtr, pos0: u32, rows: u32) -> Result<()> {
        let kbase = pos0 & !31;
        self.go(q, "kv_tree_tail", &[&kc, &vc, &tk, &tv, &tree, &pos0, &kbase], [TREE_TAIL_KEYS, rows, 1])
    }
    /// Decode attention partials of a tree window (after kv_tree_tail): max_depth = the deepest row.
    pub fn attn_dec_tree(&self, q: &Queue, qv: DevPtr, kc: DevPtr, vc: DevPtr, po: DevPtr, pm: DevPtr, pl: DevPtr, pos0: u32, rows: u32, max_depth: u32, tree: DevPtr, tk: DevPtr, tv: DevPtr) -> Result<()> {
        let tiles = (6 * rows + 7) / 8;
        let tg = if tiles >= 4 { 4 } else if tiles >= 2 { 2 } else { 1 };
        self.attn_dec_tree_tg(q, qv, kc, vc, po, pm, pl, pos0, rows, max_depth, tree, tk, tv, tg)
    }
    /// [`KxCore::attn_dec_tree`] with `tg` (1, 2, 4) tiles a work-group: the same bits for every form.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_dec_tree_tg(&self, q: &Queue, qv: DevPtr, kc: DevPtr, vc: DevPtr, po: DevPtr, pm: DevPtr, pl: DevPtr, pos0: u32, rows: u32, max_depth: u32, tree: DevPtr, tk: DevPtr, tv: DevPtr, tg: u32) -> Result<()> {
        let nch = (pos0 + max_depth + 1).div_ceil(DEC_CHUNK);
        let tiles = (6 * rows + 7) / 8;
        let name = match tg {
            4 => "attn_dec_tree4",
            2 => "attn_dec_tree2",
            _ => "attn_dec_tree",
        };
        self.go(q, name, &[&qv, &kc, &vc, &po, &pm, &pl, &pos0, &ATTN_SCALE, &MAX_CHUNKS, &rows, &tree, &tk, &tv], [4 * tiles.div_ceil(tg), nch, 1])
    }
    /// Merge + gate of a tree window (row z: pos0 + depth[z] + 1 keys).
    pub fn attn_long_merge_tr(&self, q: &Queue, po: DevPtr, pm: DevPtr, pl: DevPtr, qg: DevPtr, out: DevPtr, pos0: u32, rows: u32, tree: DevPtr) -> Result<()> {
        self.go(q, "attn_long_merge_tr", &[&po, &pm, &pl, &qg, &out, &pos0, &MAX_CHUNKS, &tree], [HEADS, rows, 1])
    }
    /// After acceptance: the path rows (path[0] = 0, n <= 16) -> cache positions pos0 .., K and V.
    pub fn kv_tree_commit(&self, q: &Queue, kc: DevPtr, vc: DevPtr, path: DevPtr, n: u32, pos0: u32) -> Result<()> {
        self.go(q, "kv_tree_commit", &[&kc, &vc, &path, &n, &pos0], [1, 1, 1])
    }
    /// dst[r] = src[idx[r]] for rows of row_elems u16 (gather an accepted path's saved inputs for the commit replay).
    pub fn rows_gather_u16(&self, q: &Queue, src: DevPtr, dst: DevPtr, idx: DevPtr, rows: u32, row_elems: u32) -> Result<()> {
        self.go(q, "rows_gather_u16", &[&src, &dst, &idx, &row_elems], [rows, row_elems.div_ceil(256), 1])
    }

    // ---- embedding, MLP glue, head --------------------------------------------------------------------------------

    /// y [rows][dim] = w[ids[row]] (bf16 table).
    pub fn embed_bf16(&self, q: &Queue, w: DevPtr, ids: DevPtr, y: DevPtr, rows: u32, dim: u32) -> Result<()> {
        self.go(q, "embed_bf16", &[&w, &ids, &y, &dim], [rows, 1, 1])
    }
    /// act = bf16(silu(g) * u) over n elements (n a multiple of 64).
    pub fn swiglu(&self, q: &Queue, g: DevPtr, u: DevPtr, act: DevPtr, n: u32) -> Result<()> {
        self.go(q, "swiglu", &[&g, &u, &act, &n], [n / 64, 1, 1])
    }
    /// fp32 logits rounded to bf16 in place.
    pub fn round_bf16_f32(&self, q: &Queue, x: DevPtr, n: u32) -> Result<()> {
        self.go(q, "round_bf16_f32", &[&x, &n], [n.div_ceil(64), 1, 1])
    }
    /// Greedy token of each row of logits [rows][n] (fp32) into out [rows] (i32); part: [rows][64] u64 scratch.
    pub fn argmax(&self, q: &Queue, logits: DevPtr, part: DevPtr, out: DevPtr, rows: u32, n: u32) -> Result<()> {
        let per = n.div_ceil(ARGMAX_PARTS);
        self.go(q, "argmax_partial", &[&logits, &part, &n, &per], [ARGMAX_PARTS, rows, 1])?;
        self.go(q, "argmax_final", &[&part, &out, &ARGMAX_PARTS], [rows, 1, 1])
    }
}

/// Device scratch of the prompt-window attention for windows of up to `max_rows` rows and key ranges of `range` keys
/// (a multiple of 128; arc-engine's default 16384): int8 query tiles + scales, the expanded K' (int8) / scales / mu and
/// V' (fp16, transposed; zero-initialised, as the kernel reads whole 16-key tiles past a range's end), the parked state.
pub struct PfScratch {
    pub qb: DevPtr,
    pub qs: DevPtr,
    pub kx: DevPtr,
    pub ks: DevPtr,
    pub vx: DevPtr,
    pub mu: DevPtr,
    pub st: DevPtr,
    pub range: u32,
    pub max_rows: u32,
    _bufs: Vec<fe_ze::DeviceBuffer<u8>>,
}

impl PfScratch {
    /// Scratch made of caller-owned buffers (checks that compare two runs from identical scratch contents).
    pub fn from_parts(p: [DevPtr; 7], range: u32, max_rows: u32) -> PfScratch {
        PfScratch { qb: p[0], qs: p[1], kx: p[2], ks: p[3], vx: p[4], mu: p[5], st: p[6], range, max_rows, _bufs: Vec::new() }
    }
    pub fn new(q: &Queue, max_rows: u32, range: u32) -> Result<PfScratch> {
        let range = (range / 128 * 128).max(128) as usize;
        let tiles = max_rows.div_ceil(8) as usize;
        let mut bufs = Vec::new();
        let mut mk = |bytes: usize| -> Result<DevPtr> {
            let b = fe_ze::DeviceBuffer::<u8>::zeroed(q, bytes)?;
            let p = b.ptr();
            bufs.push(b);
            Ok(p)
        };
        let qb = mk(tiles * 8 * 24 * 256 * 2)?;
        let qs = mk(tiles * 4 * 3 * 16 * 4)?;
        let kx = mk(range * 4 * 256 * 2)?;
        let ks = mk(range * 4 * 4)?;
        let vx = mk(range * 4 * 256 * 2)?;
        let mu = mk(1024 * 4)?;
        let st = mk(tiles * 4 * 8 * 1632 * 4)?;
        Ok(PfScratch { qb, qs, kx, ks, vx, mu, st, range: range as u32, max_rows, _bufs: bufs })
    }
}

/// The tree table of a draft-tree verify window (kx-core tree.rs): u32 [rows][4] = par (i32), depth, ancestor mask,
/// delta-rule state slots (save | restore << 8) for gdn_step_tr.
#[derive(Clone, Debug)]
pub struct TreeTable {
    pub words: Vec<u32>,
    pub par: Vec<i32>,
    pub depth: Vec<u32>,
    /// scratch slots gdn_step_tr needs (each 48 x 128 x 128 fp32 = 3 MB)
    pub slots: u32,
}

impl TreeTable {
    /// From the parent of every row: par[0] = -1 (the root continues the committed sequence), 0 <= par[t] < t else;
    /// at most 16 rows.
    pub fn new(par: &[i32]) -> Result<TreeTable> {
        let m = par.len();
        if m == 0 || m > 16 || par[0] != -1 || par.iter().enumerate().skip(1).any(|(t, &p)| p < 0 || p as usize >= t) {
            return Err(anyhow!("tree: rows must be 1..16 in topological order with par[0] = -1: {par:?}"));
        }
        let mut depth = vec![0u32; m];
        let mut anc = vec![1u32; m];
        for t in 1..m {
            let p = par[t] as usize;
            depth[t] = depth[p] + 1;
            anc[t] = anc[p] | (1 << t);
        }
        // state slots: row t saves its state when a child other than t + 1 comes later; the slot is freed when the last
        // such child restores it (a row restores before it saves, so it may reuse the slot it read)
        let mut last_child = vec![-1i64; m];
        let mut needs_save = vec![false; m];
        for t in 1..m {
            let p = par[t] as usize;
            if t != p + 1 {
                needs_save[p] = true;
                last_child[p] = last_child[p].max(t as i64);
            }
        }
        let mut slot_of = vec![0xffu32; m];
        let mut free: Vec<u32> = Vec::new();
        let mut next = 0u32;
        let mut words = vec![0u32; 4 * m];
        for t in 0..m {
            let mut restore = 0xffu32;
            if t > 0 && par[t] as usize != t - 1 {
                let p = par[t] as usize;
                restore = slot_of[p];
                assert!(restore != 0xff, "tree: no saved state for row {p}");
                if last_child[p] == t as i64 {
                    free.push(restore);
                }
            }
            let mut save = 0xffu32;
            if needs_save[t] {
                save = free.pop().unwrap_or_else(|| {
                    next += 1;
                    next - 1
                });
                slot_of[t] = save;
            }
            words[4 * t] = par[t] as u32;
            words[4 * t + 1] = depth[t];
            words[4 * t + 2] = anc[t];
            words[4 * t + 3] = save | (restore << 8);
        }
        Ok(TreeTable { words, par: par.to_vec(), depth, slots: next.max(1) })
    }
    /// Rows of the root -> t path, in depth order.
    pub fn path(&self, t: usize) -> Vec<usize> {
        let mut p = vec![t];
        let mut x = t;
        while self.par[x] >= 0 {
            x = self.par[x] as usize;
            p.push(x);
        }
        p.reverse();
        p
    }
    pub fn max_depth(&self) -> u32 {
        *self.depth.iter().max().unwrap()
    }
}
