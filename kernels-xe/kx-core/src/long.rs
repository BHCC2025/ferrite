//! Split-K merge of the decode attention partials + output gate: Rust port of arc-engine `kernels/xpu/qwen_attn_long.cl`
//! attn_long_merge (the q8 build, qwen_attn_long_q8.spv). Contraction sites: `den += pl * w` and `acc += po * w` (mad).
use crate::{bf, bfr, sigmoid, to_bf};
use kxcommon::{attr, id, math};

const HD: usize = 256;
const NH: usize = 24;
const LC: u32 = 1024;

/// The merge of one (head, dim) of row z with `len` keys (shared by the window and the tree forms).
#[inline(always)]
pub(crate) unsafe fn merge_row(po: *const f32, pm: *const f32, pl: *const f32, qg: *const u16, out: *mut u16, len: u32, z: u32, maxc: u32) {
    let head = id::group_id(0) as usize;
    let d = id::local_id(0) as usize;
    let nch = (len + LC - 1) / LC;
    let cb = z as usize * maxc as usize;
    let mut m = f32::NEG_INFINITY;
    let mut c = 0usize;
    while c < nch as usize {
        m = math::fmax(m, *pm.add((cb + c) * NH + head));
        c += 1;
    }
    let mut den = 0.0f32;
    let mut acc = 0.0f32;
    let mut c = 0usize;
    while c < nch as usize {
        let w = math::exp(*pm.add((cb + c) * NH + head) - m);
        den = math::mad(*pl.add((cb + c) * NH + head), w, den);
        acc = math::mad(*po.add(((cb + c) * NH + head) * HD + d), w, acc);
        c += 1;
    }
    let gate = bf(*qg.add((z as usize * NH + head) * 2 * HD + HD + d));
    *out.add((z as usize * NH + head) * HD + d) = to_bf(bfr(math::div_ocl(acc, den)) * sigmoid(gate));
}

/// Merges the chunks of one query head (global max, rescaled sums, in chunk order) and applies the output gate:
/// out [row][24][256] = bf16(bf16(o / l) * sigmoid(gate)); qg [row][24][512] (query, gate); po [row][maxc][24][256],
/// pm / pl [row][maxc][24]; row z has len0 + z keys. Grid (24, rows), local 256 (one dim an item).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn attn_long_merge(po: *const f32, pm: *const f32, pl: *const f32, qg: *const u16, out: *mut u16, len0: u32, maxc: u32) {
    attr::sg_size::<32>(); // the reference compiles at SIMD32 (no required sub-group size)
    attr::reqd_work_group_size::<256, 1, 1>();
    let z = id::group_id(1);
    merge_row(po, pm, pl, qg, out, len0 + z, z, maxc);
}
