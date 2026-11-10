//! Embedding, SwiGLU, bf16 rounding of logits, greedy argmax: Rust port of the qwen_basic.cl kernels on the default
//! path. No contraction sites (the reference has none in these kernels).
use crate::ext::reduce_max_u64;
use crate::{bf, bfr, silu, to_bf};
use kxcommon::{attr, id, slm};

/// Plain bf16 embedding rows by token id: y[t][0..dim] = w[ids[t]][0..dim]. Grid (rows), local 64.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn embed_bf16(w: *const u16, ids: *const u32, y: *mut u16, dim: u32) {
    attr::sg_size::<32>(); // the reference compiles at SIMD32 (no required sub-group size)
    let t = id::group_id(0);
    let row = *ids.add(t as usize) as usize;
    let mut i = id::local_id(0);
    while i < dim {
        *y.add((t * dim + i) as usize) = *w.add(row * dim as usize + i as usize);
        i += 64;
    }
}

/// act = bf16(silu(gate) * up), gate and up bf16 [n]. Grid (n / 64), local 64.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn swiglu(g: *const u16, u: *const u16, act: *mut u16, n: u32) {
    attr::sg_size::<32>(); // the reference compiles at SIMD32 (no required sub-group size)
    let i = id::global_id(0);
    if i >= n as usize {
        return;
    }
    let a = bf(*g.add(i));
    *act.add(i) = to_bf(silu(a) * bf(*u.add(i)));
}

/// In place: fp32 values rounded to bf16 and widened back; NaN stays NaN. Grid (n / 64), local 64.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn round_bf16_f32(x: *mut f32, n: u32) {
    attr::sg_size::<32>(); // the reference compiles at SIMD32 (no required sub-group size)
    let i = id::global_id(0);
    if i < n as usize && !(*x.add(i)).is_nan() {
        *x.add(i) = bfr(*x.add(i));
    }
}

/// Candidate (key << 32) | ~index: a plain unsigned max picks the first NaN, else the largest value (+0 == -0), ties to
/// the lowest index; 0 is the empty candidate.
#[inline(always)]
fn argmax_pack(f: f32, idx: u32) -> u64 {
    let mut w = f.to_bits();
    let key = if (w & 0x7fff_ffff) > 0x7f80_0000 {
        0xffff_ffff
    } else {
        if (w & 0x7fff_ffff) == 0 {
            w = 0;
        }
        if w & 0x8000_0000 != 0 { !w } else { w ^ 0x8000_0000 }
    };
    ((key as u64) << 32) | (!idx) as u64
}

/// Work-group max of 256 items (16 sub-groups): sub-group reduce, then the 16 parts in order.
#[inline(always)]
unsafe fn wg_max(v: u64, tmp: *mut u64) -> u64 {
    let v = reduce_max_u64(v);
    if id::lane() == 0 {
        *tmp.add(id::sg_id() as usize) = v;
    }
    slm::barrier();
    let mut r = 0u64;
    let mut i = 0;
    while i < 16 {
        r = r.max(*tmp.add(i));
        i += 1;
    }
    r
}

/// Pass 1: work-group (part, row) of 256 items scans x[row][p * per_part .. min(n, (p + 1) * per_part)).
/// Grid (parts, rows), local 256.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn argmax_partial(x: *const f32, part: *mut u64, n: u32, per_part: u32) {
    let tmp = slm::alloc::<u64, 16, 0>();
    let p = id::group_id(0);
    let row = id::group_id(1);
    let lo = p * per_part;
    let hi = n.min(lo + per_part);
    let mut best = 0u64;
    let mut i = lo + id::local_id(0);
    while i < hi {
        best = best.max(argmax_pack(*x.add(row as usize * n as usize + i as usize), i));
        i += 256;
    }
    let best = wg_max(best, tmp);
    if id::local_id(0) == 0 {
        *part.add((row * id::num_groups(0) + p) as usize) = best;
    }
}

/// Pass 2: one work-group a row merges the parts and writes the winning index. Grid (rows), local 256.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn argmax_final(part: *const u64, out: *mut i32, parts: u32) {
    let tmp = slm::alloc::<u64, 16, 0>();
    let row = id::group_id(0);
    let mut best = 0u64;
    let mut i = id::local_id(0);
    while i < parts {
        best = best.max(*part.add((row * parts + i) as usize));
        i += 256;
    }
    let best = wg_max(best, tmp);
    if id::local_id(0) == 0 {
        *out.add(row as usize) = (!(best as u32)) as i32;
    }
}
