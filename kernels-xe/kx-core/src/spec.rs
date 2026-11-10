//! Ranked int8 draft head of the MTP drafter: Rust port of arc-engine `kernels/xpu/spec_head.cl` (dhead_q8b, dhead_final).
//! logits[r] = scale[r] * sum_k q[r][k] x[k] over the selected rows, then argmax_partial (basic.rs) and dhead_final,
//! which maps the winning row back to its token id. No contraction sites (the reference has none: dot16 is OpDot x 4).
use crate::bf;
use crate::ext::{dot4, reduce_max_u64};
use kxcommon::{blk, id, sg, slm, unroll};

const DHB_ROWS: usize = 8;

unsafe extern "C" {
    fn _Z31intel_sub_group_block_read_us16PU3AS1Kt(p: *const u16) -> kxcommon::v::U16x16;
}

/// dot16 of spec_head.cl: ((dot(0..3) + dot(4..7)) + dot(8..11)) + dot(12..15).
#[inline(always)]
fn dot16(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    let d = |i: usize| dot4([a[i], a[i + 1], a[i + 2], a[i + 3]], [b[i], b[i + 1], b[i + 2], b[i + 3]]);
    d(0) + d(4) + d(8) + d(12)
}

/// 8 rows a 16-lane sub-group, 4 sub-groups a work-group; per 256-column block lane l holds columns 16 j + l of the
/// row (sub-group block reads) and of x. k a multiple of 256. Grid ((n + 31) / 32), local 64.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn dhead_q8b(q: *const i8, scale: *const f32, x: *const u16, logits: *mut f32, n: u32, k: u32) {
    let row0 = (id::group_id(0) * 4 + id::sg_id()) * DHB_ROWS as u32;
    if row0 >= n {
        return;
    }
    let lane = id::lane();
    let mut acc = [0f32; DHB_ROWS];
    let rk = k as usize;
    let nr = (DHB_ROWS as u32).min(n - row0) as usize;
    let mut b = 0usize;
    while b < rk {
        let xu = _Z31intel_sub_group_block_read_us16PU3AS1Kt(x.add(b)).to_array();
        let mut xf = [0f32; 16];
        unroll!(16, 16, |j| xf[j] = bf(xu[j]));
        unroll!(8, DHB_ROWS, |r| {
            if r < nr {
                let u = blk::global::read_u8x16((q as *const u8).add((row0 as usize + r) * rk + b)).to_array();
                let mut qf = [0f32; 16];
                unroll!(16, 16, |j| qf[j] = u[j] as i8 as f32);
                acc[r] += dot16(&qf, &xf);
            }
        });
        b += 256;
    }
    unroll!(8, DHB_ROWS, |r| {
        let s = sg::reduce_add_f32(acc[r]);
        if lane == 0 && r < nr {
            *logits.add(row0 as usize + r) = s * *scale.add(row0 as usize + r);
        }
    });
}

/// Argmax pass 2 over the parts of argmax_partial with the row index mapped to its token id: out[0] = ids[argmax].
/// Grid (1), local 256.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn dhead_final(part: *const u64, out: *mut i32, parts: u32, ids: *const u32) {
    let tmp = slm::alloc::<u64, 16, 0>();
    let mut best = 0u64;
    let mut i = id::local_id(0);
    while i < parts {
        best = best.max(*part.add(i as usize));
        i += 256;
    }
    let v = reduce_max_u64(best);
    if id::lane() == 0 {
        *tmp.add(id::sg_id() as usize) = v;
    }
    slm::barrier();
    let mut r = 0u64;
    let mut j = 0;
    while j < 16 {
        r = r.max(*tmp.add(j));
        j += 1;
    }
    if id::local_id(0) == 0 {
        *out = *ids.add(!(r as u32) as usize) as i32;
    }
}
