//! KV cache q8 append: Rust port of arc-engine `kernels/xpu/qwen_kvq.cl` kvq_quant8. bf16 rows [rows][4 kv heads][256]
//! -> records [pos][kv head][272 B]: 8 blocks of 32 dims as int8, then the 8 fp16 block scales (bytes 256..271).
//! d = fp16_rte(amax / 127) (2.5-ulp division), q = clamp(rint(v * (1 / d)), -127, 127), 1 / d also a 2.5-ulp division.
//! The reference module has no ContractionOff, but kvq_quant8 has no a * b + c to contract (its two mads are in
//! kvq_quant4).
use crate::bf;
use crate::ext::{fabs, rint};
use kxcommon::{attr, bits, cvt, id, math, unroll};

/// One work-item a (row, head, block); global size rows * 32; row r goes to position pos0 + r. Grid (rows), local 32.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn kvq_quant8(src: *const u16, dst: *mut u8, pos0: u32) {
    attr::sg_size::<32>(); // the reference compiles at SIMD32 (no required sub-group size)
    let g = id::global_id(0) as u32;
    let (r, h, b) = (g >> 5, (g >> 3) & 3, g & 7);
    let x = src.add((r as usize * 4 + h as usize) * 256 + b as usize * 32);
    let mut v = [0f32; 32];
    let mut amax = 0.0f32;
    unroll!(32, 32, |i| {
        v[i] = bf(*x.add(i));
        amax = math::fmax(amax, fabs(v[i]));
    });
    let dh = cvt::f16_rte(math::div_ocl(amax, 127.0));
    let hb: u16 = bits(dh);
    let d = dh as f32;
    let inv = if d != 0.0 { math::div_ocl(1.0, d) } else { 0.0 };
    let rec = dst.add(((pos0 + r) as usize * 4 + h as usize) * 272);
    unroll!(8, 8, |i4| {
        let i = 4 * i4;
        let mut w = 0u32;
        unroll!(4, 4, |j| {
            let q = rint(v[i + j] * inv).to_int_unchecked::<i32>().clamp(-127, 127);
            w |= (q as i8 as u8 as u32) << (8 * j);
        });
        *(rec.add(b as usize * 32 + i) as *mut u32) = w;
    });
    *(rec.add(256 + 2 * b as usize) as *mut u16) = hb;
}
