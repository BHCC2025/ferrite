//! G16 build: re-quantize the MXFP8 experts of layers 20-23 to NVFP4 in the release's NVFP4 layout (row-major codes
//! [e][row][K/2], e4m3 scales [e][row][K/16], per-expert amax with outer scale denominator 2688 = 6 * 448), following the
//! format definition: u = amax / 2688 over the expert's whole matrix, block scale
//! s = u * e4m3(bmax / div / u) for div in (6, 4), the candidate with the smaller sum of absolute errors wins (ties: 6).
use super::*;

/// Dequantized MXFP8 element: e4m3 value * 2^(s - 127), exact in f32.
#[inline(always)]
fn mx_val(v: u8, s: u8) -> f32 {
    let q = e4m3x4(v as u32)[0];
    q * ue8m0(s as u32)
}

/// Per-block partial max of |w| of expert e over its rows b, b + 64, ...: part[e][b]. grid = (128, 64), block = 256.
/// w8 = [128][rows][k] e4m3, sc = [128][rows][k / 32] ue8m0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_rq_amax_part(part: *mut f32, w8: *const u8, sc: *const u8, rows: u32, k: u32) {
    let red = static_shared!(rq_red, f32, 128);
    let e = bid_x() as usize;
    let mut m = 0.0f32;
    let mut r = bid_y();
    while r < rows {
        let row = (e * rows as usize + r as usize) * k as usize;
        let srow = (e * rows as usize + r as usize) * (k as usize / 32);
        let mut i = tid_x();
        while i < k {
            m = m.max(mx_val(*w8.add(row + i as usize), *sc.add(srow + i as usize / 32)).abs());
            i += 256;
        }
        r += 64;
    }
    let t = warp::block_max(m, red);
    if tid_x() == 0 {
        *part.add(e * 64 + bid_y() as usize) = t;
    }
}

/// amax[e] = max over the 64 partials. grid = 128, block = 64.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_rq_amax_fin(amax: *mut f32, part: *const f32) {
    let red = static_shared!(rq_red2, f32, 128);
    let e = bid_x() as usize;
    let t = warp::block_max(*part.add(e * 64 + tid_x() as usize), red);
    if tid_x() == 0 {
        *amax.add(e) = t;
    }
}

/// Quantize: one thread per 16-value block. grid = (ceil(rows * k / 16 / 256), 128), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_rq_quant(codes: *mut u8, sf: *mut u8, w8: *const u8, sc: *const u8, amax: *const f32,
    rows: u32, k: u32) {
    let e = bid_y() as usize;
    let kb = (k / 16) as usize;
    let blk = (bid_x() * 256 + tid_x()) as usize; // block index inside the expert
    if blk >= rows as usize * kb {
        return;
    }
    let r = blk / kb;
    let b = blk % kb;
    let u = *amax.add(e) / 2688.0;
    let base = (e * rows as usize + r) * k as usize + b * 16;
    let s = *sc.add((e * rows as usize + r) * (k as usize / 32) + b / 2);
    let raw = *(w8.add(base) as *const [u32; 4]);
    let mut x = [0.0f32; 16];
    kcommon::unroll!(q in 4 => {
        let v = e4m3x4(raw[q]);
        kcommon::unroll!(j in 4 => {
            x[q * 4 + j] = v[j] * ue8m0(s as u32);
        });
    });
    let mut bm = 0.0f32;
    kcommon::unroll!(i in 16 => {
        bm = bm.max(x[i].abs());
    });
    let mut best_err = 0.0f32;
    let mut best_bits = 0u32;
    let mut best_codes = [0u32; 2];
    kcommon::unroll!(c in 2 => {
        let div = if c == 0 { 6.0f32 } else { 4.0f32 };
        let q8 = to_e4m3(bm / div / u);
        let mut sv = u * e4m3_to_f32(q8);
        if !(sv > 0.0) {
            sv = u * (1.0 / 512.0);
        }
        let mut a = [0.0f32; 8];
        let mut bb = [0.0f32; 8];
        kcommon::unroll!(j in 8 => {
            a[j] = x[j] / sv;
            bb[j] = x[8 + j] / sv;
        });
        let c0 = to_e2m1x8(a);
        let c1 = to_e2m1x8(bb);
        // reconstruction error
        let da = e2m1x8(c0);
        let db = e2m1x8(c1);
        let mut err = 0.0f32;
        kcommon::unroll!(j in 8 => {
            err += (da[j] * sv - x[j]).abs();
            err += (db[j] * sv - x[8 + j]).abs();
        });
        if c == 0 || err < best_err {
            best_err = err;
            best_bits = q8;
            best_codes = [c0, c1];
        }
    });
    let o = codes.add((e * rows as usize + r) * (k as usize / 2) + b * 8) as *mut u32;
    *o = best_codes[0];
    *o.add(1) = best_codes[1];
    *sf.add((e * rows as usize + r) * kb + b) = best_bits as u8;
}
