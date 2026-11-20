//! DeepSeek-V4.1 Engram gate (layers 1 and 14, before the attention mixes). Built to one PTX module (`k_engram`) by
//! crates/fe-kbuild; loaded by fe-engram's `gpu` module.
#![no_std]
#![feature(abi_ptx, asm_experimental_arch)]
#![allow(internal_features, clippy::missing_safety_doc)]

extern crate kcommon;

use kcommon::*;

/// The engram.wkv output of a row, element `i` of `w` = (hc + 1) * D: the rank partials summed in rank order and
/// rounded to bf16 (TensorFold's add2Bf16 then its gate), or the bf16 value itself when `world` is 0.
#[inline(always)]
unsafe fn kv_at(kvp: *const f32, kvb: *const u16, world: u32, part: usize, base: usize, i: usize) -> f32 {
    if world == 0 {
        bf16::to_f32(*kvb.add(base + i))
    } else {
        let mut s = *kvp.add(base + i);
        let mut r = 1;
        while r < world {
            s += *kvp.add(r as usize * part + base + i);
            r += 1;
        }
        bf16::round(s)
    }
}

/// One block a (row, stream): grid (rows, hc), any multiple of 32 threads (256 used).
///
/// - `h` [rows, hc, d] bf16: the residual streams; `out` the same shape (may not alias `h`).
/// - `kvp` f32 [world, rows, (hc + 1) d]: the ranks' engram.wkv partials (hc keys, then the value), or with
///   `world` = 0, `kvb` bf16 [rows, (hc + 1) d] already summed.
/// - `qk` f32 [hc, d]: q_weight * k_weight.
///
/// gate = sigmoid(signed sqrt(max(|dot|, 1e-6))), dot = <h * qk, key> * rstd(h) * rstd(key) / sqrt(d);
/// out = bf16(h + gate * value).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn engram_gate(
    out: *mut u16,
    h: *const u16,
    kvp: *const f32,
    kvb: *const u16,
    qk: *const f32,
    rows: u32,
    hc: u32,
    d: u32,
    world: u32,
    eps: f32,
) {
    let scratch = static_shared!(eg_red, f32, 3 * 32 * 4);
    let r = bid_x() as usize;
    let s = bid_y() as usize;
    let d = d as usize;
    let hc = hc as usize;
    let w = (hc + 1) * d;
    let part = rows as usize * w;
    let hrow = h.add((r * hc + s) * d);
    let q = qk.add(s * d);
    let kbase = r * w + s * d;
    let vbase = r * w + hc * d;
    let (mut sh, mut sk, mut sd) = (0.0f32, 0.0f32, 0.0f32);
    let mut i = tid_x() as usize;
    while i < d {
        let x = bf16::to_f32(*hrow.add(i));
        let k = kv_at(kvp, kvb, world, part, kbase, i);
        sh = math::fma(x, x, sh);
        sk = math::fma(k, k, sk);
        sd = math::fma(x * *q.add(i), k, sd);
        i += bdim_x() as usize;
    }
    let sh = warp::block_sum(sh, scratch);
    let sk = warp::block_sum(sk, scratch.add(32));
    let sd = warp::block_sum(sd, scratch.add(64));
    let df = d as f32;
    let rstd = math::rcp(math::sqrt(sh / df + eps)) * math::rcp(math::sqrt(sk / df + eps));
    let dot = sd * rstd * math::rcp(math::sqrt(df));
    let mag = math::sqrt(math::maxf(dot.abs(), 1e-6));
    let sg = if dot < 0.0 { -mag } else { mag };
    let gate = math::rcp(1.0 + math::exp(-sg));
    let orow = out.add((r * hc + s) * d);
    let mut i = tid_x() as usize;
    while i < d {
        let x = bf16::to_f32(*hrow.add(i));
        let v = kv_at(kvp, kvb, world, part, vbase, i);
        *orow.add(i) = bf16::from_f32(math::fma(gate, v, x));
        i += bdim_x() as usize;
    }
}

/// Image-span rows pass unchanged (Engram's gate is shut there): out[r] = h[r] for rows with mask[r] != 0.
/// grid (rows), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn engram_keep_rows(out: *mut u16, h: *const u16, mask: *const u32, row_len: u32) {
    let r = bid_x() as usize;
    if *mask.add(r) == 0 {
        return;
    }
    let n = row_len as usize;
    let mut i = tid_x() as usize;
    while i < n {
        *out.add(r * n + i) = *h.add(r * n + i);
        i += bdim_x() as usize;
    }
}
