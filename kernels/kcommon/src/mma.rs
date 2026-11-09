//! Tensor-core and async-copy wrappers (sm_80+ forms; all valid on sm_121a).
//!
//! Fragment layouts are the PTX ISA ones for `mma.m16n8k16` (bf16/f16 inputs, f32 accumulate):
//! - A (16x16, row-major): 4 x b32 regs, each two consecutive-k elements. With `g = lane / 4`,
//!   `t = lane % 4`: a0 = (row g,   k 2t..2t+1), a1 = (row g+8, k 2t..2t+1),
//!   a2 = (row g,   k 2t+8..),    a3 = (row g+8, k 2t+8..).
//! - B (16x8, "col" = stored n-major/k-contiguous): b0 = (k 2t..2t+1, col g), b1 = (k 2t+8.., col g).
//! - C/D (16x8 f32): c0 = (row g, col 2t), c1 = (row g, col 2t+1), c2 = (row g+8, col 2t), c3 = (row g+8, col 2t+1).
//!
//! `ldmatrix.x4` with lane `l` pointing at row `l % 16` of a 16x16 bf16 tile, column block `(l / 16) * 8`,
//! yields exactly the A fragment; `ldmatrix.x2` on an n-major [8][16] tile (lanes 0..16 address rows
//! `l % 8`, col block `(l / 8) * 8`) yields the B fragment.
use core::arch::asm;

/// D = A * B + D, bf16 inputs, f32 accumulators.
#[inline(always)]
pub fn mma_bf16_16816(d: &mut [f32; 4], a: [u32; 4], b: [u32; 2]) {
    unsafe {
        asm!("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, {{{b0}, {b1}}}, {{{d0}, {d1}, {d2}, {d3}}};",
            d0 = inout(reg32) d[0], d1 = inout(reg32) d[1], d2 = inout(reg32) d[2], d3 = inout(reg32) d[3],
            a0 = in(reg32) a[0], a1 = in(reg32) a[1], a2 = in(reg32) a[2], a3 = in(reg32) a[3],
            b0 = in(reg32) b[0], b1 = in(reg32) b[1], options(nomem, nostack));
    }
}
/// D = A * B + D, f16 inputs, f32 accumulators.
#[inline(always)]
pub fn mma_f16_16816(d: &mut [f32; 4], a: [u32; 4], b: [u32; 2]) {
    unsafe {
        asm!("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, {{{b0}, {b1}}}, {{{d0}, {d1}, {d2}, {d3}}};",
            d0 = inout(reg32) d[0], d1 = inout(reg32) d[1], d2 = inout(reg32) d[2], d3 = inout(reg32) d[3],
            a0 = in(reg32) a[0], a1 = in(reg32) a[1], a2 = in(reg32) a[2], a3 = in(reg32) a[3],
            b0 = in(reg32) b[0], b1 = in(reg32) b[1], options(nomem, nostack));
    }
}

#[inline(always)]
pub fn ldmatrix_x4(addr: u32) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    unsafe {
        asm!("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{a}, {b}, {c}, {d}}}, [{p}];",
            a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d, p = in(reg32) addr,
            options(readonly, nostack));
    }
    [a, b, c, d]
}
#[inline(always)]
pub fn ldmatrix_x2(addr: u32) -> [u32; 2] {
    let (a, b): (u32, u32);
    unsafe {
        asm!("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{{a}, {b}}}, [{p}];",
            a = out(reg32) a, b = out(reg32) b, p = in(reg32) addr, options(readonly, nostack));
    }
    [a, b]
}
#[inline(always)]
pub fn ldmatrix_x1(addr: u32) -> u32 {
    let a: u32;
    unsafe {
        asm!("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {{{a}}}, [{p}];", a = out(reg32) a, p = in(reg32) addr,
            options(readonly, nostack));
    }
    a
}
/// Transposing variant (use for a k-major B tile stored [k][n]).
#[inline(always)]
pub fn ldmatrix_x4_trans(addr: u32) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    unsafe {
        asm!("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{{a}, {b}, {c}, {d}}}, [{p}];",
            a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d, p = in(reg32) addr,
            options(readonly, nostack));
    }
    [a, b, c, d]
}
#[inline(always)]
pub fn ldmatrix_x2_trans(addr: u32) -> [u32; 2] {
    let (a, b): (u32, u32);
    unsafe {
        asm!("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {{{a}, {b}}}, [{p}];",
            a = out(reg32) a, b = out(reg32) b, p = in(reg32) addr, options(readonly, nostack));
    }
    [a, b]
}

/// 16-byte global -> shared async copy, L2 only (`cp.async.cg`). `src` must be 16-byte aligned.
#[inline(always)]
pub unsafe fn cp_async_16(dst: u32, src: *const u8) {
    asm!("cp.async.cg.shared.global [{d}], [{s}], 16;", d = in(reg32) dst, s = in(reg64) src, options(nostack));
}
/// Same, but copies only `src_bytes` (0..=16) and zero-fills the rest (out-of-bounds tiles).
#[inline(always)]
pub unsafe fn cp_async_16_zfill(dst: u32, src: *const u8, src_bytes: u32) {
    asm!("cp.async.cg.shared.global [{d}], [{s}], 16, {n};", d = in(reg32) dst, s = in(reg64) src,
        n = in(reg32) src_bytes, options(nostack));
}
/// 4-byte async copy through L1 (`cp.async.ca`).
#[inline(always)]
pub unsafe fn cp_async_4(dst: u32, src: *const u8) {
    asm!("cp.async.ca.shared.global [{d}], [{s}], 4;", d = in(reg32) dst, s = in(reg64) src, options(nostack));
}
#[inline(always)]
pub fn cp_async_commit() {
    unsafe { asm!("cp.async.commit_group;", options(nostack)) }
}
/// Wait until at most `N` committed groups are still in flight.
#[inline(always)]
pub fn cp_async_wait<const N: u32>() {
    unsafe { asm!("cp.async.wait_group {n};", n = const N, options(nostack)) }
}
#[inline(always)]
pub fn cp_async_wait_all() {
    unsafe { asm!("cp.async.wait_all;", options(nostack)) }
}
