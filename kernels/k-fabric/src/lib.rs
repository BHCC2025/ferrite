//! k-fabric: the GPU side of the two-rank RoCE one-shot exchange (after b12x's "RoCEnante", Apache-2.0, as
//! TensorFold's roce_gather.cu): (1) every block stages its share of this rank's f32 partial into the pinned
//! send slot, (2) the last block to stage rings the proxy's doorbell (ctrl.seq) after a system fence, the host
//! proxy RDMA-writes the slot to the peer and then a sequence flag on the same queue pair, (3) every block waits
//! for the peer's flag, (4) out = bf16([resid +] (rank 0's partial + rank 1's)) — identical bits on both ranks,
//! (5) the last block to finish advances the device epoch, so a CUDA graph replays with fresh sequences.
//! A wait past the spin limit poisons the exchange (later launches return at once; the host reports it).
#![no_std]
#![feature(abi_ptx, asm_experimental_arch)]
#![allow(internal_features, clippy::missing_safety_doc)]

extern crate kcommon;

use core::arch::asm;
use kcommon::{bdim_x, bid_x, gdim_x, static_shared, sync_threads, tid_x};

// ctrl words (pinned memory the proxy polls), u32
const C_SEQ: usize = 0;
const C_NBYTES: usize = 1; // + slot
const C_ERROR: usize = 3;
// device counters, u32: epoch, poison, then stage and tail arrivals for each grid-size class
const K_EPOCH: usize = 0;
const K_POISON: usize = 1;
const K_CLASSES: usize = 4;
const K_STAGE: usize = 2;
const K_TAIL: usize = 2 + K_CLASSES;

#[inline(always)]
unsafe fn ld_relaxed_gpu(p: *const u32) -> u32 {
    let v: u32;
    asm!("ld.relaxed.gpu.global.u32 {0}, [{1}];", out(reg32) v, in(reg64) p, options(nostack));
    v
}
#[inline(always)]
unsafe fn ld_acquire_sys(p: *const u32) -> u32 {
    let v: u32;
    asm!("ld.acquire.sys.global.u32 {0}, [{1}];", out(reg32) v, in(reg64) p, options(nostack));
    v
}
#[inline(always)]
unsafe fn ld_relaxed_sys_v4(p: *const u32) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    asm!("ld.relaxed.sys.global.v4.u32 {{{0}, {1}, {2}, {3}}}, [{4}];", out(reg32) a, out(reg32) b, out(reg32) c,
         out(reg32) d, in(reg64) p, options(nostack));
    [a, b, c, d]
}
#[inline(always)]
unsafe fn st_relaxed_sys(p: *mut u32, v: u32) {
    asm!("st.relaxed.sys.global.u32 [{0}], {1};", in(reg64) p, in(reg32) v, options(nostack));
}
#[inline(always)]
unsafe fn st_release_gpu(p: *mut u32, v: u32) {
    asm!("st.release.gpu.global.u32 [{0}], {1};", in(reg64) p, in(reg32) v, options(nostack));
}
#[inline(always)]
unsafe fn arrive(p: *mut u32) -> u32 {
    let old: u32;
    asm!("atom.relaxed.gpu.global.add.u32 {0}, [{1}], 1;", out(reg32) old, in(reg64) p, options(nostack));
    old
}
#[inline(always)]
fn fence_sys() {
    unsafe { asm!("fence.sc.sys;", options(nostack)) };
}
#[inline(always)]
fn fence_gpu() {
    unsafe { asm!("fence.sc.gpu;", options(nostack)) };
}
#[inline(always)]
fn bf2(lo: f32, hi: f32) -> u32 {
    let r: u32;
    unsafe { asm!("cvt.rn.bf16x2.f32 {0}, {1}, {2};", out(reg32) r, in(reg32) hi, in(reg32) lo, options(pure, nomem, nostack)) };
    r
}

/// One exchange + sum (`klass`: the launch's grid-size class, 0..4; `split_min`: the proxy's split threshold in bytes,
/// fe_fabric::roce::SPLIT_MIN with two queue pairs, else u32::MAX). `inp`: this rank's f32 partial, `packs`
/// 16-byte packs (4 floats each); `out`: bf16
/// [packs * 4]; `resid`: bf16 added after the rank sum (null: none; may equal `out`). `region`: the pinned
/// region (recv slots at 0, flags at 2 * slot_bytes, send slots at 2 * slot_bytes + 256, ctrl after them).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fab_gather_sum_bf16(
    inp: *const u32,
    out: *mut u32,
    resid: *const u32,
    packs: u32,
    rank: u32,
    region: *mut u8,
    slot_bytes: u32,
    counters: *mut u32,
    spin_limit: u32,
    klass: u32,
    split_min: u32,
) {
    let flag_sh = static_shared!(fab_ok, u32, 4);
    gather_sum::<false>(inp, out, resid, packs, rank, region, slot_bytes, counters, spin_limit, klass, split_min, flag_sh);
}

/// fab_gather_sum_bf16 for bf16 partials: `packs` 16-byte packs of 8 bf16, out / resid bf16 [packs * 8]; each
/// output = bf16((rank 0's + rank 1's) [+ resid]) in f32. Half the bytes of the f32 exchange.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fab_gather_sum_bf16in(
    inp: *const u32,
    out: *mut u32,
    resid: *const u32,
    packs: u32,
    rank: u32,
    region: *mut u8,
    slot_bytes: u32,
    counters: *mut u32,
    spin_limit: u32,
    klass: u32,
    split_min: u32,
) {
    let flag_sh = static_shared!(fab_ok16, u32, 4);
    gather_sum::<true>(inp, out, resid, packs, rank, region, slot_bytes, counters, spin_limit, klass, split_min, flag_sh);
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn gather_sum<const BF16: bool>(
    inp: *const u32,
    out: *mut u32,
    resid: *const u32,
    packs: u32,
    rank: u32,
    region: *mut u8,
    slot_bytes: u32,
    counters: *mut u32,
    spin_limit: u32,
    klass: u32,
    split_min: u32,
    flag_sh: kcommon::smem::Shared<u32>,
) {
    let klass = klass as usize; // one per grid size: arrivals count modulo the grid
    if ld_relaxed_gpu(counters.add(K_POISON)) != 0 {
        return;
    }
    let seq = ld_relaxed_gpu(counters.add(K_EPOCH)) + 1;
    let slot = seq & 1;
    let recv = region.add((slot * slot_bytes) as usize) as *const u32;
    let flags = region.add((2 * slot_bytes + slot * 128) as usize) as *const u32;
    let send = region.add((2 * slot_bytes + 256 + slot * slot_bytes) as usize) as *mut u32;
    let ctrl = region.add((4 * slot_bytes + 256) as usize) as *mut u32;
    let at = bid_x() * bdim_x() + tid_x();
    let stride = gdim_x() * bdim_x();
    // 1. stage
    let mut i = at;
    while i < packs {
        let v = kcommon::ldg_nc_v4(inp.add(4 * i as usize));
        kcommon::stg_v4(send.add(4 * i as usize), v);
        i += stride;
    }
    sync_threads();
    // 2. the doorbell, by the last block to stage
    if tid_x() == 0 {
        fence_sys();
        if (arrive(counters.add(K_STAGE + klass)) + 1) % gdim_x() == 0 {
            st_relaxed_sys(ctrl.add(C_NBYTES + slot as usize), packs * 16);
            fence_sys();
            st_relaxed_sys(ctrl.add(C_SEQ), seq);
        }
        // 3. the peer's flag(s): above split_min bytes its proxy sent halves over two queue pairs, each flagged
        // (at + 64 bytes for the second)
        let mut polls = 0u32;
        let mut ok = 1u32;
        let nflags = if packs * 16 > split_min { 2 } else { 1 };
        let mut f = 0;
        while f < nflags {
            while ld_acquire_sys(flags.add(16 * f)) != seq {
                polls += 1;
                if polls >= spin_limit {
                    st_relaxed_sys(ctrl.add(C_ERROR), seq);
                    st_release_gpu(counters.add(K_POISON), seq);
                    ok = 0;
                    break;
                }
            }
            f += 1;
        }
        flag_sh.st(ok);
    }
    sync_threads();
    if flag_sh.ld() != 0 {
        // 4. rank 0's partial + rank 1's, then the residual
        let mut i = at;
        while i < packs {
            let own = kcommon::ldg_nc_v4(inp.add(4 * i as usize));
            let peer = ld_relaxed_sys_v4(recv.add(4 * i as usize));
            let (a, b) = if rank == 0 { (own, peer) } else { (peer, own) };
            if BF16 {
                let mut s = [0f32; 8];
                for q in 0..4 {
                    s[2 * q] = f32::from_bits(a[q] << 16) + f32::from_bits(b[q] << 16);
                    s[2 * q + 1] = f32::from_bits(a[q] & 0xffff_0000) + f32::from_bits(b[q] & 0xffff_0000);
                }
                if !resid.is_null() {
                    let r = *(resid.add(4 * i as usize) as *const [u32; 4]);
                    for q in 0..4 {
                        s[2 * q] += f32::from_bits(r[q] << 16);
                        s[2 * q + 1] += f32::from_bits(r[q] & 0xffff_0000);
                    }
                }
                *(out.add(4 * i as usize) as *mut [u32; 4]) =
                    [bf2(s[0], s[1]), bf2(s[2], s[3]), bf2(s[4], s[5]), bf2(s[6], s[7])];
            } else {
                let mut s = [0f32; 4];
                for q in 0..4 {
                    s[q] = f32::from_bits(a[q]) + f32::from_bits(b[q]);
                }
                if !resid.is_null() {
                    let r = *(resid.add(2 * i as usize) as *const [u32; 2]);
                    s[0] += f32::from_bits(r[0] << 16);
                    s[1] += f32::from_bits(r[0] & 0xffff_0000);
                    s[2] += f32::from_bits(r[1] << 16);
                    s[3] += f32::from_bits(r[1] & 0xffff_0000);
                }
                *(out.add(2 * i as usize) as *mut [u32; 2]) = [bf2(s[0], s[1]), bf2(s[2], s[3])];
            }
            i += stride;
        }
    }
    // 5. the epoch, by the last block to finish
    fence_gpu();
    sync_threads();
    if tid_x() == 0 && (arrive(counters.add(K_TAIL + klass)) + 1) % gdim_x() == 0 {
        fence_gpu();
        if ld_relaxed_gpu(counters.add(K_POISON)) == 0 {
            st_release_gpu(counters.add(K_EPOCH), seq);
        }
    }
}

/// NCCL fallback's epilogue: out = bf16([resid +] x) for an f32 all-reduced x.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fab_to_bf16(x: *const u32, out: *mut u32, resid: *const u32, packs: u32) {
    let mut i = bid_x() * bdim_x() + tid_x();
    while i < packs {
        let v = kcommon::ldg_nc_v4(x.add(4 * i as usize));
        let mut s = [f32::from_bits(v[0]), f32::from_bits(v[1]), f32::from_bits(v[2]), f32::from_bits(v[3])];
        if !resid.is_null() {
            let r = *(resid.add(2 * i as usize) as *const [u32; 2]);
            s[0] += f32::from_bits(r[0] << 16);
            s[1] += f32::from_bits(r[0] & 0xffff_0000);
            s[2] += f32::from_bits(r[1] << 16);
            s[3] += f32::from_bits(r[1] & 0xffff_0000);
        }
        *(out.add(2 * i as usize) as *mut [u32; 2]) = [bf2(s[0], s[1]), bf2(s[2], s[3])];
        i += gdim_x() * bdim_x();
    }
}

/// Logits gather epilogue: out[r][rank * half + c] = gathered[rank][r][c] for the NCCL all-gather of each
/// rank's vocabulary half ([2][rows][half] -> [rows][2 * half]). grid = (ceil(half / 2048), rows, 2), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fab_interleave_halves(out: *mut u16, gathered: *const u16, rows: u32, half: u32) {
    let r = kcommon::bid_y();
    let rank = kcommon::bid_z();
    let src = gathered.add(((rank * rows + r) * half) as usize);
    let dst = out.add((r * 2 * half + rank * half) as usize);
    let mut c = bid_x() * 2048 + tid_x() * 8;
    let end = if (bid_x() + 1) * 2048 < half { (bid_x() + 1) * 2048 } else { half };
    while c + 8 <= end {
        let v = kcommon::ldg_nc_v4(src.add(c as usize) as *const u32);
        kcommon::stg_v4(dst.add(c as usize) as *mut u32, v);
        c += 2048;
    }
}

/// NCCL fallback for bf16 partials (prompt rows): x = bf16(x + y) with y the all-reduced bf16 sum.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fab_add_bf16(x: *mut u32, y: *const u32, pairs: u32) {
    let mut i = bid_x() * bdim_x() + tid_x();
    while i < pairs {
        let (a, b) = (*x.add(i as usize), *y.add(i as usize));
        *x.add(i as usize) = bf2(
            f32::from_bits(a << 16) + f32::from_bits(b << 16),
            f32::from_bits(a & 0xffff_0000) + f32::from_bits(b & 0xffff_0000),
        );
        i += gdim_x() * bdim_x();
    }
}
