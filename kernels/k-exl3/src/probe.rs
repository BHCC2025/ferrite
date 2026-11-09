//! Bandwidth probes (test-only): the streaming roof of the device for the GEMV's access sizes.

use kcommon::{bdim_x, bid_x, gdim_x, ldg_stream_v4, tid_x};

/// Grid-stride 16-byte streaming loads over `words` u32 (multiple of 4); one xor per thread to `out`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn exl3_probe_v4(p: *const u32, words: u32, out: *mut u32) {
    let mut acc = 0u32;
    let mut i = (bid_x() * bdim_x() + tid_x()) * 4;
    let step = gdim_x() * bdim_x() * 4;
    while i < words {
        let v = ldg_stream_v4(p.add(i as usize));
        acc ^= v[0] ^ v[1] ^ v[2] ^ v[3];
        i += step;
    }
    if acc == 0x1234_5678 {
        *out = acc;
    }
}

/// The same with 4-byte loads, 4 independent a thread an iteration (the GEMV's load size).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn exl3_probe_u32(p: *const u32, words: u32, out: *mut u32) {
    let mut acc = 0u32;
    let nt = gdim_x() * bdim_x();
    let mut i = bid_x() * bdim_x() + tid_x();
    while i + 3 * nt < words {
        let a = crate::util::ldg_stream(p.add(i as usize));
        let b = crate::util::ldg_stream(p.add((i + nt) as usize));
        let c = crate::util::ldg_stream(p.add((i + 2 * nt) as usize));
        let d = crate::util::ldg_stream(p.add((i + 3 * nt) as usize));
        acc ^= a ^ b ^ c ^ d;
        i += 4 * nt;
    }
    if acc == 0x1234_5678 {
        *out = acc;
    }
}

macro_rules! op_probe {
    ($name:ident, $asm:literal) => {
        /// 8 independent chains of `iters` dependent ops a thread (instruction throughput probe).
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(out: *mut u32, iters: u32, seed: u32) {
            let mut x = [seed ^ tid_x(), seed + 1, seed + 2, seed + 3, seed + 4, seed + 5, seed + 6, seed + 7];
            let mut i = 0;
            while i < iters {
                crate::gemv::unroll_pub!(8, j, {
                    core::arch::asm!($asm, x = inout(reg32) x[j], c = in(reg32) 0x83DC_D12Du32, options(pure, nomem, nostack));
                });
                i += 1;
            }
            let r = x[0] ^ x[1] ^ x[2] ^ x[3] ^ x[4] ^ x[5] ^ x[6] ^ x[7];
            if r == 0x1234_5678 {
                *out = r;
            }
        }
    };
}
op_probe!(exl3_probe_imad, "mad.lo.u32 {x}, {x}, {c}, {c};");
op_probe!(exl3_probe_idp, "dp4a.u32.u32 {x}, {x}, {c}, {x};");
op_probe!(exl3_probe_lop, "lop3.b32 {x}, {x}, {c}, 0x0f0f0f0f, 0x6a;");
op_probe!(exl3_probe_iadd, "add.u32 {x}, {x}, {c};");
op_probe!(exl3_probe_shf, "shf.l.wrap.b32 {x}, {x}, {c}, {x};");
op_probe!(exl3_probe_hfma2, "fma.rn.f16x2 {x}, {x}, {c}, {c};");
op_probe!(exl3_probe_ffma, "fma.rn.f32 {x}, {x}, {c}, {c};");
op_probe!(exl3_probe_prmt, "prmt.b32 {x}, {x}, {c}, 0x5410;");
