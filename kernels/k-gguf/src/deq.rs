//! Dequant kernels: `ggq_deq_<ty>_f32(dst, src, n_chunks)` / `..._bf16`. One thread per 8 output values
//! (`n_chunks = n_values / 8`); blocks are contiguous, so any whole-block range works (a tensor, a row range, an
//! expert slice). Grid: `ceil(n_chunks / 256)` x 256 threads.
use crate::dec::*;
use kcommon::*;

macro_rules! deq {
    ($f32name:ident, $bf16name:ident, $dec:ident, $vals:expr, $bytes:expr) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $f32name(dst: *mut f32, src: *const u8, n_chunks: u64) {
            let t = (bid_x() as u64) * (bdim_x() as u64) + tid_x() as u64;
            if t >= n_chunks {
                return;
            }
            const CPB: u64 = $vals / 8;
            let blk = src.add(((t / CPB) * $bytes) as usize);
            let y = $dec(blk, (t % CPB) as u32);
            let o = dst.add((t * 8) as usize) as *mut u32;
            stg_v4(o, [y[0].to_bits(), y[1].to_bits(), y[2].to_bits(), y[3].to_bits()]);
            stg_v4(o.add(4), [y[4].to_bits(), y[5].to_bits(), y[6].to_bits(), y[7].to_bits()]);
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $bf16name(dst: *mut u16, src: *const u8, n_chunks: u64) {
            let t = (bid_x() as u64) * (bdim_x() as u64) + tid_x() as u64;
            if t >= n_chunks {
                return;
            }
            const CPB: u64 = $vals / 8;
            let blk = src.add(((t / CPB) * $bytes) as usize);
            let y = $dec(blk, (t % CPB) as u32);
            let o = dst.add((t * 8) as usize) as *mut u32;
            stg_v4(o, [bf16::pack2(y[0], y[1]), bf16::pack2(y[2], y[3]), bf16::pack2(y[4], y[5]), bf16::pack2(y[6], y[7])]);
        }
    };
}

deq!(ggq_deq_q8_0_f32, ggq_deq_q8_0_bf16, dec_q8_0, 32, 34);
deq!(ggq_deq_q2_0_f32, ggq_deq_q2_0_bf16, dec_q2_0, 64, 18);
deq!(ggq_deq_iq4_nl_f32, ggq_deq_iq4_nl_bf16, dec_iq4_nl, 32, 18);
deq!(ggq_deq_iq4_xs_f32, ggq_deq_iq4_xs_bf16, dec_iq4_xs, 256, 136);
deq!(ggq_deq_q4_k_f32, ggq_deq_q4_k_bf16, dec_q4_k, 256, 144);
deq!(ggq_deq_q5_k_f32, ggq_deq_q5_k_bf16, dec_q5_k, 256, 176);
deq!(ggq_deq_q6_k_f32, ggq_deq_q6_k_bf16, dec_q6_k, 256, 210);
deq!(ggq_deq_iq2_s_f32, ggq_deq_iq2_s_bf16, dec_iq2_s, 256, 82);
deq!(ggq_deq_iq3_xxs_f32, ggq_deq_iq3_xxs_bf16, dec_iq3_xxs, 256, 98);
deq!(ggq_deq_iq3_s_f32, ggq_deq_iq3_s_bf16, dec_iq3_s, 256, 110);
deq!(ggq_deq_f32_f32, ggq_deq_f32_bf16, dec_f32, 8, 32);
deq!(ggq_deq_f16_f32, ggq_deq_f16_bf16, dec_f16, 8, 16);
deq!(ggq_deq_bf16_f32, ggq_deq_bf16_bf16, dec_bf16, 8, 16);
