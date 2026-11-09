//! M0 proof kernels: shared memory (static + dynamic), block reductions, cp.async, ldmatrix, mma.sync.
use kcommon::mma::*;
use kcommon::smem::{dyn_base, Shared};
use kcommon::*;

/// out[i] = in[rev(i)] (through a static shared tile) + 2*in[rev(i)] (through dynamic shared),
/// and out[n + block] = sum of the block's inputs. blockDim = 256, dynamic smem >= 1024 + 128 bytes.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_t_smem(out: *mut f32, inp: *const f32, n: u32) {
    let t = tid_x();
    let base = bid_x() * 256;
    let tile: Shared<f32> = static_shared!(t_tile, f32, 1024);
    let dynm: Shared<f32> = dyn_base();
    let red: Shared<f32> = dynm.add(256);
    let x = *inp.add((base + t) as usize);
    tile.set(t, x);
    dynm.set(t, 2.0 * x);
    sync_threads();
    let r = 255 - t;
    *out.add((base + t) as usize) = tile.at(r) + dynm.at(r);
    let s = warp::block_sum(x, red);
    if t == 0 {
        *out.add((n + bid_x()) as usize) = s;
    }
}

/// One warp: C[16][8] (f32, row-major) = A[16][16] (bf16 row-major) * W[8][16]^T (bf16, n-major like a
/// linear weight). A and W go global -> shared by cp.async, fragments by ldmatrix, product by mma.sync.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_t_mma(c: *mut f32, a: *const u16, w: *const u16) {
    let l = lane();
    let sa: Shared<u16> = static_shared!(t_sa, u16, 512);
    let sw: Shared<u16> = static_shared!(t_sw, u16, 256);
    cp_async_16(sa.add(l * 8).addr(), a.add((l * 8) as usize) as *const u8);
    if l < 16 {
        cp_async_16(sw.add(l * 8).addr(), w.add((l * 8) as usize) as *const u8);
    }
    cp_async_commit();
    cp_async_wait_all();
    sync_warp();
    let af = ldmatrix_x4(sa.add((l % 16) * 16 + (l / 16) * 8).addr());
    let bf = ldmatrix_x2(sw.add((l % 8) * 16 + ((l / 8) % 2) * 8).addr());
    let mut d = [0.0f32; 4];
    mma_bf16_16816(&mut d, af, bf);
    let g = l / 4;
    let t = l % 4;
    *c.add((g * 8 + 2 * t) as usize) = d[0];
    *c.add((g * 8 + 2 * t + 1) as usize) = d[1];
    *c.add(((g + 8) * 8 + 2 * t) as usize) = d[2];
    *c.add(((g + 8) * 8 + 2 * t + 1) as usize) = d[3];
}

/// Fast-math sanity: out[i] = (exp, silu, rsqrt, bf16 round) of in[i].
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_t_math(out: *mut f32, inp: *const f32, n: u32) {
    let i = bid_x() * bdim_x() + tid_x();
    if i < n {
        let x = *inp.add(i as usize);
        let o = out.add((i * 4) as usize);
        *o = math::exp(x);
        *o.add(1) = math::silu(x);
        *o.add(2) = math::rsqrt(x.abs() + 1.0);
        *o.add(3) = bf16::round(x);
    }
}

/// Tensor-core peak probe: `iters` x 8 independent bf16 m16n8k16 mma per warp, register operands only.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_t_mma_peak(out: *mut f32, iters: u32) {
    let l = lane();
    let a = [l, l ^ 1, l ^ 2, l ^ 3];
    let b = [l ^ 5, l ^ 6];
    let mut acc = [[0.0f32; 4]; 8];
    let mut i = 0;
    while i < iters {
        kcommon::unroll!(j in 8 => {
            mma_bf16_16816(&mut acc[j], a, b);
        });
        i += 1;
    }
    let mut s = 0.0;
    kcommon::unroll!(j in 8 => {
        s += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
    });
    if s == 12345.0 {
        *out = s;
    }
}
