//! Sub-group block reads and writes (cl_intel_subgroups, _short, _char): one message moves sub-group-size x N
//! consecutive elements; lane l gets element `l + 16 i` of the run in component i (sub-group size 16). Pointers must be
//! 4-byte aligned (16-byte for the widest forms on global memory). OpSubgroupBlockReadINTEL / WriteINTEL.
//! [`global`] takes pointers into device buffers, [`local`] pointers into SLM.

macro_rules! rd {
    ($name:ident, $f:ident, $t:ty, $r:ty) => {
        #[inline(always)]
        pub unsafe fn $name(p: *const $t) -> $r {
            unsafe { crate::ffi::$f(p) }
        }
    };
}
macro_rules! wr {
    ($name:ident, $f:ident, $t:ty, $v:ty) => {
        #[inline(always)]
        pub unsafe fn $name(p: *mut $t, v: $v) {
            unsafe { crate::ffi::$f(p, v) }
        }
    };
}

pub mod global {
    use crate::v::*;
    rd!(read_u32, _Z26intel_sub_group_block_readPU3AS1Kj, u32, u32);
    rd!(read_u32x2, _Z27intel_sub_group_block_read2PU3AS1Kj, u32, U32x2);
    rd!(read_u32x4, _Z27intel_sub_group_block_read4PU3AS1Kj, u32, U32x4);
    rd!(read_u32x8, _Z27intel_sub_group_block_read8PU3AS1Kj, u32, U32x8);
    rd!(read_u16, _Z29intel_sub_group_block_read_usPU3AS1Kt, u16, u16);
    rd!(read_u16x2, _Z30intel_sub_group_block_read_us2PU3AS1Kt, u16, U16x2);
    rd!(read_u16x4, _Z30intel_sub_group_block_read_us4PU3AS1Kt, u16, U16x4);
    rd!(read_u16x8, _Z30intel_sub_group_block_read_us8PU3AS1Kt, u16, U16x8);
    rd!(read_u8, _Z29intel_sub_group_block_read_ucPU3AS1Kh, u8, u8);
    rd!(read_u8x2, _Z30intel_sub_group_block_read_uc2PU3AS1Kh, u8, U8x2);
    rd!(read_u8x4, _Z30intel_sub_group_block_read_uc4PU3AS1Kh, u8, U8x4);
    rd!(read_u8x8, _Z30intel_sub_group_block_read_uc8PU3AS1Kh, u8, U8x8);
    rd!(read_u8x16, _Z31intel_sub_group_block_read_uc16PU3AS1Kh, u8, U8x16);
    wr!(write_u32, _Z27intel_sub_group_block_writePU3AS1jj, u32, u32);
    wr!(write_u32x2, _Z28intel_sub_group_block_write2PU3AS1jDv2_j, u32, U32x2);
    wr!(write_u32x4, _Z28intel_sub_group_block_write4PU3AS1jDv4_j, u32, U32x4);
    wr!(write_u32x8, _Z28intel_sub_group_block_write8PU3AS1jDv8_j, u32, U32x8);
    wr!(write_u16, _Z30intel_sub_group_block_write_usPU3AS1tt, u16, u16);
    wr!(write_u16x2, _Z31intel_sub_group_block_write_us2PU3AS1tDv2_t, u16, U16x2);
    wr!(write_u16x4, _Z31intel_sub_group_block_write_us4PU3AS1tDv4_t, u16, U16x4);
    wr!(write_u16x8, _Z31intel_sub_group_block_write_us8PU3AS1tDv8_t, u16, U16x8);

    /// `prefetch(p, n)`: hint that n dwords from p will be read (OpenCL.std prefetch).
    #[inline(always)]
    pub unsafe fn prefetch_u32(p: *const u32, n: u64) {
        unsafe { crate::ffi::_Z8prefetchPU3AS1Kjm(p, n) }
    }

    /// n (<= 64, a compile-time constant after inlining) dwords a lane from a [n][16 lanes] run into `o[0..n]`, in block
    /// reads of 8, 4, 2, 1 dwords (xq.cl `rd_u`). Straight-line code (`unroll!`), so `o` can live in registers even
    /// when LLVM runs no loop passes.
    #[inline(always)]
    pub unsafe fn read_u32_run(n: usize, p: *const u32, o: &mut [u32]) {
        let n8 = n / 8 * 8;
        crate::unroll!(8, n / 8, |j| {
            let v = unsafe { read_u32x8(p.add(j * 8 * 16)) };
            crate::unroll!(8, 8, |e| o[j * 8 + e] = v[e]);
        });
        let mut i = n8;
        if n - i >= 4 {
            let v = unsafe { read_u32x4(p.add(i * 16)) };
            crate::unroll!(4, 4, |e| o[i + e] = v[e]);
            i += 4;
        }
        if n - i >= 2 {
            let v = unsafe { read_u32x2(p.add(i * 16)) };
            o[i] = v[0];
            o[i + 1] = v[1];
            i += 2;
        }
        if n - i >= 1 {
            o[i] = unsafe { read_u32(p.add(i * 16)) };
        }
    }
    /// As [`read_u32_run`] for 16-bit elements (xq.cl `rd_us`).
    #[inline(always)]
    pub unsafe fn read_u16_run(n: usize, p: *const u16, o: &mut [u16]) {
        let n8 = n / 8 * 8;
        crate::unroll!(8, n / 8, |j| {
            let v = unsafe { read_u16x8(p.add(j * 8 * 16)) };
            crate::unroll!(8, 8, |e| o[j * 8 + e] = v[e]);
        });
        let mut i = n8;
        if n - i >= 4 {
            let v = unsafe { read_u16x4(p.add(i * 16)) };
            crate::unroll!(4, 4, |e| o[i + e] = v[e]);
            i += 4;
        }
        if n - i >= 2 {
            let v = unsafe { read_u16x2(p.add(i * 16)) };
            o[i] = v[0];
            o[i + 1] = v[1];
            i += 2;
        }
        if n - i >= 1 {
            o[i] = unsafe { read_u16(p.add(i * 16)) };
        }
    }
}

pub mod local {
    use crate::v::*;
    rd!(read_u32, _Z26intel_sub_group_block_readPU3AS3Kj, u32, u32);
    rd!(read_u32x2, _Z27intel_sub_group_block_read2PU3AS3Kj, u32, U32x2);
    rd!(read_u32x4, _Z27intel_sub_group_block_read4PU3AS3Kj, u32, U32x4);
    rd!(read_u32x8, _Z27intel_sub_group_block_read8PU3AS3Kj, u32, U32x8);
    rd!(read_u16, _Z29intel_sub_group_block_read_usPU3AS3Kt, u16, u16);
    rd!(read_u16x8, _Z30intel_sub_group_block_read_us8PU3AS3Kt, u16, U16x8);
    wr!(write_u32, _Z27intel_sub_group_block_writePU3AS3jj, u32, u32);
    wr!(write_u32x8, _Z28intel_sub_group_block_write8PU3AS3jDv8_j, u32, U32x8);
    wr!(write_u16x8, _Z31intel_sub_group_block_write_us8PU3AS3tDv8_t, u16, U16x8);
}
