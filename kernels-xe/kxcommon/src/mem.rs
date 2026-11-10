//! OpenCL `vloadN` / `vstoreN` (OpenCL.std vloadn / vstoren): N consecutive elements per lane at `p + N * offset`, at the
//! element's alignment, as ONE message. Plain vector loads through a cast pointer need the vector's alignment, and N
//! scalar loads are merged by IGC only up to 16 bytes (gdn_step 0.61x -> 0.98x of OpenCL with these). The
//! arc-engine kernels use them a lot (`vload16(lane, cp)`, `vstore4(.., 1, y + at)`), so matching ports should too.
use crate::ffi::*;
use crate::v::*;

macro_rules! ld {
    ($name:ident, $f:ident, $t:ty, $r:ty) => {
        #[inline(always)]
        pub unsafe fn $name(offset: usize, p: *const $t) -> $r {
            unsafe { $f(offset as u64, p) }
        }
    };
}
macro_rules! st {
    ($name:ident, $f:ident, $t:ty, $v:ty) => {
        #[inline(always)]
        pub unsafe fn $name(v: $v, offset: usize, p: *mut $t) {
            unsafe { $f(v, offset as u64, p) }
        }
    };
}

/// Global memory.
pub mod global {
    use super::*;
    ld!(vload2_u32, _Z6vload2mPU3AS1Kj, u32, U32x2);
    ld!(vload4_u32, _Z6vload4mPU3AS1Kj, u32, U32x4);
    ld!(vload8_u32, _Z6vload8mPU3AS1Kj, u32, U32x8);
    ld!(vload16_u32, _Z7vload16mPU3AS1Kj, u32, U32x16);
    ld!(vload2_f32, _Z6vload2mPU3AS1Kf, f32, F32x2);
    ld!(vload4_f32, _Z6vload4mPU3AS1Kf, f32, F32x4);
    ld!(vload8_f32, _Z6vload8mPU3AS1Kf, f32, F32x8);
    ld!(vload4_u16, _Z6vload4mPU3AS1Kt, u16, U16x4);
    ld!(vload8_u16, _Z6vload8mPU3AS1Kt, u16, U16x8);
    ld!(vload16_u16, _Z7vload16mPU3AS1Kt, u16, U16x16);
    st!(vstore2_u32, _Z7vstore2Dv2_jmPU3AS1j, u32, U32x2);
    st!(vstore4_u32, _Z7vstore4Dv4_jmPU3AS1j, u32, U32x4);
    st!(vstore8_u32, _Z7vstore8Dv8_jmPU3AS1j, u32, U32x8);
    st!(vstore16_u32, _Z8vstore16Dv16_jmPU3AS1j, u32, U32x16);
    st!(vstore2_f32, _Z7vstore2Dv2_fmPU3AS1f, f32, F32x2);
    st!(vstore4_f32, _Z7vstore4Dv4_fmPU3AS1f, f32, F32x4);
    st!(vstore8_f32, _Z7vstore8Dv8_fmPU3AS1f, f32, F32x8);
    st!(vstore4_u16, _Z7vstore4Dv4_tmPU3AS1t, u16, U16x4);
    st!(vstore8_u16, _Z7vstore8Dv8_tmPU3AS1t, u16, U16x8);
}

/// Local memory (SLM).
pub mod local {
    use super::*;
    ld!(vload4_u32, _Z6vload4mPU3AS3Kj, u32, U32x4);
    ld!(vload8_u32, _Z6vload8mPU3AS3Kj, u32, U32x8);
    ld!(vload4_f32, _Z6vload4mPU3AS3Kf, f32, F32x4);
    ld!(vload8_f32, _Z6vload8mPU3AS3Kf, f32, F32x8);
    ld!(vload8_u16, _Z6vload8mPU3AS3Kt, u16, U16x8);
    ld!(vload8_u8, _Z6vload8mPU3AS3Kh, u8, U8x8);
    st!(vstore8_u32, _Z7vstore8Dv8_jmPU3AS3j, u32, U32x8);
    st!(vstore8_f32, _Z7vstore8Dv8_fmPU3AS3f, f32, F32x8);
    st!(vstore4_f32, _Z7vstore4Dv4_fmPU3AS3f, f32, F32x4);
}

/// Private memory (a kernel's own arrays, e.g. what a 2D block read wrote): `vload8(0, wa + 8 a)` as the OpenCL kernels
/// write it, so IGC sees the same access.
pub mod private {
    use super::*;
    #[inline(always)]
    pub unsafe fn vload8_u16(p: *const u16) -> U16x8 {
        unsafe { _Z6vload8mPKt(0, p) }
    }
    #[inline(always)]
    pub unsafe fn vload8_u32(p: *const u32) -> U32x8 {
        unsafe { _Z6vload8mPKj(0, p) }
    }
}
