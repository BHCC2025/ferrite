//! 2D block load / store / prefetch (cl_intel_subgroup_2d_block_io; IGC: `load_block2d` / `store_block2d` LSC messages).
//!
//! A surface is (base, width in bytes, height in rows, pitch in bytes); the block origin is (x in elements, y in rows).
//! Out-of-surface elements read as zero and are not written. Hardware rules: base 64-byte aligned, width and pitch
//! >= 64 bytes, pitch a multiple of 16 (8 for some forms), width <= 2^24. Names: `<elem bits>b_<rows>r<cols>x<blocks>c`;
//! `transform` = VNNI (pairs/quads of rows interleaved per dword: the DPAS B layout), `transpose` = column-major.
//! For 16-bit `rows` x 16 x `blocks`, lane l gets column l of each row, block-major (element `b * rows + r`).
//!
//! Every read has an `_into` form that writes the caller's private array, exactly as the OpenCL builtin does (use it in
//! loops: the array stays in registers). The value forms return `Simd` for up to 8 elements and a plain array above
//! that: a 16-dword `Simd` copied out of the builtin's array became one `load i512`, which SPIR-V cannot express.
use crate::ffi::*;
use crate::v::*;
use crate::Simd;
use core::mem::MaybeUninit;

/// A 2D surface in global memory.
#[derive(Copy, Clone)]
pub struct Surface {
    pub base: *const u8,
    pub width_bytes: i32,
    pub height: i32,
    pub pitch_bytes: i32,
}

macro_rules! rd2d {
    (simd $name:ident, $into:ident, $f:ident, $e:ty, $n:literal) => {
        rd2d!(@into $into, $f, $e, $n);
        #[inline(always)]
        pub unsafe fn $name(s: Surface, x: i32, y: i32) -> Simd<$e, $n> {
            let mut d = MaybeUninit::<[$e; $n]>::uninit();
            unsafe {
                $f(s.base, s.width_bytes, s.height, s.pitch_bytes, I32x2::from_array([x, y]), d.as_mut_ptr() as *mut $e);
                Simd::from_array(d.assume_init())
            }
        }
    };
    (array $name:ident, $into:ident, $f:ident, $e:ty, $n:literal) => {
        rd2d!(@into $into, $f, $e, $n);
        #[inline(always)]
        pub unsafe fn $name(s: Surface, x: i32, y: i32) -> [$e; $n] {
            let mut d = [0 as $e; $n];
            unsafe { $f(s.base, s.width_bytes, s.height, s.pitch_bytes, I32x2::from_array([x, y]), d.as_mut_ptr()) };
            d
        }
    };
    (@into $into:ident, $f:ident, $e:ty, $n:literal) => {
        #[inline(always)]
        pub unsafe fn $into(s: Surface, x: i32, y: i32, out: &mut [$e; $n]) {
            unsafe { $f(s.base, s.width_bytes, s.height, s.pitch_bytes, I32x2::from_array([x, y]), out.as_mut_ptr()) }
        }
    };
}
macro_rules! pf2d {
    ($name:ident, $f:ident) => {
        #[inline(always)]
        pub unsafe fn $name(s: Surface, x: i32, y: i32) {
            unsafe { $f(s.base, s.width_bytes, s.height, s.pitch_bytes, I32x2::from_array([x, y])) }
        }
    };
}

rd2d!(simd read_16b_8r16x1c, read_16b_8r16x1c_into, _Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt, u16, 8);
rd2d!(array read_16b_8r16x2c, read_16b_8r16x2c_into, _Z41intel_sub_group_2d_block_read_16b_8r16x2cPU3AS1viiiDv2_iPt, u16, 16);
rd2d!(array read_16b_16r16x1c, read_16b_16r16x1c_into, _Z42intel_sub_group_2d_block_read_16b_16r16x1cPU3AS1viiiDv2_iPt, u16, 16);
rd2d!(array read_16b_16r16x2c, read_16b_16r16x2c_into, _Z42intel_sub_group_2d_block_read_16b_16r16x2cPU3AS1viiiDv2_iPt, u16, 32);
rd2d!(array read_16b_32r16x1c, read_16b_32r16x1c_into, _Z42intel_sub_group_2d_block_read_16b_32r16x1cPU3AS1viiiDv2_iPt, u16, 32);
rd2d!(array read_16b_32r16x2c, read_16b_32r16x2c_into, _Z42intel_sub_group_2d_block_read_16b_32r16x2cPU3AS1viiiDv2_iPt, u16, 64);
rd2d!(array read_8b_16r32x1c, read_8b_16r32x1c_into, _Z41intel_sub_group_2d_block_read_8b_16r32x1cPU3AS1viiiDv2_iPt, u16, 16);
rd2d!(array read_8b_32r32x1c, read_8b_32r32x1c_into, _Z41intel_sub_group_2d_block_read_8b_32r32x1cPU3AS1viiiDv2_iPt, u16, 32);
rd2d!(simd read_transform_16b_16r16x1c, read_transform_16b_16r16x1c_into, _Z52intel_sub_group_2d_block_read_transform_16b_16r16x1cPU3AS1viiiDv2_iPj, u32, 8);
rd2d!(array read_transform_16b_16r16x2c, read_transform_16b_16r16x2c_into, _Z52intel_sub_group_2d_block_read_transform_16b_16r16x2cPU3AS1viiiDv2_iPj, u32, 16);
rd2d!(array read_transform_16b_32r16x2c, read_transform_16b_32r16x2c_into, _Z52intel_sub_group_2d_block_read_transform_16b_32r16x2cPU3AS1viiiDv2_iPj, u32, 32);
rd2d!(simd read_transform_8b_32r16x1c, read_transform_8b_32r16x1c_into, _Z51intel_sub_group_2d_block_read_transform_8b_32r16x1cPU3AS1viiiDv2_iPj, u32, 8);
rd2d!(array read_transform_8b_32r16x2c, read_transform_8b_32r16x2c_into, _Z51intel_sub_group_2d_block_read_transform_8b_32r16x2cPU3AS1viiiDv2_iPj, u32, 16);
rd2d!(simd read_transpose_32b_16r8x1c, read_transpose_32b_16r8x1c_into, _Z51intel_sub_group_2d_block_read_transpose_32b_16r8x1cPU3AS1viiiDv2_iPj, u32, 8);
pf2d!(prefetch_16b_8r16x2c, _Z45intel_sub_group_2d_block_prefetch_16b_8r16x2cPU3AS1viiiDv2_i);
pf2d!(prefetch_16b_16r16x2c, _Z46intel_sub_group_2d_block_prefetch_16b_16r16x2cPU3AS1viiiDv2_i);
pf2d!(prefetch_16b_32r16x2c, _Z46intel_sub_group_2d_block_prefetch_16b_32r16x2cPU3AS1viiiDv2_i);
pf2d!(prefetch_8b_32r32x1c, _Z45intel_sub_group_2d_block_prefetch_8b_32r32x1cPU3AS1viiiDv2_i);

/// Store 8 rows x 16 16-bit columns (lane l writes column l; component r = row r).
#[inline(always)]
pub unsafe fn write_16b_8r16x1c(base: *mut u8, width_bytes: i32, height: i32, pitch_bytes: i32, x: i32, y: i32, v: U16x8) {
    let mut s = v.to_array();
    unsafe {
        _Z42intel_sub_group_2d_block_write_16b_8r16x1cPU3AS1viiiDv2_iPt(
            base,
            width_bytes,
            height,
            pitch_bytes,
            I32x2::from_array([x, y]),
            s.as_mut_ptr(),
        )
    }
}
