//! Shared memory without addrspace(3) in Rust.
//!
//! Rust has no way to name an LLVM addrspace(3) global, so shared memory is reached through PTX:
//! - dynamic: one module-scope `.extern .shared .align 16 .b8 fe_dsmem[];` (declared below with
//!   `global_asm!`, lands once in every kernel crate's PTX). [`dyn_base`] returns its 32-bit
//!   shared-window address; size it at launch with the `smem` bytes argument.
//! - static: [`static_shared!`] declares a function-scope `.shared` array inside an `asm!` block
//!   (PTX rejects `.shared` inside `{ }` sub-blocks, so the name must be unique within the kernel).
//!
//! Loads and stores go through [`Shared<T>`], a typed 32-bit shared address whose `ld`/`st` emit real
//! `ld.shared`/`st.shared` (LDS/STS in SASS), and whose address feeds `ldmatrix`/`cp.async` directly.
//! A generic pointer (`cvta.shared`) is available via [`Shared::generic`] but compiles to generic LD/ST.
use core::arch::{asm, global_asm};
use core::marker::PhantomData;

global_asm!(".extern .shared .align 16 .b8 fe_dsmem[];");

/// A typed address in the CTA's `.shared` window.
#[repr(transparent)]
pub struct Shared<T> {
    addr: u32,
    _t: PhantomData<T>,
}
impl<T> Clone for Shared<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Shared<T> {}

impl<T> Shared<T> {
    #[inline(always)]
    pub const fn from_addr(addr: u32) -> Self {
        Shared { addr, _t: PhantomData }
    }
    #[inline(always)]
    pub const fn addr(self) -> u32 {
        self.addr
    }
    /// Element offset.
    #[inline(always)]
    pub const fn add(self, n: u32) -> Self {
        Shared::from_addr(self.addr + n * core::mem::size_of::<T>() as u32)
    }
    #[inline(always)]
    pub const fn byte_add(self, bytes: u32) -> Self {
        Shared::from_addr(self.addr + bytes)
    }
    #[inline(always)]
    pub const fn cast<U>(self) -> Shared<U> {
        Shared::from_addr(self.addr)
    }
    /// Generic pointer to the same location (slower generic LD/ST; use for passing to helpers only).
    #[inline(always)]
    pub fn generic(self) -> *mut T {
        let p: u64;
        unsafe {
            asm!("cvta.shared.u64 {p}, {a};", p = out(reg64) p, a = in(reg64) self.addr as u64,
                options(pure, nomem, nostack));
        }
        p as *mut T
    }
}
impl<T: SmemIo> Shared<T> {
    #[inline(always)]
    pub fn ld(self) -> T {
        unsafe { T::lds(self.addr) }
    }
    #[inline(always)]
    pub fn st(self, v: T) {
        unsafe { T::sts(self.addr, v) }
    }
    #[inline(always)]
    pub fn at(self, i: u32) -> T {
        self.add(i).ld()
    }
    #[inline(always)]
    pub fn set(self, i: u32, v: T) {
        self.add(i).st(v)
    }
}

/// Base of the dynamic shared allocation (`fe_dsmem`).
#[inline(always)]
pub fn dyn_base<T>() -> Shared<T> {
    let a: u32;
    unsafe { asm!("mov.u32 {a}, fe_dsmem;", a = out(reg32) a, options(pure, nomem, nostack)) }
    Shared::from_addr(a)
}
/// Dynamic shared bytes requested at launch (`%dynamic_smem_size`).
#[inline(always)]
pub fn dyn_size() -> u32 {
    let n: u32;
    unsafe { asm!("mov.u32 {n}, %dynamic_smem_size;", n = out(reg32) n, options(pure, nomem, nostack)) }
    n
}

/// Declare a function-scope static shared array and get its typed address:
/// `let tile = static_shared!(tile_a, f32, 4096);` (third arg = BYTES, a const expression).
/// The name must be unique within the kernel (ptxas reports a duplicate otherwise).
#[macro_export]
macro_rules! static_shared {
    ($name:ident, $ty:ty, $bytes:expr) => {{
        let a: u32;
        unsafe {
            core::arch::asm!(
                concat!(".shared .align 16 .b8 ", stringify!($name), "[{n}];\n\tmov.u32 {a}, ", stringify!($name), ";"),
                n = const $bytes, a = out(reg32) a, options(nomem, nostack));
        }
        $crate::smem::Shared::<$ty>::from_addr(a)
    }};
}

/// Types with a native `ld.shared`/`st.shared` form.
pub trait SmemIo: Copy {
    unsafe fn lds(addr: u32) -> Self;
    unsafe fn sts(addr: u32, v: Self);
}

macro_rules! scalar_io {
    ($t:ty, $sfx:literal, $cls:ident) => {
        impl SmemIo for $t {
            #[inline(always)]
            unsafe fn lds(addr: u32) -> Self {
                let v: $t;
                asm!(concat!("ld.shared.", $sfx, " {v}, [{a}];"), v = out($cls) v, a = in(reg32) addr,
                    options(readonly, nostack));
                v
            }
            #[inline(always)]
            unsafe fn sts(addr: u32, v: Self) {
                asm!(concat!("st.shared.", $sfx, " [{a}], {v};"), v = in($cls) v, a = in(reg32) addr,
                    options(nostack));
            }
        }
    };
}
scalar_io!(u32, "u32", reg32);
scalar_io!(i32, "s32", reg32);
scalar_io!(f32, "f32", reg32);
scalar_io!(u16, "u16", reg16);
scalar_io!(u64, "u64", reg64);

impl SmemIo for [u32; 2] {
    #[inline(always)]
    unsafe fn lds(addr: u32) -> Self {
        let (x, y): (u32, u32);
        asm!("ld.shared.v2.u32 {{{x}, {y}}}, [{a}];", x = out(reg32) x, y = out(reg32) y, a = in(reg32) addr,
            options(readonly, nostack));
        [x, y]
    }
    #[inline(always)]
    unsafe fn sts(addr: u32, v: Self) {
        asm!("st.shared.v2.u32 [{a}], {{{x}, {y}}};", x = in(reg32) v[0], y = in(reg32) v[1], a = in(reg32) addr,
            options(nostack));
    }
}
impl SmemIo for [u32; 4] {
    #[inline(always)]
    unsafe fn lds(addr: u32) -> Self {
        let (x, y, z, w): (u32, u32, u32, u32);
        asm!("ld.shared.v4.u32 {{{x}, {y}, {z}, {w}}}, [{a}];", x = out(reg32) x, y = out(reg32) y,
            z = out(reg32) z, w = out(reg32) w, a = in(reg32) addr, options(readonly, nostack));
        [x, y, z, w]
    }
    #[inline(always)]
    unsafe fn sts(addr: u32, v: Self) {
        asm!("st.shared.v4.u32 [{a}], {{{x}, {y}, {z}, {w}}};", x = in(reg32) v[0], y = in(reg32) v[1],
            z = in(reg32) v[2], w = in(reg32) v[3], a = in(reg32) addr, options(nostack));
    }
}
impl SmemIo for [f32; 4] {
    #[inline(always)]
    unsafe fn lds(addr: u32) -> Self {
        let v = <[u32; 4]>::lds(addr);
        [f32::from_bits(v[0]), f32::from_bits(v[1]), f32::from_bits(v[2]), f32::from_bits(v[3])]
    }
    #[inline(always)]
    unsafe fn sts(addr: u32, v: Self) {
        <[u32; 4]>::sts(addr, [v[0].to_bits(), v[1].to_bits(), v[2].to_bits(), v[3].to_bits()])
    }
}
