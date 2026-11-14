//! f32 math for no_std device code. `+ - * /` are native (div.rn); everything transcendental goes
//! through the PTX approx instructions (MUFU), matching CUDA's `__expf`-style fast math.
use core::arch::asm;

pub const LOG2E: f32 = core::f32::consts::LOG2_E;
pub const LN2: f32 = core::f32::consts::LN_2;

#[inline(always)]
pub fn fma(a: f32, b: f32, c: f32) -> f32 {
    let r: f32;
    unsafe {
        asm!("fma.rn.f32 {r}, {a}, {b}, {c};", r = out(reg32) r, a = in(reg32) a, b = in(reg32) b, c = in(reg32) c,
            options(pure, nomem, nostack))
    }
    r
}
#[inline(always)]
pub fn exp2(x: f32) -> f32 {
    let r: f32;
    unsafe { asm!("ex2.approx.ftz.f32 {r}, {x};", r = out(reg32) r, x = in(reg32) x, options(pure, nomem, nostack)) }
    r
}
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    exp2(x * LOG2E)
}
#[inline(always)]
pub fn log2(x: f32) -> f32 {
    let r: f32;
    unsafe { asm!("lg2.approx.ftz.f32 {r}, {x};", r = out(reg32) r, x = in(reg32) x, options(pure, nomem, nostack)) }
    r
}
#[inline(always)]
pub fn ln(x: f32) -> f32 {
    log2(x) * LN2
}
#[inline(always)]
pub fn rsqrt(x: f32) -> f32 {
    let r: f32;
    unsafe { asm!("rsqrt.approx.ftz.f32 {r}, {x};", r = out(reg32) r, x = in(reg32) x, options(pure, nomem, nostack)) }
    r
}
#[inline(always)]
pub fn sqrt(x: f32) -> f32 {
    let r: f32;
    unsafe { asm!("sqrt.rn.f32 {r}, {x};", r = out(reg32) r, x = in(reg32) x, options(pure, nomem, nostack)) }
    r
}
#[inline(always)]
pub fn rcp(x: f32) -> f32 {
    let r: f32;
    unsafe { asm!("rcp.rn.f32 {r}, {x};", r = out(reg32) r, x = in(reg32) x, options(pure, nomem, nostack)) }
    r
}
#[inline(always)]
pub fn tanh(x: f32) -> f32 {
    let r: f32;
    unsafe { asm!("tanh.approx.f32 {r}, {x};", r = out(reg32) r, x = in(reg32) x, options(pure, nomem, nostack)) }
    r
}
#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + exp(-x))
}
/// torch.nn.functional.softplus (beta 1, threshold 20).
#[inline(always)]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { ln(1.0 + exp(x)) }
}
#[inline(always)]
pub fn maxf(a: f32, b: f32) -> f32 {
    a.max(b)
}
#[inline(always)]
pub fn minf(a: f32, b: f32) -> f32 {
    a.min(b)
}

/// exp(x) from IEEE mul / add / fma only (no MUFU approximation), so the CPU reproduces it bit for bit
/// (fe-ggq `cpu::exp_det`): 2^t, t = x * log2(e), as 2^n * p(t - n) with Cephes' exp2f polynomial (rel err ~2e-7).
#[inline(always)]
pub fn exp_det(x: f32) -> f32 {
    let t = x * LOG2E;
    let t = if t > 128.0 { 128.0 } else if t < -150.0 { -150.0 } else { t };
    let n = (t + 12582912.0) - 12582912.0;
    let f = t - n;
    let mut p = 1.535336188319500e-4f32;
    p = fma(p, f, 1.339887440266574e-3);
    p = fma(p, f, 9.618437357674640e-3);
    p = fma(p, f, 5.550332471162809e-2);
    p = fma(p, f, 2.402264791363012e-1);
    p = fma(p, f, 6.931472028550421e-1);
    let r = fma(f, p, 1.0);
    let ni = n as i32;
    let n1 = ni / 2;
    let n2 = ni - n1;
    r * f32::from_bits(((n1 + 127) as u32) << 23) * f32::from_bits(((n2 + 127) as u32) << 23)
}
/// x * sigmoid(x) = x / (1 + exp_det(-x)).
#[inline(always)]
pub fn silu_det(x: f32) -> f32 {
    x / (1.0 + exp_det(-x))
}
