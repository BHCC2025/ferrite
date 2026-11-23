//! bf16 conversion helpers (round to nearest even), shared by the engine's host code and `ink-run`.

pub fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1));
    (r >> 16) as u16
}
pub fn bf16_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}
