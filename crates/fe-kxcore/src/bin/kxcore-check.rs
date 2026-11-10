//! kx-core checks: every Rust kernel against arc-engine's OpenCL kernel on identical inputs, bitwise, plus window
//! invariance (an m-row launch == m one-row launches, row for row; for the delta rule: m sequential steps), plus device
//! timing of both on the same shapes.
//!
//! usage: kxcore-check [--kernels a,b,..] [--rows 1,2,3,4,5,8,9,16] [--len 1500,3000,20000] [--time] [--iters 50]
//!                     [--seed N] [--kd DIR|FILE.kd,..]   (--kd: replay arc-engine ARC_KDUMP launches instead)
//! Kernel groups: rmsnorm, gdn_prep, gdn_vconv, gdn_gates, gdn_step, gdn_norm, gdn_commit, attn_prep, kvq, attn_dec,
//! merge, one (one-token forms, also == the window forms at one row), dhead, prefill (prompt-window attention), tree
//! (draft-tree verify windows), gdnpre (fused GDN window chain), embed, swiglu, round, argmax (default: all). Every buffer of a launch is compared (stray writes count).
//! Exit 1 on any mismatch.
use anyhow::{bail, Result};
use fe_kxcore::*;
use fe_ze::{DevPtr, DeviceBuffer, Queue, Ze};
use std::sync::Arc;

// ---- data ----------------------------------------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn f32(&mut self) -> f32 {
        (self.u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// ~N(0, 1) (sum of 4 uniforms, scaled).
    fn n(&mut self) -> f32 {
        (self.f32() + self.f32() + self.f32() + self.f32() - 2.0) * 1.732
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.u64() % n as u64) as u32
    }
}

fn bf16(x: f32) -> u16 {
    let u = x.to_bits();
    (u.wrapping_add(0x7fff + ((u >> 16) & 1)) >> 16) as u16
}
fn bfr(x: f32) -> f32 {
    f32::from_bits((bf16(x) as u32) << 16)
}
fn f16(x: f32) -> u16 {
    // RNE fp32 -> fp16 (normal range only; test scales stay normal)
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if e <= 0 {
        return sign;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    sign | r as u16
}
fn bytes<T: Copy>(v: &[T]) -> Vec<u8> {
    let n = std::mem::size_of_val(v);
    let mut out = vec![0u8; n];
    unsafe { std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, out.as_mut_ptr(), n) };
    out
}
fn bf16v(r: &mut Rng, n: usize, scale: f32) -> Vec<u8> {
    let v: Vec<u16> = (0..n).map(|i| bf16(r.n() * scale * if i % 97 == 13 { 6.0 } else { 1.0 })).collect();
    bytes(&v)
}
fn f32v(r: &mut Rng, n: usize, f: impl Fn(&mut Rng) -> f32) -> Vec<u8> {
    let v: Vec<f32> = (0..n).map(|_| f(r)).collect();
    bytes(&v)
}
fn fill(n: usize, b: u8) -> Vec<u8> {
    vec![b; n]
}

// ---- cases ---------------------------------------------------------------------------------------------------------

type Launch = Box<dyn Fn(&KxCore, &Queue, &[DevPtr]) -> Result<()>>;

/// A launch sequence over a set of buffers (initial contents); every buffer is compared after it.
struct Case {
    name: String,
    bufs: Vec<(&'static str, Vec<u8>, usize)>, // name, initial bytes, element bytes (for reporting)
    launch: Launch,
}

fn upload(q: &Queue, c: &Case) -> Result<Vec<DeviceBuffer<u8>>> {
    c.bufs.iter().map(|(_, b, _)| Ok(DeviceBuffer::from_slice(q, b)?)).collect()
}

fn run(kx: &KxCore, q: &Queue, c: &Case) -> Result<Vec<Vec<u8>>> {
    let d = upload(q, c)?;
    let p: Vec<DevPtr> = d.iter().map(|b| b.ptr()).collect();
    (c.launch)(kx, q, &p)?;
    q.sync()?;
    d.iter().map(|b| Ok(b.to_vec(q)?)).collect()
}

/// Mismatching elements of buffer i (element size es).
fn diff(a: &[u8], b: &[u8], es: usize) -> (usize, Option<usize>) {
    let mut n = 0;
    let mut first = None;
    for (i, (x, y)) in a.chunks(es).zip(b.chunks(es)).enumerate() {
        if x != y {
            n += 1;
            first.get_or_insert(i);
        }
    }
    (n, first)
}

fn hex(b: &[u8]) -> String {
    b.iter().rev().map(|x| format!("{x:02x}")).collect()
}

/// Compare two result sets of a case; prints the mismatches; returns the count.
fn compare(c: &Case, what: &str, a: &[Vec<u8>], b: &[Vec<u8>]) -> usize {
    let mut tot = 0;
    for (i, (name, _, es)) in c.bufs.iter().enumerate() {
        let (n, first) = diff(&a[i], &b[i], *es);
        if n > 0 {
            let f = first.unwrap();
            println!(
                "    MISMATCH {what} {}: buffer {name}: {n} of {} elements, first [{f}] {} vs {}",
                c.name,
                a[i].len() / es,
                hex(&a[i][f * es..(f + 1) * es]),
                hex(&b[i][f * es..(f + 1) * es])
            );
        }
        tot += n;
    }
    tot
}

/// Median device time (us) of one invocation of the case's launches (sum over its kernels), `iters` invocations on
/// resident buffers (inputs re-uploaded never: kernels that update in place keep running on their own output, which
/// changes no work).
fn time(ze: &Arc<Ze>, kx: &KxCore, q: &Queue, c: &Case, iters: usize) -> Result<f64> {
    let d = upload(q, c)?;
    let p: Vec<DevPtr> = d.iter().map(|b| b.ptr()).collect();
    for _ in 0..3 {
        (c.launch)(kx, q, &p)?;
    }
    q.sync()?;
    kx.set_timing(ze, true)?;
    let mut t = Vec::new();
    let mut per: Vec<Vec<f64>> = Vec::new();
    for _ in 0..iters {
        (c.launch)(kx, q, &p)?;
        q.sync()?;
        let v = kx.take_times()?;
        if per.is_empty() {
            per = vec![Vec::new(); v.len()];
        }
        for (i, x) in v.iter().enumerate() {
            if i < per.len() {
                per[i].push(*x);
            }
        }
        t.push(v.iter().sum::<f64>());
    }
    kx.set_timing(ze, false)?;
    if std::env::var("KX_BREAKDOWN").is_ok() && per.len() > 1 {
        let med: Vec<String> = per
            .iter_mut()
            .map(|v| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                format!("{:.1}", v[v.len() / 2] / 1000.0)
            })
            .collect();
        println!("      {} launches (us): {}", kx.label, med.join(" "));
    }
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(t[t.len() / 2] / 1000.0)
}

// ---- case builders (shapes and grids of the Qwen3.8 forward) -------------------------------------------------------

const EPS: f32 = 1e-6;

fn case_add_rms(r: &mut Rng, m: u32) -> Case {
    let n = HIDDEN as usize;
    let rows = m as usize;
    let w: Vec<u16> = (0..n).map(|_| bf16(1.0 + 0.1 * r.n())).collect();
    Case {
        name: format!("add_rmsnorm_r m={m}"),
        bufs: vec![("x", bf16v(r, rows * n, 1.0), 2), ("d", bf16v(r, rows * n, 0.5), 2), ("w", bytes(&w), 2), ("y", fill(rows * n * 2, 0x55), 2)],
        launch: Box::new(move |k, q, p| k.add_rmsnorm_r(q, p[0], p[1], p[2], p[3], m, HIDDEN, EPS)),
    }
}
fn case_rms(r: &mut Rng, m: u32) -> Case {
    let n = HIDDEN as usize;
    let rows = m as usize;
    let w: Vec<u16> = (0..n).map(|_| bf16(1.0 + 0.1 * r.n())).collect();
    Case {
        name: format!("rmsnorm_r m={m}"),
        bufs: vec![("x", bf16v(r, rows * n, 2.0), 2), ("w", bytes(&w), 2), ("y", fill(rows * n * 2, 0x55), 2)],
        launch: Box::new(move |k, q, p| k.rmsnorm_r(q, p[0], p[1], p[2], m, HIDDEN, EPS)),
    }
}
/// GDN conv inputs: qkv [max(m, 16)][10240] (rows past m unused), conv state [3][10240], conv weights [10240][4].
fn gdn_conv_bufs(r: &mut Rng, m: u32) -> Vec<(&'static str, Vec<u8>, usize)> {
    let c = GDN_QKV as usize;
    let rows = (m as usize).max(16);
    vec![("qkv", bf16v(r, rows * c, 1.0), 2), ("cstate", bf16v(r, 3 * c, 1.0), 2), ("conv", bf16v(r, 4 * c, 0.4), 2)]
}
fn case_gdn_prep(r: &mut Rng, m: u32) -> Case {
    let mut bufs = gdn_conv_bufs(r, m);
    let rows = (m as usize).max(16);
    bufs.push(("qn", fill(rows * 2048 * 4, 0x55), 4));
    bufs.push(("kn", fill(rows * 2048 * 4, 0x55), 4));
    Case { name: format!("gdn_conv_prep_r m={m}"), bufs, launch: Box::new(move |k, q, p| k.gdn_conv_prep_r(q, p[0], p[1], p[2], p[3], p[4], m)) }
}
fn case_gdn_vconv(r: &mut Rng, m: u32) -> Case {
    let mut bufs = gdn_conv_bufs(r, m);
    bufs.push(("vconv", fill((m as usize).max(16) * GDN_V as usize * 2, 0x55), 2));
    Case { name: format!("gdn_vconv_r m={m}"), bufs, launch: Box::new(move |k, q, p| k.gdn_vconv_r(q, p[0], p[1], p[2], p[3], m)) }
}
fn case_gdn_gates(r: &mut Rng, m: u32) -> Case {
    let h = GDN_HEADS as usize;
    let rows = m as usize;
    // a + dt_bias crosses the softplus threshold 20 for some heads
    let a: Vec<u16> = (0..rows * h).map(|i| bf16(if i % 7 == 3 { 18.0 + 6.0 * r.f32() } else { 3.0 * r.n() })).collect();
    Case {
        name: format!("gdn_gates_r m={m}"),
        bufs: vec![
            ("b", bf16v(r, rows * h, 2.0), 2),
            ("a", bytes(&a), 2),
            ("a_log", f32v(r, h, |r| -2.0 + 4.5 * r.f32()), 4),
            ("dt_bias", f32v(r, h, |r| r.n()), 4),
            ("beta", fill(rows * h * 4, 0x55), 4),
            ("g", fill(rows * h * 4, 0x55), 4),
        ],
        launch: Box::new(move |k, q, p| k.gdn_gates_r(q, p[0], p[1], p[2], p[3], p[4], p[5], m)),
    }
}
/// Delta-rule inputs for max(m, 16) rows: qn, kn (bf16-rounded, unit-ish), vconv, beta (0..1), g (0.5..1), state.
fn gdn_step_bufs(r: &mut Rng, m: u32) -> Vec<(&'static str, Vec<u8>, usize)> {
    let s = (GDN_HEADS * 128 * 128) as usize;
    let rows = (m as usize).max(16);
    vec![
        ("qn", f32v(r, rows * 2048, |r| bfr(r.n() * 0.0884 / 128.0 * 11.3)), 4),
        ("kn", f32v(r, rows * 2048, |r| bfr(r.n() * 0.0884)), 4),
        ("vconv", bf16v(r, rows * GDN_V as usize, 0.5), 2),
        ("beta", f32v(r, rows * 48, |r| r.f32()), 4),
        ("g", f32v(r, rows * 48, |r| 0.5 + 0.5 * r.f32()), 4),
        ("s_in", f32v(r, s, |r| 0.05 * r.n()), 4),
        ("s_out", fill(s * 4, 0x55), 4),
        ("y", fill(rows * GDN_V as usize * 2, 0x55), 2),
    ]
}
fn case_gdn_step(r: &mut Rng, m: u32, inplace: bool) -> Case {
    Case {
        name: format!("gdn_step_r2x4 m={m}{}", if inplace { " in place" } else { "" }),
        bufs: gdn_step_bufs(r, m),
        launch: Box::new(move |k, q, p| k.gdn_step(q, p[0], p[1], p[2], p[3], p[4], p[5], if inplace { p[5] } else { p[6] }, p[7], false, m)),
    }
}
fn case_gdn_norm(r: &mut Rng, m: u32) -> Case {
    let rows = m as usize;
    let w: Vec<u16> = (0..128).map(|_| bf16(0.8 + 0.2 * r.f32())).collect();
    Case {
        name: format!("gdn_gate_norm_r m={m}"),
        bufs: vec![
            ("y", bf16v(r, rows * GDN_V as usize, 0.3), 2),
            ("z", bf16v(r, rows * GDN_V as usize, 2.0), 2),
            ("w", bytes(&w), 2),
            ("out", fill(rows * GDN_V as usize * 2, 0x55), 2),
        ],
        launch: Box::new(move |k, q, p| k.gdn_gate_norm_r(q, p[0], p[1], p[2], p[3], m, EPS)),
    }
}
fn case_gdn_commit(r: &mut Rng, c: u32) -> Case {
    let mut bufs = gdn_conv_bufs(r, 16);
    bufs.truncate(2);
    Case { name: format!("gdn_conv_commit c={c}"), bufs, launch: Box::new(move |k, q, p| k.gdn_conv_commit(q, p[0], p[1], c)) }
}
/// RoPE table [rows][cos 32 | sin 32] for positions pos0 .. (theta 1e7, as the host computes it).
fn rope(pos0: u32, rows: u32) -> Vec<u8> {
    let mut v = Vec::new();
    for r in 0..rows {
        let pos = (pos0 + r) as f32;
        let ang: Vec<f32> = (0..32).map(|i| pos * (1e7f64.powf(-(i as f64) / 32.0) as f32)).collect();
        v.extend(ang.iter().map(|a| a.cos()));
        v.extend(ang.iter().map(|a| a.sin()));
    }
    bytes(&v)
}
fn case_attn_prep(r: &mut Rng, m: u32, pos0: u32) -> Case {
    let rows = m as usize;
    let qw: Vec<u16> = (0..256).map(|_| bf16(1.0 + 0.2 * r.n())).collect();
    let kw: Vec<u16> = (0..256).map(|_| bf16(1.0 + 0.2 * r.n())).collect();
    let k_off = pos0 as u64 * KV_DIM as u64;
    Case {
        name: format!("attn_prep_r m={m} pos0={pos0}"),
        bufs: vec![
            ("qg", bf16v(r, rows * 24 * 512, 3.0), 2),
            ("kraw", bf16v(r, rows * 1024, 3.0), 2),
            ("qw", bytes(&qw), 2),
            ("kw", bytes(&kw), 2),
            ("rope", rope(pos0, m), 4),
            ("qo", fill(rows * Q_DIM as usize * 2, 0x55), 2),
            ("kc", fill((pos0 as usize + rows) * KV_DIM as usize * 2, 0x55), 2),
        ],
        launch: Box::new(move |k, q, p| k.attn_prep_r(q, p[0], p[1], p[2], p[3], p[4], p[5], p[6], k_off, m, EPS)),
    }
}
fn case_kvq(r: &mut Rng, m: u32, pos0: u32) -> Case {
    let rows = m as usize;
    // some all-zero blocks (d == 0 path) and outliers
    let mut src: Vec<u16> = (0..rows * 1024).map(|i| bf16(r.n() * if i % 61 == 7 { 9.0 } else { 1.3 })).collect();
    for i in 0..32 {
        src[64 + i] = 0;
    }
    let mut garbage = vec![0u8; (pos0 as usize + rows) * 4 * KV_REC as usize];
    for b in garbage.iter_mut() {
        *b = r.u64() as u8;
    }
    Case {
        name: format!("kvq_quant8 m={m} pos0={pos0}"),
        bufs: vec![("src", bytes(&src), 2), ("cache", garbage, 1)],
        launch: Box::new(move |k, q, p| k.kvq_quant8(q, p[0], p[1], pos0, m)),
    }
}
/// A q8 KV cache of `n` positions: random int8 data, fp16 block scales of realistic size.
fn kv_cache(r: &mut Rng, n: usize) -> Vec<u8> {
    let mut v = vec![0u8; n * 4 * KV_REC as usize];
    for rec in v.chunks_mut(KV_REC as usize) {
        for b in rec[..256].iter_mut() {
            let x = (r.n() * 40.0).clamp(-127.0, 127.0) as i8;
            *b = x as u8;
        }
        for bl in 0..8 {
            let s = f16((0.5 + 3.0 * r.f32()) / 127.0);
            rec[256 + 2 * bl..258 + 2 * bl].copy_from_slice(&s.to_le_bytes());
        }
    }
    v
}
fn dec_bufs(r: &mut Rng, m: u32, len0: u32) -> Vec<(&'static str, Vec<u8>, usize)> {
    let rows = m as usize;
    let n = len0 as usize + rows;
    let po = rows * MAX_CHUNKS as usize * Q_DIM as usize * 4;
    let pmn = rows * MAX_CHUNKS as usize * 24 * 4;
    vec![
        ("q", bf16v(r, rows * Q_DIM as usize, 2.0), 2),
        ("kc", kv_cache(r, n), 1),
        ("vc", kv_cache(r, n), 1),
        ("po", fill(po, 0x55), 4),
        ("pm", fill(pmn, 0x55), 4),
        ("pl", fill(pmn, 0x55), 4),
    ]
}
fn case_dec(r: &mut Rng, name: &'static str, m: u32, len0: u32) -> Case {
    Case {
        name: format!("{name} m={m} len0={len0}"),
        bufs: dec_bufs(r, m, len0),
        launch: Box::new(move |k, q, p| k.attn_dec_with(q, name, p[0], p[1], p[2], p[3], p[4], p[5], len0, m)),
    }
}
fn case_merge(r: &mut Rng, m: u32, len0: u32) -> Case {
    let rows = m as usize;
    let nmax = rows * MAX_CHUNKS as usize * 24;
    Case {
        name: format!("attn_long_merge m={m} len0={len0}"),
        bufs: vec![
            ("po", f32v(r, nmax * 256, |r| r.n() * 3.0), 4),
            ("pm", f32v(r, nmax, |r| r.n() * 4.0), 4),
            ("pl", f32v(r, nmax, |r| 1.0 + 50.0 * r.f32()), 4),
            ("qg", bf16v(r, rows * 24 * 512, 2.0), 2),
            ("out", fill(rows * Q_DIM as usize * 2, 0x55), 2),
        ],
        launch: Box::new(move |k, q, p| k.attn_long_merge(q, p[0], p[1], p[2], p[3], p[4], len0, m)),
    }
}
fn case_embed(r: &mut Rng, m: u32) -> Case {
    let vocab = 2048usize;
    let ids: Vec<u32> = (0..m).map(|_| r.below(vocab as u32)).collect();
    Case {
        name: format!("embed_bf16 m={m}"),
        bufs: vec![("w", bf16v(r, vocab * HIDDEN as usize, 0.05), 2), ("ids", bytes(&ids), 4), ("y", fill(m as usize * HIDDEN as usize * 2, 0x55), 2)],
        launch: Box::new(move |k, q, p| k.embed_bf16(q, p[0], p[1], p[2], m, HIDDEN)),
    }
}
fn case_swiglu(r: &mut Rng, m: u32) -> Case {
    let n = m * 17408;
    Case {
        name: format!("swiglu m={m}"),
        bufs: vec![("g", bf16v(r, n as usize, 3.0), 2), ("u", bf16v(r, n as usize, 1.0), 2), ("act", fill(n as usize * 2, 0x55), 2)],
        launch: Box::new(move |k, q, p| k.swiglu(q, p[0], p[1], p[2], n)),
    }
}
fn case_round(r: &mut Rng, m: u32) -> Case {
    let n = m * 248320;
    let mut x: Vec<f32> = (0..n).map(|_| r.n() * 8.0).collect();
    x[5] = f32::NAN;
    x[6] = -0.0;
    Case { name: format!("round_bf16_f32 m={m}"), bufs: vec![("x", bytes(&x), 4)], launch: Box::new(move |k, q, p| k.round_bf16_f32(q, p[0], n)) }
}
fn case_argmax(r: &mut Rng, m: u32) -> Case {
    let n = 248320u32;
    let mut x: Vec<f32> = (0..(m * n) as usize).map(|_| r.n() * 3.0).collect();
    // ties (lowest index wins), -0 / +0, a NaN row
    for row in 0..m as usize {
        let a = r.below(n) as usize;
        let b = r.below(n) as usize;
        x[row * n as usize + a] = 40.0;
        x[row * n as usize + b] = 40.0;
    }
    if m > 2 {
        x[2 * n as usize + 777] = f32::NAN;
    }
    Case {
        name: format!("argmax m={m}"),
        bufs: vec![("logits", bytes(&x), 4), ("part", fill(m as usize * 64 * 8, 0x55), 8), ("out", fill(m as usize * 4, 0x55), 4)],
        launch: Box::new(move |k, q, p| k.argmax(q, p[0], p[1], p[2], m, n)),
    }
}

/// One-token GDN chain inputs (union with the window chain's scratch): see gdn1_launch / gdnw_launch.
fn gdn1_bufs(r: &mut Rng) -> Vec<(&'static str, Vec<u8>, usize)> {
    let c = GDN_QKV as usize;
    let h = GDN_HEADS as usize;
    let s = h * 128 * 128;
    let a: Vec<u16> = (0..h).map(|i| bf16(if i % 7 == 3 { 18.0 + 6.0 * r.f32() } else { 3.0 * r.n() })).collect();
    let gw: Vec<u16> = (0..128).map(|_| bf16(0.8 + 0.2 * r.f32())).collect();
    vec![
        ("qkv", bf16v(r, c, 1.0), 2),             // 0
        ("cstate", bf16v(r, 3 * c, 1.0), 2),      // 1
        ("conv", bf16v(r, 4 * c, 0.4), 2),        // 2
        ("qn", fill(2048 * 4, 0x55), 4),          // 3
        ("kn", fill(2048 * 4, 0x55), 4),          // 4
        ("b", bf16v(r, h, 2.0), 2),               // 5
        ("a", bytes(&a), 2),                      // 6
        ("a_log", f32v(r, h, |r| -2.0 + 4.5 * r.f32()), 4), // 7
        ("dt_bias", f32v(r, h, |r| r.n()), 4),    // 8
        ("sstate", f32v(r, s, |r| 0.05 * r.n()), 4), // 9
        ("y", fill(GDN_V as usize * 2, 0x55), 2), // 10
        ("z", bf16v(r, GDN_V as usize, 2.0), 2),  // 11
        ("gnorm", bytes(&gw), 2),                 // 12
        ("out", fill(GDN_V as usize * 2, 0x55), 2), // 13
        ("vconv", fill(GDN_V as usize * 2, 0x55), 2), // 14 (window chain only)
        ("beta", fill(h * 4, 0x55), 4),           // 15 (window chain only)
        ("g", fill(h * 4, 0x55), 4),              // 16 (window chain only)
    ]
}
fn gdn1_launch(k: &KxCore, q: &Queue, p: &[DevPtr]) -> Result<()> {
    k.gdn_conv_prep(q, p[0], p[1], p[2], p[3], p[4])?;
    k.gdn_step1(q, p[3], p[4], p[0], p[1], p[2], p[5], p[6], p[7], p[8], p[9], p[10], false)?;
    k.gdn_gate_norm(q, p[10], p[11], p[12], p[13], EPS)
}
fn gdnw_launch(k: &KxCore, q: &Queue, p: &[DevPtr]) -> Result<()> {
    k.gdn_conv_prep_r(q, p[0], p[1], p[2], p[3], p[4], 1)?;
    k.gdn_vconv_r(q, p[0], p[1], p[2], p[14], 1)?;
    k.gdn_gates_r(q, p[5], p[6], p[7], p[8], p[15], p[16], 1)?;
    k.gdn_step(q, p[3], p[4], p[14], p[15], p[16], p[9], p[9], p[10], false, 1)?;
    k.gdn_gate_norm_r(q, p[10], p[11], p[12], p[13], 1, EPS)?;
    k.gdn_conv_commit(q, p[0], p[1], 1)
}

/// Rows checked against 1-row launches: all up to 16, else the first 4, two in the middle and the last 2.
fn sample_rows(m: u32) -> Vec<usize> {
    let m = m as usize;
    if m <= 16 { (0..m).collect() } else { vec![0, 1, 2, 3, m / 2, m / 2 + 1, m - 2, m - 1] }
}

// ---- window invariance -----------------------------------------------------------------------------------------------

/// Row-parallel kernels: the m-row result's rows == one-row launches on that row's inputs. `rowb` lists, per buffer, the
/// bytes of one row (0: shared, not row-indexed). Runs the Rust kernels only (the reference check covers the m-row form).
fn window_rows(kx: &KxCore, q: &Queue, c: &Case, m: u32, rowb: &[usize], one: &dyn Fn(u32) -> Launch) -> Result<usize> {
    let full = run(kx, q, c)?;
    let mut bad = 0;
    for row in sample_rows(m) {
        // inputs: the row's slice moved to row 0, everything else as in the case
        let bufs: Vec<(&'static str, Vec<u8>, usize)> = c
            .bufs
            .iter()
            .zip(rowb)
            .map(|((n, b, es), &rb)| {
                let mut v = b.clone();
                if rb > 0 {
                    v[..rb].copy_from_slice(&b[row * rb..(row + 1) * rb]);
                }
                (*n, v, *es)
            })
            .collect();
        let c1 = Case { name: format!("{} row {row}", c.name), bufs, launch: one(row as u32) };
        let res = run(kx, q, &c1)?;
        for (i, ((name, _, es), &rb)) in c.bufs.iter().zip(rowb).enumerate() {
            if rb == 0 {
                continue;
            }
            let (n, f) = diff(&full[i][row * rb..(row + 1) * rb], &res[i][..rb], *es);
            if n > 0 {
                println!("    WINDOW {}: row {row} buffer {name}: {n} elements differ from the 1-row launch (first {})", c.name, f.unwrap());
                bad += n;
            }
        }
    }
    Ok(bad)
}

// ---- main ------------------------------------------------------------------------------------------------------------

struct Opts {
    kernels: Vec<String>,
    rows: Vec<u32>,
    lens: Vec<u32>,
    time: bool,
    iters: usize,
    seed: u64,
}

fn want(o: &Opts, k: &str) -> bool {
    o.kernels.is_empty() || o.kernels.iter().any(|x| x == k)
}

struct Tally {
    fails: usize,
    table: Vec<(String, String, f64, f64)>, // kernel, shape, rust us, ocl us
}

/// Reference vs Rust on one case (+ timing).
fn check(ze: &Arc<Ze>, q: &Queue, rs: &KxCore, cl: &KxCore, c: &Case, o: &Opts, t: &mut Tally, time_it: bool) -> Result<()> {
    let a = run(cl, q, c)?;
    let b = run(rs, q, c)?;
    let mut n = compare(c, "rust vs opencl", &a, &b);
    // the comparison means something only if the launch wrote: count the elements the reference changed
    let written: usize = c.bufs.iter().zip(&a).map(|((_, init, es), out)| diff(init, out, *es).0).sum();
    if written == 0 {
        println!("    NO OUTPUT: {} changed nothing", c.name);
        n += 1;
    }
    t.fails += n;
    let mut line = format!("  {:44} {:>8} written  {}", c.name, written, if n == 0 { "bitwise OK".to_string() } else { format!("{n} MISMATCHES") });
    if time_it && o.time {
        // alternate the two in rounds (clock / power drift hits both), keep the best round median of each
        let (mut tr, mut tc) = (f64::MAX, f64::MAX);
        for _ in 0..3 {
            tr = tr.min(time(ze, rs, q, c, o.iters)?);
            tc = tc.min(time(ze, cl, q, c, o.iters)?);
        }
        line += &format!("   rust {tr:8.2} us   opencl {tc:8.2} us   ratio {:.3}", tc / tr);
        let (k, shape) = c.name.split_once(' ').unwrap_or((&c.name, ""));
        t.table.push((k.to_string(), shape.to_string(), tr, tc));
    }
    println!("{line}");
    Ok(())
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut opt = |name: &str| -> Option<String> {
        let i = args.iter().position(|a| a == name)?;
        args.remove(i);
        if i < args.len() && !args[i].starts_with("--") { Some(args.remove(i)) } else { Some(String::new()) }
    };
    let list = |s: Option<String>, d: &str| -> Vec<u32> { s.unwrap_or(d.into()).split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect() };
    let o = Opts {
        kernels: opt("--kernels").map(|s| s.split(',').map(String::from).collect()).unwrap_or_default(),
        rows: list(opt("--rows"), "1,2,3,4,5,8,9,16"),
        lens: list(opt("--len"), "1500,3000,20000"),
        time: opt("--time").is_some(),
        iters: opt("--iters").map(|s| s.parse().unwrap()).unwrap_or(50),
        seed: opt("--seed").map(|s| s.parse().unwrap()).unwrap_or(0x5eed_1234_abcd_0001),
    };
    let kd = opt("--kd");
    if !args.is_empty() {
        bail!("unknown arguments {args:?}");
    }
    let ze = Ze::new()?;
    let q = Queue::new(&ze)?;
    let t0 = fe_ze::now_ns();
    let rs = KxCore::rust(&ze)?;
    let t1 = fe_ze::now_ns();
    let cl = KxCore::reference(&ze)?;
    println!("modules: rust {:.1} s, opencl {:.1} s", (t1 - t0) as f64 / 1e9, (fe_ze::now_ns() - t1) as f64 / 1e9);
    for name in ["attn_dec_partial", "attn_dec_pack", "attn_dec_pack2", "attn_dec_pack4", "attn_prefill_s", "attn_pfs_prep_i8", "kvq_expand_k8", "kvq_expand_vt", "gdn_step_r2x4", "dhead_q8b"] {
        let (a, b) = (rs.kernel(name).props()?, cl.kernel(name).props()?);
        println!("  {name}: spill rust {} B / opencl {} B, private {} / {}, sg {} / {}", a.spill_bytes, b.spill_bytes, a.private_bytes, b.private_bytes, a.max_sg_size, b.max_sg_size);
    }
    if let Some(kd) = kd {
        return replay_dumps(&q, &rs, &cl, &kd);
    }
    let mut r = Rng(o.seed);
    let mut t = Tally { fails: 0, table: Vec::new() };
    let tsel = |m: u32| m == 1 || m == 16 || m == 4 || m >= 512;

    if want(&o, "rmsnorm") {
        println!("== norms");
        for &m in &o.rows {
            let c = case_add_rms(&mut r, m);
            check(&ze, &q, &rs, &cl, &c, &o, &mut t, tsel(m))?;
            let rb = (HIDDEN * 2) as usize;
            t.fails += window_rows(&rs, &q, &c, m, &[rb, rb, 0, rb], &|_| Box::new(|k, q, p| k.add_rmsnorm_r(q, p[0], p[1], p[2], p[3], 1, HIDDEN, EPS)))?;
            let c = case_rms(&mut r, m);
            check(&ze, &q, &rs, &cl, &c, &o, &mut t, tsel(m))?;
            t.fails += window_rows(&rs, &q, &c, m, &[rb, 0, rb], &|_| Box::new(|k, q, p| k.rmsnorm_r(q, p[0], p[1], p[2], 1, HIDDEN, EPS)))?;
        }
    }
    if want(&o, "gdn_prep") || want(&o, "gdn_vconv") {
        println!("== gdn conv");
        for &m in &o.rows {
            // qkv rows are row-indexed; the conv of row t reads rows t-3 .. t, so the 1-row form needs the state as of
            // row t: rebuilt on the host (rows t-3 .. t-1 come from qkv, the older ones from the state)
            for kind in ["gdn_prep", "gdn_vconv"] {
                if !want(&o, kind) {
                    continue;
                }
                let c = if kind == "gdn_prep" { case_gdn_prep(&mut r, m) } else { case_gdn_vconv(&mut r, m) };
                check(&ze, &q, &rs, &cl, &c, &o, &mut t, tsel(m))?;
                let full = run(&rs, &q, &c)?;
                let cw = GDN_QKV as usize * 2;
                for row in sample_rows(m) {
                    // ext(0..2) for row t = ext rows t, t+1, t+2 of the m-row form
                    let ext = |j: usize| -> Vec<u8> { if j < 3 { c.bufs[1].1[j * cw..(j + 1) * cw].to_vec() } else { c.bufs[0].1[(j - 3) * cw..(j - 2) * cw].to_vec() } };
                    let mut st = Vec::new();
                    for j in row..row + 3 {
                        st.extend(ext(j));
                    }
                    let mut qkv = c.bufs[0].1.clone();
                    qkv[..cw].copy_from_slice(&c.bufs[0].1[row * cw..(row + 1) * cw]);
                    let mut bufs = c.bufs.iter().map(|(n, b, e)| (*n, b.clone(), *e)).collect::<Vec<_>>();
                    bufs[0].1 = qkv;
                    bufs[1].1 = st;
                    let c1 = Case {
                        name: format!("{} row {row}", c.name),
                        bufs,
                        launch: if kind == "gdn_prep" { Box::new(|k, q, p| k.gdn_conv_prep_r(q, p[0], p[1], p[2], p[3], p[4], 1)) } else { Box::new(|k, q, p| k.gdn_vconv_r(q, p[0], p[1], p[2], p[3], 1)) },
                    };
                    let res = run(&rs, &q, &c1)?;
                    let outs: &[(usize, usize)] = if kind == "gdn_prep" { &[(3, 2048 * 4), (4, 2048 * 4)] } else { &[(3, GDN_V as usize * 2)] };
                    for &(bi, rb) in outs {
                        let (n, f) = diff(&full[bi][row * rb..(row + 1) * rb], &res[bi][..rb], c.bufs[bi].2);
                        if n > 0 {
                            println!("    WINDOW {}: row {row} buffer {}: {n} differ (first {})", c.name, c.bufs[bi].0, f.unwrap());
                            t.fails += n;
                        }
                    }
                }
            }
        }
    }
    if want(&o, "gdn_gates") {
        println!("== gdn gates");
        for &m in &o.rows {
            let c = case_gdn_gates(&mut r, m);
            check(&ze, &q, &rs, &cl, &c, &o, &mut t, tsel(m))?;
            t.fails += window_rows(&rs, &q, &c, m, &[96, 96, 0, 0, 192, 192], &|_| Box::new(|k, q, p| k.gdn_gates_r(q, p[0], p[1], p[2], p[3], p[4], p[5], 1)))?;
        }
    }
    if want(&o, "gdn_step") {
        println!("== gdn delta rule");
        for &m in &o.rows {
            let c = case_gdn_step(&mut r, m, false);
            check(&ze, &q, &rs, &cl, &c, &o, &mut t, tsel(m))?;
            check(&ze, &q, &rs, &cl, &case_gdn_step(&mut r, m, true), &o, &mut t, false)?;
            // the r2 form (1 value row a sub-group) of the reference == the Rust r2x4
            let c2 = Case { name: format!("gdn_step_r2 (ref) vs r2x4 m={m}"), bufs: c.bufs.iter().map(|(n, b, e)| (*n, b.clone(), *e)).collect(), launch: Box::new(move |k, q, p| k.gdn_step_r2(q, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], false, m)) };
            let a = run(&cl, &q, &c2)?;
            let b = run(&rs, &q, &c)?;
            t.fails += compare(&c, "opencl r2 vs rust r2x4", &a, &b);
            // window: m sequential one-row steps in place == the m-row launch (state and every y row)
            let full = run(&rs, &q, &c)?;
            let d = upload(&q, &c)?;
            let p: Vec<DevPtr> = d.iter().map(|b| b.ptr()).collect();
            for row in 0..m as usize {
                let off2048 = row * 2048 * 4;
                let off48 = row * 48 * 4;
                let offv = row * GDN_V as usize * 2;
                rs.gdn_step(&q, p[0].byte_add(off2048), p[1].byte_add(off2048), p[2].byte_add(offv), p[3].byte_add(off48), p[4].byte_add(off48), p[5], p[5], p[7].byte_add(offv), false, 1)?;
            }
            q.sync()?;
            let st = d[5].to_vec(&q)?;
            let y = d[7].to_vec(&q)?;
            let (n1, _) = diff(&full[6], &st, 4);
            let (n2, _) = diff(&full[7], &y, 2);
            if n1 + n2 > 0 {
                println!("    WINDOW {}: state {n1} / y {n2} differ from {m} sequential 1-row steps", c.name);
            }
            t.fails += n1 + n2;
        }
    }
    if want(&o, "gdn_norm") {
        println!("== gdn gated norm");
        for &m in &o.rows {
            let c = case_gdn_norm(&mut r, m);
            check(&ze, &q, &rs, &cl, &c, &o, &mut t, tsel(m))?;
            let rb = GDN_V as usize * 2;
            t.fails += window_rows(&rs, &q, &c, m, &[rb, rb, 0, rb], &|_| Box::new(|k, q, p| k.gdn_gate_norm_r(q, p[0], p[1], p[2], p[3], 1, EPS)))?;
        }
    }
    if want(&o, "gdn_commit") {
        println!("== gdn conv commit");
        for c in [1u32, 2, 3, 4, 7, 16] {
            check(&ze, &q, &rs, &cl, &case_gdn_commit(&mut r, c), &o, &mut t, c == 1)?;
        }
    }
    if want(&o, "attn_prep") {
        println!("== attention prep");
        for &m in &o.rows {
            let c = case_attn_prep(&mut r, m, 0);
            check(&ze, &q, &rs, &cl, &c, &o, &mut t, tsel(m))?;
            let qr = 24 * 512 * 2;
            t.fails += window_rows(&rs, &q, &c, m, &[qr, 2048, 0, 0, 256, Q_DIM as usize * 2, 2048], &|_| Box::new(|k, q, p| k.attn_prep_r(q, p[0], p[1], p[2], p[3], p[4], p[5], p[6], 0, 1, EPS)))?;
            check(&ze, &q, &rs, &cl, &case_attn_prep(&mut r, m, 37), &o, &mut t, false)?;
        }
    }
    if want(&o, "kvq") {
        println!("== kv q8 append");
        for &m in &o.rows {
            let c = case_kvq(&mut r, m, 5);
            check(&ze, &q, &rs, &cl, &c, &o, &mut t, tsel(m))?;
        }
    }
    if want(&o, "attn_dec") {
        println!("== decode attention (q8, matrix engine)");
        for &len0 in &o.lens {
            for &m in &o.rows {
                // the kernel the engine picks for m rows, then every pack form, all vs the reference
                let picked = if m == 1 { "attn_dec_partial" } else if (6 * m + 7) / 8 >= 4 { "attn_dec_pack4" } else if (6 * m + 7) / 8 >= 2 { "attn_dec_pack2" } else { "attn_dec_pack" };
                let names: Vec<&'static str> = if m == 1 { vec!["attn_dec_partial", "attn_dec_pack"] } else { vec![picked, "attn_dec_pack", "attn_dec_pack2", "attn_dec_pack4"].into_iter().fold(Vec::new(), |mut v, x| { if !v.contains(&x) { v.push(x) } v }) };
                for (i, name) in names.iter().enumerate() {
                    let c = case_dec(&mut r, name, m, len0);
                    let all = std::env::var("KX_TIME_ALL").is_ok();
                    check(&ze, &q, &rs, &cl, &c, &o, &mut t, (i == 0 || all) && (tsel(m) || m == 8 || all))?;
                    if m > 1 {
                        // (every pack form: the engine picks the form by occupancy, so each must give these bits)
                        // window: each row == attn_dec_partial on that row alone (the DEC_CHECK of arc-engine), po / pm / pl
                        let full = run(&rs, &q, &c)?;
                        let d = upload(&q, &c)?;
                        let p: Vec<DevPtr> = d.iter().map(|b| b.ptr()).collect();
                        let qrow = Q_DIM as usize * 2;
                        let por = MAX_CHUNKS as usize * Q_DIM as usize * 4;
                        let pmr = MAX_CHUNKS as usize * 24 * 4;
                        let mut bad = 0;
                        for z in 0..m as usize {
                            q.fill(p[3], &[0x55], c.bufs[3].1.len())?;
                            q.fill(p[4], &[0x55], c.bufs[4].1.len())?;
                            q.fill(p[5], &[0x55], c.bufs[5].1.len())?;
                            rs.attn_dec_with(&q, "attn_dec_partial", p[0].byte_add(z * qrow), p[1], p[2], p[3], p[4], p[5], len0 + z as u32, 1)?;
                            q.sync()?;
                            let nch = (len0 as usize + z).div_ceil(DEC_CHUNK as usize);
                            let (po1, pm1, pl1) = (d[3].to_vec(&q)?, d[4].to_vec(&q)?, d[5].to_vec(&q)?);
                            bad += diff(&full[3][z * por..z * por + nch * Q_DIM as usize * 4], &po1[..nch * Q_DIM as usize * 4], 4).0;
                            bad += diff(&full[4][z * pmr..z * pmr + nch * 24 * 4], &pm1[..nch * 24 * 4], 4).0;
                            bad += diff(&full[5][z * pmr..z * pmr + nch * 24 * 4], &pl1[..nch * 24 * 4], 4).0;
                        }
                        if bad > 0 {
                            println!("    WINDOW {}: {bad} partial values differ from 1-row attn_dec_partial", c.name);
                        }
                        t.fails += bad;
                    }
                }
            }
        }
    }
    if want(&o, "merge") {
        println!("== attention merge + gate");
        for &len0 in &o.lens {
            for &m in &o.rows {
                let c = case_merge(&mut r, m, len0);
                check(&ze, &q, &rs, &cl, &c, &o, &mut t, tsel(m) && len0 == o.lens[0])?;
            }
        }
    }
    if want(&o, "one") {
        println!("== one-token forms (decode step / MTP)");
        for n in [HIDDEN, 1024] {
            let w: Vec<u16> = (0..n).map(|_| bf16(1.0 + 0.1 * r.n())).collect();
            let c = Case {
                name: format!("add_rmsnorm n={n}"),
                bufs: vec![("x", bf16v(&mut r, n as usize, 1.0), 2), ("d", bf16v(&mut r, n as usize, 0.5), 2), ("w", bytes(&w), 2), ("y", fill(n as usize * 2, 0x55), 2)],
                launch: Box::new(move |k, q, p| k.add_rmsnorm(q, p[0], p[1], p[2], p[3], n, EPS)),
            };
            check(&ze, &q, &rs, &cl, &c, &o, &mut t, n == HIDDEN)?;
            // == the window form at one row
            let cw = Case { name: c.name.clone(), bufs: c.bufs.iter().map(|(a, b, e)| (*a, b.clone(), *e)).collect(), launch: Box::new(move |k, q, p| k.add_rmsnorm_r(q, p[0], p[1], p[2], p[3], 1, n, EPS)) };
            t.fails += compare(&c, "one-token vs add_rmsnorm_r", &run(&rs, &q, &c)?, &run(&rs, &q, &cw)?);
            let c = Case {
                name: format!("rmsnorm512 n={n}"),
                bufs: vec![("x", bf16v(&mut r, n as usize, 2.0), 2), ("w", bytes(&w), 2), ("y", fill(n as usize * 2, 0x55), 2)],
                launch: Box::new(move |k, q, p| k.rmsnorm512(q, p[0], p[1], p[2], n, EPS)),
            };
            check(&ze, &q, &rs, &cl, &c, &o, &mut t, n == HIDDEN)?;
            let cw = Case { name: c.name.clone(), bufs: c.bufs.iter().map(|(a, b, e)| (*a, b.clone(), *e)).collect(), launch: Box::new(move |k, q, p| k.rmsnorm_r(q, p[0], p[1], p[2], 1, n, EPS)) };
            t.fails += compare(&c, "one-token vs rmsnorm_r", &run(&rs, &q, &c)?, &run(&rs, &q, &cw)?);
        }
        let c = Case {
            name: "add_bf16 n=5120".into(),
            bufs: vec![("x", bf16v(&mut r, 5120, 1.0), 2), ("d", bf16v(&mut r, 5120, 1.0), 2)],
            launch: Box::new(|k, q, p| k.add_bf16(q, p[0], p[1], 5120)),
        };
        check(&ze, &q, &rs, &cl, &c, &o, &mut t, true)?;
        for _ in 0..3 {
            let c = Case { name: "gdn one-token chain".into(), bufs: gdn1_bufs(&mut r), launch: Box::new(gdn1_launch) };
            check(&ze, &q, &rs, &cl, &c, &o, &mut t, false)?;
            // == the window chain at one row (conv_prep_r, vconv_r, gates_r, step_r2x4 in place, gate_norm_r, commit)
            let a = run(&rs, &q, &c)?;
            let cw = Case { name: c.name.clone(), bufs: c.bufs.iter().map(|(a, b, e)| (*a, b.clone(), *e)).collect(), launch: Box::new(gdnw_launch) };
            let b = run(&rs, &q, &cw)?;
            for i in [1usize, 3, 4, 9, 10, 13] {
                let (n, f) = diff(&a[i], &b[i], c.bufs[i].2);
                if n > 0 {
                    println!("    ONE vs WINDOW gdn: buffer {}: {n} differ (first {})", c.bufs[i].0, f.unwrap());
                    t.fails += n;
                }
            }
        }
        if o.time {
            // per-kernel timing of the one-token gdn kernels
            let b = gdn1_bufs(&mut r);
            for (name, f) in [
                ("gdn_conv_prep", Box::new(|k: &KxCore, q: &Queue, p: &[DevPtr]| k.gdn_conv_prep(q, p[0], p[1], p[2], p[3], p[4])) as Launch),
                ("gdn_step", Box::new(|k: &KxCore, q: &Queue, p: &[DevPtr]| k.gdn_step1(q, p[3], p[4], p[0], p[1], p[2], p[5], p[6], p[7], p[8], p[9], p[10], false))),
                ("gdn_gate_norm", Box::new(|k: &KxCore, q: &Queue, p: &[DevPtr]| k.gdn_gate_norm(q, p[10], p[11], p[12], p[13], EPS))),
            ] {
                let c = Case { name: format!("{name} one-token"), bufs: b.iter().map(|(a, b, e)| (*a, b.clone(), *e)).collect(), launch: f };
                check(&ze, &q, &rs, &cl, &c, &o, &mut t, true)?;
            }
        }
        for pos in [0u32, 37] {
            let c1 = case_attn_prep(&mut r, 1, pos);
            let k_off = pos * KV_DIM;
            let c = Case { name: format!("attn_prep pos={pos}"), bufs: c1.bufs.iter().map(|(a, b, e)| (*a, b.clone(), *e)).collect(), launch: Box::new(move |k, q, p| k.attn_prep(q, p[0], p[1], p[2], p[3], p[4], p[5], p[6], k_off, EPS)) };
            check(&ze, &q, &rs, &cl, &c, &o, &mut t, pos == 0)?;
            t.fails += compare(&c, "one-token vs attn_prep_r", &run(&rs, &q, &c)?, &run(&rs, &q, &c1)?);
        }
    }
    if want(&o, "dhead") {
        println!("== MTP draft head");
        for n in [65536u32, 65536 + 8192, 37, 1000] {
            let k = HIDDEN;
            let ids: Vec<u32> = (0..n).map(|i| (i * 7919) % 248320).collect();
            let mut rows = vec![0u8; (n * k) as usize];
            for b in rows.iter_mut() {
                *b = (r.n() * 30.0).clamp(-127.0, 127.0) as i8 as u8;
            }
            let c = Case {
                name: format!("dhead n={n}"),
                bufs: vec![
                    ("rows", rows, 1),
                    ("scale", f32v(&mut r, n as usize, |r| 0.001 + 0.002 * r.f32()), 4),
                    ("x", bf16v(&mut r, k as usize, 1.0), 2),
                    ("logits", fill(n as usize * 4, 0x55), 4),
                    ("part", fill(64 * 8, 0x55), 8),
                    ("out", fill(4, 0x55), 4),
                    ("ids", bytes(&ids), 4),
                ],
                launch: Box::new(move |kx, q, p| kx.dhead(q, p[0], p[1], p[2], p[3], p[4], p[5], p[6], n, k)),
            };
            check(&ze, &q, &rs, &cl, &c, &o, &mut t, n == 65536)?;
        }
    }
    if want(&o, "prefill") {
        println!("== prompt-window attention (int8 scores, q8 cache)");
        for &(rows, pos0, range) in &[(16u32, 0u32, 16384u32), (100, 37, 16384), (100, 1500, 1024), (512, 3000, 16384), (4096, 0, 16384), (4096, 28672, 16384), (300, 2000, 512)] {
            let n = (pos0 + rows) as usize;
            let tiles = rows.div_ceil(8) as usize;
            let rr = (range / 128 * 128) as usize;
            let c = Case {
                name: format!("attn_prefill rows={rows} pos0={pos0} range={range}"),
                bufs: vec![
                    ("q", bf16v(&mut r, rows as usize * Q_DIM as usize, 2.0), 2),
                    ("kc", kv_cache(&mut r, n), 1),
                    ("vc", kv_cache(&mut r, n), 1),
                    ("qg", bf16v(&mut r, rows as usize * 24 * 512, 2.0), 2),
                    ("out", fill(rows as usize * Q_DIM as usize * 2, 0x55), 2),
                    ("qb", fill(tiles * 8 * 24 * 256 * 2, 0), 4),
                    ("qs", fill(tiles * 4 * 3 * 16 * 4, 0), 4),
                    ("kx", fill(rr * 1024 * 2, 0), 4),
                    ("ks", fill(rr * 16, 0), 4),
                    ("vx", fill(rr * 1024 * 2, 0), 2),
                    ("mu", fill(4096, 0), 4),
                    ("st", fill(tiles * 4 * 8 * 1632 * 4, 0), 4),
                ],
                launch: Box::new(move |k, q, p| {
                    let s = fe_kxcore::PfScratch::from_parts([p[5], p[6], p[7], p[8], p[9], p[10], p[11]], range, rows);
                    k.attn_prefill(q, &s, p[0], p[1], p[2], p[3], p[4], pos0, rows)
                }),
            };
            check(&ze, &q, &rs, &cl, &c, &o, &mut t, rows >= 512)?;
        }
    }
    if want(&o, "gdnpre") {
        println!("== fused GDN window chain (gdn_pre_r / gdn_pre_tr == conv_prep + vconv + gates)");
        for &m in &o.rows {
            let mut bufs = gdn_conv_bufs(&mut r, m);
            let mu = (m as usize).max(16);
            let h = 48usize;
            bufs.push(("qn", fill(mu * 2048 * 4, 0x55), 4));
            bufs.push(("kn", fill(mu * 2048 * 4, 0x55), 4));
            bufs.push(("vconv", fill(mu * GDN_V as usize * 2, 0x55), 2));
            let a: Vec<u16> = (0..mu * h).map(|i| bf16(if i % 7 == 3 { 18.0 + 6.0 * r.f32() } else { 3.0 * r.n() })).collect();
            bufs.push(("b", bf16v(&mut r, mu * h, 2.0), 2));
            bufs.push(("a", bytes(&a), 2));
            bufs.push(("a_log", f32v(&mut r, h, |r| -2.0 + 4.5 * r.f32()), 4));
            bufs.push(("dt_bias", f32v(&mut r, h, |r| r.n()), 4));
            bufs.push(("beta", fill(mu * h * 4, 0x55), 4));
            bufs.push(("g", fill(mu * h * 4, 0x55), 4));
            let chain = Case {
                name: format!("gdn chain m={m}"),
                bufs: bufs.iter().map(|(a, b, e)| (*a, b.clone(), *e)).collect(),
                launch: Box::new(move |k, q, p| {
                    k.gdn_conv_prep_r(q, p[0], p[1], p[2], p[3], p[4], m)?;
                    k.gdn_vconv_r(q, p[0], p[1], p[2], p[5], m)?;
                    k.gdn_gates_r(q, p[6], p[7], p[8], p[9], p[10], p[11], m)
                }),
            };
            let fused = Case { name: format!("gdn_pre_r m={m}"), bufs, launch: Box::new(move |k, q, p| k.gdn_pre_r(q, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], m, true)) };
            // the chain on the reference kernels == the fused Rust kernel
            let a = run(&cl, &q, &chain)?;
            let b = run(&rs, &q, &fused)?;
            let n = compare(&fused, "opencl chain vs rust gdn_pre_r", &a, &b);
            t.fails += n;
            let mut line = format!("  {:44} {}", fused.name, if n == 0 { "bitwise == chain".to_string() } else { format!("{n} MISMATCHES") });
            if o.time && (m == 1 || m == 8 || m == 16 || m >= 512) {
                let (mut tc, mut tf, mut tr) = (f64::MAX, f64::MAX, f64::MAX);
                for _ in 0..3 {
                    tc = tc.min(time(&ze, &cl, &q, &chain, o.iters)?);
                    tr = tr.min(time(&ze, &rs, &q, &chain, o.iters)?);
                    tf = tf.min(time(&ze, &rs, &q, &fused, o.iters)?);
                }
                line += &format!("   opencl chain {tc:7.2} us, rust chain {tr:7.2} us, fused {tf:7.2} us");
                t.table.push(("gdn chain->pre_r".into(), format!("m={m}"), tf, tc));
            }
            println!("{line}");
        }
        // the tree form against the tree chain
        for par in [vec![-1, 0, 0, 1, 1, 2, 2], vec![-1, 0, 1, 2, 1, 0, 5, 6, 5], (0..16).map(|t| t - 1).collect::<Vec<i32>>()] {
            let tt = TreeTable::new(&par)?;
            let m = par.len() as u32;
            let mut bufs = gdn_conv_bufs(&mut r, m);
            let mu = 16usize;
            bufs.push(("qn", fill(mu * 2048 * 4, 0x55), 4));
            bufs.push(("kn", fill(mu * 2048 * 4, 0x55), 4));
            bufs.push(("vconv", fill(mu * GDN_V as usize * 2, 0x55), 2));
            bufs.push(("b", bf16v(&mut r, mu * 48, 2.0), 2));
            bufs.push(("a", bf16v(&mut r, mu * 48, 3.0), 2));
            bufs.push(("a_log", f32v(&mut r, 48, |r| -2.0 + 4.5 * r.f32()), 4));
            bufs.push(("dt_bias", f32v(&mut r, 48, |r| r.n()), 4));
            bufs.push(("beta", fill(mu * 48 * 4, 0x55), 4));
            bufs.push(("g", fill(mu * 48 * 4, 0x55), 4));
            bufs.push(("tree", bytes(&tt.words), 4));
            let chain = Case {
                name: format!("gdn tree chain m={m}"),
                bufs: bufs.iter().map(|(a, b, e)| (*a, b.clone(), *e)).collect(),
                launch: Box::new(move |k, q, p| {
                    k.gdn_conv_prep_tr(q, p[0], p[1], p[2], p[3], p[4], m, p[12])?;
                    k.gdn_vconv_tr(q, p[0], p[1], p[2], p[5], m, p[12])?;
                    k.gdn_gates_r(q, p[6], p[7], p[8], p[9], p[10], p[11], m)
                }),
            };
            let fused = Case { name: format!("gdn_pre_tr m={m}"), bufs, launch: Box::new(move |k, q, p| k.gdn_pre_tr(q, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], m, true, p[12])) };
            let a = run(&rs, &q, &chain)?;
            let b = run(&rs, &q, &fused)?;
            let n = compare(&fused, "tree chain vs gdn_pre_tr", &a, &b);
            t.fails += n;
            println!("  {:44} {}", fused.name, if n == 0 { "bitwise == tree chain" } else { "MISMATCH" });
        }
    }
    if want(&o, "tree") {
        println!("== tree verify windows (each row == its root path decoded alone with the linear kernels)");
        let trees: Vec<(&str, Vec<i32>)> = vec![
            ("chain4", vec![-1, 0, 1, 2]),
            ("star5", vec![-1, 0, 0, 0, 0]),
            ("binary7", vec![-1, 0, 0, 1, 1, 2, 2]),
            ("dfs9", vec![-1, 0, 1, 2, 1, 0, 5, 6, 5]),
            ("mix16", vec![-1, 0, 1, 2, 3, 1, 5, 0, 7, 8, 7, 2, 11, 3, 13, 14]),
            ("chain16", (0..16).map(|t| t - 1).collect()),
        ];
        for (tn, par) in &trees {
            let tt = TreeTable::new(par)?;
            let m = par.len() as u32;
            for &pos0 in &[37u32, 1000, 2047, 20000] {
                t.fails += tree_check(&q, &rs, &mut r, tn, &tt, m, pos0, o.time && pos0 == 2047)?;
            }
        }
    }
    if want(&o, "embed") {
        println!("== embedding");
        for &m in &o.rows {
            check(&ze, &q, &rs, &cl, &case_embed(&mut r, m), &o, &mut t, tsel(m))?;
        }
    }
    if want(&o, "swiglu") {
        println!("== swiglu");
        for &m in &o.rows {
            let c = case_swiglu(&mut r, m);
            check(&ze, &q, &rs, &cl, &c, &o, &mut t, tsel(m))?;
        }
    }
    if want(&o, "round") {
        println!("== bf16 logits rounding");
        for m in [1u32, 4] {
            check(&ze, &q, &rs, &cl, &case_round(&mut r, m), &o, &mut t, m == 1)?;
        }
    }
    if want(&o, "argmax") {
        println!("== argmax");
        for &m in &o.rows {
            check(&ze, &q, &rs, &cl, &case_argmax(&mut r, m), &o, &mut t, tsel(m))?;
        }
    }
    if !t.table.is_empty() {
        println!("\n{:24} {:22} {:>10} {:>10} {:>7}", "kernel", "shape", "rust us", "opencl us", "ocl/rs");
        for (k, s, a, b) in &t.table {
            println!("{k:24} {s:22} {a:10.2} {b:10.2} {:7.3}", b / a);
        }
    }
    println!("\n{}", if t.fails == 0 { "ALL BITWISE OK".to_string() } else { format!("FAILED: {} mismatching values", t.fails) });
    std::process::exit(if t.fails == 0 { 0 } else { 1 });
}

/// Replays every ARC_KDUMP launch of a kx-core kernel: the reference first (the replay must reproduce the dump), then
/// the Rust kernel, both against the dumped `post` bytes.
fn replay_dumps(q: &Queue, rs: &KxCore, cl: &KxCore, spec: &str) -> Result<()> {
    let mut files = Vec::new();
    for p in spec.split(',') {
        let p = std::path::PathBuf::from(p);
        if p.is_dir() {
            let mut v: Vec<_> = std::fs::read_dir(&p)?.flatten().map(|e| e.path()).filter(|f| f.extension().map(|x| x == "kd").unwrap_or(false)).collect();
            v.sort();
            files.extend(v);
        } else {
            files.push(p);
        }
    }
    let mut fails = 0;
    let mut done = 0;
    for f in files {
        let d = fe_kxcore::kd::KernelDump::read(&f)?;
        if !KxCore::has(&d.name) {
            println!("  {:56} skip ({} not in kx-core)", f.file_name().unwrap().to_string_lossy(), d.name);
            continue;
        }
        let a = fe_kxcore::kd::replay(cl, q, &d)?;
        let b = fe_kxcore::kd::replay(rs, q, &d)?;
        let words: usize = b.outputs.iter().map(|x| x.2).sum();
        let (ref_bad, rs_bad, how) = if a.truncated {
            // a buffer was dumped in part: the replays read zeros past it, so compare rust with the reference replay
            let mut n = 0;
            for (x, y) in a.after.iter().zip(&b.after) {
                if let (Some(x), Some(y)) = (x, y) {
                    n += x.chunks(4).zip(y.chunks(4)).filter(|(p, q)| p != q).count();
                }
            }
            (0, n, "rust vs opencl replay, truncated dump")
        } else {
            (a.outputs.iter().map(|x| x.1).sum::<usize>() + a.unchanged_bad, b.outputs.iter().map(|x| x.1).sum::<usize>() + b.unchanged_bad, "vs post")
        };
        println!(
            "  {:44} {:>9} words in {} outputs: opencl {} / rust {} mismatches ({how}) {}",
            f.file_name().unwrap().to_string_lossy(),
            words,
            b.outputs.len(),
            ref_bad,
            rs_bad,
            if rs_bad == 0 && ref_bad == 0 { "OK" } else { "FAIL" }
        );
        fails += rs_bad + ref_bad;
        done += 1;
    }
    println!("\n{done} dumps replayed: {}", if fails == 0 { "ALL BITWISE OK".to_string() } else { format!("FAILED ({fails} words)") });
    std::process::exit(if fails == 0 { 0 } else { 1 });
}

fn up<T: Copy>(q: &Queue, v: &[T]) -> Result<DeviceBuffer<u8>> {
    Ok(DeviceBuffer::from_slice(q, &bytes(v))?)
}
fn down<T: Copy + Default>(q: &Queue, b: &DeviceBuffer<u8>) -> Result<Vec<T>> {
    let raw = b.to_vec(q)?;
    let n = raw.len() / std::mem::size_of::<T>();
    let mut v = vec![T::default(); n];
    unsafe { std::ptr::copy_nonoverlapping(raw.as_ptr(), v.as_mut_ptr() as *mut u8, raw.len()) };
    Ok(v)
}
fn diffn<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Tree window vs the linear kernels on each row's gathered path: GDN (q / k conv + norm, v conv, delta rule y) and
/// decode attention (tail windows + attn_dec_tree + merge vs attn_dec_partial + merge over a cache holding the path).
fn tree_check(q: &Queue, k: &KxCore, r: &mut Rng, name: &str, tt: &TreeTable, m: u32, pos0: u32, time_it: bool) -> Result<usize> {
    let mu = m as usize;
    let c = GDN_QKV as usize;
    let (vq, vv) = (2048usize, GDN_V as usize);
    let sz = (GDN_HEADS * 128 * 128) as usize;
    // ---- GDN ----
    let qkv: Vec<u16> = (0..mu * c).map(|_| bf16(r.n())).collect();
    let cst: Vec<u16> = (0..3 * c).map(|_| bf16(r.n())).collect();
    let cw: Vec<u16> = (0..4 * c).map(|_| bf16(r.n() * 0.4)).collect();
    let bb: Vec<u16> = (0..mu * 48).map(|_| bf16(r.n() * 2.0)).collect();
    let aa: Vec<u16> = (0..mu * 48).map(|_| bf16(r.n() * 3.0)).collect();
    let alog: Vec<f32> = (0..48).map(|_| -2.0 + 4.5 * r.f32()).collect();
    let dtb: Vec<f32> = (0..48).map(|_| r.n()).collect();
    let s_in: Vec<f32> = (0..sz).map(|_| 0.05 * r.n()).collect();
    let tree = up(q, &tt.words)?;
    let (d_qkv, d_cst, d_cw, d_b, d_a, d_al, d_dt, d_s) = (up(q, &qkv)?, up(q, &cst)?, up(q, &cw)?, up(q, &bb)?, up(q, &aa)?, up(q, &alog)?, up(q, &dtb)?, up(q, &s_in)?);
    let z = |n: usize| DeviceBuffer::<u8>::zeroed(q, n);
    let (qn, kn, vc, beta, g, y) = (z(mu * vq * 4)?, z(mu * vq * 4)?, z(mu * vv * 2)?, z(mu * 48 * 4)?, z(mu * 48 * 4)?, z(mu * vv * 2)?);
    let sscr = z(tt.slots as usize * sz * 4)?;
    let tree_gdn = |k: &KxCore| -> Result<()> {
        k.gdn_conv_prep_tr(q, d_qkv.ptr(), d_cst.ptr(), d_cw.ptr(), qn.ptr(), kn.ptr(), m, tree.ptr())?;
        k.gdn_vconv_tr(q, d_qkv.ptr(), d_cst.ptr(), d_cw.ptr(), vc.ptr(), m, tree.ptr())?;
        k.gdn_gates_r(q, d_b.ptr(), d_a.ptr(), d_al.ptr(), d_dt.ptr(), beta.ptr(), g.ptr(), m)?;
        k.gdn_step_tr(q, qn.ptr(), kn.ptr(), vc.ptr(), beta.ptr(), g.ptr(), d_s.ptr(), sscr.ptr(), y.ptr(), false, m, tree.ptr())
    };
    tree_gdn(k)?;
    q.sync()?;
    let (tqn, tkn, tvc, ty) = (down::<f32>(q, &qn)?, down::<f32>(q, &kn)?, down::<u16>(q, &vc)?, down::<u16>(q, &y)?);
    let s_after: Vec<f32> = down(q, &d_s)?;
    let mut bad = diffn(&s_after, &s_in); // s_in must stay the committed state
    for t in 0..mu {
        let path = tt.path(t);
        let l = path.len();
        let gather = |src: &[u16], w: usize| -> Vec<u16> { path.iter().flat_map(|&p| src[p * w..(p + 1) * w].iter().copied()).collect() };
        let (lq, lb, la) = (up(q, &gather(&qkv, c))?, up(q, &gather(&bb, 48))?, up(q, &gather(&aa, 48))?);
        let (qn1, kn1, vc1, be1, g1, y1, so1) = (z(l * vq * 4)?, z(l * vq * 4)?, z(l * vv * 2)?, z(l * 48 * 4)?, z(l * 48 * 4)?, z(l * vv * 2)?, z(sz * 4)?);
        let lu = l as u32;
        k.gdn_conv_prep_r(q, lq.ptr(), d_cst.ptr(), d_cw.ptr(), qn1.ptr(), kn1.ptr(), lu)?;
        k.gdn_vconv_r(q, lq.ptr(), d_cst.ptr(), d_cw.ptr(), vc1.ptr(), lu)?;
        k.gdn_gates_r(q, lb.ptr(), la.ptr(), d_al.ptr(), d_dt.ptr(), be1.ptr(), g1.ptr(), lu)?;
        k.gdn_step(q, qn1.ptr(), kn1.ptr(), vc1.ptr(), be1.ptr(), g1.ptr(), d_s.ptr(), so1.ptr(), y1.ptr(), false, lu)?;
        q.sync()?;
        let e = l - 1;
        let (a_qn, a_kn, a_vc, a_y) = (down::<f32>(q, &qn1)?, down::<f32>(q, &kn1)?, down::<u16>(q, &vc1)?, down::<u16>(q, &y1)?);
        let n = diffn(&tqn[t * vq..(t + 1) * vq], &a_qn[e * vq..(e + 1) * vq])
            + diffn(&tkn[t * vq..(t + 1) * vq], &a_kn[e * vq..(e + 1) * vq])
            + diffn(&tvc[t * vv..(t + 1) * vv], &a_vc[e * vv..(e + 1) * vv])
            + diffn(&ty[t * vv..(t + 1) * vv], &a_y[e * vv..(e + 1) * vv]);
        if n > 0 {
            println!("    TREE {name} pos0={pos0} GDN row {t} (path {path:?}): {n} values differ from the linear path");
        }
        bad += n;
    }
    // ---- attention ----
    let ncache = pos0 as usize + mu;
    let kcv = kv_cache(r, ncache);
    let vcv = kv_cache(r, ncache);
    let qv: Vec<u16> = (0..mu * Q_DIM as usize).map(|_| bf16(r.n() * 2.0)).collect();
    let qg: Vec<u16> = (0..mu * 24 * 512).map(|_| bf16(r.n() * 2.0)).collect();
    let (d_kc, d_vc, d_q, d_qg) = (DeviceBuffer::from_slice(q, &kcv)?, DeviceBuffer::from_slice(q, &vcv)?, up(q, &qv)?, up(q, &qg)?);
    let por = MAX_CHUNKS as usize * Q_DIM as usize * 4;
    let pmr = MAX_CHUNKS as usize * 24 * 4;
    let (po, pm, pl, out) = (z(mu * por)?, z(mu * pmr)?, z(mu * pmr)?, z(mu * Q_DIM as usize * 2)?);
    let (tk, tv) = (z(mu * TREE_TAIL_BYTES)?, z(mu * TREE_TAIL_BYTES)?);
    let tree_attn_tg = |k: &KxCore, tg: u32| -> Result<()> {
        k.kv_tree_tail(q, d_kc.ptr(), d_vc.ptr(), tk.ptr(), tv.ptr(), tree.ptr(), pos0, m)?;
        k.attn_dec_tree_tg(q, d_q.ptr(), d_kc.ptr(), d_vc.ptr(), po.ptr(), pm.ptr(), pl.ptr(), pos0, m, tt.max_depth(), tree.ptr(), tk.ptr(), tv.ptr(), tg)?;
        k.attn_long_merge_tr(q, po.ptr(), pm.ptr(), pl.ptr(), d_qg.ptr(), out.ptr(), pos0, m, tree.ptr())
    };
    let tree_attn = |k: &KxCore| -> Result<()> {
        k.kv_tree_tail(q, d_kc.ptr(), d_vc.ptr(), tk.ptr(), tv.ptr(), tree.ptr(), pos0, m)?;
        k.attn_dec_tree(q, d_q.ptr(), d_kc.ptr(), d_vc.ptr(), po.ptr(), pm.ptr(), pl.ptr(), pos0, m, tt.max_depth(), tree.ptr(), tk.ptr(), tv.ptr())?;
        k.attn_long_merge_tr(q, po.ptr(), pm.ptr(), pl.ptr(), d_qg.ptr(), out.ptr(), pos0, m, tree.ptr())
    };
    // every work-group form (1 / 2 / 4 tiles: the engine picks by occupancy) must give the same bits
    let mut forms = Vec::new();
    for tg in [1u32, 2, 4] {
        tree_attn_tg(k, tg)?;
        q.sync()?;
        forms.push(down::<u16>(q, &out)?);
    }
    for (i, tg) in [2u32, 4].iter().enumerate() {
        let n = diffn(&forms[i + 1], &forms[0]);
        if n > 0 {
            println!("    TREE {name} pos0={pos0}: attn_dec_tree with {tg} tiles a group differs from 1 tile in {n} values");
        }
        bad += n;
    }
    tree_attn(k)?;
    q.sync()?;
    let tout = down::<u16>(q, &out)?;
    let rec = 4 * KV_REC as usize;
    let (po1, pm1, pl1, out1) = (z(por)?, z(pmr)?, z(pmr)?, z(Q_DIM as usize * 2)?);
    for t in 0..mu {
        let path = tt.path(t);
        let len = pos0 as usize + path.len();
        let mut kp = kcv[..len * rec].to_vec();
        let mut vp = vcv[..len * rec].to_vec();
        for (e, &p) in path.iter().enumerate() {
            let (dst, src) = ((pos0 as usize + e) * rec, (pos0 as usize + p) * rec);
            kp[dst..dst + rec].copy_from_slice(&kcv[src..src + rec]);
            vp[dst..dst + rec].copy_from_slice(&vcv[src..src + rec]);
        }
        let (dk, dv) = (DeviceBuffer::from_slice(q, &kp)?, DeviceBuffer::from_slice(q, &vp)?);
        let qrow = d_q.ptr().byte_add(t * Q_DIM as usize * 2);
        k.attn_dec_with(q, "attn_dec_partial", qrow, dk.ptr(), dv.ptr(), po1.ptr(), pm1.ptr(), pl1.ptr(), len as u32, 1)?;
        k.attn_long_merge(q, po1.ptr(), pm1.ptr(), pl1.ptr(), d_qg.ptr().byte_add(t * 24 * 512 * 2), out1.ptr(), len as u32, 1)?;
        q.sync()?;
        let a = down::<u16>(q, &out1)?;
        let n = diffn(&tout[t * Q_DIM as usize..(t + 1) * Q_DIM as usize], &a);
        if n > 0 {
            println!("    TREE {name} pos0={pos0} attention row {t} (path {path:?}): {n} values differ from attn_dec_partial on the path");
        }
        bad += n;
    }
    // ---- commit: the deepest path to positions pos0 .. ----
    let deepest = (0..mu).max_by_key(|&t| tt.depth[t]).unwrap();
    let path = tt.path(deepest);
    let pidx: Vec<u32> = path.iter().map(|&p| p as u32).collect();
    let d_path = up(q, &pidx)?;
    k.kv_tree_commit(q, d_kc.ptr(), d_vc.ptr(), d_path.ptr(), pidx.len() as u32, pos0)?;
    q.sync()?;
    let (kc2, vc2) = (d_kc.to_vec(q)?, d_vc.to_vec(q)?);
    let mut nc = 0;
    for (e, &p) in path.iter().enumerate() {
        let (dst, src) = ((pos0 as usize + e) * rec, (pos0 as usize + p) * rec);
        nc += diffn(&kc2[dst..dst + rec], &kcv[src..src + rec]) + diffn(&vc2[dst..dst + rec], &vcv[src..src + rec]);
    }
    nc += diffn(&kc2[..pos0 as usize * rec], &kcv[..pos0 as usize * rec]);
    if nc > 0 {
        println!("    TREE {name} pos0={pos0} kv_tree_commit: {nc} bytes wrong");
    }
    bad += nc;
    let mut line = format!("  tree {name:8} rows {m:2} depth {:2} slots {} pos0 {pos0:6}  {}", tt.max_depth(), tt.slots, if bad == 0 { "bitwise == linear paths" } else { "MISMATCH" });
    if time_it {
        let ze = q.ze().clone();
        let tm = |f: &dyn Fn(&KxCore) -> Result<()>| -> Result<f64> {
            for _ in 0..3 {
                f(k)?;
            }
            q.sync()?;
            k.set_timing(&ze, true)?;
            let mut v = Vec::new();
            for _ in 0..15 {
                f(k)?;
                q.sync()?;
                v.push(k.take_times()?.iter().sum::<f64>());
            }
            k.set_timing(&ze, false)?;
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            Ok(v[v.len() / 2] / 1000.0)
        };
        let tg = tm(&tree_gdn)?;
        let ta = tm(&tree_attn)?;
        line += &format!("   gdn {tg:.1} us, attention {ta:.1} us");
    }
    println!("{line}");
    Ok(bad)
}
