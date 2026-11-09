//! `gdn-check [model_dir] [len,len,...]`: fe_gdn_chunk vs an f64 row-by-row reference of fe_gdn_rec2's math
//! (same inputs: random post-conv activations and a / b projections, real A_log / dt_bias of layer 0, a
//! random initial state), several segments of the given lengths in one launch; then timing at 2048 rows.
use std::path::PathBuf;
use anyhow::{ensure, Result};
use fe_cuda::{Context, DeviceBuffer, Event, Stream};
use fe_fp4::format::{bf16_to_f32, f32_to_bf16};

const CONV: usize = 10240;
const NVH: usize = 48;
const D: usize = 128;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn normal(&mut self) -> f32 {
        let u = ((self.next() >> 40) as f32 + 0.5) / (1u64 << 24) as f32;
        let v = ((self.next() >> 40) as f32 + 0.5) / (1u64 << 24) as f32;
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// f64 reference of one head over rows [r0, r0 + n): updates `st` ([128][128], S[k][v]) and writes o rows.
fn reference(h: usize, r0: usize, n: usize, conv: &[u16], ax: &[u16], bx: &[u16], a_log: f32, dtb: f32, st: &mut [f64],
    o: &mut [f64]) {
    let kh = h / 3;
    let a_neg = -(a_log as f64).exp();
    for r in r0..r0 + n {
        let row = &conv[r * CONV..(r + 1) * CONV];
        let mut q: Vec<f64> = (0..D).map(|i| bf16_to_f32(row[kh * D + i]) as f64).collect();
        let mut k: Vec<f64> = (0..D).map(|i| bf16_to_f32(row[2048 + kh * D + i]) as f64).collect();
        let nq = 1.0 / (q.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt();
        let nk = 1.0 / (k.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt();
        q.iter_mut().for_each(|x| *x *= nq / (128f64).sqrt());
        k.iter_mut().for_each(|x| *x *= nk);
        let v: Vec<f64> = (0..D).map(|j| bf16_to_f32(row[4096 + h * D + j]) as f64).collect();
        let beta = bf16_to_f32(f32_to_bf16(sigmoid(bf16_to_f32(bx[r * NVH + h])))) as f64;
        let g = (a_neg * softplus(bf16_to_f32(ax[r * NVH + h]) as f64 + dtb as f64)).exp();
        st.iter_mut().for_each(|x| *x *= g);
        for jj in 0..D {
            let kv: f64 = (0..D).map(|i| st[i * D + jj] * k[i]).sum();
            let d = (v[jj] - kv) * beta;
            for i in 0..D {
                st[i * D + jj] += k[i] * d;
            }
            o[r * NVH * D + h * D + jj] = (0..D).map(|i| st[i * D + jj] * q[i]).sum();
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).map(PathBuf::from).unwrap_or_else(fe_site::nvfp4_dir);
    let lens: Vec<usize> = args.get(2).map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or(vec![77, 200, 32, 1]);
    let _ctx = Context::new(0)?;
    let g = fe_gdn::Gdn::load()?;
    println!("kernel: {}", ["fe_gdn_chunk (fp32 SIMT)", "fe_gdn_chunk_tc1 (TF32 mma)", "", "fe_gdn_chunk_tc (3xTF32 mma)"][g.tc.min(3) as usize]);
    let s = Stream::new()?;
    let ck = fe_fp4::Checkpoint::open(&dir)?;
    let rd = |name: &str| -> Result<Vec<u16>> {
        let b = ck.read(name)?;
        Ok(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    };
    let a_log = rd("model.language_model.layers.0.linear_attn.A_log")?;
    let dt_bias = rd("model.language_model.layers.0.linear_attn.dt_bias")?;
    let rows: usize = lens.iter().sum();
    let nseg = lens.len();
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let conv: Vec<u16> = (0..rows * CONV).map(|_| f32_to_bf16(rng.normal() * 0.5)).collect();
    let ax: Vec<u16> = (0..rows * NVH).map(|_| f32_to_bf16(rng.normal())).collect();
    let bx: Vec<u16> = (0..rows * NVH).map(|_| f32_to_bf16(rng.normal())).collect();
    let st0: Vec<f32> = (0..nseg * NVH * D * D).map(|_| rng.normal() * 0.05).collect();
    let (mut slot, mut row0) = (Vec::new(), Vec::new());
    let mut r = 0;
    for (i, &n) in lens.iter().enumerate() {
        slot.push(i as u32);
        row0.push(r as u32);
        r += n;
    }
    let len32: Vec<u32> = lens.iter().map(|&n| n as u32).collect();
    let (dconv, dax, dbx) = (DeviceBuffer::from_slice(&conv)?, DeviceBuffer::from_slice(&ax)?, DeviceBuffer::from_slice(&bx)?);
    let (dal, ddt) = (DeviceBuffer::from_slice(&a_log)?, DeviceBuffer::from_slice(&dt_bias)?);
    let dst = DeviceBuffer::from_slice(&st0)?;
    let (dslot, drow0, dlen) = (DeviceBuffer::from_slice(&slot)?, DeviceBuffer::from_slice(&row0)?, DeviceBuffer::from_slice(&len32)?);
    let dout = DeviceBuffer::<u16>::zeroed(rows * NVH * D)?;
    unsafe {
        g.run(&s, dout.ptr(), dconv.ptr(), dax.ptr(), dbx.ptr(), NVH, dal.ptr(), ddt.ptr(), dst.ptr(), dslot.ptr(), drow0.ptr(),
            dlen.ptr(), nseg)?;
    }
    s.sync()?;
    let og = dout.to_vec()?;
    let sg = dst.to_vec()?;
    // determinism: rerun from the same initial state, compare bitwise
    let reps: usize = std::env::var("FE_GDN_DET").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    for rep in 0..reps {
        let d2 = DeviceBuffer::from_slice(&st0)?;
        let o2 = DeviceBuffer::<u16>::zeroed(rows * NVH * D)?;
        unsafe {
            g.run(&s, o2.ptr(), dconv.ptr(), dax.ptr(), dbx.ptr(), NVH, dal.ptr(), ddt.ptr(), d2.ptr(), dslot.ptr(), drow0.ptr(),
                dlen.ptr(), nseg)?;
        }
        s.sync()?;
        let same = o2.to_vec()? == og && d2.to_vec()? == sg;
        println!("determinism rerun {rep}: {}", if same { "bitwise identical" } else { "DIFFERS" });
        ensure!(same, "k-gdn nondeterministic");
    }

    // f64 reference, one thread per head
    let mut oref = vec![0.0f64; rows * NVH * D];
    let mut sref: Vec<f64> = st0.iter().map(|&x| x as f64).collect();
    std::thread::scope(|sc| {
        let mut ochunks: Vec<&mut [f64]> = Vec::new();
        let _ = &mut ochunks;
        let hs: Vec<_> = (0..NVH)
            .map(|h| {
                let (conv, ax, bx, a_log, dt_bias, lens) = (&conv, &ax, &bx, &a_log, &dt_bias, &lens);
                let st0 = &st0;
                sc.spawn(move || {
                    let mut o = vec![0.0f64; rows * NVH * D];
                    let mut sts = Vec::new();
                    let mut r0 = 0;
                    for (i, &n) in lens.iter().enumerate() {
                        let mut st: Vec<f64> = st0[(i * NVH + h) * D * D..(i * NVH + h + 1) * D * D].iter().map(|&x| x as f64).collect();
                        reference(h, r0, n, conv, ax, bx, bf16_to_f32(a_log[h]), bf16_to_f32(dt_bias[h]), &mut st, &mut o);
                        sts.push(st);
                        r0 += n;
                    }
                    (h, o, sts)
                })
            })
            .collect();
        for hd in hs {
            let (h, o, sts) = hd.join().unwrap();
            for r in 0..rows {
                for jj in 0..D {
                    oref[r * NVH * D + h * D + jj] = o[r * NVH * D + h * D + jj];
                }
            }
            for (i, st) in sts.into_iter().enumerate() {
                sref[(i * NVH + h) * D * D..(i * NVH + h + 1) * D * D].copy_from_slice(&st);
            }
        }
    });
    // errors: o relative to each row's per-head rms; state relative to its rms
    let (mut worst_o, mut se, mut ss) = (0.0f64, 0.0f64, 0.0f64);
    for r in 0..rows {
        for h in 0..NVH {
            let b = r * NVH * D + h * D;
            let rms = (oref[b..b + D].iter().map(|x| x * x).sum::<f64>() / D as f64).sqrt().max(1e-12);
            for jj in 0..D {
                let e = (bf16_to_f32(og[b + jj]) as f64 - oref[b + jj]).abs();
                worst_o = worst_o.max(e / rms);
                se += e * e;
                ss += oref[b + jj] * oref[b + jj];
            }
        }
    }
    let (mut sse, mut sss, mut worst_s) = (0.0f64, 0.0f64, 0.0f64);
    let srms = (sref.iter().map(|x| x * x).sum::<f64>() / sref.len() as f64).sqrt();
    for (a, b) in sg.iter().zip(&sref) {
        let e = (*a as f64 - b).abs();
        sse += e * e;
        sss += b * b;
        worst_s = worst_s.max(e / srms);
    }
    let (rel_o, rel_s) = ((se / ss).sqrt(), (sse / sss).sqrt());
    println!("segments {lens:?}: o rel rms {rel_o:.2e} (max |err| / row rms {worst_o:.2e}; bf16 output rounding ~2e-3), \
              state rel rms {rel_s:.2e} (max {worst_s:.2e})");
    let ok = rel_o < 4e-3 && worst_o < 3e-2 && rel_s < if g.tc == 1 { 2e-3 } else { 1e-4 };
    println!("{}", if ok { "OK" } else { "FAIL" });

    if std::env::var("FE_GDN_NOTIME").is_ok() {
        ensure!(ok, "gdn-check failed");
        return Ok(());
    }
    // timing: one 2048-row segment
    let n = 2048usize;
    let conv2: Vec<u16> = (0..n * CONV).map(|_| f32_to_bf16(rng.normal() * 0.5)).collect();
    let ab2: Vec<u16> = (0..n * NVH).map(|_| f32_to_bf16(rng.normal())).collect();
    let (c2, a2) = (DeviceBuffer::from_slice(&conv2)?, DeviceBuffer::from_slice(&ab2)?);
    let o2 = DeviceBuffer::<u16>::alloc(n * NVH * D)?;
    let st2 = DeviceBuffer::<f32>::zeroed(NVH * D * D)?;
    let (s1, r1, l1) = (DeviceBuffer::from_slice(&[0u32])?, DeviceBuffer::from_slice(&[0u32])?, DeviceBuffer::from_slice(&[n as u32])?);
    let run = || unsafe {
        g.run(&s, o2.ptr(), c2.ptr(), a2.ptr(), a2.ptr(), NVH, dal.ptr(), ddt.ptr(), st2.ptr(), s1.ptr(), r1.ptr(), l1.ptr(), 1)
    };
    run()?;
    let (e0, e1) = (Event::new()?, Event::new()?);
    s.record(&e0)?;
    for _ in 0..10 {
        run()?;
    }
    s.record(&e1)?;
    e1.sync()?;
    println!("2048 rows x 48 heads: {:.3} ms (fe_gdn_rec2 ~5 ms in the engine)", e1.elapsed_since(&e0)? / 10.0);
    ensure!(ok, "gdn-check failed");
    Ok(())
}
