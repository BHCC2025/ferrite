//! fnext-binv: batch invariance of the fe-fnext decode kernels (test tool; launches k-fnext kernels by name on
//! synthetic inputs, no engine state). Every verify row at B = 2..5 must have the bits the same row gets from 1-row
//! decode steps run one after another (stateful kernels: KV / indexer caches, GDN conv + recurrent state).
//!
//! fnext-binv [model.gguf]: prints per kernel group the rows that differ per B; "OK" when none.
use anyhow::Result;
use fe_cuda::{Context, DevPtr, DeviceBuffer, Function, Module, Stream};
use fe_fnext::config::Config;

// step parameter words (kernels/k-fnext/src/fast.rs)
const P_POS: usize = 0;
const P_ROWS: usize = 1;
const P_NB: usize = 2;
const P_BLO: usize = 3;
const P_CS: usize = 16;
const MAX_ROWS: usize = 24;
const MAX_BLK: usize = 8;
const P_SN: usize = P_CS + MAX_ROWS * 32;
const P_PCS: usize = P_SN + MAX_ROWS * 32;
const P_PSN: usize = P_PCS + MAX_BLK * 32;
const P_WORDS: usize = P_PSN + MAX_BLK * 32;
/// Rows per verify step to check (env FNEXT_BINV_B, default 5 = 1 lane x MTP; the engine takes up to MAX_ROWS = 24; fx_gemv_h runs
/// in 8-row launches as the engine's hrun does).
fn bmax() -> usize {
    std::env::var("FNEXT_BINV_B").ok().and_then(|v| v.parse().ok()).unwrap_or(5)
}

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn v(&mut self, n: usize, s: f32) -> Vec<f32> {
        (0..n).map(|_| self.f() * s).collect()
    }
    fn i8v(&mut self, n: usize) -> Vec<i8> {
        (0..n).map(|_| (self.f() * 127.0) as i8).collect()
    }
}

fn g256(n: usize) -> u32 {
    n.div_ceil(256) as u32
}
fn same_rows(a: &[f32], b: &[f32], n: usize) -> usize {
    a.chunks(n).zip(b.chunks(n)).filter(|(x, y)| x.iter().zip(y.iter()).any(|(p, q)| p.to_bits() != q.to_bits())).count()
}
fn bits_eq<T: Copy + PartialEq>(a: &[T], b: &[T]) -> bool {
    a == b
}
fn dup<T: Copy>(b: &DeviceBuffer<T>) -> Result<DeviceBuffer<T>> {
    Ok(DeviceBuffer::from_slice(&b.to_vec()?)?)
}

/// Step parameters for `rows` rows at pos0.. (as fe-fnext Fast::fill_prm).
fn prm(c: &Config, pos0: usize, rows: usize) -> Vec<u32> {
    let mut p = vec![0u32; P_WORDS];
    p[P_POS] = pos0 as u32;
    p[P_ROWS] = rows as u32;
    let kp = c.kpool;
    let (b_lo, b_hi) = (pos0 / kp, (pos0 + rows) / kp);
    p[P_NB] = (b_hi - b_lo) as u32;
    p[P_BLO] = b_lo as u32;
    let rope = |pos: &[usize], cs: &mut [f32], sn: &mut [f32]| {
        let half = c.n_rot / 2;
        let ts = c.rope_base.powf(-2.0 / c.n_rot as f32);
        for (r, &q) in pos.iter().enumerate() {
            let mut th = q as f32;
            for k in 0..half {
                cs[r * 32 + k] = th.cos();
                sn[r * 32 + k] = th.sin();
                th *= ts;
            }
        }
    };
    let (mut cs, mut sn, mut pcs, mut psn) = (vec![0f32; MAX_ROWS * 32], vec![0f32; MAX_ROWS * 32], vec![0f32; MAX_BLK * 32],
        vec![0f32; MAX_BLK * 32]);
    rope(&(0..rows).map(|r| pos0 + r).collect::<Vec<_>>(), &mut cs, &mut sn);
    rope(&(b_lo..b_hi).map(|b| b * kp).collect::<Vec<_>>(), &mut pcs, &mut psn);
    for i in 0..MAX_ROWS * 32 {
        p[P_CS + i] = cs[i].to_bits();
        p[P_SN + i] = sn[i].to_bits();
    }
    for i in 0..MAX_BLK * 32 {
        p[P_PCS + i] = pcs[i].to_bits();
        p[P_PSN + i] = psn[i].to_bits();
    }
    p
}

struct T<'a> {
    s: &'a Stream,
    m: &'a Module,
    bad_total: usize,
}
impl T<'_> {
    fn f(&self, n: &str) -> Result<Function> {
        Ok(self.m.function(n)?)
    }
    fn report(&mut self, name: &str, bad: &[usize]) {
        let t: usize = bad.iter().sum();
        self.bad_total += t;
        if t == 0 {
            println!("{name:52} OK");
        } else {
            println!("{name:52} DIFFER: rows per B=1..{} {bad:?}", bmax());
        }
    }
}

fn main() -> Result<()> {
    let path = std::env::args().nth(1).map(std::path::PathBuf::from).unwrap_or_else(fe_fnext::default_model);
    let g = fe_fnext::open_model(&path)?;
    let c = Config::from_gguf(&g)?;
    let _cu = Context::new(0)?;
    let m = Module::load(fe_fnext::K_FNEXT_CUBIN)?;
    let s = Stream::new()?;
    let mut t = T { s: &s, m: &m, bad_total: 0 };
    let mut rng = Lcg(2026);
    rmsnorm(&mut t, &c, &mut rng)?;
    gdn(&mut t, &c, &mut rng)?;
    attention(&mut t, &c, &mut rng, 501)?;
    attention(&mut t, &c, &mut rng, 3001)?;
    router_argmax(&mut t, &c, &mut rng)?;
    mtp_parts(&mut t, &c, &mut rng)?;
    println!("fnext batch invariance: {} differing rows in total", t.bad_total);
    Ok(())
}

/// fn_rmsnorm with the engine's block choice (1024 threads up to 32 groups, else 256), groups = rows * per-row groups.
fn rmsnorm(t: &mut T, c: &Config, rng: &mut Lcg) -> Result<()> {
    let f = t.f("fn_rmsnorm")?;
    for (name, d, gpr, ngw) in [("rmsnorm hc (d 2560, 4 groups/row)", c.n_embd, c.hc, c.hc), ("rmsnorm indexer q (d 128, 4/row)", c.idx_dim, c.idx_heads, 1),
        ("rmsnorm MTP emb (d 2560, 1/row)", c.n_embd, 1, 1)] {
        let x = rng.v(bmax() * gpr * d, 3.0);
        let w = rng.v(ngw * d, 1.0);
        let (dx, dw) = (DeviceBuffer::from_slice(&x)?, DeviceBuffer::from_slice(&w)?);
        let dy = DeviceBuffer::<f32>::zeroed(bmax() * gpr * d)?;
        let run = |rows: usize, r0: usize| -> Result<Vec<f32>> {
            let groups = rows * gpr;
            let bs = if groups <= 32 { 1024 } else { 256 };
            unsafe { t.s.launch(&f, groups as u32, bs, 0, (dy.ptr(), dx.ptr().add::<f32>(r0 * gpr * d), dw.ptr(), d as u32, ngw as u32, c.eps))? };
            t.s.sync()?;
            Ok(dy.to_vec()?[..groups * d].to_vec())
        };
        let mut one = vec![];
        for r in 0..bmax() {
            one.extend(run(1, r)?);
        }
        let mut bad = vec![0; bmax()];
        for b in 1..=bmax() {
            bad[b - 1] = same_rows(&run(b, 0)?, &one[..b * gpr * d], gpr * d);
        }
        t.report(name, &bad);
        // 256- vs 1024-thread blocks on the same groups (prompt rows vs decode rows), every row
        let groups = bmax() * gpr;
        let mut y2 = vec![];
        for bs in [256u32, 1024] {
            unsafe { t.s.launch(&f, groups as u32, bs, 0, (dy.ptr(), dx.ptr(), dw.ptr(), d as u32, ngw as u32, c.eps))? };
            t.s.sync()?;
            y2.push(dy.to_vec()?[..groups * d].to_vec());
        }
        let d1 = same_rows(&y2[0], &y2[1], gpr * d);
        println!("{:52} {}", format!("  {name}: 256- vs 1024-thread block"), if d1 == 0 { "same bits".to_string() } else { format!("DIFFER on {d1} of {} rows", bmax()) });
    }
    Ok(())
}

/// GDN: conv + recurrence (+ gated norm) of a k-row verify (ro = 1) vs k one-row decode steps (ro = 0), and the commit
/// of n kept rows (conv_shift + recurrence replay) vs the state after n decode steps.
fn gdn(t: &mut T, c: &Config, rng: &mut Lcg) -> Result<()> {
    let (conv, rec, gnorm, shift) = (t.f("fn_gdn_conv")?, t.f("fn_gdn_rec")?, t.f("fn_gdn_gnorm")?, t.f("fn_conv_shift")?);
    let cd = c.conv_dim();
    let (vh, kh, hd) = (c.ssm_vh, c.ssm_kh, c.ssm_state);
    let qkv = rng.v(bmax() * cd, 1.0);
    let ab = rng.v(bmax() * 2 * vh, 1.0);
    let z = rng.v(bmax() * c.ssm_inner, 1.0);
    let (dqkv, dab, dz) = (DeviceBuffer::from_slice(&qkv)?, DeviceBuffer::from_slice(&ab)?, DeviceBuffer::from_slice(&z)?);
    let (cw, dtb, sa, nw) = (DeviceBuffer::from_slice(&rng.v(4 * cd, 0.5))?, DeviceBuffer::from_slice(&rng.v(vh, 0.5))?,
        DeviceBuffer::from_slice(&rng.v(vh, 0.5))?, DeviceBuffer::from_slice(&rng.v(hd, 1.0))?);
    let cstate0 = DeviceBuffer::from_slice(&rng.v((c.ssm_conv - 1) * cd, 1.0))?;
    let sstate0 = DeviceBuffer::from_slice(&rng.v(vh * hd * hd, 0.05))?;
    let convo = DeviceBuffer::<f32>::zeroed(bmax() * cd)?;
    let o = DeviceBuffer::<f32>::zeroed(bmax() * vh * hd)?;
    let y = DeviceBuffer::<f32>::zeroed(bmax() * c.ssm_inner)?;
    // one launch group of `rows` rows from r0 with conv / ssm state `cs` / `ss`
    let step = |rows: usize, r0: usize, cs: &DeviceBuffer<f32>, ss: &DeviceBuffer<f32>, ro: u32| -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        unsafe {
            t.s.launch(&conv, g256(cd), 256, 0, (convo.ptr(), dqkv.ptr().add::<f32>(r0 * cd), cd as u32, cw.ptr(), cs.ptr(), rows as u32,
                cd as u32, ro))?;
            t.s.launch(&rec, (vh as u32, 4), 128, 0, (o.ptr(), convo.ptr(), dab.ptr().add::<f32>(r0 * 2 * vh), (2 * vh) as u32, dtb.ptr(),
                sa.ptr(), ss.ptr(), rows as u32, kh as u32, vh as u32, c.eps, ro))?;
            t.s.launch(&gnorm, (rows as u32, vh.div_ceil(8) as u32), 256, 0, (y.ptr(), o.ptr(), dz.ptr().add::<f32>(r0 * c.ssm_inner),
                c.ssm_inner as u32, nw.ptr(), vh as u32, c.eps))?;
        }
        t.s.sync()?;
        Ok((convo.to_vec()?[..rows * cd].to_vec(), o.to_vec()?[..rows * vh * hd].to_vec(), y.to_vec()?[..rows * c.ssm_inner].to_vec()))
    };
    // reference: bmax() one-row decode steps (ro = 0), states after each
    let (cs1, ss1) = (dup(&cstate0)?, dup(&sstate0)?);
    let (mut rc, mut ro_, mut ry) = (vec![], vec![], vec![]);
    let mut states = vec![(cs1.to_vec()?, ss1.to_vec()?)];
    for r in 0..bmax() {
        let (a, b, cc) = step(1, r, &cs1, &ss1, 0)?;
        rc.extend(a);
        ro_.extend(b);
        ry.extend(cc);
        states.push((cs1.to_vec()?, ss1.to_vec()?));
    }
    let (mut bc, mut bo, mut by, mut bst) = (vec![0; bmax()], vec![0; bmax()], vec![0; bmax()], vec![0; bmax()]);
    for b in 1..=bmax() {
        let (cs, ss) = (dup(&cstate0)?, dup(&sstate0)?);
        let (a, bb, cc) = step(b, 0, &cs, &ss, 1)?;
        bc[b - 1] = same_rows(&a, &rc[..b * cd], cd);
        bo[b - 1] = same_rows(&bb, &ro_[..b * vh * hd], vh * hd);
        by[b - 1] = same_rows(&cc, &ry[..b * c.ssm_inner], c.ssm_inner);
        // commit n = b kept rows: conv_shift of the history + the recurrence replayed with ro = 0 (no output)
        unsafe {
            t.s.launch(&shift, g256(cd), 256, 0, (cs.ptr(), dqkv.ptr(), cd as u32, b as u32, cd as u32, (c.ssm_conv - 1) as u32))?;
            t.s.launch(&rec, (vh as u32, 4), 128, 0, (DevPtr::NULL, convo.ptr(), dab.ptr(), (2 * vh) as u32, dtb.ptr(), sa.ptr(),
                ss.ptr(), b as u32, kh as u32, vh as u32, c.eps, 0u32))?;
        }
        t.s.sync()?;
        let (wc, ws) = &states[b];
        bst[b - 1] = (!bits_eq(&cs.to_vec()?.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), &wc.iter().map(|v| v.to_bits()).collect::<Vec<_>>())) as usize
            + (!bits_eq(&ss.to_vec()?.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), &ws.iter().map(|v| v.to_bits()).collect::<Vec<_>>())) as usize;
    }
    t.report("GDN conv (verify ro=1 vs 1-row steps)", &bc);
    t.report("GDN recurrence o (verify vs 1-row steps)", &bo);
    t.report("GDN gated norm y (verify vs 1-row steps)", &by);
    t.report("GDN commit state (conv + ssm) vs n 1-row steps [0-2]", &bst);
    Ok(())
}

/// Attention with QSA (positions past the dense range): indexer key store / pool / score / top-k, attn_prep, attn of a
/// k-row verify vs k one-row steps (caches copied from the same start).
fn attention(t: &mut T, c: &Config, rng: &mut Lcg, p0: usize) -> Result<()> {
    let (kstore, pool, score, topk, prep, attn) = (t.f("fx_idx_kstore")?, t.f("fx_idx_pool")?, t.f("fx_idx_score")?, t.f("fx_topk")?,
        t.f("fx_attn_prep")?, t.f("fx_attn")?);
    let ctx = 4096usize;
    let (nq, nkv, hdim, dd, kp) = (c.n_head, c.n_head_kv, c.head_dim, c.idx_dim, c.kpool);
    let qd = nq * hdim;
    let kvd = nkv * hdim;
    let rn = 8192usize;
    let top = (c.idx_top_k / kp) as u32;
    let ldsc = (ctx / kp + 1) as u32;
    let ldsel = (c.idx_top_k + kp - 1) as u32;
    // caches (cells < p0 filled)
    let kq0 = DeviceBuffer::from_slice(&rng.i8v(ctx * kvd))?;
    let vq0 = DeviceBuffer::from_slice(&rng.i8v(ctx * kvd))?;
    let ks0 = DeviceBuffer::from_slice(&rng.v(ctx * kvd / 32, 0.02).iter().map(|v| v.abs() + 0.001).collect::<Vec<_>>())?;
    let vs0 = DeviceBuffer::from_slice(&rng.v(ctx * kvd / 32, 0.02).iter().map(|v| v.abs() + 0.001).collect::<Vec<_>>())?;
    let ring0 = DeviceBuffer::from_slice(&rng.v(rn * dd, 1.0))?;
    let pooled0 = DeviceBuffer::from_slice(&rng.v((ctx / kp + 1) * dd, 1.0))?;
    // per-row inputs
    let qf = DeviceBuffer::from_slice(&rng.v(bmax() * 2 * qd, 2.0))?;
    let kx = DeviceBuffer::from_slice(&rng.v(bmax() * kvd, 2.0))?;
    let vx = DeviceBuffer::from_slice(&rng.v(bmax() * kvd, 2.0))?;
    let ikx = DeviceBuffer::from_slice(&rng.v(bmax() * dd, 1.0))?;
    let iqv = DeviceBuffer::from_slice(&rng.v(bmax() * c.idx_heads * dd, 1.0))?;
    let (qn, kn, ikn) = (DeviceBuffer::from_slice(&rng.v(hdim, 1.0))?, DeviceBuffer::from_slice(&rng.v(hdim, 1.0))?,
        DeviceBuffer::from_slice(&rng.v(dd, 1.0))?);
    let dprm = DeviceBuffer::<u32>::zeroed(P_WORDS)?;
    let sc = DeviceBuffer::<f32>::zeroed(bmax() * ldsc as usize)?;
    let sel = DeviceBuffer::<u32>::zeroed(bmax() * ldsel as usize)?;
    let seln = DeviceBuffer::<u32>::zeroed(bmax())?;
    let q = DeviceBuffer::<f32>::zeroed(bmax() * qd)?;
    let att = DeviceBuffer::<f32>::zeroed(bmax() * qd)?;
    struct St {
        kq: DeviceBuffer<i8>,
        vq: DeviceBuffer<i8>,
        ks: DeviceBuffer<f32>,
        vs: DeviceBuffer<f32>,
        ring: DeviceBuffer<f32>,
        pooled: DeviceBuffer<f32>,
    }
    let fresh = || -> Result<St> {
        Ok(St { kq: dup(&kq0)?, vq: dup(&vq0)?, ks: dup(&ks0)?, vs: dup(&vs0)?, ring: dup(&ring0)?, pooled: dup(&pooled0)? })
    };
    // rows r0 .. r0 + rows at positions p0 + r0 ..: returns (att rows, sorted selections per row)
    let step = |st: &St, rows: usize, r0: usize| -> Result<(Vec<f32>, Vec<Vec<u32>>, Vec<Vec<u32>>)> {
        fe_cuda::memcpy_htod(dprm.ptr(), &prm(c, p0 + r0, rows))?;
        let pp = dprm.ptr();
        unsafe {
            t.s.launch(&kstore, rows as u32, dd as u32, 0, (st.ring.ptr(), ikx.ptr().add::<f32>(r0 * dd), pp, rn as u32))?;
            t.s.launch(&pool, (rows / kp + 1) as u32, dd as u32, 0, (st.pooled.ptr(), st.ring.ptr(), ikn.ptr(), pp, kp as u32,
                c.n_rot as u32, c.eps, rn as u32, pp.add::<u32>(P_PCS), pp.add::<u32>(P_PSN)))?;
            t.s.launch(&score, (rows as u32, 72), 256, 0, (sc.ptr(), iqv.ptr().add::<f32>(r0 * c.idx_heads * dd), st.pooled.ptr(),
                c.idx_heads as u32, pp, kp as u32, ldsc, top, 1.0 / (dd as f32).sqrt()))?;
            t.s.launch(&topk, rows as u32, 1024, 0, (sel.ptr(), seln.ptr(), sc.ptr(), pp, kp as u32, top, ldsc, ldsel))?;
            t.s.launch(&prep, (rows as u32, (nq + nkv) as u32), 256, 0, (q.ptr(), st.kq.ptr(), st.ks.ptr(), st.vq.ptr(), st.vs.ptr(),
                qf.ptr().add::<f32>(r0 * 2 * qd), (2 * qd) as u32, (2 * hdim) as u32, kx.ptr().add::<f32>(r0 * kvd),
                vx.ptr().add::<f32>(r0 * kvd), kvd as u32, qn.ptr(), kn.ptr(), pp, nq as u32, nkv as u32, c.n_rot as u32, c.eps,
                pp.add::<u32>(P_CS), pp.add::<u32>(P_SN)))?;
            t.s.launch(&attn, (rows as u32, nq as u32), 256, 0, (att.ptr(), q.ptr(), st.kq.ptr(), st.ks.ptr(), st.vq.ptr(), st.vs.ptr(),
                qf.ptr().add::<f32>(r0 * 2 * qd + hdim), (2 * qd) as u32, (2 * hdim) as u32, nq as u32, nkv as u32, pp,
                1.0 / (hdim as f32).sqrt(), sel.ptr(), seln.ptr(), ldsel))?;
        }
        t.s.sync()?;
        let (sv, nv) = (sel.to_vec()?, seln.to_vec()?);
        let mut sets = vec![];
        let mut raw = vec![];
        for r in 0..rows {
            let row = sv[r * ldsel as usize..r * ldsel as usize + nv[r] as usize].to_vec();
            let mut srt = row.clone();
            srt.sort();
            sets.push(srt);
            raw.push(row);
        }
        Ok((att.to_vec()?[..rows * qd].to_vec(), sets, raw))
    };
    let st1 = fresh()?;
    let (mut ra, mut rsets, mut rraw) = (vec![], vec![], vec![]);
    for r in 0..bmax() {
        let (a, s1, r1) = step(&st1, 1, r)?;
        ra.extend(a);
        rsets.extend(s1);
        rraw.extend(r1);
    }
    let (mut ba, mut bset, mut bord) = (vec![0; bmax()], vec![0; bmax()], vec![0; bmax()]);
    for b in 1..=bmax() {
        let st = fresh()?;
        let (a, sets, raw) = step(&st, b, 0)?;
        ba[b - 1] = same_rows(&a, &ra[..b * qd], qd);
        bset[b - 1] = (0..b).filter(|&r| sets[r] != rsets[r]).count();
        bord[b - 1] = (0..b).filter(|&r| raw[r] != rraw[r]).count();
    }
    // run-to-run: the same 1-row step twice from the same caches
    let (sa1, sb1) = (fresh()?, fresh()?);
    let (_, _, x1) = step(&sa1, 1, 0)?;
    let (_, _, x2) = step(&sb1, 1, 0)?;
    let tag = if (p0 + 1) / kp > top as usize { "QSA" } else { "dense" };
    t.report(&format!("{tag} pos {p0}: selected set (verify vs 1-row steps)"), &bset);
    t.report(&format!("{tag} pos {p0}: selection order (sel as written)"), &bord);
    t.report(&format!("{tag} pos {p0}: attention out (verify vs 1-row)"), &ba);
    println!("{:52} {}", format!("  {tag} pos {p0}: selection order, same step twice"), if x1 == x2 { "same" } else { "DIFFERS run to run" });
    Ok(())
}

/// fx_route8 (router top-k) and the two-stage argmax, per row.
fn router_argmax(t: &mut T, c: &Config, rng: &mut Lcg) -> Result<()> {
    let r8 = t.f("fx_route8")?;
    let ne = c.n_expert;
    let k = c.n_expert_used;
    let lg = DeviceBuffer::from_slice(&rng.v(bmax() * ne, 4.0))?;
    let ids = DeviceBuffer::<i32>::zeroed(bmax() * k)?;
    let wts = DeviceBuffer::<f32>::zeroed(bmax() * k)?;
    let run = |rows: usize, r0: usize| -> Result<(Vec<i32>, Vec<f32>)> {
        unsafe { t.s.launch(&r8, rows as u32, 256, 0, (ids.ptr(), wts.ptr(), lg.ptr().add::<f32>(r0 * ne), ne as u32, ne as u32, k as u32,
            c.expert_weights_scale))? };
        t.s.sync()?;
        Ok((ids.to_vec()?[..rows * k].to_vec(), wts.to_vec()?[..rows * k].to_vec()))
    };
    let (mut oi, mut ow) = (vec![], vec![]);
    for r in 0..bmax() {
        let (a, b) = run(1, r)?;
        oi.extend(a);
        ow.extend(b);
    }
    let mut bad = vec![0; bmax()];
    for b in 1..=bmax() {
        let (i, w) = run(b, 0)?;
        bad[b - 1] = (0..b).filter(|&r| i[r * k..(r + 1) * k] != oi[r * k..(r + 1) * k]
            || w[r * k..(r + 1) * k].iter().zip(&ow[r * k..(r + 1) * k]).any(|(x, y)| x.to_bits() != y.to_bits())).count();
    }
    t.report("router fx_route8 (ids + weights)", &bad);
    let (part, fin) = (t.f("fx_argmax_part")?, t.f("fx_argmax_fin")?);
    let nv = c.n_vocab;
    let pp = 64u32;
    let x = DeviceBuffer::from_slice(&rng.v(bmax() * nv, 8.0))?;
    let pt = DeviceBuffer::<f32>::zeroed(bmax() * pp as usize * 3)?;
    let out = DeviceBuffer::<u32>::zeroed(bmax())?;
    let pmax = DeviceBuffer::<f32>::zeroed(bmax())?;
    let am = |rows: usize, r0: usize| -> Result<(Vec<u32>, Vec<f32>)> {
        unsafe {
            t.s.launch(&part, (pp, rows as u32), 256, 0, (pt.ptr(), x.ptr().add::<f32>(r0 * nv), nv as u32))?;
            t.s.launch(&fin, rows as u32, 32, 0, (out.ptr(), pt.ptr(), pp, pmax.ptr()))?;
        }
        t.s.sync()?;
        Ok((out.to_vec()?[..rows].to_vec(), pmax.to_vec()?[..rows].to_vec()))
    };
    let (mut o1, mut p1) = (vec![], vec![]);
    for r in 0..bmax() {
        let (a, b) = am(1, r)?;
        o1.extend(a);
        p1.extend(b);
    }
    let mut bad = vec![0; bmax()];
    for b in 1..=bmax() {
        let (o, p) = am(b, 0)?;
        bad[b - 1] = (0..b).filter(|&r| o[r] != o1[r] || p[r].to_bits() != p1[r].to_bits()).count();
    }
    t.report("argmax (two-stage, index + pmax)", &bad);
    Ok(())
}

/// Compare `run(rows, r0)` (the outputs of rows r0 .. r0 + rows, `n` values per row) at B = 1..bmax() against B = 1 runs.
fn per_row(t: &mut T, name: &str, n: usize, run: &dyn Fn(usize, usize) -> Result<Vec<f32>>) -> Result<()> {
    let mut one = vec![];
    for r in 0..bmax() {
        one.extend(run(1, r)?);
    }
    let mut bad = vec![0; bmax()];
    for b in 1..=bmax() {
        bad[b - 1] = same_rows(&run(b, 0)?, &one[..b * n], n);
    }
    t.report(name, &bad);
    Ok(())
}

/// fx_gemv_h tpr choice of fe-fnext (fast/mod.rs tpr_for).
fn tpr_for(k: usize, n: usize) -> u32 {
    let chunks = k / 8;
    let mut t = 8usize;
    while t < 256 && chunks > 5 * t {
        t *= 2;
    }
    while t < 256 && n.div_ceil(256 / t) < 72 {
        t *= 2;
    }
    t as u32
}

/// The kernels the MTP head (enqueue_mtp) runs besides the GEMVs / attention / router / argmax covered elsewhere:
/// fx_gemv_h (bf16 / f16 GEMV with the engine's tpr), fx_mtp_cat, fn_hc_mix, fx_scale_ld, fx_silu_ld, fn_swiglu,
/// fx_hc_combine, fx_moe_fin.
fn mtp_parts(t: &mut T, c: &Config, rng: &mut Lcg) -> Result<()> {
    let s = t.s;
    let (d, hc) = (c.n_embd, c.hc);
    let hd = hc * d;
    // fx_gemv_h on the engine's shapes (bf16 and f16 weights, strided rows)
    let gh = t.f("fx_gemv_h")?;
    for (nm, k, n, ldx, ldy, bf) in [("indexer k [2560 -> 128]", d, c.idx_dim, d, c.idx_dim, 1u32),
        ("indexer q [2560 -> 512]", d, c.idx_dim * c.idx_heads, d, c.idx_dim * c.idx_heads, 1),
        ("router gate [2560 -> 257] ld 257", d, c.n_expert + 1, d, c.n_expert + 1, 1),
        ("hc down+inject [10240 -> 324] ld 384", hd, c.hc_lr + hc, hd, 384, 1), ("f16 [2560 -> 512]", d, 512, d, 512, 0)] {
        let w: Vec<u16> = rng.v(n * k, 0.05).iter().map(|v| if bf == 1 { (v.to_bits() >> 16) as u16 } else { fe_ggq::refq::f32_to_f16(*v) }).collect();
        let (dw, dx) = (DeviceBuffer::from_slice(&w)?, DeviceBuffer::from_slice(&rng.v(bmax() * ldx, 1.0))?);
        let dy = DeviceBuffer::<f32>::zeroed(bmax() * ldy)?;
        let tpr = tpr_for(k, n);
        per_row(t, &format!("fx_gemv_h {nm} (tpr {tpr})"), ldy, &|rows, r0| {
            let mut c0 = 0;
            while c0 < rows {
                let cr = (rows - c0).min(8);
                unsafe { s.launch(&gh, n.div_ceil(256 / tpr as usize) as u32, 256, 0, (dy.ptr().add::<f32>(c0 * ldy),
                    dx.ptr().add::<f32>((r0 + c0) * ldx), dw.ptr(), k as u32, n as u32, cr as u32, 1.0f32, bf, 0u32, ldx as u32,
                    ldy as u32, tpr))? };
                c0 += cr;
            }
            s.sync()?;
            let mut v = dy.to_vec()?[..rows * ldy].to_vec();
            for r in 0..rows {
                v[r * ldy + n..(r + 1) * ldy].fill(0.0);
            }
            Ok(v)
        })?;
    }
    // fx_mtp_cat
    let cat = t.f("fx_mtp_cat")?;
    let (en, hn) = (DeviceBuffer::from_slice(&rng.v(bmax() * d, 1.0))?, DeviceBuffer::from_slice(&rng.v(bmax() * hd, 1.0))?);
    let dc = DeviceBuffer::<f32>::zeroed(bmax() * hc * 2 * d)?;
    per_row(t, "fx_mtp_cat", hc * 2 * d, &|rows, r0| {
        unsafe { s.launch(&cat, (g256(2 * d), (rows * hc) as u32), 256, 0, (dc.ptr(), en.ptr().add::<f32>(r0 * d), hn.ptr().add::<f32>(r0 * hd),
            d as u32, hc as u32))? };
        s.sync()?;
        Ok(dc.to_vec()?[..rows * hc * 2 * d].to_vec())
    })?;
    // fn_hc_mix
    let mix = t.f("fn_hc_mix")?;
    let (xn, gate) = (DeviceBuffer::from_slice(&rng.v(bmax() * hd, 1.0))?, DeviceBuffer::from_slice(&rng.v(bmax() * hd, 2.0))?);
    let dm = DeviceBuffer::<f32>::zeroed(bmax() * d)?;
    per_row(t, "fn_hc_mix", d, &|rows, r0| {
        let n = rows * d;
        unsafe { s.launch(&mix, g256(n), 256, 0, (dm.ptr(), xn.ptr().add::<f32>(r0 * hd), gate.ptr().add::<f32>(r0 * hd), d as u32, hc as u32,
            n as u32))? };
        s.sync()?;
        Ok(dm.to_vec()?[..n].to_vec())
    })?;
    // in-place row kernels: fx_scale_ld, fx_silu_ld on [rows][ld], fn_swiglu
    let (scl, sil, swi) = (t.f("fx_scale_ld")?, t.f("fx_silu_ld")?, t.f("fn_swiglu")?);
    let ld = c.hc_lr + hc;
    let x0 = rng.v(bmax() * ld, 3.0);
    let dxx = DeviceBuffer::<f32>::zeroed(bmax() * ld)?;
    per_row(t, "fx_scale_ld + fx_silu_ld (hc low rank)", ld, &|rows, r0| {
        fe_cuda::memcpy_htod(dxx.ptr(), &x0[r0 * ld..(r0 + rows) * ld])?;
        unsafe {
            s.launch(&scl, (g256(ld), rows as u32), 256, 0, (dxx.ptr(), ld as u32, ld as u32, 1.0 / hc as f32))?;
            s.launch(&sil, (g256(c.hc_lr), rows as u32), 256, 0, (dxx.ptr(), c.hc_lr as u32, ld as u32))?;
        }
        s.sync()?;
        Ok(dxx.to_vec()?[..rows * ld].to_vec())
    })?;
    let nf = c.n_ff_shexp;
    let (sg, su) = (DeviceBuffer::from_slice(&rng.v(bmax() * nf, 3.0))?, DeviceBuffer::from_slice(&rng.v(bmax() * nf, 1.0))?);
    let so = DeviceBuffer::<f32>::zeroed(bmax() * nf)?;
    per_row(t, "fn_swiglu (shared expert)", nf, &|rows, r0| {
        unsafe { s.launch(&swi, g256(rows * nf), 256, 0, (so.ptr(), sg.ptr().add::<f32>(r0 * nf), su.ptr().add::<f32>(r0 * nf), (rows * nf) as u32))? };
        s.sync()?;
        Ok(so.to_vec()?[..rows * nf].to_vec())
    })?;
    // fx_hc_combine: res += out * 2 sigmoid(inj)
    let hcc = t.f("fx_hc_combine")?;
    let res0 = rng.v(bmax() * hd, 1.0);
    let (outv, inj) = (DeviceBuffer::from_slice(&rng.v(bmax() * d, 1.0))?, DeviceBuffer::from_slice(&rng.v(bmax() * ld, 2.0))?);
    let dres = DeviceBuffer::<f32>::zeroed(bmax() * hd)?;
    per_row(t, "fx_hc_combine", hd, &|rows, r0| {
        fe_cuda::memcpy_htod(dres.ptr(), &res0[r0 * hd..(r0 + rows) * hd])?;
        let n = rows * hd;
        unsafe { s.launch(&hcc, g256(n), 256, 0, (dres.ptr(), outv.ptr().add::<f32>(r0 * d), inj.ptr().add::<f32>(r0 * ld + c.hc_lr), d as u32,
            hc as u32, ld as u32, n as u32))? };
        s.sync()?;
        Ok(dres.to_vec()?[..n].to_vec())
    })?;
    // fx_moe_fin: experts on the card or on the host, + shared expert * sigmoid(gate), into the hc streams
    let fin = t.f("fx_moe_fin")?;
    let topk = c.n_expert_used;
    let ne = c.n_expert;
    let ids: Vec<i32> = (0..bmax() * topk).map(|p| ((p * 37 + p / topk * 11) % ne) as i32).collect();
    let tab: Vec<u64> = (0..ne).map(|e| if e % 5 == 2 { 0 } else { 1 }).collect();
    let (dids, dtab) = (DeviceBuffer::from_slice(&ids)?, DeviceBuffer::from_slice(&tab)?);
    let (ey, host) = (DeviceBuffer::from_slice(&rng.v(bmax() * topk * d, 1.0))?, DeviceBuffer::from_slice(&rng.v(bmax() * topk * d, 1.0))?);
    let wts = DeviceBuffer::from_slice(&rng.v(bmax() * topk, 0.3))?;
    let sh = DeviceBuffer::from_slice(&rng.v(bmax() * d, 1.0))?;
    let ldg = ne + 1;
    let glog = DeviceBuffer::from_slice(&rng.v(bmax() * ldg, 2.0))?;
    per_row(t, "fx_moe_fin (card + host rows, shared expert)", hd, &|rows, r0| {
        fe_cuda::memcpy_htod(dres.ptr(), &res0[r0 * hd..(r0 + rows) * hd])?;
        unsafe { s.launch(&fin, (g256(d), rows as u32), 256, 0, (dres.ptr(), ey.ptr().add::<f32>(r0 * topk * d), host.ptr().add::<f32>(r0 * topk * d),
            dids.ptr().add::<i32>(r0 * topk), dtab.ptr(), wts.ptr().add::<f32>(r0 * topk), sh.ptr().add::<f32>(r0 * d),
            glog.ptr().add::<f32>(r0 * ldg + ne), ldg as u32, inj.ptr().add::<f32>(r0 * ld + c.hc_lr), ld as u32, d as u32, hc as u32,
            topk as u32, DevPtr::NULL))? };
        s.sync()?;
        Ok(dres.to_vec()?[..rows * hd].to_vec())
    })?;
    println!("  (the MTP head's q8 GEMVs, gemv_bf16, moe_gu_q8 / moe_down: fe-ggq `ggq-check binv`; attention, router, argmax,");
    println!("   rmsnorm: above)");
    Ok(())
}
