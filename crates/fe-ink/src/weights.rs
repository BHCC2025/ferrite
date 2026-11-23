//! Weights on the device, as shipped: experts NVFP4 (layers 0-19) / MXFP8 (20-23), everything else bf16 / f32.
use crate::ckpt::Ckpt;
use crate::*;
use anyhow::Result;
use fe_cuda::{DeviceBuffer, Stream};

pub type Buf = DeviceBuffer<u8>;

/// A dense bf16-activation linear: bf16 weights, or FP8 (E4M3, one f32 scale per output row; effective weight
/// bf16(q * scale)).
pub struct Lin {
    pub n: usize,
    pub k: usize,
    pub bf: Option<Buf>,
    pub f8: Option<fe_inkk::Fp8Linear>,
}

fn lin_cat(ck: &mut Ckpt, kk: &fe_inkk::Kern, s: &Stream, names: &[String], fp8: bool) -> Result<Lin> {
    let k = ck.info(&names[0])?.shape[1];
    let n: usize = names.iter().map(|x| ck.info(x).map(|i| i.shape[0])).sum::<Result<usize>>()?;
    let b = Buf::alloc(n * k * 2)?;
    let mut off = 0;
    for nm in names {
        off += ck.upload_to(nm, b.ptr().byte_add(off))?;
    }
    if fp8 {
        let f = kk.fp8_from_bf16(s, b.ptr(), n, k)?;
        s.sync()?;
        Ok(Lin { n, k, bf: None, f8: Some(f) })
    } else {
        Ok(Lin { n, k, bf: Some(b), f8: None })
    }
}

fn lin(ck: &mut Ckpt, kk: &fe_inkk::Kern, s: &Stream, name: &str, fp8: bool, rows: Option<usize>) -> Result<Lin> {
    let info = ck.info(name)?.clone();
    let (n, k) = (rows.unwrap_or(info.shape[0]), info.shape[1]);
    let b = ck.upload(name, Some(n * k * 2))?;
    if fp8 {
        let f = kk.fp8_from_bf16(s, b.ptr(), n, k)?;
        s.sync()?;
        Ok(Lin { n, k, bf: None, f8: Some(f) })
    } else {
        Ok(Lin { n, k, bf: Some(b), f8: None })
    }
}

pub enum Experts {
    /// NVFP4: one copy of the weights, in the repacked tensor-core layout (w13 rows permuted + tiled scales, w2 codes as stored +
    /// tiled scales); both the decode kernels (k-inkm rp.rs) and the grouped prompt kernels (k-ink) read it. `pf.w2` borrows `w2`.
    Fp4 { pf: fe_inkk::PrefillLayer, f13: Buf, f2: Buf, am13: Buf, am2: Buf },
    /// MXFP8: the release layout for the decode kernels plus the repacked copy for the grouped prompt kernels (0.7 GB a layer)
    Mx { pf: fe_inkk::PrefillLayer },
    /// G16 (INK_HOST_EXPERTS=N): NVFP4 in the repacked layout with some experts in pinned host memory behind the pointer tables
    /// (fe_inkk::slots); `keep` holds the amax buffers the tables' kernels read. All 24 layers (20-23 re-quantized from MXFP8).
    Slot { sl: std::sync::Mutex<fe_inkk::slots::SlotLayer>, keep: Vec<Buf> },
}

pub struct Gate {
    pub down: Buf,
    pub forget: Buf,
    pub gain: Buf,
}

pub struct Layer {
    pub full: bool,
    /// wq | gate_attn | wr stacked: [3072 + 3072 + 384][2304], one GEMV on the query-side input
    pub wqgr: Lin,
    pub wk: Lin,
    pub wv: Lin,
    pub wo: Lin,
    pub sink: Buf,
    pub q_scale: Buf,
    pub rel_scale: Buf,
    pub proj: Buf,
    /// sliding: [C][4] taps; full: static [4][C]
    pub kconv: Buf,
    pub vconv: Buf,
    /// full only: weight_proj [32][2304]
    pub kdyn: Option<Buf>,
    pub vdyn: Option<Buf>,
    pub attn_conv: Buf,
    pub mlp_conv: Buf,
    /// sliding: [input_layernorm]; full: [q, k, v]
    pub ln: Vec<Buf>,
    pub post_ln: Buf,
    pub attn_q: Option<Buf>,
    pub mlp_q: Buf,
    pub router: Buf,
    pub router_bias: Buf,
    pub experts: std::sync::Arc<Experts>,
    pub gates: Option<Vec<Gate>>,
}

pub struct Weights {
    pub layers: Vec<Layer>,
    pub embed_norm: Buf,
    pub final_norm: Buf,
    pub out_q: Buf,
    /// [12][4][2304] f32: query, key, value, mlp
    pub msg_q: Buf,
    pub unembed: Lin,
}

pub fn load(ck: &mut Ckpt, kk: &fe_inkk::Kern, st: &Stream, fp8: bool) -> Result<Weights> {
    let mut layers: Vec<Layer> = Vec::new();
    // INK_HOST_EXPERTS=N: the N coldest experts anywhere (heat share) live in pinned host memory, all layers NVFP4
    let host_n: usize = std::env::var("INK_HOST_EXPERTS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    // INK_SLOTS=1 with N = 0: slot layers with everything resident (pointer tables + one shared pool: the same bytes as the plain layout), so
    // the resident count can be moved later with `set_host_experts` / `fit_host_experts`
    let resident: Option<Vec<Vec<bool>>> = if host_n > 0 {
        Some(fe_inkk::slots::resident_global(&crate::heat_share()?, host_n))
    } else if crate::env_flag("INK_SLOTS") {
        Some(vec![vec![true; 128]; NL])
    } else {
        None
    };
    let slot_pool = fe_inkk::slots::SlotPool::new();
    for li in 0..NL {
        let p = format!("model.llm.layers.{li}.");
        let full = is_full(li);
        macro_rules! g { ($n:expr) => { ck.upload(&format!("{p}{}", $n), None) }; }
        macro_rules! l { ($n:expr) => { lin(ck, kk, st, &format!("{p}{}", $n), fp8, None) }; }
        let ln = if full {
            vec![g!("q_input_norm.weight")?, g!("k_input_norm.weight")?, g!("v_input_norm.weight")?]
        } else {
            vec![g!("attn_norm.weight")?]
        };
        let mut raw4 = if li < 20 {
            Some((g!("mlp.experts.w13_weight")?, g!("mlp.experts.w13_weight_scale")?, Buf::alloc(512)?,
                g!("mlp.experts.w2_weight")?, g!("mlp.experts.w2_weight_scale")?, Buf::alloc(512)?))
        } else {
            None
        };
        let mx = if li >= 20 {
            let (w13, s13) = (g!("mlp.experts.w13_weight")?, g!("mlp.experts.w13_weight_scale")?);
            let (w2, s2) = (g!("mlp.experts.w2_weight")?, g!("mlp.experts.w2_weight_scale")?);
            let rel = fe_inkk::ExpertLayer { fmt: fe_inkk::ExpertFmt::Mxfp8, w13: w13.ptr(), s13: s13.ptr(), f13: fe_cuda::DevPtr(0),
                w2: w2.ptr(), s2: s2.ptr(), f2: fe_cuda::DevPtr(0) };
            let g16 = crate::env_flag("INK_G16");
            if resident.is_some() || g16 {
                // G16: re-quantize to NVFP4 (study rule), repack, move the cold experts to pinned host memory (INK_G16=1 alone: all resident)
                let rq = kk.requant_mxfp8_to_nvfp4(st, &rel)?;
                st.sync()?;
                drop((w13, s13, s2, w2));
                let (am13, am2) = (Buf::alloc(512)?, Buf::alloc(512)?);
                fe_cuda::memcpy_dtod(am13.ptr(), rq.amax13.ptr(), 512)?;
                fe_cuda::memcpy_dtod(am2.ptr(), rq.amax2.ptr(), 512)?;
                let pf = kk.repack_nvfp4(st, &rq.layer(), am13.ptr(), am2.ptr(), fe_inkk::Requant::DENOMINATOR, fe_inkk::Requant::DENOMINATOR)?;
                st.sync()?;
                drop(rq);
                if let Some(res) = &resident {
                    let sl = kk.make_slot_layer_in(st, pf, &res[li], &slot_pool)?;
                    Some(Experts::Slot { sl: std::sync::Mutex::new(sl), keep: vec![am13, am2] })
                } else {
                    Some(Experts::Fp4 { pf, f13: Buf::alloc(512)?, f2: Buf::alloc(512)?, am13, am2 })
                }
            } else {
                let pf = kk.repack_mxfp8(st, &rel)?;
                st.sync()?;
                drop((w13, s13, s2, w2));
                Some(Experts::Mx { pf })
            }
        } else {
            None
        };
        let gates = if (LOOP0..LOOP1).contains(&li) {
            let mut v = Vec::new();
            for c in ["attn", "k", "v", "mlp"] {
                v.push(Gate { down: g!(format!("controllers.{c}.down_weight"))?,
                    forget: g!(format!("controllers.{c}.forget_weight"))?,
                    gain: g!(format!("controllers.{c}.gain_weight"))? });
            }
            Some(v)
        } else {
            None
        };
        let (kdyn, vdyn) = if full {
            (Some(g!("attn.k_sconv.weight_proj")?), Some(g!("attn.v_sconv.weight_proj")?))
        } else {
            (None, None)
        };
        let (kconv, vconv) = if full { (g!("attn.k_sconv.static_w")?, g!("attn.v_sconv.static_w")?) }
            else { (g!("attn.k_sconv.weight")?, g!("attn.v_sconv.weight")?) };
        // per-expert factors amax / outer_scale_denominator (f32), what the reference decode multiplies in
        if let Some((_, _, f13, _, _, f2)) = &mut raw4 {
            for (dst, nm) in [(f13, "w13"), (f2, "w2")] {
                let am = ck.read_f32(&format!("{p}mlp.experts.{nm}_weight_amax"))?;
                let den = ck.read_f32(&format!("{p}mlp.experts.{nm}_weight_outer_scale_denominator"))?[0];
                let fac: Vec<f32> = am.iter().map(|a| a / den).collect();
                let b: Vec<u8> = fac.iter().flat_map(|x| x.to_le_bytes()).collect();
                fe_cuda::memcpy_htod(dst.ptr(), &b)?;
            }
        }
        // repack in place of the release layout: the release w13 / s13 / s2 are freed, so a layer holds its weights once
        let experts: std::sync::Arc<Experts> = if let Some((w13, s13, f13, w2, s2, f2)) = raw4 {
            let rel = fe_inkk::ExpertLayer { fmt: fe_inkk::ExpertFmt::Nvfp4, w13: w13.ptr(), s13: s13.ptr(), f13: f13.ptr(),
                w2: w2.ptr(), s2: s2.ptr(), f2: f2.ptr() };
            // raw per-expert amax (the native activation path scales with it) and the outer denominators
            let (mut am13, mut am2) = (Buf::alloc(512)?, Buf::alloc(512)?);
            let mut den = [0f32; 2];
            for (k, (dst, nm)) in [(&mut am13, "w13"), (&mut am2, "w2")].into_iter().enumerate() {
                let am = ck.read_f32(&format!("{p}mlp.experts.{nm}_weight_amax"))?;
                den[k] = ck.read_f32(&format!("{p}mlp.experts.{nm}_weight_outer_scale_denominator"))?[0];
                let b: Vec<u8> = am.iter().flat_map(|x| x.to_le_bytes()).collect();
                fe_cuda::memcpy_htod(dst.ptr(), &b)?;
            }
            let pf = kk.repack_nvfp4(st, &rel, am13.ptr(), am2.ptr(), den[0], den[1])?;
            st.sync()?;
            drop((w13, s13, s2, w2));
            if let Some(res) = &resident {
                let sl = kk.make_slot_layer_in(st, pf, &res[li], &slot_pool)?;
                std::sync::Arc::new(Experts::Slot { sl: std::sync::Mutex::new(sl), keep: vec![f13, f2, am13, am2] })
            } else {
                std::sync::Arc::new(Experts::Fp4 { pf, f13, f2, am13, am2 })
            }
        } else {
            std::sync::Arc::new(mx.unwrap())
        };
        let attn_q = if li > 0 {
            let name = format!("model.llm.attn_res.attn_queries.{}", li - 1);
            let v = ck.read_f32(&name)?;
            Some(ck.upload_f32(&v)?)
        } else {
            None
        };
        let mq = ck.read_f32(&format!("model.llm.attn_res.mlp_queries.{li}"))?;
        let mlp_q = ck.upload_f32(&mq)?;
        let rb = ck.read_f32(&format!("{p}mlp.gate.bias"))?;
        let router_bias = ck.upload_f32(&rb)?;
        layers.push(Layer {
            full,
            wqgr: lin_cat(ck, kk, st, &["attn.wq_du.weight", "attn.gate_attn.weight", "attn.wr_du.weight"].map(|n| format!("{p}{n}")), fp8)?,
            wk: l!("attn.wk_dv.weight")?, wv: l!("attn.wv_dv.weight")?, wo: l!("attn.wo_ud.weight")?,
            sink: g!("attn.attn_sink_proj.weight")?, q_scale: g!("attn.q_l2_clip.scale")?,
            rel_scale: g!("attn.rel_l2_clip.scale")?, proj: g!("attn.rel_logits_proj.proj")?,
            kconv, vconv, kdyn, vdyn,
            attn_conv: g!("attn_sconv.weight")?, mlp_conv: g!("mlp_sconv.weight")?,
            ln, post_ln: g!("mlp_norm.weight")?, attn_q, mlp_q,
            router: g!("mlp.gate.weight")?, router_bias, experts, gates,
        });
        eprintln!("ink: layer {li} loaded");
    }
    if host_n > 0 {
        // the shared spare slots of the pipelined adapt: a few experts for the whole model
        let _ = slot_pool.reserve_free(slot_pool.reserve());
    }
    // message queries: [q, mlp] and [k, v] of each loop layer -> [layer][q, k, v, mlp]
    let qs = ck.read_f32("model.llm.middle_cycle_cross_pass_queries")?;
    let kvs = ck.read_f32("model.llm.middle_cycle_cross_pass_kv_queries")?;
    let mut merged = vec![0f32; 12 * 4 * D];
    for l in 0..12 {
        let put = |slot: usize, src: &[f32], row: usize, merged: &mut Vec<f32>| {
            merged[(l * 4 + slot) * D..(l * 4 + slot + 1) * D].copy_from_slice(&src[row * D..(row + 1) * D]);
        };
        put(0, &qs, 2 * l, &mut merged);
        put(1, &kvs, 2 * l, &mut merged);
        put(2, &kvs, 2 * l + 1, &mut merged);
        put(3, &qs, 2 * l + 1, &mut merged);
    }
    let msg_q = ck.upload_f32(&merged)?;
    let oq = ck.read_f32("model.llm.attn_res.output_query")?;
    let out_q = ck.upload_f32(&oq)?;
    Ok(Weights {
        layers,
        embed_norm: ck.upload("model.llm.embed_norm.weight", None)?,
        final_norm: ck.upload("model.llm.norm.weight", None)?,
        out_q,
        msg_q,
        unembed: lin(ck, kk, st, "model.llm.unembed.weight", fp8, Some(VOCAB))?,
    })
}
