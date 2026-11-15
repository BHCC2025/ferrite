//! Two-Spark tensor parallelism for the Qwen3.5 engine. Each rank owns half of the attention heads (12 q /
//! 2 kv), GDN heads (8 key / 24 value, with their conv channels, A_log, dt_bias, a/b rows) and MLP columns
//! (`fe_exl3::tp::qwen35_split`); the residual stream is replicated. The three row-parallel linears of a
//! layer (o_proj, linear_attn.out_proj, mlp.down_proj) end in a fabric reduction that also adds the
//! residual, and lm_head is vocabulary-parallel with a logits gather. Both ranks run the same forward calls
//! in lockstep (same driver code, same inputs); every rank ends with identical residuals and logits.
use anyhow::{ensure, Result};
use fe_cuda::DeviceBuffer;

use crate::safetensors::SafeTensors;

pub struct Tp {
    pub rank: usize,
    pub world: usize,
    pub fab: fe_fabric::Fabric,
    /// rank 0 with a following rank 1: state-changing engine calls are sent over the link (crate::follow)
    pub mirror: bool,
    /// f32 partials of a row-parallel GEMV (decode rows) [X3_ROWS][hidden]
    pub part: DeviceBuffer<f32>,
    /// decode / verify rows exchange bf16 partials (half the bytes, summed in f32 and rounded once; KL TP2 code
    /// 0.0218 vs 0.02187 f32, text 0.04645 vs 0.04633); FE_TP_PART=f32 keeps f32 partials (both ranks must
    /// agree; checked at startup)
    pub part_bf16_dec: bool,
    /// bf16 partials of a row-parallel GEMM (prompt rows) [max_rows][hidden]
    pub part_bf16: DeviceBuffer<u16>,
    /// this rank's vocabulary half of the logits [max_logit_rows][vocab / world]
    pub logits_half: DeviceBuffer<u16>,
    /// all-gather landing zone [world][max_logit_rows][vocab / world]
    pub gather: DeviceBuffer<u16>,
}

impl Tp {
    /// The TP state over a connected fabric; brings up the RoCE path (FE_ROCE=0: NCCL only).
    #[allow(clippy::too_many_arguments)]
    pub fn with_fabric(mut fab: fe_fabric::Fabric, rank: usize, world: usize, hidden: usize, max_rows: usize,
        max_logit_rows: usize, vocab: usize, mirror: bool) -> Result<Tp> {
        ensure!(world == 2, "TP world {world}: only two ranks are supported");
        // the slot holds X3_PASS (64) rows of f32 partials (1.31 MB): the target's decode / verify rows (bf16 or f32
        // partials) and the TP drafter's f32 partials (64 rows at 4 streams) all go over RoCE. Above it they would fall
        // back to NCCL: slower (~330 us for 64 f32 rows), and for bf16 partials a different rounding.
        let roce = fab.enable_roce(crate::model::X3_PASS * hidden * 4)?;
        let part_bf16_dec = std::env::var("FE_TP_PART").map_or(true, |v| v != "f32");
        let theirs = fab.link.exchange(&[part_bf16_dec as u8])?;
        ensure!(theirs == [part_bf16_dec as u8], "TP: the ranks disagree on FE_TP_PART");
        eprintln!("ferrite TP{world}: rank {rank} (RoCE {roce}, NCCL {}{})", fab.nccl_version(),
            if mirror { ", rank 1 follows" } else { "" });
        let half = vocab / world;
        let t = Tp {
            rank,
            world,
            fab,
            mirror,
            part: DeviceBuffer::alloc(crate::model::X3_ROWS * hidden)?,
            part_bf16_dec,
            part_bf16: DeviceBuffer::zeroed(max_rows * hidden)?,
            logits_half: DeviceBuffer::zeroed(max_logit_rows * half)?,
            gather: DeviceBuffer::alloc(world * max_logit_rows * half)?,
        };
        t.warm_up(max_rows * hidden, max_logit_rows * half)?;
        Ok(t)
    }

    /// NCCL connects (and sizes its protocol buffers) lazily on the first collectives: run the prompt-row
    /// all-reduce at small / mid / full size and the logits all-gather once here, or the first prompt pays it.
    fn warm_up(&self, reduce: usize, gather: usize) -> Result<()> {
        use fe_fabric::DataType;
        let s = fe_cuda::Stream::new()?;
        for n in [256, 64 << 10, reduce] {
            let n = n.min(reduce) & !1;
            self.fab.all_reduce_sum(&s, self.part_bf16.ptr(), self.part_bf16.ptr(), n, DataType::Bf16)?;
        }
        self.fab.all_gather(&s, self.logits_half.ptr(), self.gather.ptr(), gather, DataType::Bf16)?;
        s.sync()?;
        Ok(())
    }
}

impl Drop for Tp {
    fn drop(&mut self) {
        if self.mirror {
            let _ = self.fab.link.send(&crate::follow::Call::Stop.encode());
        }
    }
}

/// Build the engine the way every entry point does (CLI, scheduler / server): whole model, or one TP rank
/// when FE_TP_WORLD / FE_TP_RANK are set. `follower`: this rank only replays rank 0's calls
/// (`Engine::follow`); its sizes come from rank 0. `spec`: speculative depth if the checkpoint has MTP.
pub fn load_engine(model: &std::path::Path, n_slots: usize, max_ctx: usize, max_rows: usize, spec: usize, follower: bool)
    -> Result<(std::sync::Arc<fe_cuda::Context>, crate::model::Engine)> {
    use crate::config::Config;
    use crate::model::{Engine, Kernels, LoadOpts, Weights};
    let ctx = fe_cuda::Context::new(0)?;
    let full = Config::load(&Config::locate(model))?;
    let tp = from_env();
    let cfg = match tp {
        Some((_, world)) => full.shard(world),
        None => full,
    };
    let m = fe_cuda::Module::load(crate::K_CORE_CUBIN)?;
    let mut k = Kernels::load(&m)?;
    if tp.is_some() {
        k = k.with_tp2(&fe_cuda::Module::load(crate::K_CORE_TP2_CUBIN)?)?;
    }
    // TP: connect first, so a follower can take the leader's sizes
    let (mut sizes, mut fab) = ([n_slots, max_ctx, max_rows, spec], None);
    if let Some((rank, _)) = tp {
        let mut f = fe_fabric::Fabric::connect(rank as u32, &addr_from_env())?;
        // [follows?][hash of the command line]: ranks that both run their own driver must run the same one,
        // or they would part ways inside a collective and wait forever
        let mut hello = vec![follower as u8];
        hello.extend_from_slice(&argv_hash().to_le_bytes());
        let peer = f.link.exchange(&hello)?;
        ensure!(peer.len() == 9, "TP: bad hello from the peer");
        if !follower && peer[0] == 0 {
            ensure!(peer[1..] == hello[1..], "TP: the two ranks run different commands (lockstep needs identical argv)");
        }
        let mirror = rank == 0 && peer[0] == 1;
        if mirror {
            let b: Vec<u8> = sizes.iter().flat_map(|&v| (v as u32).to_le_bytes()).collect();
            f.link.send(&b)?;
        } else if follower {
            let b = f.link.recv()?;
            ensure!(b.len() == 16, "follower: bad sizes message");
            for (i, c) in b.chunks_exact(4).enumerate() {
                sizes[i] = u32::from_le_bytes(c.try_into()?) as usize;
            }
            eprintln!("ferrite TP: following rank 0 ({} slots x {} ctx, {} rows, spec {})", sizes[0], sizes[1], sizes[2], sizes[3]);
        }
        fab = Some((f, mirror));
    }
    let (free, total) = ctx.mem_info()?;
    crate::lowmem::init(total, ctx.compute_capability()?);
    eprintln!("ferrite: {} free {:.1} GB{}; loading {model:?}", ctx.name(), free as f64 / 1e9,
        if crate::lowmem::on() { " (small-memory forms on)" } else { "" });
    crate::lowmem::trace("the context and kernels");
    let t = std::time::Instant::now();
    let w = Weights::load(&cfg, model, &LoadOpts::from_env(), ctx.sm_count()? as u32)?;
    eprintln!("ferrite: weights loaded in {:.1}s", t.elapsed().as_secs_f32());
    // speculation needs a drafter: the MTP head, or an external one (FE_DRAFT=dflash; under TP it lives on rank 0,
    // whose verify forwards the follower replays). A follower takes rank 0's depth, it never drafts itself.
    let ext = std::env::var("FE_DRAFT").is_ok_and(|v| v == "dflash");
    let spec = if follower || w.mtp.is_some() || ext { sizes[3] } else { 0 };
    let mut e = Engine::with_spec(cfg, k, w, sizes[0], sizes[1], sizes[2], spec)?;
    if let (Some((rank, world)), Some((f, mirror))) = (tp, fab) {
        let (mr, ml) = e.tp_sizes();
        let t = Tp::with_fabric(f, rank, world, e.cfg.hidden, mr, ml, e.cfg.vocab, mirror)?;
        e.attach_tp(t)?;
    }
    let (free, _) = ctx.mem_info()?;
    eprintln!("ferrite: engine ready ({} slots x {} ctx), free {:.1} GB", sizes[0], sizes[1], free as f64 / 1e9);
    Ok((ctx, e))
}

/// FNV-1a of the command line (NUL-separated arguments).
fn argv_hash() -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for a in std::env::args() {
        for b in a.bytes().chain(std::iter::once(0)) {
            h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

/// (rank, world) from FE_TP_RANK / FE_TP_WORLD (unset or world 1: no tensor parallelism).
pub fn from_env() -> Option<(usize, usize)> {
    let world: usize = std::env::var("FE_TP_WORLD").ok()?.parse().ok()?;
    let rank: usize = std::env::var("FE_TP_RANK").ok()?.parse().ok()?;
    (world > 1).then_some((rank, world))
}

/// The fabric rendezvous: rank 0's fabric address and port (fe_site::tp_addr, FERRITE_TP_ADDR).
pub fn addr_from_env() -> String {
    fe_site::tp_addr()
}

/// Row ranges of a plain tensor kept by `rank`: GDN per-value-head tensors (A_log, dt_bias, in_proj_a/b)
/// split by value head, conv1d by the q / k / v channel sections; everything else whole.
pub fn plain_rows(name: &str, rows: usize, rank: usize, world: usize) -> Vec<(usize, usize)> {
    let part = |len: usize, off: usize| (off + rank * len / world, off + (rank + 1) * len / world);
    if name.ends_with("linear_attn.A_log")
        || name.ends_with("linear_attn.dt_bias")
        || name.ends_with("linear_attn.in_proj_a.weight")
        || name.ends_with("linear_attn.in_proj_b.weight")
    {
        vec![part(rows, 0)]
    } else if name.ends_with("linear_attn.conv1d.weight") {
        // channels [q 2048 | k 2048 | v 6144] (rows = 10240)
        let qk = (rows - rows * 3 / 5) / 2;
        vec![part(qk, 0), part(qk, qk), part(rows - 2 * qk, 2 * qk)]
    } else {
        vec![(0, rows)]
    }
}

/// A plain BF16 / F16 / F32 tensor's row ranges (row = first dim) as bf16 bits, concatenated.
pub fn rows_bf16(st: &SafeTensors, name: &str, ranges: &[(usize, usize)]) -> Result<Vec<u16>> {
    let t = st.get(name)?;
    let rows = t.shape[0];
    let per: usize = t.shape[1..].iter().product::<usize>().max(1);
    let dtype = t.dtype.clone();
    let raw = st.read_host(name)?;
    let esz = raw.len() / (rows * per);
    let mut out = Vec::new();
    for &(a, b) in ranges {
        ensure!(b <= rows, "{name}: rows {a}..{b} of {rows}");
        for i in a * per..b * per {
            let e = &raw[i * esz..(i + 1) * esz];
            out.push(match dtype.as_str() {
                "BF16" => u16::from_le_bytes([e[0], e[1]]),
                "F16" => crate::model::f16_bits_to_bf16(u16::from_le_bytes([e[0], e[1]])),
                "F32" => crate::model::f32_to_bf16_bits(f32::from_le_bytes([e[0], e[1], e[2], e[3]])),
                d => anyhow::bail!("{name}: dtype {d}"),
            });
        }
    }
    Ok(out)
}

/// This rank's [`fe_fp4::Slice`] of a ModelOpt NVFP4 / FP8 prompt weight (`base` without `.weight`): the same
/// heads / columns / input rows as its EXL3 linear (`fe_exl3::tp::qwen35_split`).
pub fn fp4_slice(ck: &fe_fp4::Checkpoint, base: &str, rank: usize, world: usize) -> Result<fe_fp4::Slice> {
    use fe_exl3::tp::Split;
    let t = ck.get(&format!("{base}.weight"))?;
    let n = t.shape[0];
    let k = if ck.algo(base).as_deref() == Some("NVFP4") { t.shape[1] * 2 } else { t.shape[1] };
    Ok(match fe_exl3::tp::qwen35_split(base, k, n, rank, world) {
        Split::Cols(rows) => fe_fp4::Slice { rows, k: (0, k) },
        Split::Rows(a, b) => fe_fp4::Slice { rows: vec![(0, n)], k: (a, b) },
        Split::Replicate => fe_fp4::Slice { rows: vec![(0, n)], k: (0, k) },
    })
}
