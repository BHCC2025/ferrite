//! fe-fabric: the two-Spark TP2 collectives. A TCP bootstrap between the ranks (rank 0 listens), NCCL as the
//! baseline (dlopen'd), and the RoCE one-shot gather-sum (Rust kernel + ibverbs proxy) for decode-sized
//! reductions.
//!
//! ```ignore
//! let mut fab = fe_fabric::Fabric::connect(rank, "10.0.0.1:29511")?;     // both ranks, rank 0's fabric address
//! fab.enable_roce(1 << 20)?;                                             // both ranks (FE_ROCE=0: NCCL only)
//! fab.all_reduce_sum(&stream, buf.ptr(), buf.ptr(), n, DataType::F32)?;  // NCCL, in place
//! fab.reduce_to_bf16(&stream, partial_f32, hidden_bf16, Some(hidden_bf16), n)?;  // hidden += sum of partials
//! ```

pub mod boot;
pub mod nccl;
pub mod roce;

use anyhow::{bail, Result};
use fe_cuda::{DevPtr, Function, Module, Stream};

pub use nccl::DataType;

pub static K_FABRIC_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/k_fabric.cubin"));

/// Exchange grid by size (the kernel counts arrivals per grid-size class): small messages are latency-bound
/// (fewer blocks fence and poll less), large ones staging-bound. Measured on the two Sparks: 1 block 10.6 us
/// at 4 KB, 4 blocks 13.7 us at 20 KB / 36 us at 160 KB, 16 blocks 131 us at 1 MB.
pub fn exchange_grid(bytes: usize) -> (u32, u32) {
    if bytes <= 8 << 10 {
        (1, 0)
    } else if bytes <= 192 << 10 {
        (4, 1)
    } else {
        (16, 2)
    }
}

pub struct Fabric {
    pub rank: u32,
    pub world: u32,
    nccl: nccl::Nccl,
    comm: nccl::NcclComm,
    /// the bootstrap link stays open for host-side exchanges (barriers, small control messages)
    pub link: boot::Link,
    pub roce: Option<roce::Roce>,
    /// route reductions through RoCE when it is up (true after `enable_roce`; false forces NCCL)
    pub use_roce: bool,
    _module: Module,
    k_gather_sum: Function,
    k_gather_sum16: Function,
    k_to_bf16: Function,
    k_interleave: Function,
    k_add: Function,
    /// polls before an exchange gives up on the peer (poisoning the path)
    pub spin_limit: u32,
}

unsafe impl Send for Fabric {}
unsafe impl Sync for Fabric {}

impl Fabric {
    /// Two ranks: rank 0 listens on `addr` (ip:port), rank 1 connects to it; then the NCCL communicator.
    /// The CUDA context must be current (NCCL uses the current device).
    pub fn connect(rank: u32, addr: &str) -> Result<Self> {
        let nccl = nccl::Nccl::open()?;
        let mut link = boot::Link::connect(rank, addr)?;
        let id = if rank == 0 {
            let id = nccl.unique_id()?;
            link.send(&id.0)?;
            id
        } else {
            let mut id = nccl::UniqueId([0; 128]);
            id.0.copy_from_slice(&link.recv()?);
            id
        };
        let comm = nccl.comm_init_rank(2, id, rank as i32)?;
        let module = Module::load(K_FABRIC_CUBIN)?;
        let k_gather_sum = module.function("fab_gather_sum_bf16")?;
        let k_gather_sum16 = module.function("fab_gather_sum_bf16in")?;
        let k_to_bf16 = module.function("fab_to_bf16")?;
        let k_interleave = module.function("fab_interleave_halves")?;
        let k_add = module.function("fab_add_bf16")?;
        Ok(Self { rank, world: 2, nccl, comm, link, roce: None,
            use_roce: false,
            _module: module,
            k_gather_sum,
            k_gather_sum16,
            k_to_bf16,
            k_interleave,
            k_add,
            spin_limit: 200_000_000,
        })
    }

    /// Bring up the RoCE one-shot path for reductions up to `cap` bytes of f32 partials (both ranks must call;
    /// FE_ROCE=0 on either side keeps NCCL for both).
    pub fn enable_roce(&mut self, cap: usize) -> Result<bool> {
        let mine = std::env::var("FE_ROCE").map(|v| v != "0").unwrap_or(true);
        let theirs = self.link.exchange(&[mine as u8])?;
        if !(mine && theirs[0] == 1) {
            return Ok(false);
        }
        self.roce = Some(roce::Roce::connect(&mut self.link, cap)?);
        self.use_roce = true;
        Ok(true)
    }

    /// out (bf16 [n]) = [resid +] (rank 0's partial + rank 1's) for f32 partials [n] (n % 4 == 0): the TP2
    /// reduction after a row-parallel linear. RoCE when enabled and it fits, else NCCL (the partial is
    /// all-reduced in place) + a conversion kernel. Both ranks get identical bits. Enqueued on `s`.
    pub fn reduce_to_bf16(&self, s: &Stream, partial: DevPtr, out: DevPtr, resid: Option<DevPtr>, n: usize) -> Result<()> {
        if n % 4 != 0 {
            bail!("reduce_to_bf16: n = {n} not a multiple of 4");
        }
        let packs = (n / 4) as u32;
        let resid = resid.unwrap_or(DevPtr(0));
        match &self.roce {
            Some(r) if self.use_roce && n * 4 <= r.layout.slot => unsafe {
                let (blocks, klass) = exchange_grid(n * 4);
                s.launch(
                    &self.k_gather_sum,
                    blocks,
                    256,
                    0,
                    (partial, out, resid, packs, self.rank, r.region_ptr(), r.layout.slot as u32, r.counters.ptr(), self.spin_limit, klass,
                     r.split_min),
                )?;
            },
            _ => unsafe {
                self.nccl.all_reduce_sum(self.comm, partial.0, partial.0, n, DataType::F32, s.raw() as nccl::CuStream)?;
                s.launch(&self.k_to_bf16, packs.div_ceil(256).min(64), 256, 0, (partial, out, resid, packs))?;
            },
        }
        Ok(())
    }

    /// out (bf16 [n]) = [resid +] (rank 0's partial + rank 1's) for bf16 partials [n] (n % 8 == 0), summed in f32
    /// and rounded once: the decode-row reduction at half the bytes of `reduce_to_bf16`. RoCE when enabled and it
    /// fits, else NCCL (bf16 all-reduce in place, then out = bf16(resid + sum): out must equal resid then).
    pub fn reduce_bf16_to_bf16(&self, s: &Stream, partial: DevPtr, out: DevPtr, resid: Option<DevPtr>, n: usize) -> Result<()> {
        if n % 8 != 0 {
            bail!("reduce_bf16_to_bf16: n = {n} not a multiple of 8");
        }
        let packs = (n / 8) as u32;
        match &self.roce {
            Some(r) if self.use_roce && n * 2 <= r.layout.slot => unsafe {
                let (blocks, klass) = exchange_grid(n * 2);
                s.launch(&self.k_gather_sum16, blocks, 256, 0, (partial, out, resid.unwrap_or(DevPtr(0)), packs, self.rank,
                    r.region_ptr(), r.layout.slot as u32, r.counters.ptr(), self.spin_limit, klass, r.split_min))?;
            },
            _ => {
                if resid.map(|r| r.0) != Some(out.0) {
                    bail!("reduce_bf16_to_bf16: the NCCL path adds in place (out == resid)");
                }
                self.reduce_add_bf16(s, partial, out, n)?;
            }
        }
        Ok(())
    }

    /// Prompt-row reduction of bf16 partials: x = bf16(x + bf16(p0 + p1)) (NCCL bf16 all-reduce of `partial`
    /// in place, then the add). `n` even.
    pub fn reduce_add_bf16(&self, s: &Stream, partial: DevPtr, x: DevPtr, n: usize) -> Result<()> {
        unsafe {
            self.nccl.all_reduce_sum(self.comm, partial.0, partial.0, n, DataType::Bf16, s.raw() as nccl::CuStream)?;
            let pairs = (n / 2) as u32;
            s.launch(&self.k_add, pairs.div_ceil(256).min(1024), 256, 0, (x, partial, pairs))?;
        }
        Ok(())
    }

    /// Vocabulary-parallel logits: every rank's [rows][half] bf16 block -> [rows][2 * half] on both ranks.
    /// `scratch`: bf16 room for 2 * rows * half. `half` % 8 == 0.
    pub fn gather_halves(&self, s: &Stream, mine: DevPtr, out: DevPtr, scratch: DevPtr, rows: usize, half: usize) -> Result<()> {
        unsafe {
            self.nccl.all_gather(self.comm, mine.0, scratch.0, rows * half, DataType::Bf16, s.raw() as nccl::CuStream)?;
            s.launch(&self.k_interleave, ((half as u32).div_ceil(2048), rows as u32, 2), 256, 0, (out, scratch, rows as u32, half as u32))?;
        }
        Ok(())
    }

    /// The RoCE path's error state (call after a synchronize that read a reduction).
    pub fn check(&self) -> Result<()> {
        match &self.roce {
            Some(r) => r.check(),
            None => Ok(()),
        }
    }

    pub fn nccl_version(&self) -> i32 {
        self.nccl.version
    }

    /// recv = send summed over the ranks (in place when equal), `count` elements, enqueued on `s`.
    pub fn all_reduce_sum(&self, s: &Stream, send: DevPtr, recv: DevPtr, count: usize, dt: DataType) -> Result<()> {
        unsafe { self.nccl.all_reduce_sum(self.comm, send.0, recv.0, count, dt, s.raw() as nccl::CuStream) }
    }

    /// recv = [rank 0's send | rank 1's send], `count` elements a rank, enqueued on `s`.
    pub fn all_gather(&self, s: &Stream, send: DevPtr, recv: DevPtr, count: usize, dt: DataType) -> Result<()> {
        unsafe { self.nccl.all_gather(self.comm, send.0, recv.0, count, dt, s.raw() as nccl::CuStream) }
    }

    /// Host barrier over the bootstrap link.
    pub fn barrier(&mut self) -> Result<()> {
        self.link.send(&[1u8])?;
        self.link.recv()?;
        Ok(())
    }
}

impl Drop for Fabric {
    fn drop(&mut self) {
        self.nccl.destroy(self.comm);
    }
}
