//! Chunked gated delta rule for GDN prompt rows (kernels/k-gdn): drop-in for k-core's `fe_gdn_rec2` on
//! segments without a state history (prefill), 32 rows a step instead of one. See the kernel crate's docs
//! for the math; `gdn-check` verifies it against an f64 row-by-row reference.
use anyhow::Result;
use fe_cuda::{DevPtr, Function, Module, Stream};

pub static K_GDN_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/k_gdn.cubin"));
/// One TP2 rank (8 key / 24 value heads, conv 5120): kernels/k-gdn-tp2.
pub static K_GDN_TP2_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/k_gdn_tp2.cubin"));

/// Must match kernels/k-gdn (SMEM_FLOATS * 4, T_FLOATS * 4).
pub const GDN_CHUNK_SMEM: u32 = 24868 * 4;
pub const GDN_TC_SMEM: u32 = 21508 * 4;

pub struct Gdn {
    pub module: Module,
    /// fp32 SIMT kernel (512 threads)
    pub chunk: Function,
    /// TF32 tensor-core kernel (256 threads) with 3xTF32 state products, the default
    pub chunk_tc: Function,
    /// same with single TF32 state products (FE_GDN_TC=1)
    pub chunk_tc1: Function,
    /// FE_GDN_TC: 0 = fp32 SIMT, 1 = single TF32, 3 = 3xTF32 (default)
    pub tc: u32,
    /// value heads (the grid): 48, or 24 for a TP2 rank
    pub nvh: u32,
}

impl Gdn {
    pub fn load() -> Result<Gdn> {
        Self::load_from(K_GDN_CUBIN, 48)
    }

    /// The kernels for one rank of a two-way head split (its 24 value heads; conv rows of 5120).
    pub fn load_tp2() -> Result<Gdn> {
        Self::load_from(K_GDN_TP2_CUBIN, 24)
    }

    fn load_from(cubin: &[u8], nvh: u32) -> Result<Gdn> {
        let module = Module::load(cubin)?;
        let chunk = module.function("fe_gdn_chunk")?;
        chunk.set_max_dynamic_smem(GDN_CHUNK_SMEM)?;
        let chunk_tc = module.function("fe_gdn_chunk_tc")?;
        chunk_tc.set_max_dynamic_smem(GDN_TC_SMEM)?;
        let chunk_tc1 = module.function("fe_gdn_chunk_tc1")?;
        chunk_tc1.set_max_dynamic_smem(GDN_TC_SMEM)?;
        let tc = std::env::var("FE_GDN_TC").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
        Ok(Gdn { module, chunk, chunk_tc, chunk_tc1, tc, nvh })
    }

    /// Same arguments as `fe_gdn_rec2` minus the history: o [rows][48*128] bf16, conv [rows][10240], a / b rows
    /// (stride ldab) of 48, a_log / dt_bias [48] bf16, state [slot][48][128][128] f32, segment tables (device
    /// u32 arrays of `nseg`).
    pub unsafe fn run(&self, s: &Stream, o: DevPtr, conv: DevPtr, ax: DevPtr, bx: DevPtr, ldab: usize, a_log: DevPtr,
        dt_bias: DevPtr, state: DevPtr, seg_slot: DevPtr, seg_row0: DevPtr, seg_len: DevPtr, nseg: usize) -> Result<()> {
        let (f, threads, smem) = match self.tc {
            0 => (&self.chunk, 512, GDN_CHUNK_SMEM),
            1 => (&self.chunk_tc1, 256, GDN_TC_SMEM),
            _ => (&self.chunk_tc, 256, GDN_TC_SMEM),
        };
        unsafe {
            s.launch(f, (self.nvh, nseg as u32), threads, smem, (o, conv, ax, bx, ldab as u32, a_log, dt_bias, state, seg_slot,
                seg_row0, seg_len))?;
        }
        Ok(())
    }
}
