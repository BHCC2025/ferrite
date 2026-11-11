//! The RoCE one-shot exchange between the two ranks (host side): libibverbs (system library, NON-RUST.md),
//! an RC queue pair on each HCA (FE_ROCE_HCA, comma-separated: by default both PCIe functions of the fabric port,
//! each its own x4 link), a pinned region both the GPU and the NICs reach, and a proxy thread that turns the
//! kernel's doorbell into RDMA writes: the shard (split across the queue pairs above SPLIT_MIN bytes), then on
//! each queue pair its sequence flag, so a flag never lands before its part. The GPU half is kernels/k-fabric
//! (`fab_gather_sum_bf16`). The protocol is TensorFold's (zig/src/cuda/roce.zig + roce_gather.cu, Apache-2.0),
//! after b12x's RoCEnante (Apache-2.0); see NOTICE.
//!
//! Region (each rank): recv[2][slot] (the peer's NIC writes) | flags[2][128 B] (the peer's NICs write, queue pair
//! k at + 64 k) | send[2][slot] (the kernel stages) | ctrl {seq, nbytes[2], error} (the kernel rings) | src[2]
//! (flag sources).
//! Struct layouts below are rdma-core 50 (aarch64), checked with offsetof (tools/ibv_offsets.c).

use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use fe_cuda::{DevPtr, DeviceBuffer, HostBuffer};

use crate::boot::Link;

// libibverbs is linked on the DGX Spark build (aarch64) or with the `roce` feature. Other builds (one GPU, no
// fabric) get stubs that report no RDMA device, so they need neither the library nor its headers.
#[cfg(any(target_arch = "aarch64", feature = "roce"))]
#[link(name = "ibverbs")]
unsafe extern "C" {
    fn ibv_get_device_list(num: *mut c_int) -> *mut *mut c_void;
    fn ibv_free_device_list(list: *mut *mut c_void);
    fn ibv_get_device_name(dev: *mut c_void) -> *const c_char;
    fn ibv_open_device(dev: *mut c_void) -> *mut c_void;
    fn ibv_close_device(ctx: *mut c_void) -> c_int;
    fn ibv_alloc_pd(ctx: *mut c_void) -> *mut c_void;
    fn ibv_dealloc_pd(pd: *mut c_void) -> c_int;
    fn ibv_reg_mr(pd: *mut c_void, addr: *mut c_void, len: usize, access: c_int) -> *mut c_void;
    fn ibv_dereg_mr(mr: *mut c_void) -> c_int;
    fn ibv_create_cq(ctx: *mut c_void, cqe: c_int, cq_ctx: *mut c_void, ch: *mut c_void, vec: c_int) -> *mut c_void;
    fn ibv_destroy_cq(cq: *mut c_void) -> c_int;
    fn ibv_create_qp(pd: *mut c_void, attr: *mut QpInitAttr) -> *mut c_void;
    fn ibv_destroy_qp(qp: *mut c_void) -> c_int;
    fn ibv_modify_qp(qp: *mut c_void, attr: *mut u8, mask: c_int) -> c_int;
    fn ibv_query_gid(ctx: *mut c_void, port: u8, index: c_int, gid: *mut [u8; 16]) -> c_int;
}

#[cfg(not(any(target_arch = "aarch64", feature = "roce")))]
mod no_verbs {
    use super::*;
    pub unsafe fn ibv_get_device_list(_num: *mut c_int) -> *mut *mut c_void { std::ptr::null_mut() }
    pub unsafe fn ibv_free_device_list(_list: *mut *mut c_void) {  }
    pub unsafe fn ibv_get_device_name(_dev: *mut c_void) -> *const c_char { std::ptr::null() }
    pub unsafe fn ibv_open_device(_dev: *mut c_void) -> *mut c_void { std::ptr::null_mut() }
    pub unsafe fn ibv_close_device(_ctx: *mut c_void) -> c_int { -1 }
    pub unsafe fn ibv_alloc_pd(_ctx: *mut c_void) -> *mut c_void { std::ptr::null_mut() }
    pub unsafe fn ibv_dealloc_pd(_pd: *mut c_void) -> c_int { -1 }
    pub unsafe fn ibv_reg_mr(_pd: *mut c_void, _addr: *mut c_void, _len: usize, _access: c_int) -> *mut c_void { std::ptr::null_mut() }
    pub unsafe fn ibv_dereg_mr(_mr: *mut c_void) -> c_int { -1 }
    pub unsafe fn ibv_create_cq(_ctx: *mut c_void, _cqe: c_int, _cq_ctx: *mut c_void, _ch: *mut c_void, _vec: c_int) -> *mut c_void { std::ptr::null_mut() }
    pub unsafe fn ibv_destroy_cq(_cq: *mut c_void) -> c_int { -1 }
    pub unsafe fn ibv_create_qp(_pd: *mut c_void, _attr: *mut QpInitAttr) -> *mut c_void { std::ptr::null_mut() }
    pub unsafe fn ibv_destroy_qp(_qp: *mut c_void) -> c_int { -1 }
    pub unsafe fn ibv_modify_qp(_qp: *mut c_void, _attr: *mut u8, _mask: c_int) -> c_int { -1 }
    pub unsafe fn ibv_query_gid(_ctx: *mut c_void, _port: u8, _index: c_int, _gid: *mut [u8; 16]) -> c_int { -1 }
}
#[cfg(not(any(target_arch = "aarch64", feature = "roce")))]
use no_verbs::*;

#[repr(C)]
struct QpInitAttr {
    qp_context: *mut c_void,
    send_cq: *mut c_void,
    recv_cq: *mut c_void,
    srq: *mut c_void,
    max_send_wr: u32,
    max_recv_wr: u32,
    max_send_sge: u32,
    max_recv_sge: u32,
    max_inline_data: u32,
    qp_type: u32,
    sq_sig_all: c_int,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Sge {
    addr: u64,
    length: u32,
    lkey: u32,
}

// struct ibv_send_wr is 128 bytes; fields written by offset
#[repr(C, align(8))]
struct SendWr([u8; 128]);

impl SendWr {
    fn rdma_write(sge: *mut Sge, remote: u64, rkey: u32, flags: u32, next: *mut SendWr) -> Self {
        let mut w = SendWr([0; 128]);
        let b = &mut w.0;
        b[8..16].copy_from_slice(&(next as u64).to_le_bytes()); // next
        b[16..24].copy_from_slice(&(sge as u64).to_le_bytes()); // sg_list
        b[24..28].copy_from_slice(&1i32.to_le_bytes()); // num_sge
        b[28..32].copy_from_slice(&0u32.to_le_bytes()); // opcode IBV_WR_RDMA_WRITE
        b[32..36].copy_from_slice(&flags.to_le_bytes()); // send_flags
        b[40..48].copy_from_slice(&remote.to_le_bytes()); // wr.rdma.remote_addr
        b[48..52].copy_from_slice(&rkey.to_le_bytes()); // wr.rdma.rkey
        w
    }
}

const QPS_INIT: u32 = 1;
const QPS_RTR: u32 = 2;
const QPS_RTS: u32 = 3;
const QPT_RC: u32 = 2;
const SEND_SIGNALED: u32 = 2;
const SEND_INLINE: u32 = 8;
const MTU_4096: u32 = 5;
const ACCESS: c_int = 1 | 2 | 4; // local write, remote write, remote read

const OFF_POLL_CQ: usize = 96; // ibv_context.ops.poll_cq
const OFF_POST_SEND: usize = 208; // ibv_context.ops.post_send
const OFF_QP_NUM: usize = 52;
const OFF_MR_LKEY: usize = 36;
const OFF_MR_RKEY: usize = 40;

type PostSend = unsafe extern "C" fn(*mut c_void, *mut SendWr, *mut *mut SendWr) -> c_int;
type PollCq = unsafe extern "C" fn(*mut c_void, c_int, *mut u8) -> c_int;

fn put<T: Copy>(b: &mut [u8], off: usize, v: T) {
    let n = std::mem::size_of::<T>();
    unsafe { std::ptr::copy_nonoverlapping(&v as *const T as *const u8, b[off..off + n].as_mut_ptr(), n) };
}

/// Layout of a rank's region for `slot` bytes a slot.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub slot: usize,
}

impl Layout {
    pub fn recv(&self, s: usize) -> usize {
        s * self.slot
    }
    pub fn flag(&self, s: usize) -> usize {
        2 * self.slot + 128 * s
    }
    pub fn send(&self, s: usize) -> usize {
        2 * self.slot + 256 + s * self.slot
    }
    pub fn ctrl(&self) -> usize {
        4 * self.slot + 256
    }
    pub fn src(&self, s: usize) -> usize {
        self.ctrl() + 64 + 4 * s
    }
    pub fn bytes(&self) -> usize {
        self.ctrl() + 128
    }
}

struct Verbs {
    ctx: *mut c_void,
    pd: *mut c_void,
    mr: *mut c_void,
    cq: *mut c_void,
    qp: *mut c_void,
}

/// Exchanges above this many bytes are split across the queue pairs (both halves on their own PCIe link);
/// smaller ones go whole through the first (latency). The kernel waits for the matching flags (`split_min`).
pub const SPLIT_MIN: usize = 32 << 10;
unsafe impl Send for Verbs {}
unsafe impl Sync for Verbs {}

pub struct Roce {
    pub layout: Layout,
    pub hca: String,
    region: HostBuffer<u8>,
    /// device counters: epoch, poison, stage and tail arrivals a grid class
    pub counters: DeviceBuffer<u32>,
    /// the kernel's split threshold in bytes (u32::MAX: one queue pair)
    pub split_min: u32,
    verbs: Arc<Vec<Verbs>>,
    stop: Arc<AtomicBool>,
    proxy: Option<std::thread::JoinHandle<Result<u64>>>,
}

unsafe impl Send for Roce {}
unsafe impl Sync for Roce {}

impl Roce {
    /// Both ranks: open the HCAs (fe_site::roce_hca, FERRITE_ROCE_HCA; port 1, GID index FERRITE_ROCE_GID), exchange
    /// queue pair / region keys over `link`, connect, start the proxy. `cap` = the largest exchange in bytes.
    pub fn connect(link: &mut Link, cap: usize) -> Result<Self> {
        let hca = fe_site::roce_hca();
        let gid_index: c_int = fe_site::roce_gid();
        let layout = Layout { slot: cap.div_ceil(256) * 256 };
        let names: Vec<&str> = hca.split(',').filter(|h| !h.is_empty()).collect();
        if names.is_empty() || names.len() > 2 {
            bail!("roce: FE_ROCE_HCA names 1 or 2 HCAs ({hca})");
        }
        let mut region = HostBuffer::<u8>::alloc(layout.bytes())?;
        region.as_mut_slice().fill(0);
        // both ranks must open as many queue pairs
        let peer_n = link.exchange(&[names.len() as u8])?;
        if peer_n != [names.len() as u8] {
            bail!("roce: {} HCAs here, {} at the peer", names.len(), peer_n[0]);
        }
        let mut verbs = Vec::new();
        let mut keys = Vec::new(); // (lkey, peer rkey)
        let mut paddr = 0u64;
        for name in &names {
            let (v, lkey, prkey, pa) = unsafe { Self::open_qp(link, name, gid_index, &region, layout)? };
            verbs.push(v);
            keys.push((lkey, prkey));
            paddr = pa;
        }
        link.exchange(&[1])?; // both connected
        let verbs = Arc::new(verbs);
        let stop = Arc::new(AtomicBool::new(false));
        let base = region.as_ptr() as usize;
        let (v2, s2) = (verbs.clone(), stop.clone());
        let proxy = std::thread::Builder::new()
            .name("fe-roce-proxy".into())
            .spawn(move || {
                let r = proxy_loop(v2, s2, base, layout, keys, paddr);
                if let Err(e) = &r {
                    eprintln!("fe-roce proxy: {e:#}");
                }
                r
            })
            .context("spawn proxy")?;
        let counters = DeviceBuffer::<u32>::zeroed(16)?;
        let split_min = if names.len() > 1 { SPLIT_MIN as u32 } else { u32::MAX };
        Ok(Self { layout, hca, region, counters, split_min, verbs, stop, proxy: Some(proxy) })
    }

    /// One HCA: open it, register the region, a connected RC queue pair to the peer's same-index HCA (keys and
    /// addresses exchanged over `link`). Returns (verbs, lkey, the peer's rkey, the peer's region address).
    unsafe fn open_qp(link: &mut Link, hca: &str, gid_index: c_int, region: &HostBuffer<u8>, layout: Layout)
        -> Result<(Verbs, u32, u32, u64)> {
        unsafe {
            let mut num = 0;
            let list = ibv_get_device_list(&mut num);
            if list.is_null() {
                bail!("roce: no RDMA devices");
            }
            let mut dev = std::ptr::null_mut();
            for i in 0..num as usize {
                let d = *list.add(i);
                if CStr::from_ptr(ibv_get_device_name(d)).to_string_lossy() == hca {
                    dev = d;
                }
            }
            if dev.is_null() {
                ibv_free_device_list(list);
                bail!("roce: HCA {hca} not found");
            }
            let ctx = ibv_open_device(dev);
            ibv_free_device_list(list);
            if ctx.is_null() {
                bail!("roce: open {hca}");
            }
            let pd = ibv_alloc_pd(ctx);
            let mr = ibv_reg_mr(pd, region.as_ptr() as *mut c_void, layout.bytes(), ACCESS);
            if pd.is_null() || mr.is_null() {
                bail!("roce: pd / reg_mr failed (errno {})", std::io::Error::last_os_error());
            }
            let cq = ibv_create_cq(ctx, 256, std::ptr::null_mut(), std::ptr::null_mut(), 0);
            let mut ia = QpInitAttr {
                qp_context: std::ptr::null_mut(),
                send_cq: cq,
                recv_cq: cq,
                srq: std::ptr::null_mut(),
                max_send_wr: 64,
                max_recv_wr: 1,
                max_send_sge: 1,
                max_recv_sge: 1,
                max_inline_data: 64,
                qp_type: QPT_RC,
                sq_sig_all: 0,
                _pad: 0,
            };
            let qp = ibv_create_qp(pd, &mut ia);
            if cq.is_null() || qp.is_null() {
                bail!("roce: cq / qp failed (errno {})", std::io::Error::last_os_error());
            }
            let mut gid = [0u8; 16];
            if ibv_query_gid(ctx, 1, gid_index, &mut gid) != 0 {
                bail!("roce: query gid {gid_index}");
            }
            let qpn = *((qp as *const u8).add(OFF_QP_NUM) as *const u32);
            let lkey = *((mr as *const u8).add(OFF_MR_LKEY) as *const u32);
            let rkey = *((mr as *const u8).add(OFF_MR_RKEY) as *const u32);
            let psn: u32 = 0x1234 + 0x100 * (qpn & 0xff);
            let mut mine = Vec::with_capacity(40);
            mine.extend_from_slice(&qpn.to_le_bytes());
            mine.extend_from_slice(&psn.to_le_bytes());
            mine.extend_from_slice(&gid);
            mine.extend_from_slice(&(region.as_ptr() as u64).to_le_bytes());
            mine.extend_from_slice(&rkey.to_le_bytes());
            mine.extend_from_slice(&(layout.slot as u32).to_le_bytes());
            let peer = link.exchange(&mine)?;
            let pqpn = u32::from_le_bytes(peer[0..4].try_into()?);
            let ppsn = u32::from_le_bytes(peer[4..8].try_into()?);
            let mut pgid = [0u8; 16];
            pgid.copy_from_slice(&peer[8..24]);
            let paddr = u64::from_le_bytes(peer[24..32].try_into()?);
            let prkey = u32::from_le_bytes(peer[32..36].try_into()?);
            let pslot = u32::from_le_bytes(peer[36..40].try_into()?) as usize;
            if pslot != layout.slot {
                bail!("roce: slot sizes differ ({} here, {pslot} at the peer)", layout.slot);
            }
            // INIT
            let mut a = [0u8; 144];
            put(&mut a, 0, QPS_INIT);
            put(&mut a, 120, 0u16); // pkey_index
            put(&mut a, 129, 1u8); // port_num
            put(&mut a, 32, ACCESS as u32); // qp_access_flags
            if ibv_modify_qp(qp, a.as_mut_ptr(), 1 | 16 | 32 | 8) != 0 {
                bail!("roce: qp -> INIT (errno {})", std::io::Error::last_os_error());
            }
            // RTR
            let mut a = [0u8; 144];
            put(&mut a, 0, QPS_RTR);
            put(&mut a, 8, MTU_4096);
            put(&mut a, 28, pqpn);
            put(&mut a, 20, ppsn); // rq_psn
            put(&mut a, 127, 1u8); // max_dest_rd_atomic
            put(&mut a, 128, 12u8); // min_rnr_timer
            a[56..72].copy_from_slice(&pgid); // ah_attr.grh.dgid
            put(&mut a, 56 + 20, gid_index as u8); // grh.sgid_index
            put(&mut a, 56 + 21, 64u8); // grh.hop_limit
            put(&mut a, 56 + 29, 1u8); // ah_attr.is_global
            put(&mut a, 56 + 30, 1u8); // ah_attr.port_num
            if ibv_modify_qp(qp, a.as_mut_ptr(), 1 | 128 | 256 | 1048576 | 4096 | 131072 | 32768) != 0 {
                bail!("roce: qp -> RTR (errno {})", std::io::Error::last_os_error());
            }
            // RTS
            let mut a = [0u8; 144];
            put(&mut a, 0, QPS_RTS);
            put(&mut a, 130, 14u8); // timeout
            put(&mut a, 131, 7u8); // retry_cnt
            put(&mut a, 132, 7u8); // rnr_retry
            put(&mut a, 24, psn); // sq_psn
            put(&mut a, 126, 1u8); // max_rd_atomic
            if ibv_modify_qp(qp, a.as_mut_ptr(), 1 | 512 | 1024 | 2048 | 65536 | 8192) != 0 {
                bail!("roce: qp -> RTS (errno {})", std::io::Error::last_os_error());
            }
            Ok((Verbs { ctx, pd, mr, cq, qp }, lkey, prkey, paddr))
        }
    }

    /// The region's base (host pointer; the GPU uses the same address).
    pub fn region_ptr(&self) -> DevPtr {
        DevPtr(self.region.as_ptr() as u64)
    }

    /// Error state: the sequence that timed out on the GPU side, or the proxy's failure.
    pub fn check(&self) -> Result<()> {
        let ctrl = unsafe { self.region.as_ptr().add(self.layout.ctrl()) as *const u32 };
        let err = unsafe { std::ptr::read_volatile(ctrl.add(3)) };
        if err != 0 {
            bail!("roce: exchange {err} timed out waiting for the peer");
        }
        if let Some(p) = &self.proxy {
            if p.is_finished() {
                bail!("roce: proxy thread stopped");
            }
        }
        Ok(())
    }
}

/// Doorbell -> RDMA writes. Returns the number of exchanges served when stopped.
fn proxy_loop(v: Arc<Vec<Verbs>>, stop: Arc<AtomicBool>, base: usize, l: Layout, keys: Vec<(u32, u32)>, paddr: u64)
    -> Result<u64> {
    unsafe {
        let fns: Vec<(PostSend, PollCq)> = v.iter().map(|v| {
            let ctx = *(v.qp as *const *mut c_void);
            let post_send: PostSend = std::mem::transmute(*((ctx as *const u8).add(OFF_POST_SEND) as *const *const c_void));
            let poll_cq: PollCq = std::mem::transmute(*(((*(v.cq as *const *mut c_void)) as *const u8).add(OFF_POLL_CQ)
                as *const *const c_void));
            (post_send, poll_cq)
        }).collect();
        let ctrl = (base + l.ctrl()) as *const u32;
        let mut last = 0u32;
        let mut served = 0u64;
        let debug = std::env::var("FE_ROCE_DEBUG").is_ok();
        let mut wc = [0u8; 48];
        while !stop.load(Ordering::Relaxed) {
            let seq = std::ptr::read_volatile(ctrl);
            if seq == last {
                std::hint::spin_loop();
                continue;
            }
            std::sync::atomic::fence(Ordering::Acquire);
            // A rank's kernel only waits for the PEER's shard, so it can finish exchange q and ring q + 1 before
            // this thread saw q: serve every sequence since the last one, in order. Their slots are intact (the
            // kernel for q + 2 cannot start before the peer received q).
            let mut q = last;
            while q != seq {
                q = q.wrapping_add(1);
                let s = (q & 1) as usize;
                let n = std::ptr::read_volatile(ctrl.add(1 + s));
                let src = (base + l.src(s)) as *mut u32;
                std::ptr::write_volatile(src, q);
                // the parts: whole on the first queue pair, or halves (a 256-byte boundary) on both
                let nq = if v.len() > 1 && n as usize > SPLIT_MIN { v.len() } else { 1 };
                let h = if nq > 1 { (n / 2).div_ceil(256) * 256 } else { n };
                let mut sges = [Sge { addr: 0, length: 0, lkey: 0 }; 4];
                let mut wrs: Vec<SendWr> = Vec::with_capacity(4);
                for k in 0..nq {
                    let (lkey, prkey) = keys[k];
                    let (off, len) = if k == 0 { (0, h) } else { (h, n - h) };
                    sges[2 * k] = Sge { addr: (base + l.send(s)) as u64 + off as u64, length: len, lkey };
                    sges[2 * k + 1] = Sge { addr: src as u64, length: 4, lkey };
                    wrs.push(SendWr::rdma_write(&mut sges[2 * k + 1], paddr + (l.flag(s) + 64 * k) as u64, prkey,
                        SEND_SIGNALED | SEND_INLINE, std::ptr::null_mut()));
                    wrs.push(SendWr::rdma_write(&mut sges[2 * k], paddr + (l.recv(s) + off as usize) as u64, prkey, 0,
                        std::ptr::null_mut()));
                }
                for k in 0..nq {
                    let flag: *mut SendWr = &mut wrs[2 * k];
                    let data = &mut wrs[2 * k + 1];
                    data.0[8..16].copy_from_slice(&(flag as u64).to_le_bytes()); // data -> flag
                    let mut bad = std::ptr::null_mut();
                    if (fns[k].0)(v[k].qp, data, &mut bad) != 0 {
                        bail!("roce proxy: post_send failed at seq {q} (queue pair {k})");
                    }
                }
                for k in 0..nq {
                    loop {
                        let got = (fns[k].1)(v[k].cq, 1, wc.as_mut_ptr());
                        if got < 0 {
                            bail!("roce proxy: poll_cq failed");
                        }
                        if got == 1 {
                            let status = u32::from_le_bytes(wc[8..12].try_into().unwrap());
                            if status != 0 {
                                bail!("roce proxy: completion status {status} at seq {q} (queue pair {k})");
                            }
                            break;
                        }
                    }
                }
                if debug {
                    eprintln!("fe-roce proxy: served seq {q} slot {s} {n} bytes on {nq} queue pair(s)");
                }
            }
            served += seq.wrapping_sub(last) as u64;
            last = seq;
        }
        Ok(served)
    }
}

impl Drop for Roce {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(p) = self.proxy.take() {
            let _ = p.join();
        }
        for v in self.verbs.iter() {
            unsafe {
                ibv_destroy_qp(v.qp);
                ibv_destroy_cq(v.cq);
                ibv_dereg_mr(v.mr);
                ibv_dealloc_pd(v.pd);
                ibv_close_device(v.ctx);
            }
        }
    }
}
