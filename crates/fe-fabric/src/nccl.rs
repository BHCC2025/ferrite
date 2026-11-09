//! NCCL, opened at run time (dlopen, as the driver is): the TP2 baseline collective. Not Rust (NON-RUST.md).
//! Library: fe_site::nccl_libs (FERRITE_NCCL_LIB, else libnccl.so.2 on the loader path).

use std::ffi::{c_char, c_int, c_void, CStr, CString};

use anyhow::{anyhow, bail, Result};

pub type NcclComm = *mut c_void;
pub type CuStream = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct UniqueId(pub [u8; 128]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum DataType {
    U8 = 1,
    I32 = 2,
    F16 = 6,
    F32 = 7,
    Bf16 = 9,
}

impl DataType {
    pub fn size(self) -> usize {
        match self {
            DataType::U8 => 1,
            DataType::F16 | DataType::Bf16 => 2,
            DataType::I32 | DataType::F32 => 4,
        }
    }
}

const NCCL_SUM: c_int = 0;

type GetUniqueId = unsafe extern "C" fn(*mut UniqueId) -> c_int;
type CommInitRank = unsafe extern "C" fn(*mut NcclComm, c_int, UniqueId, c_int) -> c_int;
type AllReduce = unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int, c_int, NcclComm, CuStream) -> c_int;
type AllGather = unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int, NcclComm, CuStream) -> c_int;
type CommDestroy = unsafe extern "C" fn(NcclComm) -> c_int;
type GetErrorString = unsafe extern "C" fn(c_int) -> *const c_char;
type GetVersion = unsafe extern "C" fn(*mut c_int) -> c_int;

pub struct Nccl {
    _lib: *mut c_void,
    get_unique_id: GetUniqueId,
    comm_init_rank: CommInitRank,
    all_reduce: AllReduce,
    all_gather: AllGather,
    comm_destroy: CommDestroy,
    error_string: GetErrorString,
    pub version: i32,
}

unsafe impl Send for Nccl {}
unsafe impl Sync for Nccl {}

fn sym<T: Copy>(lib: *mut c_void, name: &str) -> Result<T> {
    let c = CString::new(name).unwrap();
    let p = unsafe { libc::dlsym(lib, c.as_ptr()) };
    if p.is_null() {
        bail!("nccl: symbol {name} missing");
    }
    Ok(unsafe { std::mem::transmute_copy(&p) })
}

impl Nccl {
    pub fn open() -> Result<Self> {
        for c in fe_site::nccl_libs() {
            let cs = CString::new(c.clone()).unwrap();
            let lib = unsafe { libc::dlopen(cs.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
            if lib.is_null() {
                continue;
            }
            let get_version: GetVersion = sym(lib, "ncclGetVersion")?;
            let mut version = 0;
            unsafe { get_version(&mut version) };
            return Ok(Self {
                _lib: lib,
                get_unique_id: sym(lib, "ncclGetUniqueId")?,
                comm_init_rank: sym(lib, "ncclCommInitRank")?,
                all_reduce: sym(lib, "ncclAllReduce")?,
                all_gather: sym(lib, "ncclAllGather")?,
                comm_destroy: sym(lib, "ncclCommDestroy")?,
                error_string: sym(lib, "ncclGetErrorString")?,
                version,
            });
        }
        bail!("nccl: no libnccl.so.2 (set FE_NCCL_LIB)")
    }

    fn check(&self, r: c_int, what: &str) -> Result<()> {
        if r == 0 {
            return Ok(());
        }
        let s = unsafe { CStr::from_ptr((self.error_string)(r)) }.to_string_lossy();
        Err(anyhow!("{what}: NCCL error {r} ({s})"))
    }

    pub fn unique_id(&self) -> Result<UniqueId> {
        let mut id = UniqueId([0; 128]);
        self.check(unsafe { (self.get_unique_id)(&mut id) }, "ncclGetUniqueId")?;
        Ok(id)
    }

    pub fn comm_init_rank(&self, world: i32, id: UniqueId, rank: i32) -> Result<NcclComm> {
        let mut c = std::ptr::null_mut();
        self.check(unsafe { (self.comm_init_rank)(&mut c, world, id, rank) }, "ncclCommInitRank")?;
        Ok(c)
    }

    /// Sum all-reduce of `count` elements (in place when send == recv), enqueued on `stream`.
    ///
    /// # Safety
    /// Device pointers valid for `count` elements of `dt`.
    pub unsafe fn all_reduce_sum(&self, comm: NcclComm, send: u64, recv: u64, count: usize, dt: DataType, stream: CuStream) -> Result<()> {
        self.check(
            unsafe { (self.all_reduce)(send as *const c_void, recv as *mut c_void, count, dt as c_int, NCCL_SUM, comm, stream) },
            "ncclAllReduce",
        )
    }

    /// All-gather: recv = [rank 0's send | rank 1's send | ...], `count` elements a rank.
    ///
    /// # Safety
    /// Device pointers valid (recv for world * count elements).
    pub unsafe fn all_gather(&self, comm: NcclComm, send: u64, recv: u64, count: usize, dt: DataType, stream: CuStream) -> Result<()> {
        self.check(unsafe { (self.all_gather)(send as *const c_void, recv as *mut c_void, count, dt as c_int, comm, stream) }, "ncclAllGather")
    }

    pub fn destroy(&self, comm: NcclComm) {
        unsafe { (self.comm_destroy)(comm) };
    }
}
