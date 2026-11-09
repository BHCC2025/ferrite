//! Thin, owned bindings to the CUDA driver API (libcuda only; no runtime, no cudarc).
//!
//! Context (primary context of one device), [`Stream`], [`Event`], [`DeviceBuffer`], [`HostBuffer`]
//! (pinned), [`Module`]/[`Function`] (cubin load + typed launch), and [`Graph`] (stream capture).
pub mod ffi;

use ffi::*;
use std::ffi::{c_void, CStr, CString};
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

// ---- errors -----------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CuError {
    pub code: CUresult,
    pub call: &'static str,
}
impl fmt::Display for CuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (mut n, mut s) = (std::ptr::null(), std::ptr::null());
        unsafe {
            cuGetErrorName(self.code, &mut n);
            cuGetErrorString(self.code, &mut s);
        }
        let cs = |p: *const std::ffi::c_char| if p.is_null() { "?".into() } else { unsafe { CStr::from_ptr(p) }.to_string_lossy() };
        write!(f, "{} failed: {} ({}): {}", self.call, cs(n), self.code, cs(s))
    }
}
impl std::error::Error for CuError {}
pub type Result<T> = std::result::Result<T, CuError>;

#[inline]
fn check(r: CUresult, call: &'static str) -> Result<()> {
    if r == CUDA_SUCCESS { Ok(()) } else { Err(CuError { code: r, call }) }
}
macro_rules! cu {
    ($f:ident ( $($a:expr),* $(,)? )) => {
        check(unsafe { $f($($a),*) }, stringify!($f))
    };
}

// ---- context ----------------------------------------------------------------------------------

pub struct Context {
    ctx: CUcontext,
    dev: CUdevice,
}
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    /// Initialise the driver, retain the device's primary context and make it current on this thread.
    /// Also raises this process's OOM score: GPU memory is host memory on the GB10, and a process launched over
    /// ssh inherits sshd's -1000 (unkillable), so a GPU OOM would take the whole box down instead of this process.
    pub fn new(ordinal: i32) -> Result<Arc<Context>> {
        let _ = std::fs::write("/proc/self/oom_score_adj", "500");
        cu!(cuInit(0))?;
        let mut dev = 0;
        cu!(cuDeviceGet(&mut dev, ordinal))?;
        let mut ctx = std::ptr::null_mut();
        cu!(cuDevicePrimaryCtxRetain(&mut ctx, dev))?;
        let c = Arc::new(Context { ctx, dev });
        c.bind()?;
        Ok(c)
    }
    /// Make this context current on the calling thread (needed once per thread).
    pub fn bind(&self) -> Result<()> {
        cu!(cuCtxSetCurrent(self.ctx))
    }
    pub fn attribute(&self, attr: i32) -> Result<i32> {
        let mut v = 0;
        cu!(cuDeviceGetAttribute(&mut v, attr, self.dev))?;
        Ok(v)
    }
    pub fn name(&self) -> String {
        let mut buf = [0 as std::ffi::c_char; 256];
        unsafe { cuDeviceGetName(buf.as_mut_ptr(), 256, self.dev) };
        unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
    }
    pub fn sm_count(&self) -> Result<i32> {
        self.attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    }
    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        Ok((self.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
            self.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?))
    }
    pub fn max_smem_optin(&self) -> Result<i32> {
        self.attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    }
    /// (free, total) bytes as the driver sees them (unified memory on GB10).
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let (mut f, mut t) = (0, 0);
        cu!(cuMemGetInfo_v2(&mut f, &mut t))?;
        Ok((f, t))
    }
    pub fn synchronize(&self) -> Result<()> {
        cu!(cuCtxSynchronize())
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe { cuDevicePrimaryCtxRelease_v2(self.dev) };
    }
}

// ---- device memory ----------------------------------------------------------------------------

/// A raw device address; what kernels receive for pointer parameters.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DevPtr(pub u64);
impl DevPtr {
    pub const NULL: DevPtr = DevPtr(0);
    #[inline]
    pub fn byte_add(self, bytes: usize) -> DevPtr {
        DevPtr(self.0 + bytes as u64)
    }
    /// Offset by `n` elements of `T`.
    #[inline]
    pub fn add<T>(self, n: usize) -> DevPtr {
        DevPtr(self.0 + (n * std::mem::size_of::<T>()) as u64)
    }
    pub fn is_null(self) -> bool {
        self.0 == 0
    }
}

/// Owned device allocation of `len` elements of `T`.
pub struct DeviceBuffer<T> {
    ptr: DevPtr,
    len: usize,
    _t: PhantomData<T>,
}
unsafe impl<T> Send for DeviceBuffer<T> {}
unsafe impl<T> Sync for DeviceBuffer<T> {}

impl<T: Copy> DeviceBuffer<T> {
    pub fn alloc(len: usize) -> Result<Self> {
        let bytes = len * std::mem::size_of::<T>();
        let mut p = 0;
        if bytes > 0 {
            if canary::on() {
                cu!(cuMemAlloc_v2(&mut p, bytes + canary::PAD))?;
                canary::arm(p, bytes)?;
            } else {
                cu!(cuMemAlloc_v2(&mut p, bytes))?;
            }
            if canary::zero() {
                cu!(cuMemsetD8Async(p, 0, bytes, std::ptr::null_mut()))?;
                cu!(cuCtxSynchronize())?;
            }
        }
        Ok(DeviceBuffer { ptr: DevPtr(p), len, _t: PhantomData })
    }
    /// Allocate and zero (synchronous).
    pub fn zeroed(len: usize) -> Result<Self> {
        let b = Self::alloc(len)?;
        if b.bytes() > 0 {
            cu!(cuMemsetD8Async(b.ptr.0, 0, b.bytes(), std::ptr::null_mut()))?;
            cu!(cuCtxSynchronize())?;
        }
        Ok(b)
    }
    pub fn from_slice(src: &[T]) -> Result<Self> {
        let mut b = Self::alloc(src.len())?;
        b.copy_from(src)?;
        Ok(b)
    }
    #[inline]
    pub fn ptr(&self) -> DevPtr {
        self.ptr
    }
    /// Device address of element `i`.
    #[inline]
    pub fn at(&self, i: usize) -> DevPtr {
        self.ptr.add::<T>(i)
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub fn bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
    /// Synchronous host -> device copy into the start of the buffer, complete and ordered against every stream
    /// ([`memcpy_htod`]).
    pub fn copy_from(&mut self, src: &[T]) -> Result<()> {
        assert!(src.len() <= self.len);
        memcpy_htod(self.ptr, src)
    }
    /// Synchronous device -> host copy from the start of the buffer, after all queued work ([`memcpy_dtoh`]).
    pub fn copy_to(&self, dst: &mut [T]) -> Result<()> {
        assert!(dst.len() <= self.len);
        memcpy_dtoh(dst, self.ptr)
    }
    pub fn to_vec(&self) -> Result<Vec<T>> {
        let mut v = Vec::with_capacity(self.len);
        unsafe { v.set_len(self.len) };
        self.copy_to(&mut v)?;
        Ok(v)
    }
}
impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.ptr.0 != 0 {
            canary::disarm(self.ptr.0);
            unsafe { cuMemFree_v2(self.ptr.0) };
        }
    }
}

/// Debug aid (FE_CANARY=1): every DeviceBuffer gets a 64 KB tail filled with 0xA5; [`canary::tails`] reports
/// any tail that was written, i.e. a kernel that overflowed its buffer (memcheck cannot see an overflow that
/// lands in a neighbouring allocation).
pub mod canary {
    use super::*;
    use std::sync::Mutex;
    pub const PAD: usize = 64 << 10;
    static LIVE: Mutex<Vec<(u64, usize)>> = Mutex::new(Vec::new());
    /// Zero every allocation (default; FE_ZERO_ALLOC=0 turns it off). Some kernel reads device memory it never
    /// wrote: with recycled (non-zero) pages the hybrid run was nondeterministic, zeroed it is not. Still hunted.
    pub fn zero() -> bool {
        static Z: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *Z.get_or_init(|| std::env::var("FE_ZERO_ALLOC").map(|v| v != "0").unwrap_or(true))
    }
    pub fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("FE_CANARY").map(|v| v == "1").unwrap_or(false))
    }
    pub(crate) fn arm(base: u64, bytes: usize) -> Result<()> {
        cu!(cuMemsetD8Async(base + bytes as u64, 0xA5, PAD, std::ptr::null_mut()))?;
        cu!(cuCtxSynchronize())?;
        LIVE.lock().unwrap().push((base, bytes));
        Ok(())
    }
    pub(crate) fn disarm(base: u64) {
        if on() {
            LIVE.lock().unwrap().retain(|e| e.0 != base);
        }
    }
    /// (buffer base, buffer bytes, first corrupted tail offset, corrupted byte count) for every damaged tail.
    pub fn tails() -> Result<Vec<(u64, usize, usize, usize)>> {
        let live = LIVE.lock().unwrap().clone();
        let mut bad = Vec::new();
        let mut buf = vec![0u8; PAD];
        for (base, bytes) in live {
            cu!(cuMemcpyDtoH_v2(buf.as_mut_ptr() as *mut c_void, base + bytes as u64, PAD))?;
            let n = buf.iter().filter(|&&b| b != 0xA5).count();
            if n > 0 {
                let first = buf.iter().position(|&b| b != 0xA5).unwrap();
                bad.push((base, bytes, first, n));
            }
        }
        Ok(bad)
    }
}

/// Synchronous copies on raw pointers (for views into a larger arena), safe next to non-blocking streams.
/// cuMemcpyHtoD from pageable memory only orders against the legacy stream and returns once the bytes are staged,
/// before the DMA lands: a kernel on a non-blocking stream could still be reading the destination, or read it
/// before the copy arrives (the hybrid run's random bad steps, and the 9-16-row GEMV report that was really its
/// test tool's upload). Both copies therefore wait for every stream first, and the upload also for its own DMA.
/// Per-step uploads should stay on the engine stream from pinned memory (Stream::memcpy_htod).
pub fn memcpy_htod<T: Copy>(dst: DevPtr, src: &[T]) -> Result<()> {
    if src.is_empty() {
        return Ok(());
    }
    cu!(cuCtxSynchronize())?;
    cu!(cuMemcpyHtoD_v2(dst.0, src.as_ptr() as *const c_void, std::mem::size_of_val(src)))?;
    cu!(cuCtxSynchronize())
}
pub fn memcpy_dtoh<T: Copy>(dst: &mut [T], src: DevPtr) -> Result<()> {
    if dst.is_empty() {
        return Ok(());
    }
    cu!(cuCtxSynchronize())?;
    cu!(cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut c_void, src.0, std::mem::size_of_val(dst)))
}

/// Synchronous device -> device copy.
pub fn memcpy_dtod(dst: DevPtr, src: DevPtr, bytes: usize) -> Result<()> {
    cu!(cuMemcpyDtoDAsync_v2(dst.0, src.0, bytes, std::ptr::null_mut()))?;
    cu!(cuCtxSynchronize())
}

/// Page-locked host memory (cuMemAllocHost), for staging uploads and async readback.
pub struct HostBuffer<T> {
    ptr: *mut T,
    len: usize,
}
unsafe impl<T> Send for HostBuffer<T> {}
impl<T: Copy> HostBuffer<T> {
    pub fn alloc(len: usize) -> Result<Self> {
        let mut p = std::ptr::null_mut();
        cu!(cuMemAllocHost_v2(&mut p, len.max(1) * std::mem::size_of::<T>()))?;
        Ok(HostBuffer { ptr: p as *mut T, len })
    }
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
impl<T> Drop for HostBuffer<T> {
    fn drop(&mut self) {
        unsafe { cuMemFreeHost(self.ptr as *mut c_void) };
    }
}

// ---- modules, functions, launch ---------------------------------------------------------------

struct ModInner(CUmodule);
unsafe impl Send for ModInner {}
unsafe impl Sync for ModInner {}
impl Drop for ModInner {
    fn drop(&mut self) {
        unsafe { cuModuleUnload(self.0) };
    }
}

/// A loaded cubin (one per kernels/k-* crate).
#[derive(Clone)]
pub struct Module(Arc<ModInner>);
impl Module {
    /// Load a cubin image (or NUL-terminated PTX).
    pub fn load(image: &[u8]) -> Result<Module> {
        // cuModuleLoadData reads the ELF header for size; copy into an aligned buffer to be safe.
        let mut v: Vec<u64> = vec![0; image.len().div_ceil(8) + 1];
        unsafe { std::ptr::copy_nonoverlapping(image.as_ptr(), v.as_mut_ptr() as *mut u8, image.len()) };
        let mut m = std::ptr::null_mut();
        cu!(cuModuleLoadData(&mut m, v.as_ptr() as *const c_void))?;
        Ok(Module(Arc::new(ModInner(m))))
    }
    pub fn function(&self, name: &str) -> Result<Function> {
        let c = CString::new(name).unwrap();
        let mut f = std::ptr::null_mut();
        check(unsafe { cuModuleGetFunction(&mut f, self.0.0, c.as_ptr()) }, "cuModuleGetFunction")
            .map_err(|e| {
                eprintln!("fe-cuda: kernel `{name}` not found in module");
                e
            })?;
        Ok(Function { f, _m: self.0.clone(), name: name.to_string() })
    }
}

#[derive(Clone)]
pub struct Function {
    f: CUfunction,
    _m: Arc<ModInner>,
    pub name: String,
}
unsafe impl Send for Function {}
unsafe impl Sync for Function {}
impl Function {
    pub fn raw(&self) -> CUfunction {
        self.f
    }
    /// Allow more than 48 KB of dynamic shared memory.
    pub fn set_max_dynamic_smem(&self, bytes: u32) -> Result<()> {
        cu!(cuFuncSetAttribute(self.f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes as i32))
    }
    pub fn attribute(&self, attr: i32) -> Result<i32> {
        let mut v = 0;
        cu!(cuFuncGetAttribute(&mut v, attr, self.f))?;
        Ok(v)
    }
    pub fn num_regs(&self) -> Result<i32> {
        self.attribute(CU_FUNC_ATTRIBUTE_NUM_REGS)
    }
    pub fn static_smem(&self) -> Result<i32> {
        self.attribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}
impl From<u32> for Dim3 {
    fn from(x: u32) -> Self {
        Dim3 { x, y: 1, z: 1 }
    }
}
impl From<(u32, u32)> for Dim3 {
    fn from((x, y): (u32, u32)) -> Self {
        Dim3 { x, y, z: 1 }
    }
}
impl From<(u32, u32, u32)> for Dim3 {
    fn from((x, y, z): (u32, u32, u32)) -> Self {
        Dim3 { x, y, z }
    }
}

/// Plain-old-data kernel parameter. Must match the kernel's parameter type exactly (size and alignment).
pub unsafe trait KernelArg: Copy {}
unsafe impl KernelArg for u32 {}
unsafe impl KernelArg for i32 {}
unsafe impl KernelArg for u64 {}
unsafe impl KernelArg for i64 {}
unsafe impl KernelArg for f32 {}
unsafe impl KernelArg for f64 {}
unsafe impl KernelArg for DevPtr {}

/// A tuple of kernel arguments; `launch` passes pointers to each element.
pub trait KernelArgs {
    fn with_ptrs<Ret>(&self, f: impl FnOnce(*mut *mut c_void) -> Ret) -> Ret;
}
impl KernelArgs for () {
    fn with_ptrs<Ret>(&self, f: impl FnOnce(*mut *mut c_void) -> Ret) -> Ret {
        f(std::ptr::null_mut())
    }
}
macro_rules! impl_args {
    ($($T:ident $i:tt),+) => {
        impl<$($T: KernelArg),+> KernelArgs for ($($T,)+) {
            fn with_ptrs<Ret>(&self, f: impl FnOnce(*mut *mut c_void) -> Ret) -> Ret {
                let mut p = [$(&self.$i as *const $T as *mut c_void),+];
                f(p.as_mut_ptr())
            }
        }
    };
}
impl_args!(A 0);
impl_args!(A 0, B 1);
impl_args!(A 0, B 1, C 2);
impl_args!(A 0, B 1, C 2, D 3);
impl_args!(A 0, B 1, C 2, D 3, E 4);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14, P 15);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14, P 15, Q 16);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14, P 15, Q 16, R 17);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14, P 15, Q 16, R 17, S 18);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14, P 15, Q 16, R 17, S 18, T 19);

// ---- streams, events, graphs ------------------------------------------------------------------

pub struct Stream {
    s: CUstream,
}
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    /// A non-blocking stream.
    pub fn new() -> Result<Stream> {
        let mut s = std::ptr::null_mut();
        cu!(cuStreamCreate(&mut s, CU_STREAM_NON_BLOCKING))?;
        Ok(Stream { s })
    }
    pub fn raw(&self) -> CUstream {
        self.s
    }
    pub fn sync(&self) -> Result<()> {
        cu!(cuStreamSynchronize(self.s))
    }
    /// Enqueue `f<<<grid, block, smem, self>>>(args...)`.
    ///
    /// # Safety
    /// `args` must match the kernel's parameter list exactly; pointers must be valid for the access.
    pub unsafe fn launch<A: KernelArgs>(&self, f: &Function, grid: impl Into<Dim3>, block: impl Into<Dim3>, smem: u32,
        args: A) -> Result<()> {
        let (g, b) = (grid.into(), block.into());
        let r = args.with_ptrs(|p| unsafe {
            cuLaunchKernel(f.f, g.x, g.y, g.z, b.x, b.y, b.z, smem, self.s, p, std::ptr::null_mut())
        });
        check(r, "cuLaunchKernel").map_err(|e| {
            eprintln!("fe-cuda: launch of `{}` grid {g:?} block {b:?} smem {smem} failed", f.name);
            e
        })
    }
    /// # Safety
    /// `src` must stay alive and unmodified until the copy completes (use pinned memory for true async).
    pub unsafe fn memcpy_htod<T: Copy>(&self, dst: DevPtr, src: &[T]) -> Result<()> {
        cu!(cuMemcpyHtoDAsync_v2(dst.0, src.as_ptr() as *const c_void, std::mem::size_of_val(src), self.s))
    }
    /// # Safety
    /// `dst` must stay alive until the copy completes.
    pub unsafe fn memcpy_dtoh<T: Copy>(&self, dst: &mut [T], src: DevPtr) -> Result<()> {
        cu!(cuMemcpyDtoHAsync_v2(dst.as_mut_ptr() as *mut c_void, src.0, std::mem::size_of_val(dst), self.s))
    }
    pub fn memcpy_dtod(&self, dst: DevPtr, src: DevPtr, bytes: usize) -> Result<()> {
        cu!(cuMemcpyDtoDAsync_v2(dst.0, src.0, bytes, self.s))
    }
    pub fn memset_d8(&self, dst: DevPtr, v: u8, n: usize) -> Result<()> {
        cu!(cuMemsetD8Async(dst.0, v, n, self.s))
    }
    pub fn memset_d32(&self, dst: DevPtr, v: u32, n: usize) -> Result<()> {
        cu!(cuMemsetD32Async(dst.0, v, n, self.s))
    }
    pub fn record(&self, e: &Event) -> Result<()> {
        cu!(cuEventRecord(e.e, self.s))
    }
    pub fn wait(&self, e: &Event) -> Result<()> {
        cu!(cuStreamWaitEvent(self.s, e.e, 0))
    }
    /// Start capturing launches into a graph (thread-local mode).
    pub fn begin_capture(&self) -> Result<()> {
        cu!(cuStreamBeginCapture_v2(self.s, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL))
    }
    /// Finish capture and instantiate.
    pub fn end_capture(&self) -> Result<Graph> {
        let mut g = std::ptr::null_mut();
        cu!(cuStreamEndCapture(self.s, &mut g))?;
        let mut e = std::ptr::null_mut();
        cu!(cuGraphInstantiateWithFlags(&mut e, g, 0))?;
        Ok(Graph { g, e })
    }
    /// Time a closure that enqueues work on this stream; returns milliseconds (synchronises).
    pub fn time<F: FnOnce(&Stream) -> Result<()>>(&self, f: F) -> Result<f32> {
        let (a, b) = (Event::new()?, Event::new()?);
        self.record(&a)?;
        f(self)?;
        self.record(&b)?;
        b.sync()?;
        b.elapsed_since(&a)
    }
}
impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { cuStreamDestroy_v2(self.s) };
    }
}

pub struct Event {
    e: CUevent,
}
unsafe impl Send for Event {}
unsafe impl Sync for Event {}
impl Event {
    pub fn new() -> Result<Event> {
        let mut e = std::ptr::null_mut();
        cu!(cuEventCreate(&mut e, CU_EVENT_DEFAULT))?;
        Ok(Event { e })
    }
    pub fn new_no_timing() -> Result<Event> {
        let mut e = std::ptr::null_mut();
        cu!(cuEventCreate(&mut e, CU_EVENT_DISABLE_TIMING))?;
        Ok(Event { e })
    }
    pub fn sync(&self) -> Result<()> {
        cu!(cuEventSynchronize(self.e))
    }
    pub fn is_done(&self) -> Result<bool> {
        let r = unsafe { cuEventQuery(self.e) };
        if r == CUDA_ERROR_NOT_READY { Ok(false) } else { check(r, "cuEventQuery").map(|_| true) }
    }
    /// Milliseconds between `start` and this event (both recorded and complete).
    pub fn elapsed_since(&self, start: &Event) -> Result<f32> {
        let mut ms = 0.0;
        cu!(cuEventElapsedTime_v2(&mut ms, start.e, self.e))?;
        Ok(ms)
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        unsafe { cuEventDestroy_v2(self.e) };
    }
}

/// An instantiated CUDA graph from stream capture.
pub struct Graph {
    g: CUgraph,
    e: CUgraphExec,
}
unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}
impl Graph {
    pub fn launch(&self, s: &Stream) -> Result<()> {
        cu!(cuGraphLaunch(self.e, s.s))
    }
}
impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            cuGraphExecDestroy(self.e);
            cuGraphDestroy(self.g);
        }
    }
}
