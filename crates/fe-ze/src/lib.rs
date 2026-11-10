//! Thin, owned bindings to Level Zero for the Intel Arc Pro B70 (no SYCL, no oneAPI runtime; libze_loader.so.1 is
//! dlopen'ed, so nothing links against it).
//!
//! [`Ze`] (driver, device, context; shared by `Arc`), [`Queue`] (an in-order immediate command list), USM memory
//! ([`DeviceBuffer`], [`HostBuffer`], [`SharedBuffer`], raw [`DevPtr`]), [`Module`] (SPIR-V or native binary, build
//! flags, build log), [`Kernel`] (group size, arguments incl. dynamic SLM, properties) and [`EventPool`] / [`Event`]
//! with kernel timestamps (device-side timing of single launches).
pub mod ffi;
pub mod graph;
pub use graph::Graph;

use ffi::*;
use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

// ---- errors -----------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ZeError {
    pub code: ze_result_t,
    pub call: &'static str,
    /// Module build log, when a build failed.
    pub log: Option<String>,
}
impl fmt::Display for ZeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} failed: {} (0x{:x})", self.call, result_name(self.code), self.code)?;
        if let Some(l) = &self.log {
            write!(f, "\nbuild log:\n{l}")?;
        }
        Ok(())
    }
}
impl std::error::Error for ZeError {}
pub type Result<T> = std::result::Result<T, ZeError>;

#[inline]
fn check(r: ze_result_t, call: &'static str) -> Result<()> {
    if r == ZE_RESULT_SUCCESS { Ok(()) } else { Err(ZeError { code: r, call, log: None }) }
}
macro_rules! ze {
    ($api:expr, $f:ident ( $($a:expr),* $(,)? )) => {
        check(unsafe { ($api.$f)($($a),*) }, stringify!($f))
    };
}

// ---- driver / device / context ----------------------------------------------------------------

/// Device facts fe-ze reads once.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub core_clock_mhz: u32,
    pub eus_per_subslice: u32,
    pub subslices_per_slice: u32,
    pub slices: u32,
    pub threads_per_eu: u32,
    pub simd_width: u32,
    pub max_alloc: u64,
    pub mem_total: u64,
    /// Timer ticks per second (device and kernel timestamps).
    pub timer_hz: u64,
    pub kernel_ts_valid_bits: u32,
    pub max_group_size: u32,
    pub max_slm: u32,
    pub sg_sizes: Vec<u32>,
}

/// One device, one context. Shared (`Arc<Ze>`) by every object made from it.
pub struct Ze {
    pub api: Api,
    pub driver: ze_driver_handle_t,
    pub device: ze_device_handle_t,
    pub ctx: ze_context_handle_t,
    pub info: DeviceInfo,
}
unsafe impl Send for Ze {}
unsafe impl Sync for Ze {}

impl Ze {
    /// Load Level Zero, take the first GPU of the first driver, create a context.
    pub fn new() -> Result<Arc<Ze>> {
        let api = Api::load().map_err(|e| {
            eprintln!("fe-ze: {e}");
            ZeError { code: 0x78000001, call: "dlopen libze_loader.so.1", log: Some(e) }
        })?;
        ze!(api, zeInit(ZE_INIT_FLAG_GPU_ONLY))?;
        let mut n = 1u32;
        let mut driver = std::ptr::null_mut();
        ze!(api, zeDriverGet(&mut n, &mut driver))?;
        let mut n = 1u32;
        let mut device = std::ptr::null_mut();
        ze!(api, zeDeviceGet(driver, &mut n, &mut device))?;
        let mut ctx = std::ptr::null_mut();
        let cd = ze_context_desc_t { stype: ZE_STRUCTURE_TYPE_CONTEXT_DESC, pNext: std::ptr::null(), flags: 0 };
        ze!(api, zeContextCreate(driver, &cd, &mut ctx))?;

        let mut p: ze_device_properties_t = unsafe { std::mem::zeroed() };
        p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
        ze!(api, zeDeviceGetProperties(device, &mut p))?;
        let mut c: ze_device_compute_properties_t = unsafe { std::mem::zeroed() };
        c.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
        ze!(api, zeDeviceGetComputeProperties(device, &mut c))?;
        let mut nm = 1u32;
        let mut mp: ze_device_memory_properties_t = unsafe { std::mem::zeroed() };
        mp.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
        ze!(api, zeDeviceGetMemoryProperties(device, &mut nm, &mut mp))?;
        let info = DeviceInfo {
            name: unsafe { CStr::from_ptr(p.name.as_ptr()) }.to_string_lossy().into_owned(),
            vendor_id: p.vendorId,
            device_id: p.deviceId,
            core_clock_mhz: p.coreClockRate,
            eus_per_subslice: p.numEUsPerSubslice,
            subslices_per_slice: p.numSubslicesPerSlice,
            slices: p.numSlices,
            threads_per_eu: p.numThreadsPerEU,
            simd_width: p.physicalEUSimdWidth,
            max_alloc: p.maxMemAllocSize,
            mem_total: mp.totalSize,
            timer_hz: p.timerResolution,
            kernel_ts_valid_bits: p.kernelTimestampValidBits,
            max_group_size: c.maxTotalGroupSize,
            max_slm: c.maxSharedLocalMemory,
            sg_sizes: c.subGroupSizes[..c.numSubGroupSizes as usize].to_vec(),
        };
        Ok(Arc::new(Ze { api, driver, device, ctx, info }))
    }

    /// Device ticks -> nanoseconds.
    pub fn ticks_ns(&self, ticks: u64) -> f64 {
        ticks as f64 * 1e9 / self.info.timer_hz as f64
    }
}
impl Drop for Ze {
    fn drop(&mut self) {
        unsafe { (self.api.zeContextDestroy)(self.ctx) };
    }
}

// ---- queue (immediate command list) -----------------------------------------------------------

/// An in-order, asynchronous immediate command list on the compute engine (ordinal 0). Launches and copies are queued
/// in submission order; [`Queue::sync`] waits for all of them.
pub struct Queue {
    ze: Arc<Ze>,
    list: ze_command_list_handle_t,
}
unsafe impl Send for Queue {}

impl Queue {
    pub fn new(ze: &Arc<Ze>) -> Result<Queue> {
        let d = ze_command_queue_desc_t {
            stype: ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
            pNext: std::ptr::null(),
            ordinal: 0,
            index: 0,
            flags: ZE_COMMAND_QUEUE_FLAG_IN_ORDER,
            mode: ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
            priority: 0,
        };
        let mut list = std::ptr::null_mut();
        ze!(ze.api, zeCommandListCreateImmediate(ze.ctx, ze.device, &d, &mut list))?;
        Ok(Queue { ze: ze.clone(), list })
    }
    pub fn ze(&self) -> &Arc<Ze> {
        &self.ze
    }
    pub fn raw(&self) -> ze_command_list_handle_t {
        self.list
    }
    /// Wait until everything queued so far has finished.
    pub fn sync(&self) -> Result<()> {
        ze!(self.ze.api, zeCommandListHostSynchronize(self.list, u64::MAX))
    }
    /// Queue `kernel` over `groups` work-groups (its group size and arguments as currently set).
    pub fn launch(&self, kernel: &Kernel, groups: [u32; 3]) -> Result<()> {
        self.launch_ev(kernel, groups, None)
    }
    /// As [`Queue::launch`], signalling `ev` (from a timestamp pool: the launch's device start/end times).
    pub fn launch_ev(&self, kernel: &Kernel, groups: [u32; 3], ev: Option<&Event>) -> Result<()> {
        let g = ze_group_count_t { groupCountX: groups[0], groupCountY: groups[1], groupCountZ: groups[2] };
        let e = ev.map(|e| e.h).unwrap_or(std::ptr::null_mut());
        ze!(self.ze.api, zeCommandListAppendLaunchKernel(self.list, kernel.h, &g, e, 0, std::ptr::null_mut()))
    }
    /// Queue a copy of `bytes` bytes (any USM pointers or host memory).
    pub unsafe fn copy_raw(&self, dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<()> {
        ze!(self.ze.api, zeCommandListAppendMemoryCopy(self.list, dst, src, bytes, std::ptr::null_mut(), 0, std::ptr::null_mut()))
    }
    /// Host slice -> device, then wait (the host memory may be reused right after).
    pub fn upload<T: Copy>(&self, dst: DevPtr, src: &[T]) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        unsafe { self.copy_raw(dst.0 as *mut c_void, src.as_ptr() as *const c_void, std::mem::size_of_val(src))? };
        self.sync()
    }
    /// Device -> host slice after everything queued before, then wait.
    pub fn download<T: Copy>(&self, dst: &mut [T], src: DevPtr) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        unsafe { self.copy_raw(dst.as_mut_ptr() as *mut c_void, src.0 as *const c_void, std::mem::size_of_val(dst))? };
        self.sync()
    }
    /// Queue a device -> device copy.
    pub fn copy_dtod(&self, dst: DevPtr, src: DevPtr, bytes: usize) -> Result<()> {
        unsafe { self.copy_raw(dst.0 as *mut c_void, src.0 as *const c_void, bytes) }
    }
    /// Queue a fill of `bytes` bytes with a repeated pattern (1, 2, 4, ... 128 bytes).
    pub fn fill(&self, dst: DevPtr, pattern: &[u8], bytes: usize) -> Result<()> {
        ze!(
            self.ze.api,
            zeCommandListAppendMemoryFill(
                self.list,
                dst.0 as *mut c_void,
                pattern.as_ptr() as *const c_void,
                pattern.len(),
                bytes,
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut()
            )
        )
    }
    /// Queue an execution barrier (in-order lists are ordered already; kept for explicit dependencies).
    pub fn barrier(&self) -> Result<()> {
        ze!(self.ze.api, zeCommandListAppendBarrier(self.list, std::ptr::null_mut(), 0, std::ptr::null_mut()))
    }
}
impl Drop for Queue {
    fn drop(&mut self) {
        unsafe {
            (self.ze.api.zeCommandListHostSynchronize)(self.list, 10_000_000_000);
            (self.ze.api.zeCommandListDestroy)(self.list);
        }
    }
}

// ---- memory -----------------------------------------------------------------------------------

/// A raw USM device address; what kernels receive for pointer parameters.
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Kind {
    Device,
    Host,
    Shared,
}

struct Alloc {
    ze: Arc<Ze>,
    p: *mut c_void,
    bytes: usize,
}
impl Alloc {
    fn new(ze: &Arc<Ze>, bytes: usize, kind: Kind) -> Result<Alloc> {
        let mut p = std::ptr::null_mut();
        let dd = ze_device_mem_alloc_desc_t { stype: ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, pNext: std::ptr::null(), flags: 0, ordinal: 0 };
        let hd = ze_host_mem_alloc_desc_t { stype: ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, pNext: std::ptr::null(), flags: 0 };
        let n = bytes.max(1);
        match kind {
            Kind::Device => ze!(ze.api, zeMemAllocDevice(ze.ctx, &dd, n, 256, ze.device, &mut p))?,
            Kind::Host => ze!(ze.api, zeMemAllocHost(ze.ctx, &hd, n, 256, &mut p))?,
            Kind::Shared => ze!(ze.api, zeMemAllocShared(ze.ctx, &dd, &hd, n, 256, ze.device, &mut p))?,
        }
        Ok(Alloc { ze: ze.clone(), p, bytes })
    }
}
impl Drop for Alloc {
    fn drop(&mut self) {
        unsafe { (self.ze.api.zeMemFree)(self.ze.ctx, self.p) };
    }
}

/// Owned device allocation of `len` elements of `T` (USM device memory, 256-byte aligned).
pub struct DeviceBuffer<T> {
    a: Alloc,
    len: usize,
    _t: PhantomData<T>,
}
unsafe impl<T> Send for DeviceBuffer<T> {}
unsafe impl<T> Sync for DeviceBuffer<T> {}
impl<T: Copy> DeviceBuffer<T> {
    pub fn alloc(ze: &Arc<Ze>, len: usize) -> Result<Self> {
        Ok(DeviceBuffer { a: Alloc::new(ze, len * std::mem::size_of::<T>(), Kind::Device)?, len, _t: PhantomData })
    }
    /// Allocate and zero (waits).
    pub fn zeroed(q: &Queue, len: usize) -> Result<Self> {
        let b = Self::alloc(q.ze(), len)?;
        if b.bytes() > 0 {
            q.fill(b.ptr(), &[0u8], b.bytes())?;
            q.sync()?;
        }
        Ok(b)
    }
    pub fn from_slice(q: &Queue, src: &[T]) -> Result<Self> {
        let b = Self::alloc(q.ze(), src.len())?;
        q.upload(b.ptr(), src)?;
        Ok(b)
    }
    #[inline]
    pub fn ptr(&self) -> DevPtr {
        DevPtr(self.a.p as u64)
    }
    /// Device address of element `i`.
    #[inline]
    pub fn at(&self, i: usize) -> DevPtr {
        self.ptr().add::<T>(i)
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
        self.a.bytes
    }
    /// Host -> device into the start of the buffer (waits).
    pub fn copy_from(&self, q: &Queue, src: &[T]) -> Result<()> {
        assert!(src.len() <= self.len);
        q.upload(self.ptr(), src)
    }
    /// Device -> host from the start of the buffer, after all queued work (waits).
    pub fn copy_to(&self, q: &Queue, dst: &mut [T]) -> Result<()> {
        assert!(dst.len() <= self.len);
        q.download(dst, self.ptr())
    }
    pub fn to_vec(&self, q: &Queue) -> Result<Vec<T>> {
        let mut v = Vec::with_capacity(self.len);
        unsafe { v.set_len(self.len) };
        self.copy_to(q, &mut v)?;
        Ok(v)
    }
}

/// Host-resident USM (pinned, device-readable over PCIe; also the fast staging buffer for copies).
pub struct HostBuffer<T> {
    a: Alloc,
    len: usize,
    _t: PhantomData<T>,
}
unsafe impl<T> Send for HostBuffer<T> {}
impl<T: Copy> HostBuffer<T> {
    pub fn alloc(ze: &Arc<Ze>, len: usize) -> Result<Self> {
        Ok(HostBuffer { a: Alloc::new(ze, len * std::mem::size_of::<T>(), Kind::Host)?, len, _t: PhantomData })
    }
    pub fn ptr(&self) -> DevPtr {
        DevPtr(self.a.p as u64)
    }
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.a.p as *const T, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.a.p as *mut T, self.len) }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Shared USM (migrates between host and device on access).
pub struct SharedBuffer<T> {
    a: Alloc,
    len: usize,
    _t: PhantomData<T>,
}
unsafe impl<T> Send for SharedBuffer<T> {}
impl<T: Copy> SharedBuffer<T> {
    pub fn alloc(ze: &Arc<Ze>, len: usize) -> Result<Self> {
        Ok(SharedBuffer { a: Alloc::new(ze, len * std::mem::size_of::<T>(), Kind::Shared)?, len, _t: PhantomData })
    }
    pub fn ptr(&self) -> DevPtr {
        DevPtr(self.a.p as u64)
    }
    /// Host view; only touch it while no kernel using the buffer is in flight.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.a.p as *const T, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.a.p as *mut T, self.len) }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// ---- modules and kernels ----------------------------------------------------------------------

struct ModInner {
    ze: Arc<Ze>,
    h: ze_module_handle_t,
}
unsafe impl Send for ModInner {}
unsafe impl Sync for ModInner {}
impl Drop for ModInner {
    fn drop(&mut self) {
        unsafe { (self.ze.api.zeModuleDestroy)(self.h) };
    }
}

/// A built module (SPIR-V compiled by IGC at creation, or a native binary).
#[derive(Clone)]
pub struct Module {
    m: Arc<ModInner>,
    /// IGC's build log (warnings; empty when clean).
    pub log: String,
}

fn take_log(ze: &Ze, log: ze_module_build_log_handle_t) -> String {
    if log.is_null() {
        return String::new();
    }
    let mut n = 0usize;
    unsafe { (ze.api.zeModuleBuildLogGetString)(log, &mut n, std::ptr::null_mut()) };
    let mut buf = vec![0 as c_char; n.max(1)];
    unsafe {
        (ze.api.zeModuleBuildLogGetString)(log, &mut n, buf.as_mut_ptr());
        (ze.api.zeModuleBuildLogDestroy)(log);
    }
    unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().trim_end().to_string()
}

impl Module {
    /// Build a SPIR-V module. `flags` are IGC build options (e.g. "-cl-intel-256-GRF-per-thread"); the build log is
    /// returned in the error on failure and kept in [`Module::log`] on success.
    pub fn from_spirv(ze: &Arc<Ze>, spv: &[u8], flags: &str) -> Result<Module> {
        Self::create(ze, spv, flags, ZE_MODULE_FORMAT_IL_SPIRV)
    }
    /// Load a device binary (from [`Module::native_binary`] or ocloc).
    pub fn from_native(ze: &Arc<Ze>, bin: &[u8]) -> Result<Module> {
        Self::create(ze, bin, "", ZE_MODULE_FORMAT_NATIVE)
    }
    fn create(ze: &Arc<Ze>, input: &[u8], flags: &str, format: u32) -> Result<Module> {
        let f = CString::new(flags).unwrap();
        let d = ze_module_desc_t {
            stype: ZE_STRUCTURE_TYPE_MODULE_DESC,
            pNext: std::ptr::null(),
            format,
            inputSize: input.len(),
            pInputModule: input.as_ptr(),
            pBuildFlags: f.as_ptr(),
            pConstants: std::ptr::null(),
        };
        let mut h = std::ptr::null_mut();
        let mut log = std::ptr::null_mut();
        let r = unsafe { (ze.api.zeModuleCreate)(ze.ctx, ze.device, &d, &mut h, &mut log) };
        let text = take_log(ze, log);
        if r != ZE_RESULT_SUCCESS {
            return Err(ZeError { code: r, call: "zeModuleCreate", log: Some(text) });
        }
        Ok(Module { m: Arc::new(ModInner { ze: ze.clone(), h }), log: text })
    }
    pub fn kernel_names(&self) -> Result<Vec<String>> {
        let api = &self.m.ze.api;
        let mut n = 0u32;
        ze!(api, zeModuleGetKernelNames(self.m.h, &mut n, std::ptr::null_mut()))?;
        let mut v = vec![std::ptr::null::<c_char>(); n as usize];
        ze!(api, zeModuleGetKernelNames(self.m.h, &mut n, v.as_mut_ptr()))?;
        Ok(v.iter().map(|p| unsafe { CStr::from_ptr(*p) }.to_string_lossy().into_owned()).collect())
    }
    /// The compiled device binary (zebin: `ocloc disasm` shows its ISA).
    pub fn native_binary(&self) -> Result<Vec<u8>> {
        let api = &self.m.ze.api;
        let mut n = 0usize;
        ze!(api, zeModuleGetNativeBinary(self.m.h, &mut n, std::ptr::null_mut()))?;
        let mut v = vec![0u8; n];
        ze!(api, zeModuleGetNativeBinary(self.m.h, &mut n, v.as_mut_ptr()))?;
        Ok(v)
    }
    /// A kernel of this module with the given work-group size.
    pub fn kernel(&self, name: &str, group: [u32; 3]) -> Result<Kernel> {
        let n = CString::new(name).unwrap();
        let d = ze_kernel_desc_t { stype: ZE_STRUCTURE_TYPE_KERNEL_DESC, pNext: std::ptr::null(), flags: 0, pKernelName: n.as_ptr() };
        let mut h = std::ptr::null_mut();
        ze!(self.m.ze.api, zeKernelCreate(self.m.h, &d, &mut h)).map_err(|mut e| {
            e.log = Some(format!("kernel `{name}`"));
            e
        })?;
        let k = Kernel { m: self.m.clone(), h, name: name.to_string(), group };
        ze!(self.m.ze.api, zeKernelSetGroupSize(h, group[0], group[1], group[2]))?;
        Ok(k)
    }
}

/// What the compiler made of a kernel.
#[derive(Copy, Clone, Debug)]
pub struct KernelProps {
    pub num_args: u32,
    pub required_group: [u32; 3],
    pub required_sg_size: u32,
    pub max_sg_size: u32,
    pub max_num_sg: u32,
    /// Static SLM bytes a work-group.
    pub slm_bytes: u32,
    /// Private (stack) bytes a work-item.
    pub private_bytes: u32,
    /// Register spill bytes (non-zero = the kernel spilled).
    pub spill_bytes: u32,
}

/// A kernel argument value.
pub trait KernelArg {
    /// (bytes, pointer to the value; null = dynamic SLM of `bytes`).
    fn raw(&self) -> (usize, *const c_void);
}
impl KernelArg for DevPtr {
    fn raw(&self) -> (usize, *const c_void) {
        (8, self as *const DevPtr as *const c_void)
    }
}
macro_rules! scalar_arg {
    ($($t:ty),*) => { $(impl KernelArg for $t { fn raw(&self) -> (usize, *const c_void) { (std::mem::size_of::<$t>(), self as *const $t as *const c_void) } })* };
}
scalar_arg!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);
/// Dynamic SLM argument: a `__local` pointer parameter (`kxcommon::attr::local_arg`) of this many bytes.
#[derive(Copy, Clone, Debug)]
pub struct Local(pub usize);
impl KernelArg for Local {
    fn raw(&self) -> (usize, *const c_void) {
        (self.0, std::ptr::null())
    }
}

/// A kernel object: work-group size and arguments are its state (set them, then launch; launches copy them).
pub struct Kernel {
    m: Arc<ModInner>,
    h: ze_kernel_handle_t,
    pub name: String,
    pub group: [u32; 3],
}
unsafe impl Send for Kernel {}

impl Kernel {
    pub fn set_group_size(&mut self, group: [u32; 3]) -> Result<()> {
        ze!(self.m.ze.api, zeKernelSetGroupSize(self.h, group[0], group[1], group[2]))?;
        self.group = group;
        Ok(())
    }
    pub fn arg(&self, index: u32, v: &dyn KernelArg) -> Result<&Self> {
        let (n, p) = v.raw();
        ze!(self.m.ze.api, zeKernelSetArgumentValue(self.h, index, n, p)).map_err(|mut e| {
            e.log = Some(format!("kernel `{}` argument {index} ({n} bytes)", self.name));
            e
        })?;
        Ok(self)
    }
    /// Set arguments 0.. in order.
    pub fn args(&self, vals: &[&dyn KernelArg]) -> Result<&Self> {
        for (i, v) in vals.iter().enumerate() {
            self.arg(i as u32, *v)?;
        }
        Ok(self)
    }
    pub fn props(&self) -> Result<KernelProps> {
        let mut p: ze_kernel_properties_t = unsafe { std::mem::zeroed() };
        p.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
        ze!(self.m.ze.api, zeKernelGetProperties(self.h, &mut p))?;
        Ok(KernelProps {
            num_args: p.numKernelArgs,
            required_group: [p.requiredGroupSizeX, p.requiredGroupSizeY, p.requiredGroupSizeZ],
            required_sg_size: p.requiredSubgroupSize,
            max_sg_size: p.maxSubgroupSize,
            max_num_sg: p.maxNumSubgroups,
            slm_bytes: p.localMemSize,
            private_bytes: p.privateMemSize,
            spill_bytes: p.spillMemSize,
        })
    }
    pub fn raw(&self) -> ze_kernel_handle_t {
        self.h
    }
}
impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { (self.m.ze.api.zeKernelDestroy)(self.h) };
    }
}

// ---- events and kernel timestamps -------------------------------------------------------------

struct PoolInner {
    ze: Arc<Ze>,
    h: ze_event_pool_handle_t,
}
impl Drop for PoolInner {
    fn drop(&mut self) {
        unsafe { (self.ze.api.zeEventPoolDestroy)(self.h) };
    }
}

/// A pool of host-visible events that record kernel timestamps.
pub struct EventPool {
    p: Arc<PoolInner>,
    count: u32,
}
unsafe impl Send for EventPool {}

impl EventPool {
    pub fn new(ze: &Arc<Ze>, count: u32) -> Result<EventPool> {
        let d = ze_event_pool_desc_t {
            stype: ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
            pNext: std::ptr::null(),
            flags: ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP,
            count,
        };
        let mut h = std::ptr::null_mut();
        let mut dev = ze.device;
        ze!(ze.api, zeEventPoolCreate(ze.ctx, &d, 1, &mut dev, &mut h))?;
        Ok(EventPool { p: Arc::new(PoolInner { ze: ze.clone(), h }), count })
    }
    pub fn event(&self, index: u32) -> Result<Event> {
        assert!(index < self.count);
        let d = ze_event_desc_t {
            stype: ZE_STRUCTURE_TYPE_EVENT_DESC,
            pNext: std::ptr::null(),
            index,
            signal: ZE_EVENT_SCOPE_FLAG_HOST,
            wait: ZE_EVENT_SCOPE_FLAG_HOST,
        };
        let mut h = std::ptr::null_mut();
        ze!(self.p.ze.api, zeEventCreate(self.p.h, &d, &mut h))?;
        Ok(Event { p: self.p.clone(), h })
    }
}

/// One event: signalled by a launch, then read for the kernel's device timestamps.
pub struct Event {
    p: Arc<PoolInner>,
    h: ze_event_handle_t,
}
unsafe impl Send for Event {}

impl Event {
    pub fn sync(&self) -> Result<()> {
        ze!(self.p.ze.api, zeEventHostSynchronize(self.h, u64::MAX))
    }
    pub fn reset(&self) -> Result<()> {
        ze!(self.p.ze.api, zeEventHostReset(self.h))
    }
    /// Device execution time of the launch that signalled this event, ns (waits for it).
    pub fn kernel_ns(&self) -> Result<f64> {
        self.sync()?;
        let mut r = ze_kernel_timestamp_result_t::default();
        ze!(self.p.ze.api, zeEventQueryKernelTimestamp(self.h, &mut r))?;
        let ze = &self.p.ze;
        let bits = ze.info.kernel_ts_valid_bits.min(64);
        let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
        let t = r.global.kernelEnd.wrapping_sub(r.global.kernelStart) & mask;
        Ok(ze.ticks_ns(t))
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        unsafe { (self.p.ze.api.zeEventDestroy)(self.h) };
    }
}

/// Host wall-clock nanoseconds (monotonic) for host-side timing of launch sequences.
pub fn now_ns() -> u64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static T0: OnceLock<Instant> = OnceLock::new();
    T0.get_or_init(Instant::now).elapsed().as_nanos() as u64
}
