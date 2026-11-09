//! Raw libcuda (driver API) declarations: only what ferrite uses. Names and ABI follow cuda.h (CUDA 13).
#![allow(non_camel_case_types, non_snake_case, dead_code)]
use std::ffi::{c_char, c_int, c_uint, c_void};

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CUDA_ERROR_NOT_READY: CUresult = 600;

pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: c_int = 97;
pub const CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: c_int = 38;

pub const CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 0;
pub const CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: c_int = 1;
pub const CU_FUNC_ATTRIBUTE_NUM_REGS: c_int = 4;
pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;

pub const CU_STREAM_DEFAULT: c_uint = 0;
pub const CU_STREAM_NON_BLOCKING: c_uint = 1;
pub const CU_EVENT_DEFAULT: c_uint = 0;
pub const CU_EVENT_DISABLE_TIMING: c_uint = 2;
pub const CU_STREAM_CAPTURE_MODE_GLOBAL: c_int = 0;
pub const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: c_int = 1;
pub const CU_STREAM_CAPTURE_MODE_RELAXED: c_int = 2;
pub const CU_MEMHOSTREGISTER_PORTABLE: c_uint = 1;
pub const CU_MEMHOSTREGISTER_READ_ONLY: c_uint = 8;

#[link(name = "cuda")]
unsafe extern "C" {
    pub fn cuInit(flags: c_uint) -> CUresult;
    pub fn cuGetErrorName(e: CUresult, s: *mut *const c_char) -> CUresult;
    pub fn cuGetErrorString(e: CUresult, s: *mut *const c_char) -> CUresult;
    pub fn cuDriverGetVersion(v: *mut c_int) -> CUresult;

    pub fn cuDeviceGet(d: *mut CUdevice, ordinal: c_int) -> CUresult;
    pub fn cuDeviceGetName(name: *mut c_char, len: c_int, d: CUdevice) -> CUresult;
    pub fn cuDeviceGetAttribute(v: *mut c_int, attr: c_int, d: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, d: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxRelease_v2(d: CUdevice) -> CUresult;
    pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    pub fn cuCtxSynchronize() -> CUresult;

    pub fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;
    pub fn cuMemAlloc_v2(p: *mut CUdeviceptr, bytes: usize) -> CUresult;
    pub fn cuMemFree_v2(p: CUdeviceptr) -> CUresult;
    pub fn cuMemAllocHost_v2(p: *mut *mut c_void, bytes: usize) -> CUresult;
    pub fn cuMemFreeHost(p: *mut c_void) -> CUresult;
    pub fn cuMemHostRegister_v2(p: *mut c_void, bytes: usize, flags: c_uint) -> CUresult;
    pub fn cuMemHostUnregister(p: *mut c_void) -> CUresult;
    pub fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult;
    pub fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult;
    pub fn cuMemcpyHtoDAsync_v2(dst: CUdeviceptr, src: *const c_void, bytes: usize, s: CUstream) -> CUresult;
    pub fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: CUdeviceptr, bytes: usize, s: CUstream) -> CUresult;
    pub fn cuMemcpyDtoDAsync_v2(dst: CUdeviceptr, src: CUdeviceptr, bytes: usize, s: CUstream) -> CUresult;
    pub fn cuMemsetD8Async(dst: CUdeviceptr, v: u8, n: usize, s: CUstream) -> CUresult;
    pub fn cuMemsetD32Async(dst: CUdeviceptr, v: c_uint, n: usize, s: CUstream) -> CUresult;

    pub fn cuModuleLoadData(m: *mut CUmodule, image: *const c_void) -> CUresult;
    pub fn cuModuleUnload(m: CUmodule) -> CUresult;
    pub fn cuModuleGetFunction(f: *mut CUfunction, m: CUmodule, name: *const c_char) -> CUresult;
    pub fn cuFuncGetAttribute(v: *mut c_int, attr: c_int, f: CUfunction) -> CUresult;
    pub fn cuFuncSetAttribute(f: CUfunction, attr: c_int, v: c_int) -> CUresult;
    pub fn cuLaunchKernel(f: CUfunction, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint,
        smem: c_uint, s: CUstream, params: *mut *mut c_void, extra: *mut *mut c_void) -> CUresult;

    pub fn cuStreamCreate(s: *mut CUstream, flags: c_uint) -> CUresult;
    pub fn cuStreamDestroy_v2(s: CUstream) -> CUresult;
    pub fn cuStreamSynchronize(s: CUstream) -> CUresult;
    pub fn cuStreamWaitEvent(s: CUstream, e: CUevent, flags: c_uint) -> CUresult;
    pub fn cuStreamBeginCapture_v2(s: CUstream, mode: c_int) -> CUresult;
    pub fn cuStreamEndCapture(s: CUstream, g: *mut CUgraph) -> CUresult;

    pub fn cuEventCreate(e: *mut CUevent, flags: c_uint) -> CUresult;
    pub fn cuEventDestroy_v2(e: CUevent) -> CUresult;
    pub fn cuEventRecord(e: CUevent, s: CUstream) -> CUresult;
    pub fn cuEventSynchronize(e: CUevent) -> CUresult;
    pub fn cuEventQuery(e: CUevent) -> CUresult;
    pub fn cuEventElapsedTime_v2(ms: *mut f32, start: CUevent, end: CUevent) -> CUresult;

    pub fn cuGraphInstantiateWithFlags(e: *mut CUgraphExec, g: CUgraph, flags: u64) -> CUresult;
    pub fn cuGraphLaunch(e: CUgraphExec, s: CUstream) -> CUresult;
    pub fn cuGraphExecDestroy(e: CUgraphExec) -> CUresult;
    pub fn cuGraphDestroy(g: CUgraph) -> CUresult;
}
