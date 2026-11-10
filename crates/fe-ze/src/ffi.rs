//! Raw Level Zero (oneAPI L0 1.x core) declarations, as ze_api.h (loader 1.32) defines them: only what ferrite uses.
//! The entry points are resolved at runtime from libze_loader.so.1 (dlopen), so nothing links against Level Zero.
#![allow(non_camel_case_types, non_snake_case, dead_code)]
use std::ffi::{c_char, c_int, c_void};

pub type ze_result_t = u32;
pub type ze_driver_handle_t = *mut c_void;
pub type ze_device_handle_t = *mut c_void;
pub type ze_context_handle_t = *mut c_void;
pub type ze_command_list_handle_t = *mut c_void;
pub type ze_module_handle_t = *mut c_void;
pub type ze_module_build_log_handle_t = *mut c_void;
pub type ze_kernel_handle_t = *mut c_void;
pub type ze_event_pool_handle_t = *mut c_void;
pub type ze_event_handle_t = *mut c_void;

pub const ZE_RESULT_SUCCESS: ze_result_t = 0;
pub const ZE_RESULT_NOT_READY: ze_result_t = 1;
pub const ZE_INIT_FLAG_GPU_ONLY: u32 = 1;

pub const ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES: u32 = 0x4;
pub const ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES: u32 = 0x7;
pub const ZE_STRUCTURE_TYPE_CONTEXT_DESC: u32 = 0xd;
pub const ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC: u32 = 0xe;
pub const ZE_STRUCTURE_TYPE_EVENT_POOL_DESC: u32 = 0x10;
pub const ZE_STRUCTURE_TYPE_EVENT_DESC: u32 = 0x11;
pub const ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC: u32 = 0x15;
pub const ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC: u32 = 0x16;
pub const ZE_STRUCTURE_TYPE_MODULE_DESC: u32 = 0x1b;
pub const ZE_STRUCTURE_TYPE_KERNEL_DESC: u32 = 0x1d;
pub const ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES: u32 = 0x1e;
pub const ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2: u32 = 0x0002_0006;

pub const ZE_COMMAND_QUEUE_FLAG_IN_ORDER: u32 = 1 << 1;
pub const ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS: u32 = 2;
pub const ZE_MODULE_FORMAT_IL_SPIRV: u32 = 0;
pub const ZE_MODULE_FORMAT_NATIVE: u32 = 1;
pub const ZE_EVENT_POOL_FLAG_HOST_VISIBLE: u32 = 1 << 0;
pub const ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP: u32 = 1 << 2;
pub const ZE_EVENT_SCOPE_FLAG_HOST: u32 = 1 << 2;
pub const ZE_MAX_DEVICE_NAME: usize = 256;
pub const ZE_SUBGROUPSIZE_COUNT: usize = 8;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct ze_device_properties_t {
    pub stype: u32,
    pub pNext: *mut c_void,
    pub type_: u32,
    pub vendorId: u32,
    pub deviceId: u32,
    pub flags: u32,
    pub subdeviceId: u32,
    pub coreClockRate: u32,
    pub maxMemAllocSize: u64,
    pub maxHardwareContexts: u32,
    pub maxCommandQueuePriority: u32,
    pub numThreadsPerEU: u32,
    pub physicalEUSimdWidth: u32,
    pub numEUsPerSubslice: u32,
    pub numSubslicesPerSlice: u32,
    pub numSlices: u32,
    pub timerResolution: u64,
    pub timestampValidBits: u32,
    pub kernelTimestampValidBits: u32,
    pub uuid: [u8; 16],
    pub name: [c_char; ZE_MAX_DEVICE_NAME],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct ze_device_compute_properties_t {
    pub stype: u32,
    pub pNext: *mut c_void,
    pub maxTotalGroupSize: u32,
    pub maxGroupSizeX: u32,
    pub maxGroupSizeY: u32,
    pub maxGroupSizeZ: u32,
    pub maxGroupCountX: u32,
    pub maxGroupCountY: u32,
    pub maxGroupCountZ: u32,
    pub maxSharedLocalMemory: u32,
    pub numSubGroupSizes: u32,
    pub subGroupSizes: [u32; ZE_SUBGROUPSIZE_COUNT],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct ze_device_memory_properties_t {
    pub stype: u32,
    pub pNext: *mut c_void,
    pub flags: u32,
    pub maxClockRate: u32,
    pub maxBusWidth: u32,
    pub totalSize: u64,
    pub name: [c_char; ZE_MAX_DEVICE_NAME],
}

#[repr(C)]
pub struct ze_context_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub flags: u32,
}

#[repr(C)]
pub struct ze_command_queue_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub ordinal: u32,
    pub index: u32,
    pub flags: u32,
    pub mode: u32,
    pub priority: u32,
}

#[repr(C)]
pub struct ze_device_mem_alloc_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub ordinal: u32,
}

#[repr(C)]
pub struct ze_host_mem_alloc_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub flags: u32,
}

#[repr(C)]
pub struct ze_module_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub format: u32,
    pub inputSize: usize,
    pub pInputModule: *const u8,
    pub pBuildFlags: *const c_char,
    pub pConstants: *const c_void,
}

#[repr(C)]
pub struct ze_kernel_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub pKernelName: *const c_char,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct ze_group_count_t {
    pub groupCountX: u32,
    pub groupCountY: u32,
    pub groupCountZ: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct ze_kernel_properties_t {
    pub stype: u32,
    pub pNext: *mut c_void,
    pub numKernelArgs: u32,
    pub requiredGroupSizeX: u32,
    pub requiredGroupSizeY: u32,
    pub requiredGroupSizeZ: u32,
    pub requiredNumSubGroups: u32,
    pub requiredSubgroupSize: u32,
    pub maxSubgroupSize: u32,
    pub maxNumSubgroups: u32,
    pub localMemSize: u32,
    pub privateMemSize: u32,
    pub spillMemSize: u32,
    pub kid: [u8; 16],
    pub mid: [u8; 16],
}

#[repr(C)]
pub struct ze_event_pool_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub count: u32,
}

#[repr(C)]
pub struct ze_event_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub index: u32,
    pub signal: u32,
    pub wait: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct ze_kernel_timestamp_data_t {
    pub kernelStart: u64,
    pub kernelEnd: u64,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct ze_kernel_timestamp_result_t {
    pub global: ze_kernel_timestamp_data_t,
    pub context: ze_kernel_timestamp_data_t,
}

type Ev = ze_event_handle_t;
type EvP = *mut ze_event_handle_t;

macro_rules! api {
    ($($name:ident : fn($($a:ty),*);)*) => {
        /// Level Zero entry points, resolved from libze_loader.so.1.
        #[allow(non_snake_case)]
        pub struct Api { $(pub $name: unsafe extern "C" fn($($a),*) -> ze_result_t,)* _lib: *mut c_void }
        unsafe impl Send for Api {}
        unsafe impl Sync for Api {}
        impl Api {
            pub fn load() -> Result<Api, String> {
                let lib = unsafe { dlopen(c"libze_loader.so.1".as_ptr(), RTLD_NOW | RTLD_LOCAL) };
                if lib.is_null() {
                    return Err(format!("dlopen libze_loader.so.1: {}", dl_error()));
                }
                Ok(Api {
                    $($name: unsafe {
                        let p = dlsym(lib, concat!(stringify!($name), "\0").as_ptr() as *const c_char);
                        if p.is_null() {
                            return Err(format!("dlsym {}: {}", stringify!($name), dl_error()));
                        }
                        std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($a),*) -> ze_result_t>(p)
                    },)*
                    _lib: lib,
                })
            }
        }
    };
}

// ---- recorded (regular) command lists and the mutable-command-list extension (ZE_experimental_mutable_command_list) -------
pub const ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC: u32 = 0xf;
pub const ZE_COMMAND_LIST_FLAG_IN_ORDER: u32 = 1 << 3;
pub const ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_DESC: u32 = 0x0002_0016;
pub const ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_ID_EXP_DESC: u32 = 0x0002_0017;
pub const ZE_STRUCTURE_TYPE_MUTABLE_COMMANDS_EXP_DESC: u32 = 0x0002_0018;
pub const ZE_STRUCTURE_TYPE_MUTABLE_KERNEL_ARGUMENT_EXP_DESC: u32 = 0x0002_0019;
pub const ZE_STRUCTURE_TYPE_MUTABLE_GROUP_COUNT_EXP_DESC: u32 = 0x0002_001A;
pub const ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_ARGUMENTS: u32 = 1 << 0;
pub const ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_COUNT: u32 = 1 << 1;

#[repr(C)]
pub struct ze_command_list_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub commandQueueGroupOrdinal: u32,
    pub flags: u32,
}
#[repr(C)]
pub struct ze_mutable_command_list_exp_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub flags: u32,
}
#[repr(C)]
pub struct ze_mutable_command_id_exp_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub flags: u32,
}
#[repr(C)]
pub struct ze_mutable_commands_exp_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub flags: u32,
}
#[repr(C)]
pub struct ze_mutable_kernel_argument_exp_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub commandId: u64,
    pub argIndex: u32,
    pub argSize: usize,
    pub pArgValue: *const c_void,
}
#[repr(C)]
pub struct ze_mutable_group_count_exp_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub commandId: u64,
    pub pGroupCount: *const ze_group_count_t,
}

api! {
    zeInit: fn(u32);
    zeDriverGet: fn(*mut u32, *mut ze_driver_handle_t);
    zeDeviceGet: fn(ze_driver_handle_t, *mut u32, *mut ze_device_handle_t);
    zeDeviceGetProperties: fn(ze_device_handle_t, *mut ze_device_properties_t);
    zeDeviceGetComputeProperties: fn(ze_device_handle_t, *mut ze_device_compute_properties_t);
    zeDeviceGetMemoryProperties: fn(ze_device_handle_t, *mut u32, *mut ze_device_memory_properties_t);
    zeContextCreate: fn(ze_driver_handle_t, *const ze_context_desc_t, *mut ze_context_handle_t);
    zeContextDestroy: fn(ze_context_handle_t);
    zeCommandListCreateImmediate: fn(ze_context_handle_t, ze_device_handle_t, *const ze_command_queue_desc_t, *mut ze_command_list_handle_t);
    zeCommandListDestroy: fn(ze_command_list_handle_t);
    zeCommandListHostSynchronize: fn(ze_command_list_handle_t, u64);
    zeCommandListAppendBarrier: fn(ze_command_list_handle_t, Ev, u32, EvP);
    zeCommandListAppendMemoryCopy: fn(ze_command_list_handle_t, *mut c_void, *const c_void, usize, Ev, u32, EvP);
    zeCommandListAppendMemoryFill: fn(ze_command_list_handle_t, *mut c_void, *const c_void, usize, usize, Ev, u32, EvP);
    zeCommandListAppendLaunchKernel: fn(ze_command_list_handle_t, ze_kernel_handle_t, *const ze_group_count_t, Ev, u32, EvP);
    zeMemAllocDevice: fn(ze_context_handle_t, *const ze_device_mem_alloc_desc_t, usize, usize, ze_device_handle_t, *mut *mut c_void);
    zeMemAllocHost: fn(ze_context_handle_t, *const ze_host_mem_alloc_desc_t, usize, usize, *mut *mut c_void);
    zeMemAllocShared: fn(ze_context_handle_t, *const ze_device_mem_alloc_desc_t, *const ze_host_mem_alloc_desc_t, usize, usize, ze_device_handle_t, *mut *mut c_void);
    zeMemFree: fn(ze_context_handle_t, *mut c_void);
    zeModuleCreate: fn(ze_context_handle_t, ze_device_handle_t, *const ze_module_desc_t, *mut ze_module_handle_t, *mut ze_module_build_log_handle_t);
    zeModuleDestroy: fn(ze_module_handle_t);
    zeModuleBuildLogGetString: fn(ze_module_build_log_handle_t, *mut usize, *mut c_char);
    zeModuleBuildLogDestroy: fn(ze_module_build_log_handle_t);
    zeModuleGetKernelNames: fn(ze_module_handle_t, *mut u32, *mut *const c_char);
    zeModuleGetNativeBinary: fn(ze_module_handle_t, *mut usize, *mut u8);
    zeKernelCreate: fn(ze_module_handle_t, *const ze_kernel_desc_t, *mut ze_kernel_handle_t);
    zeKernelDestroy: fn(ze_kernel_handle_t);
    zeKernelSetGroupSize: fn(ze_kernel_handle_t, u32, u32, u32);
    zeKernelSuggestGroupSize: fn(ze_kernel_handle_t, u32, u32, u32, *mut u32, *mut u32, *mut u32);
    zeKernelSetArgumentValue: fn(ze_kernel_handle_t, u32, usize, *const c_void);
    zeKernelGetProperties: fn(ze_kernel_handle_t, *mut ze_kernel_properties_t);
    zeEventPoolCreate: fn(ze_context_handle_t, *const ze_event_pool_desc_t, u32, *mut ze_device_handle_t, *mut ze_event_pool_handle_t);
    zeEventPoolDestroy: fn(ze_event_pool_handle_t);
    zeEventCreate: fn(ze_event_pool_handle_t, *const ze_event_desc_t, *mut ze_event_handle_t);
    zeEventDestroy: fn(ze_event_handle_t);
    zeEventHostSynchronize: fn(ze_event_handle_t, u64);
    zeEventHostReset: fn(ze_event_handle_t);
    zeEventQueryStatus: fn(ze_event_handle_t);
    zeEventQueryKernelTimestamp: fn(ze_event_handle_t, *mut ze_kernel_timestamp_result_t);
    zeCommandListCreate: fn(ze_context_handle_t, ze_device_handle_t, *const ze_command_list_desc_t, *mut ze_command_list_handle_t);
    zeCommandListClose: fn(ze_command_list_handle_t);
    zeCommandListReset: fn(ze_command_list_handle_t);
    zeCommandListImmediateAppendCommandListsExp: fn(ze_command_list_handle_t, u32, *mut ze_command_list_handle_t, Ev, u32, EvP);
    zeCommandListGetNextCommandIdExp: fn(ze_command_list_handle_t, *const ze_mutable_command_id_exp_desc_t, *mut u64);
    zeCommandListUpdateMutableCommandsExp: fn(ze_command_list_handle_t, *const ze_mutable_commands_exp_desc_t);
}

pub const RTLD_NOW: c_int = 2;
pub const RTLD_LOCAL: c_int = 0;
#[link(name = "dl")]
unsafe extern "C" {
    pub fn dlopen(file: *const c_char, mode: c_int) -> *mut c_void;
    pub fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    pub fn dlerror() -> *mut c_char;
}
fn dl_error() -> String {
    let e = unsafe { dlerror() };
    if e.is_null() { "?".into() } else { unsafe { std::ffi::CStr::from_ptr(e) }.to_string_lossy().into_owned() }
}

/// ze_result_t name (the common ones).
pub fn result_name(r: ze_result_t) -> &'static str {
    match r {
        0 => "ZE_RESULT_SUCCESS",
        1 => "ZE_RESULT_NOT_READY",
        0x70000001 => "ZE_RESULT_ERROR_DEVICE_LOST",
        0x70000002 => "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY",
        0x70000003 => "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY",
        0x70000004 => "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE",
        0x70000005 => "ZE_RESULT_ERROR_MODULE_LINK_FAILURE",
        0x70000006 => "ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET",
        0x70000007 => "ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE",
        0x78000001 => "ZE_RESULT_ERROR_UNINITIALIZED",
        0x78000002 => "ZE_RESULT_ERROR_UNSUPPORTED_VERSION",
        0x78000003 => "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE",
        0x78000004 => "ZE_RESULT_ERROR_INVALID_ARGUMENT",
        0x78000005 => "ZE_RESULT_ERROR_INVALID_NULL_HANDLE",
        0x78000006 => "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE",
        0x78000007 => "ZE_RESULT_ERROR_INVALID_NULL_POINTER",
        0x78000008 => "ZE_RESULT_ERROR_INVALID_SIZE",
        0x78000009 => "ZE_RESULT_ERROR_UNSUPPORTED_SIZE",
        0x7800000a => "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT",
        0x7800000b => "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT",
        0x7800000c => "ZE_RESULT_ERROR_INVALID_ENUMERATION",
        0x7800000d => "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION",
        0x7800000e => "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT",
        0x7800000f => "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY",
        0x78000010 => "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME",
        0x78000011 => "ZE_RESULT_ERROR_INVALID_KERNEL_NAME",
        0x78000012 => "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME",
        0x78000013 => "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION",
        0x78000014 => "ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION",
        0x78000015 => "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX",
        0x78000016 => "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE",
        0x78000017 => "ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE",
        0x78000018 => "ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED",
        0x78000019 => "ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE",
        0x7800001a => "ZE_RESULT_ERROR_OVERLAPPING_REGIONS",
        0x7fffffff => "ZE_RESULT_ERROR_UNKNOWN",
        _ => "ZE_RESULT_?",
    }
}
