//! Programmatic dependent launch (cuLaunchKernelEx + PROGRAMMATIC_STREAM_SERIALIZATION): a kernel may start
//! while the previous one on the stream finishes; k-exl3 kernels prefetch weights first and
//! `griddepcontrol.wait` before touching anything the previous kernel wrote. Captured into CUDA graphs as
//! programmatic edges. FE_EXL3_PDL=0 turns it off (plain launches).

use std::ffi::{c_uint, c_void};
use std::sync::OnceLock;

use fe_cuda::{Function, KernelArgs, Stream};

use crate::{Error, Result};

#[repr(C)]
struct LaunchAttribute {
    id: u32,
    pad: [u8; 4],
    value: [u8; 64],
}

#[repr(C)]
struct LaunchConfig {
    grid: [c_uint; 3],
    block: [c_uint; 3],
    smem: c_uint,
    stream: *mut c_void,
    attrs: *mut LaunchAttribute,
    num_attrs: c_uint,
}

const PROGRAMMATIC_STREAM_SERIALIZATION: u32 = 6;

#[link(name = "cuda")]
unsafe extern "C" {
    fn cuLaunchKernelEx(config: *const LaunchConfig, f: *mut c_void, params: *mut *mut c_void, extra: *mut *mut c_void) -> i32;
}

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FE_EXL3_PDL").map(|v| v != "0").unwrap_or(true))
}

/// Launch `f` on `s`, programmatic (PDL) when enabled.
///
/// # Safety
/// As `Stream::launch`: `args` must match the kernel's parameters exactly.
pub unsafe fn launch<A: KernelArgs>(s: &Stream, f: &Function, grid: (u32, u32), block: u32, smem: u32, args: A) -> Result<()> {
    if !enabled() {
        return Ok(unsafe { s.launch(f, grid, block, smem, args)? });
    }
    let mut attr = LaunchAttribute { id: PROGRAMMATIC_STREAM_SERIALIZATION, pad: [0; 4], value: [0; 64] };
    attr.value[..4].copy_from_slice(&1i32.to_le_bytes());
    let cfg = LaunchConfig {
        grid: [grid.0, grid.1, 1],
        block: [block, 1, 1],
        smem,
        stream: s.raw() as *mut c_void,
        attrs: &mut attr,
        num_attrs: 1,
    };
    let r = args.with_ptrs(|p| unsafe { cuLaunchKernelEx(&cfg, f.raw() as *mut c_void, p, std::ptr::null_mut()) });
    if r != 0 {
        return Err(Error::msg(format!("cuLaunchKernelEx({}) failed: CUDA error {r}", f.name)));
    }
    Ok(())
}
