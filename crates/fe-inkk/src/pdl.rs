//! Programmatic dependent launch (cuLaunchKernelEx + PROGRAMMATIC_STREAM_SERIALIZATION). Every k-ink kernel that can be
//! launched this way (all but the load-time repack / quantize ones) waits with `griddepcontrol.wait` before reading
//! anything the previous kernel wrote, so a PDL launch lets its grid start (and prefetch weights) while the previous grid
//! drains. Captured into CUDA graphs as programmatic edges.
use std::ffi::{c_uint, c_void};

use anyhow::{bail, Result};
use fe_cuda::{Dim3, Function, KernelArgs, Stream};

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
    true
}

/// Launch `f` on `s`, programmatic (PDL) when enabled.
///
/// # Safety
/// As `Stream::launch`; the kernel must call `griddepcontrol.wait` before consuming the previous kernel's output.
pub unsafe fn launch<A: KernelArgs>(s: &Stream, f: &Function, grid: impl Into<Dim3>, block: u32, smem: u32, args: A) -> Result<()> {
    if !enabled() {
        return Ok(unsafe { s.launch(f, grid, block, smem, args)? });
    }
    let g: Dim3 = grid.into();
    let mut attr = LaunchAttribute { id: PROGRAMMATIC_STREAM_SERIALIZATION, pad: [0; 4], value: [0; 64] };
    attr.value[..4].copy_from_slice(&1i32.to_le_bytes());
    let cfg = LaunchConfig {
        grid: [g.x, g.y, g.z],
        block: [block, 1, 1],
        smem,
        stream: s.raw() as *mut c_void,
        attrs: &mut attr,
        num_attrs: 1,
    };
    let r = args.with_ptrs(|p| unsafe { cuLaunchKernelEx(&cfg, f.raw() as *mut c_void, p, std::ptr::null_mut()) });
    if r != 0 {
        bail!("cuLaunchKernelEx({}) failed: CUDA error {r}", f.name);
    }
    Ok(())
}
