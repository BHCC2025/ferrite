# How to write a kernel in ferrite

GPU kernels are Rust (`#![no_std]`, nightly, target `nvptx64-nvidia-cuda`, `target-cpu=sm_121a`). Each
`kernels/k-*` crate is one PTX module, assembled by `ptxas -arch=sm_121a -O3` into a cubin that a host crate
embeds with `include_bytes!`. kernels/ is its own cargo workspace (members `kcommon` + glob `k-*`), selected
toolchain nightly via `rust-toolchain.toml`, target and `build-std=core` via `.cargo/config.toml`.

## 1. A new kernel crate

```
kernels/k-foo/Cargo.toml
  [package] name = "k-foo", version.workspace = true, edition.workspace = true
  [lib] crate-type = ["cdylib"]
  [dependencies] kcommon = { path = "../kcommon" }
  [lints] workspace = true
kernels/k-foo/src/lib.rs
  #![no_std]
  #![feature(abi_ptx, asm_experimental_arch)]
  extern crate kcommon;            // brings the panic handler and the dynamic-smem symbol
```

A kernel is an exported `extern "ptx-kernel"` function. Parameters are plain `u32/i32/u64/f32` and raw pointers:

```rust
use kcommon::*;

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn foo_scale(out: *mut f32, inp: *const f32, n: u32, s: f32) {
    let i = bid_x() * bdim_x() + tid_x();
    if i < n {
        *out.add(i as usize) = *inp.add(i as usize) * s;
    }
}
```

## 2. Embedding it in a host crate

`crates/fe-foo/build.rs`:

```rust
fn main() { fe_kbuild::build("k-foo"); }        // [build-dependencies] fe-kbuild.workspace = true
```

`crates/fe-foo/src/lib.rs`:

```rust
pub static K_FOO_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/k_foo.cubin"));
```

The build script runs `cargo build --release -p k-foo` in `kernels/` with a clean environment (so the
stable host flags never leak in), then `ptxas`. It only watches `kernels/k-foo` and `kernels/kcommon`, so
other agents' kernel crates never trigger your rebuild. `FE_PTXAS_FLAGS="-v"` prints per-kernel registers,
spills and smem as cargo warnings. The PTX is kept next to the cubin in `OUT_DIR` for inspection
(`find target -name k_foo.ptx`). `cuobjdump -sass <cubin>` shows SASS.

## 3. Launching from Rust (fe-cuda)

```rust
use fe_cuda::{Context, DeviceBuffer, Module, Stream};
let ctx = Context::new(0)?;                     // primary context, made current on this thread
let m = Module::load(fe_foo::K_FOO_CUBIN)?;
let f = m.function("foo_scale")?;
let s = Stream::new()?;
let x = DeviceBuffer::from_slice(&host)?;
let y = DeviceBuffer::<f32>::alloc(n)?;
unsafe { s.launch(&f, n.div_ceil(256), 256, /*dyn smem bytes*/ 0, (y.ptr(), x.ptr(), n as u32, 2.0f32))? };
s.sync()?;
```

Arguments are a tuple of `KernelArg` types (`u32 i32 u64 i64 f32 f64 DevPtr`); they must match the kernel
signature exactly (a `u32` where the kernel has `u64` is silent garbage). Pointers are `DevPtr`
(`buf.ptr()`, `buf.at(i)`, `ptr.add::<T>(n)`, `ptr.byte_add(b)`). Grid/block are `u32`, `(x, y)` or `(x, y, z)`.
More than 48 KB dynamic smem: `f.set_max_dynamic_smem(bytes)?` once (GB10 opt-in max 101376 B).
Graphs: `s.begin_capture()?; ...launches...; let g = s.end_capture()?; g.launch(&s)?`.
Timing: `let ms = s.time(|s| { ...launches...; Ok(()) })?`. All of this is exercised by `ferrite selftest`.

## 4. kcommon reference

| module | what |
|---|---|
| root | `tid_x/y`, `bid_x/y/z`, `bdim_x/y`, `gdim_x/y`, `lane`, `warp_id`, `nwarps`; `sync_threads`, `sync_warp`, `bar_sync(id, n)`, `fence_block`; `ldg_nc_v4`, `ldg_stream_v4` (128-bit `ld.global.nc[.L1::no_allocate]`), `stg_v4`, `atomic_add_f32` (red), `atomic_add_u32` |
| `warp` | `shfl_xor/down/idx` (f32) and `_u32` forms, `sum`, `max`, `sum_width::<W>`, `block_sum(v, scratch)`, `block_max` |
| `smem` | `Shared<T>` (typed 32-bit shared address: `ld/st/at/set/add/byte_add/cast/generic`), `dyn_base::<T>()`, `dyn_size()`, `static_shared!(name, T, BYTES)` |
| `bf16` | bf16 = `u16` bits: `to_f32`, `from_f32` (RNE), `pack2(lo, hi)`, `lo/hi/unpack2`, `unpack8`, `round` |
| `math` | `fma`, `exp/exp2/ln/log2` (MUFU approx), `rsqrt`, `sqrt` (rn), `rcp` (rn), `tanh`, `sigmoid`, `silu`, `softplus` |
| `mma` | `mma_bf16_16816`, `mma_f16_16816` (f32 acc), `ldmatrix_x1/x2/x4[_trans]`, `cp_async_16[_zfill]`, `cp_async_4`, `cp_async_commit`, `cp_async_wait::<N>`, `cp_async_wait_all`; fragment layouts documented in the module |

## 5. Shared memory (the one non-obvious part)

Rust cannot name an LLVM `addrspace(3)` global, so shared memory goes through PTX:

- **Dynamic**: kcommon declares `.extern .shared .align 16 .b8 fe_dsmem[];` once at module scope with
  `global_asm!`. `smem::dyn_base::<T>()` returns its address; carve it with `.add(n)` / `.byte_add(b)` and pass
  the byte count as the launch `smem` argument. Verified: lands once in every kernel crate's PTX.
- **Static**: `let t = static_shared!(my_tile, f32, 4096);` emits a *function-scope* `.shared` array (PTX
  rejects `.shared` inside `{ }` sub-blocks, so names must be unique per kernel; a duplicate is a loud ptxas
  error, never silent). Use it at the top of the kernel, outside loops/branches.
- Access through `Shared<T>`: `ld`/`st` emit real `ld.shared`/`st.shared` (SASS LDS/STS, verified), and the raw
  `addr()` feeds `ldmatrix` / `cp.async` directly. Supported element types: `u16 u32 i32 f32 u64 [u32;2]
  [u32;4] [f32;4]`. `Shared::generic()` gives a generic pointer (generic LD/ST: slower, avoid in hot loops).
- The asm wrappers are not `pure`, so LLVM treats them as memory side effects: they will not be reordered
  across `sync_threads()` or each other. Barriers are still your job.

## 6. Inline PTX rules of thumb

- `asm!` on nvptx needs `#![feature(asm_experimental_arch)]`. Register classes: `reg16` (u16/i16), `reg32`
  (u32/i32/f32), `reg64` (u64/i64/f64/pointers). No predicate class: compute predicates inside the asm with
  `{ .reg .pred p; setp...; @p ...; }` (braced sub-blocks are fine for `.reg`, not for `.shared`).
- `const N` operands interpolate integers into the template (`cp.async.wait_group {n};`, array sizes).
- Literal braces in PTX vector syntax are doubled: `"{{{a}, {b}}}"`.
- Mark pure register math `options(pure, nomem, nostack)`; anything warp-synchronous (`shfl.sync`,
  `mma.sync`, `ldmatrix`) should NOT be `pure` (keeps LLVM from hoisting/sinking it into divergent code).
- f32 `+ - * /` are native; there is no libm in `core`, so use `kcommon::math` for exp/log/sqrt.
- `f32::max/min/abs` work in core. Avoid `u64` division in hot code (it is a software routine).
- Bounds checks and panics compile to `trap`; use raw pointers (`*p.add(i)`) in hot loops, not slices.

## 7. Checking the generated code

```
cd kernels && cargo build --release -p k-foo        # PTX at target/nvptx64-nvidia-cuda/release/k_foo.ptx
ptxas -arch=sm_121a -O3 -v target/nvptx64-nvidia-cuda/release/k_foo.ptx -o /tmp/k.cubin
cuobjdump -sass /tmp/k.cubin | grep -E "HMMA|LDSM|LDGSTS|LDS|STS"
```
