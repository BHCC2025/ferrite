//! The kernel inventory of the B70 forward: every device kernel the serving path launches, as data (name, arc-engine source kernel it
//! must reproduce bit for bit, work shape, priority). DESIGN.md has the full table with shapes and per-token launch counts; this list is
//! what fe-xe's loader resolves against the kernels-xe SPIR-V modules (a missing entry is a load error, not a silent fallback).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path {
    /// one-token decode (Model.forward) and the 1..16-row verify windows
    Decode,
    /// prompt windows (> 16 rows)
    Prefill,
    /// both
    Both,
}

#[derive(Clone, Copy, Debug)]
pub struct KernelSpec {
    /// kernels-xe entry point
    pub name: &'static str,
    /// arc-engine kernel (file:kernel) whose output it must equal bitwise
    pub reference: &'static str,
    pub path: Path,
    /// 1 = on the critical path of the X2 decode forward, 2 = prefill / speculation (X3 / X5), 3 = later
    pub priority: u8,
}

/// Decode-path kernels of the default profile (XQ mix5.0, KV q8, ARC_DEC=1); prefill / MTP / spec entries follow DESIGN.md.
pub const DECODE: &[KernelSpec] = &[
    KernelSpec { name: "xq_mv", reference: "xq.cl:xq_mv_{g}_{MW}x{NR}_{NB} (xq3/4/5/6/8.spv)", path: Path::Decode, priority: 1 },
    KernelSpec { name: "xq_prep", reference: "xq.cl:xq_prep", path: Path::Decode, priority: 1 },
    KernelSpec { name: "xq_add_rms_prep", reference: "xq.cl:xq_add_rms_prep", path: Path::Decode, priority: 1 },
    KernelSpec { name: "xq_prep_swiglu", reference: "xq.cl:xq_prep_swiglu", path: Path::Decode, priority: 1 },
    KernelSpec { name: "embed_bf16", reference: "qwen_basic.cl:embed_bf16", path: Path::Both, priority: 1 },
    KernelSpec { name: "add_bf16", reference: "qwen_basic.cl:add_bf16", path: Path::Decode, priority: 1 },
    KernelSpec { name: "add_rmsnorm", reference: "qwen_basic.cl:add_rmsnorm", path: Path::Decode, priority: 1 },
    KernelSpec { name: "rmsnorm512", reference: "qwen_basic.cl:rmsnorm512", path: Path::Decode, priority: 1 },
    KernelSpec { name: "gdn_conv_prep", reference: "qwen_gdn.cl:gdn_conv_prep", path: Path::Decode, priority: 1 },
    KernelSpec { name: "gdn_step", reference: "qwen_gdn.cl:gdn_step", path: Path::Decode, priority: 1 },
    KernelSpec { name: "gdn_gate_norm", reference: "qwen_gdn.cl:gdn_gate_norm", path: Path::Decode, priority: 1 },
    KernelSpec { name: "attn_prep", reference: "qwen_attn.cl:attn_prep", path: Path::Decode, priority: 1 },
    KernelSpec { name: "kvq_append", reference: "qwen_kvq.cl (Kvq.append, q8)", path: Path::Both, priority: 1 },
    KernelSpec { name: "attn_dec", reference: "qwen_attn_dec.cl (ARC_DEC=1, q8) / qwen_attn_long.cl", path: Path::Decode, priority: 1 },
    KernelSpec { name: "round_bf16_f32", reference: "qwen_basic.cl:round_bf16_f32", path: Path::Decode, priority: 2 },
    KernelSpec { name: "argmax_partial", reference: "qwen_basic.cl:argmax_partial", path: Path::Decode, priority: 1 },
    KernelSpec { name: "argmax_final", reference: "qwen_basic.cl:argmax_final", path: Path::Decode, priority: 1 },
];
