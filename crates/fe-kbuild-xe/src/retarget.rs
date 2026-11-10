//! amdgcn-amd-amdhsa LLVM IR (rustc output, every value named by `opt -passes=instnamer`) -> spirv64 LLVM IR in the
//! OpenCL flavour `llvm-spirv` turns into Kernel-capability SPIR-V for Level Zero.
//!
//! Why amdgcn: rustc's LLVM has no SPIR-V target, so the kernels are compiled for a GPU target rustc does have and the
//! IR is moved over. amdgcn (not nvptx) because it is the one that accepts SIMD vector types in `extern "C"`
//! signatures (the DPAS / block-IO builtins take `<8 x i32>` etc.), and its address spaces are all explicit.
//!
//! What changes, all as text on the named IR:
//! * datalayout + triple -> spirv64; `module asm` dropped; amdgpu-only attribute strings dropped.
//! * address spaces, a bijection applied to every type and intrinsic suffix: flat 0 -> generic 4, private 5 -> 0,
//!   global 1 -> 1, local 3 -> 3, constant 4 -> 2. The module stays type-consistent by construction.
//! * `amdgpu_kernel` -> `spir_kernel`; pointer parameters become global (`ptr addrspace(1) %x.kxg`) and the body gets
//!   `%x = addrspacecast .. to generic` at its entry (`opt -passes=infer-address-spaces` then pushes `global` through).
//! * calls of OpenCL builtins (`_Z..` mangled names) get their pointer arguments cast to the address space the
//!   mangling names (`PU3AS1` global, `PU3AS3` local, `PU3AS2` constant, plain `P` private), and their declarations
//!   are retyped to match: the same calls clang + llvm-spirv emit for OpenCL C, which IGC resolves by name.
//! * markers from kxcommon (opaque `extern "C"` calls rustc cannot remove):
//!   - `__xe_slm(bytes, id)` -> a `addrspace(3)` (Workgroup) global `@__xe_slm_<bytes>_<id>` cast to generic;
//!   - `__xe_fdiv_ocl(a, b)` -> `fdiv` with `!fpmath 2.5` (OpenCL's 2.5 ulp division; plain `/` stays IEEE);
//!   - `__xe_attr_reqd_wg_size(x, y, z)`, `__xe_attr_sg_size(n)` -> kernel metadata (sub-group size 16 by default);
//!   - `llvm.trap` (panic paths) dropped (the following `unreachable` stays).
use std::collections::BTreeMap;
use std::fmt::Write as _;

pub const SPIRV_DATALAYOUT: &str =
    "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1";
pub const SPIRV_TRIPLE: &str = "spirv64-unknown-unknown";
/// Sub-group size every kernel gets unless it calls `kxcommon::attr::sg_size::<N>()`.
pub const DEFAULT_SG_SIZE: u32 = 16;

fn map_as(a: u32) -> Result<u32, String> {
    match a {
        0 => Ok(4),
        5 => Ok(0),
        1 => Ok(1),
        3 => Ok(3),
        4 => Ok(2),
        _ => Err(format!("amdgcn address space {a} has no SPIR-V mapping")),
    }
}

fn ptr_ty(a: u32) -> String {
    if a == 0 { "ptr".into() } else { format!("ptr addrspace({a})") }
}

fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.' | b'$' | b'-')
}

/// `(N)` at the start of `s` (after optional spaces): (N, bytes consumed).
fn parse_parens_num(s: &str) -> Option<(u32, usize)> {
    let b = s.as_bytes();
    if b.first() != Some(&b'(') {
        return None;
    }
    let close = s.find(')')?;
    let n = s[1..close].trim().parse().ok()?;
    Some((n, close + 1))
}

/// ` addrspace(N)` right after a `ptr` word.
fn parse_ptr_as(s: &str) -> Option<(u32, usize)> {
    let t = s.strip_prefix(" addrspace")?;
    let (n, len) = parse_parens_num(t)?;
    Some((n, " addrspace".len() + len))
}

/// `@llvm.foo.p0.p5.i64` -> `@llvm.foo.p4.p0.i64`.
fn remap_intrinsic(name: &str) -> Result<String, String> {
    if name.starts_with("@llvm.amdgcn") || name.starts_with("@llvm.r600") {
        return Err(format!("amdgpu intrinsic {name} in the kernel IR: no SPIR-V equivalent"));
    }
    let mut out = String::new();
    for (i, part) in name.split('.').enumerate() {
        if i > 0 {
            out.push('.');
        }
        if part.len() > 1 && part.starts_with('p') && part[1..].bytes().all(|c| c.is_ascii_digit()) {
            write!(out, "p{}", map_as(part[1..].parse().unwrap())?).unwrap();
        } else {
            out.push_str(part);
        }
    }
    Ok(out)
}

/// Address-space remap of one line of IR code (string literals untouched).
pub fn remap_code(line: &str) -> Result<String, String> {
    let b = line.as_bytes();
    let mut out = String::with_capacity(line.len() + 32);
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == b'"' {
            let j = line[i + 1..].find('"').map(|k| i + 1 + k).unwrap_or(b.len() - 1);
            out.push_str(&line[i..=j]);
            i = j + 1;
            continue;
        }
        if matches!(c, b'%' | b'@' | b'!' | b'#' | b'$') {
            let s = i;
            i += 1;
            if i < b.len() && b[i] == b'"' {
                let j = line[i + 1..].find('"').map(|k| i + 1 + k).unwrap_or(b.len() - 1);
                out.push_str(&line[s..=j]);
                i = j + 1;
                continue;
            }
            while i < b.len() && is_ident(b[i]) {
                i += 1;
            }
            let id = &line[s..i];
            if id.starts_with("@llvm.") { out.push_str(&remap_intrinsic(id)?) } else { out.push_str(id) }
            continue;
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            let s = i;
            while i < b.len() && is_ident(b[i]) {
                i += 1;
            }
            let w = &line[s..i];
            if w == "ptr" {
                let (a, len) = parse_ptr_as(&line[i..]).unwrap_or((0, 0));
                out.push_str(&ptr_ty(map_as(a)?));
                i += len;
            } else if w == "addrspace" {
                if let Some((n, len)) = parse_parens_num(&line[i..]) {
                    write!(out, "addrspace({})", map_as(n)?).unwrap();
                    i += len;
                } else {
                    out.push_str(w);
                }
            } else {
                out.push_str(w);
            }
            continue;
        }
        // copy one UTF-8 character
        let ch = line[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    Ok(out)
}

/// Split at top-level `sep` (outside (), [], {}, <> and quotes).
fn split_top(s: &str, sep: u8) -> Vec<&str> {
    let mut parts = Vec::new();
    let (mut depth, mut start, mut q) = (0i32, 0usize, false);
    for (i, &c) in s.as_bytes().iter().enumerate() {
        if q {
            q = c != b'"';
            continue;
        }
        match c {
            b'"' => q = true,
            b'(' | b'[' | b'{' | b'<' => depth += 1,
            b')' | b']' | b'}' | b'>' => depth -= 1,
            _ if c == sep && depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if !s[start..].trim().is_empty() || !parts.is_empty() {
        parts.push(&s[start..]);
    }
    parts
}

/// Index of the `)` matching the `(` at `open`.
fn matching_paren(s: &str, open: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut q = false;
    for (i, &c) in s.as_bytes().iter().enumerate().skip(open) {
        if q {
            q = c != b'"';
            continue;
        }
        match c {
            b'"' => q = true,
            b'(' | b'[' | b'{' | b'<' => depth += 1,
            b')' | b']' | b'}' | b'>' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Address spaces of the pointer parameters of an OpenCL builtin, in order, from its SPIR (Itanium) mangling:
/// `PU3AS1..` global (1), `PU3AS3..` local (3), `PU3AS2..` constant (2), `PU3AS4..` generic (4), plain `P..` private (0).
pub fn mangled_ptr_spaces(name: &str) -> Result<Vec<u32>, String> {
    let rest = name.strip_prefix("_Z").ok_or_else(|| format!("{name}: not a mangled name"))?;
    let nd = rest.bytes().take_while(|c| c.is_ascii_digit()).count();
    let len: usize = rest[..nd].parse().map_err(|_| format!("{name}: bad mangled length"))?;
    let params = &rest[nd + len..];
    let b = params.as_bytes();
    let mut spaces = Vec::new();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'P' => {
                let mut a = 0;
                if params[i + 1..].starts_with("U3AS") {
                    a = (b[i + 5] - b'0') as u32;
                    i += 5;
                }
                spaces.push(a);
                i += 1;
            }
            b'S' if spaces.len() > 0 => {
                // a substitution after a pointer may name that pointer type again: not supported (none of the builtins
                // kxcommon declares has one)
                return Err(format!("{name}: substitution after a pointer parameter is not handled by the retarget pass"));
            }
            _ => i += 1,
        }
    }
    Ok(spaces)
}

#[derive(Default)]
struct KernelAttrs {
    sg_size: Option<u32>,
    wg_size: Option<[u32; 3]>,
    /// kernel parameters marked `__local` by `kxcommon::attr::local_arg`
    local_params: Vec<String>,
}

struct Ctx {
    next_md: u32,
    extra_md: Vec<String>,
    fpmath: Option<u32>,
    slm: BTreeMap<String, u32>,
    tmp: u32,
    retyped_decls: BTreeMap<String, Vec<u32>>,
}

impl Ctx {
    fn md(&mut self, body: String) -> u32 {
        let n = self.next_md;
        self.next_md += 1;
        self.extra_md.push(format!("!{n} = {body}"));
        n
    }
}

/// The integer constants of a call's arguments (`i32 noundef 512, i32 16` -> [512, 16]).
fn const_args(args: &str) -> Result<Vec<u64>, String> {
    split_top(args, b',')
        .iter()
        .map(|a| {
            let last = a.split_whitespace().last().unwrap_or("");
            last.parse::<u64>().map_err(|_| format!("marker argument `{}` is not a constant", a.trim()))
        })
        .collect()
}

/// Splits `.. call .. @callee(args) rest` -> (head up to and incl. callee, callee, args, rest).
fn split_call(l: &str) -> Option<(&str, &str, &str, &str)> {
    if !(l.contains(" call ") || l.trim_start().starts_with("call ")) {
        return None;
    }
    let at = l.find(" @")? + 1;
    let open = at + l[at..].find('(')?;
    let close = matching_paren(l, open)?;
    Some((&l[..open], &l[at..open], &l[open + 1..close], &l[close + 1..]))
}

/// The value of a pointer argument `ptr addrspace(4) noundef nonnull %x` (attributes dropped) -> `%x`.
fn ptr_arg_value(a: &str) -> Option<&str> {
    let t = a.trim().strip_prefix("ptr addrspace(4)")?;
    let mut t = t.trim_start();
    loop {
        let w = t.split_whitespace().next().unwrap_or("");
        let skip = if matches!(w, "noundef" | "nonnull" | "noalias" | "readonly" | "writeonly" | "nocapture" | "nofree" | "returned") {
            w.len()
        } else if w == "align" {
            let rest = &t[5..].trim_start();
            t.len() - rest.len() + rest.split_whitespace().next().unwrap_or("").len()
        } else if w.starts_with("dereferenceable") || w.starts_with("captures(") || w.starts_with("dead_on") {
            match t.find(')') {
                Some(p) => p + 1,
                None => w.len(),
            }
        } else {
            break;
        };
        t = t[skip..].trim_start();
    }
    if t.is_empty() { None } else { Some(t) }
}

/// One function body line (already address-space remapped).
fn rewrite_body_line(
    cx: &mut Ctx,
    l: &str,
    attrs: Option<&mut KernelAttrs>,
    casts: &BTreeMap<String, String>,
    out: &mut Vec<String>,
) -> Result<(), String> {
    let Some((head, callee, args, rest)) = split_call(l) else {
        out.push(l.to_string());
        return Ok(());
    };
    let lhs = l.split_once(" = ").map(|(v, _)| v.trim()).filter(|v| v.starts_with('%') && !v.contains(' '));
    match callee {
        "@llvm.trap" | "@llvm.debugtrap" => return Ok(()),
        "@__xe_slm" => {
            let v = const_args(args)?;
            let (bytes, id) = (v[0], v[1]);
            let g = format!("@__xe_slm_{bytes}_{id}");
            cx.slm.insert(g.clone(), bytes as u32);
            let lhs = lhs.ok_or("__xe_slm result unused")?;
            out.push(format!("  {lhs} = addrspacecast ptr addrspace(3) {g} to ptr addrspace(4)"));
            return Ok(());
        }
        "@__xe_fdiv_ocl" => {
            let a = split_top(args, b',');
            let val = |s: &str| s.split_whitespace().last().unwrap_or("").to_string();
            let fm = match cx.fpmath {
                Some(n) => n,
                None => {
                    let n = cx.md("!{float 2.500000e+00}".into());
                    cx.fpmath = Some(n);
                    n
                }
            };
            let lhs = lhs.ok_or("__xe_fdiv_ocl result unused")?;
            out.push(format!("  {lhs} = fdiv float {}, {}, !fpmath !{fm}", val(a[0]), val(a[1])));
            return Ok(());
        }
        _ => {}
    }
    if callee == "@__xe_local_arg" {
        let at = attrs.ok_or("local_arg outside a kernel")?;
        let mut v = args.split_whitespace().last().unwrap_or("").to_string();
        while let Some(src) = casts.get(&v) {
            v = src.clone();
        }
        let lhs = lhs.ok_or("local_arg result unused")?;
        out.push(format!("  {lhs} = addrspacecast ptr addrspace(3) {v}.kxl to ptr addrspace(4)"));
        at.local_params.push(v);
        return Ok(());
    }
    if let Some(m) = callee.strip_prefix("@__xe_attr_") {
        let v = const_args(args)?;
        let at = attrs.ok_or_else(|| format!("kernel attribute marker {callee} outside a kernel"))?;
        match m {
            "reqd_wg_size" => at.wg_size = Some([v[0] as u32, v[1] as u32, v[2] as u32]),
            "sg_size" => at.sg_size = Some(v[0] as u32),
            _ => return Err(format!("unknown kernel attribute marker {callee}")),
        }
        return Ok(());
    }
    if callee.starts_with("@__xe_") {
        return Err(format!("unknown kxcommon marker {callee}"));
    }
    if let Some(fname) = callee.strip_prefix('@').filter(|n| n.starts_with("_Z")) {
        let parts = split_top(args, b',');
        let nptr = parts.iter().filter(|a| a.trim().starts_with("ptr")).count();
        if nptr == 0 {
            out.push(l.to_string());
            return Ok(());
        }
        let spaces = mangled_ptr_spaces(fname)?;
        if spaces.len() != nptr {
            return Err(format!("{fname}: {nptr} pointer arguments but the mangling names {}", spaces.len()));
        }
        let mut k = 0;
        let mut new_args = Vec::new();
        for a in parts {
            if a.trim().starts_with("ptr") {
                let tgt = spaces[k];
                k += 1;
                let v = ptr_arg_value(a).ok_or_else(|| format!("{fname}: pointer argument `{}` is not generic", a.trim()))?;
                cx.tmp += 1;
                let nv = format!("%kx.as{tgt}.{}", cx.tmp);
                if tgt == 4 {
                    new_args.push(format!("ptr addrspace(4) {v}"));
                } else {
                    out.push(format!("  {nv} = addrspacecast ptr addrspace(4) {v} to {}", ptr_ty(tgt)));
                    new_args.push(format!("{} {nv}", ptr_ty(tgt)));
                }
            } else {
                new_args.push(a.trim().to_string());
            }
        }
        cx.retyped_decls.insert(fname.to_string(), spaces);
        out.push(format!("{head}({}){rest}", new_args.join(", ")));
        return Ok(());
    }
    out.push(l.to_string());
    Ok(())
}

/// `define .. @name(params) .. {` of a kernel -> (new header without `{`, entry casts).
fn kernel_header(h: &str, local: &[String]) -> Result<(String, String, Vec<String>), String> {
    let at = h.find(" @").ok_or("kernel header without a name")? + 1;
    let open = at + h[at..].find('(').ok_or("kernel header without parameters")?;
    let close = matching_paren(h, open).ok_or("kernel header: unbalanced parameters")?;
    let name = h[at + 1..open].to_string();
    let mut casts = Vec::new();
    let mut params = Vec::new();
    for p in split_top(&h[open + 1..close], b',') {
        let p = p.trim();
        if p.starts_with("ptr addrspace(4)") {
            let v = p.split_whitespace().last().unwrap();
            if !v.starts_with('%') {
                return Err(format!("kernel {name}: unnamed pointer parameter `{p}` (run instnamer first)"));
            }
            if local.iter().any(|l| l == v) {
                params.push(format!("ptr addrspace(3) {v}.kxl"));
                continue;
            }
            let attrs: Vec<&str> = p["ptr addrspace(4)".len()..]
                .split_whitespace()
                .filter(|w| *w != v && !w.starts_with("dereferenceable") && !w.starts_with("align") && w.parse::<u64>().is_err())
                .collect();
            params.push(format!("ptr addrspace(1) {} {v}.kxg", attrs.join(" ")).replace("  ", " "));
            casts.push(format!("  {v} = addrspacecast ptr addrspace(1) {v}.kxg to ptr addrspace(4)"));
        } else if p.starts_with("ptr") {
            return Err(format!("kernel {name}: pointer parameter `{p}` in a non-flat address space"));
        } else {
            params.push(p.to_string());
        }
    }
    let pre = h[..open].replacen("amdgpu_kernel", "spir_kernel", 1);
    let post = h[close + 1..].trim_end().trim_end_matches('{').trim_end();
    Ok((name, format!("{pre}({}){post}", params.join(", ")), casts))
}

fn strip_amdgpu_attrs(l: &str) -> String {
    // drop `"key"="value"` / `"key"` entries whose key is amdgpu-specific
    let mut out = String::new();
    let mut rest = l;
    while let Some(q) = rest.find('"') {
        let (before, after) = rest.split_at(q);
        out.push_str(before);
        let Some(e) = after[1..].find('"') else {
            out.push_str(after);
            rest = "";
            break;
        };
        let key = &after[1..1 + e];
        let mut consumed = e + 2;
        if after[consumed..].starts_with("=\"") {
            consumed += 2 + after[consumed + 2..].find('"').map(|x| x + 1).unwrap_or(0);
        }
        let drop = key == "target-cpu" || key == "target-features" || key.starts_with("amdgpu-") || key == "uniform-work-group-size";
        if !drop {
            out.push_str(&after[..consumed]);
        }
        rest = &after[consumed..];
    }
    out.push_str(rest);
    out
}

fn max_md(ir: &str) -> u32 {
    let b = ir.as_bytes();
    let mut m = 0;
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'!' {
            let s = i + 1;
            let mut j = s;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            if j > s {
                m = m.max(ir[s..j].parse().unwrap_or(0));
            }
            i = j.max(i + 1);
        } else {
            i += 1;
        }
    }
    m
}

/// The pass. Input: `opt -passes=instnamer` output of the rustc amdgcn bitcode. Output: spirv64 IR text.
pub fn retarget(ir: &str) -> Result<String, String> {
    let mut cx = Ctx { next_md: max_md(ir) + 1, extra_md: Vec::new(), fpmath: None, slm: BTreeMap::new(), tmp: 0, retyped_decls: BTreeMap::new() };
    let mut out: Vec<String> = Vec::new();
    let lines: Vec<&str> = ir.lines().collect();
    let mut i = 0;
    let mut kernels = Vec::new();
    while i < lines.len() {
        let l = lines[i];
        i += 1;
        if l.starts_with("module asm") {
            while i < lines.len() && lines[i].starts_with("    \"") {
                i += 1;
            }
            continue;
        }
        if l.starts_with("target datalayout") {
            out.push(format!("target datalayout = \"{SPIRV_DATALAYOUT}\""));
            continue;
        }
        if l.starts_with("target triple") {
            out.push(format!("target triple = \"{SPIRV_TRIPLE}\""));
            continue;
        }
        if l.starts_with('!') || l.starts_with(';') || l.starts_with("source_filename") {
            out.push(l.to_string());
            continue;
        }
        if l.starts_with("attributes #") {
            out.push(strip_amdgpu_attrs(l));
            continue;
        }
        let r = remap_code(l).map_err(|e| format!("line {i}: {e}"))?;
        if r.starts_with("define ") {
            // collect the whole function
            let mut body = Vec::new();
            while i < lines.len() && lines[i] != "}" {
                body.push(remap_code(lines[i]).map_err(|e| format!("line {}: {e}", i + 1))?);
                i += 1;
            }
            i += 1; // the closing brace
            let is_kernel = r.contains(" amdgpu_kernel ");
            let mut attrs = KernelAttrs::default();
            // `%a = addrspacecast <ty> %b to <ty>`: a -> b (local_arg looks through the casts amdgpu put on parameters)
            let mut casts = BTreeMap::new();
            for b in &body {
                if let Some((lhs, rhs)) = b.split_once(" = addrspacecast ") {
                    let src = rhs.split(" to ").next().unwrap_or("").split_whitespace().last().unwrap_or("");
                    casts.insert(lhs.trim().to_string(), src.to_string());
                }
            }
            let mut nb = Vec::new();
            for b in &body {
                rewrite_body_line(&mut cx, b, if is_kernel { Some(&mut attrs) } else { None }, &casts, &mut nb)
                    .map_err(|e| format!("in `{}`: {e}", r.trim_end_matches('{').trim()))?;
            }
            if is_kernel {
                let (name, hdr, casts) = kernel_header(&r, &attrs.local_params)?;
                let sg = cx.md(format!("!{{i32 {}}}", attrs.sg_size.unwrap_or(DEFAULT_SG_SIZE)));
                let mut md = format!(" !intel_reqd_sub_group_size !{sg}");
                if let Some([x, y, z]) = attrs.wg_size {
                    let w = cx.md(format!("!{{i32 {x}, i32 {y}, i32 {z}}}"));
                    write!(md, " !reqd_work_group_size !{w}").unwrap();
                }
                out.push(format!("{hdr}{md} {{"));
                // entry casts go after the entry label
                let mut k = 0;
                let is_label = |f: &str| {
                    let f = f.split(';').next().unwrap().trim_end();
                    !f.starts_with(' ') && f.ends_with(':') && f[..f.len() - 1].bytes().all(is_ident)
                };
                if nb.first().map(|f| is_label(f)).unwrap_or(false) {
                    out.push(nb[0].clone());
                    k = 1;
                }
                out.extend(casts);
                out.extend(nb.into_iter().skip(k));
                kernels.push(name);
            } else {
                out.push(r);
                out.extend(nb);
            }
            out.push("}".into());
            continue;
        }
        out.push(r);
    }
    if kernels.is_empty() {
        return Err("no kernels (`extern \"gpu-kernel\"` functions) in the module".into());
    }
    // declarations: retype pointer params of builtins whose calls were rewritten, drop marker declarations
    let mut fin = Vec::with_capacity(out.len() + 16);
    for l in out {
        if l.starts_with("declare ") {
            if l.contains("@__xe_") || l.contains("@llvm.trap(") || l.contains("@llvm.debugtrap(") {
                continue;
            }
            if let Some(at) = l.find(" @_Z") {
                let open = at + l[at..].find('(').unwrap();
                let name = &l[at + 2..open];
                if let Some(spaces) = cx.retyped_decls.get(name) {
                    let close = matching_paren(&l, open).unwrap();
                    let mut k = 0;
                    let ps: Vec<String> = split_top(&l[open + 1..close], b',')
                        .iter()
                        .map(|p| {
                            let p = p.trim();
                            if let Some(rest) = p.strip_prefix("ptr addrspace(4)") {
                                let t = spaces[k];
                                k += 1;
                                format!("{}{rest}", ptr_ty(t))
                            } else {
                                p.to_string()
                            }
                        })
                        .collect();
                    fin.push(format!("{}({}){}", &l[..open], ps.join(", "), &l[close + 1..]));
                    continue;
                }
            }
        }
        fin.push(l);
    }
    let mut s = fin.join("\n");
    s.push('\n');
    for (g, bytes) in &cx.slm {
        writeln!(s, "{g} = internal addrspace(3) global [{bytes} x i8] undef, align 64").unwrap();
    }
    let ocl = cx.md("!{i32 3, i32 0}".into());
    let src = cx.md("!{i32 3, i32 300000}".into());
    writeln!(s, "!opencl.ocl.version = !{{!{ocl}}}").unwrap();
    writeln!(s, "!spirv.Source = !{{!{src}}}").unwrap();
    for m in &cx.extra_md {
        writeln!(s, "{m}").unwrap();
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn remap() {
        assert_eq!(remap_code("  %x = load i32, ptr %p, align 4").unwrap(), "  %x = load i32, ptr addrspace(4) %p, align 4");
        assert_eq!(remap_code("  %a = alloca [4 x i8], align 4, addrspace(5)").unwrap(), "  %a = alloca [4 x i8], align 4, addrspace(0)");
        assert_eq!(remap_code("  %c = addrspacecast ptr addrspace(5) %a to ptr").unwrap(), "  %c = addrspacecast ptr %a to ptr addrspace(4)");
        assert_eq!(remap_code("call void @llvm.memcpy.p0.p5.i64(ptr %d, ptr addrspace(5) %s, i64 4, i1 false)").unwrap(),
            "call void @llvm.memcpy.p4.p0.i64(ptr addrspace(4) %d, ptr %s, i64 4, i1 false)");
        assert_eq!(remap_code("  %ptr.x = getelementptr i8, ptr %ptr, i64 1 ; \"ptr\"").unwrap(),
            "  %ptr.x = getelementptr i8, ptr addrspace(4) %ptr, i64 1 ; \"ptr\"");
    }
    #[test]
    fn mangling() {
        assert_eq!(mangled_ptr_spaces("_Z27intel_sub_group_block_read8PU3AS1Kj").unwrap(), vec![1]);
        assert_eq!(mangled_ptr_spaces("_Z42intel_sub_group_2d_block_read_16b_16r16x1cPU3AS1viiiDv2_iPt").unwrap(), vec![1, 0]);
        assert_eq!(mangled_ptr_spaces("_Z36intel_sub_group_i8_i8_matrix_mad_k32Dv8_sDv8_iS0_").unwrap(), Vec::<u32>::new());
        assert_eq!(mangled_ptr_spaces("_Z27intel_sub_group_block_read8PU3AS3Kj").unwrap(), vec![3]);
    }
}
