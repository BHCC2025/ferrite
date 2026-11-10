//! Kernel registry of the forward: kernels by (module tag, name, work-group size), created on first use and cached.
//!
//! The forward names kernels exactly as arc-engine does (and passes the same arguments and grids), so a kernel can come from either
//! - the Rust kernels (kernels-xe: kx-core, kx-xq, ...; built by build.rs through fe-kbuild-xe and embedded: [`embedded`],
//!   [`rust_modules`]; FE_XE_KERNELS=rust, the engine path), or
//! - arc-engine's reference SPIR-V (`kernels/xpu/*.spv`), **test only** (cargo feature `ref-spv`): it lets the host side of the forward
//!   (buffers, launch order, arguments) be validated bit for bit against the oracle before the Rust kernels exist, and then each Rust
//!   kernel be swapped in alone and checked against the same forward. Never the shipped path (PLAN-XE rules; NON-RUST.md).
//!   FE_XE_KERNELS=mixed (also test only): the embedded Rust kernel where its module has the name, else the reference.
//! FE_XE_RUST_SPV adds Rust modules from files on top of any source (experiments, kernels not merged yet). Kx::origin_line reports
//! which kernels a run took from where.
//!
//! Module tags (arc-engine module -> build flags): basic, rows, gdn, attn, kvq (no flags); attn_dec_q8, attn_long_q8 (256 GRF);
//! xq3 xq4 xq5 xq6 xq8 (XQ decode, no flags); spec_head; prompt path: xqpfd3..8 (no flags), xq_pfg and pfxi_q8 (256 GRF).
use anyhow::{anyhow, bail, Context as _, Result};
use fe_ze::{Event, EventPool, Kernel, KernelArg, Module, Queue, Ze};
use std::cell::RefCell;
use std::collections::HashMap;
#[cfg(feature = "ref-spv")]
use std::path::PathBuf;
use std::sync::Arc;

/// (file, IGC flags) of a module tag in the reference tree.
fn ref_module(tag: &str) -> Option<(&'static str, &'static str)> {
    const GRF256: &str = "-cl-intel-256-GRF-per-thread";
    Some(match tag {
        "basic" => ("qwen_basic.spv", ""),
        "rows" => ("qwen_rows.spv", ""),
        "gdn" => ("qwen_gdn.spv", ""),
        "attn" => ("qwen_attn.spv", ""),
        "kvq" => ("qwen_kvq.spv", ""),
        "attn_dec_q8" => ("qwen_attn_dec_q8.spv", GRF256),
        "attn_long_q8" => ("qwen_attn_long_q8.spv", GRF256),
        "xq3" => ("xq3.spv", ""),
        "xq4" => ("xq4.spv", ""),
        "xq5" => ("xq5.spv", ""),
        "xq6" => ("xq6.spv", ""),
        "xq8" => ("xq8.spv", ""),
        "spec_head" => ("spec_head.spv", ""),
        "xqpfd3" => ("xqpfd3.spv", ""),
        "xqpfd4" => ("xqpfd4.spv", ""),
        "xqpfd5" => ("xqpfd5.spv", ""),
        "xqpfd6" => ("xqpfd6.spv", ""),
        "xqpfd8" => ("xqpfd8.spv", ""),
        "xq_pfg" => ("xq_pfg.spv", GRF256),
        "pfxi_q8" => ("qwen_attn_pfxi_q8.spv", GRF256),
        _ => return None,
    })
}

/// The Rust kernels, built from kernels-xe by build.rs (fe-kbuild-xe) and embedded.
pub mod embedded {
    /// kx-core: norms, GDN window + one-token kernels, attention prep, KV q8 append, chunk merge, embedding, SwiGLU, argmax, the MTP
    /// draft head
    pub static KX_CORE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kx_core.spv"));
    /// kx-core feature dec: the matrix-engine decode attention and the prompt-window attention (256 GRF)
    pub static KX_CORE_DEC: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kx_core_dec.spv"));
    /// kx-xq per code width: XQ decode matvec 1..16 rows, prepasses, the prompt GEMM's prep and weight decode
    pub static KX_XQ3: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kx_xq3.spv"));
    pub static KX_XQ4: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kx_xq4.spv"));
    pub static KX_XQ5: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kx_xq5.spv"));
    pub static KX_XQ6: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kx_xq6.spv"));
    pub static KX_XQ8: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kx_xq8.spv"));
    /// kx-xqpf: the prompt GEMM (xq_pfgemm_f16_c*t*, 256 GRF)
    pub static KX_XQPF: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kx_xqpf.spv"));
}

/// Module tag -> (embedded Rust SPIR-V, IGC flags) for every tag the Rust kernels cover so far. A tag may map to a module that lacks
/// some of the tag's kernels (kx-core has the row norms, not the one-row `rmsnorm512`): [`Source::Mixed`] falls back per kernel name.
pub fn rust_modules() -> HashMap<String, (&'static [u8], &'static str)> {
    const GRF256: &str = "-cl-intel-256-GRF-per-thread";
    use embedded::*;
    let mut m: HashMap<String, (&'static [u8], &'static str)> = HashMap::new();
    for t in ["rows", "basic", "kvq", "attn_long_q8", "gdn", "attn", "spec_head"] {
        m.insert(t.into(), (KX_CORE, ""));
    }
    for t in ["attn_dec_q8", "pfxi_q8"] {
        m.insert(t.into(), (KX_CORE_DEC, GRF256));
    }
    for (t, b) in [("xq3", KX_XQ3), ("xq4", KX_XQ4), ("xq5", KX_XQ5), ("xq6", KX_XQ6), ("xq8", KX_XQ8)] {
        m.insert(t.into(), (b, ""));
        // the prompt GEMM's prepass and weight decode live in the same per-width module
        m.insert(format!("xqpfd{}", &t[2..]), (b, ""));
    }
    m.insert("xq_pfg".into(), (KX_XQPF, GRF256));
    m
}

pub enum Source {
    /// arc-engine's `kernels/xpu` directory (test only)
    #[cfg(feature = "ref-spv")]
    RefSpv(PathBuf),
    /// Rust kernels: module tag -> (SPIR-V bytes, flags) ([`rust_modules`])
    Rust(HashMap<String, (&'static [u8], &'static str)>),
    /// The Rust module where it has the kernel, else the reference (test only: runs the whole engine while kernels are missing;
    /// the origin line lists what came from the reference)
    #[cfg(feature = "ref-spv")]
    Mixed(HashMap<String, (&'static [u8], &'static str)>, PathBuf),
}

/// Launch profiling (FE_XE_PROFILE=1): every launch signals a timestamp event; device times are summed per kernel.
struct Prof {
    pool: EventPool,
    evs: Vec<(String, Event)>,
    /// "tag:name" -> (launches, device ns)
    acc: std::collections::BTreeMap<String, (u64, f64)>,
}

const PROF_EVENTS: u32 = 4096;

impl Prof {
    /// Reads (and waits for) the pending events into the totals.
    fn harvest(&mut self) -> Result<()> {
        for (k, e) in self.evs.drain(..) {
            let ns = e.kernel_ns().map_err(|e| anyhow!("{e}"))?;
            let a = self.acc.entry(k).or_insert((0, 0.0));
            a.0 += 1;
            a.1 += ns;
        }
        Ok(())
    }
}

pub struct Kx {
    pub ze: Arc<Ze>,
    prof: RefCell<Option<Prof>>,
    src: Source,
    /// modules by key: "env:<tag>" (FE_XE_RUST_SPV tag=file), "ref:<tag>", "rust:<bytes>:<flags>"
    mods: RefCell<HashMap<String, Module>>,
    /// kernel names of the Rust modules (Mixed: which names fall back)
    #[cfg(feature = "ref-spv")]
    names: RefCell<HashMap<String, Vec<String>>>,
    /// kernel name -> Rust module providing it (overrides the source's module for that name: one kernel at a time)
    overrides: HashMap<String, Module>,
    kernels: RefCell<HashMap<(String, String, [u32; 3]), Kernel>>,
    /// module tags replaced by Rust modules (FE_XE_RUST_SPV tag=file)
    rust_tags: std::collections::HashSet<String>,
    /// "tag:name" -> whether the kernel came from a Rust module (every kernel created so far)
    origin: RefCell<std::collections::BTreeMap<String, bool>>,
    /// launches since creation (statistics)
    pub launches: std::cell::Cell<u64>,
}

impl Kx {
    pub fn new(ze: &Arc<Ze>, src: Source) -> Kx {
        let prof = if std::env::var("FE_XE_PROFILE").is_ok_and(|v| v != "0") {
            EventPool::new(ze, PROF_EVENTS).ok().map(|pool| Prof { pool, evs: Vec::new(), acc: Default::default() })
        } else {
            None
        };
        Kx { ze: ze.clone(), prof: RefCell::new(prof), src, mods: RefCell::new(HashMap::new()), #[cfg(feature = "ref-spv")]
            names: RefCell::new(HashMap::new()),
            overrides: HashMap::new(), kernels: RefCell::new(HashMap::new()), launches: 0.into(),
             rust_tags: Default::default(), origin: RefCell::new(Default::default()) }
    }

    /// Takes the kernels `only` (all of the module's when None) from a Rust SPIR-V module instead of the source; returns their names.
    pub fn add_override(&mut self, spv: &[u8], flags: &str, only: Option<&[String]>) -> Result<Vec<String>> {
        let m = Module::from_spirv(&self.ze, spv, flags).map_err(|e| anyhow!("{e}"))?;
        let mut taken = Vec::new();
        for n in m.kernel_names().map_err(|e| anyhow!("{e}"))? {
            // (the driver lists its own symbol-table kernel in every module)
            if n.starts_with("Intel_Symbol_Table") {
                continue;
            }
            if only.is_none_or(|o| o.iter().any(|x| *x == n)) {
                self.overrides.insert(n.clone(), m.clone());
                taken.push(n);
            }
        }
        Ok(taken)
    }

    /// FE_XE_RUST_SPV=item,... with item = `file[@flags]` (its kernels replace those names in every module; FE_XE_RUST_ONLY=name,...
    /// restricts to these) or `tag=file[@flags]` (the module replaces the module tag, e.g. xq5=kx_xq5.spv: same kernel names per width).
    pub fn overrides_from_env(&mut self) -> Result<Vec<String>> {
        let Ok(list) = std::env::var("FE_XE_RUST_SPV") else { return Ok(Vec::new()) };
        let only: Option<Vec<String>> = std::env::var("FE_XE_RUST_ONLY").ok().map(|v| v.split(',').map(str::to_string).collect());
        let mut all = Vec::new();
        for item in list.split(',').filter(|x| !x.is_empty()) {
            let (tag, rest) = match item.split_once('=') {
                Some((t, r)) => (Some(t), r),
                None => (None, item),
            };
            let (file, flags) = rest.split_once('@').unwrap_or((rest, ""));
            let spv = std::fs::read(file).with_context(|| format!("reading {file}"))?;
            match tag {
                Some(t) => {
                    let m = Module::from_spirv(&self.ze, &spv, flags).map_err(|e| anyhow!("{file}: {e}"))?;
                    self.mods.borrow_mut().insert(format!("env:{t}"), m);
                    self.rust_tags.insert(t.to_string());
                    all.push(format!("{t}:*"));
                }
                None => all.extend(self.add_override(&spv, flags, only.as_deref())?),
            }
        }
        Ok(all)
    }

    /// The reference kernels of `dir` (default: FE_XE_REF_SPV). Test only.
    #[cfg(feature = "ref-spv")]
    pub fn reference(ze: &Arc<Ze>, dir: Option<PathBuf>) -> Kx {
        let d = dir.unwrap_or_else(|| {
            PathBuf::from(std::env::var("FE_XE_REF_SPV").expect("FE_XE_REF_SPV: the folder of the reference .spv modules"))
        });
        Kx::new(ze, Source::RefSpv(d))
    }

    /// The Rust kernels embedded in this build (FE_XE_KERNELS=rust, the engine path).
    pub fn rust(ze: &Arc<Ze>) -> Kx {
        Kx::new(ze, Source::Rust(rust_modules()))
    }

    /// The embedded Rust kernels where they exist, the reference SPIR-V for the rest. Test only.
    #[cfg(feature = "ref-spv")]
    pub fn mixed(ze: &Arc<Ze>, dir: Option<PathBuf>) -> Kx {
        let d = dir.unwrap_or_else(|| {
            PathBuf::from(std::env::var("FE_XE_REF_SPV").expect("FE_XE_REF_SPV: the folder of the reference .spv modules"))
        });
        Kx::new(ze, Source::Mixed(rust_modules(), d))
    }

    /// A cached module: key, SPIR-V loader, flags.
    fn cached(&self, key: String, flags: &str, spv: impl FnOnce() -> Result<Vec<u8>>) -> Result<Module> {
        if let Some(m) = self.mods.borrow().get(&key) {
            return Ok(m.clone());
        }
        let m = Module::from_spirv(&self.ze, &spv()?, flags).map_err(|e| anyhow!("{key}: {e}"))?;
        self.mods.borrow_mut().insert(key, m.clone());
        Ok(m)
    }

    #[cfg(feature = "ref-spv")]
    fn ref_mod(&self, dir: &std::path::Path, tag: &str) -> Result<Module> {
        let (file, flags) = ref_module(tag).ok_or_else(|| anyhow!("unknown module tag {tag}"))?;
        let p = dir.join(file);
        self.cached(format!("ref:{tag}"), flags, || std::fs::read(&p).with_context(|| format!("reading {p:?}")))
    }

    /// The Rust module of `tag` (modules shared by several tags are created once).
    fn rust_mod(&self, map: &HashMap<String, (&'static [u8], &'static str)>, tag: &str) -> Result<Option<Module>> {
        let Some(&(spv, flags)) = map.get(tag) else { return Ok(None) };
        self.cached(format!("rust:{:p}:{flags}", spv.as_ptr()), flags, || Ok(spv.to_vec())).map(Some)
    }

    /// Whether module `m` (cached under `key`) has a kernel `name`.
    #[cfg(feature = "ref-spv")]
    fn has_kernel(&self, key: &str, m: &Module, name: &str) -> Result<bool> {
        if let Some(n) = self.names.borrow().get(key) {
            return Ok(n.iter().any(|x| x == name));
        }
        let n = m.kernel_names().map_err(|e| anyhow!("{e}"))?;
        let has = n.iter().any(|x| x == name);
        self.names.borrow_mut().insert(key.to_string(), n);
        Ok(has)
    }

    /// The module providing (tag, name) and whether it is a Rust module.
    fn module_for(&self, tag: &str, name: &str) -> Result<(Module, bool)> {
        if let Some(m) = self.overrides.get(name) {
            return Ok((m.clone(), true));
        }
        if self.rust_tags.contains(tag) {
            return Ok((self.mods.borrow()[&format!("env:{tag}")].clone(), true));
        }
        let _ = ref_module; // (the table documents the tags for every source)
        match &self.src {
            #[cfg(feature = "ref-spv")]
            Source::RefSpv(dir) => Ok((self.ref_mod(dir, tag)?, false)),
            Source::Rust(map) => {
                let m = self.rust_mod(map, tag)?.ok_or_else(|| anyhow!("no Rust module for {tag}:{name} yet (kernels-xe)"))?;
                Ok((m, true))
            }
            #[cfg(feature = "ref-spv")]
            Source::Mixed(map, dir) => {
                if let Some(m) = self.rust_mod(map, tag)? {
                    let key = format!("{:p}", map[tag].0.as_ptr());
                    if self.has_kernel(&key, &m, name)? {
                        return Ok((m, true));
                    }
                }
                Ok((self.ref_mod(dir, tag)?, false))
            }
        }
    }

    /// Whether `name` can be launched under module tag `tag` from the current source (e.g. a Rust-only kernel such as
    /// kx-core gdn_pre_r is absent from arc-engine's reference modules).
    pub fn provides(&self, tag: &str, name: &str) -> bool {
        match self.module_for(tag, name) {
            Ok((m, _)) => m.kernel_names().map(|n| n.iter().any(|x| x == name)).unwrap_or(false),
            Err(_) => false,
        }
    }

    /// Sets the arguments and queues `name` from module `tag` with work-group size `local` over `groups`.
    pub fn run(&self, q: &Queue, tag: &str, name: &str, local: [u32; 3], groups: [u32; 3], args: &[&dyn KernelArg]) -> Result<()> {
        let key = (tag.to_string(), name.to_string(), local);
        if !self.kernels.borrow().contains_key(&key) {
            let (m, rust) = self.module_for(tag, name)?;
            self.origin.borrow_mut().insert(format!("{tag}:{name}"), rust);
            let k = m.kernel(name, local).map_err(|e| anyhow!("{tag}:{name}: {e}"))?;
            self.kernels.borrow_mut().insert(key.clone(), k);
        }
        let ks = self.kernels.borrow();
        let k = &ks[&key];
        k.args(args).map_err(|e| anyhow!("{tag}:{name}: {e}"))?;
        if groups.iter().any(|&g| g == 0) {
            bail!("{tag}:{name}: empty grid {groups:?}");
        }
        let mut prof = self.prof.borrow_mut();
        match prof.as_mut() {
            Some(p) => {
                if p.evs.len() as u32 >= PROF_EVENTS {
                    p.harvest()?;
                }
                let ev = p.pool.event(p.evs.len() as u32).map_err(|e| anyhow!("{e}"))?;
                q.launch_ev(k, groups, Some(&ev)).map_err(|e| anyhow!("{tag}:{name}: {e}"))?;
                p.evs.push((format!("{tag}:{name}"), ev));
            }
            None => q.launch(k, groups).map_err(|e| anyhow!("{tag}:{name}: {e}"))?,
        }
        self.launches.set(self.launches.get() + 1);
        Ok(())
    }

    /// FE_XE_PROFILE: device time per kernel since the start (or the last reset), longest first, with each kernel's origin; None when
    /// profiling is off.
    pub fn profile_report(&self) -> Result<Option<String>> {
        let mut prof = self.prof.borrow_mut();
        let Some(p) = prof.as_mut() else { return Ok(None) };
        p.harvest()?;
        let total: f64 = p.acc.values().map(|a| a.1).sum();
        let mut rows: Vec<_> = p.acc.iter().collect();
        rows.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
        let origin = self.origin.borrow();
        let mut out = format!("device time by kernel: {:.1} ms in {} launches\n", total / 1e6, p.acc.values().map(|a| a.0).sum::<u64>());
        for (k, (n, ns)) in rows {
            let o = match origin.get(k) {
                Some(true) => "rust",
                Some(false) => "ref",
                None => "?",
            };
            out += &format!("  {:<40} {:<4} {:>8} launches {:>10.2} ms {:>9.2} us/launch {:>5.1} %\n", k, o, n, ns / 1e6, ns / 1e3 / *n as f64, 100.0 * ns / total);
        }
        Ok(Some(out))
    }

    /// Clears the profile totals (e.g. after a prompt, to profile decode alone).
    pub fn profile_reset(&self) -> Result<()> {
        if let Some(p) = self.prof.borrow_mut().as_mut() {
            p.harvest()?;
            p.acc.clear();
        }
        Ok(())
    }

    /// The kernels used so far by origin: (Rust, reference SPIR-V), as "tag:name".
    pub fn origins(&self) -> (Vec<String>, Vec<String>) {
        let o = self.origin.borrow();
        let rust = o.iter().filter(|e| *e.1).map(|e| e.0.clone()).collect();
        let refs = o.iter().filter(|e| !*e.1).map(|e| e.0.clone()).collect();
        (rust, refs)
    }

    /// One line for logs: how many distinct kernels came from Rust modules, and the reference ones by name.
    pub fn origin_line(&self) -> String {
        let (r, f) = self.origins();
        if f.is_empty() {
            format!("kernels used: {} Rust, 0 reference (all Rust)", r.len())
        } else {
            format!("kernels used: {} Rust, {} reference: {}", r.len(), f.len(), f.join(" "))
        }
    }

    /// As [`Kx::run`] with a work-group size that changes per call (the XQ matvec: 16 x split sub-groups).
    pub fn run_sized(&self, q: &Queue, tag: &str, name: &str, local: [u32; 3], groups: [u32; 3], args: &[&dyn KernelArg]) -> Result<()> {
        self.run(q, tag, name, local, groups, args)
    }
}
