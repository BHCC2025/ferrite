//! GGUF reader (format v2/v3): header, metadata key/values (arrays included), tensor infos, multi-shard files
//! (`*-00001-of-0000N.gguf`), and the tensor data through a read-only `mmap` of every shard. Nothing is read
//! besides the headers; tensor bytes are borrowed slices of the mappings, so the page cache only holds what a
//! caller actually touches.
//!
//! ```ignore
//! let g = fe_gguf::Gguf::open("model-00001-of-00003.gguf")?;   // finds the other shards
//! let t = g.tensor("blk.0.ffn_gate_exps.weight").unwrap();
//! let bytes: &[u8] = g.data(t);                                  // the raw blocks (t.ty, t.dims)
//! ```
use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

/// ggml tensor types (the numeric values are the on-disk `ggml_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,
    /// Fork-specific (llama.cpp-flashnext): 64 values per block, f16 d + 2-bit codes {-1, 0, 1, 2} * d.
    Q2_0 = 42,
    /// Not a file type: IQ1_M rows repacked for the CPU expert lane (`fe_ggq::cpu::repack`); never read from a GGUF.
    RepIq1M = 1001,
    /// Not a file type: MXFP4 repacked to 18-byte blocks (the e8m0 scale as a u16, then the 16 nibble bytes) so the GPU reads 2-byte aligned.
    RepMxfp4 = 1002,
    /// Not a file type: IQ2_XXS / IQ2_XS / IQ1_M repacked for the CPU expert lane into magnitude (2 bit) and sign (1 bit) planes + scale bytes (fe_ggq::cpu::repack::rep3).
    RepR3Xxs = 1003,
    RepR3Xs = 1004,
    RepR3Iq1M = 1005,
    /// IQ2_XXS with the 8-bit grid indices kept and the signs resolved into a bit plane (76 B per 256 values).
    RepHXxs = 1006,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<GgmlType> {
        use GgmlType::*;
        Some(match v {
            0 => F32, 1 => F16, 2 => Q4_0, 3 => Q4_1, 6 => Q5_0, 7 => Q5_1, 8 => Q8_0, 9 => Q8_1, 10 => Q2K,
            11 => Q3K, 12 => Q4K, 13 => Q5K, 14 => Q6K, 15 => Q8K, 16 => IQ2XXS, 17 => IQ2XS, 18 => IQ3XXS,
            19 => IQ1S, 20 => IQ4NL, 21 => IQ3S, 22 => IQ2S, 23 => IQ4XS, 24 => I8, 25 => I16, 26 => I32,
            27 => I64, 28 => F64, 29 => IQ1M, 30 => BF16, 34 => TQ1_0, 35 => TQ2_0, 39 => MXFP4, 40 => NVFP4,
            41 => Q1_0, 42 => Q2_0,
            _ => return None,
        })
    }
    /// (values per block, bytes per block)
    pub fn block(self) -> (usize, usize) {
        use GgmlType::*;
        match self {
            F32 | I32 => (1, 4),
            F16 | BF16 | I16 => (1, 2),
            I8 => (1, 1),
            I64 | F64 => (1, 8),
            Q4_0 => (32, 18),
            Q4_1 => (32, 20),
            Q5_0 => (32, 22),
            Q5_1 => (32, 24),
            Q8_0 => (32, 34),
            Q8_1 => (32, 36),
            Q2K => (256, 84),
            Q3K => (256, 110),
            Q4K => (256, 144),
            Q5K => (256, 176),
            Q6K => (256, 210),
            Q8K => (256, 292),
            IQ2XXS => (256, 66),
            IQ2XS => (256, 74),
            IQ3XXS => (256, 98),
            IQ1S => (256, 50),
            IQ4NL => (32, 18),
            IQ3S => (256, 110),
            IQ2S => (256, 82),
            IQ4XS => (256, 136),
            IQ1M => (256, 56),
            TQ1_0 => (256, 54),
            TQ2_0 => (256, 66),
            MXFP4 => (32, 17),
            NVFP4 => (64, 36),
            Q1_0 => (128, 18),
            Q2_0 => (64, 18),
            RepIq1M => (256, 56),
            RepMxfp4 => (32, 18),
            RepR3Xxs | RepR3Xs | RepR3Iq1M => (256, 108),
            RepHXxs => (256, 76),
        }
    }
    pub fn name(self) -> &'static str {
        use GgmlType::*;
        match self {
            F32 => "F32", F16 => "F16", Q4_0 => "Q4_0", Q4_1 => "Q4_1", Q5_0 => "Q5_0", Q5_1 => "Q5_1",
            Q8_0 => "Q8_0", Q8_1 => "Q8_1", Q2K => "Q2_K", Q3K => "Q3_K", Q4K => "Q4_K", Q5K => "Q5_K",
            Q6K => "Q6_K", Q8K => "Q8_K", IQ2XXS => "IQ2_XXS", IQ2XS => "IQ2_XS", IQ3XXS => "IQ3_XXS",
            IQ1S => "IQ1_S", IQ4NL => "IQ4_NL", IQ3S => "IQ3_S", IQ2S => "IQ2_S", IQ4XS => "IQ4_XS", I8 => "I8",
            I16 => "I16", I32 => "I32", I64 => "I64", F64 => "F64", IQ1M => "IQ1_M", BF16 => "BF16",
            TQ1_0 => "TQ1_0", TQ2_0 => "TQ2_0", MXFP4 => "MXFP4", NVFP4 => "NVFP4", Q1_0 => "Q1_0", Q2_0 => "Q2_0", RepIq1M => "IQ1_M-rep", RepMxfp4 => "MXFP4-rep", RepR3Xxs => "IQ2_XXS-r3", RepR3Xs => "IQ2_XS-r3", RepR3Iq1M => "IQ1_M-r3", RepHXxs => "IQ2_XXS-h",
        }
    }
    /// Bytes of one row of `n` values (n must be a multiple of the block size).
    pub fn row_bytes(self, n: usize) -> usize {
        if self == GgmlType::RepIq1M {
            // 160-value units + per-row header, see fe_ggq::cpu::repack
            return (8 + (n / 32).div_ceil(5) * 40 + n / 256 * 4 + 7) & !7;
        }
        let (bs, by) = self.block();
        n / bs * by
    }
}

/// A metadata value.
#[derive(Clone, Debug)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<Value>),
}

impl Value {
    /// Any integer type as i64 (bools as 0/1).
    pub fn as_i64(&self) -> Option<i64> {
        Some(match *self {
            Value::U8(v) => v as i64,
            Value::I8(v) => v as i64,
            Value::U16(v) => v as i64,
            Value::I16(v) => v as i64,
            Value::U32(v) => v as i64,
            Value::I32(v) => v as i64,
            Value::U64(v) => v as i64,
            Value::I64(v) => v,
            Value::Bool(v) => v as i64,
            _ => return None,
        })
    }
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            _ => self.as_i64().map(|v| v as f64),
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Value::Str(s) = self { Some(s) } else { None }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        if let Value::Array(a) = self { Some(a) } else { None }
    }
    /// An array of numbers (or a single number) as i64s.
    pub fn as_i64_vec(&self) -> Option<Vec<i64>> {
        match self {
            Value::Array(a) => a.iter().map(|v| v.as_i64()).collect(),
            v => v.as_i64().map(|x| vec![x]),
        }
    }
    pub fn as_f32_vec(&self) -> Option<Vec<f32>> {
        match self {
            Value::Array(a) => a.iter().map(|v| v.as_f64().map(|x| x as f32)).collect(),
            v => v.as_f64().map(|x| vec![x as f32]),
        }
    }
    /// Short printable form (arrays abbreviated).
    pub fn short(&self) -> String {
        match self {
            Value::Str(s) => {
                if s.len() > 80 { format!("{:?}...({} bytes)", &s[..s.floor_char_boundary(80)], s.len()) } else { format!("{s:?}") }
            }
            Value::Array(a) => {
                let head: Vec<String> = a.iter().take(8).map(|v| v.short()).collect();
                format!("[{}{}] (n={})", head.join(", "), if a.len() > 8 { ", ..." } else { "" }, a.len())
            }
            Value::F32(v) => format!("{v}"),
            Value::F64(v) => format!("{v}"),
            Value::Bool(v) => format!("{v}"),
            v => format!("{}", v.as_i64().unwrap()),
        }
    }
}

/// One tensor: name, type, dims (ggml order: `dims[0]` is the contiguous row length), shard and byte range.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub ty: GgmlType,
    pub dims: Vec<u64>,
    pub shard: usize,
    /// Absolute byte offset in the shard file.
    pub offset: u64,
    pub bytes: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
    /// Row length (`dims[0]`).
    pub fn ne0(&self) -> usize {
        self.dims[0] as usize
    }
    /// Number of rows over all higher dims.
    pub fn n_rows(&self) -> usize {
        (self.n_elements() / self.dims[0]) as usize
    }
    pub fn row_bytes(&self) -> usize {
        self.ty.row_bytes(self.ne0())
    }
    /// For a 3-D tensor `[ne0, ne1, ne2]` (e.g. experts: cols, rows, experts): bytes of one `ne2` slice.
    pub fn slice_bytes(&self) -> usize {
        self.row_bytes() * self.dims.get(1).copied().unwrap_or(1) as usize
    }
}

pub struct Shard {
    pub path: PathBuf,
    pub map: Mmap,
}

pub struct Gguf {
    pub version: u32,
    pub kv: Vec<(String, Value)>,
    kv_idx: HashMap<String, usize>,
    pub tensors: Vec<TensorInfo>,
    t_idx: HashMap<String, usize>,
    pub shards: Vec<Shard>,
    /// `open_partial` / `attach_partial` only: shard paths that do not exist yet and tensors whose bytes run past
    /// the end of a file still being downloaded (neither is in `tensors`).
    pub missing: Vec<String>,
}

struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.p + n > self.b.len() {
            bail!("gguf: truncated header at byte {}", self.p);
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> { Ok(self.take(1)?[0]) }
    fn u16(&mut self) -> Result<u16> { Ok(u16::from_le_bytes(self.take(2)?.try_into()?)) }
    fn u32(&mut self) -> Result<u32> { Ok(u32::from_le_bytes(self.take(4)?.try_into()?)) }
    fn u64(&mut self) -> Result<u64> { Ok(u64::from_le_bytes(self.take(8)?.try_into()?)) }
    fn str(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.str()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 24));
                for _ in 0..n {
                    v.push(self.value(et)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            t => bail!("gguf: unknown metadata type {t}"),
        })
    }
}

/// `name-00001-of-00003.gguf` -> all shard paths; any other name -> just itself.
pub fn shard_paths(first: &Path) -> Vec<PathBuf> {
    let s = first.to_string_lossy();
    if let Some(pos) = s.rfind("-of-") {
        let (pre, post) = s.split_at(pos);
        // pre ends with -NNNNN, post = -of-NNNNN.gguf
        if pre.len() >= 6 && post.len() >= 9 {
            let idx_w = pre.len() - pre.rfind('-').unwrap() - 1;
            let base = &pre[..pre.rfind('-').unwrap()];
            let n_str = &post[4..post.len() - 5];
            if let Ok(n) = n_str.parse::<usize>() {
                return (1..=n).map(|i| PathBuf::from(format!("{base}-{i:0idx_w$}{post}"))).collect();
            }
        }
    }
    vec![first.to_path_buf()]
}

impl Gguf {
    /// Open a GGUF file; a split name (`-00001-of-0000N`) opens all shards (any shard path works).
    pub fn open(path: impl AsRef<Path>) -> Result<Gguf> {
        Self::open_impl(path.as_ref(), false)
    }
    /// Like [`Gguf::open`] for a download in progress: a shard that is not there yet is skipped (listed in
    /// `missing`; its tensors are unknown), and a tensor that runs past the end of its file is left out (also in
    /// `missing`). The first shard must exist. Never use this to serve: it exists for inventory and tests.
    pub fn open_partial(path: impl AsRef<Path>) -> Result<Gguf> {
        Self::open_impl(path.as_ref(), true)
    }
    fn open_impl(path: &Path, lenient: bool) -> Result<Gguf> {
        let paths = shard_paths(path);
        let mut g = Gguf { version: 0, kv: vec![], kv_idx: HashMap::new(), tensors: vec![], t_idx: HashMap::new(), shards: vec![], missing: vec![] };
        for (si, p) in paths.iter().enumerate() {
            if lenient && si > 0 && !p.exists() {
                g.missing.push(format!("shard {}", p.display()));
                g.shards.push(Shard { path: p.clone(), map: memmap2::MmapMut::map_anon(1)?.make_read_only()? });
                continue;
            }
            let f = File::open(p).with_context(|| format!("open {}", p.display()))?;
            let map = unsafe { Mmap::map(&f) }.with_context(|| format!("mmap {}", p.display()))?;
            let (version, kv, tensors) = parse(&map, si).with_context(|| format!("parse {}", p.display()))?;
            if si == 0 {
                g.version = version;
                for (k, v) in kv {
                    g.kv_idx.insert(k.clone(), g.kv.len());
                    g.kv.push((k, v));
                }
            }
            for t in tensors {
                if t.offset + t.bytes > map.len() as u64 {
                    if lenient {
                        g.missing.push(t.name.clone());
                        continue;
                    }
                    bail!("gguf: tensor {} runs past the end of {}", t.name, p.display());
                }
                g.t_idx.insert(t.name.clone(), g.tensors.len());
                g.tensors.push(t);
            }
            g.shards.push(Shard { path: p.clone(), map });
        }
        Ok(g)
    }
    /// Add the tensors of another (single-file) GGUF as one more shard, e.g. a separate draft head next to a split
    /// model. Tensor names must not repeat; the file's metadata is returned, not merged (see [`Gguf::set`]).
    pub fn attach(&mut self, path: impl AsRef<Path>) -> Result<Vec<(String, Value)>> {
        self.attach_impl(path.as_ref(), false)
    }
    /// [`Gguf::attach`] for a file still downloading: tensors past its end are left out and listed in `missing`.
    pub fn attach_partial(&mut self, path: impl AsRef<Path>) -> Result<Vec<(String, Value)>> {
        self.attach_impl(path.as_ref(), true)
    }
    fn attach_impl(&mut self, p: &Path, lenient: bool) -> Result<Vec<(String, Value)>> {
        let f = File::open(p).with_context(|| format!("open {}", p.display()))?;
        let map = unsafe { Mmap::map(&f) }.with_context(|| format!("mmap {}", p.display()))?;
        let si = self.shards.len();
        let (_, kv, tensors) = parse(&map, si).with_context(|| format!("parse {}", p.display()))?;
        let mut keep = Vec::with_capacity(tensors.len());
        for t in tensors {
            if t.offset + t.bytes > map.len() as u64 {
                if lenient {
                    self.missing.push(t.name.clone());
                    continue;
                }
                bail!("gguf: tensor {} runs past the end of {}", t.name, p.display());
            }
            if self.t_idx.contains_key(&t.name) {
                bail!("gguf: tensor {} of {} is already in the model", t.name, p.display());
            }
            keep.push(t);
        }
        for t in keep {
            self.t_idx.insert(t.name.clone(), self.tensors.len());
            self.tensors.push(t);
        }
        self.shards.push(Shard { path: p.to_path_buf(), map });
        Ok(kv)
    }
    /// Set (or add) a metadata value in memory; the files are never written.
    pub fn set(&mut self, key: &str, v: Value) {
        match self.kv_idx.get(key) {
            Some(&i) => self.kv[i].1 = v,
            None => {
                self.kv_idx.insert(key.to_string(), self.kv.len());
                self.kv.push((key.to_string(), v));
            }
        }
    }
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.kv_idx.get(key).map(|&i| &self.kv[i].1)
    }
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(|v| v.as_i64())
    }
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(|v| v.as_f64())
    }
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(|v| v.as_str())
    }
    /// `general.architecture` (e.g. "qwen4exp").
    pub fn arch(&self) -> &str {
        self.get_str("general.architecture").unwrap_or("")
    }
    /// `<arch>.<key>`.
    pub fn arch_get(&self, key: &str) -> Option<&Value> {
        self.get(&format!("{}.{key}", self.arch()))
    }
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.t_idx.get(name).map(|&i| &self.tensors[i])
    }
    /// The tensor's raw bytes (borrowed from the mmap).
    pub fn data(&self, t: &TensorInfo) -> &[u8] {
        &self.shards[t.shard].map[t.offset as usize..(t.offset + t.bytes) as usize]
    }
    /// Bytes of expert/slice `i` of a 3-D tensor (`dims[2]` slices).
    pub fn slice(&self, t: &TensorInfo, i: usize) -> &[u8] {
        let sb = t.slice_bytes();
        &self.data(t)[i * sb..(i + 1) * sb]
    }
    /// madvise(WILLNEED) / (DONTNEED) helpers are left to callers (`memmap2::Mmap::advise_range`).
    pub fn total_tensor_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.bytes).sum()
    }
}

fn parse(b: &[u8], shard: usize) -> Result<(u32, Vec<(String, Value)>, Vec<TensorInfo>)> {
    let mut c = Cur { b, p: 0 };
    if c.take(4)? != b"GGUF" {
        bail!("gguf: bad magic");
    }
    let version = c.u32()?;
    if !(2..=3).contains(&version) {
        bail!("gguf: unsupported version {version}");
    }
    let nt = c.u64()? as usize;
    let nkv = c.u64()? as usize;
    let mut kv = Vec::with_capacity(nkv);
    let mut align = 32u64;
    for _ in 0..nkv {
        let k = c.str()?;
        let ty = c.u32()?;
        let v = c.value(ty)?;
        if k == "general.alignment" {
            align = v.as_i64().unwrap_or(32) as u64;
        }
        kv.push((k, v));
    }
    let mut ts = Vec::with_capacity(nt);
    for _ in 0..nt {
        let name = c.str()?;
        let nd = c.u32()? as usize;
        let mut dims = Vec::with_capacity(nd);
        for _ in 0..nd {
            dims.push(c.u64()?);
        }
        let tyv = c.u32()?;
        let ty = GgmlType::from_u32(tyv).with_context(|| format!("tensor {name}: unknown ggml type {tyv}"))?;
        let off = c.u64()?;
        let n: u64 = dims.iter().product();
        let (bs, by) = ty.block();
        if dims[0] % bs as u64 != 0 {
            bail!("tensor {name}: row {} not a multiple of the {} block {}", dims[0], ty.name(), bs);
        }
        let bytes = n / bs as u64 * by as u64;
        ts.push(TensorInfo { name, ty, dims, shard, offset: off, bytes });
    }
    let data_start = (c.p as u64).div_ceil(align) * align;
    for t in &mut ts {
        t.offset += data_start;
    }
    Ok((version, kv, ts))
}
