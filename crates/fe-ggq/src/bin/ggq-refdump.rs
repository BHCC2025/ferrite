//! ggq-refdump <model.gguf> <outdir>: for every quant type in the file, take the first tensor of that type, dump
//! up to 64 rows of raw blocks (<ty>.bin) and our host reference dequant (<ty>.ref.f32). tools/ggml-deq then
//! dequantizes the same bytes with libggml and ggq-refdump --cmp compares.
use fe_gguf::Gguf;
use std::collections::BTreeMap;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let g = Gguf::open(&a[1])?;
    let out = std::path::Path::new(&a[2]);
    std::fs::create_dir_all(out)?;
    let cmp = a.iter().any(|x| x == "--cmp");
    let mut first: BTreeMap<u32, &fe_gguf::TensorInfo> = BTreeMap::new();
    for t in &g.tensors {
        first.entry(t.ty as u32).or_insert(t);
    }
    for (tyid, t) in first {
        if !fe_ggq::refq::supported(t.ty) {
            println!("skip {}", t.ty.name());
            continue;
        }
        let rows = t.n_rows().min(64);
        let n = rows * t.ne0();
        let bytes = &g.data(t)[..t.ty.row_bytes(t.ne0()) * rows];
        let mut y = vec![0f32; n];
        fe_ggq::refq::dequant(t.ty, bytes, &mut y);
        let name = t.ty.name();
        if cmp {
            let gg = std::fs::read(out.join(format!("{name}.ggml.f32")))?;
            let gg: Vec<f32> = gg.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            let mut nbad = 0usize;
            let mut maxd = 0f32;
            for (a, b) in y.iter().zip(&gg) {
                if a.to_bits() != b.to_bits() {
                    nbad += 1;
                    maxd = maxd.max((a - b).abs());
                }
            }
            println!("{name:8} {n:8} values from {:40} bit-exact: {} (mismatches {nbad}, max abs diff {maxd:e})",
                t.name, nbad == 0);
        } else {
            std::fs::write(out.join(format!("{name}.bin")), bytes)?;
            std::fs::write(out.join(format!("{name}.ref.f32")), y.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            println!("{tyid} {name} {n}");
        }
    }
    Ok(())
}
