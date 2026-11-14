//! Print the model config and the VRAM placement plan: fnext-plan [ctx] [card MiB]
use anyhow::Result;
use fe_fnext::{config::Config, plan::Plan};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let ctx = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(131072);
    let card_mib: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(16311);
    let g = fe_fnext::open_model(&fe_fnext::default_model())?;
    if let Ok(pre) = std::env::var("FNEXT_LIST") {
        for t in &g.tensors {
            if t.name.starts_with(&pre) {
                println!("{} {:?} {:?}", t.name, t.ty, t.dims);
            }
        }
        return Ok(());
    }
    let c = Config::from_gguf(&g)?;
    println!("{c:#?}");
    let p = Plan::new(&c, &g, card_mib << 20, ctx, 2048, 0.7, true)?;
    p.print(&c);
    Ok(())
}
