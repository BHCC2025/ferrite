//! dsc-mkmodel <out dir> <chat_template.jinja> (scripts/fetch-deepseek.sh runs it): write the serving folder fe-server reads (tokenizer.json built from the GGUF,
//! the chat template, tokenizer_config.json with bos / eos, generation_config.json: greedy default, eos id).
use anyhow::{anyhow, Result};
use fe_dsc::{config::DscConfig, tok};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let out = std::path::PathBuf::from(&a[0]);
    std::fs::create_dir_all(&out)?;
    let path = fe_dsc::model_path();
    let eng = fe_dsc::engram_path(&path);
    let g = fe_dsc::open_model(&path, eng.as_deref(), true)?;
    let c = DscConfig::from_gguf(&g)?;
    let t = tok::build(&g, &c)?;
    t.save(out.join("tokenizer.json"), false).map_err(|e| anyhow!("{e}"))?;
    std::fs::copy(&a[1], out.join("chat_template.jinja"))?;
    let eos = t.token_to_id("<｜end▁of▁sentence｜>").ok_or_else(|| anyhow!("no eos token"))?;
    std::fs::write(out.join("tokenizer_config.json"), r#"{"bos_token":"<｜begin▁of▁sentence｜>","eos_token":"<｜end▁of▁sentence｜>"}"#)?;
    std::fs::write(out.join("generation_config.json"), format!(r#"{{"do_sample":false,"temperature":0.0,"eos_token_id":[{eos}]}}"#))?;
    println!("wrote {} (eos id {eos})", out.display());
    Ok(())
}
