//! `fe-server` binary: `fe-server --model-dir DIR [--mock | --engine ferrite --weights-dir W] [--host H] [--port P]`.
//! Site settings come from flags or their FERRITE_* environment variables (fe_site; the README's table).
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;

use fe_server::engine::Engine;
use fe_server::mock::{MockConfig, MockEngine};
use fe_server::server::{AppState, GenDefaults, ServerConfig, router};
use fe_server::template::ChatTemplate;
use fe_server::tokenizer::Tok;

#[derive(Parser, Debug)]
#[command(name = "fe-server", about = "OpenAI-compatible server for the ferrite engine")]
struct Args {
    /// Model directory (tokenizer.json, chat_template.jinja, generation_config.json, weights).
    #[arg(long, env = "FERRITE_EXL3_DIR")]
    model_dir: PathBuf,
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// Model id reported by the API (requests may name any model).
    #[arg(long, env = "FERRITE_MODEL_ID", default_value = "spark")]
    served_model_name: String,
    /// Real engine: the NVIDIA NVFP4 / FP8 checkpoint for prompt rows (FE_PROMPT=fp4); default fe_site::nvfp4_dir.
    #[arg(long, env = "FERRITE_NVFP4_DIR")]
    nvfp4_dir: Option<PathBuf>,
    /// Real engine: the DFlash2 drafter checkpoint (FE_DRAFT=dflash); default fe_site::draft_dir.
    #[arg(long, env = "FERRITE_DRAFT_DIR")]
    draft_dir: Option<PathBuf>,
    /// Use the deterministic mock engine instead of the real one.
    #[arg(long)]
    mock: bool,
    /// Engine when not --mock: `ferrite` = core's fe_model::sched::Scheduler, `ferrite-batch` = fe-server's
    /// BatchEngine over fe_model::model::Engine (both need `--features ferrite`); `fake` = BatchEngine over
    /// a CPU fake model (random tokens; tests the real-engine path without a GPU).
    #[arg(long, default_value = "ferrite")]
    engine: String,
    /// Weights (config.json + safetensors, BF16 or EXL3) for the real engine; default: --model-dir.
    #[arg(long)]
    weights_dir: Option<PathBuf>,
    /// Real engine: cache slots (= concurrent requests).
    #[arg(long, default_value_t = 4)]
    slots: usize,
    /// Real engine: prompt tokens per prefill chunk.
    #[arg(long, default_value_t = 2048)]
    prefill_chunk: usize,
    /// Mock: milliseconds per decode step.
    #[arg(long, default_value_t = 5)]
    mock_delay_ms: u64,
    /// Mock: tokens per decode step (>1 imitates speculative decoding).
    #[arg(long, default_value_t = 1)]
    mock_burst: usize,
    /// Mock: simulated prefill speed in tokens/s (0 = instant).
    #[arg(long, default_value_t = 0.0)]
    mock_prefill_tps: f64,
    /// Requests generating at once (default: the engine's batch size); more wait in a queue.
    #[arg(long)]
    max_concurrent: Option<usize>,
    /// Context window in tokens (prompt + output).
    #[arg(long, default_value_t = 131_072)]
    max_model_len: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    // the engine reads its checkpoint folders from the environment (fe_site): hand it the flags before any thread
    // starts (a TP2 peer gets them through its own environment, tools/site.sh)
    for (k, v) in [("FERRITE_NVFP4_DIR", &args.nvfp4_dir), ("FERRITE_DRAFT_DIR", &args.draft_dir)] {
        if let Some(v) = v {
            // SAFETY: still single-threaded (the runtime and the engine thread start below)
            unsafe { std::env::set_var(k, v) };
        }
    }
    tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(serve(args))
}

async fn serve(args: Args) -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let tok = Arc::new(Tok::from_dir(&args.model_dir)?);
    let template = Arc::new(ChatTemplate::from_dir(&args.model_dir)?);
    let defaults = GenDefaults::from_dir(&args.model_dir, &tok);
    let engine: Arc<dyn Engine> = if args.mock {
        Arc::new(MockEngine::new(
            tok.clone(),
            MockConfig {
                delay: Duration::from_millis(args.mock_delay_ms),
                burst: args.mock_burst.max(1),
                prefill_tps: args.mock_prefill_tps,
                max_batch: args.max_concurrent.unwrap_or(4),
                max_context: args.max_model_len,
            },
        ))
    } else {
        if args.engine == "fake" { fake_engine(&args, &tok)? } else if args.engine == "xe" || args.engine == "xe-batch" { xe_engine(&args)? } else { real_engine(&args)? }
    };
    let info = engine.info();
    let cfg = ServerConfig {
        model_id: args.served_model_name.clone(),
        max_concurrent: args.max_concurrent.unwrap_or(info.max_batch).max(1),
        max_context: args.max_model_len.min(info.max_context),
        defaults,
    };
    tracing::info!(
        "engine {} | model id {} | {} concurrent | context {} | eos {:?} | defaults temp {} top_p {} top_k {}",
        info.name,
        cfg.model_id,
        cfg.max_concurrent,
        cfg.max_context,
        cfg.defaults.eos_ids,
        cfg.defaults.temperature,
        cfg.defaults.top_p,
        cfg.defaults.top_k
    );
    let state = AppState::new(engine, tok, template, cfg);
    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("binding {addr}"))?;
    tracing::info!("listening on http://{addr}");
    axum::serve(listener, router(state)).with_graceful_shutdown(shutdown()).await?;
    Ok(())
}

/// BatchEngine over the CPU fake model, with the real vocabulary and eos (`--engine fake`).
fn fake_engine(args: &Args, tok: &Tok) -> Result<Arc<dyn Engine>> {
    use fe_server::batch::fake::FakeModel;
    use fe_server::batch::{BatchConfig, BatchEngine};
    let (slots, ctx, chunk) = (args.slots.max(1), args.max_model_len, args.prefill_chunk.max(16));
    let vocab = tok.vocab_size();
    let eos = tok.token_id("<|im_end|>").unwrap_or(0);
    let text_vocab = tok.token_id("<|endoftext|>").map(|t| t as usize).unwrap_or(vocab);
    let delay = Duration::from_millis(args.mock_delay_ms);
    let row_delay =
        if args.mock_prefill_tps > 0.0 { Duration::from_secs_f64(1.0 / args.mock_prefill_tps) } else { Duration::ZERO };
    let cfg = BatchConfig { name: "fake".into(), prefill_chunk: chunk };
    Ok(Arc::new(BatchEngine::spawn(cfg, move || {
        let mut m = FakeModel::new(slots, ctx, chunk + slots, vocab, eos);
        m.text_vocab = text_vocab;
        m.eos_period = 997;
        m.delay = delay;
        m.row_delay = row_delay;
        Ok(m)
    })?))
}

#[cfg(feature = "ferrite")]
fn real_engine(args: &Args) -> Result<Arc<dyn Engine>> {
    use fe_server::batch::{BatchConfig, BatchEngine};
    use fe_server::ferrite::FeModel;
    let dir = args.weights_dir.clone().unwrap_or_else(|| args.model_dir.clone());
    let (slots, ctx, chunk) = (args.slots.max(1), args.max_model_len, args.prefill_chunk.max(16));
    match args.engine.as_str() {
        "ferrite" => Ok(Arc::new(fe_server::ferrite::sched::FerriteEngine::start(&dir, slots, ctx, chunk)?)),
        "ferrite-batch" => {
            // one forward holds a prefill chunk plus one decode row per slot
            let rows = chunk + slots;
            let cfg = BatchConfig { name: "ferrite-batch".into(), prefill_chunk: chunk };
            Ok(Arc::new(BatchEngine::spawn(cfg, move || FeModel::load(&dir, slots, ctx, rows))?))
        }
        other => bail!("unknown --engine {other} (choose ferrite, ferrite-batch, fake)"),
    }
}

#[cfg(not(feature = "ferrite"))]
fn real_engine(_: &Args) -> Result<Arc<dyn Engine>> {
    bail!("this fe-server was built without the real engine: run with --mock, or build inside the ferrite \
           workspace with --features ferrite (see README)")
}

/// `--engine xe`: the Intel Arc B70 engine (fe-xe), shaped like arc-engine's server (fe_server::xe::XeEngine); `--engine xe-batch`: fe-xe behind the
/// BatchEngine (one slot, prompt windows of --prefill-chunk rows, plain decode).
#[cfg(feature = "xe")]
fn xe_engine(args: &Args) -> Result<Arc<dyn Engine>> {
    use fe_server::batch::{BatchConfig, BatchEngine};
    let dir = args.weights_dir.clone().unwrap_or_else(|| args.model_dir.clone());
    let (ctx, rcap) = (args.max_model_len, args.prefill_chunk.clamp(16, 4096));
    if args.engine == "xe" {
        // arc-engine's serving shape: FIFO, GDN-state checkpoints, speculative decoding (env: FE_XE_K, FE_XE_CKPT_SLOTS, MTP_DHEAD)
        let opts = fe_server::xe::XeOpts::from_env(ctx, rcap)?;
        return Ok(Arc::new(fe_server::xe::XeEngine::start(&dir, opts)?));
    }
    let cfg = BatchConfig { name: "xe-batch".into(), prefill_chunk: rcap };
    Ok(Arc::new(BatchEngine::spawn(cfg, move || fe_server::xe::XeModel::load(&dir, ctx, rcap))?))
}

#[cfg(not(feature = "xe"))]
fn xe_engine(_: &Args) -> Result<Arc<dyn Engine>> {
    bail!("this fe-server was built without the B70 engine: build with --features xe")
}

async fn shutdown() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
    tracing::info!("shutting down (open streams get 5 s to finish)");
    // graceful shutdown waits for every open connection; do not let a long stream hold the exit
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(5));
        std::process::exit(0);
    });
}
