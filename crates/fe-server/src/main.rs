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

/// `fe-server --version`: the release and the engines this binary was built with
static VERSION: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    let mut e = vec!["mock"];
    for (on, name) in [(cfg!(feature = "ferrite"), "ferrite"), (cfg!(feature = "fnext"), "fnext"), (cfg!(feature = "dsc"), "dsc"), (cfg!(feature = "xe"), "xe")] {
        if on {
            e.push(name);
        }
    }
    #[cfg(feature = "ferrite")]
    let karch = Some(fe_model::KARCH);
    #[cfg(all(feature = "fnext", not(feature = "ferrite")))]
    let karch = Some(fe_fnext::KARCH);
    #[cfg(not(any(feature = "ferrite", feature = "fnext")))]
    let karch: Option<&str> = None;
    match karch {
        Some(k) => format!("{} (engines: {}; kernels {k})", env!("CARGO_PKG_VERSION"), e.join(" ")),
        None => format!("{} (engines: {})", env!("CARGO_PKG_VERSION"), e.join(" ")),
    }
});

#[derive(Parser, Debug)]
#[command(name = "fe-server", about = "OpenAI-compatible server for the ferrite engine", version = VERSION.as_str())]
struct Args {
    /// Model directory (tokenizer.json, chat_template.jinja, generation_config.json, weights).
    #[arg(long, env = "FERRITE_EXL3_DIR")]
    model_dir: PathBuf,
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// Model id reported by the API (requests may name any model).
    #[arg(long, env = "FERRITE_MODEL_ID", default_value = "qwen3.8-27b")]
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
    // still single-threaded: no other thread allocates yet
    fixed_malloc_thresholds(!args.mock && args.engine == "fnext");
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
        // colours only on a terminal: a log file gets plain text
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stdout()))
        .init();
    tracing::info!("fe-server {}", *VERSION);
    // bind before loading: a port in use fails at once instead of after the model load; connections made while the
    // model loads wait in the listen queue and are served once it is ready
    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("binding {addr}"))?;
    if !args.mock {
        check_gpu_arch(&args.engine)?;
    }
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
                cached_prompt_tokens: 0,
            },
        ))
    } else {
        if args.engine == "fake" { fake_engine(&args, &tok)? } else if args.engine == "fnext" { fnext_engine(&args)? } else if args.engine == "dsc" { dsc_engine(&args)? } else if args.engine == "xe" || args.engine == "xe-batch" { xe_engine(&args)? } else { real_engine(&args)? }
    };
    let info = engine.info();
    let cfg = ServerConfig {
        model_id: args.served_model_name.clone(),
        max_concurrent: args.max_concurrent.unwrap_or(info.max_batch).max(1),
        // an engine with a shared context pool (FE_POOL) holds more than --max-model-len in one request
        max_context: if info.max_context > args.max_model_len && std::env::var("FE_POOL").is_ok_and(|v| v == "1") { info.max_context } else { args.max_model_len.min(info.max_context) },
        defaults,
    };
    if args.max_concurrent.is_some_and(|n| n > info.max_batch) {
        tracing::warn!(
            "--max-concurrent {} is above the {} requests the {} engine decodes at once: running {} at a time",
            cfg.max_concurrent, info.max_batch, info.name, info.max_batch
        );
    }
    // AppState::new keeps both limits within what the engine reports: /health never claims more
    let state = AppState::new(engine, tok, template, cfg);
    let cfg = &state.cfg;
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
    // an engine that stops for good (a panic, a lost TP2 peer) takes the process down with a non-zero status, after a
    // moment in which /health and new requests answer 503 with the reason: a launcher or systemd restarts it
    let watch = state.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(250)).await;
            if let Some(why) = watch.engine.dead() {
                tracing::error!("the engine stopped ({why}); exiting with status {ENGINE_DIED} so that it can be restarted");
                tokio::time::sleep(Duration::from_secs(2)).await;
                // _exit, not exit: a hung engine thread (a TP2 collective whose peer is gone) would block the CUDA
                // driver's exit handlers forever
                unsafe extern "C" {
                    fn _exit(code: i32) -> !;
                }
                unsafe { _exit(ENGINE_DIED) }
            }
        }
    });
    tracing::info!("listening on http://{addr}");
    axum::serve(listener, router(state)).with_graceful_shutdown(shutdown()).await?;
    Ok(())
}

/// BatchEngine over the CPU fake model, with the real vocabulary and eos (`--engine fake`).
/// The engine's kernels must match the visible cards' architecture (else cuModuleLoadData fails after the load).
fn check_gpu_arch(engine: &str) -> Result<()> {
    let r: std::result::Result<(), String> = match engine {
        #[cfg(feature = "fnext")]
        "fnext" => fe_fnext::check_arch(),
        #[cfg(feature = "ferrite")]
        "ferrite" | "ferrite-batch" => fe_model::check_arch(),
        _ => Ok(()),
    };
    r.map_err(|e| anyhow::anyhow!(e))
}

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

/// `--engine dsc`: DeepSeek-V4.1-Flash Coder from GGUF on three cards (DSC_* environment, see fe_server::dsc).
#[cfg(feature = "dsc")]
fn dsc_engine(args: &Args) -> Result<Arc<dyn Engine>> {
    Ok(Arc::new(fe_server::dsc::DscEngine::start(args.max_model_len)?))
}

#[cfg(not(feature = "dsc"))]
fn dsc_engine(_: &Args) -> Result<Arc<dyn Engine>> {
    bail!("this fe-server was built without the DeepSeek engine: build with --features dsc")
}

/// `--engine fnext`: Qwen3.8-Flash-Next from GGUF (FNEXT_MODEL, FNEXT_COUNTS, FNEXT_THREADS, FNEXT_MTP ...).
#[cfg(feature = "fnext")]
fn fnext_engine(args: &Args) -> Result<Arc<dyn Engine>> {
    Ok(Arc::new(fe_server::fnext::FnextEngine::start(args.max_model_len)?))
}

#[cfg(not(feature = "fnext"))]
fn fnext_engine(_: &Args) -> Result<Arc<dyn Engine>> {
    bail!("this fe-server was built without Flash-Next: build with --features fnext")
}

/// Exit status when the engine stopped serving (EX_SOFTWARE).
const ENGINE_DIED: i32 = 70;

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

/// glibc raises its mmap threshold (up to 32 MiB) each time a large block is freed, so the MB-sized buffers the
/// engines make every step and every prompt (logit rows, embeddings, PLE rows) end up in the per-thread heaps, where
/// freed space fragments and is not given back: a 4-client Flash-Next soak grew by ~2.5 GiB an hour that way, with
/// the live Rust heap flat. Fixed thresholds (128 KiB) put such blocks in their own mappings, returned on free
/// (the same soak then stays flat). Set before any thread exists; MALLOC_MMAP_THRESHOLD_ / MALLOC_TRIM_THRESHOLD_
/// in the environment still win. On by default for `--engine fnext` only: the 27B engine creeps ~0.6 MB/min with
/// glibc's defaults (an RTX 5080 soak) and decodes 1.6% slower with the fixed thresholds. FERRITE_MALLOC=1 / 0
/// turns them on / off for any engine.
fn fixed_malloc_thresholds(by_default: bool) {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    if match std::env::var("FERRITE_MALLOC").as_deref() { Ok("1") => true, Ok("0") => false, _ => by_default } {
        unsafe extern "C" {
            fn mallopt(param: i32, value: i32) -> i32;
        }
        const M_TRIM_THRESHOLD: i32 = -1;
        const M_MMAP_THRESHOLD: i32 = -3;
        if std::env::var_os("MALLOC_MMAP_THRESHOLD_").is_none() {
            unsafe { mallopt(M_MMAP_THRESHOLD, 128 << 10) };
        }
        if std::env::var_os("MALLOC_TRIM_THRESHOLD_").is_none() {
            unsafe { mallopt(M_TRIM_THRESHOLD, 128 << 10) };
        }
    }
}
