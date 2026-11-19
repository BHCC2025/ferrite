//! fe-server: the OpenAI-compatible HTTP front end of ferrite.
//!
//! The server owns everything text-shaped (chat template, tokenizer, streaming detokenization, stop
//! strings, reasoning and tool-call parsing, OpenAI JSON / SSE); the model engine behind
//! [`engine::Engine`] only maps prompt token ids to generated token ids.
pub mod api;
pub mod batch;
pub mod engine;
#[cfg(feature = "fnext")]
pub mod fnext;
#[cfg(feature = "ferrite")]
pub mod ferrite;
pub mod guard;
pub mod loopdet;
pub mod mock;
pub mod parser;
pub mod pyjson;
pub mod sampler;
pub mod server;
pub mod snapshots;
pub mod stop;
pub mod template;
pub mod tokenizer;
pub mod toolcall;
pub mod util;
#[cfg(feature = "xe")]
pub mod xe;

/// Live bytes the Rust code holds on the heap (every allocation goes through this counting wrapper of the system
/// allocator), for the engines' memory logs: compared with RssAnon it tells Rust-side growth from growth inside C
/// libraries (the CUDA driver).
pub fn rust_heap_bytes() -> usize {
    LIVE.load(std::sync::atomic::Ordering::Relaxed).max(0) as usize
}

static LIVE: std::sync::atomic::AtomicIsize = std::sync::atomic::AtomicIsize::new(0);

struct Counting;

// SAFETY: forwards to the system allocator unchanged; only adds a relaxed counter
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: std::alloc::Layout) -> *mut u8 {
        let p = unsafe { std::alloc::System.alloc(l) };
        if !p.is_null() {
            LIVE.fetch_add(l.size() as isize, std::sync::atomic::Ordering::Relaxed);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: std::alloc::Layout) -> *mut u8 {
        let p = unsafe { std::alloc::System.alloc_zeroed(l) };
        if !p.is_null() {
            LIVE.fetch_add(l.size() as isize, std::sync::atomic::Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(p, l) };
        LIVE.fetch_sub(l.size() as isize, std::sync::atomic::Ordering::Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: std::alloc::Layout, n: usize) -> *mut u8 {
        let q = unsafe { std::alloc::System.realloc(p, l, n) };
        if !q.is_null() {
            LIVE.fetch_add(n as isize - l.size() as isize, std::sync::atomic::Ordering::Relaxed);
        }
        q
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;
