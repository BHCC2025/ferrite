//! A persistent spin pool for latency-critical fork/join (one MoE layer every few hundred microseconds).
//!
//! - Workers can be pinned to given CPUs ([`Pool::pinned`]); they spin on a generation counter between jobs (for
//!   `spin`, then park), so a job starts on every core within ~a microsecond, no syscalls.
//! - [`Pool::run`]: `f(i, n)` on all `n` threads (the caller is thread 0); returns when every thread has run it.
//!   For statically split work: a preempted worker delays the call.
//! - [`Pool::run_dyn`]: for work-claiming jobs (units taken from a shared counter until none are left, so any subset
//!   of threads finishes the work): returns as soon as the caller has run `f` and no worker is inside it. Workers
//!   that wake late see the job closed and skip it; nothing waits for them. Lock-free (no mutex on the hot path).
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

type Job = *const (dyn Fn(usize, usize) + Sync);

struct Shared {
    /// 2 * generation + 1 while job `generation` is open, 2 * generation once it is closed
    state: AtomicU64,
    running: AtomicUsize,
    /// workers that finished the current job (run)
    done: AtomicUsize,
    job: UnsafeCell<Option<Job>>,
    stop: AtomicBool,
    n: usize,
    spin: Duration,
}
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct Pool {
    sh: Arc<Shared>,
    workers: Vec<(JoinHandle<()>, Thread)>,
}

/// Pin the calling thread to one CPU (Linux; no-op elsewhere). Returns false if the kernel refused.
pub fn pin_current(cpu: usize) -> bool {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cpu;
        false
    }
}

/// Parse a CPU list like "0-7" or "1-7,16,18-19".
pub fn parse_cpus(s: &str) -> Vec<usize> {
    let mut v = vec![];
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        if let Some((a, b)) = part.split_once('-') {
            if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                v.extend(a..=b);
            }
        } else if let Ok(a) = part.parse() {
            v.push(a);
        }
    }
    v
}

impl Pool {
    /// `n` threads in total (the caller counts as one), unpinned.
    pub fn new(n: usize, spin: Duration) -> Pool {
        Self::build(n.max(1), None, spin)
    }
    /// One worker pinned to each of `cpus`, plus the caller as thread 0 (`n = cpus.len() + 1`). Pin the caller
    /// yourself if you want ([`pin_current`]).
    pub fn pinned(cpus: &[usize], spin: Duration) -> Pool {
        Self::build(cpus.len() + 1, Some(cpus.to_vec()), spin)
    }
    fn build(n: usize, cpus: Option<Vec<usize>>, spin: Duration) -> Pool {
        let sh = Arc::new(Shared {
            state: AtomicU64::new(0),
            running: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            job: UnsafeCell::new(None),
            stop: AtomicBool::new(false),
            n,
            spin,
        });
        let mut workers = vec![];
        for i in 1..n {
            let sh2 = sh.clone();
            let cpu = cpus.as_ref().map(|c| c[i - 1]);
            let h = thread::Builder::new()
                .name(format!("ggq-cpu-{i}"))
                .spawn(move || {
                    if let Some(c) = cpu {
                        pin_current(c);
                    }
                    worker(sh2, i)
                })
                .unwrap();
            let t = h.thread().clone();
            workers.push((h, t));
        }
        Pool { sh, workers }
    }
    /// Threads in total (caller included).
    pub fn n(&self) -> usize {
        self.sh.n
    }
    fn post(&self, f: &(dyn Fn(usize, usize) + Sync)) {
        // SAFETY: the job is closed and every worker inside it has left before run/run_dyn return.
        let job: Job = unsafe { std::mem::transmute::<&(dyn Fn(usize, usize) + Sync), Job>(f) };
        unsafe { *self.sh.job.get() = Some(job) };
        self.sh.done.store(0, SeqCst);
        let st = self.sh.state.load(SeqCst);
        self.sh.state.store((st & !1) + 3, SeqCst); // next generation, open
        for (_, t) in &self.workers {
            t.unpark();
        }
    }
    fn close(&self) {
        let st = self.sh.state.load(SeqCst);
        self.sh.state.store(st & !1, SeqCst);
        while self.sh.running.load(SeqCst) != 0 {
            std::hint::spin_loop();
        }
    }
    /// Run `f(thread_index, n_threads)` on every thread; returns when all have finished.
    pub fn run<F: Fn(usize, usize) + Sync>(&self, f: F) {
        let n = self.sh.n;
        if n == 1 {
            f(0, 1);
            return;
        }
        self.post(&f);
        f(0, n);
        while self.sh.done.load(SeqCst) != n - 1 {
            std::hint::spin_loop();
        }
        self.close();
    }
    /// Run a work-claiming `f` (see the module docs).
    pub fn run_dyn<F: Fn(usize, usize) + Sync>(&self, f: F) {
        let n = self.sh.n;
        if n == 1 {
            f(0, 1);
            return;
        }
        self.post(&f);
        f(0, n);
        self.close();
    }
}

fn worker(sh: Arc<Shared>, i: usize) {
    let mut seen = 0u64;
    loop {
        let t0 = Instant::now();
        let mut k = 0u32;
        let st = loop {
            if sh.stop.load(SeqCst) {
                return;
            }
            let st = sh.state.load(SeqCst);
            if st & 1 == 1 && st != seen {
                break st;
            }
            std::hint::spin_loop();
            k += 1;
            if k % 4096 == 0 && t0.elapsed() > sh.spin {
                thread::park_timeout(Duration::from_millis(20));
            }
        };
        seen = st;
        sh.running.fetch_add(1, SeqCst);
        if sh.state.load(SeqCst) == st {
            let f = unsafe { &*(*sh.job.get()).unwrap() };
            f(i, sh.n);
            sh.done.fetch_add(1, SeqCst);
        }
        sh.running.fetch_sub(1, SeqCst);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.sh.stop.store(true, SeqCst);
        for (h, t) in self.workers.drain(..) {
            t.unpark();
            let _ = h.join();
        }
    }
}
