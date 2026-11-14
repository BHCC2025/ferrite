//! The process core plan: which CPU the step launcher, the CPU expert thread and the uploader / prefetch thread
//! run on, and which CPUs the CPU expert pool owns (one pinned worker each).
//!
//! FNEXT_CORES="main=0,cpu=1,up=2,pool=8-23" overrides any part (FNEXT_CPU_POOL still sets the pool). Defaults:
//! main 0, cpu 1, up 2 (the first allowed CPUs); the pool is the E-cores on a hybrid Intel part
//! (/sys/devices/cpu_atom/cpus), else one thread per physical core of the rest (e.g. a 12-core 5900X: cores 3..11).
//! Only CPUs the process may run on (taskset / cgroup) are used.
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct Cores {
    pub main: usize,
    pub cpu: usize,
    pub up: usize,
    pub pool: Vec<usize>,
    /// every CPU the process may run on (read before any thread is pinned)
    pub all: Vec<usize>,
}

fn parse_list(s: &str) -> Vec<usize> {
    let mut v = vec![];
    for part in s.trim().split(',').filter(|p| !p.is_empty()) {
        if let Some((a, b)) = part.split_once('-') {
            if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                v.extend(a..=b);
            }
        } else if let Ok(a) = part.trim().parse::<usize>() {
            v.push(a);
        }
    }
    v
}

fn allowed() -> Vec<usize> {
    std::fs::read_to_string("/proc/self/status").ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("Cpus_allowed_list:")).map(|l| parse_list(&l[18..])))
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| (0..std::thread::available_parallelism().map_or(1, |n| n.get())).collect())
}

/// The first CPU of each physical core among `cpus` (SMT siblings dropped).
fn one_per_core(cpus: &[usize]) -> Vec<usize> {
    let mut seen: Vec<Vec<usize>> = vec![];
    let mut out = vec![];
    for &c in cpus {
        let sib = std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"))
            .map(|s| parse_list(&s)).unwrap_or_else(|_| vec![c]);
        if !seen.contains(&sib) {
            seen.push(sib);
            out.push(c);
        }
    }
    out
}

fn siblings(c: usize) -> Vec<usize> {
    std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"))
        .map(|s| parse_list(&s)).unwrap_or_else(|_| vec![c])
}

fn auto() -> Cores {
    let al = allowed();
    let pick = |i: usize| al.get(i).copied().unwrap_or(0);
    let (main, cpu, up) = (pick(0), pick(1), pick(2));
    let mut busy: Vec<usize> = vec![];
    for c in [main, cpu, up] {
        busy.extend(siblings(c));
    }
    let atom = std::fs::read_to_string("/sys/devices/cpu_atom/cpus").map(|s| parse_list(&s)).unwrap_or_default();
    let mut pool: Vec<usize> = if !atom.is_empty() {
        atom.into_iter().filter(|c| al.contains(c) && !busy.contains(c)).collect()
    } else {
        one_per_core(&al.iter().copied().filter(|c| !busy.contains(c)).collect::<Vec<_>>())
    };
    if pool.is_empty() {
        pool = vec![cpu];
    }
    Cores { main, cpu, up, pool, all: al }
}

/// The plan for this process (computed once, printed with FNEXT_VERBOSE).
pub fn plan() -> &'static Cores {
    static P: OnceLock<Cores> = OnceLock::new();
    P.get_or_init(|| {
        let mut p = auto();
        if let Ok(s) = std::env::var("FNEXT_CPU_POOL") {
            p.pool = parse_list(&s);
        }
        if let Ok(s) = std::env::var("FNEXT_CORES") {
            for kv in s.split(|c| c == ';' || c == ' ') {
                if let Some((k, v)) = kv.split_once('=') {
                    match k.trim() {
                        "main" => p.main = v.trim().parse().unwrap_or(p.main),
                        "cpu" => p.cpu = v.trim().parse().unwrap_or(p.cpu),
                        "up" => p.up = v.trim().parse().unwrap_or(p.up),
                        "pool" => p.pool = parse_list(v),
                        _ => {}
                    }
                }
            }
        }
        if std::env::var("FNEXT_VERBOSE").is_ok() {
            eprintln!("fnext-fast: cores {p:?}");
        }
        p
    })
}

/// Let the calling thread run on every CPU of the process again (threads inherit the affinity of the thread that
/// spawns them: anything started from the pinned launching thread would otherwise share its one CPU).
pub fn unpin_current() {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for &c in &plan().all {
            libc::CPU_SET(c, &mut set);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

/// Build rayons global pool with every worker on all of the processs CPUs (it is otherwise built lazily by the
/// first parallel call, from whatever thread makes it: the pinned launching thread put all workers on CPU 0).
pub fn init_rayon() {
    let _ = plan();
    let _ = rayon::ThreadPoolBuilder::new().start_handler(|_| unpin_current()).build_global();
}
