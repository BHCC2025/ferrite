//! leakprobe [test] [iters]: repeat one CUDA driver pattern the Flash-Next prompt path uses and print RssAnon,
//! to find host memory the driver keeps. Tests: thread (a thread that binds the context, makes a stream and two
//! events, then exits), host (48 MiB pinned alloc + free), vm (grow + shrink a 512 MiB virtual buffer), event.
use fe_cuda::*;
fn anon() -> usize {
    std::fs::read_to_string("/proc/self/status").unwrap().lines().find(|l| l.starts_with("RssAnon:"))
        .and_then(|l| l.split_whitespace().nth(1)?.parse().ok()).unwrap_or(0usize) >> 10
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let test = a.get(1).cloned().unwrap_or("thread".into());
    let n: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(2000);
    let cu = Context::new(0).unwrap();
    let mut vm = VmBuffer::reserve(1 << 30, 0).unwrap();
    println!("{test}: start anon {} MiB", anon());
    for i in 1..=n {
        match test.as_str() {
            "thread" => {
                let c = cu.clone();
                std::thread::spawn(move || {
                    c.bind().unwrap();
                    let s = Stream::new().unwrap();
                    let e = [Event::new_blocking().unwrap(), Event::new_no_timing().unwrap()];
                    s.record(&e[0]).unwrap();
                    s.sync().unwrap();
                }).join().unwrap();
            }
            "host" => drop(HostBuffer::<u8>::alloc(48 << 20).unwrap()),
            "vm" => {
                vm.grow(512 << 20).unwrap();
                vm.shrink(0).unwrap();
            }
            "event" => drop((Event::new_no_timing().unwrap(), Event::new_blocking().unwrap())),
            _ => panic!("test: thread | host | vm | event"),
        }
        if i % (n / 5).max(1) == 0 {
            println!("{test}: {i} iterations, anon {} MiB", anon());
        }
    }
}
