//! TCP bootstrap between the two ranks: length-prefixed messages over one socket.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

pub struct Link {
    s: TcpStream,
}

impl Link {
    /// Rank 0 binds `addr` and accepts one peer, rank 1 connects; either side gives up after
    /// FE_TP_CONNECT_SECS (default 900) so a missing peer never holds a GPU lock for long.
    pub fn connect(rank: u32, addr: &str) -> Result<Self> {
        let window = Duration::from_secs(std::env::var("FE_TP_CONNECT_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(900));
        let t0 = Instant::now();
        let s = if rank == 0 {
            let l = TcpListener::bind(addr).with_context(|| format!("bind {addr}"))?;
            l.set_nonblocking(true)?;
            loop {
                match l.accept() {
                    Ok((s, _)) => {
                        s.set_nonblocking(false)?;
                        break s;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock && t0.elapsed() < window => {
                        std::thread::sleep(Duration::from_millis(100));
                    }
                    Err(e) => bail!("accept on {addr}: {e} (no peer within {window:?})"),
                }
            }
        } else {
            loop {
                match TcpStream::connect(addr) {
                    Ok(s) => break s,
                    Err(_) if t0.elapsed() < window => std::thread::sleep(Duration::from_millis(200)),
                    Err(e) => bail!("connect {addr}: {e}"),
                }
            }
        };
        s.set_nodelay(true)?;
        Ok(Self { s })
    }

    pub fn send(&mut self, b: &[u8]) -> Result<()> {
        self.s.write_all(&(b.len() as u64).to_le_bytes())?;
        self.s.write_all(b)?;
        Ok(())
    }

    pub fn recv(&mut self) -> Result<Vec<u8>> {
        let mut n = [0u8; 8];
        self.s.read_exact(&mut n)?;
        let mut b = vec![0u8; u64::from_le_bytes(n) as usize];
        self.s.read_exact(&mut b)?;
        Ok(b)
    }

    /// `recv`, spinning on the socket (non-blocking reads) for up to `spin` before it blocks. A blocking read pays
    /// the receiving core's wake-up from deep idle (GB10: ~0.3-0.6 ms a message after a few ms of quiet), which the
    /// follower (TP rank 1) would add to every call that follows a host sync on rank 0.
    pub fn recv_spin(&mut self, spin: Duration) -> Result<Vec<u8>> {
        let mut n = [0u8; 8];
        self.read_spin(&mut n, spin)?;
        let mut b = vec![0u8; u64::from_le_bytes(n) as usize];
        self.read_spin(&mut b, spin)?;
        Ok(b)
    }

    fn read_spin(&mut self, buf: &mut [u8], spin: Duration) -> Result<()> {
        use std::io::ErrorKind;
        let t0 = Instant::now();
        let mut got = 0;
        self.s.set_nonblocking(true)?;
        let r = loop {
            if got == buf.len() {
                break Ok(());
            }
            match self.s.read(&mut buf[got..]) {
                Ok(0) => break Err(anyhow::anyhow!("link closed by the peer")),
                Ok(k) => got += k,
                Err(e) if e.kind() == ErrorKind::WouldBlock => {
                    if t0.elapsed() >= spin {
                        self.s.set_nonblocking(false)?;
                        break self.s.read_exact(&mut buf[got..]).map_err(Into::into);
                    }
                    std::hint::spin_loop();
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => break Err(e.into()),
            }
        };
        self.s.set_nonblocking(false)?;
        r
    }

    /// The CPU that handled the socket's last incoming packet (its receive queue's interrupt), if the kernel says.
    pub fn incoming_cpu(&self) -> Option<usize> {
        use std::os::fd::AsRawFd;
        let mut v: libc::c_int = -1;
        let mut len = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
        let rc = unsafe {
            libc::getsockopt(self.s.as_raw_fd(), libc::SOL_SOCKET, libc::SO_INCOMING_CPU,
                &mut v as *mut libc::c_int as *mut libc::c_void, &mut len)
        };
        (rc == 0 && v >= 0).then_some(v as usize)
    }

    /// Send ours, receive theirs.
    pub fn exchange(&mut self, b: &[u8]) -> Result<Vec<u8>> {
        self.send(b)?;
        self.recv()
    }
}

/// The kernel's relative capacity of `cpu` (1024 = the fastest core; GB10's little cores ~720).
pub fn cpu_capacity(cpu: usize) -> Option<usize> {
    std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/cpu_capacity")).ok()?.trim().parse().ok()
}

/// Let the calling thread run on every CPU again.
pub fn unpin_thread() {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for c in 0..(8 * std::mem::size_of::<libc::cpu_set_t>()) {
            libc::CPU_SET(c, &mut set);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

/// Pin the calling thread to `cpu` (false if the kernel refuses).
pub fn pin_thread(cpu: usize) -> bool {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
    }
}

/// A thread that keeps `cpu` out of deep idle while the link is busy, so the interrupt of a socket's receive queue on
/// that CPU is taken at once: it spins there while `busy` (the owner's last message, ms on `clock`) is within
/// `window`, else sleeps 1 ms at a time; it ends with `stop`.
pub fn keep_awake(cpu: usize, stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    busy: std::sync::Arc<std::sync::atomic::AtomicU64>, clock: Instant, window: Duration) -> std::thread::JoinHandle<()> {
    use std::sync::atomic::Ordering::Relaxed;
    std::thread::spawn(move || {
        if !pin_thread(cpu) {
            return;
        }
        let w = window.as_millis() as u64;
        while !stop.load(Relaxed) {
            if (clock.elapsed().as_millis() as u64).saturating_sub(busy.load(Relaxed)) > w {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            for _ in 0..256 {
                std::hint::spin_loop();
            }
        }
    })
}
