//! graph-check: fe-ze command-list replay (Graph) on the B70, test only (needs --features ref-spv for a kernel to launch).
//! 1. correctness: N launches of qwen_basic `add_bf16` (x = bf16(x + d)) recorded once and replayed == the same launches queued one by one;
//!    a mutable graph with a changed element-count argument == direct launches with that count;
//! 2. host cost: 1,000 tiny launches per step, immediate vs replayed (the per-token launch count of a decode step).
use anyhow::{anyhow, bail, ensure, Result};
use fe_ze::{DeviceBuffer, Graph, Module, Queue, Ze};

fn main() -> Result<()> {
    let ze = Ze::new().map_err(|e| anyhow!("{e}"))?;
    let q = Queue::new(&ze).map_err(|e| anyhow!("{e}"))?;
    let dir = std::env::var("FE_XE_REF_SPV").map_err(|_| anyhow!("FE_XE_REF_SPV is not set: the folder of the reference .spv modules"))?;
    let spv = std::fs::read(format!("{dir}/qwen_basic.spv"))?;
    let m = Module::from_spirv(&ze, &spv, "").map_err(|e| anyhow!("{e}"))?;
    let k = m.kernel("add_bf16", [64, 1, 1]).map_err(|e| anyhow!("{e}"))?;
    let n = 5120usize;
    let x0: Vec<u16> = (0..n).map(|i| 0x3f80 + (i % 97) as u16).collect();
    let d: Vec<u16> = (0..n).map(|i| 0x3c00 + (i % 31) as u16).collect();
    let run_direct = |steps: usize, count: u32| -> Result<Vec<u16>> {
        let x = DeviceBuffer::from_slice(&q, &x0).map_err(|e| anyhow!("{e}"))?;
        let dd = DeviceBuffer::from_slice(&q, &d).map_err(|e| anyhow!("{e}"))?;
        for _ in 0..steps {
            k.args(&[&x.ptr(), &dd.ptr(), &count]).map_err(|e| anyhow!("{e}"))?;
            q.launch(&k, [count / 64, 1, 1]).map_err(|e| anyhow!("{e}"))?;
        }
        x.to_vec(&q).map_err(|e| anyhow!("{e}"))
    };
    // 1a. replay == direct
    let want = run_direct(10, n as u32)?;
    let x = DeviceBuffer::from_slice(&q, &x0).map_err(|e| anyhow!("{e}"))?;
    let dd = DeviceBuffer::from_slice(&q, &d).map_err(|e| anyhow!("{e}"))?;
    let mut g = Graph::new(&ze, false).map_err(|e| anyhow!("{e}"))?;
    for _ in 0..5 {
        k.args(&[&x.ptr(), &dd.ptr(), &(n as u32)]).map_err(|e| anyhow!("{e}"))?;
        g.launch(&k, [n as u32 / 64, 1, 1]).map_err(|e| anyhow!("{e}"))?;
    }
    // (two appends of the same list in flight at once overlap: only one took effect in the first test; sync between replays)
    g.replay(&q).map_err(|e| anyhow!("{e}"))?;
    q.sync().map_err(|e| anyhow!("{e}"))?;
    g.replay(&q).map_err(|e| anyhow!("{e}"))?;
    let got = x.to_vec(&q).map_err(|e| anyhow!("{e}"))?;
    if got != want {
        let applied = (0..14).find(|&k| run_direct(k, n as u32).map(|v| v == got).unwrap_or(false));
        bail!("graph replay != 10 direct launches: the result equals {applied:?} direct launches");
    }
    println!("replay: 2 x 5 recorded launches == 10 direct launches (bitwise)");
    // 1b. mutable: record with count n, change to n/2 (and the group count), replay == direct with n/2
    let want2 = run_direct(3, n as u32 / 2)?;
    let x = DeviceBuffer::from_slice(&q, &x0).map_err(|e| anyhow!("{e}"))?;
    let mut gm = Graph::new(&ze, true).map_err(|e| anyhow!("{e}"))?;
    k.args(&[&x.ptr(), &dd.ptr(), &(n as u32)]).map_err(|e| anyhow!("{e}"))?;
    let id = gm.launch(&k, [n as u32 / 64, 1, 1]).map_err(|e| anyhow!("{e}"))?.unwrap();
    gm.close().map_err(|e| anyhow!("{e}"))?;
    let half = n as u32 / 2;
    gm.set_args(&[(id, 2, &half)]).map_err(|e| anyhow!("{e}"))?;
    gm.set_groups(&[(id, [half / 64, 1, 1])]).map_err(|e| anyhow!("{e}"))?;
    for _ in 0..3 {
        gm.replay(&q).map_err(|e| anyhow!("{e}"))?;
        q.sync().map_err(|e| anyhow!("{e}"))?;
    }
    let got2 = x.to_vec(&q).map_err(|e| anyhow!("{e}"))?;
    ensure!(got2 == want2, "mutable graph (count n/2) != direct launches");
    println!("mutable: updated argument + group count, 3 replays == 3 direct launches (bitwise)");
    // 2. host cost of 1,000 launches a step
    let steps = 20;
    let t0 = fe_ze::now_ns();
    for _ in 0..steps {
        for _ in 0..1000 {
            k.args(&[&x.ptr(), &dd.ptr(), &64u32]).map_err(|e| anyhow!("{e}"))?;
            q.launch(&k, [1, 1, 1]).map_err(|e| anyhow!("{e}"))?;
        }
        q.sync().map_err(|e| anyhow!("{e}"))?;
    }
    let direct = (fe_ze::now_ns() - t0) as f64 / steps as f64 / 1e6;
    let mut gb = Graph::new(&ze, false).map_err(|e| anyhow!("{e}"))?;
    k.args(&[&x.ptr(), &dd.ptr(), &64u32]).map_err(|e| anyhow!("{e}"))?;
    for _ in 0..1000 {
        gb.launch(&k, [1, 1, 1]).map_err(|e| anyhow!("{e}"))?;
    }
    gb.close().map_err(|e| anyhow!("{e}"))?;
    let t0 = fe_ze::now_ns();
    for _ in 0..steps {
        gb.replay(&q).map_err(|e| anyhow!("{e}"))?;
        q.sync().map_err(|e| anyhow!("{e}"))?;
    }
    let replay = (fe_ze::now_ns() - t0) as f64 / steps as f64 / 1e6;
    println!("1,000 tiny launches a step: immediate {direct:.2} ms, replayed {replay:.2} ms");
    Ok(())
}
