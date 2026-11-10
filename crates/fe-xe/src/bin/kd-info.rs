//! kd-info FILE.kd...: what an ARC_KDUMP launch dump holds (kernel, shape, arguments; which buffers the launch wrote and how many bytes
//! changed). The oracle for porting one arc-engine kernel to Rust: replay the launch on the Rust kernel and compare the written buffers.
use anyhow::Result;
use fe_xe::oracle::{KArg, KernelDump};
use std::path::Path;

fn main() -> Result<()> {
    for f in std::env::args().skip(1) {
        let d = KernelDump::read(Path::new(&f))?;
        println!("{f}: {} local {:?} groups {:?}, {} args", d.name, d.local, d.groups, d.args.len());
        for (i, a) in d.args.iter().enumerate() {
            match a {
                KArg::Unset => println!("  {i:>2}: unset"),
                KArg::Scalar(b) => println!("  {i:>2}: scalar {} bytes = u32 {:?} / f32 {:?} / raw {:02x?}", b.len(), a.u32(), a.f32(), b),
                KArg::Local(n) => println!("  {i:>2}: local memory {n} bytes"),
                KArg::Buffer { ptr, offset, alloc, pre, post } => {
                    let changed = post.as_ref().map(|p| p.iter().zip(pre).filter(|(x, y)| x != y).count());
                    let p = ptr.map(|v| format!("0x{v:x} = ")).unwrap_or_default();
                    println!("  {i:>2}: buffer {p}alloc+{offset} of {alloc} bytes, {} dumped, {}", pre.len(),
                        match changed { Some(c) => format!("WRITTEN ({c} bytes changed)"), None => "read only".into() });
                }
            }
        }
    }
    Ok(())
}
