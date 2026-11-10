//! `kbuild-xe <kx-crate> [--features a,b --name ident] [out_dir]`: build one kernels-xe crate to SPIR-V by hand (prints
//! the .spv path; intermediates next to it).
fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(|a| a == "--retarget").unwrap_or(false) {
        // kbuild-xe --retarget in.ll out.ll: only the IR pass (instnamer'd amdgcn IR -> spirv64 IR)
        let ir = std::fs::read_to_string(&args[1]).expect("read input");
        match fe_kbuild_xe::retarget::retarget(&ir) {
            Ok(s) => std::fs::write(&args[2], s).expect("write output"),
            Err(e) => {
                eprintln!("retarget: {e}");
                std::process::exit(1);
            }
        }
        return;
    }
    let mut features = Vec::new();
    let mut name = None;
    while let Some(i) = args.iter().position(|a| a == "--features" || a == "--name") {
        let v = args.remove(i + 1);
        if args.remove(i) == "--features" {
            features.extend(v.split(',').map(String::from));
        } else {
            name = Some(v);
        }
    }
    if args.is_empty() {
        eprintln!("usage: kbuild-xe <kx-crate> [--features a,b --name ident] [out_dir]   (run inside the ferrite checkout)");
        std::process::exit(2);
    }
    let root = fe_kbuild_xe::workspace_root();
    let out = args.get(1).map(std::path::PathBuf::from).unwrap_or_else(|| root.join("kernels-xe/target/spv"));
    let name = name.unwrap_or_else(|| args[0].replace('-', "_"));
    let f: Vec<&str> = features.iter().map(|s| s.as_str()).collect();
    match fe_kbuild_xe::build_variant_to(&root, &args[0], &f, &name, &out) {
        Ok(p) => println!("{}", p.display()),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
