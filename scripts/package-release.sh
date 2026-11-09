#!/usr/bin/env bash
# Build a Ferrite release tarball for DGX Spark (aarch64, GB10 sm_121a): the fe-server and ferrite binaries (the GPU
# kernels are cubins embedded in them), the run scripts and the docs.
#
#   scripts/package-release.sh [--no-build] [--dev] [--out DIR]
#
# Output: DIR/ferrite-<version>-dgx-spark-aarch64.tar.gz and its .sha256 (DIR defaults to dist/), <version> = the
# workspace version (Cargo.toml), the name the docs and scripts/install.sh use. --dev adds `git describe` to the
# name (ferrite-<version>-<describe>-dgx-spark-aarch64) for builds that are not a release; BUILD-INFO records the
# commit either way. Unpacked, the release runs from its own folder (the same name without .tar.gz):
#   bin/fe-server, bin/ferrite        the binaries (target/release is a link to bin/ for the tools/ scripts)
#   scripts/                          install.sh, fetch-models.sh, serve.sh (one node), check-deps.sh, models.manifest
#   tools/                            site.sh (site settings) + fabric-detect.sh, serve-tp2.sh + tp2.sh + tp2-exec.sh
#                                     (two nodes), serve.sh, gate-serve-tp2.sh
#   README.md, AGENTS.md, LICENSE, NOTICE, CREDITS.md, NON-RUST.md, TP2.md, BUILD-INFO, and docs/: quickstart.md,
#   prompt-cache.md, benchmarks.md, model-card.md (the pages README.md and the quickstart link to)
# Build requirements are in docs/quickstart.md ("Build from source"); the runtime libraries are checked with ldd.
# The build remaps this machine's paths out of the binaries (panic locations, the cargo registry):
# --remap-path-prefix $HOME=/home/user, $CARGO_HOME=/cargo, the repository=/ferrite (added to RUSTFLAGS), and the
# packed tree is scanned for this machine's home folder, user and host names and private addresses before it is
# written: any hit stops the release.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build=1
dev=0
out=$root/dist
while [ $# -gt 0 ]; do
  case $1 in
    --no-build) build=0 ;;
    --dev) dev=1 ;;
    --out) out=$2; shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) echo "package-release: unknown argument '$1'" >&2; exit 1 ;;
  esac
  shift
done
die() { echo "package-release: $*" >&2; exit 1; }
cd "$root"

[ "$(uname -m)" = aarch64 ] || die "build on the DGX Spark itself (aarch64); this machine is $(uname -m)"
ver=$(awk -F'"' '/^\[workspace.package\]/ { w = 1 } w && /^version/ { print $2; exit }' Cargo.toml)
desc=$(git describe --tags --always --dirty 2>/dev/null || echo unknown)
name=ferrite-$ver-dgx-spark-aarch64
[ $dev = 1 ] && name=ferrite-$ver-$desc-dgx-spark-aarch64
case $desc in *-dirty) echo "package-release: warning: the work tree has uncommitted changes ($desc)" >&2 ;; esac

if [ $build = 1 ]; then
  command -v cargo > /dev/null || die "cargo not found (see docs/quickstart.md, Build from source)"
  command -v ptxas > /dev/null || [ -x /usr/local/cuda/bin/ptxas ] || die "ptxas not found: install the CUDA 13 toolkit (/usr/local/cuda/bin)"
  # the last matching --remap-path-prefix wins: the general home folder first, the more specific ones after it
  ch=${CARGO_HOME:-$HOME/.cargo}
  export RUSTFLAGS="${RUSTFLAGS:-} --remap-path-prefix=$HOME=/home/user --remap-path-prefix=$ch=/cargo --remap-path-prefix=$root=/ferrite"
  cargo build --release -p fe-server --features ferrite -p fe-cli
fi
for b in fe-server ferrite; do
  [ -x target/release/$b ] || die "target/release/$b missing (build first, or drop --no-build)"
  if command -v file > /dev/null; then
    file -b target/release/$b | grep -q 'ARM aarch64' || die "target/release/$b is not an aarch64 executable"
  fi
done
scripts/check-deps.sh --libs-only target/release/fe-server target/release/ferrite \
  || die "a linked library is missing on this machine; the release would not start here either"
if ! ldconfig -p 2>/dev/null | grep -q 'libnccl.so.2' && [ -z "${FERRITE_NCCL_LIB:-${FE_NCCL_LIB:-}}" ]; then
  echo "package-release: note: no libnccl.so.2 here; two-node (TP2) runs dlopen it (one node does not need it)" >&2
fi

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
d=$stage/$name
mkdir -p "$d/bin" "$d/scripts" "$d/tools" "$d/docs" "$d/target"
cp target/release/fe-server target/release/ferrite "$d/bin/"
ln -s ../bin "$d/target/release"
cp scripts/install.sh scripts/fetch-models.sh scripts/serve.sh scripts/check-deps.sh scripts/models.manifest "$d/scripts/"
cp tools/site.sh tools/fabric-detect.sh tools/serve.sh tools/serve-tp2.sh tools/tp2.sh tools/tp2-exec.sh \
  tools/gate-serve-tp2.sh "$d/tools/"
for f in quickstart.md prompt-cache.md benchmarks.md model-card.md; do
  if [ -f "docs/$f" ]; then cp "docs/$f" "$d/docs/"; else echo "package-release: warning: no docs/$f" >&2; fi
done
for f in README.md AGENTS.md LICENSE NOTICE CREDITS.md NON-RUST.md TP2.md; do
  if [ -f "$f" ]; then cp "$f" "$d/"; else echo "package-release: warning: no $f in the repository root" >&2; fi
done
{
  echo "ferrite $ver ($desc), built $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "commit $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "rustc $(rustc --version 2>/dev/null | cut -d' ' -f2-), nightly $(cd kernels && rustc --version 2>/dev/null | cut -d' ' -f2-)"
  px=$(command -v ptxas || echo /usr/local/cuda/bin/ptxas)
  echo "ptxas $("$px" --version 2>/dev/null | awk -F', ' '/release/ { print $2 ", " $3 }')"
  echo "kernels: sm_121a cubins embedded in the binaries"
  (cd "$d/bin" && sha256sum fe-server ferrite)
} > "$d/BUILD-INFO"
# nothing of this machine in the release: its home folder, user name in a path, host name (5+ characters, as a
# word) and private IPv4 addresses (10.x / 192.168.x, octets without leading zeros)
o='(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])'
pat="$HOME|/home/$(id -un)|(^|[^0-9.])(10\.$o|192\.168)\.$o\.$o([^0-9.]|\$)"
host=$(hostname -s)
leaks=$(cd "$stage" && find "$name" -type f | while read -r f; do
  { strings -n 4 "$f" | grep -E -o -- "$pat"
    [ ${#host} -ge 5 ] && strings -n 4 "$f" | grep -w -o -F -- "$host"; } | sort -u | sed "s|^|$f: |"; done) || true
[ -z "$leaks" ] || { echo "$leaks" | head -n 20 >&2; die "the release contains this machine's paths, names or addresses (above); with --no-build, rebuild without it"; }
mkdir -p "$out"
tar -C "$stage" -czf "$out/$name.tar.gz" "$name"
(cd "$out" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")
echo "package-release: $out/$name.tar.gz ($(du -h "$out/$name.tar.gz" | cut -f1))"
cat "$d/BUILD-INFO"
