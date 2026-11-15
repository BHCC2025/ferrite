#!/usr/bin/env bash
# Check that this machine can run the Ferrite binaries: the shared libraries they link (ldd), the GPU (GB10,
# compute capability 12.1, the kernels are sm_121a cubins), the driver, free memory, and for two nodes NCCL and
# an RDMA device.
#
#   scripts/check-deps.sh [--two-node] [--libs-only] [binary ...]
#   scripts/check-deps.sh --build [--install]
#
# Binaries default to fe-server and ferrite in bin/ or target/release/ next to this script's folder. Exit status 1
# when something required is missing; warnings do not fail. --libs-only checks the architecture and the linked
# libraries only (a build machine without the GPU).
# --build checks what building from source needs instead: rustup and a stable Rust (1.99 or newer) for the host
# crates, the exact nightly the GPU kernels are built with (kernels/rust-toolchain.toml: its channel, components and
# the nvptx64-nvidia-cuda target), a C linker (cc) and ptxas from the CUDA 13 toolkit. With --install it installs
# the missing Rust toolchains through rustup.
# The kernel architecture is read from the binaries (sm_121a = GB10 / DGX Spark, sm_120a = RTX 50 cards); FE_KARCH
# sets it for --build (default: sm_121a on aarch64, sm_120a on x86_64) and when no binary is there yet.
# A clean machine needs a few system packages first (docs/quickstart.md, Clean machine); --build names the missing
# ones with the apt command.
# Environment: the site settings of tools/site.sh: FERRITE_MIN_FREE_GB (warn below this much available memory;
# scripts/serve.sh passes its own estimate), FERRITE_NCCL_LIB, FERRITE_ROCE_HCA / FERRITE_NET_IF (two nodes).
set -uo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../tools/site.sh
. "$root/tools/site.sh"
two=0
libs_only=0
build=0
install=0
bins=()
for a in "$@"; do
  case $a in
    --two-node) two=1 ;;
    --libs-only) libs_only=1 ;;
    --build) build=1 ;;
    --install) install=1 ;;
    -h|--help) sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) bins+=("$a") ;;
  esac
done
if [ ${#bins[@]} -eq 0 ]; then
  for b in fe-server ferrite; do
    if [ -x "$root/bin/$b" ]; then bins+=("$root/bin/$b")
    elif [ -x "$root/target/release/$b" ]; then bins+=("$root/target/release/$b")
    fi
  done
fi

fail=0
err() { echo "check-deps: ERROR: $*" >&2; fail=1; }
warn() { echo "check-deps: warning: $*" >&2; }
ok() { echo "check-deps: ok: $*"; }

fl_name=check-deps
. "$root/tools/launch-lib.sh"
host_karch=$([ "$(uname -m)" = aarch64 ] && echo sm_121a || echo sm_120a)
built_karch=""
[ $build = 1 ] || [ ${#bins[@]} -eq 0 ] || built_karch=$(fl_built_arch "${bins[0]}")
if [ -n "$built_karch" ] && [ -n "${FE_KARCH:-}" ] && [ "$built_karch" != "$FE_KARCH" ]; then
  warn "FE_KARCH=$FE_KARCH, but $(basename "${bins[0]}") was built for $built_karch: checking for $built_karch"
fi
karch=${built_karch:-${FE_KARCH:-$host_karch}}
case $karch in
  sm_121a) want_cap=12.1; want_gpu="GB10, DGX Spark"
    [ "$(uname -m)" = aarch64 ] || err "this build is for aarch64 (DGX Spark); this machine is $(uname -m)" ;;
  sm_120a) want_cap=12.0; want_gpu="RTX 50 series"
    [ "$(uname -m)" = x86_64 ] || err "the RTX 50 build is for x86_64; this machine is $(uname -m)" ;;
  *) err "FE_KARCH=$karch: sm_121a or sm_120a"; want_cap=; want_gpu= ;;
esac

if [ $build = 1 ]; then
  # probe without rustup installing anything on its own (rustup auto-installs a missing toolchain when one is named)
  export RUSTUP_AUTO_INSTALL=0
  tc=$root/kernels/rust-toolchain.toml
  [ -f "$tc" ] || { err "no $tc (run this from a source checkout)"; exit 1; }
  chan=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$tc")
  comps=$(sed -n 's/^components *= *\[\(.*\)\]/\1/p' "$tc" | tr -d '" ' | tr ',' ' ')
  targets=$(sed -n 's/^targets *= *\[\(.*\)\]/\1/p' "$tc" | tr -d '" ' | tr ',' ' ')
  # the system packages a clean Ubuntu lacks (one apt line for all of them)
  pk=""
  command -v curl > /dev/null || pk="$pk curl"
  [ -f /etc/ssl/certs/ca-certificates.crt ] || [ -d /etc/ca-certificates ] || pk="$pk ca-certificates"
  command -v git > /dev/null || pk="$pk git"
  command -v cc > /dev/null || pk="$pk build-essential"
  command -v python3 > /dev/null || pk="$pk python3 python3-venv"
  [ -z "$pk" ] || err "system packages missing:$pk: sudo apt install -y$pk (docs/quickstart.md, Clean machine)"
  if ! command -v rustup > /dev/null; then
    err "rustup not found: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y, then . \"\$HOME/.cargo/env\" (docs/quickstart.md, Build from source)"
    exit 1
  fi
  # the host crates: the default toolchain outside kernels/ (stable)
  if ! (cd "$root" && rustc --version > /dev/null 2>&1); then
    if [ $install = 1 ]; then rustup toolchain install stable --profile minimal && rustup default stable || err "rustup could not install stable"
    else err "no default Rust toolchain: rustup default stable (or rerun with --install)"
    fi
  fi
  if v=$(cd "$root" && rustc --version 2>/dev/null); then
    read -r maj min <<< "$(echo "$v" | sed -n 's/^rustc \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')"
    if [ "${maj:-0}" -gt 1 ] || [ "${min:-0}" -ge 99 ]; then ok "host Rust: $v"
    else err "host Rust is $v; the host crates need 1.99 or newer: rustup update stable"
    fi
  fi
  # the kernels: exactly the pinned nightly, with its components and target
  have=0
  rustup toolchain list | grep -q "^$chan-" && have=1
  missing=""
  if [ $have = 1 ]; then
    inst_c=$(rustup component list --toolchain "$chan" --installed 2>/dev/null)
    for c in $comps; do printf '%s\n' "$inst_c" | grep -q "^$c\(-\|$\)" || missing="$missing $c"; done
    inst_t=$(rustup target list --toolchain "$chan" --installed 2>/dev/null)
    for t in $targets; do printf '%s\n' "$inst_t" | grep -qx "$t" || missing="$missing $t"; done
  fi
  if [ $have = 0 ] || [ -n "$missing" ]; then
    if [ $install = 1 ]; then
      echo "check-deps: installing $chan with $comps and $targets (rustup)"
      rustup toolchain install "$chan" --profile minimal --component "${comps// /,}" --target "${targets// /,}" \
        || err "rustup could not install $chan"
    elif [ $have = 0 ]; then
      err "the kernels' toolchain $chan is not installed: rerun with --install, or rustup toolchain install $chan --profile minimal --component ${comps// /,} --target ${targets// /,}"
    else
      err "$chan lacks$missing: rerun with --install"
    fi
  fi
  if v=$(rustc "+$chan" --version 2>/dev/null); then ok "kernel Rust ($chan): $v, with $comps, $targets"; fi
  command -v cc > /dev/null && ok "C linker: $(command -v cc)"
  px=$(command -v ptxas || true)
  [ -n "$px" ] || { [ -x /usr/local/cuda/bin/ptxas ] && px=/usr/local/cuda/bin/ptxas; }
  if [ -n "$px" ]; then
    pv=$("$px" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    if [ "${pv%%.*}" -ge 13 ] 2>/dev/null; then ok "ptxas $pv ($px)"
    else err "ptxas $pv at $px; the kernels need the CUDA 13 toolkit ($karch)"
    fi
  else
    err "ptxas not found (on PATH or /usr/local/cuda/bin): NVIDIA's CUDA repository has it as cuda-nvcc-13-<x> (no full toolkit needed; docs/quickstart.md, Clean machine)"
  fi
  # the CUDA engine (--features ferrite) links cuBLASLt (the measured GEMM baseline, NON-RUST.md); Flash-Next and
  # the Arc engine do not
  if ls /usr/local/cuda/lib64/libcublasLt.so /usr/local/cuda/targets/*/lib/libcublasLt.so > /dev/null 2>&1 \
     || { ldconfig -p 2> /dev/null || true; } | grep 'libcublasLt\.so ' > /dev/null; then
    ok "libcublasLt (the 27B engine links it)"
  else
    err "libcublasLt.so not found (/usr/local/cuda/lib64): the 27B engine (--features ferrite) links it: sudo apt install -y libcublas-dev-13-<x> from NVIDIA's CUDA repository (Flash-Next alone does not need it)"
  fi
  exit $fail
fi

[ ${#bins[@]} -gt 0 ] || err "no fe-server / ferrite binary found under $root/bin or $root/target/release"
for b in "${bins[@]}"; do
  [ -x "$b" ] || { err "$b is missing or not executable"; continue; }
  missing=$(ldd "$b" 2>&1 | awk '/not found/ { print $1 }')
  if [ -n "$missing" ]; then
    for lib in $missing; do
      case $lib in
        libcuda.so*) err "$(basename "$b") needs $lib: the NVIDIA driver library (install or repair the driver)" ;;
        libibverbs.so*|libnl-*) err "$(basename "$b") needs $lib: rdma-core (sudo apt install ibverbs-providers libibverbs1)" ;;
        *) err "$(basename "$b") needs $lib (not found by the dynamic loader)" ;;
      esac
    done
  else
    ok "$(basename "$b"): all linked libraries found"
  fi
done

[ $libs_only = 1 ] && exit $fail

# the Intel Arc engine (scripts/serve-xe.sh): Level Zero and the xe driver instead of the NVIDIA checks
xe_bin=0
[ ${#bins[@]} -gt 0 ] && "${bins[0]}" --version 2>/dev/null | grep -q 'engines:.* xe' && ! "${bins[0]}" --version 2>/dev/null | grep -q 'ferrite\|fnext' && xe_bin=1
if [ $xe_bin = 1 ] || { ! command -v nvidia-smi > /dev/null && [ "${FE_ENGINE:-}" != cuda ]; }; then
  xe=""
  for d in /sys/class/drm/card[0-9]*; do
    [ "$(basename "$(readlink -f "$d/device/driver" 2> /dev/null)")" = xe ] && xe="$xe $(basename "$(readlink -f "$d/device")")"
  done
  if [ -n "$xe" ]; then ok "Intel GPU on the xe driver:$xe"; else err "no GPU on the xe kernel driver (docs/arc-b70.md, Requirements)"; fi
  if { ldconfig -p 2> /dev/null || true; } | grep 'libze_loader\.so\.1' > /dev/null || ls /usr/lib/libze_loader.so.1 /usr/lib64/libze_loader.so.1 /usr/lib/x86_64-linux-gnu/libze_loader.so.1 > /dev/null 2>&1; then
    ok "Level Zero loader (libze_loader.so.1)"
  else
    err "the Level Zero loader (libze_loader.so.1) is not installed: docs/arc-b70.md, Requirements"
  fi
  exit $fail
fi

if command -v nvidia-smi > /dev/null; then
  read -r gpu cap drv < <(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null \
    | head -1 | awk -F', ' '{ gsub(/ /, "_", $1); print $1, $2, $3 }')
  if [ -z "${cap:-}" ]; then
    err "nvidia-smi found no GPU"
  elif [ "$cap" != "$want_cap" ]; then
    case $cap in 12.1) fix=sm_121a ;; 12.0) fix=sm_120a ;; *) fix="" ;; esac
    err "GPU ${gpu//_/ } has compute capability $cap; the kernels are built for $karch ($want_gpu)${fix:+: rebuild with FE_KARCH=$fix (docs/quickstart.md, Build from source)}"
  else
    ok "GPU ${gpu//_/ } (${karch%a}), driver $drv"
    major=${drv%%.*}
    [ "${major:-0}" -ge 580 ] || warn "driver $drv; Ferrite was tested with 580.x (CUDA 13.0)"
  fi
else
  err "nvidia-smi not found (is the NVIDIA driver installed?)"
fi

avail=$(awk '/MemAvailable/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
# inside a container or a memory-limited cgroup the limit counts, not the host's memory
cg=$(awk -F: '$1 == "0" { print $3 }' /proc/self/cgroup 2>/dev/null || true)
if [ -n "$cg" ] && [ -r "/sys/fs/cgroup$cg/memory.max" ] && [ "$(cat "/sys/fs/cgroup$cg/memory.max")" != max ]; then
  lim=$(( $(cat "/sys/fs/cgroup$cg/memory.max") - $(cat "/sys/fs/cgroup$cg/memory.current" 2>/dev/null || echo 0) ))
  [ $((lim >> 30)) -lt "$avail" ] && avail=$((lim >> 30)) && echo "check-deps: note: a memory limit applies here (cgroup $cg): $avail GB of it is free"
fi
need=$FERRITE_MIN_FREE_GB
if [ "$avail" -lt "$need" ]; then
  warn "only $avail GB of memory available (the GPU shares it), $need GB wanted: lower FERRITE_CTX or FERRITE_SLOTS, or stop other GPU jobs"
else
  ok "$avail GB memory available"
fi

if [ $two = 1 ]; then
  nccl=""
  for c in "${FERRITE_NCCL_LIB:-${FE_NCCL_LIB:-}}" libnccl.so.2 $HOME/.venvs/nccl/lib/python3*/site-packages/nvidia/nccl/lib/libnccl.so.2; do
    [ -n "$c" ] || continue
    if [ -f "$c" ] || ldconfig -p 2>/dev/null | grep -q " $c "; then nccl=$c; break; fi
  done
  if [ -n "$nccl" ]; then
    ok "NCCL: $nccl"
    case $nccl in /*) [ -n "${FERRITE_NCCL_LIB:-${FE_NCCL_LIB:-}}" ] || warn "found NCCL at $nccl: set FERRITE_NCCL_LIB=$nccl for the two-node scripts" ;; esac
  else err "two nodes need NCCL 2.30+ (libnccl.so.2). Without sudo: python3 -m venv ~/.venvs/nccl && ~/.venvs/nccl/bin/pip install nvidia-nccl-cu13, then FERRITE_NCCL_LIB=\$(ls ~/.venvs/nccl/lib/python3*/site-packages/nvidia/nccl/lib/libnccl.so.2) (on both nodes); or sudo apt install libnccl2 (TP2.md, NCCL)"
  fi
  if command -v ibv_devices > /dev/null; then
    devs=$(ibv_devices 2>/dev/null | awk 'NR > 2 && NF { print $1 }')
    for h in ${FERRITE_ROCE_HCA//,/ }; do
      if printf '%s\n' "$devs" | grep -qx "$h"; then ok "RDMA device $h"
      else err "RDMA device $h (FERRITE_ROCE_HCA) not found; present: $(echo $devs) (FERRITE_FABRIC=auto detects them)"
      fi
    done
  else
    warn "ibv_devices not found (apt install ibverbs-utils) to check the RoCE devices"
  fi
  if ip link show "$FERRITE_NET_IF" > /dev/null 2>&1; then ok "fabric interface $FERRITE_NET_IF"
  else err "network interface $FERRITE_NET_IF (FERRITE_NET_IF) not found"
  fi
fi

exit $fail
