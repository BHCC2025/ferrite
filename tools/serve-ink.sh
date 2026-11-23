#!/usr/bin/env bash
# Serve Inkling-2-Mini (docs/inkling-2-mini.md) with Ferrite on one NVIDIA card:
#
#   tools/serve-ink.sh [extra fe-server arguments]
#
# Run scripts/fetch-inkling.sh first. The layout is chosen from the card's memory:
#   24 GB (RTX PRO 4000 Blackwell and the like): the shipped weights as they are (NVFP4 experts in layers 0-19, MXFP8 in 20-23, FP8 attention and
#                     unembedding), every expert on the card, 6-bit K/V cache mapped as the context grows, bf16 draft head, 128K context.
#   16 GB (RTX 5060 Ti and the like): all experts as NVFP4 (layers 20-23 requantised at load), experts that do not fit stay in pinned host memory and
#                     are read from there as needed (more of them as the context grows), 6-bit K/V cache mapped as the context grows, FP8 draft head.
# It runs in the foreground and its process becomes the server: Ctrl-C or `kill <pid>` stops it.
#
# Environment (defaults in brackets; FERRITE_MODELS / FERRITE_GPU_LOCK come from tools/site.sh):
#   INK_DIR           folder from scripts/fetch-inkling.sh [$FERRITE_MODELS/Inkling-2-Mini]
#   FERRITE_HOST      listen address [0.0.0.0: every machine on your network can reach it, and there is no API key,
#                     so use it on a trusted network or firewall the port; 127.0.0.1 keeps it on this machine]
#   FERRITE_PORT      [8000]; checked before the model loads
#   FERRITE_MODEL_ID  model id reported by /v1/models [inkling-2-mini]
#   FERRITE_CTX       context length, prompt plus answer [131072]
#   FERRITE_BIN       fe-server built with --features ink [bin/fe-server, else target/release/fe-server]
#   FERRITE_LOCK_WAIT 1: wait for a busy card instead of failing [0]
#   INK_KV            6bit | bf16: the K/V cache of the 8 global layers [6bit on both layouts; bf16 is the opt-out, docs/inkling-2-mini.md has what it costs]
#   INK_DRAFT         on | off: the model's own draft head [on]
#   INK_LAYOUT        24gb | 16gb: force a layout [from the card's memory]
#   INK_NUMA          numactl binding for the server, "" to run unbound [--cpunodebind=0 --membind=0 when numactl is installed and the machine
#                     has more than one NUMA node; bind it to the node the card hangs off]
#   FE_DEFAULT_TEMPERATURE / FE_DEFAULT_REASONING_EFFORT / FE_DEFAULT_MAX_TOKENS
#                     request defaults, see docs/inkling-2-mini.md [1.0 / the template's 0.9 / 32768]
#   INK_*             the engine's own settings, passed through (docs/inkling-2-mini.md lists them)
# The card is the first visible one (CUDA_VISIBLE_DEVICES: an index or a UUID), else the first one nvidia-smi lists.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fl_name=serve-ink
case ${1:-} in -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;; esac
# shellcheck source=site.sh
. "$root/tools/site.sh"
# shellcheck source=launch-lib.sh
. "$root/tools/launch-lib.sh"
die() { fl_die "$@"; }

dir=${INK_DIR:-$FERRITE_MODELS/Inkling-2-Mini}
for f in "$dir/config.json" "$dir/tokenizer.json" "$dir/chat_template.jinja" "$dir/model-00001-of-00001.safetensors" "$dir/mtp.safetensors"; do
  [ -f "$f" ] || die "missing $f: run scripts/fetch-inkling.sh (or set INK_DIR)"
done
export INK_DIR=$dir
bin=${FERRITE_BIN:-}
if [ -z "$bin" ]; then
  if [ -x "$root/bin/fe-server" ]; then bin=$root/bin/fe-server; else bin=$root/target/release/fe-server; fi
fi
[ -x "$bin" ] || die "no fe-server at $bin (cargo build --release -p fe-server --features ink -p fe-ink)"

command -v nvidia-smi >/dev/null || die "nvidia-smi not found (NVIDIA driver)"
card=$(fl_nvidia_cards | head -n 1)
[ -n "$card" ] || die "no NVIDIA card visible (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}')"
IFS=, read -r idx uuid mib_total mib_free name <<<"$card"
cap=$(nvidia-smi -i "$idx" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ')
fl_check_arch "$bin" "$cap"
export CUDA_VISIBLE_DEVICES=$uuid

layout=${INK_LAYOUT:-}
if [ -z "$layout" ]; then
  if [ "$mib_total" -ge 23000 ]; then layout=24gb
  elif [ "$mib_total" -ge 15000 ]; then layout=16gb
  else die "$name has $mib_total MiB: Inkling-2-Mini needs a 16 GB card or larger"; fi
fi
case $layout in 24gb|16gb) ;; *) die "INK_LAYOUT is 24gb or 16gb, not '$layout'" ;; esac
ctx=${FERRITE_CTX:-131072}
fl_int FERRITE_CTX "$ctx" 1024 131072
draft=${INK_DRAFT:-on}
case $draft in on|off) ;; *) die "INK_DRAFT is on or off, not '$draft'" ;; esac
[ "$draft" = off ] && export INK_NODRAFT=1
if [ "$layout" = 16gb ]; then
  kv=${INK_KV:-6bit}
  # all experts NVFP4 in a shared slot pool, parked ones in host memory (INK_HOST_EXPERTS at the start; more are parked as K/V grows: INK_FIT_MB)
  export INK_G16=1 INK_SLOTS=1 INK_KV_LAZY=1 INK_CHUNK=${INK_CHUNK:-0} INK_FIT_MB=${INK_FIT_MB:-150}
  export INK_DRAFT_FP8=${INK_DRAFT_FP8:-2}
  # the first chunk of memory the draft head needs is left free at load (INK_RESERVE_MB), and 250 experts start parked instead of 130
  if [ "$draft" = off ]; then export INK_HOST_EXPERTS=${INK_HOST_EXPERTS:-130} INK_RESERVE_MB=${INK_RESERVE_MB:-0}
  else export INK_HOST_EXPERTS=${INK_HOST_EXPERTS:-250} INK_RESERVE_MB=${INK_RESERVE_MB:-500}; fi
  mem_gib=$(awk '/^MemTotal:/ { print int($2 / 1048576) }' /proc/meminfo)
  [ "$mem_gib" -ge 24 ] || fl_warn "${mem_gib} GiB of host memory: the experts parked in pinned host memory take up to 4 GiB beside the page cache"
else
  kv=${INK_KV:-6bit}
  export INK_CHUNK=${INK_CHUNK:-256}
fi
case $kv in
  6bit) export INK_KVQ6=1; [ "$layout" = 24gb ] && export INK_KV_LAZY=1 ;;
  bf16) unset INK_KVQ6 ;;
  *) die "INK_KV is bf16 or 6bit, not '$kv'" ;;
esac

numa=()
if [ -z "${INK_NUMA+x}" ]; then
  if command -v numactl >/dev/null && [ "$(numactl -H 2>/dev/null | awk '/^available:/ { print $2 }')" -gt 1 ]; then
    numa=(numactl --cpunodebind=0 --membind=0)
  fi
elif [ -n "$INK_NUMA" ]; then
  read -ra opts <<<"$INK_NUMA"; numa=(numactl "${opts[@]}")
fi

fl_say "Inkling-2-Mini from $dir on $name (${mib_total} MiB): $layout layout, $kv K/V, draft head $draft, ctx $ctx${numa:+, ${numa[*]}}"
fl_net
fl_take_locks "$uuid"
fl_banner "${FERRITE_MODEL_ID:-inkling-2-mini}" "loading takes one to two minutes"
exec "${numa[@]}" "$bin" --engine ink --model-dir "$dir" \
  --host "$FERRITE_HOST" --port "$FERRITE_PORT" --served-model-name "${FERRITE_MODEL_ID:-inkling-2-mini}" \
  --max-model-len "$ctx" "$@"
