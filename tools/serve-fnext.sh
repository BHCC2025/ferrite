#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next (docs/flash-next.md) with Ferrite on one or two 16 GB cards:
#
#   tools/serve-fnext.sh [extra fe-server arguments]
#
# Run scripts/fetch-flashnext.sh first. The preset follows the visible cards (CUDA_VISIBLE_DEVICES, else every card
# nvidia-smi lists): two or more -> tensor parallel over the first two (FNEXT_TP=2, half the heads and half the
# experts on each); one -> the single-card engine (experts that do not fit on the card run on the CPU). The host
# memory picks the pinned pools inside the engine (below 40 GiB: one prompt-cache snapshot buffer, no prompt
# stages); this script picks the request slots. It runs in the foreground and its process becomes the server: Ctrl-C
# or `kill <pid>` stops it. scripts/ferritectl.sh runs it in the background.
#
# Environment (defaults in brackets; FERRITE_MODELS / FERRITE_GPU_LOCK come from tools/site.sh; the server holds
# FERRITE_GPU_LOCK shared and a lock of its own per card, so jobs on other cards are not blocked):
#   FNEXT_DIR         folder from scripts/fetch-flashnext.sh [$FERRITE_MODELS/Qwen3.8-Flash-Next]
#   FERRITE_GPUS      auto, 1 or 2 [auto]
#   FERRITE_HOST      listen address [0.0.0.0: every machine on your network can reach it, and there is no API key,
#                     so use it on a trusted network or firewall the port; 127.0.0.1 keeps it on this machine]
#   FERRITE_PORT      [8000]; checked before the model loads
#   FERRITE_GPU_LOCK  the node's GPU lock, held shared, plus each used card's own lock (ferrite-gpu.<card UUID>.lock)
#                     exclusively: a 27B server on the other card does not block this one; a second launcher on the
#                     same card stops at once and names the holder (FERRITE_LOCK_WAIT=1 waits) [/var/tmp/ferrite-gpu.lock]
#   FERRITE_MODEL_ID  model id reported by /v1/models [qwen3.8-flash-next]
#   FERRITE_CTX       context length [131072]
#   FERRITE_SLOTS     requests at once (1-8) [4 with two cards, or one card and >= 40 GiB of memory; 1 below]
#   FERRITE_DRAFT     mtp (speculative decoding with the MTP draft file) or off [mtp when the draft file is there]
#   FERRITE_BIN       fe-server built with --features fnext [bin/fe-server, else target/release/fe-server]
#   FERRITE_GPU_WAIT  1: wait for a busy card instead of failing [0]
#   FNEXT_*           passed through to the engine (e.g. FNEXT_COUNTS, FNEXT_CPU_POOL, FNEXT_RESERVE_MIB)
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fl_name=serve-fnext
case ${1:-} in -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;; esac
# shellcheck source=site.sh
. "$root/tools/site.sh"
# shellcheck source=launch-lib.sh
. "$root/tools/launch-lib.sh"
die() { fl_die "$@"; }

dir=${FNEXT_DIR:-$FERRITE_MODELS/Qwen3.8-Flash-Next}
model=$(ls "$dir"/*-00001-of-0000[0-9].gguf 2>/dev/null | head -1 || true)
[ -n "$model" ] || die "no Flash-Next GGUF in $dir: run scripts/fetch-flashnext.sh (or set FNEXT_DIR)"
[ -f "$dir/tokenizer.json" ] && [ -f "$dir/chat_template.jinja" ] || die "$dir lacks tokenizer.json / chat_template.jinja: rerun scripts/fetch-flashnext.sh"
bin=${FERRITE_BIN:-}
if [ -z "$bin" ]; then
  if [ -x "$root/bin/fe-server" ]; then bin=$root/bin/fe-server; else bin=$root/target/release/fe-server; fi
fi
[ -x "$bin" ] || die "no fe-server at $bin (cargo build --release -p fe-server --features fnext)"

# cards: the visible ones, at least 16 GB each
command -v nvidia-smi >/dev/null || die "nvidia-smi not found (NVIDIA driver)"
mapfile -t cards < <(nvidia-smi --query-gpu=uuid,memory.total,name --format=csv,noheader,nounits)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=, read -ra vis <<<"$CUDA_VISIBLE_DEVICES"
  sel=()
  for v in "${vis[@]}"; do
    for i in "${!cards[@]}"; do
      u=${cards[$i]%%,*}
      if [ "$v" = "$i" ] || [ "$v" = "$u" ] || [[ $u == "$v"* && $v == GPU-* ]]; then sel+=("${cards[$i]}"); fi
    done
  done
  cards=("${sel[@]}")
fi
[ ${#cards[@]} -ge 1 ] || die "no visible NVIDIA card"
for c in "${cards[@]}"; do
  mib=$(echo "$c" | cut -d, -f2 | tr -d ' ')
  [ "$mib" -ge 15000 ] || die "card $(echo "$c" | cut -d, -f3) has ${mib} MiB: Flash-Next needs 16 GB cards"
done
gpus=${FERRITE_GPUS:-auto}
[ "$gpus" = auto ] && gpus=$([ ${#cards[@]} -ge 2 ] && echo 2 || echo 1)
case $gpus in
  1) ids=$(echo "${cards[0]}" | cut -d, -f1); unset FNEXT_TP ;;
  2) [ ${#cards[@]} -ge 2 ] || die "FERRITE_GPUS=2 but only one card is visible"
     ids=$(echo "${cards[0]}" | cut -d, -f1),$(echo "${cards[1]}" | cut -d, -f1); export FNEXT_TP=2 ;;
  *) die "FERRITE_GPUS is auto, 1 or 2, not '$gpus'" ;;
esac
export CUDA_VISIBLE_DEVICES=$ids

# host memory (the cgroup limit when lower): slots and a check against the 32 GB minimum
mem_gib=$(awk '/^MemTotal:/ { print int($2 / 1048576) }' /proc/meminfo)
cg=$(awk -F: '$1 == "0" { print $3 }' /proc/self/cgroup 2>/dev/null || true)
if [ -n "$cg" ] && [ -r "/sys/fs/cgroup$cg/memory.max" ]; then
  lim=$(cat "/sys/fs/cgroup$cg/memory.max")
  [ "$lim" != max ] && [ $((lim >> 30)) -lt "$mem_gib" ] && mem_gib=$((lim >> 30))
fi
[ "$mem_gib" -ge 27 ] || die "${mem_gib} GiB of memory: Flash-Next needs a 32 GB machine (the 27.5 GiB PLE table is read from the page cache)"
slots=${FERRITE_SLOTS:-}
[ -n "$slots" ] || { [ "$gpus" = 2 ] || [ "$mem_gib" -ge 40 ] && slots=4 || slots=1; }
export FNEXT_SLOTS=$slots FNEXT_CTX=${FERRITE_CTX:-131072}
export FNEXT_MODEL=$model
export FNEXT_COUNTS=${FNEXT_COUNTS:-$root/crates/fe-fnext/data/routing-mix.json}
[ -f "$FNEXT_COUNTS" ] || die "missing $FNEXT_COUNTS (the expert routing counts that pick the experts kept on the card)"

draft=$(ls "$dir"/*[Mm][Tt][Pp]*.gguf 2>/dev/null | head -1 || true)
dr=${FERRITE_DRAFT:-$([ -n "$draft" ] && echo mtp || echo off)}
case $dr in
  mtp) [ -n "$draft" ] || die "FERRITE_DRAFT=mtp needs the draft file in $dir (scripts/fetch-flashnext.sh)"
       export FNEXT_MTP=1 FNEXT_DRAFT=$draft ;;
  off) export FNEXT_MTP=0 FNEXT_DRAFT=none ;;
  *) die "FERRITE_DRAFT is mtp or off, not '$dr'" ;;
esac

echo "serve-fnext: $model$([ "$dr" = mtp ] && echo " + $(basename "$draft")")"
echo "serve-fnext: $([ "$gpus" = 2 ] && echo "two cards, tensor parallel" || echo "one card") ($ids), ${mem_gib} GiB host memory, \
ctx $FNEXT_CTX x $slots slots, drafting $dr"
fl_net
fl_take_locks ${ids//,/ }
fl_sampling "$dir" qwen3.8-flash-next
fl_banner "${FERRITE_MODEL_ID:-qwen3.8-flash-next}" "loading takes a minute or two"
exec "$bin" --engine fnext --model-dir "$dir" \
  --host "$FERRITE_HOST" --port "$FERRITE_PORT" --served-model-name "${FERRITE_MODEL_ID:-qwen3.8-flash-next}" \
  --max-model-len "$FNEXT_CTX" "$@"
