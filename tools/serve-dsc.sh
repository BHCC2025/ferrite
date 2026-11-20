#!/usr/bin/env bash
# Serve DeepSeek-V4.1-Flash Coder (docs/deepseek-coder.md) with Ferrite on three or four NVIDIA cards:
#
#   tools/serve-dsc.sh [extra fe-server arguments]
#
# Run scripts/fetch-deepseek.sh first. Card roles, in the order of the cards (below): the first is
# the main card (attention, the dense layers and its share of the experts: put it in your fastest x16 slot), the others
# hold experts only, the drafter lives on the third. Three 24 GB cards hold all but about 1,400 of the 7,680 experts
# (those stay in pinned host memory and are copied to a card when used); a fourth card of 16 GB or more holds the rest.
# The experts on the cards are the most used ones, from the placement heat this tree ships (crates/fe-dsc/heat/coder.cnt,
# measured on code, agent sessions and chat). It runs in the foreground and its process becomes the server: Ctrl-C or
# `kill <pid>` stops it.
#
# Environment (defaults in brackets; FERRITE_MODELS / FERRITE_GPU_LOCK come from tools/site.sh):
#   DSC_DIR           folder from scripts/fetch-deepseek.sh [$FERRITE_MODELS/DeepSeek-V4.1-Flash-Coder]
#   FERRITE_HOST      listen address [0.0.0.0: every machine on your network can reach it, and there is no API key,
#                     so use it on a trusted network or firewall the port; 127.0.0.1 keeps it on this machine]
#   FERRITE_PORT      [8000]; checked before the model loads
#   FERRITE_MODEL_ID  model id reported by /v1/models [deepseek-v4.1-flash-coder]
#   FERRITE_CTX       context length [131072]
#   FERRITE_SLOTS     requests at once, 1-4 [1]. One request alone is fastest; 4 gives more in total (code on four
#                     cards: about 110-130 tok/s alone, about 140 in total over four), see docs/deepseek-coder.md
#   FERRITE_BIN       fe-server built with --features dsc [bin/fe-server, else target/release/fe-server]
#   FERRITE_LOCK_WAIT 1: wait for busy cards instead of failing [0]
#   DSC_CARDS         3 or 4: how many of the visible cards to use [4 when four are visible, else 3]
#   DSC_NUMA          numactl binding for the server, "" to run unbound [--cpunodebind=0 --membind=0 when numactl is
#                     installed and the machine has more than one NUMA node; bind it to the node the main card hangs off]
#   DSC_*             engine settings, passed through (docs/deepseek-coder.md lists them)
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fl_name=serve-dsc
case ${1:-} in -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;; esac
# shellcheck source=site.sh
. "$root/tools/site.sh"
# shellcheck source=launch-lib.sh
. "$root/tools/launch-lib.sh"
die() { fl_die "$@"; }

dir=${DSC_DIR:-$FERRITE_MODELS/DeepSeek-V4.1-Flash-Coder}
export DSC_MODEL=${DSC_MODEL:-$dir/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-00001-of-00006.gguf}
export DSC_DRAFT_MODEL=${DSC_DRAFT_MODEL:-$dir/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf}
export DSC_ENGRAM=${DSC_ENGRAM:-$dir/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf}
for f in "$DSC_MODEL" "$DSC_DRAFT_MODEL" "$DSC_ENGRAM" "$dir/tokenizer.json" "$dir/chat_template.jinja"; do
  [ -f "$f" ] || die "missing $f: run scripts/fetch-deepseek.sh (or set DSC_DIR)"
done
export DSC_HEAT=${DSC_HEAT:-$root/crates/fe-dsc/heat/coder.cnt}
[ -f "$DSC_HEAT" ] || die "missing $DSC_HEAT (the placement heat that picks the experts kept on the cards)"
bin=${FERRITE_BIN:-}
if [ -z "$bin" ]; then
  if [ -x "$root/bin/fe-server" ]; then bin=$root/bin/fe-server; else bin=$root/target/release/fe-server; fi
fi
[ -x "$bin" ] || die "no fe-server at $bin (cargo build --release -p fe-server --features dsc)"

# cards: the visible ones (CUDA_VISIBLE_DEVICES: nvidia-smi indices or UUIDs, in the order given), else every card with
# the largest first (bus order among equals); the server gets them by UUID in that order
command -v nvidia-smi >/dev/null || die "nvidia-smi not found (NVIDIA driver)"
mapfile -t all < <(nvidia-smi --query-gpu=uuid,memory.total,name --format=csv,noheader,nounits)
cards=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=, read -ra vis <<<"$CUDA_VISIBLE_DEVICES"
  for v in "${vis[@]}"; do
    for i in "${!all[@]}"; do
      u=${all[$i]%%,*}
      if [ "$v" = "$i" ] || [ "$v" = "$u" ] || [[ $u == "$v"* && $v == GPU-* ]]; then cards+=("${all[$i]}"); fi
    done
  done
else
  mapfile -t cards < <(printf '%s\n' "${all[@]}" | sort -t, -k2,2nr -s)
fi
n=${DSC_CARDS:-$([ ${#cards[@]} -ge 4 ] && echo 4 || echo 3)}
case $n in 3|4) ;; *) die "DSC_CARDS is 3 or 4, not '$n'" ;; esac
[ ${#cards[@]} -ge "$n" ] || die "${#cards[@]} visible card(s): DeepSeek Coder needs three 24 GB cards (a fourth of 16 GB or more is optional)"
cards=("${cards[@]:0:$n}")
for k in 0 1 2; do
  mib=$(echo "${cards[$k]}" | cut -d, -f2 | tr -d ' ')
  [ "$mib" -ge 23000 ] || die "card $((k + 1)) ($(echo "${cards[$k]}" | cut -d, -f3)) has ${mib} MiB: the first three cards must have 24 GB"
done
if [ "$n" = 4 ]; then
  mib=$(echo "${cards[3]}" | cut -d, -f2 | tr -d ' ')
  [ "$mib" -ge 15000 ] || die "card 4 has ${mib} MiB: the fourth card needs 16 GB (or DSC_CARDS=3)"
fi
ids=$(for c in "${cards[@]}"; do echo "${c%%,*}"; done | paste -sd,)
export CUDA_VISIBLE_DEVICES=$ids
# engine layout: ordinal 0 = main card, 1.. = expert cards, the drafter on ordinal 2; card memory left free per card
if [ "$n" = 4 ]; then
  : "${DSC_XCARDS:=1,2,3}" "${DSC_RESERVE_GBS:=0.5,0.15,-0.3,0.4}" "${DSC_PLACE_SPEED:=1,1,1,0.6}" "${DSC_ZC_W:=1,1,0.5,0.5}"
else
  : "${DSC_XCARDS:=1,2}" "${DSC_RESERVE_GBS:=0.5,0.15,-0.3}" "${DSC_ZC_W:=1,1,0.5}"
fi
export DSC_XCARDS DSC_RESERVE_GBS DSC_ZC_W ${DSC_PLACE_SPEED:+DSC_PLACE_SPEED}
export DSC_DRAFT_CARD=${DSC_DRAFT_CARD:-2} DSC_CTX=${FERRITE_CTX:-131072} DSC_RING=${DSC_RING:-4096}
export DSC_PREFILL_ROWS=${DSC_PREFILL_ROWS:-32} DSC_PF_ROWS=${DSC_PF_ROWS:-256} DSC_PF_CHUNK=${DSC_PF_CHUNK:-2048}
export DSC_PF_SLOTS=${DSC_PF_SLOTS:-14} DSC_PF_GROUP=${DSC_PF_GROUP:-4} DSC_PF_IDXTC=${DSC_PF_IDXTC:-1}
export DSC_DEPTH_BUCKET=${DSC_DEPTH_BUCKET:-1} DSC_PF_MAIN_KEEP=${DSC_PF_MAIN_KEEP:-0.1} DSC_PF_MAIN_PEN=${DSC_PF_MAIN_PEN:-40}
slots=${FERRITE_SLOTS:-1}
fl_int FERRITE_SLOTS "$slots" 1 4
export DSC_LANES=$slots

# host memory: the experts that stay off the cards are pinned (about 12 GiB with three cards) beside the page cache
mem_gib=$(awk '/^MemTotal:/ { print int($2 / 1048576) }' /proc/meminfo)
[ "$mem_gib" -ge 60 ] || fl_warn "${mem_gib} GiB of memory: tested with 128 GB; below 64 GB the start may fail"

numa=()
if [ -z "${DSC_NUMA+x}" ]; then
  if command -v numactl >/dev/null && [ "$(numactl -H 2>/dev/null | awk '/^available:/ { print $2 }')" -gt 1 ]; then
    numa=(numactl --cpunodebind=0 --membind=0)
  fi
elif [ -n "$DSC_NUMA" ]; then
  read -ra opts <<<"$DSC_NUMA"; numa=(numactl "${opts[@]}")
fi

fl_say "$DSC_MODEL + $(basename "$DSC_DRAFT_MODEL")"
fl_say "$n cards ($CUDA_VISIBLE_DEVICES; main card first), ${mem_gib} GiB host memory, ctx $DSC_CTX, $slots request(s) at once${numa:+, ${numa[*]}}"
fl_net
fl_take_locks ${ids//,/ }
fl_sampling "$dir" deepseek-v4.1-flash-coder
fl_banner "${FERRITE_MODEL_ID:-deepseek-v4.1-flash-coder}" "loading takes two to four minutes"
exec "${numa[@]}" "$bin" --engine dsc --model-dir "$dir" \
  --host "$FERRITE_HOST" --port "$FERRITE_PORT" --served-model-name "${FERRITE_MODEL_ID:-deepseek-v4.1-flash-coder}" \
  --max-model-len "$DSC_CTX" "$@"
