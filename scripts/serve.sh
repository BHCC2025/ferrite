#!/usr/bin/env bash
# Start Ferrite's OpenAI-compatible server for Qwen3.8-27B on one NVIDIA GPU (one command once the models are
# downloaded):
#
#   FERRITE_MODELS=/path/to/models scripts/serve.sh [extra fe-server arguments]
#
# On a DGX Spark it serves 4 x 128K. On an RTX 5080 / 5060 Ti (16 GB) it hands over to tools/serve-5080.sh, the same
# script with the small-memory preset (docs/rtx-5080.md). Serves /v1/chat/completions, /v1/completions, /v1/models
# and /health (streaming, tool calls, thinking via chat_template_kwargs.enable_thinking) at
# http://<this machine>:$FERRITE_PORT/v1 under the model id $FERRITE_MODEL_ID. It runs in the foreground and its
# process becomes the server: Ctrl-C or `kill <pid>` stops it. scripts/ferritectl.sh runs it in the background.
#
# Environment (defaults in brackets). The site settings (model folders, GPU lock, ...) come from tools/site.sh,
# whose FERRITE_* table is in the README; this script adds the server settings below.
#   FERRITE_MODELS    folder with the model folders from scripts/fetch-models.sh [/var/tmp/models]; or set
#                     FERRITE_EXL3_DIR / FERRITE_NVFP4_DIR / FERRITE_DRAFT_DIR one by one
#   FERRITE_HOST      listen address [0.0.0.0: every machine on your network can reach it, and there is no API key,
#                     so use it on a trusted network or firewall the port; 127.0.0.1 keeps it on this machine]
#   FERRITE_PORT      [8000]; checked before the model loads
#   FERRITE_MODEL_ID  model id reported by /v1/models; requests may name any model [qwen3.8-27b]
#   FERRITE_GPU_LOCK  the node's GPU lock, held shared while the server runs, with its card's own lock
#                     (ferrite-gpu.<card UUID>.lock next to it) held exclusively; a second launcher on the card stops at
#                     once and names the holder (FERRITE_LOCK_WAIT=1 waits instead) [/var/tmp/ferrite-gpu.lock]
#   FERRITE_CTX       context length per request [Spark: 131072 with fp4 prompt weights, 32768 with bf16; lower it
#                     to save memory: the KV cache is ~64 KB a token a slot]
#   FERRITE_SLOTS     requests at once [4; at most 4 with dflash drafting: more requests wait in a queue]
#   FERRITE_PROMPT    prompt (prefill) weights: fp4 (the NVIDIA NVFP4 checkpoint) or bf16 (reconstructed from the
#                     EXL3 quant at load: no NVIDIA download, about half the prefill speed and ~30 GB more memory)
#                     [fp4 when FERRITE_NVFP4_DIR is present, else bf16]
#   FERRITE_DRAFT     speculative decoding: dflash (DFlash2 drafter with verify trees), mtp (the model's own MTP
#                     head), auto (DFlash2 for up to 2 streams, MTP above) or off [dflash when the drafter is
#                     present, else mtp]
#   FERRITE_CACHE     prompt cache: 1 keeps each conversation's state at its turn boundaries (and the shared system
#                     prompt), so a later turn, or another conversation with the same system prompt, skips what was
#                     already computed; 0 turns it off. It works with every drafting mode [1]
#   FERRITE_PARK_GB   (or FE_PARK_GB) memory for the cache's parked conversations and shared prefixes, on top of the
#                     model and KV cache; the server shrinks it if needed to keep FE_PARK_RESERVE_GB (20) of memory
#                     available [12 on one Spark; tools/serve-tp2.sh takes 24 a node]
#   FERRITE_FORCE     1 starts even when the memory check says it does not fit, or the model folder check fails
#   FERRITE_BIN       fe-server binary [bin/fe-server, else target/release/fe-server]
#   Advanced (the defaults are the measured best): FERRITE_CHUNK prompt rows a forward [2048], FERRITE_SPEC draft
#   depth [16 dflash, 8 mtp], FERRITE_COPY copy rows [47], FERRITE_TREE verify trees [1], FERRITE_BLOCK DFlash2
#   block [16].
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fl_name=serve
case ${1:-} in -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;; esac

user_min_free=${FERRITE_MIN_FREE_GB:-}
# shellcheck source=../tools/site.sh
. "$root/tools/site.sh"
# shellcheck source=../tools/launch-lib.sh
. "$root/tools/launch-lib.sh"
quant=$FERRITE_EXL3_DIR
nvfp4=$FERRITE_NVFP4_DIR
draft=$FERRITE_DRAFT_DIR
bin=${FERRITE_BIN:-}
if [ -z "$bin" ]; then
  if [ -x "$root/bin/fe-server" ]; then bin=$root/bin/fe-server; else bin=$root/target/release/fe-server; fi
fi
[ -x "$bin" ] || fl_die "no fe-server binary at $bin (unpack a release, or build: docs/quickstart.md, Build from source)"

# the GPU: a GB10 (DGX Spark) or an RTX 50 card (the small-memory preset)
command -v nvidia-smi > /dev/null || fl_die "nvidia-smi not found: the NVIDIA driver is missing"
mapfile -t cards < <(fl_nvidia_cards)
[ ${#cards[@]} -ge 1 ] && [ -n "${cards[0]}" ] || fl_die "no visible NVIDIA GPU (nvidia-smi; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset})"
IFS=, read -r gidx guuid gtotal gfree gname <<< "${cards[0]}"
cap=$(nvidia-smi -i "$gidx" --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' ')
profile=${FERRITE_PROFILE:-}
if [ -z "$profile" ]; then
  case $cap in
    12.1) profile=spark ;;
    12.0) fl_say "$gname (compute capability 12.0): starting the RTX preset, tools/serve-5080.sh"
          exec "$root/tools/serve-5080.sh" "$@" ;;
    *) fl_die "the GPU is $gname (compute capability $cap); Ferrite's CUDA kernels run on GB10 (12.1) and RTX 50 cards (12.0)" ;;
  esac
fi
fl_check_arch "$bin" "$cap"
FE_KARCH=$([ "$cap" = 12.0 ] && echo sm_120a || echo sm_121a) FERRITE_MIN_FREE_GB=0 \
  "$root/scripts/check-deps.sh" "$bin" > /dev/null || fl_die "fix the errors above first (scripts/check-deps.sh)"

[ -f "$quant/config.json" ] || fl_die "no model at $quant: run scripts/fetch-models.sh (FERRITE_MODELS=$FERRITE_MODELS)"
prompt=${FERRITE_PROMPT:-}
[ -n "$prompt" ] || { [ -f "$nvfp4/config.json" ] && prompt=fp4 || prompt=bf16; }
dr=${FERRITE_DRAFT:-}
if [ -z "$dr" ]; then
  if [ "$profile" = rtx ]; then dr=mtp; else [ -f "$draft/config.json" ] && dr=dflash || dr=mtp; fi
fi

case $prompt in
  fp4) [ -f "$nvfp4/config.json" ] || fl_die "FERRITE_PROMPT=fp4 needs $nvfp4 (scripts/fetch-models.sh prompt)"
       export FE_PROMPT=fp4 ;;
  bf16) unset FE_PROMPT ;;
  *) fl_die "FERRITE_PROMPT is fp4 or bf16, not '$prompt'" ;;
esac
slots=${FERRITE_SLOTS:-4}
fl_int FERRITE_SLOTS "$slots" 1 8
case $dr in
  dflash|auto)
    [ -f "$draft/config.json" ] || fl_die "FERRITE_DRAFT=$dr needs the DFlash2 drafter in $draft (scripts/fetch-models.sh draft; FERRITE_DRAFT_DIR)"
    export FE_DRAFT=dflash FE_SPEC=${FERRITE_SPEC:-16} FE_TREE=${FERRITE_TREE:-1} FE_DF_BLOCK=${FERRITE_BLOCK:-16} \
      FE_COPY_ROWS=${FERRITE_COPY:-47}
    if [ "$dr" = auto ]; then export FE_MTP=1; else export FE_MTP=0; fi ;;
  mtp) unset FE_DRAFT; export FE_MTP=1 FE_SPEC=${FERRITE_SPEC:-8}
       if [ -n "${FERRITE_COPY:-}" ]; then export FE_COPY_ROWS=$FERRITE_COPY; else unset FE_COPY_ROWS; fi ;;
  off) unset FE_DRAFT; export FE_MTP=0 FE_SPEC=0 FE_COPY_ROWS=0 ;;
  *) fl_die "FERRITE_DRAFT is dflash, mtp, auto or off, not '$dr'" ;;
esac
if [ "$slots" -gt 4 ] && [ "$dr" = dflash ]; then
  fl_die "FERRITE_SLOTS=$slots is not supported with DFlash2 drafting (FERRITE_DRAFT=dflash): one draft pass serves at most 4 streams in this build. Use FERRITE_SLOTS=4, or FERRITE_DRAFT=auto / mtp for more slots (MTP drafting above 2 streams; not tested above 4 slots)"
fi
cache=${FERRITE_CACHE:-${CACHE:-1}}
case $cache in
  0|1) export FE_CACHE=$cache ;;
  *) fl_die "FERRITE_CACHE is 1 or 0, not '$cache'" ;;
esac
# one Spark: a 12 GB pool (24 bottoms out at ~20 GB available under 4 agents with thinking; 12 keeps ~34 GB and
# still parks ~92K tokens); the server also shrinks it if needed to keep 20 GB available
park=${FERRITE_PARK_GB:-${FE_PARK_GB:-12}}
fl_int FERRITE_PARK_GB "${park%.*}" 0 512
if [ "$cache" = 1 ]; then export FE_PARK_GB=$park; else park=0; fi
chunk=${FERRITE_CHUNK:-2048}
fl_int FERRITE_CHUNK "$chunk" 64 16384

# the model files: sizes against scripts/models.manifest, so a missing or cut-off file is named before the load
roles="quant"; [ "$prompt" = fp4 ] && roles="$roles prompt"; case $dr in dflash|auto) roles="$roles draft" ;; esac
if ! vout=$("$root/scripts/fetch-models.sh" --verify-only --quick --essential $roles 2>&1); then
  printf '%s\n' "$vout" | grep -E '^  (missing|size)' | head -n 8 >&2
  if [ "${FERRITE_FORCE:-0}" = 1 ]; then fl_warn "the model folders do not match scripts/models.manifest (above); FERRITE_FORCE=1: starting anyway"
  else fl_die "the model folders do not match scripts/models.manifest (above): run scripts/fetch-models.sh $roles to fetch the missing parts (FERRITE_FORCE=1 starts anyway)"
  fi
fi

# the port and the GPU lock before the memory checks: a second launcher names the server that is already running
fl_net
fl_take_locks "$guuid"
avail=$(awk '/MemAvailable/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
if [ "$profile" = spark ]; then
  # bf16 prompt weights take ~30 GB more than the NVFP4 ones: keep the default KV cache inside one Spark's memory
  ctx=${FERRITE_CTX:-}
  [ -n "$ctx" ] || { [ "$prompt" = fp4 ] && ctx=131072 || ctx=32768; }
  fl_int FERRITE_CTX "$ctx" 1024 1048576
  # memory estimate (GB): EXL3 13 + prompt weights (NVFP4 21 | bf16 52) + drafter 4 + KV 64 KB a token a slot + ~10
  # + the prompt cache's pool (FE_PARK_GB; the server shrinks it if less is free, keeping 20 GB available)
  wts=$([ "$prompt" = fp4 ] && echo 21 || echo 52)
  need=${user_min_free:-$(( 13 + wts + 4 + ctx * slots * 64 / 1048576 + 10 + ${park%.*} ))}
  if [ "$avail" -lt "$need" ]; then
    msg="only $avail GB of memory is available; this setup needs about $need GB (ctx $ctx x $slots slots, prompt weights $prompt, a $park GB cache pool). The GPU shares this memory and running out of it can stall or reboot the machine: stop other GPU jobs, or lower FERRITE_CTX / FERRITE_SLOTS / FERRITE_PARK_GB"
    [ "${FERRITE_FORCE:-0}" = 1 ] && fl_warn "$msg; FERRITE_FORCE=1: starting anyway" || fl_die "$msg (FERRITE_FORCE=1 starts anyway)"
  fi
  memline="$avail GB available, about $need GB needed"
else
  # RTX preset (tools/serve-5080.sh): host memory for the pinned prompt weights, the embedding table and the pool;
  # the card's free memory decides the context (FE_KVQ bits a K/V value)
  hneed=$(( $([ "$prompt" = fp4 ] && echo 20 || echo 4) + ${park%.*} + 2 ))
  if [ "$avail" -lt "$hneed" ]; then
    msg="only $avail GB of host memory is available; the RTX preset needs about $hneed GB (prompt weights $prompt, a $park GB cache pool)"
    [ "${FERRITE_FORCE:-0}" = 1 ] && fl_warn "$msg; FERRITE_FORCE=1: starting anyway" || fl_die "$msg: lower FERRITE_PARK_GB (FERRITE_FORCE=1 starts anyway)"
  fi
  # device memory: a fixed part (weights, MTP head, workspaces, prompt ring; measured on a 16 GB card: 12000 MiB with
  # one slot, 12736 with the 4-slot pool) and the K/V cache, bytes a token by FE_KVQ (the engine's own figures);
  # FERRITE_VRAM_RESERVE_MIB stays free for other programs [128; 768 when the card drives a display]. A 120K-token
  # prompt adds under 10 MiB to the load-time use.
  case ${FE_KVQ:-6} in 6) bpt=28288 ;; 4) bpt=19584 ;; 0) bpt=69632 ;; *) bpt=28288 ;; esac
  fixed=${FERRITE_VRAM_FIXED_MIB:-$([ "$slots" -gt 1 ] && echo 12736 || echo 12000)}
  # a server that just let go of the card frees its memory a moment after its lock: give it up to 30 s
  for _ in $(seq 1 15); do
    [ "$gfree" -ge "$fixed" ] && break
    sleep 2
    gfree=$(nvidia-smi -i "$gidx" --query-gpu=memory.free --format=csv,noheader,nounits 2> /dev/null | tr -d ' ')
  done
  disp=$(nvidia-smi -i "$gidx" --query-gpu=display_active --format=csv,noheader 2> /dev/null | tr -d ' ')
  reserve=${FERRITE_VRAM_RESERVE_MIB:-$([ "$disp" = Enabled ] && echo 768 || echo 128)}
  room=$(( gfree - fixed - reserve ))
  units=$([ "$slots" -gt 1 ] && echo "$slots" || echo 1)
  fit=$(( room > 0 ? room * 1048576 / bpt / units / 256 * 256 : 0 ))
  ctx=${FERRITE_CTX:-}
  maxctx=$([ "$slots" -gt 1 ] && echo 28160 || echo 131072)
  if [ -z "$ctx" ]; then
    ctx=$(( fit < maxctx ? fit : maxctx ))
    [ "$ctx" -ge 8192 ] || fl_die "the card has $gfree MiB free; the model needs about $fixed MiB plus the K/V cache, so even an 8K context does not fit: close what else uses the card (nvidia-smi), or FE_KVQ=4"
    [ "$ctx" -lt "$maxctx" ] && fl_say "the card has $gfree MiB free (with $reserve MiB kept for $( [ "$disp" = Enabled ] && echo "the display and " )other programs): context $ctx instead of $maxctx (FERRITE_CTX sets it; FE_KVQ=4 fits more)"
  else
    fl_int FERRITE_CTX "$ctx" 1024 1048576
    if [ "$ctx" -gt "$fit" ]; then
      msg="FERRITE_CTX=$ctx does not fit the card: $gfree MiB free, about $fixed MiB for the model, $(( ctx * units * bpt / 1048576 )) MiB of K/V cache ($bpt bytes a token, FE_KVQ=${FE_KVQ:-6}) and $reserve MiB kept free; at most about $fit fits"
      [ "${FERRITE_FORCE:-0}" = 1 ] && fl_warn "$msg; FERRITE_FORCE=1: starting anyway" || fl_die "$msg (lower FERRITE_CTX, FE_KVQ=4, or FERRITE_FORCE=1 to try anyway)"
    fi
  fi
  memline="$avail GB host memory available, $gfree MiB free on the card"
fi

fl_say "$quant, prompt weights $prompt, drafting $dr, ctx $ctx x $slots slots, prompt cache \
$([ "$cache" = 1 ] && echo "on (a pool of up to ${park} GB, keeping 20 GB available)" || echo off); $memline"
fl_sampling "$quant" qwen3.8-27b
fl_banner "${FERRITE_MODEL_ID:-qwen3.8-27b}" "loading takes a minute"
exec "$bin" --engine ferrite --model-dir "$quant" --weights-dir "$quant" \
  --nvfp4-dir "$nvfp4" --draft-dir "$draft" \
  --host "$FERRITE_HOST" --port "$FERRITE_PORT" --served-model-name "${FERRITE_MODEL_ID:-qwen3.8-27b}" \
  --slots "$slots" --max-model-len "$ctx" --prefill-chunk "$chunk" "$@"
