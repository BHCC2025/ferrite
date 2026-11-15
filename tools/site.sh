#!/bin/bash
# tools/site.sh: the site settings of the tools/ scripts (each one sources this file). Every setting is an
# environment variable whose default is the two-node setup ferrite was built on (two DGX Sparks on a direct RoCE
# link); set it to change it. The engine reads the same names (crates/fe-site). Values must not contain spaces.
# Run directly (`tools/site.sh`) it prints the resolved settings. FERRITE_FABRIC=auto takes the fabric settings
# from tools/fabric-detect.sh instead of the defaults; the two-node scripts do that too while FERRITE_TP_HOST is unset.
# Local settings can live in tools/site.local.sh (not tracked; FERRITE_SITE_FILE names another file), read first;
# tools/site.local.sh.example shows the format.
_sf=${FERRITE_SITE_FILE:-$(dirname "${BASH_SOURCE[0]}")/site.local.sh}
# shellcheck source=/dev/null
[ -f "$_sf" ] && . "$_sf"
unset _sf

# model folders: one root holding the three standard folders, or each set on its own
: "${FERRITE_MODELS:=/var/tmp/models}"
: "${FERRITE_EXL3_DIR:=$FERRITE_MODELS/Qwen3.8-27B-EXL3-3.0bpw-mix}"
: "${FERRITE_NVFP4_DIR:=${FE_PROMPT_WEIGHTS:-$FERRITE_MODELS/nvidia-Qwen3.8-27B-NVFP4}}"
: "${FERRITE_DRAFT_DIR:=${FE_DRAFT_DIR:-$FERRITE_MODELS/z-lab-Qwen3.8-27B-DFlash2}}"
# the XQ quant of the Intel Arc Pro B70 engine (scripts/serve-xe.sh; docs/arc-b70.md)
: "${FERRITE_XQ_DIR:=$FERRITE_MODELS/Qwen3.8-27B-XQ-5.0bpw-mix}"
# state: logs and run files, and the locks: the node's GPU lock (servers hold it shared, plus their own card's lock
# ${FERRITE_GPU_LOCK%.lock}.<card UUID>.lock exclusively; whole-node jobs such as tests and KL runs take it
# exclusively), and the TP2 lock on rank 0's node. Without FERRITE_STATE_DIR they stay at their /var/tmp paths.
_fs=${FERRITE_STATE_DIR:-}
: "${FERRITE_STATE_DIR:=/var/tmp/ferrite}"
: "${FERRITE_GPU_LOCK:=${_fs:+$_fs/gpu.lock}}"
: "${FERRITE_GPU_LOCK:=/var/tmp/ferrite-gpu.lock}"
: "${FERRITE_TP2_LOCK:=${_fs:+$_fs/tp2.lock}}"
: "${FERRITE_TP2_LOCK:=/var/tmp/ferrite-tp2.lock}"
unset _fs
# two nodes: rank 1's host (ssh; no default), rank 0's fabric address (the rendezvous; unset = detected by the
# two-node scripts), the NCCL socket interface, the RDMA devices (one or two, comma-separated) and the RoCE v2 GID
# index of the fabric address on them. The interface and device names are the DGX Spark's ConnectX-7 ports.
: "${FERRITE_PEER:=${PEER:-}}"
if [ "${FERRITE_FABRIC:-}" = auto ] && [ -n "$FERRITE_PEER" ]; then
  eval "$("$(dirname "${BASH_SOURCE[0]}")"/fabric-detect.sh "$FERRITE_PEER")"
fi
: "${FERRITE_TP_HOST:=}"
: "${FERRITE_NET_IF:=enp1s0f0np0}"
: "${FERRITE_ROCE_HCA:=${FE_ROCE_HCA:-rocep1s0f0,roceP2p1s0f0}}"
: "${FERRITE_ROCE_GID:=${FE_ROCE_GID:-5}}"
# memory (GB) each node must have available before a TP2 job starts
: "${FERRITE_MIN_FREE_GB:=45}"
# the KL reference of tools/kl.sh / kl-tp2.sh: token ids, BF16 reference logits and the scorer
: "${FERRITE_KL_DIR:=/var/tmp/kl27b}"
# ferrite_gpu_exec CARDS CMD...: exec CMD holding this node's GPU lock SHARED and one EXCLUSIVE lock per card it uses
# (CARDS: comma-separated GPU UUIDs, ${FERRITE_GPU_LOCK%.lock}.<uuid>.lock). Servers on different cards run side by
# side; a job that takes $FERRITE_GPU_LOCK itself (exclusively: tests, KL runs) still excludes every server and the
# reverse. A held lock fails at once with exit 75 and says which; FERRITE_GPU_WAIT=1 waits for it instead.
ferrite_gpu_exec() {
  local cards=$1 c l me nw=(-n) cs=()
  shift
  me=$(basename "$0")
  [ "${FERRITE_GPU_WAIT:-0}" = 1 ] && nw=()
  mkdir -p "$(dirname "$FERRITE_GPU_LOCK")"
  local cmd=(flock "${nw[@]}" -s -E 75 "$FERRITE_GPU_LOCK")
  if ! flock -n -s "$FERRITE_GPU_LOCK" true; then
    [ ${#nw[@]} = 1 ] && { echo "$me: a whole-node Ferrite GPU job holds $FERRITE_GPU_LOCK" >&2; return 75; }
    echo "$me: waiting for a whole-node Ferrite GPU job ($FERRITE_GPU_LOCK)" >&2
  fi
  IFS=, read -ra cs <<<"$cards"
  [ ${#cs[@]} -ge 1 ] || { echo "$me: ferrite_gpu_exec: no card" >&2; return 2; }
  for c in "${cs[@]}"; do
    l=${FERRITE_GPU_LOCK%.lock}.$c.lock
    if ! flock -n "$l" true; then
      [ ${#nw[@]} = 1 ] && { echo "$me: another Ferrite GPU job uses card $c (it holds $l)" >&2; return 75; }
      echo "$me: waiting for card $c (another Ferrite GPU job holds $l)" >&2
    fi
    cmd+=(flock "${nw[@]}" -E 75 "$l")
  done
  exec "${cmd[@]}" "$@"
}
# ferrite_card_uuid: the UUID of the card a one-card job uses (the first CUDA_VISIBLE_DEVICES entry, an index or a
# UUID, else card 0), for ferrite_gpu_exec; card<N> when nvidia-smi cannot tell
ferrite_card_uuid() {
  local v=${CUDA_VISIBLE_DEVICES:-0} u=
  v=${v%%,*}
  if [[ $v == GPU-* ]]; then u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | grep -m1 "^$v" || true)
  else u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F', ' -v i="$v" '$1 == i { print $2 }' || true); fi
  echo "${u:-card$v}"
}
export FERRITE_MODELS FERRITE_EXL3_DIR FERRITE_NVFP4_DIR FERRITE_DRAFT_DIR FERRITE_XQ_DIR FERRITE_STATE_DIR FERRITE_GPU_LOCK \
  FERRITE_TP2_LOCK FERRITE_PEER FERRITE_TP_HOST FERRITE_NET_IF FERRITE_ROCE_HCA FERRITE_ROCE_GID FERRITE_MIN_FREE_GB \
  FERRITE_KL_DIR
# the environment both TP ranks start with (the peer's arrives over ssh): NCCL's and the engine's fabric settings,
# the checkpoint folders and the NCCL library when it is set
_ferrite_fab() {
  FERRITE_FAB="NCCL_IB_GID_INDEX=$FERRITE_ROCE_GID NCCL_SOCKET_IFNAME=$FERRITE_NET_IF NCCL_IB_HCA=$FERRITE_ROCE_HCA \
NCCL_DEBUG=${NCCL_DEBUG:-WARN} FERRITE_ROCE_HCA=$FERRITE_ROCE_HCA FERRITE_ROCE_GID=$FERRITE_ROCE_GID \
FERRITE_EXL3_DIR=$FERRITE_EXL3_DIR FERRITE_NVFP4_DIR=$FERRITE_NVFP4_DIR FERRITE_DRAFT_DIR=$FERRITE_DRAFT_DIR\
${FERRITE_NCCL_LIB:+ FERRITE_NCCL_LIB=$FERRITE_NCCL_LIB}"
}
_ferrite_fab
# ferrite_two_node: every two-node script calls this after sourcing this file. It needs FERRITE_PEER and, while
# FERRITE_TP_HOST is unset, takes the fabric settings from tools/fabric-detect.sh (as FERRITE_FABRIC=auto does).
ferrite_two_node() {
  if [ -z "$FERRITE_PEER" ]; then
    echo "two nodes: set FERRITE_PEER to node B's ssh host (e.g. in tools/site.local.sh)" >&2
    return 1
  fi
  if [ -z "$FERRITE_TP_HOST" ]; then
    local fab
    fab=$("$(dirname "${BASH_SOURCE[0]}")"/fabric-detect.sh "$FERRITE_PEER") || {
      echo "two nodes: set FERRITE_TP_HOST (node A's address on the RoCE link) and the fabric settings of tools/site.sh" >&2
      return 1
    }
    eval "$fab"
    echo "two nodes: detected" $fab >&2
    export FERRITE_TP_HOST FERRITE_NET_IF FERRITE_ROCE_HCA FERRITE_ROCE_GID
    _ferrite_fab
  fi
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  # what the two-node scripts will use: with FERRITE_PEER set and FERRITE_TP_HOST unset they detect the fabric
  if [ -n "$FERRITE_PEER" ] && [ -z "$FERRITE_TP_HOST" ]; then
    if fab=$("$(dirname "${BASH_SOURCE[0]}")"/fabric-detect.sh "$FERRITE_PEER" 2> /dev/null); then
      eval "$fab"; echo "(the fabric settings below are detected from both nodes, as the two-node scripts do)"
    else
      echo "(fabric detection with $FERRITE_PEER failed: the two-node scripts would stop; these are the defaults)"
    fi
  fi
  for v in FERRITE_MODELS FERRITE_EXL3_DIR FERRITE_NVFP4_DIR FERRITE_DRAFT_DIR FERRITE_XQ_DIR FERRITE_STATE_DIR FERRITE_GPU_LOCK \
    FERRITE_TP2_LOCK FERRITE_PEER FERRITE_TP_HOST FERRITE_NET_IF FERRITE_ROCE_HCA FERRITE_ROCE_GID FERRITE_MIN_FREE_GB \
    FERRITE_KL_DIR; do
    printf '%-20s %s\n' "$v" "${!v}"
  done
fi
