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
# state: logs and run files, and the locks (a GPU lock on each node, the TP2 lock on rank 0's node; every GPU job
# on a node must take the same lock, so without FERRITE_STATE_DIR they stay at their /var/tmp paths)
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
  for v in FERRITE_MODELS FERRITE_EXL3_DIR FERRITE_NVFP4_DIR FERRITE_DRAFT_DIR FERRITE_XQ_DIR FERRITE_STATE_DIR FERRITE_GPU_LOCK \
    FERRITE_TP2_LOCK FERRITE_PEER FERRITE_TP_HOST FERRITE_NET_IF FERRITE_ROCE_HCA FERRITE_ROCE_GID FERRITE_MIN_FREE_GB \
    FERRITE_KL_DIR; do
    printf '%-20s %s\n' "$v" "${!v}"
  done
fi
