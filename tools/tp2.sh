#!/bin/bash
# Run one ferrite command as TP2 across this host (rank 0) and the peer (rank 1): rank 1 is started detached on
# the peer (same binary path and model path), rank 0 runs in the foreground; both under their node's GPU lock.
# Lock order, the same for every TP2 job (tp2.sh, serve-tp2.sh, kl-tp2.sh): the global TP2 lock (on rank 0's
# node, held for the whole job until rank 1 has also let go of its node), then rank 0's GPU lock, then rank 1 takes
# the peer's. One TP2 job at a time, so two TP2 jobs can never hold one node each; TP1 jobs take only their own
# node's GPU lock and just queue.
# This script's process holds the locks (on file descriptors) and forwards Ctrl-C / SIGTERM to rank 0, so stopping
# it (kill <pid>) stops both ranks and frees both nodes.
# usage: tools/tp2.sh <ferrite args...>      e.g. tools/tp2.sh "$FERRITE_EXL3_DIR" bench 2048 128 4
#        FOLLOW=<model dir> tools/tp2.sh <any rank-0 command...>: rank 1 runs `ferrite <model> follow` and replays
#        rank 0's engine calls (rank 0 = the scheduler / server: e.g. `target/release/ferrite <model> sched-test`)
# env: FE_BIN, FE_TP_PORT (29620), extra FE_* variables are passed to both ranks via TP_ENV, TP2_TIMEOUT (seconds,
#      default 3600; 0 = none, e.g. a server), TP2_DRY=1 (print what would run, run nothing), TP2_NOWAIT=1 (stop at
#      once, naming the holder, when a lock is taken; the default waits for it); the site settings (peer, fabric,
#      locks, state dir) are tools/site.sh's FERRITE_* variables
set -e
. "$(dirname "$0")/site.sh"
ferrite_two_node || exit 1
BIN=$(readlink -f ${FE_BIN:-$(dirname $0)/../target/release/ferrite})
PEER=$FERRITE_PEER
FAB="$FERRITE_FAB FE_TP_WORLD=2 FE_TP_ADDR=$FERRITE_TP_HOST:${FE_TP_PORT:-29620} FE_MTP=0 $TP_ENV"
# the command line travels as base64 of NUL-separated args (any prompt survives ssh + bash -c)
B64=$(printf '%s\0' "$@" | base64 -w0)
B64_1=$B64; BIN0=$BIN; B64_0=$B64
if [ -n "$FOLLOW" ]; then
  B64_1=$(printf '%s\0' "$FOLLOW" follow | base64 -w0)
  BIN0=$(readlink -f "$1"); B64_0=$(printf '%s\0' "${@:2}" | base64 -w0)
fi
EXEC=$(readlink -f $(dirname $0)/tp2-exec.sh)
if [ "${TP2_DRY:-0}" = 1 ]; then
  if [ -n "$FOLLOW" ]; then R1="$FOLLOW follow"; R0="${*:2}"; else R1="$*"; R0="$*"; fi
  echo "locks:  $FERRITE_TP2_LOCK, then $FERRITE_GPU_LOCK here and on $PEER"
  echo "rank 1: ssh $PEER env $FAB FE_TP_RANK=1 timeout ${TP2_TIMEOUT:-3600} flock $FERRITE_GPU_LOCK $BIN $R1"
  echo "rank 0: env $FAB FE_TP_RANK=0 timeout ${TP2_TIMEOUT:-3600} $BIN0 $R0"
  echo "logs:   $FERRITE_STATE_DIR (rank 1's on $PEER)"
  exit 0
fi
mkdir -p $FERRITE_STATE_DIR
LOG=$FERRITE_STATE_DIR/tp2-r1-$(date +%H%M%S).log
# both ranks must run the same build (the bootstrap hello and every collective assume it)
[ "$(sha256sum < $BIN)" = "$(ssh -n $PEER "sha256sum < $BIN")" ] || { echo "tp2.sh: $BIN differs between $(hostname) and $PEER: rebuild both"; exit 1; }
# the locks on this process's file descriptors (the TP2 lock first, then this node's GPU lock)
take() { # lock-file name
  local fd
  mkdir -p "$(dirname "$1")"
  exec {fd}>> "$1"
  flock -n "$fd" && return 0
  if [ "${TP2_NOWAIT:-0}" = 1 ]; then
    local pid h=""
    pid=$(lslocks -n -o PID,PATH 2> /dev/null | awk -v p="$(readlink -f "$1")" '$2 == p { print $1; exit }')
    [ -n "$pid" ] && h="pid $pid: $(tr '\0' ' ' < /proc/$pid/cmdline 2> /dev/null | cut -c1-160)"
    echo "tp2.sh: another Ferrite job holds the $2 ($1): ${h:-$(head -c 200 "$1" 2> /dev/null)}" >&2
    echo "        stop it first (or start without TP2_NOWAIT=1 / with FERRITE_LOCK_WAIT=1 to wait for it)" >&2
    exit 75
  fi
  echo "tp2.sh: waiting for the $2 ($1)" >&2
  flock "$fd"
}
take "$FERRITE_TP2_LOCK" "TP2 lock"
take "$FERRITE_GPU_LOCK" "GPU lock of $(hostname -s)"
printf 'pid %s: tp2.sh (%s) since %s\n' "$$" "${*:1:3}" "$(date -u +%FT%TZ)" | tee "$FERRITE_TP2_LOCK" > "$FERRITE_GPU_LOCK" 2> /dev/null || true
min=$FERRITE_MIN_FREE_GB
here=$(free -g | awk '/^Mem/ { print $7 }')
there=$(ssh -n $PEER "free -g" | awk '/^Mem/ { print $7 }')
[ "${here:-0}" -ge "$min" ] || { echo "tp2.sh: $(hostname): only $here GB available, $min wanted (FERRITE_MIN_FREE_GB)"; exit 1; }
[ "${there:-0}" -ge "$min" ] || { echo "tp2.sh: $PEER: only $there GB available, $min wanted (FERRITE_MIN_FREE_GB)"; exit 1; }
ssh -n $PEER "mkdir -p $FERRITE_STATE_DIR && cd /tmp && setsid -f bash -c 'env $FAB FE_TP_RANK=1 timeout ${TP2_TIMEOUT:-3600} flock $FERRITE_GPU_LOCK $EXEC $BIN $B64_1 > $LOG 2>&1' < /dev/null > /dev/null 2>&1"
# rank 0 in the background of this shell so a signal to this script reaches it (bash waits otherwise)
env $FAB FE_TP_RANK=0 timeout ${TP2_TIMEOUT:-3600} $EXEC $BIN0 $B64_0 &
r0=$!
trap 'kill -TERM $r0 2> /dev/null' TERM INT HUP
rc=0
wait $r0 || rc=$?
# a signal interrupts wait: wait again for rank 0's own shutdown
while kill -0 $r0 2> /dev/null; do wait $r0 || rc=$?; done
trap - TERM INT HUP
# keep the TP2 lock until rank 1 has left the peer's GPU lock (it exits right after rank 0), so the next TP2
# job's rank 1 does not queue behind this one
ssh -n $PEER "flock -w 120 $FERRITE_GPU_LOCK true" || true
echo "(rank 1 log: $PEER:$LOG)" >&2
exit $rc
