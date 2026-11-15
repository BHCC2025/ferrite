#!/bin/bash
# tools/launch-lib.sh: what every Ferrite launcher does the same way. Sourced (after tools/site.sh) by
# scripts/serve.sh (DGX Spark, one node), tools/serve-tp2.sh (two Sparks), tools/serve-5080.sh (RTX 5080 / 5060 Ti),
# tools/serve-fnext.sh (Qwen3.8-Flash-Next) and scripts/serve-xe.sh (Intel Arc Pro B70):
#   - network: FERRITE_HOST [0.0.0.0: every machine on the local network can reach the server; there is no API key,
#     so use it on a trusted network or firewall the port; FERRITE_HOST=127.0.0.1 keeps it on this machine],
#     FERRITE_PORT [8000]. The port is checked before the model loads.
#   - GPU locks: a server holds the node's lock FERRITE_GPU_LOCK [/var/tmp/ferrite-gpu.lock] shared and its own
#     card's lock ${FERRITE_GPU_LOCK%.lock}.<card UUID>.lock exclusively, so servers on different cards run side by
#     side while a whole-node job (a test, a KL run, a TP2 job: they take the node lock exclusively) excludes them all
#     (the scheme of tools/site.sh's ferrite_gpu_exec). A launcher whose card is taken stops at once and names the
#     process holding it; FERRITE_LOCK_WAIT=1 (or FERRITE_GPU_WAIT=1) waits instead.
#   - the launcher's own process becomes fe-server (exec, the locks held on inherited file descriptors), so its pid is
#     the server's: Ctrl-C, `kill <pid>` or `kill $!` stop it cleanly and free the card, the port and the locks.
#   - FERRITE_FORCE=1 starts even when a memory check says it will not fit (at your own risk).
# Nothing here prints secrets or changes the system.

# fl_die / fl_say use the launcher's name ($fl_name, set by the launcher)
fl_die() { echo "${fl_name:-ferrite}: $*" >&2; exit 1; }
fl_say() { echo "${fl_name:-ferrite}: $*"; }
fl_warn() { echo "${fl_name:-ferrite}: warning: $*" >&2; }

# fl_int NAME VALUE MIN MAX: VALUE must be a whole number in [MIN, MAX]
fl_int() {
  case $2 in
    ''|*[!0-9]*) fl_die "$1 must be a whole number from $3 to $4, not '$2'" ;;
  esac
  [ "$2" -ge "$3" ] && [ "$2" -le "$4" ] || fl_die "$1 must be from $3 to $4, not $2"
}

# fl_net: FERRITE_HOST / FERRITE_PORT defaults and checks; then the port must be free
fl_net() {
  export FERRITE_HOST=${FERRITE_HOST:-0.0.0.0}
  export FERRITE_PORT=${FERRITE_PORT:-8000}
  fl_int FERRITE_PORT "$FERRITE_PORT" 1 65535
  fl_port_free "$FERRITE_HOST" "$FERRITE_PORT"
}

# fl_port_free HOST PORT: stop now (before the model loads) when something listens on the port
fl_port_free() {
  local host=$1 port=$2 who=""
  if command -v python3 > /dev/null; then
    # the same bind the server makes (SO_REUSEADDR, as tokio sets it): fails only when someone listens there
    python3 - "$host" "$port" <<'EOF' 2> /dev/null && return 0
import socket, sys
s = socket.socket(socket.AF_INET6 if ":" in sys.argv[1] else socket.AF_INET)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((sys.argv[1], int(sys.argv[2])))
EOF
  elif ! (exec 3<> "/dev/tcp/127.0.0.1/$port") 2> /dev/null; then
    return 0
  fi
  command -v ss > /dev/null && who=$(ss -ltnpH "sport = :$port" 2> /dev/null | grep -o 'users:(([^)]*' | head -n 1 | sed 's/users:((//')
  fl_die "port $port is already in use${who:+ (by $who)}: stop that server, or pick another port with FERRITE_PORT"
}

# fl_nvidia_cards: the visible NVIDIA cards, one "index,uuid,MiB total,MiB free,name" line each, in CUDA's order
# (CUDA_VISIBLE_DEVICES: indices or UUIDs, as CUDA reads it)
fl_nvidia_cards() {
  local all vis v line
  all=$(nvidia-smi --query-gpu=index,uuid,memory.total,memory.free,name --format=csv,noheader,nounits 2> /dev/null \
    | sed 's/, /,/g') || return 1
  if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then printf '%s\n' "$all"; return 0; fi
  IFS=, read -ra vis <<< "$CUDA_VISIBLE_DEVICES"
  for v in "${vis[@]}"; do
    v=${v// /}
    while IFS= read -r line; do
      local idx=${line%%,*} rest=${line#*,}; local uuid=${rest%%,*}
      if [ "$v" = "$idx" ] || [ "$v" = "$uuid" ] || { [[ $v == GPU-* ]] && [[ $uuid == "$v"* ]]; }; then
        printf '%s\n' "$line"
      fi
    done <<< "$all"
  done
}

# fl_holder FILE: the process holding a lock file (lslocks), else the note the holder wrote into it
fl_holder() {
  local pid h=""
  if command -v lslocks > /dev/null; then
    pid=$(lslocks -n -o PID,PATH 2> /dev/null | awk -v p="$(readlink -f "$1")" '$2 == p { print $1; exit }')
    [ -n "$pid" ] && h="pid $pid: $(tr '\0' ' ' < "/proc/$pid/cmdline" 2> /dev/null | cut -c1-160)"
  fi
  [ -n "$h" ] || h=$(head -c 200 "$1" 2> /dev/null | tr -d '\n')
  echo "$h"
}

# fl_lock FILE MODE(-s|-x) WHAT: take the lock on a new file descriptor (kept by the server after exec)
fl_lock() {
  local l=$1 mode=$2 what=$3 fd h
  mkdir -p "$(dirname "$l")" || fl_die "cannot create the lock folder $(dirname "$l")"
  exec {fd}>> "$l" || fl_die "cannot open the GPU lock $l"
  flock -n "$mode" "$fd" && return 0
  h=$(fl_holder "$l")
  if [ "${FERRITE_LOCK_WAIT:-${FERRITE_GPU_WAIT:-0}}" = 1 ]; then
    fl_say "waiting for $what ($l, held by ${h:-another Ferrite job})"
    flock "$mode" "$fd"
  else
    fl_die "another Ferrite GPU job holds $what ($l${h:+; $h}): stop it first, or set FERRITE_LOCK_WAIT=1 to wait for it"
  fi
}

# fl_take_locks ID...: the node lock shared, then each card's lock (by UUID, or PCI address for an Intel card)
fl_take_locks() {
  local id l
  fl_lock "$FERRITE_GPU_LOCK" -s "the whole node"
  for id in $(printf '%s\n' "$@" | sort -u); do
    l=${FERRITE_GPU_LOCK%.lock}.${id//[^A-Za-z0-9._-]/_}.lock
    fl_lock "$l" -x "card $id"
    # the holder note (the process that will be fe-server, its port) for the next launcher's message
    printf 'pid %s: %s on port %s since %s\n' "$$" "${fl_name:-ferrite}" "${FERRITE_PORT:-?}" "$(date -u +%FT%TZ)" > "$l" 2> /dev/null || true
  done
}

# fl_banner URL line: where the server answers and who can reach it
fl_banner() {
  local model_id=$1 what=${2:-}
  if [ "$FERRITE_HOST" = 0.0.0.0 ] || [ "$FERRITE_HOST" = "::" ]; then
    fl_say "http://$(uname -n):$FERRITE_PORT/v1 as model '$model_id'${what:+ ($what)}"
    fl_say "listening on every interface ($FERRITE_HOST): any machine on the network can use it, and there is no API key; FERRITE_HOST=127.0.0.1 keeps it on this machine"
  else
    fl_say "http://$FERRITE_HOST:$FERRITE_PORT/v1 as model '$model_id'${what:+ ($what)}"
  fi
}

# fl_built_arch BIN: the GPU architecture the binary's kernels were built for (sm_121a / sm_120a), or empty:
# `fe-server --version` names it ("kernels sm_12Xa"); else the architecture named most often in the binary (each
# embedded cubin names its own)
fl_built_arch() {
  local a
  a=$("$1" --version 2> /dev/null | grep -o 'kernels sm_12[0-9]a' | cut -d' ' -f2)
  [ -n "$a" ] || a=$(grep -a -o 'sm_12[0-9]a' "$1" 2> /dev/null | sort | uniq -c | sort -rn | awk 'NR == 1 { print $2 }')
  echo "$a"
}

# fl_check_arch BIN CAP: stop when the binary's kernels were built for another GPU than this compute capability
fl_check_arch() {
  local b=$1 cap=$2 built want
  case $cap in 12.1) want=sm_121a ;; 12.0) want=sm_120a ;; *) want="" ;; esac
  built=$(fl_built_arch "$b")
  [ -n "$built" ] && [ -n "$want" ] || return 0
  [ "$built" = "$want" ] && return 0
  fl_die "$b was built for $built, but this GPU (compute capability $cap) needs $want: rebuild with FE_KARCH=$want cargo build --release ... (docs/quickstart.md, Clean machine)"
}
