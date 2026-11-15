#!/usr/bin/env bash
# Run a Ferrite server in the background and manage it: the same commands on every target.
#
#   scripts/ferritectl.sh start [TARGET] [-- extra fe-server arguments]
#   scripts/ferritectl.sh stop | restart [TARGET] | status | log [-f]
#
# TARGET picks the launcher [auto: the one for this machine's GPU]:
#   spark   scripts/serve.sh        Qwen3.8-27B on one DGX Spark
#   tp2     tools/serve-tp2.sh      Qwen3.8-27B on two DGX Sparks (run on node A; FERRITE_PEER = node B)
#   rtx     tools/serve-5080.sh     Qwen3.8-27B on one RTX 5080 / RTX 5060 Ti 16 GB
#   fnext   tools/serve-fnext.sh    Qwen3.8-Flash-Next on one or two RTX 5060 Ti / 16 GB cards
#   b70     scripts/serve-xe.sh     Qwen3.8-27B on one Intel Arc Pro B70
# Every launcher setting applies (FERRITE_HOST [0.0.0.0], FERRITE_PORT [8000], FERRITE_CTX, ...: the launcher's
# --help). One server a port: the log is <ferrite folder>/logs/server-<port>.log (the previous run's is kept as
# .log.1) and the server's pid is in logs/server-<port>.pid (FERRITE_LOG_DIR moves both).
# start   starts the launcher in its own session, waits until /health answers (FERRITE_START_TIMEOUT seconds, 900)
#         and prints the address; if the server exits while loading, it prints the end of the log and fails
# stop    stops it (SIGTERM; open streams get 5 s; SIGKILL after 60 s) and waits until the card is free
# status  running or not, and /health
# log     the log (log -f follows it)
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
die() { echo "ferritectl: $*" >&2; exit 1; }
cmd=${1:-}; shift || true
case $cmd in -h|--help|'') sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;; esac
port=${FERRITE_PORT:-8000}
logs=${FERRITE_LOG_DIR:-$root/logs}
log=$logs/server-$port.log
pidf=$logs/server-$port.pid
pid() { [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2> /dev/null && cat "$pidf"; }

launcher_of() {
  case $1 in
    spark) echo scripts/serve.sh ;;
    tp2) echo tools/serve-tp2.sh ;;
    rtx|5080|5060ti) echo tools/serve-5080.sh ;;
    fnext) echo tools/serve-fnext.sh ;;
    b70|xe) echo scripts/serve-xe.sh ;;
    auto)
      local cap d
      cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | head -n 1 | tr -d ' ' || true)
      case $cap in
        12.1) echo scripts/serve.sh; return ;;
        12.0) echo tools/serve-5080.sh; return ;;
      esac
      for d in /sys/class/drm/card*/device/driver; do
        [ "$(basename "$(readlink -f "$d" 2> /dev/null)")" = xe ] && { echo scripts/serve-xe.sh; return; }
      done
      die "no DGX Spark, RTX 50 or Arc B70 GPU found; name the target (ferritectl.sh start spark|tp2|rtx|fnext|b70)" ;;
    *) die "unknown target '$1': spark, tp2, rtx, fnext or b70" ;;
  esac
}

health() { curl -s -m 3 "http://127.0.0.1:$port/health" 2> /dev/null; }

start() {
  local target=auto
  if [ $# -gt 0 ] && [ "$1" != -- ]; then target=$1; shift; fi
  [ "${1:-}" = -- ] && shift
  local l; l=$(launcher_of "$target")
  if p=$(pid); then die "a server started by ferritectl already runs on port $port (pid $p): ferritectl.sh stop first"; fi
  mkdir -p "$logs"
  [ -s "$log" ] && mv -f "$log" "$log.1"   # the previous run's log stays as .1
  echo "ferritectl: $l $* (log $log)"
  # its own session: the launcher's process becomes the server, so the pid file holds the server's pid
  (cd "$root" && exec setsid "$l" "$@" > "$log" 2>&1 < /dev/null) &
  echo $! > "$pidf"
  local t0=$SECONDS p
  p=$(cat "$pidf")
  until health | grep -q '"status"'; do
    if ! kill -0 "$p" 2> /dev/null; then
      tail -n 15 "$log" | sed 's/^/  | /' >&2
      rm -f "$pidf"
      die "the server stopped while starting (the end of $log is above)"
    fi
    [ $((SECONDS - t0)) -lt "${FERRITE_START_TIMEOUT:-900}" ] || die "no answer on port $port after ${FERRITE_START_TIMEOUT:-900} s (still loading? see $log)"
    sleep 3
  done
  echo "ferritectl: ready after $((SECONDS - t0)) s: $(health)"
  grep -m 2 -E '^[a-z0-9-]+: (http|listening)' "$log" | sed 's/^/  /' || true
}

stop() {
  local p
  p=$(pid) || { rm -f "$pidf"; echo "ferritectl: no server started by ferritectl on port $port"; return 0; }
  echo "ferritectl: stopping pid $p (port $port)"
  kill -TERM "$p" 2> /dev/null || true
  local i
  for i in $(seq 1 60); do kill -0 "$p" 2> /dev/null || break; sleep 1; done
  if kill -0 "$p" 2> /dev/null; then echo "ferritectl: still running after 60 s: SIGKILL"; kill -KILL "$p" 2> /dev/null || true; sleep 2; fi
  rm -f "$pidf"
  echo "ferritectl: stopped"
}

case $cmd in
  start) start "$@" ;;
  stop) stop ;;
  restart) stop; start "$@" ;;
  status)
    if p=$(pid); then echo "ferritectl: running, pid $p, port $port: $(health || echo 'no answer yet (loading?)')"
    else echo "ferritectl: not running on port $port"; h=$(health) && [ -n "$h" ] && echo "  (but something answers on port $port: $h)"; exit 3
    fi ;;
  log) if [ "${1:-}" = -f ]; then exec tail -f "$log"; else cat "$log"; fi ;;
  *) die "unknown command '$cmd' (start, stop, restart, status, log)" ;;
esac
