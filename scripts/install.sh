#!/usr/bin/env bash
# Install Ferrite on a DGX Spark and start it: one command from a fresh machine to a running server.
#
#   curl -fsSL https://raw.githubusercontent.com/BHCC2025/ferrite/main/scripts/install.sh | bash
#   scripts/install.sh [options]        (inside a checkout or an unpacked release: installs and starts that one)
#
# Each step says what it does and is skipped when it is already done, so running it again continues where it stopped.
#   1. checks: aarch64, a GB10 GPU (compute capability 12.1) with the NVIDIA driver, enough free memory
#   2. Ferrite: this checkout or release folder; otherwise, in $FERRITE_HOME, the release tarball of this version
#      (ready-made binaries), or when there is none a clone of the repository and a source build (rustup, the pinned
#      toolchains through scripts/check-deps.sh --build --install, cargo build)
#   3. the Hugging Face CLI, in ~/.venvs/hf
#   4. the models: scripts/fetch-models.sh into $FERRITE_MODELS, skipped when they are already there and verified
#   5. the server: scripts/serve.sh in the background (log in <ferrite folder>/logs/server.log), until it answers;
#      it runs in its own process group, whose id is in logs/server.pid: stop it with kill -- -$(cat logs/server.pid)
#   6. a test request: OK, or what went wrong
# It needs no sudo: when a system package is missing it stops and prints the apt command to run first.
#
# Options:
#   --source       build from source even when a release tarball exists
#   --no-serve     stop after the models (step 4)
#   -h, --help     this text
# Environment:
#   FERRITE_HOME         where it installs when not run from a checkout or release folder [~/ferrite]
#   FERRITE_MODELS       the model folder [/var/tmp/models]; the other settings of scripts/serve.sh (FERRITE_PORT,
#                        FERRITE_HOST, FERRITE_CTX, FERRITE_SLOTS, ...) apply to the server it starts
#   FERRITE_MIN_FREE_GB  memory that must be available before it starts the server [95]
#   FERRITE_VERSION      the release to install [0.4.2]
#   FERRITE_RELEASE_URL  the release tarball [the GitHub release of FERRITE_VERSION]; its .sha256 is checked
#   FERRITE_REPO_URL     the repository to clone when there is no release [https://github.com/BHCC2025/ferrite.git]
set -euo pipefail

VERSION=${FERRITE_VERSION:-0.4.2}
REPO_URL=${FERRITE_REPO_URL:-https://github.com/BHCC2025/ferrite.git}
RELEASE_URL=${FERRITE_RELEASE_URL:-https://github.com/BHCC2025/ferrite/releases/download/v$VERSION/ferrite-$VERSION-dgx-spark-aarch64.tar.gz}
source_build=0
serve=1
for a in "$@"; do
  case $a in
    --source) source_build=1 ;;
    --no-serve) serve=0 ;;
    -h|--help)
      if [ -f "${BASH_SOURCE[0]:-}" ]; then sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'
      else echo "usage: install.sh [--source] [--no-serve] (see the comment at the top of scripts/install.sh)"; fi
      exit 0 ;;
    *) echo "install: unknown argument '$a' (--help)" >&2; exit 1 ;;
  esac
done

n=0
step() { n=$((n + 1)); echo; echo "==> [$n/6] $*"; }
say() { echo "    $*"; }
die() { echo; echo "install: ERROR: $*" >&2; exit 1; }
need_pkg() { die "$1 is missing. It needs a system package (the one step that needs sudo): sudo apt install -y $2, then run this again"; }

# ---------------------------------------------------------------------------------------------------- 1. checks
step "Checking this machine"
[ "$(uname -m)" = aarch64 ] || die "Ferrite runs on DGX Spark (aarch64); this machine is $(uname -m)"
command -v nvidia-smi > /dev/null || die "nvidia-smi not found: the NVIDIA driver is missing (DGX OS ships it)"
read -r gpu cap drv < <(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null \
  | head -n 1 | awk -F', ' '{ gsub(/ /, "_", $1); print $1, $2, $3 }') || true
[ -n "${cap:-}" ] || die "nvidia-smi found no GPU"
[ "$cap" = 12.1 ] || die "the GPU is ${gpu//_/ } (compute capability $cap); Ferrite's kernels are built for the GB10 of DGX Spark (12.1) only"
say "GPU ${gpu//_/ } (compute capability $cap), driver $drv"
for t in curl python3; do command -v $t > /dev/null || need_pkg "$t" "$t"; done
port=${FERRITE_PORT:-8000}
running=0
curl -sf -m 3 "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1 && running=1
avail=$(awk '/MemAvailable/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
min=${FERRITE_MIN_FREE_GB:-95}
if [ $running = 1 ]; then
  say "a server already answers on port $port ($avail GB of memory available)"
elif [ $serve = 1 ] && [ "$avail" -lt "$min" ]; then
  die "only $avail GB of memory is available; the server needs about $min GB (the model, 4 x 128K of KV cache and the prompt cache). Stop other GPU jobs, or lower FERRITE_CTX / FERRITE_SLOTS and set FERRITE_MIN_FREE_GB to match"
elif [ $serve = 1 ]; then
  say "$avail GB of memory available (about $min GB needed)"
else
  say "$avail GB of memory available (the server will need about $min GB)"
fi

# ---------------------------------------------------------------------------------------------------- 2. Ferrite
step "Getting Ferrite"
root=""
if [ -f "${BASH_SOURCE[0]:-}" ]; then
  d=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  [ -f "$d/scripts/serve.sh" ] && [ -f "$d/tools/site.sh" ] && root=$d
fi
if [ -n "$root" ]; then
  say "using $root"
else
  root=${FERRITE_HOME:-$HOME/ferrite}
  if [ -x "$root/bin/fe-server" ] || [ -f "$root/Cargo.toml" ]; then
    say "using the Ferrite folder already in $root"
  elif [ -e "$root" ] && [ -n "$(ls -A "$root" 2> /dev/null)" ]; then
    die "$root exists and is not a Ferrite folder; move it away or set FERRITE_HOME"
  else
    got=0
    if [ $source_build = 0 ]; then
      say "downloading the release tarball: $RELEASE_URL"
      tmp=$(mktemp -d)
      if curl -fsSL -o "$tmp/r.tar.gz" "$RELEASE_URL" 2>/dev/null && curl -fsSL -o "$tmp/r.sha256" "$RELEASE_URL.sha256" 2>/dev/null; then
        [ "$(sha256sum < "$tmp/r.tar.gz" | cut -c1-64)" = "$(cut -c1-64 "$tmp/r.sha256")" ] \
          || die "the tarball's sha256 does not match $RELEASE_URL.sha256 (a broken download? run this again)"
        mkdir -p "$root"
        tar xzf "$tmp/r.tar.gz" --strip-components=1 -C "$root"
        say "unpacked into $root (sha256 checked)"
        got=1
      else
        say "no release tarball there; building from source instead"
      fi
      rm -rf "$tmp"
    fi
    if [ $got = 0 ]; then
      command -v git > /dev/null || need_pkg git git
      say "cloning $REPO_URL into $root"
      git clone -q "$REPO_URL" "$root" || die "git clone of $REPO_URL failed (network?)"
    fi
  fi
fi
cd "$root"
if [ -x bin/fe-server ] && [ ! -f Cargo.toml ]; then
  say "release binaries: bin/fe-server, bin/ferrite"
else
  say "building from source (the first build takes a few minutes; later runs only check it)"
  command -v cc > /dev/null || need_pkg "a C compiler / linker (cc)" build-essential
  if ! command -v rustup > /dev/null && [ ! -x "$HOME/.cargo/bin/rustup" ]; then
    say "installing rustup (Rust's installer) with a stable toolchain into ~/.cargo and ~/.rustup"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path -q \
      || die "the rustup installer failed (network?)"
    say "(rustup did not touch your shell profile; to use cargo yourself later: . \"\$HOME/.cargo/env\")"
  fi
  # shellcheck source=/dev/null
  [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
  say "the toolchains (scripts/check-deps.sh --build --install)"
  scripts/check-deps.sh --build --install | sed 's/^/    /' || die "the build toolchain is not complete (see above)"
  say "cargo build --release -p fe-server --features ferrite -p fe-cli"
  cargo build --release -q -p fe-server --features ferrite -p fe-cli || die "the build failed (see above)"
fi
scripts/check-deps.sh "$( [ -x bin/fe-server ] && echo bin/fe-server || echo target/release/fe-server)" 2>&1 \
  | grep -v ' memory available' | sed 's/^/    /' || die "the runtime checks failed (see above)"

# ---------------------------------------------------------------------------------------------------- 3. hf
step "The Hugging Face CLI"
hf=$HOME/.venvs/hf/bin/hf
if [ -x "$hf" ]; then
  say "already in ~/.venvs/hf"
else
  say "python3 -m venv ~/.venvs/hf && ~/.venvs/hf/bin/pip install -U huggingface_hub"
  python3 -m venv "$HOME/.venvs/hf" > /dev/null 2>&1 || need_pkg "python3-venv" python3-venv
  "$HOME/.venvs/hf/bin/pip" install -q -U huggingface_hub || die "pip install huggingface_hub failed (network?)"
  say "installed: hf $("$hf" version 2>/dev/null | awk '/version:/ { print $2 }' | head -n 1)"
fi

# ---------------------------------------------------------------------------------------------------- 4. models
export FERRITE_MODELS=${FERRITE_MODELS:-/var/tmp/models}
step "The models in $FERRITE_MODELS"
stamp=$FERRITE_MODELS/.ferrite-verified
mhash=$(sha256sum < scripts/models.manifest | cut -c1-64)
if scripts/fetch-models.sh --verify-only --quick > /dev/null 2>&1; then
  if [ "$(cat "$stamp" 2>/dev/null)" = "$mhash" ]; then
    say "present and verified earlier"
  else
    say "present; checking every file's sha256 against scripts/models.manifest (a few minutes, once)"
    scripts/fetch-models.sh --verify-only | sed 's/^/    /' || die "the model files do not match the manifest (see above)"
    echo "$mhash" > "$stamp"
  fi
else
  mkdir -p "$FERRITE_MODELS"
  free=$(df -BG --output=avail "$FERRITE_MODELS" | tail -n 1 | tr -dc 0-9)
  [ "$free" -ge 40 ] || say "warning: only $free GB free under $FERRITE_MODELS; the models take about 38 GB"
  say "downloading about 38 GB (scripts/fetch-models.sh; it resumes when run again)"
  scripts/fetch-models.sh | sed 's/^/    /' || die "the model download failed (see above)"
  echo "$mhash" > "$stamp"
fi
if [ $serve = 0 ]; then
  echo; echo "install: done (--no-serve). Start the server with: cd $root && FERRITE_MODELS=$FERRITE_MODELS scripts/serve.sh"
  exit 0
fi

# ---------------------------------------------------------------------------------------------------- 5. server
step "The server on port $port"
mkdir -p logs
if [ $running = 1 ]; then
  say "already running"
else
  say "FERRITE_MODELS=$FERRITE_MODELS scripts/serve.sh  (in the background; log: $root/logs/server.log)"
  # its own session and process group (setsid): the id in server.pid stops the group, flock and fe-server together
  # (kill -- -<id>); flock does not pass a signal on to fe-server
  setsid nohup scripts/serve.sh > logs/server.log 2>&1 < /dev/null &
  echo $! > logs/server.pid
  t0=$SECONDS
  until curl -sf -m 3 "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; do
    if ! kill -0 "$(cat logs/server.pid)" 2>/dev/null; then
      tail -n 15 logs/server.log | sed 's/^/    | /'
      die "the server stopped while starting (the end of logs/server.log is above)"
    fi
    [ $((SECONDS - t0)) -lt 900 ] || die "the server did not answer within 15 minutes (see $root/logs/server.log)"
    sleep 5
  done
  say "ready after $((SECONDS - t0)) s: $(grep -m 1 'engine ready' logs/server.log | sed 's/^ferrite: //')"
fi

# ---------------------------------------------------------------------------------------------------- 6. test
step "A test request"
body='{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Reply with the single word OK."}], "max_tokens": 16,
       "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}}'
reply=$(curl -s -m 120 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' -d "$body") \
  || die "no answer from http://127.0.0.1:$port/v1/chat/completions"
text=$(printf '%s' "$reply" | python3 -c 'import json, sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' 2>/dev/null) \
  || die "unexpected answer: $(printf '%s' "$reply" | head -c 300)"
[ -n "$text" ] || die "the server answered with an empty reply: $(printf '%s' "$reply" | head -c 300)"
say "the model answered: $text"
echo
echo "OK: Ferrite is serving http://${FERRITE_HOST:-127.0.0.1}:$port/v1 (model id '${FERRITE_MODEL_ID:-qwen3.8-27b}')"
echo "    log: $root/logs/server.log    stop: kill -- -\$(cat $root/logs/server.pid)    docs: $root/docs/quickstart.md"
