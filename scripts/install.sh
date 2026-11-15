#!/usr/bin/env bash
# Install Ferrite on a DGX Spark and start it: one command from a fresh machine to a running server.
#
#   curl -fsSL https://raw.githubusercontent.com/BHCC2025/ferrite/main/scripts/install.sh | bash
#   scripts/install.sh [options]        (inside a checkout or an unpacked release: installs and starts that one)
#
# Each step says what it does and is skipped when it is already done, so running it again continues where it stopped.
#   1. checks: aarch64, a GB10 GPU (compute capability 12.1) with the NVIDIA driver, enough free memory, the port
#   2. Ferrite: this checkout or release folder; otherwise, in $FERRITE_HOME, the latest published release (the
#      GitHub API names it; FERRITE_VERSION or --version pins one): its tarball with ready-made binaries, or when the
#      release has none (or --source) a clone of its tag and a source build (rustup, the pinned toolchains through
#      scripts/check-deps.sh --build --install, cargo build). It says which of the two it does, and why.
#      An older release already in $FERRITE_HOME is upgraded: the old folder is kept as $FERRITE_HOME-<old version>
#      (a source checkout is moved to the new tag when it has no local changes, else it says why not).
#   3. the Hugging Face CLI, in ~/.venvs/hf
#   4. the models: scripts/fetch-models.sh into $FERRITE_MODELS, skipped when they are already there and verified
#   5. the server: scripts/ferritectl.sh start (in the background, log <ferrite folder>/logs/server-<port>.log),
#      until it answers; stop it with <ferrite folder>/scripts/ferritectl.sh stop
#   6. a test request: OK, or what went wrong
# It needs no sudo: when a system package is missing it stops and prints the apt command to run first.
# The server listens on every interface (FERRITE_HOST=0.0.0.0, the default): other machines on your network can use
# it, and there is no API key. FERRITE_HOST=127.0.0.1 keeps it on this machine.
#
# Options:
#   --version V    install release V (e.g. 0.6.0) instead of the latest one
#   --source       build from source even when a release tarball exists
#   --no-serve     stop after the models (step 4)
#   -h, --help     this text
# Environment:
#   FERRITE_HOME         where it installs when not run from a checkout or release folder [~/ferrite]
#   FERRITE_MODELS       the model folder [/var/tmp/models]; the other settings of scripts/serve.sh (FERRITE_PORT,
#                        FERRITE_HOST, FERRITE_CTX, FERRITE_SLOTS, ...) apply to the server it starts
#   FERRITE_MIN_FREE_GB  memory that must be available before it starts the server [95]
#   FERRITE_VERSION      the release to install [the latest published release; 0.6.1 when GitHub cannot be asked]
#   FERRITE_RELEASE_URL  the release tarball [the GitHub release of that version]; its .sha256 is checked
#   FERRITE_REPO_URL     the repository to clone for a source build [https://github.com/BHCC2025/ferrite.git]
set -euo pipefail

REPO=BHCC2025/ferrite
DEFAULT_VERSION=0.6.1   # the release this installer comes with: used when GitHub cannot be asked for the latest one
REPO_URL=${FERRITE_REPO_URL:-https://github.com/$REPO.git}
VERSION=${FERRITE_VERSION:-}
source_build=0
serve=1
while [ $# -gt 0 ]; do
  case $1 in
    --source) source_build=1 ;;
    --no-serve) serve=0 ;;
    --version) VERSION=${2:-}; [ -n "$VERSION" ] || { echo "install: --version needs a version (e.g. 0.6.0)" >&2; exit 1; }; shift ;;
    --version=*) VERSION=${1#--version=} ;;
    -h|--help)
      if [ -f "${BASH_SOURCE[0]:-}" ]; then sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'
      else echo "usage: install.sh [--version V] [--source] [--no-serve] (see the comment at the top of scripts/install.sh)"; fi
      exit 0 ;;
    *) echo "install: unknown argument '$1' (--help)" >&2; exit 1 ;;
  esac
  shift
done
VERSION=${VERSION#v}

n=0
step() { n=$((n + 1)); echo; echo "==> [$n/6] $*"; }
say() { echo "    $*"; }
die() { echo; echo "install: ERROR: $*" >&2; exit 1; }
need_pkg() { die "$1 is missing. It needs a system package (the one step that needs sudo): sudo apt install -y $2, then run this again"; }
# a.b.c newer than x.y.z?
newer() { [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -n 1)" = "$1" ]; }

# ---------------------------------------------------------------------------------------------------- 1. checks
step "Checking this machine"
[ "$(uname -m)" = aarch64 ] || die "this installer is for DGX Spark (aarch64); this machine is $(uname -m). The RTX 50, Flash-Next and Arc B70 targets are built from source: see docs/rtx-5080.md, docs/flash-next.md, docs/arc-b70.md"
command -v nvidia-smi > /dev/null || die "nvidia-smi not found: the NVIDIA driver is missing (DGX OS ships it)"
read -r gpu cap drv < <(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null \
  | head -n 1 | awk -F', ' '{ gsub(/ /, "_", $1); print $1, $2, $3 }') || true
[ -n "${cap:-}" ] || die "nvidia-smi found no GPU"
[ "$cap" = 12.1 ] || die "the GPU is ${gpu//_/ } (compute capability $cap); this installer is for the GB10 of DGX Spark (12.1). RTX 50 cards: docs/rtx-5080.md"
say "GPU ${gpu//_/ } (compute capability $cap), driver $drv"
for t in curl python3; do command -v $t > /dev/null || need_pkg "$t" "$t"; done
port=${FERRITE_PORT:-8000}
running=0
if h=$(curl -sf -m 3 "http://127.0.0.1:$port/health" 2>/dev/null); then
  eng=$(printf '%s' "$h" | python3 -c 'import json, sys; print(json.load(sys.stdin).get("engine", ""))' 2>/dev/null || true)
  case $eng in
    ferrite) running=1 ;;
    *) die "port $port is taken by another server (its /health says: $(printf '%s' "$h" | head -c 160)). Stop it, or pick another port with FERRITE_PORT" ;;
  esac
elif curl -s -m 3 -o /dev/null "http://127.0.0.1:$port/" 2>/dev/null; then
  die "port $port is taken by another program (it answers HTTP but is not Ferrite). Stop it, or pick another port with FERRITE_PORT"
fi
avail=$(awk '/MemAvailable/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
min=${FERRITE_MIN_FREE_GB:-95}
if [ $running = 1 ]; then
  say "a Ferrite server already answers on port $port ($avail GB of memory available)"
elif [ $serve = 1 ] && [ "$avail" -lt "$min" ]; then
  die "only $avail GB of memory is available; the server needs about $min GB (the model, 4 x 128K of KV cache and the prompt cache). Stop other GPU jobs, or lower FERRITE_CTX / FERRITE_SLOTS and set FERRITE_MIN_FREE_GB to match"
elif [ $serve = 1 ]; then
  say "$avail GB of memory available (about $min GB needed)"
else
  say "$avail GB of memory available (the server will need about $min GB)"
fi

# ---------------------------------------------------------------------------------------------------- 2. Ferrite
step "Getting Ferrite"
# the release to install: FERRITE_VERSION / --version, else the latest published one
latest() {
  local v
  v=$(curl -fsSL -m 20 -H 'Accept: application/vnd.github+json' "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null \
      | python3 -c 'import json, sys; print(json.load(sys.stdin)["tag_name"])' 2>/dev/null) \
    || v=$(curl -fsSI -m 20 "https://github.com/$REPO/releases/latest" 2>/dev/null | tr -d '\r' \
      | sed -n 's|^[Ll]ocation: .*/releases/tag/\(.*\)$|\1|p' | tail -n 1)
  [ -n "$v" ] && echo "${v#v}"
}
pinned=1
if [ -z "$VERSION" ]; then
  pinned=0
  VERSION=$(latest) || VERSION=""
  if [ -z "$VERSION" ]; then
    VERSION=$DEFAULT_VERSION
    say "could not ask GitHub for the latest release (network, or its API limit); installing $VERSION, the release this installer comes with"
  fi
fi
[ -n "$VERSION" ] && say "release $VERSION ($([ $pinned = 1 ] && echo "pinned by FERRITE_VERSION / --version" || echo "the latest published release"))"
RELEASE_URL=${FERRITE_RELEASE_URL:-https://github.com/$REPO/releases/download/v$VERSION/ferrite-$VERSION-dgx-spark-aarch64.tar.gz}

# fetch: the release tarball into $1, else a clone of the release's tag (or main) for a source build
fetch() {
  local dest=$1 tmp code
  if [ $source_build = 0 ] && [ -n "$VERSION" ]; then
    say "downloading the release tarball: $RELEASE_URL"
    tmp=$(mktemp -d)
    code=$(curl -sSL -w '%{http_code}' -o "$tmp/r.tar.gz" "$RELEASE_URL" 2>/dev/null || true)
    if [ "$code" = 200 ] && curl -fsSL -o "$tmp/r.sha256" "$RELEASE_URL.sha256" 2>/dev/null; then
      [ "$(sha256sum < "$tmp/r.tar.gz" | cut -c1-64)" = "$(cut -c1-64 "$tmp/r.sha256")" ] \
        || die "the tarball's sha256 does not match $RELEASE_URL.sha256 (a broken download? run this again)"
      mkdir -p "$dest"
      tar xzf "$tmp/r.tar.gz" --strip-components=1 -C "$dest"
      rm -rf "$tmp"
      say "unpacked into $dest (sha256 checked): ready-made binaries, no build"
      return 0
    fi
    rm -rf "$tmp"
    say "release $VERSION has no tarball at that address (HTTP ${code:-error}): BUILDING FROM SOURCE instead (a few minutes; needs build-essential)"
  elif [ $source_build = 1 ]; then
    say "--source: building from source"
  fi
  command -v git > /dev/null || need_pkg git git
  if [ -n "$VERSION" ]; then
    say "cloning $REPO_URL at tag v$VERSION into $dest"
    git clone -q --branch "v$VERSION" "$REPO_URL" "$dest" 2>/dev/null || {
      say "no tag v$VERSION in $REPO_URL: cloning its main branch"; git clone -q "$REPO_URL" "$dest"; } \
      || die "git clone of $REPO_URL failed (network?)"
  else
    say "cloning $REPO_URL (main) into $dest"
    git clone -q "$REPO_URL" "$dest" || die "git clone of $REPO_URL failed (network?)"
  fi
}
installed_version() { # the version of the Ferrite folder $1
  if [ -f "$1/BUILD-INFO" ]; then sed -n '1s/^ferrite \([0-9][0-9.]*\).*/\1/p' "$1/BUILD-INFO"
  elif [ -f "$1/Cargo.toml" ]; then awk -F'"' '/^\[workspace.package\]/ { w = 1 } w && /^version/ { print $2; exit }' "$1/Cargo.toml"
  fi
}

root=""
if [ -f "${BASH_SOURCE[0]:-}" ]; then
  d=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  [ -f "$d/scripts/serve.sh" ] && [ -f "$d/tools/site.sh" ] && root=$d
fi
if [ -n "$root" ]; then
  say "using $root (Ferrite $(installed_version "$root"), the folder this script is in)"
else
  root=${FERRITE_HOME:-$HOME/ferrite}
  if [ -x "$root/bin/fe-server" ] || [ -f "$root/Cargo.toml" ]; then
    have=$(installed_version "$root")
    if [ -n "$VERSION" ] && [ -n "$have" ] && newer "$VERSION" "$have"; then
      say "Ferrite $have is installed in $root; $VERSION is available: upgrading"
      if [ $running = 1 ]; then
        die "a Ferrite server is running on port $port. Stop it first ($root/scripts/ferritectl.sh stop, or kill the pid in $root/logs/server.pid for an install older than 0.6.1), then run this again to upgrade"
      fi
      if [ -d "$root/.git" ] && [ -f "$root/Cargo.toml" ]; then
        if [ -n "$(git -C "$root" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
          die "$root is a git checkout with local changes; not moving it to v$VERSION. Commit or stash them (or move the folder away), then run this again"
        fi
        git -C "$root" fetch -q --tags || die "git fetch in $root failed (network?)"
        git -C "$root" checkout -q "v$VERSION" || die "no tag v$VERSION in $root's remote"
        say "checked out v$VERSION in $root (it is rebuilt below)"
      else
        old=$root-$have
        [ -e "$old" ] && old=$root-$have-$(date +%Y%m%d%H%M%S)
        mv "$root" "$old"
        say "the old install is kept in $old"
        fetch "$root"
        [ -d "$old/logs" ] && mkdir -p "$root/logs" && cp -a "$old/logs/." "$root/logs/" 2>/dev/null || true
      fi
    elif [ -n "$VERSION" ] && [ -n "$have" ] && newer "$have" "$VERSION"; then
      say "Ferrite $have is installed in $root, newer than $VERSION: keeping it (move it away to install $VERSION)"
    else
      say "using Ferrite ${have:-?} already in $root$([ -n "$VERSION" ] && [ "$have" = "$VERSION" ] && echo " (the current release)")"
    fi
  elif [ -e "$root" ] && [ -n "$(ls -A "$root" 2> /dev/null)" ]; then
    die "$root exists and is not a Ferrite folder; move it away or set FERRITE_HOME"
  else
    fetch "$root"
  fi
fi
cd "$root"
if [ -x bin/fe-server ] && [ ! -f Cargo.toml ]; then
  say "release binaries: bin/fe-server, bin/ferrite ($(bin/fe-server --version 2>/dev/null || sed -n 1p BUILD-INFO))"
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
FERRITE_MIN_FREE_GB=0 scripts/check-deps.sh "$( [ -x bin/fe-server ] && echo bin/fe-server || echo target/release/fe-server)" 2>&1 \
  | sed 's/^/    /' || die "the runtime checks failed (see above)"

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
    scripts/fetch-models.sh --verify-only | sed 's/^/    /' || die "the model files do not match the manifest (see above); run this again to fetch the wrong ones"
    echo "$mhash" > "$stamp"
  fi
else
  say "downloading about 38 GB (scripts/fetch-models.sh; a rerun fetches only what is missing or wrong)"
  scripts/fetch-models.sh | sed 's/^/    /' || die "the model download failed (see above); run this again to continue"
  echo "$mhash" > "$stamp"
fi
# releases before 0.6.1 have no scripts/ferritectl.sh: start scripts/serve.sh the way they did
if [ -x scripts/ferritectl.sh ]; then
  start_cmd="scripts/ferritectl.sh start spark"; stop_cmd="$root/scripts/ferritectl.sh stop"; log=$root/logs/server-$port.log
else
  start_cmd="scripts/serve.sh"; stop_cmd="kill -- -\$(cat $root/logs/server.pid)"; log=$root/logs/server.log
fi
if [ $serve = 0 ]; then
  echo; echo "install: done (--no-serve). Start the server with: cd $root && FERRITE_MODELS=$FERRITE_MODELS $start_cmd"
  exit 0
fi

# ---------------------------------------------------------------------------------------------------- 5. server
step "The server on port $port"
if [ $running = 1 ]; then
  say "already running"
elif [ -x scripts/ferritectl.sh ]; then
  say "FERRITE_MODELS=$FERRITE_MODELS $start_cmd  (in the background; log: $log)"
  scripts/ferritectl.sh start spark 2>&1 | sed 's/^/    /' || die "the server did not start (see above)"
else
  say "FERRITE_MODELS=$FERRITE_MODELS scripts/serve.sh  (in the background; log: $log)"
  mkdir -p logs
  # its own session and process group: the id in server.pid stops the group (kill -- -<id>)
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
  say "ready after $((SECONDS - t0)) s"
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
host=${FERRITE_HOST:-$([ -x scripts/ferritectl.sh ] && echo 0.0.0.0 || echo 127.0.0.1)}
echo
if [ "$host" = 0.0.0.0 ]; then
  echo "OK: Ferrite is serving http://$(hostname -s):$port/v1 (model id '${FERRITE_MODEL_ID:-qwen3.8-27b}'), and http://127.0.0.1:$port/v1 on this machine"
  echo "    It listens on every interface with no API key: use it on a trusted network or firewall port $port (FERRITE_HOST=127.0.0.1 keeps it local)."
else
  echo "OK: Ferrite is serving http://$host:$port/v1 (model id '${FERRITE_MODEL_ID:-qwen3.8-27b}')"
fi
echo "    log: $log    stop: $stop_cmd    docs: $root/docs/quickstart.md"
