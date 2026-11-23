#!/usr/bin/env bash
# Download and verify the Inkling-2-Mini files Ferrite serves (docs/inkling-2-mini.md), into one folder:
#
#   scripts/fetch-inkling.sh [--verify-only] [--dir DIR]
#
# From thinkingmachines/Inkling-2-Mini (Apache-2.0), pinned to revision ed40e7da625f594fc9c4990f5f66be24873c22d7, the one Ferrite was tested with:
#   model-00001-of-00001.safetensors   the model, NVFP4 / MXFP8 experts and FP8 attention as published (~19.5 GB)
#   mtp.safetensors                    the draft head (~0.6 GB)
#   config.json, generation_config.json, chat_template.jinja, tokenizer.json, tokenizer_config.json, special_tokens_map.json
# The draft head's fold tables are derived from two of these tensors on the first start (a minute of CPU time) and kept as fold.safetensors beside them.
#
# Every downloaded file is checked against the sizes and sha256 below. Downloads go through the Hugging Face CLI (`hf`)
# when it is installed, else through curl in INK_FETCH_PARTS parallel ranges; both resume where they stopped and retry
# with a growing pause, so a dropped connection does not end the run. A complete file whose sha256 differs is moved
# aside (NAME.bad) and fetched again.
# Options:
#   --dir DIR       target folder [$FERRITE_MODELS/Inkling-2-Mini, FERRITE_MODELS default /var/tmp/models]
#   --verify-only   only check what is there
#   -h, --help      this text
# Environment:
#   HF_TOKEN          sent to Hugging Face when set (the repository needs none)
#   INK_FETCH         hf | curl: force one downloader [hf when the `hf` (or `huggingface-cli`) command exists]
#   INK_FETCH_PARTS   parallel ranges per file with curl [8]
#   INK_FETCH_TRIES   attempts per range (or per hf download) before giving up [10]
# Needs curl, sha256sum and python3 (standard library only). Space: ~20 GB.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../tools/site.sh
. "$here/../tools/site.sh"
dir=$FERRITE_MODELS/Inkling-2-Mini
mode=fetch
die() { echo "fetch-inkling: $*" >&2; exit 1; }
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }
while [ $# -gt 0 ]; do
  case $1 in
    --dir) [ $# -ge 2 ] || die "--dir needs a folder"; dir=$2; shift ;;
    --verify-only) mode=verify ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument '$1' (see --help)" ;;
  esac
  shift
done
for c in curl sha256sum python3; do command -v $c >/dev/null || die "$c is not installed"; done

HF=https://huggingface.co
INK="thinkingmachines/Inkling-2-Mini ed40e7da625f594fc9c4990f5f66be24873c22d7"
# repo revision path-in-repo local-name bytes sha256
files=(
  "$INK model-00001-of-00001.safetensors model-00001-of-00001.safetensors 19526019168 65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1"
  "$INK mtp.safetensors mtp.safetensors 635801416 7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2"
  "$INK tokenizer.json tokenizer.json 26178678 7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6"
  "$INK tokenizer_config.json tokenizer_config.json 13235 bf09342d285139fd37ce10c70878af1d52cb390e9901e10248654293a80ee636"
  "$INK special_tokens_map.json special_tokens_map.json 529 c16f44b3594563d82743ddb0417a2be1bfe3e2000dfad2da4b8f32554593e1a0"
  "$INK chat_template.jinja chat_template.jinja 6868 53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc"
  "$INK config.json config.json 2722 a7a349601f0984c4abf999c480ef775d8567d21f5f84b4e8a3e035b44f288dc2"
  "$INK generation_config.json generation_config.json 98 a08737c07a65a8fdf69f7cf086576210e5aad4b241437090b26d7b0c4f7905f7"
)

mkdir -p "$dir"
auth=()
[ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
bad=0
tries=${INK_FETCH_TRIES:-10}
parts=${INK_FETCH_PARTS:-8}
case $tries$parts in *[!0-9]*) die "INK_FETCH_TRIES / INK_FETCH_PARTS must be numbers" ;; esac
[ "$parts" -ge 1 ] || parts=1
fetcher=${INK_FETCH:-}
hfcli=
if command -v hf >/dev/null; then hfcli=hf; elif command -v huggingface-cli >/dev/null; then hfcli=huggingface-cli; fi
[ -z "$fetcher" ] && { [ -n "$hfcli" ] && fetcher=hf || fetcher=curl; }
case $fetcher in
  hf) [ -n "$hfcli" ] || die "INK_FETCH=hf but neither hf nor huggingface-cli is installed" ;;
  curl) ;;
  *) die "INK_FETCH=$fetcher: hf or curl" ;;
esac
pause() { local s=$((5 * 2 ** ($1 - 1))); [ $s -gt 120 ] && s=120; echo "  (attempt $1 failed, retrying in ${s}s)" >&2; sleep $s; }
size() { stat -c %s "$1" 2>/dev/null || echo 0; }
# one byte range [a, b] of url into part file f, appended from where it stopped, with retries
fetch_range() {  # url a b f
  local url=$1 a=$2 b=$3 f=$4 n=$(($3 - $2 + 1)) i have
  for i in $(seq 1 "$tries"); do
    have=$(size "$f")
    [ "$have" -ge "$n" ] && return 0
    curl -fsSL --http1.1 --connect-timeout 30 --speed-limit 100000 --speed-time 60 "${auth[@]}" \
      -r "$((a + have))-$b" "$url" >> "$f" && [ "$(size "$f")" -ge "$n" ] && return 0
    [ "$(size "$f")" -gt "$n" ] && { echo "  range $a-$b: server sent more than asked, starting it over" >&2; : > "$f"; }
    [ "$i" -lt "$tries" ] && pause "$i"
  done
  return 1
}
# curl: the file in $parts ranges at once (one range for small files), then joined
fetch_curl() {  # url name bytes
  local url=$1 name=$2 bytes=$3 out=$dir/$2 np=$parts k a b step ok=1 pids=() pf=() alive p got
  [ "$bytes" -lt $((256 << 20)) ] && np=1
  # a single-stream .part from an earlier run: carry it on as one range
  if [ -f "$out.part" ] && [ ! -f "$out.part.0" ]; then np=1; mv "$out.part" "$out.part.0"; fi
  # ranges of an earlier run (every range file is created when its curl starts): keep that run's split
  if [ -f "$out.part.0" ]; then np=0; while [ -f "$out.part.$np" ]; do np=$((np + 1)); done; fi
  step=$(((bytes + np - 1) / np))
  for k in $(seq 0 $((np - 1))); do
    a=$((k * step)); b=$(((k + 1) * step - 1)); [ $b -ge "$bytes" ] && b=$((bytes - 1))
    pf+=("$out.part.$k")
    fetch_range "$url" $a $b "$out.part.$k" & pids+=($!)
  done
  while [ $np -gt 1 ]; do  # progress while the ranges run
    alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ $alive = 1 ] || break
    sleep 15
    got=0; for p in "${pf[@]}"; do got=$((got + $(size "$p"))); done
    printf "  %s: %d of %d MB\n" "$name" $((got / 1000000)) $((bytes / 1000000))
  done
  for p in "${pids[@]}"; do wait "$p" || ok=0; done
  [ $ok = 1 ] || return 1
  cat "${pf[@]}" > "$out.joined" && mv "$out.joined" "$out" && rm -f "${pf[@]}"
}
# hf CLI: into a staging folder in $dir (it resumes there), then moved into place
fetch_hf() {  # repo rev path name
  local i stage=$dir/.hf-download
  for i in $(seq 1 "$tries"); do
    if $hfcli download "$1" "$3" --revision "$2" --local-dir "$stage" >/dev/null; then
      mv "$stage/$3" "$dir/$4" && return 0
    fi
    [ "$i" -lt "$tries" ] && pause "$i"
  done
  return 1
}
fetch() {  # repo rev path name bytes
  echo "download $4 ($(($5 / 1000000)) MB) from $1 (with $fetcher)"
  if [ $fetcher = hf ]; then fetch_hf "$1" "$2" "$3" "$4"; else fetch_curl "$HF/$1/resolve/$2/$3" "$4" "$5"; fi \
    || die "download of $4 failed after $tries attempts (rerun to resume)"
}
check() {  # name bytes sha
  local f=$dir/$1
  [ -f "$f" ] || { echo "missing  $1"; return 1; }
  [ "$(stat -c %s "$f")" = "$2" ] || { echo "SIZE     $1: $(stat -c %s "$f") bytes, expected $2"; return 1; }
  [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$3" ] || { echo "SHA256   $1 differs"; return 2; }
  echo "ok       $1"
}
for line in "${files[@]}"; do
  read -r repo rev path name bytes sha <<<"$line"
  if [ $mode = fetch ] && ! { [ -f "$dir/$name" ] && [ "$(stat -c %s "$dir/$name")" = "$bytes" ]; }; then
    [ -f "$dir/$name" ] && mv "$dir/$name" "$dir/$name.bad"  # a wrong-size file: not resumable, set aside
    fetch "$repo" "$rev" "$path" "$name" "$bytes"
  fi
  rc=0; check "$name" "$bytes" "$sha" || rc=$?
  if [ $rc = 2 ] && [ $mode = fetch ]; then
    # right size, wrong content: keep it as .bad (replacing an older .bad) and download it again
    echo "         moving it to $name.bad and downloading it again"
    mv -f "$dir/$name" "$dir/$name.bad"
    fetch "$repo" "$rev" "$path" "$name" "$bytes"
    rc=0; check "$name" "$bytes" "$sha" || rc=$?
    [ $rc = 0 ] && rm -f "$dir/$name.bad"
  elif [ $rc = 2 ]; then
    echo "         run without --verify-only to download it again"
  fi
  [ $rc = 0 ] || bad=1
done

rm -rf "$dir/.hf-download"
[ $bad = 0 ] || die "some files are missing or differ (see above)"
echo "Inkling-2-Mini files ready in $dir"
echo "start it with: INK_DIR=$dir tools/serve-ink.sh"
