#!/usr/bin/env bash
# Download and verify the DeepSeek-V4.1-Flash Coder files Ferrite serves (docs/deepseek-coder.md), into one folder:
#
#   scripts/fetch-deepseek.sh [--verify-only] [--dir DIR]
#
#   DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-0000[1-5]-of-00006.gguf
#                         ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF, IQ2_XXS, as published (~72 GB, five
#                         files: the repository has no part 6)
#   dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf
#                         the DSpark draft model, unsloth/DeepSeek-V4.1-Flash-1017-GGUF (~8 GB)
#   DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf
#                         the Engram table, ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF (~111 GB; read a few rows a
#                         token from the file, so keep it on an SSD). Pinned to the revision before that repository
#                         moved to the llama.cpp tensor layout (2026-10-27): this engine reads the earlier layout
#   chat_template.jinja   from the Coder repository
#   tokenizer.json, tokenizer_config.json, generation_config.json
#                         written from the first GGUF by dsc-mkmodel (built here if it is not: cargo build --release -p fe-dsc)
#
# Every downloaded file is checked against the sizes and sha256 below. Downloads go through the Hugging Face CLI (`hf`)
# when it is installed, else through curl in DSC_FETCH_PARTS parallel ranges; both resume where they stopped and retry
# with a growing pause, so a dropped connection does not end the run. A complete file whose sha256 differs is moved
# aside (NAME.bad) and fetched again.
# Options:
#   --dir DIR       target folder [$FERRITE_MODELS/DeepSeek-V4.1-Flash-Coder, FERRITE_MODELS default /var/tmp/models]
#   --verify-only   only check what is there
#   -h, --help      this text
# Environment:
#   HF_TOKEN          sent to Hugging Face when set (none of these repos needs one)
#   DSC_FETCH         hf | curl: force one downloader [hf when the `hf` (or `huggingface-cli`) command exists]
#   DSC_FETCH_PARTS   parallel ranges per file with curl [8]
#   DSC_FETCH_TRIES   attempts per range (or per hf download) before giving up [10]
# Needs curl, sha256sum and python3 (standard library only). Space: ~191 GB (more while a .bad copy is kept).
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../tools/site.sh
. "$here/../tools/site.sh"
dir=$FERRITE_MODELS/DeepSeek-V4.1-Flash-Coder
mode=fetch
die() { echo "fetch-deepseek: $*" >&2; exit 1; }
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
CODER="ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF 54229d38d588a66d1c6cff08af7b281b9326c656"
DRAFT="unsloth/DeepSeek-V4.1-Flash-1017-GGUF dfcee0ec8d6b874c6663eca0fc55d90ab32534c8"
ENGRAM="ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF 653a9874aa38c356233c5f3cc8c96b19a43b57eb"
STEM=DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS
# repo revision path-in-repo local-name bytes sha256
files=(
  "$CODER IQ2_XXS/$STEM-00001-of-00006.gguf $STEM-00001-of-00006.gguf 5780704 108e63d27240da2d99846250727a7b282e05a7fdf2448435e81904b9a98d8a4a"
  "$CODER IQ2_XXS/$STEM-00002-of-00006.gguf $STEM-00002-of-00006.gguf 19776064960 75079999eb3a492ad695f3466fd6a068a1cf1920980eac1c7f51406363fadf7a"
  "$CODER IQ2_XXS/$STEM-00003-of-00006.gguf $STEM-00003-of-00006.gguf 19939989696 ea964e525922f36d3b984299c1a5bb8d896f50b9c2942364cd221c37863331f2"
  "$CODER IQ2_XXS/$STEM-00004-of-00006.gguf $STEM-00004-of-00006.gguf 19682533472 2d95806d186597cb0ea32635c92f0b66a9a6afae1a399ff74b13aebaa4ac7a52"
  "$CODER IQ2_XXS/$STEM-00005-of-00006.gguf $STEM-00005-of-00006.gguf 12719198624 3665b48c3bd46330780caf286ee40a087ac11499e5b7df8c943b098d10f426d4"
  "$CODER chat_template.jinja chat_template.jinja 18509 4d61d22c90ea0cdf15c7e298f7215f23eeffd20f8eed8fb517fda5fb64693d99"
  "$DRAFT dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf 7976914240 6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a"
  "$ENGRAM DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf 110595290816 eb19f30d368d393784e1fc54fdfaffc427ebb2110bf3762fdf5a8fbb9a2d740a"
)

mkdir -p "$dir"
auth=()
[ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
bad=0
tries=${DSC_FETCH_TRIES:-10}
parts=${DSC_FETCH_PARTS:-8}
case $tries$parts in *[!0-9]*) die "DSC_FETCH_TRIES / DSC_FETCH_PARTS must be numbers" ;; esac
[ "$parts" -ge 1 ] || parts=1
fetcher=${DSC_FETCH:-}
hfcli=
if command -v hf >/dev/null; then hfcli=hf; elif command -v huggingface-cli >/dev/null; then hfcli=huggingface-cli; fi
[ -z "$fetcher" ] && { [ -n "$hfcli" ] && fetcher=hf || fetcher=curl; }
case $fetcher in
  hf) [ -n "$hfcli" ] || die "DSC_FETCH=hf but neither hf nor huggingface-cli is installed" ;;
  curl) ;;
  *) die "DSC_FETCH=$fetcher: hf or curl" ;;
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

# the serving files fe-server reads (tokenizer.json built from the GGUF's own vocabulary, with the think and DSML marks as
# single tokens; generation_config: greedy default, the eos id)
if [ $mode = fetch ] && [ $bad = 0 ]; then
  mk=
  for c in "$here/../bin/dsc-mkmodel" "$here/../target/release/dsc-mkmodel"; do [ -x "$c" ] && { mk=$c; break; }; done
  if [ -z "$mk" ]; then
    command -v cargo >/dev/null || die "dsc-mkmodel is not built and cargo is not installed (cargo build --release -p fe-dsc)"
    echo "building dsc-mkmodel (cargo build --release -p fe-dsc --bin dsc-mkmodel)"
    (cd "$here/.." && cargo build --release -p fe-dsc --bin dsc-mkmodel >/dev/null) || die "the dsc-mkmodel build failed"
    mk=$here/../target/release/dsc-mkmodel
  fi
  cp "$dir/chat_template.jinja" "$dir/.chat_template.src"
  DSC_MODEL="$dir/$STEM-00001-of-00006.gguf" DSC_ENGRAM="$dir/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf" "$mk" "$dir" "$dir/.chat_template.src" \
    || die "dsc-mkmodel failed"
  rm -f "$dir/.chat_template.src"
fi
for f in tokenizer.json tokenizer_config.json generation_config.json; do
  if [ -f "$dir/$f" ]; then echo "ok       $f"; else echo "missing  $f"; bad=1; fi
done
rm -rf "$dir/.hf-download"
[ $bad = 0 ] || die "some files are missing or differ (see above)"
echo "DeepSeek-V4.1-Flash Coder files ready in $dir"
echo "start it with: DSC_DIR=$dir tools/serve-dsc.sh"
