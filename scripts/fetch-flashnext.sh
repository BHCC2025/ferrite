#!/usr/bin/env bash
# Download and verify the Qwen3.8-Flash-Next files Ferrite serves (docs/flash-next.md), into one folder:
#
#   scripts/fetch-flashnext.sh [--verify-only] [--no-draft] [--dir DIR]
#
#   Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf  } ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF, IQ1_M,
#   Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf  } as published (~58.4 GB, not modified)
#   Qwen3.8-Flash-Next-MTP.gguf      the MTP draft head (~1.5 GB, $FNEXT_DRAFT_REPO); without it the model runs
#                                    without speculative decoding (about half the decode speed)
#   tokenizer.json, tokenizer_config.json, generation_config.json   Qwen/Qwen3.8-Flash-Next
#   chat_template.jinja              taken from the first GGUF's own chat template (the one the quant was made with)
#
# Every file is checked against the sizes and sha256 below. Downloads go through the Hugging Face CLI (`hf`) when it is
# installed, else through curl in FNEXT_FETCH_PARTS parallel ranges; both resume where they stopped and retry with a
# growing pause, so a dropped connection does not end the run. A complete file whose sha256 differs is moved aside
# (NAME.bad) and fetched again.
# Options:
#   --dir DIR       target folder [$FERRITE_MODELS/Qwen3.8-Flash-Next, FERRITE_MODELS default /var/tmp/models]
#   --verify-only   only check what is there
#   --no-draft      skip the MTP draft file
#   -h, --help      this text
# Environment:
#   FNEXT_DRAFT_REPO  Hugging Face repo of the draft [BHCC2025/Qwen3.8-Flash-Next-MTP-GGUF]
#   FNEXT_DRAFT_REV   its revision [the one this release was tested with]
#   HF_TOKEN          sent to Hugging Face when set (none of these repos needs one)
#   FNEXT_FETCH       hf | curl: force one downloader [hf when the `hf` (or `huggingface-cli`) command exists]
#   FNEXT_FETCH_PARTS parallel ranges per file with curl [8]
#   FNEXT_FETCH_TRIES attempts per range (or per hf download) before giving up [10]
# Needs curl, sha256sum and python3 (standard library only). Space: ~60 GB (~90 GB while a .bad copy is kept).
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../tools/site.sh
. "$here/../tools/site.sh"
dir=$FERRITE_MODELS/Qwen3.8-Flash-Next
mode=fetch
draft=1
die() { echo "fetch-flashnext: $*" >&2; exit 1; }
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }
while [ $# -gt 0 ]; do
  case $1 in
    --dir) [ $# -ge 2 ] || die "--dir needs a folder"; dir=$2; shift ;;
    --verify-only) mode=verify ;;
    --no-draft) draft=0 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument '$1' (see --help)" ;;
  esac
  shift
done
for c in curl sha256sum python3; do command -v $c >/dev/null || die "$c is not installed"; done

HF=https://huggingface.co
ISTA="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF bba88856f8c3c0c9e40e45f82d05e228b814ac43"
QWEN="Qwen/Qwen3.8-Flash-Next de4b8e4d43b917e7706784d8bb445c9af86a3540"
DRAFT="${FNEXT_DRAFT_REPO:-BHCC2025/Qwen3.8-Flash-Next-MTP-GGUF} ${FNEXT_DRAFT_REV:-e441b708cce202623551d79451cf765203865ab9}"
STEM=Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M
# repo revision path-in-repo local-name bytes sha256
files=(
  "$ISTA IQ1_M/$STEM-00001-of-00002.gguf $STEM-00001-of-00002.gguf 29608446496 bfa3644db4adfff072fc946c20994f2909d2cf89879d83f80a4c811d4016c211"
  "$ISTA IQ1_M/$STEM-00002-of-00002.gguf $STEM-00002-of-00002.gguf 28800138432 316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"
  "$QWEN tokenizer.json tokenizer.json 12809320 0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
  "$QWEN tokenizer_config.json tokenizer_config.json 17928 b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"
  "$QWEN generation_config.json generation_config.json 202 e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
)
[ $draft = 1 ] && files+=("$DRAFT Qwen3.8-Flash-Next-MTP.gguf Qwen3.8-Flash-Next-MTP.gguf 1517334816 54ba284d7538c534e3d8e6231c1a8fa46715d27e56f0f31af66e2ac39ee7914b")
TEMPLATE_SHA=12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce

mkdir -p "$dir"
auth=()
[ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
bad=0
tries=${FNEXT_FETCH_TRIES:-10}
parts=${FNEXT_FETCH_PARTS:-8}
case $tries$parts in *[!0-9]*) die "FNEXT_FETCH_TRIES / FNEXT_FETCH_PARTS must be numbers" ;; esac
[ "$parts" -ge 1 ] || parts=1
fetcher=${FNEXT_FETCH:-}
hfcli=
if command -v hf >/dev/null; then hfcli=hf; elif command -v huggingface-cli >/dev/null; then hfcli=huggingface-cli; fi
[ -z "$fetcher" ] && { [ -n "$hfcli" ] && fetcher=hf || fetcher=curl; }
case $fetcher in
  hf) [ -n "$hfcli" ] || die "FNEXT_FETCH=hf but neither hf nor huggingface-cli is installed" ;;
  curl) ;;
  *) die "FNEXT_FETCH=$fetcher: hf or curl" ;;
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

# the chat template the quant carries (tokenizer.chat_template of the first GGUF)
if [ $mode = fetch ] && [ -f "$dir/$STEM-00001-of-00002.gguf" ]; then
  python3 -I - "$dir/$STEM-00001-of-00002.gguf" "$dir/chat_template.jinja" <<'PY'
import struct, sys
src, dst = sys.argv[1:3]
SC = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
with open(src, "rb") as f:
    u = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
    assert f.read(4) == b"GGUF"; u("I"); u("Q"); nkv = u("Q")
    def skip(t):
        if t in SC: f.seek(SC[t], 1)
        elif t == 8: f.seek(u("Q"), 1)
        elif t == 9:
            et, n = u("I"), u("Q")
            if et in SC: f.seek(SC[et] * n, 1)
            else:
                for _ in range(n): skip(et)
        else: raise SystemExit(f"unknown GGUF type {t}")
    for _ in range(nkv):
        k = f.read(u("Q")); t = u("I")
        if k == b"tokenizer.chat_template" and t == 8:
            open(dst, "wb").write(f.read(u("Q"))); break
        skip(t)
    else:
        raise SystemExit("no tokenizer.chat_template in " + src)
PY
fi
if [ -f "$dir/chat_template.jinja" ] && [ "$(sha256sum "$dir/chat_template.jinja" | cut -d' ' -f1)" = $TEMPLATE_SHA ]; then
  echo "ok       chat_template.jinja"
else
  echo "missing  chat_template.jinja (or it differs)"; bad=1
fi
rm -rf "$dir/.hf-download"
[ $bad = 0 ] || die "some files are missing or differ (see above)"
echo "Flash-Next files ready in $dir"
[ $draft = 1 ] || echo "(no MTP draft: the server runs without speculative decoding)"
echo "start it with: FNEXT_DIR=$dir tools/serve-fnext.sh"
