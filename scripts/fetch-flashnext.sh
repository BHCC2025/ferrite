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
# Every file is checked against the sizes and sha256 below; a download resumes where it stopped.
# Options:
#   --dir DIR       target folder [$FERRITE_MODELS/Qwen3.8-Flash-Next, FERRITE_MODELS default /var/tmp/models]
#   --verify-only   only check what is there
#   --no-draft      skip the MTP draft file
#   -h, --help      this text
# Environment:
#   FNEXT_DRAFT_REPO  Hugging Face repo of the draft [BHCC2025/Qwen3.8-Flash-Next-MTP-GGUF]
#   FNEXT_DRAFT_REV   its revision [the one this release was tested with]
#   HF_TOKEN          sent to Hugging Face when set (none of these repos needs one)
# Needs curl, sha256sum and python3 (standard library only). Space: ~60 GB.
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
check() {  # name bytes sha
  local f=$dir/$1
  [ -f "$f" ] || { echo "missing  $1"; return 1; }
  [ "$(stat -c %s "$f")" = "$2" ] || { echo "SIZE     $1: $(stat -c %s "$f") bytes, expected $2"; return 1; }
  [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$3" ] || { echo "SHA256   $1 differs"; return 1; }
  echo "ok       $1"
}
for line in "${files[@]}"; do
  read -r repo rev path name bytes sha <<<"$line"
  if [ $mode = fetch ] && ! { [ -f "$dir/$name" ] && [ "$(stat -c %s "$dir/$name")" = "$bytes" ]; }; then
    echo "download $name ($((bytes / 1000000)) MB) from $repo"
    curl -fL --retry 5 --retry-delay 5 -C - "${auth[@]}" -o "$dir/$name.part" "$HF/$repo/resolve/$rev/$path" \
      || die "download of $name failed (rerun to resume)"
    mv "$dir/$name.part" "$dir/$name"
  fi
  check "$name" "$bytes" "$sha" || bad=1
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
[ $bad = 0 ] || die "some files are missing or differ (see above)"
echo "Flash-Next files ready in $dir"
[ $draft = 1 ] || echo "(no MTP draft: the server runs without speculative decoding)"
echo "start it with: FNEXT_DIR=$dir tools/serve-fnext.sh"
