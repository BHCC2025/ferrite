#!/usr/bin/env bash
# Download the model repositories Ferrite serves from into one folder and verify them.
#
#   scripts/fetch-models.sh [options] [quant|prompt|draft|xq ...]
#
# Folders created under $FERRITE_MODELS (default /var/tmp/models; FERRITE_EXL3_DIR / FERRITE_NVFP4_DIR /
# FERRITE_DRAFT_DIR place them one by one, see tools/site.sh), all three needed for the default setup:
#   Qwen3.8-27B-EXL3-3.0bpw-mix   quant   EXL3 3.0 bpw mix: the decode weights (~13 GB)
#   nvidia-Qwen3.8-27B-NVFP4      prompt  NVIDIA's NVFP4 checkpoint: prompt (prefill) GEMMs and the draft head (~21 GB)
#   z-lab-Qwen3.8-27B-DFlash2     draft   z-lab's DFlash2 block drafter for speculative decoding (~3.6 GB)
# Name roles to fetch only some of them (default: all three).
# The Intel Arc Pro B70 engine (docs/arc-b70.md) uses a fourth folder instead of the first two, fetched only by name
# (`scripts/fetch-models.sh xq draft`):
#   Qwen3.8-27B-XQ-5.0bpw-mix     xq      XQ 5.0 bpw mix: the weights of the Arc engine and its MTP draft head (~30 GB;
#                                         FERRITE_XQ_DIR places it)
#
# Options:
#   --verify-only   check the folders against scripts/models.manifest without downloading
#   --quick         verify file sizes only (default: sizes and sha256, a few minutes for ~38 GB)
#   --no-verify     download without verifying
#   --make-manifest print manifest lines for the folders as they are on disk (after uploading a new quant)
#   -h, --help      this text
# Environment:
#   FERRITE_MODELS      target folder (default /var/tmp/models), or FERRITE_EXL3_DIR / NVFP4_DIR / DRAFT_DIR
#   FERRITE_QUANT_REPO  Hugging Face repo of the quant (default from the manifest: BHCC2025/Qwen3.8-27B-EXL3-3.0bpw-mix)
#   HF_TOKEN            passed through to the Hugging Face CLI (only needed for gated or private repos)
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
manifest=$here/models.manifest
# shellcheck source=../tools/site.sh
. "$here/../tools/site.sh"
models=$FERRITE_MODELS
mode=fetch
quick=0
roles=()

die() { echo "fetch-models: $*" >&2; exit 1; }
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case $1 in
    --verify-only) mode=verify ;;
    --quick) quick=1 ;;
    --no-verify) mode=fetch-only ;;
    --make-manifest) mode=manifest ;;
    -h|--help) usage; exit 0 ;;
    quant|prompt|draft|xq) roles+=("$1") ;;
    *) die "unknown argument '$1' (see --help)" ;;
  esac
  shift
done
[ ${#roles[@]} -gt 0 ] || roles=(quant prompt draft)
[ -f "$manifest" ] || die "missing $manifest"

# repo lines: folder repo revision role; dir_of: where a role's folder lives (tools/site.sh)
repo_line() { awk -v r="$1" '$1 == "@" && $5 == r { print $2, $3, $4; exit }' "$manifest"; }
dir_of() {
  case $1 in quant) echo "$FERRITE_EXL3_DIR" ;; prompt) echo "$FERRITE_NVFP4_DIR" ;; draft) echo "$FERRITE_DRAFT_DIR" ;;
    xq) echo "$FERRITE_XQ_DIR" ;; esac
}

if [ $mode = manifest ]; then
  for role in "${roles[@]}"; do
    read -r folder _ _ <<< "$(repo_line "$role")"
    dir=$(dir_of "$role")
    [ -d "$dir" ] || die "no folder $dir"
    find "$dir" -maxdepth 1 -type f ! -name '.*.tsv' ! -name '.tree.json' ! -name '.quant_summary.txt' -printf '%P\n' \
      | sort | while read -r f; do
          printf '%s %s %s %s\n' "$folder" "$f" "$(stat -c %s "$dir/$f")" "$(sha256sum "$dir/$f" | cut -c1-64)"
        done
  done
  exit 0
fi

hf_cli() { # on PATH, else the virtual environment docs/quickstart.md sets up, else pip --user / pipx's folder
  local c
  for c in hf huggingface-cli "$HOME/.venvs/hf/bin/hf" "$HOME/.local/bin/hf"; do
    if command -v "$c" > /dev/null; then echo "$c"; return 0; fi
  done
  return 1
}

verify() { # folder dir -> 0 when every manifest file of the folder is in dir with the right size (and hash)
  local folder=$1 dir=$2 bad=0 n=0 f size sha got
  while read -r f size sha; do
    n=$((n + 1))
    if [ ! -f "$dir/$f" ]; then
      echo "  missing  $folder/$f"; bad=1; continue
    fi
    got=$(stat -c %s "$dir/$f")
    if [ "$got" != "$size" ]; then
      echo "  size     $folder/$f: $got bytes, expected $size"; bad=1; continue
    fi
    if [ $quick = 0 ]; then
      got=$(sha256sum "$dir/$f" | cut -c1-64)
      if [ "$got" != "$sha" ]; then
        echo "  sha256   $folder/$f: $got, expected $sha"; bad=1
      fi
    fi
  done < <(awk -v d="$folder" '$1 == d { print $2, $3, $4 }' "$manifest")
  [ $n -gt 0 ] || { echo "  no manifest entries for $folder"; return 1; }
  return $bad
}

mkdir -p "$models"
if [ $mode != verify ]; then
  cli=$(hf_cli) || die "the Hugging Face CLI (hf) is not installed: python3 -m venv ~/.venvs/hf && ~/.venvs/hf/bin/pip install -U huggingface_hub"
  need=0
  for role in "${roles[@]}"; do
    case $role in quant) need=$((need + 14)) ;; prompt) need=$((need + 22)) ;; draft) need=$((need + 4)) ;; xq) need=$((need + 31)) ;; esac
  done
  avail=$(df -BG --output=avail "$models" | tail -1 | tr -dc 0-9)
  [ "$avail" -ge "$need" ] || echo "fetch-models: warning: $avail GB free under $models, about $need GB needed" >&2
fi

status=0
for role in "${roles[@]}"; do
  read -r folder repo rev <<< "$(repo_line "$role")"
  [ -n "$folder" ] || die "no manifest line for role $role"
  dir=$(dir_of "$role")
  if [ "$role" = quant ] && [ -n "${FERRITE_QUANT_REPO:-}" ]; then
    repo=$FERRITE_QUANT_REPO
  fi
  if [ $mode != verify ]; then
    case $repo in
      *'<HF_USER>'*) die "the quant repository is not set: FERRITE_QUANT_REPO=<user>/Qwen3.8-27B-EXL3-3.0bpw-mix $0 ..." ;;
    esac
    echo "== $role: $repo @ $rev -> $dir"
    mkdir -p "$dir"
    "$cli" download "$repo" --revision "$rev" --local-dir "$dir" \
      || die "download of $repo failed (network, or HF_TOKEN for a gated repo?)"
  fi
  if [ $mode != fetch-only ]; then
    echo "== verify $dir ($( [ $quick = 1 ] && echo sizes || echo 'sizes + sha256'))"
    if verify "$folder" "$dir"; then echo "  ok"; else status=1; fi
  fi
done
if [ $status != 0 ]; then
  die "verification failed; re-run without --verify-only to fetch again (the CLI resumes partial files)"
fi
case " ${roles[*]} " in
  *" xq "*) echo "fetch-models: done. Start a server with: FERRITE_MODELS=$models scripts/serve-xe.sh" ;;
  *) echo "fetch-models: done. Start a server with: FERRITE_MODELS=$models scripts/serve.sh" ;;
esac
