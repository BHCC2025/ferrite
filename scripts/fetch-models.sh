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
# Name roles to fetch only some of them (default: all three; an RTX 5080 / 5060 Ti needs `quant prompt`).
# The Intel Arc Pro B70 engine (docs/arc-b70.md) uses a fourth folder instead of the first two, fetched only by name
# (`scripts/fetch-models.sh xq draft`):
#   Qwen3.8-27B-XQ-5.0bpw-mix     xq      XQ 5.0 bpw mix: the weights of the Arc engine and its MTP draft head (~30 GB;
#                                         FERRITE_XQ_DIR places it)
#
# What a rerun does: it downloads what is missing, retries a failed transfer (3 attempts), and after the check
# fetches again any file whose size or sha256 is wrong (the bad copy is deleted first). The Hugging Face CLI does not
# continue a cut-off file, so an interrupted file starts over; its leftover partial files are deleted once the
# folder verifies. It stops before downloading when the disk has too little room (FERRITE_FORCE=1 tries anyway).
#
# Options:
#   --verify-only   check the folders against scripts/models.manifest without downloading
#   --quick         verify file sizes only (default: sizes and sha256, a few minutes for ~38 GB)
#   --essential     with --verify-only: only the files the server loads (not README.md / sha256sums.txt); the
#                   launchers run `--verify-only --quick --essential` before they load the model
#   --no-verify     download without verifying
#   --make-manifest print manifest lines for the folders as they are on disk (after uploading a new quant)
#   -h, --help      this text
# Environment:
#   FERRITE_MODELS      target folder (default /var/tmp/models), or FERRITE_EXL3_DIR / NVFP4_DIR / DRAFT_DIR / XQ_DIR
#   FERRITE_QUANT_REPO  Hugging Face repo of the quant (default from the manifest: BHCC2025/Qwen3.8-27B-EXL3-3.0bpw-mix)
#   FERRITE_HF          the Hugging Face CLI to use [~/.venvs/hf/bin/hf, else hf on PATH]
#   HF_TOKEN            passed through to the Hugging Face CLI (only needed for gated or private repos)
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
manifest=$here/models.manifest
# shellcheck source=../tools/site.sh
. "$here/../tools/site.sh"
models=$FERRITE_MODELS
mode=fetch
quick=0
essential=0
roles=()

die() { echo "fetch-models: $*" >&2; exit 1; }
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case $1 in
    --verify-only) mode=verify ;;
    --quick) quick=1 ;;
    --essential) essential=1 ;;
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
# files of a folder in the manifest: name size sha (--essential: without the documentation files)
files_of() {
  awk -v d="$1" -v e=$essential '$1 == d && !(e && ($2 ~ /\.(md|txt|png)$/)) { print $2, $3, $4 }' "$manifest"
}

if [ $mode = manifest ]; then
  for role in "${roles[@]}"; do
    read -r folder _ _ <<< "$(repo_line "$role")"
    dir=$(dir_of "$role")
    [ -d "$dir" ] || die "no folder $dir"
    find "$dir" -maxdepth 1 -type f ! -name '.*.tsv' ! -name '.tree.json' ! -name '.quant_summary.txt' -printf '%P\n' \
      | sort | while read -r f; do
          printf '%s %s %s %s\n' "$folder" "$f" "$(stat -L -c %s "$dir/$f")" "$(sha256sum "$dir/$f" | cut -c1-64)"
        done
  done
  exit 0
fi

hf_cli() { # FERRITE_HF, the virtual environment docs/quickstart.md sets up, then PATH, then pip --user's folder
  local c
  for c in "${FERRITE_HF:-}" "$HOME/.venvs/hf/bin/hf" hf huggingface-cli "$HOME/.local/bin/hf"; do
    [ -n "$c" ] || continue
    if command -v "$c" > /dev/null; then command -v "$c"; return 0; fi
  done
  return 1
}

BAD=()
verify() { # folder dir -> 0 when every manifest file of the folder is in dir with the right size (and hash); BAD lists the rest
  local folder=$1 dir=$2 bad=0 n=0 f size sha got
  BAD=()
  while read -r f size sha; do
    n=$((n + 1))
    if [ ! -e "$dir/$f" ]; then
      echo "  missing  $folder/$f"; bad=1; BAD+=("$f"); continue
    fi
    got=$(stat -L -c %s "$dir/$f")
    if [ "$got" != "$size" ]; then
      echo "  size     $folder/$f: $got bytes, expected $size"; bad=1; BAD+=("$f"); continue
    fi
    if [ $quick = 0 ]; then
      got=$(sha256sum "$dir/$f" | cut -c1-64)
      if [ "$got" != "$sha" ]; then
        echo "  sha256   $folder/$f: $got, expected $sha"; bad=1; BAD+=("$f")
      fi
    fi
  done < <(files_of "$folder")
  [ $n -gt 0 ] || { echo "  no manifest entries for $folder"; return 1; }
  return $bad
}

# bytes still to fetch for a folder: the manifest files not on disk at the right size
missing_bytes() {
  local folder=$1 dir=$2 f size sha t=0
  while read -r f size sha; do
    [ -e "$dir/$f" ] && [ "$(stat -L -c %s "$dir/$f")" = "$size" ] || t=$((t + size))
  done < <(files_of "$folder")
  echo $t
}

# leftovers of the CLI: partial files it will not continue (Xet backend) and the metadata of deleted files
cleanup_partials() {
  local dir=$1 n
  [ -d "$dir/.cache/huggingface/download" ] || return 0
  n=$(find "$dir/.cache/huggingface/download" -name '*.incomplete' -printf '%s\n' 2> /dev/null | awk '{ t += $1 } END { printf "%d", t / 1048576 }')
  find "$dir/.cache/huggingface/download" -name '*.incomplete' -delete 2> /dev/null || true
  [ "${n:-0}" -gt 0 ] && echo "  deleted ${n} MB of partial files left by an interrupted download" || true
}

download() { # repo rev dir: up to 3 attempts
  local repo=$1 rev=$2 dir=$3 a
  for a in 1 2 3; do
    "$cli" download "$repo" --revision "$rev" --local-dir "$dir" && return 0
    [ $a = 3 ] || { echo "fetch-models: the download stopped (attempt $a of 3); trying again in $((a * 10)) s" >&2; sleep $((a * 10)); }
  done
  return 1
}

trap 'echo; echo "fetch-models: interrupted; run the same command again to continue (finished files are kept)" >&2; exit 130' INT TERM

if [ $mode != verify ]; then
  mkdir -p "$models"
  cli=$(hf_cli) || die "the Hugging Face CLI (hf) is not installed: python3 -m venv ~/.venvs/hf && ~/.venvs/hf/bin/pip install -U huggingface_hub"
  echo "fetch-models: using $cli (hf $("$cli" version 2> /dev/null | sed "s/\x1b\[[0-9;]*m//g" | grep -o "[0-9][0-9.]*" | head -n 1))"
  need=0
  for role in "${roles[@]}"; do
    read -r folder _ _ <<< "$(repo_line "$role")"
    [ -n "$folder" ] && need=$((need + $(missing_bytes "$folder" "$(dir_of "$role")")))
  done
  need_gb=$(( (need + 1073741823) / 1073741824 + 1 ))
  probe=$models; for role in "${roles[@]}"; do probe=$(dir_of "$role"); break; done
  mkdir -p "$probe"
  avail=$(df -BG --output=avail "$probe" | tail -1 | tr -dc 0-9)
  if [ "$need" -gt 0 ] && [ "$avail" -lt "$need_gb" ]; then
    if [ "${FERRITE_FORCE:-0}" = 1 ]; then echo "fetch-models: warning: $avail GB free under $probe, about $need_gb GB needed; FERRITE_FORCE=1: trying anyway" >&2
    else die "$avail GB free under $probe, about $need_gb GB still to download: free some space, or set FERRITE_MODELS to a folder on a larger disk (FERRITE_FORCE=1 tries anyway)"
    fi
  fi
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
    download "$repo" "$rev" "$dir" || die "download of $repo failed 3 times (network, or HF_TOKEN for a gated repo?); run it again later"
  fi
  if [ $mode != fetch-only ]; then
    echo "== verify $dir ($( [ $quick = 1 ] && echo sizes || echo 'sizes + sha256'))"
    if verify "$folder" "$dir"; then
      echo "  ok"
    elif [ $mode = fetch ]; then
      # a wrong copy (cut off, or the right size with other bytes): delete it and its CLI metadata, fetch it again
      echo "  fetching the ${#BAD[@]} file(s) above again"
      for f in "${BAD[@]}"; do rm -f "$dir/$f" "$dir/.cache/huggingface/download/$f.metadata" "$dir/.cache/huggingface/download/$f.lock"; done
      download "$repo" "$rev" "$dir" || die "download of $repo failed 3 times; run it again later"
      if verify "$folder" "$dir"; then echo "  ok"; else status=1; fi
    else
      status=1
    fi
    [ $status = 0 ] && [ $mode = fetch ] && cleanup_partials "$dir"
  fi
done
trap - INT TERM
if [ $status != 0 ]; then
  [ $mode = verify ] && die "verification failed; run it without --verify-only to fetch the missing or wrong files again"
  die "verification failed after fetching again (above); report it with the lines above"
fi
[ $mode = verify ] && { echo "fetch-models: verified."; exit 0; }

# the launcher for this machine's GPU, with the folder settings that differ from the defaults
set_vars="FERRITE_MODELS=$models"
for v in FERRITE_EXL3_DIR FERRITE_NVFP4_DIR FERRITE_DRAFT_DIR FERRITE_XQ_DIR; do
  case ${!v} in "$models"/*) ;; *) set_vars="$set_vars $v=${!v}" ;; esac
done
launcher=scripts/serve.sh
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | head -n 1 | tr -d ' ' || true)
case " ${roles[*]} " in
  *" xq "*) launcher=scripts/serve-xe.sh ;;
  *) case $cap in 12.0) launcher=tools/serve-5080.sh ;; 12.1) launcher=scripts/serve.sh ;;
       *) for d in /sys/class/drm/card*/device/driver; do
            [ "$(basename "$(readlink -f "$d" 2> /dev/null)")" = xe ] && launcher=scripts/serve-xe.sh
          done ;; esac ;;
esac
echo "fetch-models: done. Start a server with: $set_vars $launcher"
