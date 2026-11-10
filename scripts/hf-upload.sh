#!/usr/bin/env bash
# Stage the EXL3 quant (the decode weights) for Hugging Face, check it, and upload it only when asked. With --xq it
# stages the XQ quant of the Intel Arc Pro B70 engine instead ($FERRITE_XQ_DIR, docs/model-card-xq.md, the manifest's
# xq lines, and the MTP draft head dhead_q8.bin beside the weights).
#
#   scripts/hf-upload.sh [options]
#
# What goes up is an explicit allowlist, never a whole folder: from the quant folder ($FERRITE_EXL3_DIR) the weight
# shards named in model.safetensors.index.json, that index, the config / quantization / tokenizer files (ALLOW below)
# and LICENSE; docs/model-card.md as README.md; and sha256sums.txt (`sha256sum -c` format, the manifest of every other
# file). Everything else in the folder is listed as not uploaded. The staged set must agree with the quant's lines in
# scripts/models.manifest, so scripts/fetch-models.sh verifies a download against exactly what was uploaded. The text
# files and the safetensors headers are scanned for anything credential-like (tokens, keys) first.
#
# DRY RUN BY DEFAULT: it prints the exact file list with sizes, checks it and uploads nothing. A real upload needs all
# of: --upload, FERRITE_HF_REPO set (e.g. BHCC2025/Qwen3.8-27B-EXL3-3.0bpw-mix), the Hugging Face CLI ('hf', pip
# install -U huggingface_hub; looked up on PATH, then /usr/local/bin and ~/.local/bin) logged in with `hf auth login`,
# a model card without placeholders and a manifest that agrees. The token comes only from that standard login: this
# script never takes, reads, writes or prints a token, and it clears HF_TOKEN / HUGGING_FACE_HUB_TOKEN for the CLI.
#
# Options:
#   --xq               the XQ quant (Intel Arc Pro B70 engine) instead of the EXL3 quant
#   --upload           really upload to $FERRITE_HF_REPO (one commit; rerunning skips files already on the Hub)
#   --private          create the repository as private when it does not exist yet (default public)
#   --quick            compare sizes only (default: sizes and sha256, about a minute for 13 GB)
#   --update-manifest  rewrite the quant's file lines in scripts/models.manifest from the staged set (after editing
#                      docs/model-card.md, for example), then check again
#   --keep             keep the staging folder (hard links, no copies) after the run
#   -h, --help         this text
# Environment:
#   FERRITE_EXL3_DIR   the quant folder (tools/site.sh; default $FERRITE_MODELS/Qwen3.8-27B-EXL3-3.0bpw-mix)
#   FERRITE_HF_REPO    the target repository, required for --upload (the dry run shows the manifest's repository)
#   FERRITE_STATE_DIR  where the staging folder goes (tools/site.sh; default /var/tmp/ferrite)
set -euo pipefail

# the non-weight files of the quant that are published (the weight shards come from the index)
ALLOW=(config.json generation_config.json quantization_config.json model.safetensors.index.json tokenizer.json
       tokenizer_config.json vocab.json merges.txt chat_template.jinja preprocessor_config.json
       video_preprocessor_config.json LICENSE)

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/.." && pwd)
manifest=$here/models.manifest
card=$root/docs/model-card.md
# shellcheck source=../tools/site.sh
. "$root/tools/site.sh"
dir=$FERRITE_EXL3_DIR
role=quant
msg="Qwen3.8-27B EXL3 3.0 bpw mix"
upload=0
private=0
quick=0
update=0
keep=0

die() { echo "hf-upload: $*" >&2; exit 1; }
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case $1 in
    --xq) role=xq; dir=$FERRITE_XQ_DIR; card=$root/docs/model-card-xq.md; msg="Qwen3.8-27B XQ 5.0 bpw mix"
          ALLOW=(config.json generation_config.json xq_config.json quant_log.json model.safetensors.index.json
                 tokenizer.json tokenizer_config.json vocab.json merges.txt chat_template.jinja preprocessor_config.json
                 video_preprocessor_config.json LICENSE dhead_q8.bin) ;;
    --upload) upload=1 ;;
    --private) private=1 ;;
    --quick) quick=1 ;;
    --update-manifest) update=1 ;;
    --keep) keep=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument '$1' (see --help)" ;;
  esac
  shift
done
[ $upload = 1 ] && [ $quick = 1 ] && die "--upload checks the sha256 of every file; drop --quick"
[ $update = 1 ] && [ $quick = 1 ] && die "--update-manifest needs the sha256 of every file; drop --quick"
[ -f "$manifest" ] || die "missing $manifest"
[ -f "$card" ] || die "missing $card"
[ -d "$dir" ] || die "no quant folder $dir (set FERRITE_EXL3_DIR / FERRITE_XQ_DIR, or fetch it first)"
# the quant's manifest line: @ <folder> <repo> <revision> quant
read -r folder mrepo _ <<< "$(awk -v r="$role" '$1 == "@" && $5 == r { print $2, $3, $4; exit }' "$manifest")"
[ -n "$folder" ] || die "no $role line (@ ... $role) in $manifest"

# 1. the allowlist: the shards the index names, the ALLOW files, README.md from the model card
[ -f "$dir/model.safetensors.index.json" ] || die "no model.safetensors.index.json in $dir"
mapfile -t shards < <(python3 -c 'import json, sys; print("\n".join(sorted(set(json.load(open(sys.argv[1]))["weight_map"].values()))))' \
  "$dir/model.safetensors.index.json")
files=()
for f in "${shards[@]}" "${ALLOW[@]}"; do
  case $f in */*|.*) die "refusing a path or hidden name from the allowlist / index: '$f'" ;; esac
  [ -f "$dir/$f" ] || die "missing $dir/$f"
  files+=("$f")
done
stage=$FERRITE_STATE_DIR/hf-stage-$folder
rm -rf "$stage"; mkdir -p "$stage"
lines=$(mktemp)
cleanup() { rm -f "$lines"; [ $keep = 1 ] || rm -rf "$stage"; }
trap cleanup EXIT
linked=hard
for f in "${files[@]}"; do
  # hard links cost nothing and work for any uploader; another file system gets a symbolic link
  ln "$dir/$f" "$stage/$f" 2> /dev/null || { ln -s "$dir/$f" "$stage/$f"; linked=symbolic; }
done
cp "$card" "$stage/README.md"
names=("${files[@]}" README.md)

# 2. nothing credential-like: the text files whole, the safetensors by their JSON header
leak=$(python3 - "$stage" "${names[@]}" <<'EOF'
import json, re, struct, sys
pat = re.compile(rb"hf_[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|ssh-(rsa|ed25519|dss) AAAA|AKIA[0-9A-Z]{16}"
                 rb"|gh[pousr]_[A-Za-z0-9]{36}|xox[abprs]-[A-Za-z0-9-]{10,}|sk-[A-Za-z0-9]{32,}")
d = sys.argv[1]
for f in sys.argv[2:]:
    with open(f"{d}/{f}", "rb") as fh:
        if f.endswith(".safetensors"):
            n = struct.unpack("<Q", fh.read(8))[0]
            b = fh.read(n)
        elif f.endswith(".bin"):
            b = fh.read(4096)  # binary weights: the header only
        else:
            b = fh.read()
    m = pat.search(b)
    if m:
        print(f"{f}: credential-like text at byte {m.start()}")
EOF
)
[ -z "$leak" ] || die "credential-like content, nothing staged for upload: $leak"

# 3. sizes and sha256 (sha256sums.txt), the manifest lines of the staged set
if [ $quick = 0 ]; then
  echo "hf-upload: sha256 of ${#names[@]} files ..." >&2
  ( cd "$stage" && sha256sum -- "${names[@]}" ) > "$stage/sha256sums.txt"
  names+=(sha256sums.txt)
  for f in "${names[@]}"; do
    sha=$(awk -v f="$f" '$2 == f { print $1 }' "$stage/sha256sums.txt")
    [ "$f" = sha256sums.txt ] && sha=$(sha256sum "$stage/$f" | cut -c1-64)
    printf '%s %s %s %s\n' "$folder" "$f" "$(stat -L -c %s "$stage/$f")" "$sha"
  done | LC_ALL=C sort -k2,2 > "$lines"
else
  names+=(sha256sums.txt)
  for f in "${names[@]}"; do
    [ "$f" = sha256sums.txt ] && continue
    printf '%s %s %s -\n' "$folder" "$f" "$(stat -L -c %s "$stage/$f")"
  done | LC_ALL=C sort -k2,2 > "$lines"
fi

# 4. the exact list
total=0
printf '\n%-40s %16s  %s\n' "file in the repository" bytes source
for f in "${names[@]}"; do
  [ -e "$stage/$f" ] || { printf '%-40s %16s  %s\n' "$f" "-" "(written by a full run)"; continue; }
  n=$(stat -L -c %s "$stage/$f"); total=$((total + n))
  case $f in
    README.md) src=docs/${card##*/} ;;
    sha256sums.txt) src="generated: sha256 of the other ${#files[@]} files + README.md" ;;
    *) src=$dir/$f ;;
  esac
  printf '%-40s %16s  %s\n' "$f" "$n" "$src"
done
printf '%-40s %16s  (%s GB, %d files; staged as %s links in %s)\n' total "$total" \
  "$(awk -v b="$total" 'BEGIN { printf "%.2f", b / 1e9 }')" "${#names[@]}" "$linked" "$stage"
skipped=()
while IFS= read -r f; do
  printf '%s\n' "${files[@]}" | grep -qxF -- "$f" || skipped+=("$f")
done < <(find "$dir" -mindepth 1 -maxdepth 1 -printf '%P\n' | LC_ALL=C sort)
[ ${#skipped[@]} -eq 0 ] || echo "not uploaded from $dir (not on the allowlist): ${skipped[*]}"
echo

# 5. agree with scripts/models.manifest
want=$(awk -v d="$folder" '$1 == d' "$manifest" | LC_ALL=C sort -k2,2)
if [ $quick = 1 ]; then
  have=$(awk '{ print $1, $2, $3 }' "$lines"); want=$(awk '{ print $1, $2, $3 }' <<< "$want" | grep -v ' sha256sums.txt ' || true)
else
  have=$(cat "$lines")
fi
if [ $update = 1 ] && [ "$have" != "$want" ]; then
  tmp=$(mktemp)
  # keep the manifest's order: replace each of the folder's lines in place, drop files that are gone, append new
  # ones after the folder's last line
  awk -v d="$folder" -v l="$lines" '
    BEGIN { while ((getline x < l) > 0) { split(x, a, " "); line[a[2]] = x; order[++n] = a[2] } }
    $1 == d { if ($2 in line) { print line[$2]; seen[$2] = 1 } last = NR; next }
    last && !flushed { for (i = 1; i <= n; i++) if (!(order[i] in seen)) print line[order[i]]; flushed = 1 }
    { print }
    END { if (last && !flushed) for (i = 1; i <= n; i++) if (!(order[i] in seen)) print line[order[i]] }' "$manifest" > "$tmp"
  cat "$tmp" > "$manifest"; rm -f "$tmp"
  echo "hf-upload: rewrote the $folder lines of scripts/models.manifest"
  want=$(awk -v d="$folder" '$1 == d' "$manifest" | LC_ALL=C sort -k2,2)
fi
if [ "$have" = "$want" ]; then
  agree=1
  echo "manifest: the staged set agrees with scripts/models.manifest ($(wc -l <<< "$have") files, $( [ $quick = 1 ] && echo sizes || echo 'sizes + sha256'))"
else
  agree=0
  echo "manifest: the staged set DIFFERS from scripts/models.manifest (< manifest, > staged):"
  diff <(echo "$want") <(echo "$have") | grep '^[<>]' | sed 's/^/  /' || true
  echo "  (after a deliberate change, e.g. to docs/${card##*/}: rerun with --update-manifest and commit the manifest)"
fi
echo "credentials: none found in the text files or safetensors headers"

# 6. placeholders that must be filled in before anything is published
ph=$(grep -n -o -e '<repository URL>' -e '<HF_USER>' "$card" | tr '\n' ' ' || true)
[ -n "$ph" ] && echo "model card: placeholders left in docs/${card##*/}: $ph"

# the Hugging Face CLI: on PATH, or where pip puts it (a non-interactive ssh shell often lacks ~/.local/bin)
hf_bin=$(command -v hf || true)
for c in /usr/local/bin/hf "$HOME/.venvs/hf/bin/hf" "$HOME/.local/bin/hf"; do [ -n "$hf_bin" ] || { [ -x "$c" ] && hf_bin=$c; }; done
echo "hf CLI: ${hf_bin:-not found (python3 -m venv ~/.venvs/hf && ~/.venvs/hf/bin/pip install -U huggingface_hub)}"

repo=${FERRITE_HF_REPO:-}
vis=public; pflag=()
if [ $private = 1 ]; then vis=private; pflag=(--private); fi
# the hf CLI (typer) takes one pattern per --include / --exclude: the flag repeats for each
inc=()
for n in "${names[@]}"; do inc+=(--include "$n"); done
exc=(--exclude '.cache/*' --exclude '.git*' --exclude '*.log')
args=(upload "${repo:-$mrepo}" "$stage" . --repo-type model "${pflag[@]}" "${inc[@]}" "${exc[@]}"
      --commit-message "$msg")
if [ $upload = 0 ]; then
  echo
  echo "DRY RUN: nothing was uploaded. With --upload and FERRITE_HF_REPO=${repo:-$mrepo} it would run:"
  echo "  hf$(printf ' %q' "${args[@]}")"
  if [ $agree = 1 ] && [ -z "$ph" ]; then echo "  (ready: the manifest agrees and the model card has no placeholders)"
  else echo "  (not ready yet: --upload would stop at the manifest / placeholder check above)"; fi
  exit 0
fi

# 7. the upload itself
[ -n "$repo" ] || die "--upload needs FERRITE_HF_REPO (e.g. FERRITE_HF_REPO=$mrepo)"
case $repo in */*) ;; *) die "FERRITE_HF_REPO must be <user or org>/<name>, not '$repo'" ;; esac
[ $agree = 1 ] || die "not uploading: the staged set does not agree with scripts/models.manifest"
[ -z "$ph" ] || die "not uploading: fill in the placeholders in docs/${card##*/} first"
[ -n "$hf_bin" ] || die "the Hugging Face CLI is not installed: python3 -m venv ~/.venvs/hf && ~/.venvs/hf/bin/pip install -U huggingface_hub"
hfc() { env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN "$hf_bin" "$@"; }
hfc auth whoami > /dev/null 2>&1 || die "not logged in: run 'hf auth login' (the token stays in the Hugging Face CLI's own store)"
[ "$repo" = "$mrepo" ] || echo "hf-upload: note: scripts/models.manifest names $mrepo, uploading to $repo"
echo "hf-upload: uploading ${#names[@]} files, $((total / 1000000)) MB, to $repo ($vis)"
hfc "${args[@]}" || die "upload failed (write access for this login? network?)"
echo "hf-upload: done. Pin the commit in scripts/models.manifest:"
echo "  @ $folder $repo <commit sha from the output above> $role"
echo "then check a fresh download: scripts/fetch-models.sh $role"
