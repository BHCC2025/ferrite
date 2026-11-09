#!/bin/bash
# KL vs the BF16 reference for ferrite (same contract and scorer as $FERRITE_KL_DIR/kl.sh; does not edit it).
# usage: tools/kl.sh <model-dir> <set: code|text|long16k> [nseq] [skip] [rows] [name]
#   defaults per set: code 40 seqs row0 0 skip 1024; text 20 seqs row0 40 skip 1024; long16k 3 seqs skip 15360; rows 512
# env: FE_BIN (default target/release/ferrite), FE_BATCH, KL_OUT (dir for the logits bin, default
#      $FERRITE_STATE_DIR/kl); the reference folder and the GPU lock: tools/site.sh (FERRITE_KL_DIR, FERRITE_GPU_LOCK)
set -e
. "$(dirname "$0")/site.sh"
K=$FERRITE_KL_DIR
M=$1; SET=$2
case $SET in
  code) ref=short; row0=0; n=40; skip=1024 ;;
  text) ref=short; row0=40; n=20; skip=1024 ;;
  long16k) ref=long16k; row0=0; n=3; skip=15360 ;;
  *) echo "unknown set $SET"; exit 2 ;;
esac
n=${3:-$n}; skip=${4:-$skip}; rows=${5:-512}; name=${6:-ferrite-$SET}
BIN=${FE_BIN:-$(dirname $0)/../target/release/ferrite}
OUT=${KL_OUT:-$FERRITE_STATE_DIR/kl}; mkdir -p $OUT
ids=$OUT/ids-$SET-$n.txt
head -n $n $K/kl-ids-$SET.txt > $ids
bin=$OUT/$name.bin
echo "$(date +%T) $name: engine ($n seqs, skip $skip, rows $rows)" >&2
flock $FERRITE_GPU_LOCK $BIN $M logits $ids $bin $skip $rows
docker run --rm --gpus all -v $K:$K -v $OUT:$OUT nvcr.io/nvidia/pytorch:26.07-py3 \
  python $K/score.py $bin $K/ref/$ref.pt $row0 $n $skip $rows $name 2>/dev/null | tail -1 | tee -a $OUT/results.jsonl
rm -f $bin
