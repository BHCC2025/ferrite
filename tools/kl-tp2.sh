#!/bin/bash
# KL vs the BF16 reference for ferrite TP2 across this host (rank 0, runs this script) and the peer (rank 1, the same
# binary path and model): same contract and scorer as tools/kl.sh. Under the global TP2 lock and this host's GPU
# lock (the tools/tp2.sh lock order), rank 1 is started over ssh (it takes the peer's lock), then rank 0; both run
# `ferrite logits` on the same ids in lockstep and rank 0's logits are scored.
# usage: tools/kl-tp2.sh <model-dir> <set: code|text|long16k> [nseq] [skip] [rows] [name]
# env: FE_BIN (default target/release/ferrite, same path on both nodes), KL_OUT, FE_TP_PORT,
#      TP_ENV (extra FE_* settings for both ranks, e.g. TP_ENV=FE_PROMPT=fp4); site settings: tools/site.sh
set -e
# killable on OOM (ssh sessions may inherit sshd's oom_score_adj -1000); the engines started below inherit it
echo 500 > /proc/self/oom_score_adj 2>/dev/null || true
. "$(dirname "$0")/site.sh"
ferrite_two_node || exit 1
K=$FERRITE_KL_DIR
M=$1; SET=$2
case $SET in
  code) ref=short; row0=0; n=40; skip=1024 ;;
  text) ref=short; row0=40; n=20; skip=1024 ;;
  long16k) ref=long16k; row0=0; n=3; skip=15360 ;;
  *) echo "unknown set $SET"; exit 2 ;;
esac
n=${3:-$n}; skip=${4:-$skip}; rows=${5:-512}; name=${6:-ferrite-tp2-$SET}
BIN=$(readlink -f ${FE_BIN:-$(dirname $0)/../target/release/ferrite})
PEER=$FERRITE_PEER
OUT=${KL_OUT:-$FERRITE_STATE_DIR/kl}; mkdir -p $OUT
ids=$OUT/ids-$SET-$n.txt
head -n $n $K/kl-ids-$SET.txt > $ids
bin=$OUT/$name.bin
FAB="$FERRITE_FAB FE_TP_WORLD=2 FE_TP_ADDR=$FERRITE_TP_HOST:${FE_TP_PORT:-29611} FE_MTP=0 $TP_ENV"
ssh -n $PEER "mkdir -p $OUT" && rsync -a $ids $PEER:$ids
cat > $OUT/$name-run.sh <<EOS
#!/bin/bash
# a rank holds ~25 GB (EXL3 + bf16 prompt slices + states): keep >= 20 GB free on both nodes
free -g | awk '/^Mem/ { if (\$7 < $FERRITE_MIN_FREE_GB) { print "$(hostname): only " \$7 " GB available"; exit 1 } }' || exit 1
ssh -n $PEER "free -g" | awk '/^Mem/ { if (\$7 < $FERRITE_MIN_FREE_GB) { print "peer: only " \$7 " GB available"; exit 1 } }' || exit 1
# rank 1 fully detached on the peer (setsid, no stdio on the ssh channel, or ssh would wait for it); it writes its
# copy of the logits to the same path on the peer (identical argv: the lockstep guard) and deletes it
ssh -n $PEER "cd /tmp && setsid -f bash -c 'env $FAB FE_TP_RANK=1 timeout 3600 flock $FERRITE_GPU_LOCK $BIN $M logits $ids $bin $skip $rows > $OUT/$name-r1.log 2>&1; rm -f $bin' < /dev/null > /dev/null 2>&1"
# rank 0 listens; if rank 1 never connects it gives up with the connect window, not after hours
env $FAB FE_TP_RANK=0 timeout 3600 $BIN $M logits $ids $bin $skip $rows
rc=\$?
ssh -n $PEER "flock -w 120 $FERRITE_GPU_LOCK true" || true
exit \$rc
EOS
chmod +x $OUT/$name-run.sh
echo "$(date +%T) $name: TP2 engine ($n seqs, skip $skip, rows $rows)" >&2
flock $FERRITE_TP2_LOCK flock $FERRITE_GPU_LOCK $OUT/$name-run.sh
docker run --rm --gpus all -v $K:$K -v $OUT:$OUT nvcr.io/nvidia/pytorch:26.07-py3 \
  python $K/score.py $bin $K/ref/$ref.pt $row0 $n $skip $rows $name 2>/dev/null | tail -1 | tee -a $OUT/results.jsonl
rm -f $bin
