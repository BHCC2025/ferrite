# kl_perseq.py BIN REF ROW0 NSEQ SKIP ROWS: per-sequence KL(BF16 || engine) (same math as $FERRITE_KL_DIR/score.py),
# plus the worst positions, to find where a run goes wrong.
import sys, numpy as np, torch
binp, refp, row0, nseq, skip, rows = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:7])
ref = torch.load(refp)
V = 248320
raw = np.memmap(binp, dtype=np.uint16, mode="r").reshape(nseq, rows, -1)
dev = "cuda" if torch.cuda.is_available() else "cpu"
for s in range(nseq):
    n = row0 + s
    x = torch.from_numpy(np.array(raw[s]).astype(np.int32) << 16).view(torch.float32).to(dev)[:, :V]
    p_lp = ref["topk_lp"][n, skip:skip + rows].to(dev); p_idx = ref["topk_idx"][n, skip:skip + rows].to(dev).long()
    q_lse = torch.logsumexp(x, -1); q_at = x.gather(1, p_idx) - q_lse[:, None]
    p = p_lp.exp(); p_rest = (1 - p.sum(-1)).clamp(min=0); q_rest = (1 - q_at.exp().sum(-1)).clamp(min=1e-30)
    kl = (p * (p_lp - q_at)).sum(-1)
    kl = kl + torch.where(p_rest > 0, p_rest * (torch.log(p_rest.clamp(min=1e-30)) - torch.log(q_rest)), torch.zeros_like(kl))
    bad = (kl > 1.0).nonzero().flatten().tolist()
    nan = int(torch.isnan(x).any(-1).sum().item())
    print(f"seq {n}: kl_mean {kl.mean().item():.4f} max {kl.max().item():.2f} rows>1: {len(bad)} first {bad[:5]} nan_rows {nan}", flush=True)
