# Dump EXL3 reconstruct references with exllamav3 (one tensor per kind x bit width).
# Out: <out>/<name>.what.bin raw little-endian, plus manifest.json
import json, os, re, sys, torch
from safetensors import safe_open
import exllamav3.ext as ext_mod
ext = ext_mod.exllamav3_ext

# the EXL3 checkpoint: FERRITE_EXL3_DIR, else the standard folder under FERRITE_MODELS (tools/site.sh)
MODEL = os.environ.get("FERRITE_EXL3_DIR") or os.path.join(os.environ.get("FERRITE_MODELS", "/var/tmp/models"),
                                                         "Qwen3.8-27B-EXL3-3.0bpw-mix")
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]

bases = sorted({k[:-len(".trellis")] for k in idx if k.endswith(".trellis")})
def kind(b): return re.sub(r"layers\.\d+\.", "layers.N.", b)
handles = {}
def get(key):
    f = idx[key]
    if f not in handles: handles[f] = safe_open(f"{MODEL}/{f}", "pt", device="cpu")
    return handles[f].get_tensor(key)

picked = {}
for b in bases:
    tr_shape = handles.setdefault(idx[b + ".trellis"], safe_open(f"{MODEL}/{idx[b + '.trellis']}", "pt", device="cpu")).get_slice(b + ".trellis").get_shape()
    key = (kind(b), tr_shape[2] // 16)
    if key not in picked: picked[key] = b

H = torch.tensor([[(-1) ** bin(i & j).count("1") for j in range(128)] for i in range(128)], dtype=torch.float64) / (128 ** 0.5)
def had_blocks_l(w):  # H_blk @ w over dim 0
    k, n = w.shape
    return (H @ w.reshape(k // 128, 128, n)).reshape(k, n)
def had_blocks_r(w):
    k, n = w.shape
    return (w.reshape(k, n // 128, 128) @ H).reshape(k, n)

man = []
for (kd, bits), b in sorted(picked.items()):
    tr = get(b + ".trellis").cuda()
    suh = get(b + ".suh"); svh = get(b + ".svh")
    K = tr.shape[0] * 16; N = tr.shape[1] * 16
    nslice = N if N <= 32768 else 2048
    w_hat = torch.empty((K, nslice), dtype=torch.half, device="cuda")
    ext.reconstruct_slice(w_hat, tr, float(bits), False, True, 0)
    w_had = torch.empty((K, nslice), dtype=torch.half, device="cuda")
    ext.reconstruct_had_slice(w_had, tr, suh.cuda(), svh.cuda(), float(bits), False, True, 0)
    torch.cuda.synchronize()
    wh = w_hat.cpu()
    w64 = had_blocks_r(had_blocks_l(wh.double())) * suh.double()[:, None] * svh.double()[None, :nslice]
    name = b.replace("model.language_model.", "")
    wh.view(torch.int16).numpy().tofile(f"{OUT}/{name}.w_hat_f16.bin")
    w64.to(torch.bfloat16).view(torch.int16).numpy().tofile(f"{OUT}/{name}.w_bf16.bin")
    w_had.cpu().view(torch.int16).numpy().tofile(f"{OUT}/{name}.w_fused_f16.bin")
    w64.to(torch.float32).numpy().tofile(f"{OUT}/{name}.w_f32.bin") if K * nslice <= 40_000_000 else None
    man.append(dict(tensor=b, name=name, kind=kd, bits=bits, k=K, n=N, nslice=nslice,
                    has_f32=K * nslice <= 40_000_000))
    print(kd, bits, b, K, N, nslice, flush=True)
    del tr, w_hat, w_had
json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1)
