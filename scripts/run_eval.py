"""Score every eval prompt with a detector under one numeric format. Usage: run_eval.py MODEL_KEY CONFIG"""
import sys, time, json, os, copy
import numpy as np, pandas as pd, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.set_num_threads(2)
MODELS = {
    "protectai": ("protectai/deberta-v3-base-prompt-injection-v2", 1),
    "deepset": ("deepset/deberta-v3-base-injection", 1),
    "fmops": ("fmops/distilbert-prompt-injection", 1),
    "piguard": ("leolee99/PIGuard", 1),
}
key, cfg = sys.argv[1], sys.argv[2]
mid, pos = MODELS[key]
OUT = f"scores/{key}__{cfg}.npy"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
if os.path.exists(OUT): sys.exit(0)
tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(mid, trust_remote_code=True).eval()

def rtn_(w, bits, group):
    """Round-to-nearest asymmetric weight quantization (fake-quant), in place."""
    out_f, in_f = w.shape
    g = in_f if group is None else group
    if in_f % g: g = in_f
    W = w.view(out_f, in_f // g, g)
    mn, mx = W.amin(-1, keepdim=True), W.amax(-1, keepdim=True)
    q = 2 ** bits - 1
    s = (mx - mn).clamp(min=1e-8) / q
    z = torch.round(-mn / s)
    Wq = (torch.clamp(torch.round(W / s) + z, 0, q) - z) * s
    w.copy_(Wq.view(out_f, in_f))

def weight_only(m, bits, group):
    with torch.no_grad():
        for n, mod in m.named_modules():
            if isinstance(mod, nn.Linear):
                rtn_(mod.weight.data, bits, group)
    return m

dtype = torch.float32
if cfg == "fp32": pass
elif cfg == "bf16": model = model.to(torch.bfloat16); dtype = torch.bfloat16
elif cfg == "int8dyn": model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
elif cfg == "w8a8":  # simulated dynamic INT8: per-channel sym weights, per-token sym activations
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, nn.Linear):
                w = mod.weight.data; s_ = w.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127
                mod.weight.data = torch.round(w / s_).clamp(-127, 127) * s_
                def pre(m, inp):
                    x = inp[0]; sx = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127
                    return (torch.round(x / sx).clamp(-127, 127) * sx,)
                mod.register_forward_pre_hook(pre)
elif cfg.startswith("w"):  # e.g. w4g128, w4pc, w3g128, w8pc
    bits = int(cfg[1]); grp = None if cfg.endswith("pc") else int(cfg.split("g")[1])
    model = weight_only(model, bits, grp)
else: raise ValueError(cfg)

df = pd.read_parquet("data/evalset.parquet")
texts = df.text.tolist()
lens = [len(t) for t in texts]
order = np.argsort(lens)
CK = OUT + ".part.npz"
probs = np.zeros(len(texts), dtype=np.float32); start = 0
if os.path.exists(CK):
    z = np.load(CK); probs = z["p"]; start = int(z["i"])
t0 = time.time(); bs = 16; BUDGET = float(os.environ.get("BUDGET", 520))
with torch.inference_mode():
    i = start
    while i < len(order):
        bs = 16 if lens[order[min(i+15, len(order)-1)]] < 1000 else (1 if cfg == "int8dyn" else 4)
        if time.time() - t0 > BUDGET:
            np.savez(CK, p=probs, i=i); print("partial", key, cfg, i, flush=True); sys.exit(3)
        idx = order[i:i + bs]
        batch = [texts[j] for j in idx]
        if cfg == "int8dyn":  # bucket shapes to avoid quantized-kernel cache growth
            L = max(len(tok(b, truncation=True, max_length=512)["input_ids"]) for b in batch)
            L = next(x for x in (64, 128, 256, 512) if x >= L)
            enc = tok(batch, truncation=True, max_length=L, padding="max_length", return_tensors="pt")
        else:
            enc = tok(batch, truncation=True, max_length=512, padding=True, return_tensors="pt")
        logits = model(**enc).logits.float()
        p = torch.softmax(logits, -1)[:, pos].numpy()
        probs[idx] = p
        i += bs
np.save(OUT, probs)
if os.path.exists(CK): os.remove(CK)
print(key, cfg, f"{time.time()-t0:.0f}s", flush=True)
