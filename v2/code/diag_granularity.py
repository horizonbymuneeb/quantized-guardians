"""Which quantization granularity causes the native-INT8 collapse? Simulated W8A8 variants on a fixed random subset.
Usage: python diag_granularity.py MODEL N -> local/diag/<model>_granularity.json"""
import sys, os, json, time, numpy as np, pandas as pd, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.set_num_threads(1)
MID = {"protectai": "protectai/deberta-v3-base-prompt-injection-v2", "piguard": "leolee99/PIGuard",
       "deepset": "deepset/deberta-v3-base-injection", "fmops": "fmops/distilbert-prompt-injection"}
key, n = sys.argv[1], int(sys.argv[2])
df = pd.read_parquet("/workspace/work/qpaper/local/evalset.parquet")
idx = np.random.default_rng(1).choice(len(df), n, replace=False); texts = df.text.values[idx].tolist()
tok = AutoTokenizer.from_pretrained(MID[key])
def run(m):
    out = []
    with torch.inference_mode():
        for t in texts:
            e = tok(t, truncation=True, max_length=512, return_tensors="pt"); out.append(torch.softmax(m(**e).logits.float(), -1)[0, 1].item())
    return np.array(out)
def fresh(): return AutoModelForSequenceClassification.from_pretrained(MID[key], dtype=torch.float32).eval()
def a_tok(m, inp):
    x = inp[0]; s = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127; return (torch.round(x / s).clamp(-127, 127) * s,)
def a_ten(qmax):
    def h(m, inp):
        x = inp[0]; mn = torch.clamp(x.amin(), max=0); mx = torch.clamp(x.amax(), min=0)
        s = ((mx - mn) / qmax).clamp(min=1e-8); z = torch.clamp(torch.round(-mn / s), 0, qmax)
        return ((torch.clamp(torch.round(x / s) + z, 0, qmax) - z) * s,)
    return h
def build(wgran, act, skip=()):
    m = fresh(); stats = {}
    with torch.no_grad():
        for name, mod in m.named_modules():
            if not isinstance(mod, nn.Linear) or any((k(name) if callable(k) else k in name) for k in skip): continue
            w = mod.weight.data
            if wgran == "ch": s = w.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127; mod.weight.data = torch.round(w / s).clamp(-127, 127) * s
            elif wgran == "ten": s = w.abs().amax().clamp(min=1e-8) / 127.5; mod.weight.data = torch.round(w / s).clamp(-128, 127) * s
            if act == "tok": mod.register_forward_pre_hook(a_tok)
            elif act == "ten7": mod.register_forward_pre_hook(a_ten(127))
            elif act == "ten8": mod.register_forward_pre_hook(a_ten(255))
    return m
VARIANTS = {"Wch_Atok(=w8a8 sim)": ("ch", "tok", ()), "Wten_Aten7(=native-like)": ("ten", "ten7", ()),
            "Wten_Atok": ("ten", "tok", ()), "Wch_Aten7": ("ch", "ten7", ()), "Wten_none(weights only)": ("ten", "none", ()),
            "none_Aten7(acts only)": ("none", "ten7", ()), "Wten_Aten8(full 8-bit range)": ("ten", "ten8", ()),
            "Wten_Aten7_skipFFNout": ("ten", "ten7", (lambda nm: nm.endswith("output.dense") and "attention" not in nm,)), "Wten_Aten7_skipHead": ("ten", "ten7", ("classifier", "pooler"))}
jf = f"/workspace/work/qpaper/local/diag/{key}_granularity.json"
res = json.load(open(jf)) if os.path.exists(jf) else {"n": n, "model": key}
p32 = run(fresh())
for name, (wg, ac, sk) in VARIANTS.items():
    if name in res: continue
    t0 = time.time(); p = run(build(wg, ac, sk))
    res[name] = {"flip": float(((p >= .5) != (p32 >= .5)).mean()), "mean_abs_dp": float(np.abs(p - p32).mean()), "s": round(time.time() - t0)}
    print(name, res[name], flush=True)
    os.makedirs("/workspace/work/qpaper/local/diag", exist_ok=True)
    json.dump(res, open(f"/workspace/work/qpaper/local/diag/{key}_granularity.json", "w"), indent=1)
