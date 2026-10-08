"""Diagnose native PyTorch dynamic INT8 on this aarch64 CPU: compare quantized engines and the simulated per-tensor path on a fixed subset.
Usage: python diag_int8.py MODEL N  -> writes local/diag/<model>_int8_engines.json"""
import sys, os, json, time, numpy as np, pandas as pd, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.set_num_threads(1)
MID = {"protectai": "protectai/deberta-v3-base-prompt-injection-v2", "piguard": "leolee99/PIGuard",
       "deepset": "deepset/deberta-v3-base-injection", "fmops": "fmops/distilbert-prompt-injection"}
key, n = sys.argv[1], int(sys.argv[2])
df = pd.read_parquet("/workspace/work/qpaper/local/evalset.parquet")
idx = np.random.default_rng(0).choice(len(df), n, replace=False); texts = df.text.values[idx].tolist()
tok = AutoTokenizer.from_pretrained(MID[key])
def run(m):
    out = []
    with torch.inference_mode():
        for t in texts:
            e = tok(t, truncation=True, max_length=512, return_tensors="pt"); out.append(torch.softmax(m(**e).logits.float(), -1)[0, 1].item())
    return np.array(out)
def fresh(): return AutoModelForSequenceClassification.from_pretrained(MID[key], dtype=torch.float32).eval()
res = {"n": n, "machine": os.uname().machine, "torch": torch.__version__, "engines": torch.backends.quantized.supported_engines}
t0 = time.time(); p32 = run(fresh()); res["fp32_s"] = time.time() - t0
for eng in ["x86", "fbgemm", "qnnpack", "onednn"]:
    try:
        torch.backends.quantized.engine = eng
        m = torch.ao.quantization.quantize_dynamic(fresh(), {nn.Linear}, dtype=torch.qint8)
        t0 = time.time(); p = run(m)
        res[eng] = {"flip": float(((p >= .5) != (p32 >= .5)).mean()), "mean_abs_dp": float(np.abs(p - p32).mean()), "s": time.time() - t0, "p_head": p[:5].round(4).tolist()}
    except Exception as e:
        res[eng] = {"error": repr(e)[:200]}
# simulated per-tensor (fbgemm-like) path, as in local_run.py w8a8pt
def act(m, inp):
    x = inp[0]; mn = torch.clamp(x.amin(), max=0); mx = torch.clamp(x.amax(), min=0)
    s = ((mx - mn) / 127).clamp(min=1e-8); z = torch.clamp(torch.round(-mn / s), 0, 127)
    return ((torch.clamp(torch.round(x / s) + z, 0, 127) - z) * s,)
m = fresh()
with torch.no_grad():
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            w = mod.weight.data; s = w.abs().amax().clamp(min=1e-8) / 127.5; mod.weight.data = torch.round(w / s).clamp(-128, 127) * s
            mod.register_forward_pre_hook(act)
p = run(m); res["sim_w8a8pt"] = {"flip": float(((p >= .5) != (p32 >= .5)).mean()), "mean_abs_dp": float(np.abs(p - p32).mean())}
res["fp32_head"] = p32[:5].round(4).tolist()
os.makedirs("/workspace/work/qpaper/local/diag", exist_ok=True)
json.dump(res, open(f"/workspace/work/qpaper/local/diag/{key}_int8_engines.json", "w"), indent=1); print(json.dumps(res, indent=1))
