"""LOCAL CPU variant of colab_run.py (Colab CLI blocked by sandbox gateway). Usage: python local_run.py MODEL CFG
(orig doc follows)
Quantized Guardians v2 experiments (Colab). Usage:
    python colab_run.py gpu   # FP32/BF16/simulated INT8/RTN/GPTQ/AWQ on CUDA
    python colab_run.py cpu   # FP32 CPU reference, native INT8: torch dynamic (fbgemm/x86) + ONNX Runtime dynamic
Reads  $ROOT/evalset.parquet ; writes $ROOT/scores_<phase>/<model>__<cfg>.npy, $ROOT/progress_<phase>.log, $ROOT/meta_<phase>.json
Every score file is written only after a complete pass over all 2,755 inputs (resumable: existing files are skipped).
"""
import sys, os, re, time, json, math, platform, hashlib
import numpy as np, pandas as pd, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PHASE = "local"
ROOT = os.environ.get("ROOT", "/workspace/work/qpaper/local")
OUTD = f"{ROOT}/scores"; os.makedirs(OUTD, exist_ok=True)
LOG = open(f"{ROOT}/progress_{PHASE}.log", "a")
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

MODELS = {"protectai": "protectai/deberta-v3-base-prompt-injection-v2", "piguard": "leolee99/PIGuard",
          "deepset": "deepset/deberta-v3-base-injection", "fmops": "fmops/distilbert-prompt-injection"}
ORDER = ["protectai", "piguard", "deepset", "fmops"]
DEV = "cuda" if (PHASE == "gpu" and torch.cuda.is_available()) else "cpu"
if DEV == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
else:
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - (1 if os.environ.get("SHARED") else 0)))

df = pd.read_parquet(f"{ROOT}/evalset.parquet"); TEXTS = df.text.tolist()
LENS = np.array([len(t) for t in TEXTS]); ORD = np.argsort(LENS, kind="stable")
BS = 16  # identical batching on every device so batch-level effects are comparable

# ---------------------------------------------------------------- quantizers
def rtn(w, bits, group):
    """Asymmetric min-max round-to-nearest fake quantization (same as v1 run_eval.py)."""
    out_f, in_f = w.shape; g = in_f if group is None else group
    if in_f % g: g = in_f
    W = w.reshape(out_f, in_f // g, g); mn, mx = W.amin(-1, keepdim=True), W.amax(-1, keepdim=True)
    q = 2 ** bits - 1; s = (mx - mn).clamp(min=1e-8) / q; z = torch.round(-mn / s)
    return ((torch.clamp(torch.round(W / s) + z, 0, q) - z) * s).reshape(out_f, in_f)

def linears(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]

def groups(model):
    """Group Linear layers by encoder block (sequential order); non-block linears (pooler/head) last."""
    g = {}
    for n, m in linears(model):
        r = re.search(r"layer\.(\d+)\.", n); k = int(r.group(1)) if r else 10 ** 6
        g.setdefault(k, []).append((n, m))
    return [g[k] for k in sorted(g)]

def gptq_quant(W, H, bits, group, blocksize=128, percdamp=0.01):
    """GPTQ (Frantar et al., 2023) with lazy batch updates, asymmetric min-max grid, no act-order."""
    W = W.clone().float(); H = H.clone(); ncol = W.shape[1]
    dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(H)); H += damp * torch.eye(ncol, device=W.device)
    L = torch.linalg.cholesky(H); Hinv = torch.cholesky_inverse(L); Hinv = torch.linalg.cholesky(Hinv, upper=True)
    q = 2 ** bits - 1
    def params(x):
        mn, mx = x.amin(1, keepdim=True), x.amax(1, keepdim=True); s = (mx - mn).clamp(min=1e-8) / q
        return s, torch.round(-mn / s)
    if group is None or ncol % group: s, z = params(W)
    Q = torch.zeros_like(W)
    for i1 in range(0, ncol, blocksize):
        i2 = min(i1 + blocksize, ncol); W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]; d = Hi[i, i]
            if group is not None and ncol % group == 0 and (i1 + i) % group == 0:
                s, z = params(W1[:, i:i + group])  # blocksize is a multiple of group, so the group lies inside W1
            qq = ((torch.clamp(torch.round(w.unsqueeze(1) / s) + z, 0, q) - z) * s).flatten()
            Q1[:, i] = qq; err = (w - qq) / d
            W1[:, i:] -= err.unsqueeze(1).matmul(Hi[i, i:].unsqueeze(0)); E1[:, i] = err
        Q[:, i1:i2] = Q1; W[:, i2:] -= E1.matmul(Hinv[i1:i2, i2:])
    return Q

QKV = ("query_proj", "key_proj", "value_proj", "q_lin", "k_lin", "v_lin")

def calibrate(model, tok, calib, method, bits, group):
    """Sequential block-wise calibration: block k sees inputs produced by blocks < k already quantized."""
    model.eval(); stats = {"blocks": 0, "linears": 0, "alphas": []}
    encs = [tok(t, truncation=True, max_length=512, return_tensors="pt").to(DEV) for t in calib]
    for blk in groups(model):
        acc = {n: {"H": None, "n": 0, "X": [], "absmean": None} for n, _ in blk}
        def mk(n):
            def hook(m, inp):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float(); a = acc[n]
                if method == "gptq":
                    h = x.t().matmul(x); k = x.shape[0]
                    a["H"] = h if a["H"] is None else a["H"] + h
                    a["n"] += k
                else:
                    a["absmean"] = x.abs().sum(0) if a["absmean"] is None else a["absmean"] + x.abs().sum(0); a["n"] += x.shape[0]
                    if x.shape[0] > 32: x = x[torch.randperm(x.shape[0], device=x.device)[:32]]
                    a["X"].append(x)
            return hook
        hs = [m.register_forward_pre_hook(mk(n)) for n, m in blk]
        with torch.no_grad():
            for e in encs: model(**e)
        for h in hs: h.remove()
        with torch.no_grad():
            if method == "gptq":
                for n, m in blk:
                    if acc[n]["n"] == 0:  # module never called in forward (e.g. unused head): RTN fallback
                        m.weight.data = rtn(m.weight.data.float(), bits, group).to(m.weight.dtype); stats.setdefault("unused", []).append(n); continue
                    H = acc[n]["H"] * (2.0 / acc[n]["n"])
                    W0 = m.weight.data.float(); Qg = gptq_quant(W0, H, bits, group); Qr = rtn(W0, bits, group)
                    lg = ((W0 - Qg).matmul(H) * (W0 - Qg)).sum().item(); lr = ((W0 - Qr).matmul(H) * (W0 - Qr)).sum().item()
                    stats.setdefault("loss_ratio", []).append(round(lg / max(lr, 1e-12), 4))  # GPTQ/RTN layer-output proxy loss
                    m.weight.data = Qg.to(m.weight.dtype)
            else:  # AWQ-style activation-aware per-input-channel scaling, grid search over alpha
                for n, m in blk:
                    if acc[n]["n"] == 0: m.weight.data = rtn(m.weight.data.float(), bits, group).to(m.weight.dtype); stats.setdefault("unused", []).append(n)
                blk = [(n, m) for n, m in blk if acc[n]["n"] > 0]
                shared = [(n, m) for n, m in blk if n.split(".")[-1] in QKV]
                units = ([shared] if shared else []) + [[(n, m)] for n, m in blk if n.split(".")[-1] not in QKV]
                for unit in units:
                    n0 = unit[0][0]; X = torch.cat(acc[n0]["X"])[:4096]; sx = acc[n0]["absmean"] / acc[n0]["n"]
                    best = (float("inf"), 0.0)
                    refs = [X.matmul(m.weight.data.float().t()) for _, m in unit]
                    for alpha in np.linspace(0, 1, 21):
                        s = sx.pow(float(alpha)).clamp(min=1e-4); s = s / (s.max() * s.min()).sqrt()
                        err = 0.0
                        for (_, m), r in zip(unit, refs):
                            Wq = rtn(m.weight.data.float() * s, bits, group) / s
                            err += (X.matmul(Wq.t()) - r).pow(2).mean().item()
                        if err < best[0]: best = (err, float(alpha))
                    alpha = best[1]; stats["alphas"].append(alpha)
                    s = sx.pow(alpha).clamp(min=1e-4); s = s / (s.max() * s.min()).sqrt()
                    for _, m in unit: m.weight.data = (rtn(m.weight.data.float() * s, bits, group) / s).to(m.weight.dtype)
        stats["blocks"] += 1; stats["linears"] += len(blk); del acc
    if "loss_ratio" in stats: stats["loss_ratio_mean"] = float(np.mean(stats.pop("loss_ratio")))
    return stats

def act_hook_per_token(m, inp):
    x = inp[0]; sx = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127
    return (torch.round(x / sx).clamp(-127, 127) * sx,)

def act_hook_per_tensor_u7(m, inp):
    """Mirror of PyTorch fbgemm dynamic quant: per-tensor asymmetric uint8 with reduce_range (0..127)."""
    x = inp[0]; mn = torch.clamp(x.amin(), max=0); mx = torch.clamp(x.amax(), min=0)
    s = ((mx - mn) / 127).clamp(min=1e-8); z = torch.clamp(torch.round(-mn / s), 0, 127)
    return ((torch.clamp(torch.round(x / s) + z, 0, 127) - z) * s,)

def apply_cfg(model, tok, cfg, calib_pool):
    info = {}
    with torch.no_grad():
        if cfg in ("fp32", "fp32cpu"): pass
        elif cfg == "bf16": model.to(torch.bfloat16)
        elif cfg in ("w8a8", "w8a8pt"):
            for n, m in linears(model):
                w = m.weight.data
                if cfg == "w8a8":
                    s = w.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127; m.weight.data = torch.round(w / s).clamp(-127, 127) * s
                    m.register_forward_pre_hook(act_hook_per_token)
                else:
                    s = w.abs().amax().clamp(min=1e-8) / 127.5; m.weight.data = torch.round(w / s).clamp(-128, 127) * s
                    m.register_forward_pre_hook(act_hook_per_tensor_u7)
        else:
            mth, fmt, *seed = cfg.split("_") if "_" in cfg else ("rtn", cfg)
            bits = int(fmt[1]); grp = None if fmt.endswith("pc") else int(fmt.split("g")[1])
            if mth == "rtn":
                for n, m in linears(model): m.weight.data = rtn(m.weight.data.float(), bits, grp)
            else:
                sd = int(seed[0][1:]); rng = np.random.default_rng(sd)
                calib = [calib_pool[i] for i in rng.choice(len(calib_pool), 128, replace=False)]
                info = calibrate(model, tok, calib, mth, bits, grp); info["seed"] = sd
                info["calib_sha1"] = hashlib.sha1("\n".join(calib).encode()).hexdigest()[:12]
    return model, info

@torch.no_grad()
def score(model, tok, dtype_cast=None, bs1=False):
    """v1-identical batching: 16 per batch, 4 when the batch's longest text >= 1000 chars; bs1=True -> one input at a time, no padding."""
    probs = np.zeros(len(TEXTS), np.float32); i = 0
    while i < len(ORD):
        bs = 1 if bs1 else (16 if LENS[ORD[min(i + 15, len(ORD) - 1)]] < 1000 else 4)
        idx = ORD[i:i + bs]; i += bs
        enc = tok([TEXTS[j] for j in idx], truncation=True, max_length=512, padding=True, return_tensors="pt").to(DEV)
        lg = model(**enc).logits.float(); probs[idx] = torch.softmax(lg, -1)[:, 1].cpu().numpy()
    return probs

def load(key):
    tok = AutoTokenizer.from_pretrained(MODELS[key], trust_remote_code=True)
    m = AutoModelForSequenceClassification.from_pretrained(MODELS[key], trust_remote_code=True, dtype=torch.float32).eval().to(DEV)
    return tok, m

def calib_pool_load():
    cf = f"{ROOT}/calib_pool.json"
    if os.path.exists(cf):
        d = json.load(open(cf)); return d["pool"], d["src"]
    pool, src = _calib_pool_load(); json.dump({"pool": pool, "src": src}, open(cf, "w")); return pool, src

def _calib_pool_load():
    """Unlabelled calibration texts from TRAIN splits only, excluding any text present in the eval set."""
    from datasets import load_dataset
    ev = set(TEXTS); pool = []; src = {}
    for name, col in [("deepset/prompt-injections", "text"), ("xTRam1/safe-guard-prompt-injection", "text"),
                      ("jackhhao/jailbreak-classification", "prompt")]:
        try:
            d = load_dataset(name, split="train"); t = [x for x in d[col] if isinstance(x, str) and x.strip() and x not in ev]
            pool += t; src[name] = len(t); log("calib pool", name, len(t))
        except Exception as e:
            log("calib pool FAILED", name, repr(e)[:200])
    pool = sorted(set(pool)); return pool, src

def run_one(key, cfg, pool, meta):
    out = f"{OUTD}/{key}__{cfg}.npy"
    if os.path.exists(out): return
    t0 = time.time()
    try:
        tok, m = load(key); m, info = apply_cfg(m, tok, cfg, pool); t1 = time.time()
        p = score(m, tok); np.save(out, p)
        meta["runs"][f"{key}__{cfg}"] = {"calib_s": round(t1 - t0, 1), "eval_s": round(time.time() - t1, 1), **info}
        log("done", key, cfg, f"calib {t1-t0:.0f}s eval {time.time()-t1:.0f}s", json.dumps(info)[:200])
    except Exception as e:
        import traceback; log("FAILED", key, cfg, repr(e)[:300]); LOG.write(traceback.format_exc() + "\n"); LOG.flush()
        meta["runs"][f"{key}__{cfg}"] = {"error": repr(e)[:300]}
    finally:
        pass
        if DEV == "cuda": torch.cuda.empty_cache()

def main_gpu():
    import transformers
    meta = json.load(open(f"{ROOT}/meta_gpu.json")) if os.path.exists(f"{ROOT}/meta_gpu.json") else {"runs": {}}
    meta.update({"torch": torch.__version__, "transformers": transformers.__version__, "device": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu"})
    pool, src = calib_pool_load(); meta["calib_pool"] = src; meta["calib_pool_n"] = len(pool)
    log("start gpu", meta["device"], "pool", len(pool))
    plan = [[(k, c) for k in ORDER for c in ["fp32", "bf16", "w8a8", "w8a8pt", "w4g128", "w4pc"]]]
    for s in (0, 1, 2): plan.append([(k, f"gptq_{f}_s{s}") for k in ORDER for f in ("w4pc", "w4g128")])
    for s in (0, 1, 2): plan.append([(k, f"awq_{f}_s{s}") for k in ORDER for f in ("w4pc", "w4g128")])
    plan.append([(k, c) for k in ORDER for c in ("w3g128", "gptq_w3g128_s0", "awq_w3g128_s0")])
    for stage in plan:
        for k, c in stage: run_one(k, c, pool, meta)
    log("ALL GPU DONE")

def main_cpu():
    import transformers
    meta = json.load(open(f"{ROOT}/meta_cpu.json")) if os.path.exists(f"{ROOT}/meta_cpu.json") else {"runs": {}}
    meta.update({"torch": torch.__version__, "transformers": transformers.__version__, "cpu": platform.processor() or platform.machine(),
                 "threads": torch.get_num_threads(), "qengine": None})
    eng = "x86" if "x86" in torch.backends.quantized.supported_engines else "fbgemm"; torch.backends.quantized.engine = eng; meta["qengine"] = eng
    log("start cpu", meta)
    for k in ORDER:
        run_one(k, "fp32cpu", None, meta)
        out = f"{OUTD}/{k}__int8dyn.npy"
        if not os.path.exists(out):
            t0 = time.time()
            try:
                tok, m = load(k); m = torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
                p = score(m, tok); np.save(out, p); meta["runs"][f"{k}__int8dyn"] = {"eval_s": round(time.time() - t0, 1)}
                log("done", k, "int8dyn", f"{time.time()-t0:.0f}s")
            except Exception as e:
                log("FAILED", k, "int8dyn", repr(e)[:300]); meta["runs"][f"{k}__int8dyn"] = {"error": repr(e)[:300]}
            pass
    for k in ORDER: onnx_runs(k, meta)
    log("ALL CPU DONE")

def onnx_runs(k, meta):
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    meta["onnxruntime"] = ort.__version__
    d = f"{ROOT}/onnx/{k}"; os.makedirs(d, exist_ok=True); f32, i8 = f"{d}/model.onnx", f"{d}/model_int8.onnx"
    need = [c for c in ("onnxfp32", "int8onnx") if not os.path.exists(f"{OUTD}/{k}__{c}.npy")]
    if not need: return
    try:
        tok, m = load(k)
        if not os.path.exists(f32):
            class Wr(nn.Module):
                def __init__(s, m): super().__init__(); s.m = m
                def forward(s, input_ids, attention_mask): return s.m(input_ids=input_ids, attention_mask=attention_mask).logits
            e = tok(["hello world", "a longer example sentence here"], padding=True, return_tensors="pt")
            kw = {"dynamo": False} if "dynamo" in torch.onnx.export.__code__.co_varnames else {}
            torch.onnx.export(Wr(m), (e["input_ids"], e["attention_mask"]), f32, input_names=["input_ids", "attention_mask"],
                              output_names=["logits"], dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "logits": {0: "b"}},
                              opset_version=17, **kw)
        if not os.path.exists(i8):
            quantize_dynamic(f32, i8, weight_type=QuantType.QInt8, per_channel=False, reduce_range=False)
        so = ort.SessionOptions(); so.intra_op_num_threads = torch.get_num_threads()
        for cfg, path in (("onnxfp32", f32), ("int8onnx", i8)):
            out = f"{OUTD}/{k}__{cfg}.npy"
            if os.path.exists(out): continue
            t0 = time.time(); sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            probs = np.zeros(len(TEXTS), np.float32)
            for i in range(0, len(ORD), 1):
                idx = ORD[i:i + 1]; e = tok([TEXTS[j] for j in idx], truncation=True, max_length=512, padding=True, return_tensors="np")
                lg = sess.run(None, {"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64)})[0]
                ex = np.exp(lg - lg.max(-1, keepdims=True)); probs[idx] = (ex / ex.sum(-1, keepdims=True))[:, 1]
            np.save(out, probs); meta["runs"][f"{k}__{cfg}"] = {"eval_s": round(time.time() - t0, 1)}; log("done", k, cfg, f"{time.time()-t0:.0f}s")
    except Exception as e:
        import traceback; log("FAILED", k, "onnx", repr(e)[:300]); LOG.write(traceback.format_exc() + "\n"); LOG.flush()
        meta["runs"][f"{k}__onnx"] = {"error": repr(e)[:300]}
    pass


def main_local(key, cfg):
    import transformers
    mf = f"{ROOT}/meta/{key}__{cfg}.json"; os.makedirs(f"{ROOT}/meta", exist_ok=True)
    meta = {"runs": {}, "torch": torch.__version__, "transformers": transformers.__version__, "cpu": platform.processor() or platform.machine(), "threads": torch.get_num_threads()}
    if cfg in ("onnxfp32", "int8onnx"):
        onnx_runs(key, meta); json.dump(meta, open(mf, "w"), indent=1); return
    if cfg == "int8onnxpc":
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic, QuantType
        d = f"{ROOT}/onnx/{key}"; f32, i8 = f"{d}/model.onnx", f"{d}/model_int8pc.onnx"; out = f"{OUTD}/{key}__{cfg}.npy"
        t0 = time.time()
        try:
            assert os.path.exists(f32), "run onnxfp32 first (exports the FP32 graph)"
            if not os.path.exists(i8): quantize_dynamic(f32, i8, weight_type=QuantType.QInt8, per_channel=True, reduce_range=False)
            tok, _ = load(key); so = ort.SessionOptions(); so.intra_op_num_threads = torch.get_num_threads()
            sess = ort.InferenceSession(i8, so, providers=["CPUExecutionProvider"]); probs = np.zeros(len(TEXTS), np.float32)
            for i in range(len(ORD)):
                idx = ORD[i:i + 1]; e = tok([TEXTS[j] for j in idx], truncation=True, max_length=512, padding=True, return_tensors="np")
                lg = sess.run(None, {"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64)})[0]
                ex = np.exp(lg - lg.max(-1, keepdims=True)); probs[idx] = (ex / ex.sum(-1, keepdims=True))[:, 1]
            np.save(out, probs); meta["runs"][f"{key}__{cfg}"] = {"eval_s": round(time.time() - t0, 1), "batch": 1, "per_channel": True, "ort": ort.__version__}
            log("done", key, cfg, f"{time.time()-t0:.0f}s")
        except Exception as e:
            import traceback; log("FAILED", key, cfg, repr(e)[:300]); LOG.write(traceback.format_exc() + "\n"); LOG.flush(); meta["runs"][f"{key}__{cfg}"] = {"error": repr(e)[:300]}
        json.dump(meta, open(mf, "w"), indent=1); return
    if cfg in ("int8dyn", "fp32bs1", "int8dynpc"):
        eng = "x86" if "x86" in torch.backends.quantized.supported_engines else "fbgemm"; torch.backends.quantized.engine = eng; meta["qengine"] = eng
        out = f"{OUTD}/{key}__{cfg}.npy"
        if os.path.exists(out): return
        t0 = time.time()
        try:
            tok, m = load(key)
            if cfg == "int8dyn": m = torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
            if cfg == "int8dynpc":
                from torch.ao.quantization import per_channel_dynamic_qconfig
                m = torch.ao.quantization.quantize_dynamic(m, {nn.Linear: per_channel_dynamic_qconfig}, dtype=torch.qint8)
            p = score(m, tok, bs1=True); np.save(out, p); meta["runs"][f"{key}__{cfg}"] = {"eval_s": round(time.time() - t0, 1), "batch": 1}
            log("done", key, cfg, f"{time.time()-t0:.0f}s")
        except Exception as e:
            import traceback; log("FAILED", key, cfg, repr(e)[:300]); LOG.write(traceback.format_exc() + "\n"); LOG.flush(); meta["runs"][f"{key}__{cfg}"] = {"error": repr(e)[:300]}
        json.dump(meta, open(mf, "w"), indent=1); return
    pool = None
    if cfg.startswith(("gptq", "awq")):
        pool, src = calib_pool_load(); meta["calib_pool"] = src; meta["calib_pool_n"] = len(pool)
    run_one(key, cfg, pool, meta)
    json.dump(meta, open(mf, "w"), indent=1)

if __name__ == "__main__":
    main_local(sys.argv[1], sys.argv[2])
