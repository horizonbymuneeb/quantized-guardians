"""Compute all v2 metrics from saved local CPU scores. Usage: python local_analyze.py
Scores: local/scores/<model>__<cfg>.npy (new runs) with fallback to local/scores_v1 (v1 runs; FP32 reproduced to <1e-6).
Writes local/results/*.csv, summary.json, fig_*.pdf/png."""
import os, json, glob, itertools
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import binomtest
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/workspace/work/qpaper/local"; R = f"{ROOT}/results"; os.makedirs(R, exist_ok=True)
df = pd.read_parquet(f"{ROOT}/evalset.parquet"); y = df.label.values; src = df.src.values; N = len(y)
CLEAN = np.isin(src, ["deepset", "safeguard", "jailbreak_cls"])
SLICES = ["evade_homoglyph", "evade_zerowidth", "evade_leet"]
MODELS = ["protectai", "piguard", "deepset", "fmops"]
NAME = {"protectai": "ProtectAI-v2", "piguard": "PIGuard", "deepset": "Deepset", "fmops": "FMOps"}
B = 1000
PROV = {}

def S(m, c):
    for d, tag in ((f"{ROOT}/scores", "v2"), (f"{ROOT}/scores_v1", "v1")):
        f = f"{d}/{m}__{c}.npy"
        if os.path.exists(f): PROV[f"{m}__{c}"] = tag; return np.load(f)
    return None

def all_cfgs(m):
    cs = set()
    for d in (f"{ROOT}/scores", f"{ROOT}/scores_v1"):
        for f in glob.glob(f"{d}/{m}__*.npy"): cs.add(os.path.basename(f)[len(m) + 2:-4])
    return sorted(cs)

def ece(p, yy, bins=10):
    e = 0; b = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(b[:-1], b[1:]):
        mm = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mm.sum(): e += mm.mean() * abs(p[mm].mean() - yy[mm].mean())
    return e

BOOT = np.random.default_rng(0).integers(0, N, (B, N))
def boot_ci(v):
    bs = v[BOOT].mean(1); return np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def metrics(p, base):
    pred = p >= 0.5; bp = base >= 0.5; r = {}
    r["auc"] = roc_auc_score(y[CLEAN], p[CLEAN]); r["f1"] = f1_score(y[CLEAN], pred[CLEAN])
    # bootstrap CI on F1 (clean pool) for the aggregate-metric table
    ci = np.where(CLEAN)[0]; rng = np.random.default_rng(1); f1s = []
    for _ in range(300):
        ii = rng.choice(ci, len(ci)); f1s.append(f1_score(y[ii], pred[ii]))
    r["f1_lo"], r["f1_hi"] = np.percentile(f1s, 2.5), np.percentile(f1s, 97.5)
    r["tpr"] = pred[CLEAN & (y == 1)].mean(); r["fpr"] = pred[CLEAN & (y == 0)].mean()
    r["notinject_fpr"] = pred[src == "notinject"].mean()
    for e in SLICES: r[e] = pred[src == e].mean()
    r["ece"] = ece(p[CLEAN], y[CLEAN])
    fl = (pred != bp); r["flip_rate"] = fl.mean(); r["flip_lo"], r["flip_hi"] = boot_ci(fl.astype(float))
    r["flips"] = int(fl.sum()); r["missed"] = int((bp & ~pred & (y == 1)).sum()); r["new_fp"] = int((~bp & pred & (y == 0)).sum())
    r["fixed_attacks"] = int((~bp & pred & (y == 1)).sum()); r["removed_fp"] = int((bp & ~pred & (y == 0)).sum())
    r["mean_abs_dp"] = float(np.abs(p - base).mean()); r["max_abs_dp"] = float(np.abs(p - base).max())
    cb, cq = bp == (y == 1), pred == (y == 1); b_, c_ = int((cb & ~cq).sum()), int((~cb & cq).sum())
    r["mcnemar_worse"], r["mcnemar_better"] = b_, c_; r["mcnemar_p"] = binomtest(b_, b_ + c_, 0.5).pvalue if b_ + c_ else 1.0
    r["d_f1"] = r["f1"] - f1_score(y[CLEAN], bp[CLEAN]); r["d_auc"] = r["auc"] - roc_auc_score(y[CLEAN], base[CLEAN])
    return r

def method_of(c):
    if c.startswith(("gptq_", "awq_")): return c.split("_")[0]
    if c in ("int8dyn", "int8onnx", "int8dynpc", "int8onnxpc"): return "native"
    if c in ("fp32", "fp32bs1", "onnxfp32"): return "ref"
    if c in ("bf16", "w8a8", "w8a8pt"): return c
    return "rtn"

rows = []
for m in MODELS:
    base = S(m, "fp32")
    if base is None: continue
    for c in all_cfgs(m):
        p = S(m, c); parts = c.split("_")
        bases = [("fp32", base)]
        if c == "int8dyn" and S(m, "fp32bs1") is not None: bases.append(("fp32bs1", S(m, "fp32bs1")))
        if c in ("int8onnx", "int8onnxpc") and S(m, "onnxfp32") is not None: bases.append(("onnxfp32", S(m, "onnxfp32")))
        for bn, bb in bases:
            r = {"model": m, "cfg": c, "baseline": bn, "method": method_of(c),
                 "fmt": parts[1] if len(parts) > 2 else c, "seed": int(parts[2][1:]) if len(parts) > 2 else -1,
                 "source": PROV.get(f"{m}__{c}"), **metrics(p, bb)}
            rows.append(r)
res = pd.DataFrame(rows); res.to_csv(f"{R}/results_all.csv", index=False)
F = res[res.baseline == "fp32"]

# native vs simulated INT8
ag = []
for m in MODELS:
    bg = S(m, "fp32")
    if bg is None: continue
    sims = {k: S(m, k) for k in ("w8a8", "w8a8pt")}
    nats = {"int8dyn": (S(m, "int8dyn"), S(m, "fp32bs1") if S(m, "fp32bs1") is not None else bg),
            "int8onnx": (S(m, "int8onnx"), S(m, "onnxfp32") if S(m, "onnxfp32") is not None else bg),
            "int8dynpc": (S(m, "int8dynpc"), bg),
            "int8onnxpc": (S(m, "int8onnxpc"), S(m, "onnxfp32") if S(m, "onnxfp32") is not None else bg)}
    for (nn_, (pn, nb)), (sn, ps) in itertools.product(nats.items(), sims.items()):
        if pn is None or ps is None: continue
        fn = (pn >= .5) != (nb >= .5); fs = (ps >= .5) != (bg >= .5)
        inter = (fn & fs).sum(); uni = (fn | fs).sum()
        ag.append({"model": m, "native": nn_, "sim": sn, "decision_agree": ((pn >= .5) == (ps >= .5)).mean(),
                   "flip_native": fn.mean(), "flip_sim": fs.mean(), "flipset_jaccard": inter / uni if uni else 1.0,
                   "flip_overlap": int(inter), "flip_native_n": int(fn.sum()), "flip_sim_n": int(fs.sum()),
                   "score_corr": float(np.corrcoef(pn - nb, ps - bg)[0, 1]), "mean_abs_score_diff": float(np.abs(pn - ps).mean())})
    for a in ("fp32bs1", "onnxfp32"):
        pa = S(m, a)
        if pa is not None: ag.append({"model": m, "native": a, "sim": "fp32", "decision_agree": ((pa >= .5) == (bg >= .5)).mean(),
                                      "mean_abs_score_diff": float(np.abs(pa - bg).mean()), "max_abs_score_diff": float(np.abs(pa - bg).max())})
pd.DataFrame(ag).to_csv(f"{R}/native_vs_sim.csv", index=False)

# seed aggregation (GPTQ / AWQ)
g = F[F.method.isin(["gptq", "awq"])]; agg = []
for (m, mt, f), q in g.groupby(["model", "method", "fmt"]):
    d = {"model": m, "method": mt, "fmt": f, "n_seeds": len(q)}
    for k in ["flip_rate", "missed", "new_fp", "f1", "auc", "d_f1", "notinject_fpr", "evade_homoglyph", "evade_zerowidth", "evade_leet", "mcnemar_p", "flip_lo", "flip_hi", "max_abs_dp"]:
        d[k + "_mean"] = q[k].mean(); d[k + "_std"] = q[k].std(ddof=1) if len(q) > 1 else np.nan
    sets = [(S(m, c) >= .5) != (S(m, "fp32") >= .5) for c in q.cfg]
    js = [(a & b).sum() / max((a | b).sum(), 1) for a, b in itertools.combinations(sets, 2)]
    d["flipset_jaccard_mean"] = float(np.mean(js)) if js else np.nan
    if S(m, f) is not None:
        rs = (S(m, f) >= .5) != (S(m, "fp32") >= .5)
        d["rtn_flip_rate"] = rs.mean(); d["jaccard_vs_rtn_mean"] = float(np.mean([(a & rs).sum() / max((a | rs).sum(), 1) for a in sets]))
    agg.append(d)
pd.DataFrame(agg).to_csv(f"{R}/seed_agg.csv", index=False)

# threshold re-calibration: 5 random 50/50 splits
cal = []
for _, r in F.iterrows():
    if r.method == "ref": continue
    p = S(r.model, r.cfg); bp = S(r.model, "fp32") >= .5; ths = np.linspace(0.01, 0.99, 197)
    a5, ac, ts = [], [], []
    for sd in range(5):
        cm = np.random.default_rng(sd).random(N) < 0.5
        agree = [((p[cm] >= t) == bp[cm]).mean() for t in ths]; t_star = ths[int(np.argmax(agree))]
        a5.append(((p[~cm] >= .5) != bp[~cm]).mean()); ac.append(((p[~cm] >= t_star) != bp[~cm]).mean()); ts.append(t_star)
    cal.append({"model": r.model, "cfg": r.cfg, "flip05_mean": np.mean(a5), "flip05_std": np.std(a5, ddof=1),
                "flipcal_mean": np.mean(ac), "flipcal_std": np.std(ac, ddof=1), "tstar_mean": np.mean(ts), "tstar_min": min(ts), "tstar_max": max(ts)})
pd.DataFrame(cal).to_csv(f"{R}/recal.csv", index=False)

# per-source flip rates
ps = []
for _, r in F.iterrows():
    if r.method == "ref": continue
    fl = (S(r.model, r.cfg) >= .5) != (S(r.model, "fp32") >= .5)
    for s_ in np.unique(src): ps.append({"model": r.model, "cfg": r.cfg, "src": s_, "n": int((src == s_).sum()), "flip_rate": fl[src == s_].mean()})
pd.DataFrame(ps).to_csv(f"{R}/per_source.csv", index=False)

# boundary analysis: flip rate vs FP32 margin |p-0.5|
bd = []
for _, r in F.iterrows():
    if r.method == "ref": continue
    b = S(r.model, "fp32"); fl = (S(r.model, r.cfg) >= .5) != (b >= .5); mg = np.abs(b - .5)
    for lo, hi in ((0, .1), (.1, .3), (.3, .45), (.45, .51)):
        mm = (mg >= lo) & (mg < hi); bd.append({"model": r.model, "cfg": r.cfg, "margin_lo": lo, "margin_hi": hi, "n": int(mm.sum()), "flip_rate": fl[mm].mean() if mm.sum() else np.nan})
pd.DataFrame(bd).to_csv(f"{R}/margin.csv", index=False)

# figures
try:
    fig, ax = plt.subplots(figsize=(7.2, 2.7)); cf = ["bf16", "w8a8", "w8a8pt", "int8dyn", "int8onnx", "int8onnxpc", "w4g128", "w4pc"]
    lab = ["BF16", "W8A8\n(sim, tok)", "W8A8\n(sim, tensor)", "INT8\nPyTorch", "INT8\nONNX RT", "INT8 ORT\nper-chan.", "W4-g128", "W4-pc"]
    xs = np.arange(len(cf)); w = 0.2
    for j, m in enumerate(MODELS):
        v, lo, hi = [], [], []
        for c in cf:
            q = F[(F.model == m) & (F.cfg == c)]
            v.append(100 * q.flip_rate.iloc[0] if len(q) else np.nan); lo.append(100 * q.flip_lo.iloc[0] if len(q) else np.nan); hi.append(100 * q.flip_hi.iloc[0] if len(q) else np.nan)
        v, lo, hi = map(np.array, (v, lo, hi))
        ax.bar(xs + (j - 1.5) * w, v, w, yerr=[np.nan_to_num(v - lo), np.nan_to_num(hi - v)], capsize=1.5, label=NAME[m])
    ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=7); ax.set_ylabel("Decision flip rate (%)"); ax.legend(fontsize=7, ncol=4)
    plt.tight_layout(); plt.savefig(f"{R}/fig_flip.pdf"); plt.savefig(f"{R}/fig_flip.png", dpi=150); plt.close()
    sa = pd.DataFrame(agg)
    if len(sa):
        fig, ax = plt.subplots(figsize=(3.5, 2.6)); xs = np.arange(len(MODELS)); w = 0.27
        for j, mt in enumerate(["rtn", "gptq", "awq"]):
            vals, errs = [], []
            for m in MODELS:
                if mt == "rtn":
                    q = F[(F.model == m) & (F.cfg == "w4pc")]; vals.append(100 * q.flip_rate.iloc[0] if len(q) else np.nan); errs.append(0)
                else:
                    q = sa[(sa.model == m) & (sa.method == mt) & (sa.fmt == "w4pc")]
                    vals.append(100 * q.flip_rate_mean.iloc[0] if len(q) else np.nan); errs.append(100 * np.nan_to_num(q.flip_rate_std.iloc[0]) if len(q) else 0)
            if not np.all(np.isnan(vals)): ax.bar(xs + (j - 1) * w, vals, w, yerr=errs, capsize=2, label=mt.upper())
        ax.set_xticks(xs); ax.set_xticklabels([NAME[m] for m in MODELS], fontsize=7); ax.set_ylabel("Flip rate, W4-pc (%)"); ax.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(f"{R}/fig_methods.pdf"); plt.savefig(f"{R}/fig_methods.png", dpi=150); plt.close()
    base = S("protectai", "fp32"); p = S("protectai", "w4pc")
    fig, ax = plt.subplots(figsize=(3.4, 3.2)); fl = (p >= .5) != (base >= .5)
    ax.scatter(base[~fl], p[~fl], s=3, c="0.7", label="same decision")
    mm = fl & (y == 1) & (base >= .5); ax.scatter(base[mm], p[mm], s=8, c="tab:red", label="attack now missed")
    mm = fl & ~((y == 1) & (base >= .5)); ax.scatter(base[mm], p[mm], s=8, c="tab:blue", label="other flip")
    ax.axhline(.5, ls="--", c="k", lw=.6); ax.axvline(.5, ls="--", c="k", lw=.6); ax.set_xlabel("FP32 attack probability"); ax.set_ylabel("W4-pc (RTN) attack probability")
    ax.legend(fontsize=6, loc="upper left"); plt.tight_layout(); plt.savefig(f"{R}/fig_scatter.pdf"); plt.savefig(f"{R}/fig_scatter.png", dpi=150); plt.close()
except Exception as e:
    import traceback; traceback.print_exc()

metas = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(f"{ROOT}/meta/*.json")}
json.dump({"n": int(N), "n_clean": int(CLEAN.sum()), "provenance": PROV, "meta": metas}, open(f"{R}/summary.json", "w"), indent=1, default=str)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
print(res[["model", "cfg", "baseline", "source", "auc", "f1", "flip_rate", "flip_lo", "flip_hi", "missed", "new_fp", "mcnemar_p", "evade_homoglyph"]].round(4).to_string())
print(pd.DataFrame(ag).round(4).to_string())
if agg: print(pd.DataFrame(agg)[["model", "method", "fmt", "n_seeds", "flip_rate_mean", "flip_rate_std", "missed_mean", "missed_std", "flipset_jaccard_mean", "jaccard_vs_rtn_mean"]].round(4).to_string())
