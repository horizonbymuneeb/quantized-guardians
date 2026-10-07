"""Compute all paper metrics from saved scores."""
import numpy as np, pandas as pd, json, glob, os
from sklearn.metrics import roc_auc_score, f1_score
D = ""
df = pd.read_parquet(D + "data/evalset.parquet")
y = df.label.values; src = df.src.values
CLEAN = np.isin(src, ["deepset", "safeguard", "jailbreak_cls"])
CFGS = ["fp32", "bf16", "w8a8", "w4g128", "w4pc"]
MODELS = ["protectai", "deepset", "piguard", "fmops"]
def ece(p, yy, bins=10):
    e = 0; b = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(b[:-1], b[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum(): e += m.mean() * abs(p[m].mean() - yy[m].mean())
    return e
rows = []; flips = []
for mk in MODELS:
    f = D + f"scores/{mk}__fp32.npy"
    if not os.path.exists(f): continue
    base = np.load(f)
    for c in CFGS:
        f = D + f"scores/{mk}__{c}.npy"
        if not os.path.exists(f): continue
        p = np.load(f); pred = p >= 0.5
        r = {"model": mk, "cfg": c}
        r["auc"] = roc_auc_score(y[CLEAN], p[CLEAN])
        r["f1"] = f1_score(y[CLEAN], pred[CLEAN])
        r["tpr"] = pred[CLEAN & (y == 1)].mean()
        r["fpr"] = pred[CLEAN & (y == 0)].mean()
        r["notinject_fpr"] = pred[src == "notinject"].mean()
        for e in ["evade_homoglyph", "evade_zerowidth", "evade_leet"]:
            r[e + "_det"] = pred[src == e].mean()
        r["ece"] = ece(p[CLEAN], y[CLEAN])
        bp = base >= 0.5
        r["flip_rate"] = (pred != bp).mean()
        r["flip_to_benign"] = (bp & ~pred & (y == 1)).sum()  # attacks newly missed
        r["flip_to_attack"] = (~bp & pred & (y == 0)).sum()
        r["mean_abs_dp"] = np.abs(p - base).mean()
        r["max_abs_dp"] = np.abs(p - base).max()
        # threshold that restores fp32 TPR on clean attacks
        t_base = (base[CLEAN & (y == 1)] >= 0.5).mean()
        ths = np.linspace(0.01, 0.99, 99)
        ok = [t for t in ths if (p[CLEAN & (y == 1)] >= t).mean() >= t_base]
        r["thr_restore"] = max(ok) if ok else 0.01
        rows.append(r)
out = pd.DataFrame(rows)
out.to_csv("results/results.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(out.round(4).to_string())

# ---- Statistical test + recalibration mitigation ----
from scipy.stats import binomtest
rng = np.random.default_rng(0); cal = rng.random(len(y)) < 0.5
ext = []
for mk in MODELS:
    f = D + f"scores/{mk}__fp32.npy"
    if not os.path.exists(f): continue
    base = np.load(f); bp = base >= .5
    for c in CFGS[1:]:
        f = D + f"scores/{mk}__{c}.npy"
        if not os.path.exists(f): continue
        p = np.load(f); qp = p >= .5
        # McNemar exact test on correctness vs ground truth
        cb, cq = bp == (y == 1), qp == (y == 1)
        b_, c_ = int((cb & ~cq).sum()), int((~cb & cq).sum())
        pval = binomtest(b_, b_ + c_, 0.5).pvalue if b_ + c_ else 1.0
        # bootstrap CI of flip rate
        fl = (bp != qp).astype(float)
        bs = [fl[rng.integers(0, len(fl), len(fl))].mean() for _ in range(1000)]
        ths = np.linspace(0.01, 0.99, 197)
        agree = [((p[cal] >= t) == bp[cal]).mean() for t in ths]
        t_star = ths[int(np.argmax(agree))]
        flip_test_05 = ((p[~cal] >= .5) != bp[~cal]).mean()
        flip_test_cal = ((p[~cal] >= t_star) != bp[~cal]).mean()
        ext.append({"model": mk, "cfg": c, "mcnemar_b": b_, "mcnemar_c": c_, "mcnemar_p": pval,
                    "flip_lo": np.percentile(bs, 2.5), "flip_hi": np.percentile(bs, 97.5),
                    "t_star": t_star, "flip_test_05": flip_test_05, "flip_test_cal": flip_test_cal})
ext = pd.DataFrame(ext); ext.to_csv("results/results_ext.csv", index=False); print(ext.round(4).to_string())
