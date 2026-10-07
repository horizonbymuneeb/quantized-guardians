"""Generate LaTeX tables and figures from results."""
import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = ""; P = "results/"; os.makedirs(P, exist_ok=True)
df = pd.read_parquet(D + "data/evalset.parquet"); y = df.label.values; src = df.src.values
r = pd.read_csv("results/results.csv")
NAME = {"protectai": "ProtectAI-v2", "deepset": "Deepset", "piguard": "PIGuard", "fmops": "FMOps"}
CN = {"fp32": "FP32", "bf16": "BF16", "w8a8": "W8A8", "w4g128": "W4-g128", "w4pc": "W4-pc"}
MODELS = [m for m in ["protectai", "piguard", "deepset", "fmops"] if m in set(r.model)]
CFGS = ["fp32", "bf16", "w8a8", "w4g128", "w4pc"]

# Table 1: aggregate metrics
L = [r"\begin{table*}[t]\centering\caption{Aggregate detection metrics on the clean pool (Deepset, Safe-Guard, Jailbreak-Cls; $n=1{,}516$) and over-defense on NotInject ($n=339$ benign). Threshold 0.5. Aggregate numbers barely move under quantization.}\label{tab:agg}",
     r"\begin{tabular}{llrrrrrr}\toprule", r"Detector & Format & AUROC & F1 & TPR & FPR & NotInject FPR & ECE \\ \midrule"]
for m in MODELS:
    for c in CFGS:
        q = r[(r.model == m) & (r.cfg == c)]
        if q.empty: continue
        q = q.iloc[0]
        L.append(f"{NAME[m] if c=='fp32' else ''} & {CN[c]} & {q.auc:.3f} & {q.f1:.3f} & {q.tpr:.3f} & {q.fpr:.3f} & {q.notinject_fpr:.3f} & {q.ece:.3f} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule\end{tabular}\end{table*}"
open(P + "tab_agg.tex", "w").write("\n".join(L))

# Table 2: instance-level instability
L = [r"\begin{table}[t]\centering\caption{Instance-level decision changes relative to the FP32 detector over all 2{,}755 inputs. \emph{Missed}: attacks the FP32 model flagged that the quantized model passes. \emph{New FP}: benign inputs newly flagged.}\label{tab:flip}",
     r"\setlength{\tabcolsep}{4pt}\begin{tabular}{llrrrr}\toprule", r"Det. & Fmt & Flip \% & Missed & New FP & max$|\Delta p|$ \\ \midrule"]
for m in MODELS:
    first = True
    for c in CFGS[1:]:
        q = r[(r.model == m) & (r.cfg == c)]
        if q.empty: continue
        q = q.iloc[0]
        L.append(f"{NAME[m] if first else ''} & {CN[c]} & {100*q.flip_rate:.2f} & {int(q.flip_to_benign)} & {int(q.flip_to_attack)} & {q.max_abs_dp:.3f} \\\\"); first = False
    L.append(r"\midrule")
L[-1] = r"\bottomrule\end{tabular}\end{table}"
open(P + "tab_flip.tex", "w").write("\n".join(L))

# Table 3: evasion
L = [r"\begin{table}[t]\centering\caption{Detection rate (\%) on 300 character-perturbed attacks per transform.}\label{tab:evade}",
     r"\setlength{\tabcolsep}{4pt}\begin{tabular}{llrrr}\toprule", r"Det. & Fmt & Homoglyph & Zero-width & Leet \\ \midrule"]
for m in MODELS:
    for c in CFGS:
        q = r[(r.model == m) & (r.cfg == c)]
        if q.empty: continue
        q = q.iloc[0]
        L.append(f"{NAME[m] if c=='fp32' else ''} & {CN[c]} & {100*q.evade_homoglyph_det:.1f} & {100*q.evade_zerowidth_det:.1f} & {100*q.evade_leet_det:.1f} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule\end{tabular}\end{table}"
open(P + "tab_evade.tex", "w").write("\n".join(L))

# Figure 1: score shift scatter for strongest detector
fig, axes = plt.subplots(1, 2, figsize=(7, 3.0))
m0 = MODELS[0]
base = np.load(D + f"scores/{m0}__fp32.npy")
for ax, c in zip(axes, ["w4g128", "w4pc"]):
    f = D + f"scores/{m0}__{c}.npy"
    if not os.path.exists(f): continue
    p = np.load(f)
    ax.scatter(base[y == 0], p[y == 0], s=3, alpha=.35, label="benign", c="tab:blue")
    ax.scatter(base[y == 1], p[y == 1], s=3, alpha=.35, label="attack", c="tab:red")
    ax.axhline(.5, c="k", lw=.6, ls="--"); ax.axvline(.5, c="k", lw=.6, ls="--")
    ax.set_xlabel("FP32 attack probability"); ax.set_title(f"{NAME[m0]}: {CN[c]}", fontsize=9)
axes[0].set_ylabel("Quantized attack probability"); axes[0].legend(fontsize=7, markerscale=3, loc="upper left")
plt.tight_layout(); plt.savefig(P + "fig_scatter.pdf"); plt.savefig(P + "fig_scatter.png", dpi=130)

# Figure 2: flip rate by detector & format
fig, ax = plt.subplots(figsize=(3.5, 2.5)); w = 0.2
cf = CFGS[1:]
for i, c in enumerate(cf):
    xs = [j for j, m in enumerate(MODELS) if not r[(r.model == m) & (r.cfg == c)].empty]
    vals = [100 * r[(r.model == MODELS[j]) & (r.cfg == c)].flip_rate.iloc[0] for j in xs]
    ax.bar(np.array(xs) + (i - 1.5) * w, vals, w, label=CN[c])
ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([NAME[m] for m in MODELS], fontsize=7)
ax.set_ylabel("Decision flip rate (%)"); ax.legend(fontsize=6)
plt.tight_layout(); plt.savefig(P + "fig_flip.pdf"); plt.savefig(P + "fig_flip.png", dpi=130)

# per-source missed attacks for W4-pc (strongest detector)
out = []
for m in MODELS:
    b = np.load(D + f"scores/{m}__fp32.npy") >= .5
    for c in CFGS[1:]:
        f = D + f"scores/{m}__{c}.npy"
        if not os.path.exists(f): continue
        q = np.load(f) >= .5
        for s in np.unique(src):
            k = src == s
            out.append({"model": m, "cfg": c, "src": s, "missed": int((b & ~q & (y == 1) & k).sum()), "newfp": int((~b & q & (y == 0) & k).sum()), "n": int(k.sum())})
pd.DataFrame(out).to_csv("results/per_source.csv", index=False)
print("ok")

e = pd.read_csv("results/results_ext.csv")
L = [r"\begin{table}[t]\centering\caption{Held-out flip rate (\%) relative to FP32 before and after threshold re-calibration, with McNemar $p$ for change in correctness (all inputs).}\label{tab:cal}",
     r"\setlength{\tabcolsep}{4pt}\begin{tabular}{llrrrr}\toprule", r"Det. & Fmt & $\tau^\star$ & @0.5 & @$\tau^\star$ & McNemar $p$ \\ \midrule"]
for m in MODELS:
    first = True
    for c in CFGS[1:]:
        q = e[(e.model == m) & (e.cfg == c)]
        if q.empty: continue
        q = q.iloc[0]
        pv = "$<$0.001" if q.mcnemar_p < 1e-3 else f"{q.mcnemar_p:.2f}"
        L.append(f"{NAME[m] if first else ''} & {CN[c]} & {q.t_star:.2f} & {100*q.flip_test_05:.2f} & {100*q.flip_test_cal:.2f} & {pv} \\\\"); first = False
    L.append(r"\midrule")
L[-1] = r"\bottomrule\end{tabular}\end{table}"
open(P + "tab_cal.tex", "w").write("\n".join(L))
