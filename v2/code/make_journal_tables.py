"""Generate LaTeX tables + numbers.tex for the journal/arXiv versions from local/results/*.csv (all values from real runs)."""
import pandas as pd, numpy as np, os, sys
R = "/workspace/work/qpaper/local/results"; OUT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/work/qpaper/journal"
os.makedirs(OUT, exist_ok=True)
res = pd.read_csv(f"{R}/results_all.csv"); F = res[res.baseline == "fp32"].copy()
nat = pd.read_csv(f"{R}/native_vs_sim.csv") if os.path.getsize(f"{R}/native_vs_sim.csv") > 2 else pd.DataFrame()
sa = pd.read_csv(f"{R}/seed_agg.csv") if os.path.getsize(f"{R}/seed_agg.csv") > 2 else pd.DataFrame()
cal = pd.read_csv(f"{R}/recal.csv")
NAME = {"protectai": "ProtectAI-v2", "piguard": "PIGuard", "deepset": "Deepset", "fmops": "FMOps"}
CF = {"fp32": "FP32", "bf16": "BF16", "w8a8": "W8A8$^{s}$ (token)", "w8a8pt": "W8A8$^{s}$ (tensor)", "int8dyn": "INT8 PyTorch", "int8onnx": "INT8 ONNX RT", "int8dynpc": "INT8 PyTorch (per-ch.)", "int8onnxpc": "INT8 ONNX RT (per-ch.)",
      "w4g128": "W4-g128 RTN", "w4pc": "W4-pc RTN", "gptq_w4pc_s0": "W4-pc GPTQ (s0)", "awq_w4pc_s0": "W4-pc AWQ (s0)", "gptq_w4g128_s0": "W4-g128 GPTQ (s0)"}
ORDER = ["fp32", "bf16", "w8a8", "w8a8pt", "int8dyn", "int8onnx", "int8dynpc", "int8onnxpc", "w4g128", "w4pc", "gptq_w4pc_s0", "awq_w4pc_s0", "gptq_w4g128_s0"]
MODELS = ["protectai", "piguard", "deepset", "fmops"]
def f3(x): return f"{x:.3f}"
def pc(x, d=2): return f"{100*x:.{d}f}"
def pv(p): return "$<$0.001" if p < 1e-3 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
def rows_for(m):
    q = F[F.model == m]; return [q[q.cfg == c].iloc[0] for c in ORDER if (q.cfg == c).any()]

# Table: aggregate
L = [r"\begin{table*}[t]\centering\small", r"\caption{Aggregate metrics on the clean pool ($n{=}1{,}516$) and NotInject over-defence ($n{=}339$). F1 brackets: 95\% bootstrap CI (300 resamples). W8A8$^{s}$ = simulated (fake-quantized) INT8; INT8 PyTorch / ONNX RT = native integer kernels on CPU. RTN rows for W4 use round-to-nearest; GPTQ/AWQ rows use calibration seed 0 (all seeds in Table~\ref{tab:gptq}).}\label{tab:agg}",
     r"\begin{tabular}{llccccc}\toprule Detector & Format & AUROC & F1 [95\% CI] & TPR & FPR & NotInject FPR\\\midrule"]
for m in MODELS:
    rs = rows_for(m)
    for i, r in enumerate(rs):
        L.append(f"{NAME[m] if i == 0 else ''} & {CF[r.cfg]} & {f3(r.auc)} & {f3(r.f1)} [{f3(r.f1_lo)}, {f3(r.f1_hi)}] & {f3(r.tpr)} & {f3(r.fpr)} & {f3(r.notinject_fpr)}\\\\")
    L.append(r"\midrule" if m != MODELS[-1] else r"\bottomrule")
L += [r"\end{tabular}\end{table*}"]; open(f"{OUT}/tab_agg.tex", "w").write("\n".join(L))

# Table: flips
L = [r"\begin{table*}[t]\centering\small", r"\caption{Instance-level change relative to the same detector's FP32 decisions on all 2{,}755 inputs. Flip CI: 95\% paired bootstrap (1{,}000 resamples). Missed: attacks blocked by FP32 but passed after quantization. New FP: benign inputs newly blocked. $b/c$: inputs that became wrong / right; $p$: exact McNemar test.}\label{tab:flip}",
     r"\begin{tabular}{llcccccc}\toprule Detector & Format & Flip \% [95\% CI] & Missed & New FP & $b/c$ & McNemar $p$ & max$|\Delta p|$\\\midrule"]
for m in MODELS:
    rs = [r for r in rows_for(m) if r.cfg != "fp32"]
    for i, r in enumerate(rs):
        L.append(f"{NAME[m] if i == 0 else ''} & {CF[r.cfg]} & {pc(r.flip_rate)} [{pc(r.flip_lo)}, {pc(r.flip_hi)}] & {r.missed} & {r.new_fp} & {r.mcnemar_worse}/{r.mcnemar_better} & {pv(r.mcnemar_p)} & {r.max_abs_dp:.2f}\\\\")
    L.append(r"\midrule" if m != MODELS[-1] else r"\bottomrule")
L += [r"\end{tabular}\end{table*}"]; open(f"{OUT}/tab_flip.tex", "w").write("\n".join(L))

# Table: adversarial slices
L = [r"\begin{table}[t]\centering\small", r"\caption{Detection rate (\%) on character-perturbed attack slices (300 attacks each).}\label{tab:evade}",
     r"\begin{tabular}{llccc}\toprule Detector & Format & Homoglyph & Zero-width & Leet\\\midrule"]
for m in MODELS:
    rs = [r for r in rows_for(m) if r.cfg in ("fp32", "bf16", "w8a8", "int8dyn", "int8onnx", "w4g128", "w4pc", "gptq_w4pc_s0")]
    for i, r in enumerate(rs):
        L.append(f"{NAME[m] if i == 0 else ''} & {CF[r.cfg]} & {pc(r.evade_homoglyph,1)} & {pc(r.evade_zerowidth,1)} & {pc(r.evade_leet,1)}\\\\")
    L.append(r"\midrule" if m != MODELS[-1] else r"\bottomrule")
L += [r"\end{tabular}\end{table}"]; open(f"{OUT}/tab_evade.tex", "w").write("\n".join(L))

# Table: native vs simulated
if len(nat):
    q = nat[nat.sim.isin(["w8a8", "w8a8pt"])]
    L = [r"\begin{table}[t]\centering\small", r"\caption{Native integer execution versus simulated INT8. Flip rates are measured against each path's own FP32 reference (PyTorch FP32 for INT8 PyTorch; ONNX Runtime FP32 for INT8 ONNX RT). $J$: Jaccard overlap of the two flip sets; Agree: fraction of identical final decisions between the native and simulated models.}\label{tab:native}",
         r"\begin{tabular}{lllcccc}\toprule Detector & Native & Simulated & Flip$_{nat}$ \% & Flip$_{sim}$ \% & $J$ & Agree \%\\\midrule"]
    for m in MODELS:
        qq = q[q.model == m]
        for i, (_, r) in enumerate(qq.iterrows()):
            L.append(f"{NAME[m] if i == 0 else ''} & {CF[r.native]} & {CF[r.sim]} & {pc(r.flip_native)} & {pc(r.flip_sim)} & {r.flipset_jaccard:.2f} & {pc(r.decision_agree)}\\\\")
    L += [r"\bottomrule\end{tabular}\end{table}"]; open(f"{OUT}/tab_native.tex", "w").write("\n".join(L))

# Table: GPTQ/AWQ seeds
if len(sa):
    L = [r"\begin{table*}[t]\centering\small", r"\caption{Calibration-based 4-bit quantization versus RTN (W4 per-channel unless noted). GPTQ/AWQ values are mean $\pm$ sample std over calibration seeds (128 calibration prompts per seed drawn from training splits disjoint from the evaluation set). $J_{seed}$: mean pairwise Jaccard overlap of flip sets across seeds; $J_{RTN}$: mean overlap with the RTN flip set.}\label{tab:gptq}",
         r"\begin{tabular}{lllcccccc}\toprule Detector & Method & Seeds & Flip \% & Missed & New FP & F1 & $J_{seed}$ & $J_{RTN}$\\\midrule"]
    for m in MODELS:
        r0 = F[(F.model == m) & (F.cfg == "w4pc")]
        qq = sa[(sa.model == m)]
        if not len(qq): continue
        if len(r0):
            r0 = r0.iloc[0]; L.append(f"{NAME[m]} & RTN & -- & {pc(r0.flip_rate)} & {r0.missed} & {r0.new_fp} & {f3(r0.f1)} & -- & --\\\\")
        for _, r in qq.iterrows():
            sd = lambda k, s=1: (f"$\\pm${s*r[k+'_std']:.2f}" if r.n_seeds > 1 else "")
            fmt = "" if r.fmt == "w4pc" else " (g128)"
            jr = f"{r.jaccard_vs_rtn_mean:.2f}" if not pd.isna(r.get("jaccard_vs_rtn_mean", np.nan)) else "--"
            js = f"{r.flipset_jaccard_mean:.2f}" if r.n_seeds > 1 else "--"
            L.append(f" & {r.method.upper()}{fmt} & {r.n_seeds} & {pc(r.flip_rate_mean)}{sd('flip_rate',100)} & {r.missed_mean:.1f}{sd('missed')} & {r.new_fp_mean:.1f}{sd('new_fp')} & {f3(r.f1_mean)} & {js} & {jr}\\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"; L += [r"\end{tabular}\end{table*}"]; open(f"{OUT}/tab_gptq.tex", "w").write("\n".join(L))

# Table: recalibration
L = [r"\begin{table}[t]\centering\small", r"\caption{Held-out flip rate (\%) at the default threshold and after re-tuning $\tau$ to maximize agreement with FP32 on a disjoint half; mean $\pm$ std over 5 random 50/50 splits. $\tau^\star$: range of selected thresholds.}\label{tab:cal}",
     r"\begin{tabular}{llccc}\toprule Detector & Format & $\tau{=}0.5$ & $\tau^\star$ & $\tau^\star$ range\\\midrule"]
for m in MODELS:
    qq = cal[(cal.model == m) & cal.cfg.isin(["w8a8", "int8dyn", "int8onnx", "w4g128", "w4pc", "gptq_w4pc_s0"])]
    qq = qq.set_index("cfg").reindex([c for c in ORDER if c in set(qq.cfg)]).reset_index()
    for i, (_, r) in enumerate(qq.iterrows()):
        L.append(f"{NAME[m] if i == 0 else ''} & {CF[r.cfg]} & {pc(r.flip05_mean)}$\\pm${pc(r.flip05_std)} & {pc(r.flipcal_mean)}$\\pm${pc(r.flipcal_std)} & {r.tstar_min:.2f}--{r.tstar_max:.2f}\\\\")
    L.append(r"\midrule" if m != MODELS[-1] else r"\bottomrule")
L += [r"\end{tabular}\end{table}"]; open(f"{OUT}/tab_cal.tex", "w").write("\n".join(L))
print("tables written to", OUT)
