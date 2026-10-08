# Quantized Guardians v2 (journal version)

Code, run metadata, per-input scores and results for the extended study
"Quantized Guardians: Accuracy-Neutral Decision Churn in Compressed Prompt-Injection Detectors" (Muneeb Anjum).

- `code/` analysis, diagnostics and run scripts (local aarch64 CPU runs)
- `ci/` GitHub Actions workflow and runner used for x86_64 CPU runs
- `scores/` per-input detector scores (`<model>__<config>.npy`; `_x86` = x86_64 native INT8 / FP32 replicates)
- `meta/` per-run metadata, logs, `lscpu`, `pip freeze`
- `results/`, `diag/` aggregate CSVs and diagnostic JSONs
- `paper/` journal preprint and IEEE/arXiv-format PDF

Hardware: local 2-vCPU aarch64 (Neoverse-V2) CPU and GitHub Actions ubuntu-24.04 x86_64 CPU runners.
The v1 (simulated-quantization) release remains in the repository root.
