# Quantized Guardians

Code and per-input scores for the paper:

**Quantized Guardians: Accuracy-Neutral Decision Churn in Compressed Prompt-Injection Detectors**
Muneeb Anjum (ORCID [0009-0001-2042-7656](https://orcid.org/0009-0001-2042-7656))
Preprint: [10.5281/zenodo.23210535](https://doi.org/10.5281/zenodo.23210535)

Compressing a prompt-injection detector can leave its F1 and AUROC almost unchanged while flipping which individual inputs it blocks. This repo measures that decision churn for four public detectors under FP32, BF16, simulated W8A8 INT8, and 4-bit round-to-nearest weight quantization (group-wise g128 and per-channel).

## Detectors

| Key | Hugging Face model |
|---|---|
| `protectai` | `protectai/deberta-v3-base-prompt-injection-v2` |
| `piguard` | `leolee99/PIGuard` |
| `deepset` | `deepset/deberta-v3-base-injection` |
| `fmops` | `fmops/distilbert-prompt-injection` |

## Data

`data/evalset.parquet` holds the 2,755 benchmark inputs plus 900 character-perturbed attacks (homoglyph, zero-width, leetspeak) used in the paper. Columns: `src`, `text`, `label` (1 = attack).

Sources: Deepset prompt-injections (test), Safe-Guard prompt injection (test, balanced 1,000 sample), Jailbreak-Classification (test), and NotInject. `scripts/build_eval.py` rebuilds the file from the raw downloads in `data/` with seed 0.

## Formats

| Config | What it does |
|---|---|
| `fp32` | Full-precision baseline |
| `bf16` | bfloat16 weights and activations |
| `w8a8` | Simulated dynamic INT8 for weights and activations (quantize-dequantize) |
| `w4g128` | 4-bit RTN weights, group size 128 |
| `w4pc` | 4-bit RTN weights, per output channel |

Note: INT8 and INT4 are simulated with fake quantization, not executed with native integer kernels.

## Reproduce

```bash
pip install -r requirements.txt

# score one detector under one format (writes scores/<model>__<config>.npy)
python scripts/run_eval.py protectai w4pc

# all metrics: flip rates, McNemar tests, AUROC/F1, recalibration
python scripts/analyze.py

# tables and figures
python scripts/make_tables.py
```

`scores/` already contains the per-input attack probabilities used in the paper, so `analyze.py` and `make_tables.py` run without re-scoring. `results/` holds the computed metrics.

Everything runs on CPU. The full grid takes a few hours.

## Citation

```bibtex
@misc{anjum2026quantized,
  author = {Anjum, Muneeb},
  title  = {Quantized Guardians: Accuracy-Neutral Decision Churn in Compressed Prompt-Injection Detectors},
  year   = {2026},
  publisher = {Zenodo},
  doi    = {10.5281/zenodo.23210535}
}
```

## License

Code: MIT. The benchmark texts in `data/` keep the licenses of their original datasets.
