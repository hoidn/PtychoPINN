# Phase D3b 60-Epoch gs2_ideal Retrain — H-NEPOCHS Validation

**Date:** 2026-01-20T14:05:31Z
**Focus:** DEBUG-SIM-LINES-DOSE-001 — Phase D3b
**Branch:** paper

---

## Executive Summary

**H-NEPOCHS: NOT CONFIRMED** — Extending training from 5 epochs to 30 epochs (early stopping triggered at 30/60) produced **no meaningful improvement** in amplitude metrics. The amplitude bias remains ~2.3× undershoot regardless of training length.

---

## Experiment Details

### Configuration
- Scenario: `gs2_ideal` (gridsize=2, idealized probe)
- Requested epochs: 60
- Actual epochs: 30 (early stopping triggered via `patience=3` on `loss`)
- Group limit: 64
- Prediction scale: `least_squares`

### Training History Summary
- Epochs completed: 30
- Final loss: -3,929,574.5
- Learning rate schedule: 0.001 → 0.0001 (ReduceLROnPlateau)
- Training NaN: **No**
- val_loss: -4,441,511.5 (converged by epoch 20)

---

## Metric Comparison: 5-epoch vs 30-epoch

| Metric | 5-epoch | 30-epoch | Delta |
|--------|---------|----------|-------|
| **MAE** | 2.3646 | 2.3774 | +0.0127 |
| **pearson_r** | 0.1353 | 0.1392 | +0.0039 |
| **RMSE** | 2.5576 | 2.5626 | +0.0050 |
| **max_abs** | 4.7677 | 4.7677 | 0.0000 |

### Amplitude Bias
| Statistic | 5-epoch | 30-epoch |
|-----------|---------|----------|
| Mean | -2.2962 | -2.3064 |
| Median | -2.5225 | -2.5306 |

### Scaling Diagnostics
- Best-fit least-squares scalar: **1.797** (30-epoch)
- Scaled MAE improvement: **negligible** (2.377 → 2.377)

---

## H-NEPOCHS Verdict

**HYPOTHESIS REJECTED:** Training length is NOT the primary cause of the amplitude bias.

Evidence:
1. **6× more epochs (5→30) produced <1% metric change** — MAE delta +0.013, pearson_r delta +0.004
2. Loss converged by epoch ~20; early stopping confirmed training plateau
3. Amplitude bias mean: -2.30 (identical within noise to 5-epoch baseline)
4. Full normalization chain product: **18.571** (vs ideal 1.0) — symmetry violation persists

### Implication
The amplitude undershoot is caused by:
- Normalization/loss wiring (grouped→normalized drops 42% of amplitude)
- Model architecture or loss composition (prediction→truth ratio ~6.7×)

NOT by:
- Insufficient training epochs
- Loss function weights (already ruled out in D1)

---

## Next Actions (D3c / D4)

1. **D3c:** Document these findings in `docs/findings.md` (H-NEPOCHS ruled out)
2. **D4:** Investigate architecture/loss wiring — the ~6.7× prediction→truth gap is the primary suspect
   - Examine how `pred_intensity_loss` is computed vs ground truth
   - Check if intensity scaler inverse is correctly applied during inference
   - Compare with dose_experiments loss composition

---

## Artifacts

```
plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T140531Z/
├── gs2_ideal_nepochs60/
│   ├── train_outputs/
│   │   ├── history.json (30 epochs)
│   │   ├── history_summary.json
│   │   └── weights.*.h5
│   ├── inference_outputs/
│   │   ├── amplitude.npy
│   │   ├── amplitude.png
│   │   ├── phase.npy
│   │   └── stats.json
│   ├── comparison_metrics.json
│   ├── run_metadata.json
│   ├── intensity_stats.json
│   └── runner.log
├── bias_summary.json
├── bias_summary.md
├── analyze_intensity_bias.log
├── pytest_cli_smoke.log
└── summary.md (this file)
```

---

## Guard Selector

```bash
pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
# Result: 1 passed
```

---

### Turn Summary
Extended gs2_ideal training from 5 to 30 epochs (early stop); metrics unchanged (MAE 2.36→2.38, pearson_r 0.135→0.139).
H-NEPOCHS hypothesis REJECTED — training length is not the cause of amplitude bias; the ~6.7× prediction→truth gap persists regardless of epochs.
Next: proceed to D4 architecture/loss wiring investigation to identify why the model undershoots amplitude by ~6×.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T140531Z/ (gs2_ideal_nepochs60/, bias_summary.md)
