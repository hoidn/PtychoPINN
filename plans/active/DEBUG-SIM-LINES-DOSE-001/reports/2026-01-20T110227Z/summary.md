# Phase D1 Loss Config Diff — Summary

**Date:** 2026-01-20T110227Z
**Focus:** DEBUG-SIM-LINES-DOSE-001 Phase D1: Compare loss-function configuration between sim_lines_4x and dose_experiments

## Key Finding: Loss Weight Inversion

The comparison reveals a **critical loss configuration difference** between the two pipelines:

| Parameter | dose_experiments | sim_lines_4x | Delta |
|-----------|------------------|--------------|-------|
| `mae_weight` | **1.0** | **0.0** | -1.0 |
| `nll_weight` | **0.0** | **1.0** | +1.0 |
| `realspace_weight` | — | 0.0 | — |
| `realspace_mae_weight` | — | 0.0 | — |

**Interpretation:**
- **dose_experiments** uses **MAE loss** (`mae_weight=1.0, nll_weight=0.0`)
- **sim_lines_4x** uses **NLL loss** (`mae_weight=0.0, nll_weight=1.0`)

This is the **opposite configuration**. The MAE (Mean Absolute Error) loss directly supervises amplitude recovery, while NLL (Negative Log-Likelihood) loss optimizes for Poisson statistics of the diffraction patterns.

## Hypothesis Update

**H-LOSS-WEIGHT is now a PRIMARY SUSPECT** for the amplitude bias observed in sim_lines_4x reconstructions:
- MAE loss explicitly penalizes amplitude errors → better amplitude recovery
- NLL loss optimizes for diffraction pattern fit → may produce correct diffraction but scaled amplitudes

## Recommended Next Actions

1. **Experiment:** Rerun sim_lines_4x scenarios with `mae_weight=1.0, nll_weight=0.0` to match dose_experiments
2. **Compare:** Evaluate whether amplitude bias improves with MAE supervision
3. If successful, update TrainingConfig defaults or expose loss weights in the sim_lines pipeline

## Artifacts

- `loss_config_diff.md`: Full parameter comparison table for all 4 scenarios
- `loss_config_diff.json`: Structured diff with per-scenario deltas
- `pytest_cli_smoke.log`: pytest guard log (1 passed)

---

### Turn Summary
Phase D1 loss configuration diff is complete: dose_experiments uses MAE loss while sim_lines_4x uses NLL loss — this inversion is a primary suspect for the amplitude bias.
Verified the existing implementation in `compare_sim_lines_params.py` correctly extracts loss weights by instantiating TrainingConfig per scenario; artifacts already generated.
Next: D2 should test whether switching sim_lines to MAE loss (`mae_weight=1.0, nll_weight=0.0`) resolves the ~2.5× amplitude undershoot.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T110227Z/ (loss_config_diff.md, loss_config_diff.json, pytest_cli_smoke.log)
