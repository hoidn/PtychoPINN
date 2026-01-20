# Phase D3 Hyperparameter Audit — Analysis Summary

**Date:** 2026-01-20T133807Z
**Focus:** DEBUG-SIM-LINES-DOSE-001 — Hyperparameter delta audit
**Spec reference:** `specs/spec-ptycho-workflow.md:67` (training runs must emit histories + metrics)

---

## Key Findings

### Training Length Divergence (Critical)

| Parameter | dose_experiments | sim_lines | Delta |
|-----------|------------------|-----------|-------|
| **nepochs** | 60 | 5 | **-55 (12× shorter)** |

**Implication:** The sim_lines runs train for only 5 epochs compared to the legacy 60-epoch training regime. This **12× reduction in training length** is a plausible explanation for the amplitude collapse observed in Phase D2 telemetry:

- At 5 epochs, the model may not have converged sufficiently to learn the correct amplitude scale
- The `normalize_gain` ratios (0.56 for gs1_ideal, 0.27 for dose_legacy_gs2) may reflect insufficient training rather than a fundamental normalization bug
- The prediction→truth amplitude gaps (~3-4×) could narrow substantially with extended training

### Matching Parameters (Verified)

| Parameter | dose_experiments | sim_lines | Status |
|-----------|------------------|-----------|--------|
| batch_size | 16 | 16 | ✓ Match |
| probe.trainable | False | False | ✓ Match |
| intensity_scale.trainable | True | True | ✓ Match |
| mae_weight | 0.0 | 0.0 | ✓ Match |
| nll_weight | 1.0 | 1.0 | ✓ Match |

**Conclusion:** Loss configuration and trainability settings are identical; these are **ruled out** as sources of amplitude bias.

---

## Recommended Next Actions

1. **Schedule retrain with nepochs=60:** Re-run gs2_ideal (or gs2_custom) with `nepochs=60` using the plan-local runner to test whether training length alone closes the amplitude gap.

2. **Capture training curves:** Ensure the retrain emits epoch-by-epoch loss/metric histories to visualize convergence behavior.

3. **Compare amplitude metrics:** If the 60-epoch run achieves comparable amplitude fidelity to legacy dose_experiments, the Phase D3 hypothesis (H-NEPOCHS) is confirmed.

4. **Document findings:** Update `docs/findings.md` with `H-NEPOCHS` if confirmed, noting that sim_lines 5-epoch runs are insufficient for production-quality reconstructions.

---

## Artifacts

- `hyperparam_diff.md` / `hyperparam_diff.json` — Full parameter comparison tables
- `dose_loss_weights.json` / `dose_loss_weights.md` — Legacy loss mode snapshots
- `legacy_params_cfg_defaults.json` — Framework defaults verification
- `compare_cli.log` — CLI execution log
- `pytest_cli_smoke.log` — Test guard verification (1 passed)
