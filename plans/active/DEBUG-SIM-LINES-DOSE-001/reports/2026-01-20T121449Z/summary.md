### Turn Summary
Extended `compare_sim_lines_params.py` with `--output-dose-loss-weights-markdown` flag and regenerated all D1 artifacts with runtime cfg capture for both loss_fn modes.
Corrected the earlier misrepresentation: legacy dose_experiments does NOT set explicit loss weights under default NLL mode — it relies on `ptycho/params.cfg` defaults (`mae_weight=0.0, nll_weight=1.0`), matching sim_lines exactly.
Next: Run pytest CLI smoke guard and update docs/fix_plan.md with D1 closure evidence; pivot to D2 normalization instrumentation since H-LOSS-WEIGHT is now ruled out.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121449Z/ (loss_config_diff.{md,json}, dose_loss_weights.{json,md}, legacy_params_cfg_defaults.json)

---

# Phase D1 — Runtime Loss-Weight Capture (Corrected Evidence)

**Date:** 2026-01-20T121449Z
**Focus:** DEBUG-SIM-LINES-DOSE-001.D1a-D1c
**Guard Selector:** `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`

## Summary

Extended `bin/compare_sim_lines_params.py` with:
1. `--output-dose-loss-weights-markdown` flag to emit standalone Markdown loss-mode snapshots
2. Runtime cfg capture for both `loss_fn='nll'` (default) and `loss_fn='mae'` (conditional) modes
3. Framework defaults comparison (`ptycho/params.cfg` vs `TrainingConfig` dataclass)

Regenerated the comparison diff using the existing Phase A snapshots and captured the refreshed artifacts.

## Key Finding: H-LOSS-WEIGHT Ruled Out

The earlier D1 evidence (2026-01-20T112000Z) incorrectly showed an MAE/NLL inversion because the static parser captured the conditional `cfg['mae_weight']=1.0` / `cfg['nll_weight']=0.0` assignments from the `if loss_fn == 'mae':` branch as defaults.

**Corrected evidence (runtime captured):**

| Source | mae_weight | nll_weight | Notes |
|--------|------------|------------|-------|
| Legacy `dose_experiments` init() with `loss_fn='nll'` | — (not set) | — (not set) | Relies on framework defaults |
| Legacy `dose_experiments` init() with `loss_fn='mae'` | 1.0 | 0.0 | Explicit conditional assignment |
| `ptycho/params.cfg` framework defaults | 0.0 | 1.0 | Pure NLL loss |
| `TrainingConfig` dataclass defaults | 0.0 | 1.0 | Pure NLL loss |
| sim_lines_4x scenarios | 0.0 | 1.0 | Via TrainingConfig |

**Conclusion:** Both pipelines use identical loss weights under default operation (`mae_weight=0.0, nll_weight=1.0`). The amplitude bias (~2.3–2.7× undershoot) is NOT caused by loss configuration differences. **H-LOSS-WEIGHT is ruled out.**

## Remaining Hypotheses for Amplitude Bias

With loss weights confirmed identical, the remaining suspects are:
- **H-NORMALIZATION:** Intensity normalization pipeline introduces bias (Phase D2 in progress)
- **H-TRAINING-PARAMS:** Hyperparameters (lr, epochs, batch size) may differ
- **H-ARCHITECTURE:** Model/forward-path wiring may have changed

## Artifacts Generated

- `loss_config_diff.md` — Full parameter diff (sim_lines vs dose_experiments) with corrected loss weights
- `loss_config_diff.json` — Structured diff with dose_loss_modes section
- `dose_loss_weights.json` — Runtime cfg snapshots for both loss_fn modes
- `dose_loss_weights.md` — Markdown summary with conditional labels
- `legacy_params_cfg_defaults.json` — Framework defaults comparison

## Next Actions

- D1 is complete; H-LOSS-WEIGHT ruled out with corrected evidence
- Proceed to D2 normalization pipeline parity instrumentation
- Focus on tracing the grouped→normalized stage where ~40-70% amplitude drop occurs

## SPEC/ADR Alignment

- Per `specs/spec-ptycho-core.md`: Loss function composition uses weighted sum of MAE/NLL/realspace terms
- Per CONFIG-001 (`docs/debugging/QUICK_REFERENCE_PARAMS.md`): Default NLL weights live in `ptycho/params.cfg:64`
