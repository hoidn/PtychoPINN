# Phase D1 Complete: Loss Config Parity Confirmed

**Date:** 2026-01-20T170000Z
**Focus:** DEBUG-SIM-LINES-DOSE-001.D1 — Loss-config parity verification
**Acceptance:** AT-D1 (Loss weights match between legacy and modern pipelines)

## Key Finding

**H-LOSS-WEIGHT hypothesis RULED OUT** — The legacy `ptycho.params.cfg` framework defaults (defined at `ptycho/params.py:64`) match the modern `TrainingConfig` dataclass defaults exactly:

| Parameter | params.cfg default | TrainingConfig default | Match? |
|-----------|-------------------|------------------------|--------|
| mae_weight | 0.0 | 0.0 | ✓ |
| nll_weight | 1.0 | 1.0 | ✓ |
| realspace_weight | 0.0 | 0.0 | ✓ |
| realspace_mae_weight | 0.0 | 0.0 | ✓ |

Both pipelines use **pure NLL loss** (`mae_weight=0, nll_weight=1`) under default operation.

## Technical Details

### Prior Misinterpretation (Fixed)

The earlier Phase D1 analysis (2026-01-20T110227Z) incorrectly reported a MAE/NLL inversion because the comparison CLI captured **conditional** assignments from the legacy script:

```python
if loss_fn == 'mae':
    cfg['mae_weight'] = 1.
    cfg['nll_weight'] = 0.
elif loss_fn == 'nll':
    pass  # Keep the current behavior
```

These conditional values (`mae_weight=1, nll_weight=0`) only apply when explicitly calling `init(nphotons, loss_fn='mae')`, not during default operation.

### Solution Implemented

Extended `compare_sim_lines_params.py` with:

1. `capture_legacy_params_defaults()` — Deep-copies the actual `ptycho.params.cfg` dictionary to extract framework defaults without mutation (per CONFIG-001)
2. `--output-legacy-defaults` flag — Writes the captured defaults to a separate JSON file
3. `build_legacy_defaults_markdown()` — Generates a comparison table showing params.cfg vs TrainingConfig defaults

## Artifacts

- `legacy_params_cfg_defaults.json` — Framework defaults with match status
- `loss_config_diff.md` — Full comparison with "Legacy params.cfg Framework Defaults" section
- `loss_config_diff.json` — Structured diff with `legacy_params_cfg_defaults` node
- `dose_loss_weights.json` — Runtime cfg snapshots for both loss_fn modes
- `pytest_cli_smoke.log` — CLI smoke guard (1 passed)

## Impact on Investigation

With H-LOSS-WEIGHT ruled out, the remaining amplitude bias (~3-6x undershoot) must stem from:

- **H-NORMALIZATION:** Intensity normalization pipeline differences (likely suspect given prior stage-ratio analysis showing 44% drop at grouped→normalized)
- **H-TRAINING-PARAMS:** Hyperparameters (lr, epochs, batch size)
- **H-ARCHITECTURE:** Model architecture mismatch vs legacy

## Next Actions

- Proceed to **Phase D2** (normalization stage parity) to investigate where the remaining amplitude bias enters the workflow
- The loss_config_diff now definitively shows matching loss weights, closing Phase D1

## SPEC/ADR References

- `specs/spec-ptycho-core.md:86` — Normative loss composition
- `ptycho/params.py:64` — Legacy framework defaults (authoritative source)

---

### Turn Summary
Extended the loss-config comparison CLI with `--output-legacy-defaults` to capture and prove the legacy `ptycho.params.cfg` defaults match `TrainingConfig`; all four loss weights are identical (mae=0, nll=1, realspace=0, realspace_mae=0).
H-LOSS-WEIGHT is now definitively ruled out; remaining amplitude bias stems from normalization or other workflow differences.
Next: move to Phase D2 to instrument normalization stage parity.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T170000Z/ (legacy_params_cfg_defaults.json, loss_config_diff.md)
