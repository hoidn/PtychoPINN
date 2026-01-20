# Phase D1a-D1c: Loss Config Validation — Corrected Evidence

**Date:** 2026-01-20T112029Z
**Focus:** DEBUG-SIM-LINES-DOSE-001 — Phase D1 loss configuration parity

## Key Finding

**The prior Phase D1 diff (2026-01-20T110227Z) misrepresented the legacy loss weights.**

The previous parsing captured `mae_weight=1.0, nll_weight=0.0` from the legacy `dose_experiments_param_scan.md` script, which appeared to show a MAE/NLL inversion versus sim_lines_4x. However, those values are **conditional assignments** that only execute when `loss_fn='mae'`:

```python
if loss_fn == 'mae':
    cfg['mae_weight'] = 1.
    cfg['nll_weight'] = 0.
elif loss_fn == 'nll':
    pass  # Keep the current behavior
```

When `loss_fn='nll'` (the **default**), the legacy script does **not set** `mae_weight` or `nll_weight` — it relies on whatever defaults exist in the underlying framework (`ptycho.params.cfg`).

## Runtime-Captured Evidence

Executed the legacy `init()` function with a stubbed `ptycho.params.cfg` for both loss modes:

| Parameter | loss_fn='nll' (default) | loss_fn='mae' (override) |
|-----------|-------------------------|--------------------------|
| mae_weight | — (not set) | 1.0 |
| nll_weight | — (not set) | 0.0 |
| realspace_weight | — (not set) | — (not set) |
| realspace_mae_weight | — (not set) | — (not set) |

## Interpretation for H-LOSS-WEIGHT Hypothesis

The corrected evidence shows:

1. **dose_experiments (loss_fn='nll'):** Does NOT explicitly set `mae_weight` or `nll_weight`. The framework defaults apply.

2. **sim_lines_4x (TrainingConfig):** Uses dataclass defaults `mae_weight=0.0, nll_weight=1.0` (pure NLL loss).

3. **Conclusion:** There is no MAE/NLL inversion between the pipelines under default operation. Both pipelines appear to rely on NLL loss when using default settings.

**H-LOSS-WEIGHT status:** Cannot be confirmed as root cause from static config analysis alone. The loss weights depend on the **framework defaults** in the legacy codebase, which were not captured in the stubbed cfg. The legacy `ptycho.params.cfg` would need to be inspected to determine if its defaults differ from `TrainingConfig`.

## Artifacts

- `dose_loss_weights.json` — Raw runtime cfg snapshots for both loss_fn modes
- `loss_config_diff.md` — Updated Markdown diff with corrected loss weights
- `loss_config_diff.json` — Updated JSON diff with `dose_loss_modes` section
- `pytest_cli_smoke.log` — CLI smoke guard (1 passed)

## Next Actions

1. **Investigate framework defaults:** Determine what `mae_weight`/`nll_weight` values the legacy `ptycho.params.cfg` uses by default (requires inspecting the production params module).

2. **Proceed to D2 (Normalization):** If loss weights cannot be confirmed as the cause, continue with normalization pipeline parity analysis.

3. **Consider D3 (Hyperparameters):** The `nepochs=60` in legacy vs potentially different values in sim_lines may also contribute to the amplitude bias.

---

### Turn Summary (Supervisor — 2026-01-20T112029Z)
Reopened Phase D1 after the reviewer showed the loss diff misread dose_experiments' conditional MAE assignments; plan, fix plan, and initiative summary now require capturing actual runtime cfg snapshots instead of assuming an inversion.
Documented the D1a–D1c tasks (capture both loss_fn branches, fix the comparison CLI, rerun the diff) and rewrote input.md so Ralph stubs ptycho.params, records both cfg snapshots, emits the new artifacts, and runs the CLI pytest guard.
Next: Ralph updates `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py`, regenerates the diff under the 2026-01-20T112029Z hub, and reports whether H-LOSS-WEIGHT still holds once the real loss weights are known.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T112029Z/ (summary.md)
