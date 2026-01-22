# D6 Training Label Telemetry - 2026-01-22T021500Z

## Summary

Implemented `record_training_label_stats()` extension to capture Y_amp (amplitude), Y_I (intensity), Y_phi (phase), X (input), and Y (complex) statistics for training label analysis per parity_logging_spec.md.

## Changes Made

1. **Extended `record_training_label_stats()`** (`bin/run_phase_c2_scenario.py:335-397`):
   - Added explicit `Y_amp` output (container._Y_I_np = ground truth amplitude)
   - Computed `Y_I` as intensity (Y_amp^2) for NLL loss comparison
   - Added structured notes to each stat block
   - Maintained backward compatibility with existing fields

2. **Updated `write_intensity_stats_outputs()` label_vs_truth_analysis** (`bin/run_phase_c2_scenario.py:626-672`):
   - Changed primary comparison from Y_I to Y_amp
   - Added `ratio_truth_to_Y_amp_mean` for direct amplitude parity check
   - Added `amplitude_gap_pct` for quick assessment
   - Added `sqrt_Y_I_mean` and `ratio_truth_to_sqrt_Y_I` for intensity→amplitude comparison

## Blocker: Keras 3 API Incompatibility

**Training run blocked** by Keras 3 API change in `ptycho/tf_helper.py`:

```
AttributeError: module 'keras._tf_keras.keras.metrics' has no attribute 'mean_absolute_error'
```

The D6a fix (setting `realspace_weight=0.1`) causes `realspace_loss()` to invoke `complex_mae()`, which references the deprecated `tf.keras.metrics.mean_absolute_error` API.

**Root cause:** Keras 3 moved `mean_absolute_error` from `keras.metrics` to `keras.losses`. The code at `ptycho/tf_helper.py:1476` needs to be updated to use `tf.keras.losses.MeanAbsoluteError()` or `tf.keras.losses.mean_absolute_error`.

**Cannot fix:** Per CLAUDE.md directive #6, `ptycho/tf_helper.py` is a core module and cannot be modified without explicit authorization.

**Workaround options:**
1. Revert `realspace_weight` to 0 (loses D6a amplitude supervision fix)
2. Fix the Keras 3 API in `ptycho/tf_helper.py` (requires explicit authorization)

## Artifacts

- `logs/pytest_cli_smoke_collect.log` - Test collection verification (1 test collected)
- `logs/pytest_cli_smoke.log` - Smoke test pass (1 passed)
- `logs/gs1_ideal_runner.log` - Training run failure log

## Test Results

| Selector | Result |
|----------|--------|
| `test_sim_lines_pipeline_import_smoke` | PASSED |

## Next Actions

1. Request authorization to fix `ptycho/tf_helper.py` Keras 3 API
2. Or revert D6a realspace_weight change to unblock training
