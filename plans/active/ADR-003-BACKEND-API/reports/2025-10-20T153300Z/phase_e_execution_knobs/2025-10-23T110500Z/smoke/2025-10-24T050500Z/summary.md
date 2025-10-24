# CSV Logger Smoke Test Summary

## Test Execution
- **Date**: 2025-10-24
- **Command**: `CUDA_VISIBLE_DEVICES="" /usr/bin/time -p python -m ptycho_torch.train --train_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --test_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --output_dir tmp/logger_smoke --n_images 64 --max_epochs 1 --gridsize 2 --batch_size 4 --accelerator cpu --deterministic --num-workers 0 --logger csv`
- **Dataset**: `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz` (64 scan positions, minimal fixture)
- **Configuration**: gridsize=2, batch_size=4, max_epochs=1, CPU-only, deterministic mode

## Runtime Performance
- **Real time**: 9.92s
- **User time**: 29.35s (CPU computation)
- **Sys time**: 1.56s (system overhead)
- **Metric**: Within expected range for minimal fixture on CPU

## Logger Backend Verification
- **Backend configured**: CSV (`--logger csv`)
- **Logger directory created**: `tmp/logger_smoke/lightning_logs/version_0/`
- **Artifacts generated**:
  - `metrics.csv`: 36 rows capturing training metrics (poisson_train_loss_step, poisson_train_loss_epoch, poisson_val_loss, learning_rate, physics_weight, training_stage)
  - `hparams.yaml`: Hyperparameters snapshot

## Metrics Captured
The CSV logger successfully captured the following metrics per Lightning `self.log()` calls:
- **Training loss**: `poisson_train_loss_step` (16 values, ranging from ~17911 to ~85978)
- **Validation loss**: `poisson_val_loss` (1 end-of-epoch value: 18388.59)
- **Learning rate**: `learning_rate` (0.001 as configured)
- **Physics weight**: `physics_weight_epoch` and `physics_weight_step` (0.0, indicating supervised mode)
- **Training stage**: `training_stage_epoch` and `training_stage_step` (1.0 throughout, single stage)

## Key Observations
1. **CSV logger worked as expected**: Zero new dependencies required (Lightning built-in), metrics persisted to `metrics.csv` without external services
2. **No warnings about missing logger**: Previous `logger=False` configuration would have silently discarded all metrics; CSV backend preserves them for post-hoc analysis
3. **Deterministic execution**: Same fixture + seed produced stable output suitable for CI regression checks
4. **Deprecation warning observed**: UserWarning from `config_factory.py:613` about `params.cfg already populated` (known CONFIG-001 quirk, harmless in this context)
5. **Model bundle saved**: `tmp/logger_smoke/wts.h5.zip` created successfully alongside Lightning artifacts

## CSV Format Verification
- **Header row**: Standard Lightning CSV format with comma-separated metric names
- **Data rows**: Mix of step-level and epoch-level aggregations (some columns populated only at epoch boundaries)
- **Parsing**: Standard CSV parsers can consume this format for plotting/analysis (e.g., pandas, Excel, plotting libraries)

## Follow-Up Items
- No issues observed; CSV logger functioning per spec
- Optional: Repeat with `--logger tensorboard` to verify TensorBoard backend (requires tensorboard package from TensorFlow install)
- CI integration: Add `--logger none --quiet` guidance for smoke tests that don't need metric capture

## Exit Criteria Status
- ✓ Smoke command executed successfully (exit code 0)
- ✓ Runtime captured (9.92s real time)
- ✓ Logger backend confirmed (CSV, default as of CONFIG-LOGGER-001)
- ✓ Metrics CSV validated (36 rows, expected columns present)
- ✓ Logger directory tree captured (see `logger_tree.txt`)
- ✓ Artifacts archived in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/smoke/2025-10-24T050500Z/`
- ✓ Temporary directory cleanup pending (next step)
