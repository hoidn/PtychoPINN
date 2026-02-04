# CI Logger Integration Notes

This document provides guidance for integrating the PyTorch backend's logging capabilities with CI/CD workflows, including artifact handling, output suppression, and cleanup strategies.

## Background

As of Phase EB3 (ADR-003-BACKEND-API), the PyTorch training CLI (`ptycho_torch/train.py`) defaults to **MLflow logging**. For CI workflows that should avoid external services, use `--logger csv` to capture training and validation metrics (loss values, learning rates, etc.) to disk without requiring MLflow or TensorBoard.

For full context on logger configuration and policy decisions, see:
- **Policy**: `docs/findings.md#CONFIG-LOGGER-001`
- **Spec**: `specs/ptychodus_api_spec.md` §4.9 (PyTorchExecutionConfig.logger_backend field)
- **Smoke Evidence**: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/smoke/2025-10-24T050500Z/`

## CSV Logger Artifact Handling

### Default Behavior

When `--logger csv` is specified (recommended for CI):

```bash
python -m ptycho_torch.train \
  --train_data_file data.npz \
  --output_dir outputs/experiment_1 \
  --logger csv
```

Lightning creates the following directory structure:

```
outputs/experiment_1/
└── lightning_logs/
    └── version_0/
        ├── hparams.yaml       # Hyperparameters snapshot
        └── metrics.csv        # Training/validation metrics
```

### CI Artifact Attachment

To preserve training metrics for post-hoc analysis or regression tracking:

1. **Capture CSV artifacts**:
   ```bash
   # After training completes successfully
   cp outputs/experiment_1/lightning_logs/version_0/metrics.csv ci_artifacts/metrics_${CI_JOB_ID}.csv
   ```

2. **Upload to CI artifact storage** (example for GitHub Actions):
   ```yaml
   - name: Archive metrics
     if: always()  # Capture even on failure
     uses: actions/upload-artifact@v3
     with:
       name: training-metrics
       path: ci_artifacts/metrics_*.csv
       retention-days: 30
   ```

3. **Optional: Parse and report key metrics**:
   ```bash
   # Extract final validation loss from CSV
   python -c "
   import pandas as pd
   df = pd.read_csv('ci_artifacts/metrics_${CI_JOB_ID}.csv')
   val_loss = df['poisson_val_loss'].dropna().iloc[-1]
   print(f'Final validation loss: {val_loss:.2f}')
   "
   ```

### CSV Format Details

The `metrics.csv` file uses standard Lightning CSV format:
- **Header row**: Comma-separated metric names (`epoch`, `learning_rate`, `poisson_train_loss_step`, `poisson_val_loss`, etc.)
- **Data rows**: Mix of step-level (every batch) and epoch-level (aggregated) values
- **Parsing**: Compatible with pandas (`pd.read_csv()`), NumPy, Excel, plotting libraries

For a concrete example, see the smoke test evidence at:
`plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/smoke/2025-10-24T050500Z/metrics.csv`

## Suppressing Logs in CI (Smoke Tests)

For quick smoke tests or validation runs where metrics aren't needed:

### Recommended: Disable Logging + Suppress Progress

```bash
python -m ptycho_torch.train \
  --train_data_file data.npz \
  --output_dir /tmp/smoke \
  --logger none \
  --quiet
```

**Effect**:
- `--logger none`: Disables Lightning logger entirely (metrics from `self.log()` calls are discarded)
- `--quiet`: Suppresses progress bars and reduces console verbosity

**Result**: Minimal console output, no `lightning_logs/` directory created, faster execution (no I/O overhead)

### Deprecated: Legacy `--disable_mlflow` Flag

**WARNING**: The `--disable_mlflow` flag is **deprecated** as of Phase EB3 and will emit a `DeprecationWarning`:

```bash
# Old approach (still works but emits warning)
python -m ptycho_torch.train --disable_mlflow

# Warning message:
# DeprecationWarning: --disable_mlflow is deprecated. Use --logger none to disable logging and --quiet to suppress progress bars.
```

**Migration path**:
- Replace `--disable_mlflow` with `--logger none --quiet`
- Update CI scripts and documentation to use the modern flags
- The `--disable_mlflow` flag will be **removed in a future release** (post-ADR-003 Phase E)

## Cleanup Guidance

### Preventing Large `lightning_logs/` Accumulation

Training runs create a new `version_N/` subdirectory for each experiment. Without cleanup, this can accumulate large volumes of metric files.

**Recommended cleanup strategies**:

1. **After each training run** (if metrics not needed):
   ```bash
   rm -rf outputs/experiment_1/lightning_logs
   ```

2. **Periodic cleanup** (keep only recent versions):
   ```bash
   # Keep last 5 training runs, delete older versions
   cd outputs/experiment_1/lightning_logs
   ls -t | tail -n +6 | xargs rm -rf
   ```

3. **CI-specific cleanup** (delete after artifact upload):
   ```bash
   # In CI pipeline, after uploading metrics.csv
   rm -rf outputs/experiment_1/lightning_logs
   ```

### Checkpoint Directory Management

The PyTorch backend also saves model checkpoints under `{output_dir}/checkpoints/`. These are typically larger than CSV logs:

```
outputs/experiment_1/
├── checkpoints/
│   └── last.ckpt         # ~50-200 MB depending on model size
├── lightning_logs/       # ~1-10 KB (metrics CSV)
└── wts.h5.zip            # Final model bundle
```

**Cleanup strategy**:
- Keep `wts.h5.zip` (final model bundle, needed for inference)
- Delete `checkpoints/` after training completes successfully (unless resuming training)
- Preserve `lightning_logs/` if metrics needed for analysis

Example:
```bash
# After successful training
rm -rf outputs/experiment_1/checkpoints  # Remove large intermediate checkpoints
# Keep wts.h5.zip and lightning_logs/ for analysis
```

## TensorBoard and MLflow Backends (Optional)

### TensorBoard Backend

For richer visualization capabilities:

```bash
python -m ptycho_torch.train \
  --logger tensorboard
```

**Requirements**:
- TensorBoard package (auto-installed via TensorFlow dependency)
- No additional configuration needed

**Artifact location**: `{output_dir}/lightning_logs/version_N/`

**Visualization**:
```bash
tensorboard --logdir outputs/experiment_1/lightning_logs/
```

**CI Integration**: Upload the entire `lightning_logs/` directory as an artifact for offline visualization.

### MLflow Backend

For experiment tracking with MLflow server:

```bash
python -m ptycho_torch.train \
  --logger mlflow
```

**Requirements**:
- `mlflow` package (optional dependency, must install separately)
- MLflow tracking server URI configured (see MLflow documentation)

**Status**: As of Phase EB3, the PyTorch backend uses legacy `mlflow.pytorch.autolog()` (see `ptycho_torch/train.py:75-80`). Migration to Lightning's `MLFlowLogger` is tracked as Phase EB3.C4 backlog.

**CI Integration**: Requires MLflow server setup; not recommended for basic CI smoke tests.

## Summary: Recommended CI Patterns

| Use Case | Command | Artifacts Generated | Cleanup |
|----------|---------|---------------------|---------|
| **Full training with metrics** | `--logger csv` | `lightning_logs/version_0/metrics.csv` | Keep CSV, delete checkpoints after upload |
| **Smoke test (no metrics needed)** | `--logger none --quiet` | None | No cleanup needed |
| **Rich visualization (local)** | `--logger tensorboard` | TensorBoard event files | Upload entire `lightning_logs/` dir |
| **Experiment tracking (production)** | `--logger mlflow` | MLflow server entries | Server-side retention policy |

## Cross-References

- **Logger backend field spec**: `specs/ptychodus_api_spec.md` §4.9 (PyTorchExecutionConfig.logger_backend)
- **CLI flag documentation**: `docs/workflows/pytorch.md` §12 (Training Execution Flags, `--logger` row)
- **Policy decision**: `docs/findings.md#CONFIG-LOGGER-001` (MLflow default, allowed backends, deprecation policy)
- **Smoke test evidence**: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/smoke/2025-10-24T050500Z/summary.md`
- **Implementation plan**: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md` (Phase EB3)

---

*Last updated: 2026-01-31 (MLflow default update)*
