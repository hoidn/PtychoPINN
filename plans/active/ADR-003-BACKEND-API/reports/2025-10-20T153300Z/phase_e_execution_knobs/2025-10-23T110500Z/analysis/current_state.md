# Phase EB3.A1 — Current Logging State Audit

**Date:** 2025-10-23
**Initiative:** ADR-003-BACKEND-API Phase EB3 (Logger Governance)
**Task:** Catalogue existing logging hooks and legacy expectations

---

## Executive Summary

**Key Finding:** PyTorch backend currently has **MORE complete logging than TensorFlow baseline**, including active MLflow integration and comprehensive Python logging. Lightning logger is intentionally disabled pending Phase D governance decision.

---

## Current Logging State

### PyTorch Backend (`ptycho_torch/`)

#### 1. MLflow Integration (ACTIVE)

**Primary Integration Points:**
- `ptycho_torch/train.py:75-76` — Imports `mlflow.pytorch` and `MlflowClient`
- `ptycho_torch/train.py:147` — `disable_mlflow` parameter (default: `False`, allows opt-out)
- `ptycho_torch/train.py:306-307` — Autologging enabled:
  ```python
  mlflow.pytorch.autolog(checkpoint_monitor=val_loss_label)
  ```
- `ptycho_torch/train.py:312-325` — Experiment tracking:
  ```python
  mlflow.set_experiment(experiment_name)
  with mlflow.start_run(run_name=...):
      mlflow.log_params({...})  # Log all config parameters
      mlflow.set_tags({...})    # Add metadata tags
  ```

**Utility Functions:**
- `ptycho_torch/train_utils.py:22-23` — Imports `mlflow.pytorch` and `MlflowClient`
- `ptycho_torch/train_utils.py:90-105` — `log_parameters_mlflow()` function:
  - Logs DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig as JSON artifacts
  - Converts dataclass instances to dictionaries for MLflow storage
- `ptycho_torch/train_utils.py:180-207` — Fine-tuning tracking:
  - Creates separate MLflow run for fine-tuning phase
  - Adds stage/encoder_frozen tags for experiment organization

**Status:** Fully active and integrated at checkpoint monitoring level.

#### 2. Standard Python Logging (ACTIVE)

**Logger Initialization:**
- `ptycho_torch/utils.py:17` — Module-level logger:
  ```python
  logger = logging.getLogger(__name__)
  ```
- `ptycho_torch/workflows/components.py:80` — Same pattern

**Usage Locations:**
- `ptycho_torch/utils.py:56-92` — Config loading:
  - Warnings for missing parameters
  - Errors for YAML decode failures
- `ptycho_torch/workflows/components.py:151-187` — Training workflow:
  - Parameter synchronization logs
  - Training start/completion messages
  - Stitching and model persistence events
- `ptycho_torch/workflows/components.py:622-789` — Lightning trainer:
  - Configuration details
  - Training start/error/completion status

**Status:** Comprehensive logging across workflow orchestration and utilities.

#### 3. Lightning Logger Integration (INTENTIONALLY DISABLED)

**Critical Comment:**
- `ptycho_torch/workflows/components.py:760`:
  ```python
  logger=False  # Disable default logger for now; MLflow/TensorBoard added in Phase D
  ```

**Lightning Imports:**
- `ptycho_torch/train.py:65-67` — Callbacks imported:
  ```python
  from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
  from lightning.pytorch.strategies import DDPStrategy
  ```
- `ptycho_torch/train.py:237-258` — ModelCheckpoint and EarlyStopping configured

**Model Logging Calls:**
- `ptycho_torch/model.py:1248, 1260` — Lightning `self.log()` calls:
  ```python
  self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
  self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
  ```

**Status:** Lightning logger is disabled via `logger=False` in Trainer instantiation. Loss values logged via `self.log()` but not captured by any backend.

#### 4. CLI Flag Handling

**Current Implementation:**
- `ptycho_torch/cli/shared.py:97-117` — `--disable_mlflow` flag handling:
  ```python
  if args.disable_mlflow:
      # Maps to enable_progress_bar (semantic overload issue)
      exec_config_dict['enable_progress_bar'] = False
  ```

**Issue Identified:** Semantic overload - `--disable_mlflow` controls BOTH MLflow AND progress bar visibility. This violates separation of concerns.

---

### TensorFlow Baseline (`ptycho/`)

#### 1. TensorBoard Integration (ACTIVE)

**Primary Usage:**
- `ptycho/model.py:546-551` — TensorBoard callback configured:
  ```python
  tboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=logs,
      histogram_freq=1,
      profile_batch='500,520'
  )
  ```

**Status:** Active at model training level.

#### 2. Workflow-Level Logging (PLANNED BUT NOT IMPLEMENTED)

**TODO Comment:**
- `ptycho/workflows/components.py:749`:
  ```python
  # TODO Save training history with tensorboard / mlflow
  ```

**Status:** Unfulfilled intention to integrate experiment tracking at workflow level.

#### 3. Configuration Field (DEFINED BUT UNUSED)

**Config Definition:**
- `ptycho/config/config.py:242` — PyTorchExecutionConfig field:
  ```python
  logger_backend: Optional[str] = None
  # Experiment tracking backend: None, 'tensorboard', 'wandb', 'mlflow'
  ```

**Status:** Field exists but is not consumed by any TensorFlow code.

---

## Comparison Matrix

| Aspect | PyTorch Backend | TensorFlow Baseline | Status |
|--------|----------------|---------------------|--------|
| **Experiment Tracking** | MLflow (active, autologging) | Planned but not implemented (TODO comment) | PyTorch MORE complete |
| **Visualization** | Lightning logger disabled, pending Phase D | TensorBoard callback active | TensorFlow has UI |
| **Python Logging** | Comprehensive (utils, workflows) | Minimal/absent | PyTorch MORE complete |
| **Config Integration** | `PyTorchExecutionConfig.logger_backend` defined | Same field defined but unused | PyTorch ready for wiring |
| **CLI Flags** | `--disable_mlflow` (semantic overload) | No equivalent flag | PyTorch has control, but design flaw |
| **Loss Logging** | `self.log()` calls present but not captured | TensorBoard callback captures losses | TensorFlow functional |

---

## Legacy Expectations & Dependencies

### Existing MLflow Dependency

**Installation:**
- Already in `setup.py:50` as part of `[torch]` extras
- `pip install -e .[torch]` includes MLflow automatically
- **Grandfathered under POLICY-001** (introduced in Phase F)

### TensorBoard Dependency

**Installation:**
- Already satisfied via TensorFlow requirement (`setup.py:37`)
- `tensorflow[and-cuda]>=2.17.0` includes `tensorboard` package
- **No new dependency needed** for TensorBoardLogger

### Lightning Dependency

**Installation:**
- Already mandatory for PyTorch backend (Phase F)
- `lightning>=2.5` specified in `setup.py:49`
- All Lightning loggers are built-in (CSVLogger) or use existing deps (TensorBoard, MLflow)

---

## Critical Findings

### 1. Lightning Logger is Deliberately Disabled

**Evidence:** `ptycho_torch/workflows/components.py:760`
- Comment explicitly states "added in Phase D"
- This Phase EB3 IS "Phase D" per initiative timeline
- **Blocker removed** - we are now authorized to implement

### 2. Semantic Overload in `--disable_mlflow`

**Current Behavior:**
```python
# cli/shared.py:97-117
if args.disable_mlflow:
    exec_config_dict['enable_progress_bar'] = False
```

**Problem:** Single flag controls two orthogonal features:
1. MLflow experiment tracking (intended)
2. Progress bar visibility (unrelated)

**Recommendation:** Separate concerns:
- `--logger none` to disable experiment tracking
- `--quiet` to disable progress bar (already exists)
- Keep `--disable_mlflow` as deprecated alias for backward compat

### 3. PyTorch Has More Logging Than TensorFlow

**Observation:** PyTorch backend includes:
- Active MLflow integration (TF only has TODO comment)
- Comprehensive Python logging (TF minimal)
- Configuration parameter logging as JSON artifacts

**Implication:** PyTorch backend is already ahead of TensorFlow in experiment tracking. Enabling Lightning logger would further widen the gap.

### 4. Loss Metrics Are Not Persisted

**Evidence:**
- `ptycho_torch/model.py:1248, 1260` calls `self.log()` for train/val loss
- `ptycho_torch/workflows/components.py:760` disables Lightning logger (`logger=False`)
- **Result:** Loss values are computed but not persisted anywhere

**Impact:** Users cannot visualize training curves, only see final checkpoint metrics.

---

## Recommendations for Phase EB3.A2 (Options Analysis)

Based on this audit, the options matrix should prioritize:

1. **CSVLogger** (Tier 1):
   - Zero new dependencies (built-in to Lightning)
   - Captures loss values currently lost
   - CI-friendly (no service dependencies)
   - Aligns with POLICY-001

2. **TensorBoardLogger** (Tier 2):
   - No new dependencies (tensorboard already present via TF)
   - Provides feature parity with TensorFlow baseline
   - Industry standard UI
   - Requires separate server process (acceptable overhead)

3. **MLFlowLogger** (Tier 3):
   - Already have MLflow installed
   - Would replace manual `mlflow.pytorch.autolog()` with Lightning integration
   - Fix semantic overload by separating concerns
   - **Caution:** Heavyweight, requires server setup

4. **None** (Not Recommended):
   - Current state (loss metrics lost)
   - Maintains status quo but loses debugging capability

5. **WandB/Neptune** (Not Recommended):
   - Violate POLICY-001 (new mandatory dependencies)
   - Require governance review + user account setup

---

## File References

**PyTorch Implementation:**
- `ptycho_torch/train.py:75-80, 306-340` — MLflow integration
- `ptycho_torch/train_utils.py:90-105, 180-207` — Logging utilities
- `ptycho_torch/workflows/components.py:760` — Lightning logger disabled
- `ptycho_torch/model.py:1248, 1260` — Loss logging calls
- `ptycho_torch/cli/shared.py:97-117` — CLI flag handling

**TensorFlow Baseline:**
- `ptycho/model.py:546-551` — TensorBoard callback
- `ptycho/workflows/components.py:749` — TODO comment
- `ptycho/config/config.py:242` — Logger config field

**Dependencies:**
- `setup.py:37` — TensorFlow (includes tensorboard)
- `setup.py:49-50` — Lightning + MLflow in `[torch]` extras

---

**Next Steps:**
- EB3.A2: Build options matrix (prioritize CSVLogger, TensorBoardLogger)
- EB3.A3: Draft decision proposal recommending CSVLogger as MVP default
- Address semantic overload in `--disable_mlflow` flag
