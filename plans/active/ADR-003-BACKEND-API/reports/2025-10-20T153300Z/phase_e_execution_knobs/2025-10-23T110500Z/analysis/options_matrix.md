# Phase EB3.A2 — Lightning Logger Options Matrix

**Date:** 2025-10-23
**Initiative:** ADR-003-BACKEND-API Phase EB3 (Logger Governance)
**Task:** Evaluate Lightning logger options and dependency impact

---

## Executive Summary

**Recommendation:** Adopt **CSVLogger** as MVP default (Tier 1), with optional upgrade path to **TensorBoardLogger** (Tier 2) in future phase. Both options satisfy POLICY-001 (no new mandatory dependencies) and provide immediate value with minimal CI friction.

---

## Options Comparison Matrix

| Option | Dependencies | Pros | Cons | CI Impact | User Workflow Impact |
|--------|--------------|------|------|-----------|---------------------|
| **`None` (no logger)** | None (built-in) | - Zero dependency footprint<br>- Simplest implementation (`logger=False`)<br>- No external service dependencies<br>- Fast CI execution<br>- Already used in workflows/components.py:760 | - No experiment tracking<br>- No metric visualization<br>- No hyperparameter logging<br>- Loss of debugging capability<br>- Inconsistent with TensorFlow baseline (uses TensorBoard) | **BEST:** No impact, tests run normally | **NEGATIVE:** Users lose all logging visibility, must manually extract metrics from checkpoints |
| **`CSVLogger`** ⭐ | None (built-in to Lightning) | - **Zero external dependencies**<br>- Lightweight, filesystem-only<br>- Human-readable YAML + CSV format<br>- Easy to parse programmatically<br>- CI-friendly (no network/service)<br>- Version control compatible<br>- Works offline | - No visualization UI (need external tools)<br>- Limited query capabilities vs database<br>- Manual plotting required<br>- Not suitable for large-scale experiments | **BEST:** Zero CI impact, no new test infrastructure needed | **MINIMAL:** Users get CSV files in output_dir, can plot with pandas/matplotlib. No UI but programmatic access is simple |
| **`TensorBoardLogger`** | `tensorboard` package (already installed via TensorFlow in setup.py:34-36) | - **No NEW dependencies** (tensorboard already mandatory via TF)<br>- Rich visualization UI<br>- Industry standard for ML<br>- Supports scalars, images, histograms, embeddings<br>- Hyperparameter comparison<br>- Remote filesystem support (fsspec)<br>- Matches TensorFlow baseline (ptycho/model.py:549) | - Requires running separate `tensorboard` server process<br>- Binary log format (not human-readable)<br>- Heavier disk footprint than CSV<br>- Potential version conflicts between TF's tensorboard and Lightning's expectations | **GOOD:** No new deps, but tests may need `--logger=False` for hermetic execution. Could add optional `@pytest.mark.logger` tests | **MINIMAL:** Users run `tensorboard --logdir outputs/` to view metrics. Standard ML workflow, well-documented |
| **`MLFlowLogger`** | `mlflow` package (already in setup.py:50 `[torch]` extras) | - **Already installed** as part of `[torch]` extras<br>- Experiment tracking with versioning<br>- Model registry integration<br>- Hyperparameter search support<br>- REST API for queries<br>- Currently used in ptycho_torch/train.py:75-80 (hardcoded) | - **Requires MLflow server** (network dependency)<br>- Heavyweight (~50+ transitive dependencies)<br>- Complex setup for users (tracking URI, experiments)<br>- Hard failure mode if server unavailable<br>- CI requires `--disable_mlflow` flag (current pattern)<br>- Semantic overload with existing `--disable_mlflow` flag (see ADR-003) | **MODERATE:** Tests need `disable_mlflow=True` or mock MLflow client. Current pattern in test_cli_train_torch.py. Server unavailability causes test failures | **MODERATE:** Users must set up MLflow server or use local file store. More powerful but steeper learning curve. Current CLI already has `--disable_mlflow` flag |
| **`WandbLogger`** | `wandb>=0.12.10` (NOT installed) | - Cloud-hosted (no local server)<br>- Excellent UI/UX<br>- Team collaboration features<br>- Free tier available<br>- Hyperparameter sweeps<br>- Model versioning | - **New mandatory dependency** (violates POLICY-001)<br>- Requires internet + account<br>- Proprietary service dependency<br>- Privacy concerns (data leaves local system)<br>- No offline mode for core functionality<br>- Would need governance review | **POOR:** New dependency requires test infrastructure changes. Skip logic or mocking needed. Adds CI complexity | **HIGH FRICTION:** Users forced to create wandb account, authenticate, configure project. Not suitable for air-gapped environments |
| **`NeptuneLogger`** | `neptune>=1.0` (NOT installed) | - Similar to WandB (cloud tracking)<br>- Metadata versioning<br>- Team features | - **New mandatory dependency** (violates POLICY-001)<br>- Requires account + API key<br>- Less popular than WandB<br>- Similar privacy/connectivity concerns | **POOR:** Same as WandB - new dependency, authentication complexity in CI | **HIGH FRICTION:** Similar to WandB. Additional service to learn/configure |

⭐ = Recommended Tier 1 (MVP default)

---

## Dependency Installation Requirements

| Logger | Installation Command | Dependency Status | Notes |
|--------|---------------------|-------------------|-------|
| None | N/A | ✅ Built-in | No installation needed |
| **CSVLogger** ⭐ | N/A | ✅ Built-in to Lightning | No installation needed |
| TensorBoardLogger | N/A | ✅ Already satisfied | `tensorboard` installed via `tensorflow[and-cuda]` (setup.py:37) |
| MLFlowLogger | `pip install -e .[torch]` | ✅ Grandfathered | Already in project's `[torch]` extras (setup.py:50) |
| WandbLogger | `pip install wandb>=0.12.10` | ❌ NEW DEPENDENCY | **Not in setup.py, violates POLICY-001** |
| NeptuneLogger | `pip install neptune>=1.0` | ❌ NEW DEPENDENCY | **Not in setup.py, violates POLICY-001** |

---

## POLICY-001 Compliance Analysis

| Logger | Status | Rationale |
|--------|--------|-----------|
| None | ✅ **COMPLIANT** | No dependencies |
| **CSVLogger** ⭐ | ✅ **COMPLIANT** | Built-in to Lightning (which is already mandatory per Phase F) |
| TensorBoardLogger | ✅ **COMPLIANT** | Dependency already satisfied via TensorFlow requirement (`setup.py:37`) |
| MLFlowLogger | ⚠️ **GRANDFATHERED** | Already in `[torch]` extras since Phase F (INTEGRATE-PYTORCH-001), used in train.py:75-80 |
| WandbLogger | ❌ **VIOLATES** | Would require governance review + new mandatory dep |
| NeptuneLogger | ❌ **VIOLATES** | Would require governance review + new mandatory dep |

**Reference:** `docs/findings.md#POLICY-001` — "PyTorch (torch>=2.2) is now a mandatory dependency for PtychoPINN as of Phase F (INTEGRATE-PYTORCH-001)"

---

## Comparison to TensorFlow Baseline

### TensorFlow Current Implementation

**File:** `ptycho/model.py:546-549`
```python
tboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=logs,
    histogram_freq=1,
    profile_batch='500,520'
)
```

**Observation:** TensorFlow uses **TensorBoard** as the standard logging mechanism.

### PyTorch Current Implementation

**File:** `ptycho_torch/train.py:75-80, 306-340`
```python
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# ...

mlflow.pytorch.autolog(checkpoint_monitor=val_loss_label)
mlflow.set_experiment(experiment_name)
with mlflow.start_run(run_name=...):
    mlflow.log_params({...})
    mlflow.set_tags({...})
```

**Observation:** PyTorch uses **MLflow** (more heavyweight, different paradigm).

### Mismatch Analysis

1. **Different Logging Paradigms:**
   - TensorFlow: TensorBoard (visualization-first, local files)
   - PyTorch: MLflow (tracking-first, server/database)

2. **Lightning Logger Disabled:**
   - `ptycho_torch/workflows/components.py:760`: `logger=False` with comment "added in Phase D"
   - Current code doesn't explicitly set Trainer's `logger` parameter elsewhere
   - **Result:** Loss values logged via `self.log()` but not persisted

3. **Feature Parity Gap:**
   - TensorFlow has visualization UI (TensorBoard)
   - PyTorch has experiment tracking (MLflow) but no UI for loss curves
   - **Missing:** PyTorch users cannot visualize training progress

### Recommendations for Parity

**Option 1: TensorBoardLogger (Direct Parity)**
- ✅ Matches TensorFlow behavior exactly
- ✅ No new dependencies
- ✅ Users familiar with TensorBoard workflow
- ⚠️ Requires separate server process

**Option 2: CSVLogger (Lightweight Parity)**
- ✅ Provides same data (metrics, hyperparameters)
- ✅ Simpler than TensorBoard (no server needed)
- ✅ Programmatic access easier (pandas)
- ⚠️ No built-in UI (manual plotting)

**Option 3: Keep MLflow (Status Quo)**
- ⚠️ Maintains divergence from TensorFlow
- ⚠️ Heavier setup burden for users
- ⚠️ Semantic overload with `--disable_mlflow` flag

---

## API Surface Changes Needed

### Proposed: CSVLogger as Default (Tier 1)

**Configuration Field (PyTorchExecutionConfig):**
```python
# ptycho/config/config.py:242 (already exists, unused)
logger_backend: Optional[str] = 'csv'  # Options: None, 'csv', 'tensorboard', 'mlflow'
```

**Factory Integration:**
```python
# ptycho_torch/config_factory.py (new function or extend existing)
def _create_lightning_logger(
    logger_backend: Optional[str],
    output_dir: Path,
    experiment_name: str
) -> Union[Logger, bool]:
    """Create Lightning logger based on execution config."""
    if logger_backend == 'csv':
        from lightning.pytorch.loggers import CSVLogger
        return CSVLogger(save_dir=output_dir, name=experiment_name)
    elif logger_backend == 'tensorboard':
        from lightning.pytorch.loggers import TensorBoardLogger
        return TensorBoardLogger(save_dir=output_dir, name=experiment_name)
    elif logger_backend == 'mlflow':
        from lightning.pytorch.loggers import MLFlowLogger
        import mlflow
        return MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=mlflow.get_tracking_uri()
        )
    elif logger_backend is None or logger_backend == 'none':
        return False  # Disable Lightning logger
    else:
        raise ValueError(f"Invalid logger_backend: {logger_backend}")
```

**Workflow Integration:**
```python
# ptycho_torch/workflows/components.py:760 (replace logger=False)
logger = _create_lightning_logger(
    logger_backend=execution_config.logger_backend,
    output_dir=config.output_dir,
    experiment_name=config.experiment_name
)

trainer = L.Trainer(
    ...,
    logger=logger  # Use configured logger instead of False
)
```

**CLI Flags (train.py, inference.py):**
```python
# ptycho_torch/train.py (new flag)
parser.add_argument(
    '--logger',
    type=str,
    default='csv',
    choices=['none', 'csv', 'tensorboard', 'mlflow'],
    help='Experiment logging backend (default: csv)'
)

# Keep backward compatibility
parser.add_argument(
    '--disable_mlflow',
    action='store_true',
    help='(DEPRECATED: use --logger none) Disable MLflow tracking'
)
```

**Shared Helper (cli/shared.py):**
```python
# ptycho_torch/cli/shared.py (extend build_execution_config_from_args)
def build_execution_config_from_args(args, mode):
    # ... existing code ...

    # Handle backward compatibility
    if hasattr(args, 'disable_mlflow') and args.disable_mlflow:
        warnings.warn(
            "Flag --disable_mlflow is deprecated. Use --logger none instead.",
            DeprecationWarning,
            stacklevel=2
        )
        exec_config_dict['logger_backend'] = None
    elif hasattr(args, 'logger'):
        exec_config_dict['logger_backend'] = args.logger

    # ... rest of execution config ...
```

---

## Test Strategy

### Test Coverage Required

**Category 1: CLI Flag Parsing**
```python
# tests/torch/test_cli_shared.py (extend TestBuildExecutionConfig)
def test_logger_backend_csv_default(self):
    """Verify logger_backend defaults to 'csv'."""
    args = Namespace(logger='csv', ...)
    config = build_execution_config_from_args(args, mode='training')
    assert config.logger_backend == 'csv'

def test_logger_backend_none(self):
    """Verify logger can be disabled."""
    args = Namespace(logger='none', ...)
    config = build_execution_config_from_args(args, mode='training')
    assert config.logger_backend is None

def test_disable_mlflow_backward_compat(self):
    """Verify --disable_mlflow maps to logger=None with deprecation warning."""
    args = Namespace(disable_mlflow=True, ...)
    with pytest.warns(DeprecationWarning, match="deprecated.*--logger none"):
        config = build_execution_config_from_args(args, mode='training')
    assert config.logger_backend is None
```

**Category 2: Factory Integration**
```python
# tests/torch/test_config_factory.py (extend TestExecutionConfigOverrides)
def test_factory_creates_csv_logger(self, tmp_path):
    """Verify factory creates CSVLogger when backend='csv'."""
    execution_config = PyTorchExecutionConfig(logger_backend='csv')
    logger = _create_lightning_logger(
        logger_backend='csv',
        output_dir=tmp_path,
        experiment_name='test'
    )
    from lightning.pytorch.loggers import CSVLogger
    assert isinstance(logger, CSVLogger)

def test_factory_disables_logger_when_none(self):
    """Verify factory returns False when backend=None."""
    logger = _create_lightning_logger(
        logger_backend=None,
        output_dir=Path('/tmp'),
        experiment_name='test'
    )
    assert logger is False
```

**Category 3: Workflow Integration**
```python
# tests/torch/test_workflows_components.py (extend TestLightningExecutionConfig)
def test_trainer_receives_csv_logger(self, monkeypatch):
    """Verify Lightning Trainer receives CSVLogger when configured."""
    captured_logger = None

    def mock_trainer_init(self, *args, **kwargs):
        nonlocal captured_logger
        captured_logger = kwargs.get('logger')
        # ... rest of mock ...

    monkeypatch.setattr(L.Trainer, '__init__', mock_trainer_init)

    execution_config = PyTorchExecutionConfig(logger_backend='csv')
    # ... call _train_with_lightning ...

    from lightning.pytorch.loggers import CSVLogger
    assert isinstance(captured_logger, CSVLogger)
```

**Category 4: CSV Output Validation**
```python
# tests/torch/test_integration_workflow_torch.py (extend existing test)
def test_csv_logger_creates_metrics_file(self, tmp_path):
    """Verify CSVLogger writes metrics to CSV file."""
    # ... run minimal training with logger_backend='csv' ...

    metrics_file = tmp_path / 'lightning_logs' / 'version_0' / 'metrics.csv'
    assert metrics_file.exists()

    import pandas as pd
    df = pd.read_csv(metrics_file)
    assert 'train_loss' in df.columns or 'epoch' in df.columns
```

### CI Impact Assessment

| Logger | Test Changes Needed | CI Runtime Impact | Skip Logic Required |
|--------|---------------------|-------------------|---------------------|
| None | Minimal (just config tests) | None | No |
| **CSVLogger** | Add output validation tests | <1s per test | No (built-in) |
| TensorBoardLogger | Same as CSV + event file checks | <2s per test | Optional (`@pytest.mark.logger`) |
| MLFlowLogger | Mock MLflow client or skip | <2s per test (mocked) | Yes (server dependency) |

**Recommendation:** Start with CSVLogger to minimize CI friction. Add TensorBoardLogger tests later with optional marks.

---

## User Workflow Examples

### CSVLogger (Recommended Default)

**Training:**
```bash
python -m ptycho_torch.train \
  --train_data_file data.npz \
  --output_dir outputs/my_experiment \
  --logger csv \
  # ... other flags ...
```

**Output Structure:**
```
outputs/my_experiment/
├── lightning_logs/
│   └── version_0/
│       ├── hparams.yaml        # Hyperparameters
│       ├── metrics.csv          # Train/val losses per epoch
│       └── events.out.tfevents.* (if TensorBoard also enabled)
└── wts.h5.zip                   # Model checkpoint
```

**Visualization (Manual):**
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/my_experiment/lightning_logs/version_0/metrics.csv')
plt.plot(df['epoch'], df['train_loss'], label='Train')
plt.plot(df['epoch'], df['val_loss'], label='Val')
plt.legend()
plt.show()
```

**Pros:**
- ✅ Simple, no server needed
- ✅ Version control friendly
- ✅ Programmatic access

**Cons:**
- ⚠️ Manual plotting required

---

### TensorBoardLogger (Future Option)

**Training:**
```bash
python -m ptycho_torch.train \
  --train_data_file data.npz \
  --output_dir outputs/my_experiment \
  --logger tensorboard \
  # ... other flags ...
```

**Visualization:**
```bash
tensorboard --logdir outputs/
# Open http://localhost:6006 in browser
```

**Pros:**
- ✅ Rich UI with no coding
- ✅ Standard ML workflow
- ✅ No new dependencies

**Cons:**
- ⚠️ Separate server process
- ⚠️ Binary log format

---

### MLFlowLogger (Power Users)

**Training:**
```bash
# Set up MLflow tracking (one-time)
export MLFLOW_TRACKING_URI=file:///path/to/mlruns
# Or use remote server:
# export MLFLOW_TRACKING_URI=http://mlflow-server:5000

python -m ptycho_torch.train \
  --train_data_file data.npz \
  --output_dir outputs/my_experiment \
  --logger mlflow \
  # ... other flags ...
```

**Visualization:**
```bash
mlflow ui --backend-store-uri file:///path/to/mlruns
# Open http://localhost:5000 in browser
```

**Pros:**
- ✅ Full experiment tracking
- ✅ Model registry
- ✅ Hyperparameter search

**Cons:**
- ⚠️ Requires server setup
- ⚠️ Heavyweight
- ⚠️ Hard failure if server unavailable

---

## Recommendations

### Tier 1: MVP Default (Phase EB3.B Implementation)

**Adopt CSVLogger as default:**
- ✅ Zero new dependencies (POLICY-001 compliant)
- ✅ CI-friendly (no service dependencies, no skip logic)
- ✅ Captures currently-lost loss values
- ✅ Programmatic access via pandas
- ✅ Offline-first workflow
- ⚠️ Manual plotting required (acceptable tradeoff)

**Configuration:**
```python
logger_backend: Optional[str] = 'csv'  # Default in PyTorchExecutionConfig
```

**CLI Flag:**
```bash
--logger csv  # Default value, explicit for clarity
--logger none # Disable logger (backward compat with current behavior)
```

**Deprecation:**
```python
# Keep --disable_mlflow as deprecated alias
--disable_mlflow → --logger none (emit DeprecationWarning)
```

---

### Tier 2: Feature Parity Upgrade (Phase EB3 Follow-up or Phase F)

**Add TensorBoardLogger option:**
- ✅ No new dependencies (tensorboard already present)
- ✅ Matches TensorFlow baseline
- ✅ Industry standard
- ⚠️ Requires separate server process (standard ML workflow)
- ⚠️ Tests may need `@pytest.mark.optional`

**Configuration:**
```bash
--logger tensorboard  # User opt-in
```

**Documentation:**
```markdown
## Viewing Training Metrics

### CSV (Default)
- Metrics saved to `outputs/lightning_logs/version_0/metrics.csv`
- Plot manually with pandas/matplotlib

### TensorBoard (Optional)
- Metrics saved to `outputs/lightning_logs/version_0/events.out.tfevents.*`
- Run: `tensorboard --logdir outputs/`
- Open: http://localhost:6006
```

---

### Tier 3: Power Users (Keep as Optional)

**Maintain MLFlowLogger support:**
- ⚠️ Already in codebase (train.py:75-80)
- ⚠️ Requires server setup
- ⚠️ CI friction
- ✅ Best for teams with existing MLflow infrastructure

**Configuration:**
```bash
--logger mlflow  # User opt-in (experts only)
```

**Refactor:**
- Replace manual `mlflow.pytorch.autolog()` with Lightning integration
- Fix semantic overload (`--disable_mlflow` → `--logger none`)

---

### Tier 4: Not Recommended

**Do NOT implement:**
- `None` logger (status quo): Loses debugging capability
- `WandbLogger` / `NeptuneLogger`: Violate POLICY-001, require governance review

---

## Decision Criteria

| Criterion | CSVLogger | TensorBoardLogger | MLFlowLogger |
|-----------|-----------|-------------------|--------------|
| **POLICY-001 Compliance** | ✅ Perfect | ✅ Perfect | ⚠️ Grandfathered |
| **CI Friendliness** | ✅ Best (no infrastructure) | ✅ Good (optional tests) | ⚠️ Requires mocking |
| **User Friction** | ✅ Minimal (just read CSV) | ✅ Low (standard workflow) | ⚠️ Moderate (server setup) |
| **Feature Completeness** | ⚠️ No UI | ✅ Full UI | ✅ Full tracking |
| **TensorFlow Parity** | ⚠️ Similar data, different UI | ✅ Exact match | ❌ Different paradigm |
| **Dependency Footprint** | ✅ Zero | ✅ Zero (already have) | ⚠️ Heavy (~50+ packages) |

**Winner:** **CSVLogger** for MVP default, with optional **TensorBoardLogger** for users who want UI.

---

## Implementation Effort Estimate

### CSVLogger (Tier 1)

**Code Changes:**
- `ptycho/config/config.py`: Update `logger_backend` default to `'csv'` (1 line)
- `ptycho_torch/config_factory.py`: Add `_create_lightning_logger()` helper (~30 lines)
- `ptycho_torch/workflows/components.py`: Replace `logger=False` with factory call (~5 lines)
- `ptycho_torch/cli/shared.py`: Handle `--logger` flag + deprecation warning (~15 lines)
- `ptycho_torch/train.py`: Add `--logger` argument (~5 lines)

**Test Changes:**
- `tests/torch/test_cli_shared.py`: Add 3 logger flag tests (~60 lines)
- `tests/torch/test_config_factory.py`: Add 2 factory tests (~40 lines)
- `tests/torch/test_workflows_components.py`: Add 1 trainer integration test (~50 lines)
- `tests/torch/test_integration_workflow_torch.py`: Extend existing test to validate CSV output (~20 lines)

**Total:** ~230 lines of code, ~30 minutes implementation + 1 hour testing.

### TensorBoardLogger (Tier 2)

**Code Changes:**
- `ptycho_torch/config_factory.py`: Add `'tensorboard'` branch to logger factory (~3 lines)

**Test Changes:**
- `tests/torch/test_config_factory.py`: Add 1 TensorBoard test (~20 lines)
- Mark with `@pytest.mark.optional` for CI

**Total:** ~25 lines, ~15 minutes incremental.

---

## File References

**Dependencies:**
- `setup.py:37` — TensorFlow (includes tensorboard)
- `setup.py:49-50` — Lightning + MLflow in `[torch]` extras

**Current Implementation:**
- `ptycho_torch/train.py:75-80, 306-340` — Manual MLflow integration
- `ptycho_torch/workflows/components.py:760` — Logger disabled (`logger=False`)
- `ptycho/model.py:546-549` — TensorFlow TensorBoard callback

**Configuration:**
- `ptycho/config/config.py:242` — `logger_backend` field (defined but unused)

**Policy:**
- `docs/findings.md#POLICY-001` — PyTorch mandatory dependency policy

---

**Next Steps:**
- EB3.A3: Draft decision proposal recommending CSVLogger as MVP default
- Include API changes, test strategy, and migration path
- Address `--disable_mlflow` semantic overload
