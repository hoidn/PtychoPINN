# Phase EB3.A3 — Logger Backend Decision Proposal

**Date:** 2025-10-23
**Initiative:** ADR-003-BACKEND-API Phase EB3 (Logger Governance)
**Task:** Draft decision proposal for supervisor review
**Author:** Ralph (Engineer Agent)
**Reviewer:** Galph (Supervisor Agent)

---

## Executive Summary

**Recommendation:** Adopt **CSVLogger** as the default Lightning logger backend for PyTorch CLI with immediate implementation in Phase EB3.B, and define an optional upgrade path to **TensorBoardLogger** for future phases.

**Rationale:**
1. ✅ **POLICY-001 Compliant** — Zero new dependencies
2. ✅ **Captures Currently-Lost Metrics** — Loss values from `self.log()` currently discarded
3. ✅ **CI-Friendly** — No service dependencies, no test mocking required
4. ✅ **User-Friendly** — Programmatic access via pandas, offline-first workflow
5. ✅ **Minimal Implementation** — ~230 lines of code, <2 hour effort

---

## Decision: Enable CSVLogger by Default

### Proposed Configuration

**Default Behavior:**
```python
# ptycho/config/config.py:242 (PyTorchExecutionConfig)
logger_backend: Optional[str] = 'csv'  # NEW DEFAULT (was None)
```

**Supported Values:**
- `'csv'` — CSVLogger (recommended default, built-in to Lightning)
- `'tensorboard'` — TensorBoardLogger (optional, requires tensorboard package already installed)
- `'mlflow'` — MLFlowLogger (optional, for power users with MLflow infrastructure)
- `None` or `'none'` — Disable logger (backward compat with current `logger=False` behavior)

---

## Pros & Cons Analysis

### Pros (CSVLogger Default)

1. **Zero Dependency Footprint**
   - CSVLogger is built-in to Lightning (no new packages)
   - Satisfies POLICY-001 (no new mandatory dependencies without governance)

2. **Captures Lost Metrics**
   - Current code: `ptycho_torch/model.py:1248, 1260` calls `self.log()` but `logger=False` discards values
   - With CSVLogger: Train/val losses persisted to CSV for analysis

3. **CI-Friendly**
   - No service dependencies (MLflow server, TensorBoard server)
   - No test mocking required (unlike MLFlowLogger)
   - Fast test execution (<1s overhead per test)

4. **Programmatic Access**
   - CSV format is human-readable and parseable with pandas
   - Easy to integrate with custom visualization pipelines
   - Version control friendly (text-based format)

5. **Offline-First Workflow**
   - No internet/server required
   - Works in air-gapped environments
   - Aligns with scientific reproducibility principles

6. **Minimal Implementation Effort**
   - ~230 lines of code (50 production + 180 tests)
   - <2 hour implementation + testing
   - Low maintenance burden

### Cons (CSVLogger Default)

1. **No Built-In Visualization UI**
   - Users must manually plot metrics with matplotlib/pandas
   - Workaround: Document example plotting code in workflow guide
   - Mitigation: Provide optional TensorBoardLogger for users who need UI

2. **Not Database-Queryable**
   - CSV lacks query capabilities of MLflow/database backends
   - Workaround: Load CSV into pandas DataFrame for filtering/analysis
   - Mitigation: This is acceptable for typical use cases (10-100 experiments)

3. **Different from TensorFlow Baseline**
   - TensorFlow uses TensorBoard (`ptycho/model.py:549`)
   - PyTorch would use CSV by default
   - Mitigation: Offer TensorBoardLogger as optional upgrade (Tier 2)

---

## Alternative Considered: TensorBoardLogger Default

### Why Not TensorBoardLogger?

**Pros:**
- Exact parity with TensorFlow baseline
- Rich visualization UI (industry standard)
- No new dependencies (tensorboard already installed)

**Cons:**
- Requires separate server process (`tensorboard --logdir outputs/`)
- Binary log format (not human-readable)
- Heavier disk footprint (~10x larger than CSV)
- Slightly more complex testing (optional `@pytest.mark.logger` tests)

**Decision:** Start with CSVLogger for simplicity, offer TensorBoardLogger as opt-in upgrade path.

---

## Alternative Considered: Keep MLflowLogger

### Why Not MLflow as Default?

**Current State:**
- MLflow already used in `ptycho_torch/train.py:75-80` (manual `autolog()`)
- Heavyweight (~50+ transitive dependencies)
- Requires server setup (local file store or remote tracking server)

**Problems:**
1. **Semantic Overload** — `--disable_mlflow` flag controls BOTH MLflow AND progress bar (`cli/shared.py:97-117`)
2. **Hard Failure Mode** — Server unavailability causes training to fail (not graceful degradation)
3. **CI Friction** — Tests require `disable_mlflow=True` or mock MLflow client
4. **User Friction** — Setup complexity, tracking URI configuration, experiment management

**Decision:** Keep MLflow as **opt-in** option (`--logger mlflow`) for power users, but NOT default.

---

## Implementation Plan

### Phase EB3.B — CSVLogger Implementation (TDD)

**B1: RED Tests (1 hour)**

*Category 1: CLI Flag Parsing* (`tests/torch/test_cli_shared.py`)
```python
def test_logger_backend_csv_default()  # Verify default='csv'
def test_logger_backend_none()         # Verify logger can be disabled
def test_disable_mlflow_backward_compat()  # Verify --disable_mlflow → --logger none
```

*Category 2: Factory Integration* (`tests/torch/test_config_factory.py`)
```python
def test_factory_creates_csv_logger()  # Verify CSVLogger instantiation
def test_factory_disables_logger_when_none()  # Verify logger=False when backend=None
```

*Category 3: Workflow Integration* (`tests/torch/test_workflows_components.py`)
```python
def test_trainer_receives_csv_logger()  # Verify Lightning Trainer gets logger
```

*Category 4: Output Validation* (`tests/torch/test_integration_workflow_torch.py`)
```python
def test_csv_logger_creates_metrics_file()  # Verify CSV output exists
```

**Expected RED Evidence:** 7 tests FAIL with `AttributeError: _create_lightning_logger` or similar.

---

**B2: Implementation (1 hour)**

*Step 1: Update Config Default*
```python
# ptycho/config/config.py:242
logger_backend: Optional[str] = 'csv'  # Change from None
```

*Step 2: Add Logger Factory* (`ptycho_torch/config_factory.py`, ~30 lines)
```python
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

*Step 3: Integrate into Workflow* (`ptycho_torch/workflows/components.py:760`)
```python
# BEFORE:
logger=False  # Disable default logger for now; MLflow/TensorBoard added in Phase D

# AFTER:
logger = _create_lightning_logger(
    logger_backend=execution_config.logger_backend,
    output_dir=config.output_dir,
    experiment_name=config.experiment_name
)
```

*Step 4: CLI Flag Support* (`ptycho_torch/train.py`, ~5 lines)
```python
parser.add_argument(
    '--logger',
    type=str,
    default='csv',
    choices=['none', 'csv', 'tensorboard', 'mlflow'],
    help='Experiment logging backend (default: csv)'
)
```

*Step 5: Backward Compatibility* (`ptycho_torch/cli/shared.py`, ~15 lines)
```python
def build_execution_config_from_args(args, mode):
    # ... existing code ...

    # Handle deprecated --disable_mlflow flag
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

**B3: GREEN Validation (30 minutes)**

**Targeted Selectors:**
```bash
# CLI flag tests
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_shared.py::TestBuildExecutionConfig::test_logger_backend_csv_default -vv
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_shared.py::TestBuildExecutionConfig::test_logger_backend_none -vv
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_shared.py::TestBuildExecutionConfig::test_disable_mlflow_backward_compat -vv

# Factory tests
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py::TestExecutionConfigOverrides::test_factory_creates_csv_logger -vv
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py::TestExecutionConfigOverrides::test_factory_disables_logger_when_none -vv

# Workflow integration
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestLightningExecutionConfig::test_trainer_receives_csv_logger -vv

# Integration test (CSV output validation)
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
```

**Expected GREEN Evidence:** 7/7 tests PASS, CSV metrics file exists in integration test.

**Full Regression:**
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/ -v
```

**Expected:** Zero new failures, existing 335 passed / 17 skipped maintained.

---

### Phase EB3.C — Documentation Sync (30 minutes)

**C1: Spec Updates** (`specs/ptychodus_api_spec.md`)

*§4.9 (PyTorchExecutionConfig):*
```diff
- logger_backend (str|None, default `None`): Experiment tracking backend. Pending governance decision (Phase E.B3).
+ logger_backend (str|None, default `'csv'`): Experiment tracking backend. Options: `'csv'` (default, built-in CSVLogger), `'tensorboard'` (requires tensorboard package, already installed via TensorFlow), `'mlflow'` (requires MLflow server setup, for power users), `None` or `'none'` (disable logger). CSVLogger captures train/val losses to CSV files for programmatic analysis. See Lightning logger docs for output format.
```

*§7.1 (Training CLI Table):*
```diff
+ | `--logger` | str | `'csv'` | Experiment logging backend: `'csv'` (default, built-in), `'tensorboard'` (TensorBoard UI, requires separate server), `'mlflow'` (MLflow tracking, requires server), `'none'` (disable). CSV logger saves metrics to `<output_dir>/lightning_logs/version_N/metrics.csv`. |
```

*§7.1 (Deprecated Flags):*
```diff
- `--disable_mlflow`: MLflow integration not yet implemented; flag is accepted but has no effect. Use `--quiet` to suppress progress output instead.
+ `--disable_mlflow`: DEPRECATED. Use `--logger none` instead. Maps to disabling Lightning logger (backward compatibility maintained).
```

---

**C2: Workflow Guide Updates** (`docs/workflows/pytorch.md`)

*§12 (Training CLI Table):*
```diff
+ | `--logger` | str | `'csv'` | Experiment logging backend. Choices: `'csv'` (default, saves metrics to CSV), `'tensorboard'` (TensorBoard UI), `'mlflow'` (MLflow tracking), `'none'` (disable). |
```

*Add new subsection after table:*
```markdown
### Viewing Training Metrics

**CSV Logger (Default):**
Metrics are saved to `<output_dir>/lightning_logs/version_0/metrics.csv`:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/my_experiment/lightning_logs/version_0/metrics.csv')
plt.plot(df['epoch'], df['train_loss'], label='Train')
plt.plot(df['epoch'], df['val_loss'], label='Val')
plt.legend()
plt.show()
```

**TensorBoard Logger (Optional):**
```bash
python -m ptycho_torch.train --logger tensorboard ...
tensorboard --logdir outputs/
# Open http://localhost:6006
```

**MLflow Logger (Power Users):**
```bash
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
python -m ptycho_torch.train --logger mlflow ...
mlflow ui
```
```

---

**C3: Findings Ledger Update** (`docs/findings.md`)

*Add new entry after CONFIG-002:*
```markdown
| CONFIG-003 | 2025-10-23 | logger, execution-config, csv, tensorboard | PyTorch CLI uses CSVLogger by default (LOGGER-001). Lightning logger captures train/val losses to `<output_dir>/lightning_logs/version_N/metrics.csv` for programmatic analysis. Users can opt-in to TensorBoardLogger (`--logger tensorboard`) or MLFlowLogger (`--logger mlflow`). Disabling logger (`--logger none`) restores pre-EB3 behavior (`logger=False`). | [Link](plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/decision/proposal.md) | Active |
```

---

## Acceptance Criteria

### Functional Requirements

1. ✅ **Default Behavior:** PyTorch training with no `--logger` flag creates CSV metrics file
2. ✅ **Opt-Out:** `--logger none` disables logger (backward compat with current behavior)
3. ✅ **TensorBoard Support:** `--logger tensorboard` works (uses existing tensorboard package)
4. ✅ **MLflow Support:** `--logger mlflow` works (existing MLflow integration)
5. ✅ **Backward Compat:** `--disable_mlflow` still works with deprecation warning
6. ✅ **Loss Capture:** Train/val losses from `self.log()` persisted to CSV

### Technical Requirements

1. ✅ **POLICY-001 Compliance:** Zero new mandatory dependencies
2. ✅ **CONFIG-001 Compliance:** Logger created AFTER `update_legacy_dict()` call
3. ✅ **Factory Integration:** Logger creation via `_create_lightning_logger()` helper
4. ✅ **Execution Config:** `PyTorchExecutionConfig.logger_backend` drives behavior
5. ✅ **CLI Validation:** Invalid logger backend raises ValueError with helpful message

### Test Requirements

1. ✅ **RED Phase:** 7 tests fail with clear NotImplementedError/AttributeError
2. ✅ **GREEN Phase:** 7/7 tests pass after implementation
3. ✅ **Full Regression:** Zero new failures in full pytest suite
4. ✅ **CSV Output:** Integration test validates metrics.csv exists and contains loss columns
5. ✅ **Runtime:** Test execution <90s (alignment with TEST-PYTORCH-001 budget)

### Documentation Requirements

1. ✅ **Spec Updated:** §4.9 and §7.1 document logger_backend field and CLI flags
2. ✅ **Workflow Guide Updated:** §12 includes logger flag + visualization examples
3. ✅ **Findings Updated:** CONFIG-003 entry documents logger policy
4. ✅ **Migration Path:** Deprecated flags documented with recommended alternatives

---

## Risks & Mitigations

### Risk 1: Users Expect TensorBoard (Like TensorFlow)

**Probability:** Medium
**Impact:** Low (cosmetic difference)

**Mitigation:**
- Document TensorBoard as opt-in (`--logger tensorboard`)
- Provide example visualization code in workflow guide
- Emphasize programmatic access benefits (pandas integration)

---

### Risk 2: CSV Format Too Minimal for Large Experiments

**Probability:** Low
**Impact:** Medium (users frustrated)

**Mitigation:**
- CSV is sufficient for typical use cases (10-100 experiments)
- Power users can upgrade to MLflow (`--logger mlflow`)
- Future enhancement: Add database-backed CSVLogger variant

---

### Risk 3: Test Execution Time Increases

**Probability:** Low
**Impact:** Low (CI slowdown)

**Mitigation:**
- CSVLogger has minimal overhead (<1s per test)
- Integration test already within 90s budget (currently 16.77s)
- Monitor CI runtime post-implementation

---

### Risk 4: Semantic Overload in --disable_mlflow Not Fully Resolved

**Probability:** Medium (if users rely on current behavior)
**Impact:** Low (progress bar control)

**Mitigation:**
- Keep `--disable_mlflow` as deprecated alias (`--logger none`)
- Emit clear DeprecationWarning with migration guidance
- Document recommended flags in workflow guide

---

## Open Questions for Supervisor

### Q1: Approve CSVLogger as Default?

**Proposal:** Set `PyTorchExecutionConfig.logger_backend = 'csv'` as default.

**Alternative:** Keep `logger_backend = None` (status quo, no logger).

**Recommendation:** **Approve CSVLogger**. Benefits (captures lost metrics, POLICY-001 compliant, CI-friendly) outweigh cons (no UI).

**Decision Required:** ☐ Approve ☐ Reject ☐ Modify

---

### Q2: Implement TensorBoardLogger in Phase EB3.B?

**Proposal:** Add TensorBoardLogger support (`--logger tensorboard`) in same phase as CSVLogger.

**Alternative:** Defer TensorBoardLogger to Phase EB3 follow-up or Phase F.

**Recommendation:** **Include in EB3.B**. Requires only 3 additional lines of code, provides immediate TensorFlow parity option.

**Decision Required:** ☐ Approve ☐ Defer ☐ Skip

---

### Q3: Deprecate --disable_mlflow Immediately?

**Proposal:** Emit `DeprecationWarning` when `--disable_mlflow` used, map to `--logger none`.

**Alternative:** Remove `--disable_mlflow` entirely (breaking change).

**Recommendation:** **Deprecate with warning**. Maintains backward compatibility, guides users to new syntax.

**Decision Required:** ☐ Approve ☐ Remove ☐ Keep Forever

---

### Q4: Document MLflow Refactor as Follow-Up Task?

**Current State:** Manual `mlflow.pytorch.autolog()` in `ptycho_torch/train.py:306`.

**Proposal:** Replace with Lightning MLFlowLogger integration in future phase (cleaner architecture).

**Recommendation:** **Yes, document as Phase EB3 follow-up**. Current implementation works but mixing manual autolog + Lightning logger is architecturally unclean.

**Decision Required:** ☐ Track Follow-Up ☐ Not a Priority

---

## Artifacts Delivered (Phase EB3.A)

1. ✅ **Current State Audit** (`analysis/current_state.md`, 12 KB)
   - Comprehensive logging hooks inventory
   - PyTorch vs TensorFlow comparison
   - Critical findings (logger disabled, semantic overload)

2. ✅ **Options Matrix** (`analysis/options_matrix.md`, 26 KB)
   - 6 logger options evaluated (None, CSV, TensorBoard, MLflow, WandB, Neptune)
   - POLICY-001 compliance analysis
   - Dependency footprint, CI impact, user workflow impact
   - Test strategy for each option

3. ✅ **Decision Proposal** (`decision/proposal.md`, THIS FILE, ~18 KB)
   - Recommendation: CSVLogger as default
   - Implementation plan (TDD phases)
   - Acceptance criteria
   - Open questions for supervisor approval

**Total:** 3 deliverables, ~56 KB documentation, comprehensive governance analysis.

---

## Next Steps (Pending Supervisor Approval)

**If Q1-Q4 Approved:**
1. Supervisor updates `plan.md` rows A1-A3 to `[x]`
2. Engineer proceeds to Phase EB3.B (TDD implementation)
3. RED tests authored per §B1 (~1 hour)
4. Implementation per §B2 (~1 hour)
5. GREEN validation per §B3 (~30 minutes)
6. Documentation sync per §C1-C3 (~30 minutes)
7. Commit with message: `[ADR-003 Phase EB3] Enable CSVLogger by default, add TensorBoard/MLflow options`
8. Update fix_plan.md with Attempt #66 and artifact links

**If Modifications Required:**
1. Supervisor documents requested changes in `decision/feedback.md`
2. Engineer iterates on proposal (Phase EB3.A revision)
3. Re-submit for approval

---

## References

**Authoritative Documents:**
- `specs/ptychodus_api_spec.md` §4.9 (PyTorchExecutionConfig contract)
- `specs/ptychodus_api_spec.md` §7.1 (Training CLI flags)
- `docs/workflows/pytorch.md` §12 (PyTorch workflow guide)
- `docs/findings.md` (Knowledge ledger: POLICY-001, CONFIG-001, CONFIG-002)
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/open_questions.md` §Q2 (MLflow ownership)

**Implementation Evidence:**
- `ptycho_torch/train.py:75-80, 306-340` — Current MLflow integration
- `ptycho_torch/workflows/components.py:760` — Logger disabled (`logger=False`)
- `ptycho/model.py:546-549` — TensorFlow TensorBoard baseline
- `setup.py:37` — TensorFlow dependency (includes tensorboard)
- `setup.py:49-50` — Lightning + MLflow in `[torch]` extras

**Lightning Documentation:**
- https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.html

---

**Prepared by:** Ralph (Engineer Agent)
**Review Status:** ☐ Pending Supervisor Approval
**Date:** 2025-10-23
