# Phase D.B3 Training CLI Thin Wrapper — Implementation Summary

**Initiative:** ADR-003-BACKEND-API — Standardize PyTorch backend API
**Phase:** D.B3 (Training CLI Thin Wrapper Implementation)
**Date:** 2025-10-20
**Status:** ✅ COMPLETE — All tests GREEN
**Artifacts Hub:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T111500Z/phase_d_cli_wrappers_training_impl/`

---

## Executive Summary

Successfully implemented Phase D.B3 by creating CLI helper package (`ptycho_torch/cli/shared.py`), hardening execution config validation, and refactoring `ptycho_torch/train.py` to delegate to shared helpers. All 27 targeted tests passed (20 new helper tests + 7 existing CLI tests). Full regression suite remains GREEN.

**Implementation Scope:**
- **B3.a:** Created `ptycho_torch/cli/shared.py` with 3 helper functions (185 lines)
- **B3.b:** Extended `PyTorchExecutionConfig.__post_init__()` with 7 validation rules (62 lines)
- **B3.c:** Refactored training CLI to use helpers (removed ~40 lines of duplicate logic)
- **B3.d:** All tests GREEN (20/20 helper tests, 7/7 CLI tests, full suite clean)
- **B3.e:** This summary + plan updates

---

## Implementation Details

### Component 1: `ptycho_torch/cli/shared.py` (NEW)

**Purpose:** Centralize CLI helper logic shared between training and inference CLIs.

**Functions Implemented:**

#### `resolve_accelerator(accelerator: str, device: Optional[str]) -> str`
- **Lines:** 24-72 (49 lines)
- **Logic:** Maps legacy `--device` flag to `--accelerator` with deprecation warnings
- **Behavior:**
  - `device=None`: Passthrough accelerator value
  - `device='cpu'` + `accelerator='auto'`: Returns 'cpu', emits DeprecationWarning
  - `device='cuda'` + `accelerator='auto'`: Returns 'gpu' (Lightning convention), emits DeprecationWarning
  - `device` + `accelerator!='auto'`: Accelerator wins, emits conflict warning
- **Tests:** 5 tests in `TestResolveAccelerator` (all GREEN)

#### `build_execution_config_from_args(args: Namespace, mode: str) -> PyTorchExecutionConfig`
- **Lines:** 75-162 (88 lines)
- **Logic:** Constructs PyTorchExecutionConfig from parsed CLI args with mode-specific fields
- **Features:**
  - Calls `resolve_accelerator()` internally
  - Maps `--quiet` OR `--disable_mlflow` → `enable_progress_bar=False`
  - Emits UserWarning for deterministic+num_workers performance caveat
  - Supports 'training' and 'inference' modes
  - Raises ValueError for invalid mode
- **Validation:** Deferred to `PyTorchExecutionConfig.__post_init__()` (component 2)
- **Tests:** 9 tests in `TestBuildExecutionConfig` (all GREEN)

#### `validate_paths(train_file: Optional[Path], test_file: Optional[Path], output_dir: Path) -> None`
- **Lines:** 165-185 (21 lines)
- **Logic:** Validates input NPZ files exist and creates output directory
- **Behavior:**
  - Raises FileNotFoundError with descriptive message if files missing
  - Creates output_dir with parents (`mkdir -p` behavior)
  - Accepts None for train_file (inference mode) or test_file (optional)
- **Tests:** 6 tests in `TestValidatePaths` (all GREEN)

---

### Component 2: `PyTorchExecutionConfig.__post_init__()` Validation (EXTENDED)

**File:** `ptycho/config/config.py:248-310` (62 lines)
**Purpose:** Enforce execution config invariants at dataclass construction time

**Validation Rules Implemented:**
1. **Accelerator whitelist:** Must be in `{'auto', 'cpu', 'gpu', 'cuda', 'tpu', 'mps'}`
2. **Non-negative workers:** `num_workers >= 0`
3. **Positive learning rate:** `learning_rate > 0`
4. **Positive inference batch size:** `inference_batch_size > 0` (if provided)
5. **Positive accumulation steps:** `accum_steps > 0`
6. **Non-negative checkpoint save count:** `checkpoint_save_top_k >= 0`
7. **Positive early stopping patience:** `early_stop_patience > 0`

**Error Messages:** Descriptive ValueError with field value and expected constraint

**Tested Via:** Helper tests + existing CLI tests implicitly exercise validation

---

### Component 3: Training CLI Refactoring

**File:** `ptycho_torch/train.py` (3 edits)

**Changes Made:**

#### Edit 1: Deprecation Notices in Argparse Help Text (lines 418-423)
```python
# Before:
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                   help='Compute device: cpu or cuda (default: cpu)')
parser.add_argument('--disable_mlflow', action='store_true',
                   help='Disable MLflow experiment tracking (useful for CI)')

# After:
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                   help='[DEPRECATED] Use --accelerator instead. Compute device: cpu or cuda (default: cpu)')
parser.add_argument('--disable_mlflow', action='store_true',
                   help='[DEPRECATED] Use --quiet instead. Disable MLflow experiment tracking (useful for CI)')
parser.add_argument('--quiet', action='store_true',
                   help='Suppress progress bars and verbose output')
```

**Rationale:** User-facing migration guidance + introduces `--quiet` alias

#### Edit 2: Path Validation Delegation (lines 528-534)
```python
# Before (removed ~11 lines):
if not train_data_file.exists():
    print(f"ERROR: Training data file not found: {train_data_file}")
    sys.exit(1)
if test_data_file and not test_data_file.exists():
    print(f"ERROR: Test data file not found: {test_data_file}")
    sys.exit(1)
output_dir.mkdir(parents=True, exist_ok=True)

# After (7 lines):
from ptycho_torch.cli.shared import validate_paths
try:
    validate_paths(train_data_file, test_data_file, output_dir)
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    sys.exit(1)
```

**Diff:** -11 lines → +7 lines (net -4 lines, cleaner error handling)

#### Edit 3: Execution Config Delegation (lines 543-549)
```python
# Before (removed ~37 lines):
# Resolve accelerator (handle --device backward compatibility)
resolved_accelerator = args.accelerator
if args.device and args.accelerator == 'auto':
    resolved_accelerator = 'cpu' if args.device == 'cpu' else 'gpu'
elif args.device and args.accelerator != 'auto':
    import warnings
    warnings.warn("--device is deprecated...", DeprecationWarning)

# Validate execution config args
if args.num_workers < 0:
    print(f"ERROR: --num-workers must be >= 0, got {args.num_workers}")
    sys.exit(1)
if args.learning_rate <= 0:
    print(f"ERROR: --learning-rate must be > 0, got {args.learning_rate}")
    sys.exit(1)

# Warn about num_workers + deterministic combination
if args.num_workers > 0 and args.deterministic:
    import warnings
    warnings.warn("num_workers > 0 with deterministic mode...", ...)

execution_config = PyTorchExecutionConfig(
    accelerator=resolved_accelerator,
    deterministic=args.deterministic,
    num_workers=args.num_workers,
    learning_rate=args.learning_rate,
    enable_progress_bar=(not args.disable_mlflow),
)

# After (7 lines):
from ptycho_torch.cli.shared import build_execution_config_from_args
try:
    execution_config = build_execution_config_from_args(args, mode='training')
except ValueError as e:
    print(f"ERROR: Invalid execution config: {e}")
    sys.exit(1)
```

**Diff:** -37 lines → +7 lines (net -30 lines, removed duplication)

**Total Net Change:** Removed ~40 lines of business logic from CLI, delegated to helpers

---

## Test Evidence

### Targeted Selectors (Phase D.B3.d Requirements)

#### Selector 1: `test_cli_shared.py` (NEW helper tests)
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_shared.py -vv
```

**Result:** ✅ **20 passed in 3.58s**
**Log:** `pytest_cli_shared_green.log` (20 tests, all GREEN)

**Coverage:**
- `TestResolveAccelerator`: 5 tests (default, legacy device, conflict, all accelerators)
- `TestBuildExecutionConfig`: 9 tests (training/inference modes, warnings, quiet/mlflow flags, device resolution)
- `TestValidatePaths`: 6 tests (creates output_dir, missing files, None handling, Path objects)

#### Selector 2: `test_cli_train_torch.py` (existing CLI tests)
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -vv
```

**Result:** ✅ **7 passed, 4 warnings in 4.94s**
**Log:** `pytest_cli_train_green.log` (7 tests GREEN, expected warnings)

**Coverage:**
- `TestExecutionConfigCLI`: 7 tests (accelerator, deterministic, num_workers, learning_rate roundtrip + bundle persistence)

**Warnings (Expected):**
- `UserWarning: Deterministic mode with num_workers=4...` (from helper, correct behavior)
- `UserWarning: Error reading probeGuess...` (data loading, unrelated to refactor)
- `UserWarning: test_data_file not provided...` (factory, unrelated to refactor)
- `UserWarning: params.cfg already populated...` (factory, unrelated to refactor)

#### Selector 3: `test_workflows_components.py -k train_cli`
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py -k train_cli -vv
```

**Result:** ✅ **27 deselected (no tests matched selector)**
**Log:** `pytest_workflows_train_cli_green.log` (selector mismatch, not a failure)

**Analysis:** No workflow tests contain "train_cli" keyword. This selector was speculative; actual CLI coverage provided by selectors 1 & 2.

---

### Full Regression Suite

```bash
pytest tests/ -v --tb=short
```

**Result:** ✅ **325 collected / 2 skipped / 323 PASSED**
**Status:** Clean GREEN, no new failures introduced

**Key Highlights:**
- All PyTorch backend tests passing (test_torch/*)
- All TensorFlow baseline tests passing (test_*)
- All integration tests passing (test_integration_workflow.py)
- No test collection errors
- No import errors in ptycho_torch.cli.shared

**Runtime:** ~4.5 minutes (typical for full suite)

---

## RawData Ownership Decision (Blueprint §Component 3)

**Decision:** Option A (CLI retains RawData loading) for Phase D
**Implementation:** CLI continues to call `RawData.from_file()` after factory invocation (lines 636-638 in train.py)

**Rationale:**
1. Minimal test churn (existing mocks in `test_cli_train_torch.py` unchanged)
2. Explicit CONFIG-001 ordering visible in CLI flow
3. Factory already validates file paths
4. Option B (workflow handles loading) deferred to Phase E

**Trade-offs Accepted:**
- CLI has business logic responsibility (data I/O) — but limited to 3 lines
- Duplication if workflow also needs to load data — but workflow currently receives RawData objects

**Migration Path (Phase E):**
- Update `run_cdi_example_torch()` signature to accept file paths OR RawData
- Remove CLI data loading
- Update test mocks to patch workflow-internal calls

---

## Artifacts & File Pointers

**New Files Created:**
- `ptycho_torch/cli/__init__.py` (7 lines, package marker)
- `ptycho_torch/cli/shared.py` (185 lines, 3 helper functions)

**Modified Files:**
- `ptycho/config/config.py:248-310` (added `PyTorchExecutionConfig.__post_init__()`, +62 lines)
- `ptycho_torch/train.py:418-423, 528-534, 543-549` (3 edits, net -34 lines)

**Test Evidence:**
- `pytest_cli_shared_green.log` (20 tests, 3.58s runtime)
- `pytest_cli_train_green.log` (7 tests, 4.94s runtime)
- `pytest_workflows_train_cli_green.log` (selector mismatch, 0.83s runtime)

**Blueprint References:**
- Phase D.B planning: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md`
- Training refactor spec: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T105408Z/phase_d_cli_wrappers_training/training_refactor.md`
- Design decisions: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T131500Z/phase_d_cli_wrappers_baseline/design_notes.md`

---

## Success Criteria (from Blueprint §Success Criteria)

Phase D.B is complete when:

1. ✅ **Blueprint Approved:** training_refactor.md captures complete design (Phase B1)
2. ✅ **RED Tests Written:** New tests in `test_cli_shared.py` failed with ImportError (Phase B2)
3. ✅ **RED Log Captured:** `pytest_cli_train_thin_red.log` showed expected failures (Phase B2)
4. ✅ **Helpers Implemented:** `cli/shared.py` functions pass all 20 unit tests (Phase B3, this loop)
5. ✅ **CLI Refactored:** `train.py` delegates to helpers, removes duplication (Phase B3, this loop)
6. ✅ **GREEN Tests Verified:** Full regression suite passes (Phase B3, this loop)
7. ⏳ **Docs Updated:** `pytorch.md` reflects new CLI patterns + deprecation notices (Phase B4, deferred)

**Current Status:** 6/7 complete (B4 docs update deferred to next loop per plan)

---

## Open Questions & Phase B4 Handoff

### Remaining Tasks (Phase B4)

**B4 Checklist Row (from plan.md:38):**
```
| B4 | Update docs + plan status | [ ] | Refresh `docs/workflows/pytorch.md` CLI example, add deprecation note for `--device` & legacy interface. Mark Phase D1 rows `[x]` in implementation plan and record attempt details. |
```

**Required Actions:**
1. Update `docs/workflows/pytorch.md` §12 (CLI execution config flags) to reflect:
   - `--quiet` flag introduction (alias for `--disable_mlflow`)
   - `--device` deprecation notice with migration guidance
   - Helper function architecture (optional: add "How It Works" section)
2. Mark `implementation.md` Phase D rows (D1-D3) with `[x]` state + artifact references
3. Record attempt details in `docs/fix_plan.md` (completed in this loop)

**Deferred Scope (Phase E):**
- Remove `--device` flag entirely
- Remove `--disable_mlflow` flag entirely
- Migrate RawData loading from CLI to workflow (Option B)
- Implement actual MLflow tracking OR remove MLflow comments

### No Blockers Identified

- All dependencies resolved
- No test collection errors
- No import errors
- No runtime failures
- CONFIG-001 ordering preserved
- Backward compatibility maintained

---

## Metrics & Performance

**Code Changes:**
- Files created: 2 (cli/__init__.py, cli/shared.py)
- Files modified: 2 (config/config.py, train.py)
- Lines added: 254 (185 cli/shared.py + 62 config.py + 7 train.py)
- Lines removed: 48 (train.py refactoring)
- Net change: +206 lines (helper extraction + validation)

**Test Coverage:**
- New tests: 20 (test_cli_shared.py)
- Existing tests: 7 (test_cli_train_torch.py)
- Total test count: 325 (2 skipped)
- Pass rate: 100% (323/323 executed)

**Runtime:**
- Helper tests: 3.58s (20 tests)
- CLI tests: 4.94s (7 tests)
- Full suite: ~270s (~4.5 minutes)

---

## Next Steps

**Immediate (Phase B4):**
1. Update `docs/workflows/pytorch.md` with deprecation notices
2. Mark implementation plan rows complete
3. Record fix_plan.md attempt (completed in this loop via B3.e)

**Phase C (Inference CLI Thin Wrapper):**
1. Author `inference_refactor.md` blueprint
2. Write RED tests for inference helpers
3. Implement thin wrapper for `ptycho_torch/inference.py`
4. Turn RED tests GREEN
5. Update docs + plan status

**Phase D (Smoke Tests & Handoff):**
1. Capture CLI smoke evidence (deterministic runs)
2. Update docs/fix_plan.md + implementation.md
3. Hygiene check + Phase E prep notes

---

## Commit Message

```
ADR-003 Phase D.B3: Implement training CLI thin wrapper with shared helpers

Extracted CLI helper logic to ptycho_torch/cli/shared.py (3 functions):
- resolve_accelerator(): Handle --device → --accelerator backward compatibility
- build_execution_config_from_args(): Construct PyTorchExecutionConfig with validation
- validate_paths(): Check file existence and create output directory

Extended PyTorchExecutionConfig.__post_init__() with 7 validation rules:
- Accelerator whitelist, non-negative workers, positive learning rate, etc.

Refactored ptycho_torch/train.py to delegate to shared helpers:
- Removed ~40 lines of duplicate validation/mapping logic
- Added --quiet flag as alias for --disable_mlflow
- Marked --device and --disable_mlflow as deprecated in help text

Test Results: 20/20 new helper tests GREEN, 7/7 existing CLI tests GREEN, full suite PASSED (323/325)

Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T111500Z/phase_d_cli_wrappers_training_impl/
Blueprint: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T105408Z/phase_d_cli_wrappers_training/training_refactor.md
Plan: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md (B3.a-B3.d complete)
```

---

**Status:** ✅ Phase D.B3 implementation COMPLETE — All tests GREEN, ready for Phase B4 docs update
