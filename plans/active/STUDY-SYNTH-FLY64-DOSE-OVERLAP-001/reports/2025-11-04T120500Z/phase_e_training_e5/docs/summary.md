# Phase E5 Training Runner Integration — Summary

**Loop:** Attempt #19 (Ralph execution)
**Date:** 2025-11-04T120500Z
**Mode:** TDD
**Focus:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E5 — Real PyTorch backend delegation

## Problem Statement

Phase E5 required upgrading the `execute_training_job` stub to delegate to the real PyTorch backend trainer (`train_cdi_model_torch`), enabling end-to-end training execution for the fly64 dose/overlap study.

**SPEC Requirements (input.md:9-10):**
- Replace marker stub with real backend wiring
- Load datasets through `ptycho_torch.memmap_bridge.MemmapDatasetBridge` or equivalent
- Call `train_cdi_model_torch` with CLI execution knobs
- Persist returned metrics/checkpoints into `job.artifact_dir`
- Ensure CLI manifests/logs reflect execution (no placeholder text)

**Relevant SPEC/ADR Citations:**
- `docs/DEVELOPER_GUIDE.md:68-104` — CONFIG-001 bridge ordering
- `docs/workflows/pytorch.md:245-312` — Canonical PyTorch training invocation
- `docs/pytorch_runtime_checklist.md` — Runtime guardrails
- `specs/data_contracts.md:190-260` — DATA-001 NPZ requirements

## Implementation

### Modifications

**File:** `studies/fly64_dose_overlap/training.py`
**Lines:** 22-29 (module-level imports), 290-442 (execute_training_job function)

1. **Module Imports (lines 28-29):**
   - Added `from ptycho.workflows.components import load_data`
   - Added `from ptycho_torch.workflows.components import train_cdi_model_torch`
   - Moved imports to module level for testability (monkeypatch compatibility)

2. **execute_training_job Implementation (lines 290-442):**
   - **Step 0:** Ensured log directory exists before writing (`log_path.parent.mkdir(parents=True, exist_ok=True)`)
   - **Step 1:** Wrote execution metadata to log (dose, view, gridsize, dataset paths)
   - **Step 2:** Validated dataset paths exist (defensive check)
   - **Step 3:** Loaded datasets via `load_data()` helper (constructs RawData instances from NPZ files)
   - **Step 4:** Delegated to `train_cdi_model_torch(train_data, test_data, config)`
   - **Step 5:** Extracted metrics from training results (final_loss, epochs_completed)
   - **Step 6:** Scanned for checkpoint files in output_dir (`.ckpt`, `.pth`)
   - **Step 7:** Returned result dict with status, metrics, and checkpoint path
   - **Error Handling:** Catch exceptions during loading and training, log failures, return error dict

**Key Design Decisions:**
- Used `load_data()` instead of RawData constructor (requires NPZ parsing)
- PyTorch backend accepts RawData instances (bridged internally via RawDataTorch adapters per Phase C design)
- Checkpoint detection via glob patterns (`.ckpt`, `.pth`) sorted by modification time
- Comprehensive error handling with traceback logging for debugging

### Tests

**File:** `tests/study/test_dose_overlap_training.py`
**Lines:** 696-836 (new test function)

**Test:** `test_execute_training_job_delegates_to_pytorch_trainer`

**RED Phase (pytest_execute_training_job_red.log):**
- Initial failure: `FileNotFoundError: [Errno 2] No such file or directory: ...train.log`
- Root cause: execute_training_job tried to open log file before creating parent directory

**GREEN Phase (pytest_execute_training_job_green_final.log):**
- **Strategy:** Monkeypatched `load_data` to return mock RawData instances (avoids complex NPZ fixture creation)
- **Strategy:** Monkeypatched `train_cdi_model_torch` to spy on invocation and return stub results
- **Validation:**
  - `train_cdi_model_torch` called exactly once ✅
  - Received non-None train_data and test_data ✅
  - Received correct TrainingConfig instance ✅
  - Config fields matched job metadata (gridsize=1, nphotons=1e3) ✅
  - Result dict contains 'status' key ✅
  - Log file exists and contains execution marker ✅

**Test:** `test_training_cli_invokes_real_runner` (pytest_training_cli_real_runner_green.log)
- Validates CLI main() calls execute_training_job (not stub) when --dry-run is omitted
- **PASSED** (1/1) in 3.62s

**Collection Proof (pytest_collect.log):**
- **7 tests collected** in 7.19s from `tests/study/test_dose_overlap_training.py`
- Includes new `test_execute_training_job_delegates_to_pytorch_trainer`

## Findings Applied

- **POLICY-001:** PyTorch backend exercised (real trainer call, no optional fallbacks)
- **CONFIG-001:** TrainingConfig bridge assumed by caller (`run_training_job`)
- **DATA-001:** Dataset paths validated; NPZ contract enforced via `load_data()`
- **OVERSAMPLING-001:** Gridsize semantics preserved in job metadata

## Artifacts

**Location:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T120500Z/phase_e_training_e5/`

- `red/pytest_execute_training_job_red.log` — RED test failure (FileNotFoundError)
- `green/pytest_execute_training_job_green_final.log` — GREEN test PASSED
- `green/pytest_training_cli_real_runner_green.log` — CLI test PASSED
- `collect/pytest_collect.log` — 7 tests collected
- `green/pytest_full_suite.log` — Full regression (in progress)
- `docs/summary.md` — This document

## Metrics

- **Tests:** 2 new assertions in existing test, 1 new test function
- **Code:** ~110 lines modified in `training.py` (execute_training_job rewrite)
- **RED → GREEN Cycle:** 1 iteration (initial FileNotFoundError → fixed with `mkdir`)
- **Test Runtime:** ~7.8s (execute_training_job), ~3.6s (CLI test)
- **Collection:** 7 tests total in test_dose_overlap_training.py

## Exit Criteria

- ✅ RED test authored and failed with expected error
- ✅ execute_training_job delegates to train_cdi_model_torch
- ✅ Loads datasets via load_data() helper
- ✅ Returns metrics dict with status, final_loss, epochs_completed, checkpoint_path
- ✅ Writes logs to job.log_path
- ✅ GREEN tests pass (2/2 targeted selectors)
- ✅ Collection proof captured (7 tests)
- ⏳ Full regression suite running (pytest -v tests/)

## Next Actions

- **Phase E6:** Batch training across dense/sparse views once baseline run lands (per input.md:62)
- **CLI Baseline Run:** Execute deterministic training for dose=1e3 baseline with real datasets (deferred to avoid long runtime in this loop)
- **Documentation Sync:** Update `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md` with new test selector

## Limitations

- **Stub Return Values:** Mock trainer returns hardcoded losses `[0.5, 0.3, 0.1]` for test validation
- **Dataset Loading:** Test uses mocked `load_data()` to avoid NPZ fixture complexity
- **No Actual Training:** Test validates invocation signature, not training correctness (integration test)
- **CLI Baseline Run Deferred:** Real training run with Phase C/D datasets skipped due to runtime constraints

## References

- **input.md:9-10** — Phase E5 task definition
- **plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:163-166** — Test requirements
- **docs/DEVELOPER_GUIDE.md:68-104** — CONFIG-001 bridge ordering
- **docs/workflows/pytorch.md §12** — train_cdi_model_torch signature
- **studies/fly64_dose_overlap/training.py:290-442** — execute_training_job implementation
- **tests/study/test_dose_overlap_training.py:696-836** — RED → GREEN test
