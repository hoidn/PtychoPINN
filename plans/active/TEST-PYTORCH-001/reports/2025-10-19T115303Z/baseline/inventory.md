# TEST-PYTORCH-001 Phase A Baseline Inventory

**Date:** 2025-10-19
**Artifact Hub:** `plans/active/TEST-PYTORCH-001/reports/2025-10-19T115303Z/baseline/`
**Reviewer:** Ralph (automated baseline assessment)

---

## 1. Existing Coverage Assessment

### Test File Status
- **Location:** `tests/torch/test_integration_workflow_torch.py`
- **Test Class:** `TestPyTorchIntegrationWorkflow` (unittest.TestCase)
- **Primary Test:** `test_pytorch_train_save_load_infer_cycle` (lines 58-163)
- **Secondary Test:** `test_pytorch_tf_output_parity` (lines 166-182, SKIPPED, deferred to Phase E2.D)

### Test Implementation Style
**Finding:** Current test uses `unittest.TestCase` framework, NOT native pytest style.

**Implications:**
- Inconsistent with project standard (CLAUDE.md §4.3: "Write new tests using native pytest style")
- Mixing unittest setUp/tearDown with pytest parametrization would cause issues
- Test is subprocess-based, which is appropriate for integration tier
- Uses temp directory management via `tempfile.TemporaryDirectory()`

**Recommendation:** Modernize to pytest style during Phase C (TDD Cycle) to align with project conventions.

---

## 2. Current Test Status (Baseline Run)

### Execution Results
**Command:**
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv
```

**Outcome:** ✅ **PASSED** in 32.54s (CPU-only execution)

**Critical Observation:** Test is already GREEN and functional. This contradicts the Phase E2.B "RED phase" annotations in the test docstrings (lines 41-42, 65), which state "EXPECTED TO FAIL until Phase E2.C implementation."

### What the Test Validated
1. **Training subprocess** (`python -m ptycho_torch.train`) executed successfully
2. **Checkpoint artifact** created at expected location (one of the searched paths exists)
3. **Inference subprocess** (`python -m ptycho_torch.inference`) executed successfully
4. **Output artifacts** (reconstructed_amplitude.png, reconstructed_phase.png) created with non-trivial file sizes

### Implications for TEST-PYTORCH-001
- **No blocker exists** — the integration workflow is already functional
- **Phase A baseline confirms** readiness to proceed directly to fixture optimization (Phase B) or test modernization (Phase C)
- **Charter assumptions outdated** — `plans/pytorch_integration_test_plan.md` lines 11-12 state "PyTorch backend currently lacks a similar guardrail; existing notebooks and parity scaffolds are manual or partially broken." This is no longer accurate as of INTEGRATE-PYTORCH-001 Attempt #40 (Phase D2 completion).

---

## 3. Test Fixture & Dataset Analysis

### Dataset Used
**File:** `datasets/Run1084_recon3_postPC_shrunk_3.npz`
- **Size:** 35 MB (on-disk)
- **Format:** NPZ archive (NumPy compressed)
- **Known Issues:**
  - Legacy (H,W,N) transpose format (FORMAT-001 finding)
  - Auto-transpose heuristic implemented in dataloader (Attempt #1 of INTEGRATE-PYTORCH-001-DATALOADER-INDEXING)

### Runtime Budget
**Current Test Runtime:** 32.54s (CPU-only)
**Target Runtime:** ≤120s (2 minutes) per TEST-PYTORCH-001 charter §Acceptance Criteria

**Verdict:** ✅ Current runtime is **well under budget** (27% of target). No fixture minimization urgently required for CI performance, but smaller fixture may improve developer iteration speed.

---

## 4. CLI Parameter Coverage

### Training Command (from test lines 86-97)
```python
[sys.executable, "-m", "ptycho_torch.train",
 "--train_data_file", str(data_file),
 "--test_data_file", str(data_file),
 "--output_dir", str(training_output_dir),
 "--max_epochs", "2",
 "--n_images", "64",
 "--gridsize", "1",
 "--batch_size", "4",
 "--device", "cpu",
 "--disable_mlflow"]
```

**Parameter Audit:**
- `--train_data_file`, `--test_data_file`: Standard NPZ paths ✅
- `--output_dir`: Temp directory management ✅
- `--max_epochs`: Training duration control ✅
- `--n_images`: Sampling control (legacy alias for `n_groups` per specs §5.2) ✅
- `--gridsize`: Group cardinality ✅
- `--batch_size`: Dataloader config ✅
- `--device`: Explicit CPU enforcement ✅
- `--disable_mlflow`: MLflow suppression flag ✅

**Finding:** Test uses **10 CLI flags** that are all documented in `specs/ptychodus_api_spec.md` §5.2 TrainingConfig fields. No undocumented or experimental flags.

### Inference Command (from test lines 130-137)
```python
[sys.executable, "-m", "ptycho_torch.inference",
 "--model_path", str(training_output_dir),
 "--test_data", str(data_file),
 "--output_dir", str(inference_output_dir),
 "--n_images", "32",
 "--device", "cpu"]
```

**Parameter Audit:**
- `--model_path`: Checkpoint directory ✅
- `--test_data`: Inference NPZ input ✅
- `--output_dir`: Output destination ✅
- `--n_images`: Inference sampling control ✅
- `--device`: CPU enforcement ✅

**Finding:** Inference uses **5 CLI flags** that align with `InferenceConfig` fields (§5.3).

---

## 5. Known Blockers & Gaps (from Prior Attempts)

### Resolved Issues (GREEN in baseline)
1. **CONFIG-001 compliance** — `update_legacy_dict` executed before data access (enforced in workflows)
2. **POLICY-001 compliance** — PyTorch required, test passes without optional-import guards
3. **FORMAT-001 transpose** — Auto-transpose heuristic handles legacy NPZ format
4. **Checkpoint loading** — Lightning hyperparameter serialization fixed (INTEGRATE-PYTORCH-001 Attempt #34)
5. **Dtype enforcement** — float32 casting implemented (Attempt #37)
6. **Decoder shape mismatch** — Center-crop fix implemented (Attempt #40)

### Open Questions (from charter)
1. **Minimal fixture resolution** (charter line 54) — Current fixture already under budget; deferrable
2. **Inference module location** (charter line 55) — Already implemented as `ptycho_torch.inference` CLI
3. **Memmap management** (charter line 56) — Test uses temp directory cleanup; no persistent state issues observed

**Verdict:** All charter open questions have been resolved by INTEGRATE-PYTORCH-001 implementation work.

---

## 6. Test Harness Compatibility

### Current Framework
- **Style:** `unittest.TestCase` with setUp/tearDown lifecycle
- **Subprocess management:** Direct `subprocess.run()` calls with `capture_output=True`
- **Temp directory:** `tempfile.TemporaryDirectory()` context manager
- **Assertions:** `unittest` assertion methods (`assertEqual`, `assertTrue`, `assertGreater`)

### Alignment with CLAUDE.md §4.3
**Requirement:** "Do not mix unittest.TestCase classes with pytest parametrization. Write new tests using native pytest style."

**Finding:** Test violates project standard by using unittest.TestCase. However, no pytest parametrization is currently used, so no immediate compatibility issue exists.

**Recommendation for Phase C:** Modernize to pytest style:
```python
# Before (unittest)
class TestPyTorchIntegrationWorkflow(unittest.TestCase):
    def setUp(self): ...
    def test_pytorch_train_save_load_infer_cycle(self): ...

# After (pytest)
@pytest.fixture
def temp_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

def test_pytorch_train_save_load_infer_cycle(temp_output_dir): ...
```

---

## 7. Blockers Summary

### Confirmed Blockers: **ZERO**
The baseline run demonstrates full end-to-end functionality with no test failures.

### Deprecation Notices
1. **Test docstrings outdated** — Lines 41-42, 65 claim "EXPECTED TO FAIL" but test passes
2. **Charter assumptions stale** — `plans/pytorch_integration_test_plan.md` predates INTEGRATE-PYTORCH-001 completion

### Hygiene Improvements (non-blocking)
1. Modernize to pytest style (Phase C)
2. Update test docstrings to reflect GREEN status
3. Consider fixture minimization for faster developer iteration (optional)
4. Add explicit deterministic seeding assertions if reproducibility is critical

---

## 8. References

- **Baseline Log:** `plans/active/TEST-PYTORCH-001/reports/2025-10-19T115303Z/baseline/pytest_integration_current.log`
- **Test Source:** `tests/torch/test_integration_workflow_torch.py:58-163`
- **Charter:** `plans/pytorch_integration_test_plan.md`
- **Implementation Plan:** `plans/active/TEST-PYTORCH-001/implementation.md`
- **Prior Work:** `docs/fix_plan.md` INTEGRATE-PYTORCH-001 Attempts #10–#41

---

**Phase A Task A1 Status:** ✅ COMPLETE
**Next Step:** Document environment prerequisites and runtime observations (A3) in `summary.md`
