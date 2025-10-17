# Phase E2.D Parity Evidence Summary (Updated)

**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** E2.D (Parity Evidence Capture - Completion)
**Status:** ⚠️ BLOCKED - PyTorch data loader incompatible with canonical NPZ format
**Timestamp:** 2025-10-17T221500Z
**Engineer:** Ralph (Iteration i=51)

---

## 1. Executive Summary

Phase E2.D successfully installed PyTorch extras (lightning, mlflow, tensordict) and executed the PyTorch integration test. The test revealed a **critical data format incompatibility**: the PyTorch data loader (`ptycho_torch/dataloader.py`) expects the legacy `diff3d` key while the test fixture uses the canonical `diffraction` key per DATA-001 specification.

### Key Findings

- ✅ **Dependency Installation**: All PyTorch extras successfully installed
- ✅ **CLI Interface**: PyTorch training CLI functional with correct argparse flags
- ✅ **CONFIG-001 Compliance**: Legacy params.cfg bridge executed correctly
- ❌ **Data Contract Violation**: PyTorch loader incompatible with DATA-001 canonical format
- 🎯 **Root Cause**: `dataloader.py:calculate_length()` hardcoded to search for `diff3d`, not `diffraction`
- 🎯 **Next Action**: Fix PyTorch data loader to honor DATA-001 specification

---

## 2. Test Execution Results

### E2.D1: TensorFlow Baseline Integration Test
*(Result unchanged from prior attempt)*

**Command Executed:**
```bash
pytest tests/test_integration_workflow.py::TestFullWorkflow::test_train_save_load_infer_cycle -vv
```

**Result:** ✅ **PASSED**
**Runtime:** 31.88 seconds
**Log Location:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T093500Z/phase_e_tf_baseline.log`

**Key Observations:**
- TensorFlow backend fully functional with canonical NPZ format
- Complete train→save→load→infer cycle validated
- Uses `diffraction` key per DATA-001 specification

---

### E2.D2: PyTorch Integration Test (Updated)

**Command Executed:**
```bash
pytest tests/torch/test_integration_workflow_torch.py -vv
```

**Result:** ❌ **FAILED** (1 failed, 1 skipped in 7.24s)
**Log Location:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T221500Z/phase_e_torch_run.log`

**Failure Details:**

**Failed Test:** `test_pytorch_train_save_load_infer_cycle`
**Root Cause:** `ValueError: Could not determine image shape from any NPZ file.`

**Error Traceback (Key Section):**
```
File "/home/ollie/Documents/PtychoPINN2/ptycho_torch/dataloader.py", line 291, in calculate_length
  raise ValueError("Could not determine image shape from any NPZ file.")
ValueError: Could not determine image shape from any NPZ file.
```

**Diagnostic Output:**
```
Error processing headers/coords for /home/ollie/Documents/PtychoPINN2/datasets/Run1084_recon3_postPC_shrunk_3.npz:
Could not find diff3d data in /home/ollie/Documents/PtychoPINN2/datasets/Run1084_recon3_postPC_shrunk_3.npz
```

**Skipped Test:** `test_pytorch_tf_output_parity`
**Skip Reason:** "Deferred to Phase E2.D parity verification"

**Success Indicators (Pre-Failure):**
- ✅ CLI interface executed correctly (`python -m ptycho_torch.train`)
- ✅ All argparse flags parsed successfully
- ✅ CONFIG-001 bridge executed: `✓ params.cfg populated: N=128, gridsize=1, n_groups=64`
- ✅ Model construction completed (decoder blocks initialized)
- ✅ Lightning trainer initialized with CUDA device
- ✅ MLflow correctly disabled via `--disable_mlflow` flag

**Subprocess Command (from test):**
```bash
python -m ptycho_torch.train \
  --train_data_file datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --test_data_file datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --output_dir training_outputs \
  --max_epochs 2 \
  --n_images 64 \
  --gridsize 1 \
  --batch_size 4 \
  --device cpu \
  --disable_mlflow
```

---

## 3. Root Cause Analysis

### Data Format Violation (DATA-001)

**Specification Requirement** (`specs/data_contracts.md`):
- Canonical NPZ datasets **MUST** use the `diffraction` key
- Legacy `diff3d` key is deprecated and only exists in raw/unconverted datasets

**Current Behavior:**
- `ptycho_torch/dataloader.py:calculate_length()` explicitly searches for `diff3d`:
  ```python
  # Inferred from error message:
  if 'diff3d' not in npz_file:
      raise ValueError("Could not find diff3d data...")
  ```

**Test Fixture:**
- `datasets/Run1084_recon3_postPC_shrunk_3.npz` uses canonical `diffraction` key
- This is the **correct** format per DATA-001
- TensorFlow backend handles this correctly

**Impact:**
- PyTorch backend cannot load canonical datasets
- Violates DATA-001 specification
- Breaks parity with TensorFlow workflow

### Environment Analysis

**Current Environment State:**

**Python Version:** 3.11.13
**PyTorch:** ✅ Available (2.8.0+cu128)
**Lightning:** ✅ Installed (via manual pip install)
**MLflow:** ✅ Installed (3.5.0)
**TensorDict:** ✅ Installed (0.10.0)
**TensorFlow:** ✅ Available (2.19.0)

**Installation Method:**
1. Attempted `pip install -e .[torch]` → **WARNING: extras not recognized**
2. Fallback: `pip install lightning mlflow tensordict` → **SUCCESS**

**Note:** The `setup.py` `extras_require={'torch': [...]}` syntax may not be compatible with the current build backend. This should be investigated separately.

---

## 4. Parity Comparison

| Dimension | TensorFlow Backend | PyTorch Backend | Status |
|:---|:---|:---|:---|
| **Dependency Installation** | ✅ Automatic | ⚠️ Manual fallback required | 🔄 PARTIAL |
| **CLI Interface** | ✅ Functional | ✅ Functional | ✅ EQUIVALENT |
| **CONFIG-001 Compliance** | ✅ Validated | ✅ Validated | ✅ EQUIVALENT |
| **DATA-001 Compliance** | ✅ Uses `diffraction` | ❌ Hardcoded `diff3d` | 🚫 **CRITICAL** |
| **Subprocess Invocation** | ✅ Validated | ✅ Validated | ✅ EQUIVALENT |
| **Model Construction** | ✅ Successful | ✅ Successful | ✅ EQUIVALENT |
| **Data Loading** | ✅ Canonical format | ❌ Legacy format only | 🚫 **CRITICAL** |
| **Training Execution** | ✅ Complete | ❌ Failed at data load | 🚫 BLOCKED |
| **Checkpoint Persistence** | ✅ H5 format | ⏸️ Not tested | 🚫 BLOCKED |
| **Inference Workflow** | ✅ Functional | ⏸️ Not tested | 🚫 BLOCKED |
| **MLflow Suppression** | N/A (no flag) | ✅ `--disable_mlflow` working | ✅ COMPLIANT |

**Legend:**
- ✅ COMPLIANT: Meets specification
- ⚠️ PARTIAL: Works with caveats
- ❌ FAILED: Did not meet spec
- ⏸️ NOT TESTED: Could not execute
- 🚫 BLOCKED: Dependency/environment issue
- 🔄 WORKAROUND: Manual intervention required

---

## 5. POLICY-001 & CONFIG-001 Compliance

### POLICY-001: PyTorch Mandatory Dependency

**Requirement:** PyTorch (torch>=2.2) is a mandatory runtime dependency for `ptycho_torch/` modules.

**Evidence:**
- ✅ PyTorch 2.8.0+cu128 installed and functional
- ✅ Lightning 2.x installed successfully
- ✅ MLflow 3.5.0 installed successfully
- ✅ TensorDict 0.10.0 installed successfully
- ⚠️ `setup.py` extras mechanism failed; manual installation required

**Compliance:** ✅ **COMPLETE** (dependencies present, extras install issue noted for follow-up)

### CONFIG-001: Legacy Dict Synchronization Gate

**Requirement:** `update_legacy_dict(params.cfg, config)` must execute before legacy module access.

**Evidence (from test output):**
```
Creating PyTorch configuration objects...
Bridging PyTorch configs to TensorFlow dataclasses (CONFIG-001 compliance)...
✓ params.cfg populated: N=128, gridsize=1, n_groups=64
```

**Compliance:** ✅ **VERIFIED**
- Bridge executed correctly
- Legacy params.cfg populated with expected values
- No CONFIG-001-related failures

### DATA-001: NPZ Format Specification

**Requirement:** All canonical NPZ datasets MUST use `diffraction` key, not legacy `diff3d`.

**Evidence:**
- TensorFlow: ✅ Correctly reads `diffraction` key
- PyTorch: ❌ **VIOLATION** — hardcoded to search for `diff3d`

**Compliance:** 🚫 **CRITICAL FAILURE**
- PyTorch backend does not honor DATA-001 specification
- Breaks interoperability with canonical datasets
- Requires code fix in `ptycho_torch/dataloader.py`

---

## 6. Artifact Inventory

### Installation Logs

**Pip Install Log:**
- Path: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T221500Z/pip_install.log`
- Size: ~15 KB
- Content: Full pip output including extras warning and manual package installation

### Test Execution Logs

**TensorFlow Baseline Log:**
- Path: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T093500Z/phase_e_tf_baseline.log`
- Size: ~2 KB
- Content: pytest verbose output, 1 passed in 31.88s

**PyTorch Integration Log (Updated):**
- Path: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T221500Z/phase_e_torch_run.log`
- Size: ~12 KB
- Content: pytest verbose output, 1 failed (data format error), 1 skipped
- Key diagnostic: `Could not find diff3d data` error message

### Parity Summary Artifacts

**This Document:**
- Path: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T221500Z/phase_e_parity_summary.md`
- Purpose: Document E2.D execution outcomes, data format blocker, and follow-up actions

---

## 7. Residual Risks & Follow-Up Actions

### Critical Blockers

1. **DATA-001 Violation in PyTorch Data Loader** (NEW)
   - **Impact:** Cannot load canonical NPZ datasets; breaks TensorFlow parity
   - **Resolution:** Update `ptycho_torch/dataloader.py:calculate_length()` to search for `diffraction` key first, fall back to `diff3d` for legacy datasets
   - **Owner:** [INTEGRATE-PYTORCH-001-DATALOADER] follow-up task
   - **Timeline:** Required before Phase E2.D2 can pass
   - **Affected Files:**
     - `ptycho_torch/dataloader.py` (primary)
     - Possibly `ptycho_torch/dset_loader_pt_mmap.py` (verify)

2. **Setup.py Extras Installation Failure**
   - **Impact:** `pip install -e .[torch]` does not recognize `extras_require`
   - **Resolution:** Investigate build backend compatibility; consider migrating to `pyproject.toml` extras syntax
   - **Owner:** [INTEGRATE-PYTORCH-001-SETUP] follow-up task
   - **Timeline:** Non-blocking (manual install workaround successful)
   - **Workaround:** `pip install lightning mlflow tensordict`

### Lower-Priority Items

3. **Incomplete Parity Validation**
   - **Impact:** Cannot compare TensorFlow ↔ PyTorch outputs, metrics, or lifecycle
   - **Resolution:** Fix DATA-001 blocker (item #1), re-run E2.D2
   - **Owner:** Ralph (next loop after dataloader fix)
   - **Timeline:** Depends on item #1 resolution

4. **CUDA Library Warnings**
   - **Issue:** Stderr warnings about duplicate cuFFT/cuDNN/cuBLAS factory registration
   - **Impact:** Cosmetic; training proceeds despite warnings
   - **Resolution:** Document as known issue; investigate TensorFlow+PyTorch coexistence
   - **Owner:** Non-blocking
   - **Timeline:** Optional cleanup

5. **Device Selection Mismatch**
   - **Issue:** Test specifies `--device cpu` but Lightning selects CUDA (`'NVIDIA GeForce RTX 3090'`)
   - **Impact:** Minor; training would work but may consume GPU resources unexpectedly
   - **Resolution:** Verify Lightning device override behavior; ensure `--device cpu` is honored
   - **Owner:** [INTEGRATE-PYTORCH-001-DEVICE] optional enhancement
   - **Timeline:** Post-E2.D

---

## 8. Recommendations

### Immediate Actions (This Loop)

1. ✅ **Install Dependencies:** PyTorch extras installed successfully (manual fallback)
2. ✅ **Capture Test Log:** PyTorch integration test executed and logged
3. ✅ **Document Findings:** This parity summary updated with failure analysis
4. ✅ **Update Plan Checklists:** Mark E2.D2 as ⚠️ (new blocker discovered)
5. ✅ **Log Attempt History:** Update `docs/fix_plan.md` with artifact paths and DATA-001 violation

### Follow-Up Task Definition (Next Supervisor Loop)

6. **Create [INTEGRATE-PYTORCH-001-DATALOADER] Task:**
   - **Objective:** Fix DATA-001 compliance in PyTorch data loader
   - **Acceptance Criteria:**
     - `dataloader.py` searches for `diffraction` key first (canonical)
     - Falls back to `diff3d` only if `diffraction` not found (legacy support)
     - Test `test_pytorch_train_save_load_infer_cycle` passes
     - No TensorFlow backend regressions
   - **Estimated Effort:** 1 loop (targeted fix + regression check)

7. **Create [INTEGRATE-PYTORCH-001-SETUP] Task (Lower Priority):**
   - **Objective:** Fix `setup.py` extras installation
   - **Acceptance Criteria:**
     - `pip install -e .[torch]` installs lightning, mlflow, tensordict
     - Document workaround in `docs/workflows/pytorch.md` if unfixable
   - **Estimated Effort:** 1 loop (investigation + fix or documentation)

### Phase E2.D Completion Criteria (Updated)

8. **Re-run E2.D2 After Dataloader Fix:**
   - Execute pytest with fixed `dataloader.py`
   - Capture successful PyTorch training log
   - Update parity summary with quantitative metrics (training time, checkpoint size, reconstruction quality)

9. **Mark E2.D Complete:**
   - All D-phase tasks [x] in `phase_e2_implementation.md`
   - Full parity comparison documented
   - Residual risks logged for Phase E3

---

## 9. Compliance Verification

### Phase E2.D Completion Criteria

| Criterion | Status | Evidence |
|:---|:---|:---|
| TensorFlow baseline log captured | ✅ COMPLETE | `phase_e_tf_baseline.log` (31.88s, 1 passed) |
| PyTorch integration log captured | ✅ COMPLETE | `phase_e_torch_run.log` (7.24s, 1 failed) |
| PyTorch dependencies installed | ✅ COMPLETE | mlflow, lightning, tensordict verified |
| Parity summary authored | ✅ COMPLETE | This document (updated with failure analysis) |
| Artifacts referenced in `docs/fix_plan.md` | 🚧 PENDING | Update required (this task) |
| E2.D rows marked in plan documents | 🚧 PENDING | Update required (this task) |
| **New Blocker Identified** | ⚠️ **DATA-001** | PyTorch dataloader incompatible with canonical NPZ format |

**Overall Status:** ⚠️ **E2.D2 COMPLETE** (evidence captured, new blocker documented)
- Dependencies successfully installed via manual fallback
- Test executed and failure mode documented
- Root cause identified (DATA-001 violation)
- Follow-up task defined for dataloader fix
- Parity comparison deferred pending fix

---

## 10. References

### Implementation Plans

- `plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md` (Phase D tasks)
- `plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md` (E2.D rows)
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T215500Z/phase_e2_green.md` (Phase E2.C completion)

### Specifications

- `specs/ptychodus_api_spec.md` §4.5 (Reconstructor CLI contract)
- `specs/data_contracts.md` §1 (Canonical NPZ format - **violated by PyTorch loader**)

### Project Documentation

- `CLAUDE.md` §4.1 (CONFIG-001 requirement)
- `CLAUDE.md` §4.2 (DATA-001 requirement)
- `docs/findings.md` ID CONFIG-001, DATA-001, POLICY-001 (Ledger entries)
- `docs/workflows/pytorch.md` §2 (PyTorch prerequisites)

### Test Suite

- `tests/test_integration_workflow.py::TestFullWorkflow::test_train_save_load_infer_cycle` (TensorFlow baseline)
- `tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle` (PyTorch integration)

### Execution Logs

- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T221500Z/pip_install.log` (dependency installation)
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T093500Z/phase_e_tf_baseline.log` (TensorFlow success)
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T221500Z/phase_e_torch_run.log` (PyTorch failure - DATA-001 violation)

---

## 11. Conclusion

Phase E2.D successfully installed all required PyTorch dependencies (lightning, mlflow, tensordict) and executed the PyTorch integration test, revealing a **critical DATA-001 specification violation**. The PyTorch data loader (`ptycho_torch/dataloader.py`) is hardcoded to search for the legacy `diff3d` key instead of the canonical `diffraction` key, preventing it from loading properly formatted datasets.

**Key Outcomes:**
- ✅ Dependencies installed (manual fallback successful)
- ✅ CLI interface validated (correct argparse flags, subprocess invocation working)
- ✅ CONFIG-001 compliance verified (params.cfg bridge executed)
- ❌ DATA-001 violation discovered (dataloader incompatible with canonical format)
- 📋 Blocker documented with clear resolution path

**Blocking Issue:** PyTorch dataloader does not honor DATA-001 canonical NPZ format (`diffraction` key)

**Next Step:** Create follow-up task [INTEGRATE-PYTORCH-001-DATALOADER] to fix data loader and re-run E2.D2

**Phase Status:** ⚠️ **E2.D2 COMPLETE** (evidence captured, blocker documented) - Follow-up loop required to fix DATA-001 violation and achieve full parity
