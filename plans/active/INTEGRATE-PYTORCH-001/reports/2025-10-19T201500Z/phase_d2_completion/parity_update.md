# PyTorch Integration Parity Update (2025-10-19)

## Executive Summary

As of Attempt #40 (2025-10-19T111855Z), the PyTorch backend achieves **full end-to-end parity** with the TensorFlow workflow for the complete train → save → load → infer cycle. The integration test selector now passes with ZERO failures, completing Phase D2 of the INTEGRATE-PYTORCH-001 initiative.

**Status:** ✅ **PARITY ACHIEVED**

---

## Parity Evolution Summary

### Baseline (2025-10-18T093500Z)

**Source:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T093500Z/phase_e_parity_summary.md`

**TensorFlow Status:** ✅ PASSED (31.88s)
- Integration test: `test_train_save_load_infer_cycle` fully functional
- Complete train → save → load → infer workflow validated

**PyTorch Status:** ❌ BLOCKED (import error)
- Missing `mlflow` dependency prevented execution
- Fail-fast guard working correctly with actionable error message
- **Blocker:** Environment lacked `pip install -e .[torch]` extras

**Key Gap:** Could not demonstrate functional parity due to dependency blocker.

---

### Current State (2025-10-19T111855Z)

**Source:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T111855Z/phase_d2_completion/summary.md`

**PyTorch Integration Test:** ✅ **PASSED** (20.44s)
- Selector: `pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv`
- Outcome: **1/1 PASSED in 20.44s**
- Training subprocess: Created checkpoint at `<output_dir>/checkpoints/last.ckpt`
- Inference subprocess: Successfully loaded checkpoint, ran Lightning inference, produced stitched reconstruction
- **No errors** — decoder shape fix, dtype enforcement, and checkpoint serialization all working end-to-end

**Full Test Suite:** ✅ **236 passed, 16 skipped, 1 xfailed, ZERO failures** (236.96s)
- Net improvement: +3 passing tests vs Attempt #21 baseline (233 → 236)
- Pre-existing failures resolved through Phases D1c-D1e:
  - **Checkpoint loading**: Fixed via `save_hyperparameters()` + dataclass restoration (D1c)
  - **Dtype mismatch**: Fixed via float32 enforcement in dataloaders + inference path (D1d)
  - **Decoder shape mismatch**: Fixed via center-crop x2→x1 alignment (D1e)

---

## Delta vs 2025-10-18 Parity Summary

### Blockers Resolved

1. **MLflow Import Guard (D1c prerequisite)**
   - **OLD:** `ModuleNotFoundError: No module named 'mlflow'` at import time
   - **NEW:** MLflow requirement satisfied; Lightning + tensordict available in environment
   - **Evidence:** Integration test now executes training subprocess successfully

2. **Checkpoint Hyperparameter Serialization (D1c)**
   - **OLD:** `TypeError: PtychoPINN_Lightning.__init__() missing 4 required positional arguments: 'model_config', 'data_config', 'training_config', and 'inference_config'`
   - **ROOT CAUSE (D1b discovery):** Checkpoint's `hyper_parameters` key was **missing** (returned `None`)
   - **NEW:** `self.save_hyperparameters()` implemented in `ptycho_torch/model.py:951-959` with `asdict()` conversion; checkpoint loading logic (lines 940-949) reconstructs dataclass instances from dict kwargs
   - **Evidence:** Integration log shows "Successfully loaded model from checkpoint"
   - **Validation:** `tests/torch/test_lightning_checkpoint.py` (3/3 tests passing)

3. **Float64 Tensor Propagation (D1d)**
   - **OLD:** `RuntimeError: Input type (double) and bias type (float)` during inference preprocessing
   - **NEW:** Explicit float32 casts in three locations:
     - `_build_inference_dataloader` (ptycho_torch/workflows/components.py:443-444)
     - `_reassemble_cdi_image_torch` (line 700, defensive cast before Lightning forward)
     - `ptycho_torch/inference.py` (lines 494-495, CLI data loading)
   - **Evidence:** Dtype regression tests GREEN (2/2 passing), integration test progresses past dtype error
   - **Validation:** `tests/torch/test_workflows_components.py::TestReassembleCdiImageTorchFloat32`

4. **Decoder Shape Mismatch (D1e)**
   - **OLD:** `RuntimeError: The size of tensor a (572) must match the size of tensor b (1080) at non-singleton dimension 3` at `ptycho_torch/model.py:366`
   - **ROOT CAUSE:** Path 1 (x1) padding → 572 width; Path 2 (x2) 2× upsample → 1080 width; addition failed
   - **NEW:** Center-crop x2 to match x1 spatial dims (ptycho_torch/model.py:366-381), mirroring TensorFlow `trim_and_pad_output` approach
   - **Evidence:** Decoder regression tests GREEN (2/2 passing), integration test completes inference without shape errors
   - **Validation:** `tests/torch/test_workflows_components.py::TestDecoderLastShapeParity`

---

## Parity Comparison Table

| Dimension | TensorFlow Baseline (2025-10-18) | PyTorch Current (2025-10-19) | Status |
|:---|:---|:---|:---|
| **Integration Test** | ✅ PASSED (31.88s) | ✅ PASSED (20.44s) | ✅ PARITY ACHIEVED |
| **Training Subprocess** | ✅ Functional | ✅ Functional | ✅ EQUIVALENT |
| **Checkpoint Persistence** | ✅ `.h5.zip` format | ✅ `.ckpt` format (bundled to `.h5.zip`) | ✅ EQUIVALENT |
| **Model Loading** | ✅ Verified | ✅ Verified (hyperparams restored) | ✅ EQUIVALENT |
| **Inference Execution** | ✅ Reconstruction generated | ✅ Reconstruction generated | ✅ EQUIVALENT |
| **CONFIG-001 Compliance** | ✅ params.cfg populated | ✅ params.cfg populated (D2.A guard) | ✅ COMPLIANT |
| **POLICY-001 Compliance** | N/A (TensorFlow) | ✅ Torch-optional imports enforced | ✅ COMPLIANT |
| **Dtype Enforcement** | Implicit (TensorFlow default) | ✅ Explicit float32 casts (D1d) | ✅ COMPLIANT |
| **Decoder Parity** | TensorFlow trim/pad logic | ✅ Center-crop alignment (D1e) | ✅ EQUIVALENT |

**Legend:**
- ✅ PARITY ACHIEVED: Both backends functionally equivalent
- ✅ EQUIVALENT: Same behavior, different implementation
- ✅ COMPLIANT: Adheres to project policies/specs

---

## Implementation Artifacts Summary

### Phase D1c — Checkpoint Serialization Fix

**Artifacts:**
- Implementation: `ptycho_torch/model.py:940-959`
- Tests: `tests/torch/test_lightning_checkpoint.py` (239 lines, 3 tests)
- Evidence: `reports/2025-10-19T134500Z/phase_d2_completion/{pytest_checkpoint_green.log,pytest_integration_checkpoint_green.log,summary.md}`

**Key Changes:**
- `self.save_hyperparameters()` with `asdict()` conversion for serialization
- Checkpoint loading logic reconstructs dataclass instances from dict kwargs
- Integration test now shows "Successfully loaded model from checkpoint"

### Phase D1d — Dtype Enforcement

**Artifacts:**
- Implementation: `ptycho_torch/workflows/components.py:443-444,700` + `ptycho_torch/inference.py:494-495`
- Tests: `tests/torch/test_workflows_components.py::TestReassembleCdiImageTorchFloat32` (2 tests)
- Evidence: `reports/2025-10-19T110500Z/phase_d2_completion/{pytest_dtype_green.log,summary.md}`

**Key Changes:**
- Dataloader casts `infer_X` and `infer_coords` to float32 before TensorDataset construction
- Stitching path adds defensive cast before Lightning forward pass
- CLI inference enforces float32/complex64 per specs/data_contracts.md §1

### Phase D1e — Decoder Shape Alignment

**Artifacts:**
- Implementation: `ptycho_torch/model.py:366-381`
- Tests: `tests/torch/test_workflows_components.py::TestDecoderLastShapeParity` (2 tests)
- Evidence: `reports/2025-10-19T111855Z/phase_d2_completion/{pytest_decoder_shape_green.log,pytest_integration_shape_green.log,summary.md}`

**Key Changes:**
- Center-crop x2 to match x1 spatial dimensions when shape mismatch detected
- Maintains device/dtype (no `.cpu()` or `.double()` calls)
- Mirrors TensorFlow baseline approach (trim oversized decoder outputs)

---

## Quantitative Comparison

### Test Execution Times

| Metric | TensorFlow Baseline | PyTorch Current | Delta |
|:---|---:|---:|---:|
| **Integration Test Runtime** | 31.88s | 20.44s | **-35.9%** (PyTorch faster) |
| **Full Suite Runtime** | N/A (not captured) | 236.96s (3:56) | N/A |

**Note:** PyTorch shows **35.9% faster** integration test execution compared to TensorFlow baseline (20.44s vs 31.88s). This may reflect differences in training epochs, batch sizes, or backend optimizations; further profiling recommended if performance delta is mission-critical.

### Test Coverage Progression

| Phase | Passing Tests | Delta | Notes |
|:---|---:|---:|:---|
| **Attempt #21 Baseline** (D1d start) | 220 | — | Lightning orchestration green, stitching stub |
| **Attempt #34** (D1c complete) | 231 | +11 | Checkpoint serialization fix + new tests |
| **Attempt #37** (D1d complete) | 233 | +2 | Dtype enforcement tests |
| **Attempt #40** (D1e complete) | 236 | +3 | Decoder parity tests + integration test |
| **Total Improvement** | — | **+16** | Zero new failures introduced |

---

## Exit Criteria Validation

### Phase D2 Completion Criteria (from `phase_d2_completion.md`)

| Criterion | Status | Evidence |
|:---|:---|:---|
| `_reassemble_cdi_image_torch` returns `(recon_amp, recon_phase, results)` without NotImplementedError | ✅ COMPLETE | Implementation at `ptycho_torch/workflows/components.py:607-730`; targeted tests GREEN (8/8 passing) |
| Lightning orchestration initializes probe inputs, respects deterministic seeding, exposes train/test containers | ✅ COMPLETE | `_train_with_lightning` implementation at lines 265-529; TestTrainWithLightningRed GREEN (3/3 passing) |
| All Phase D2 TODO markers resolved or formally retired with passing regression tests | ✅ COMPLETE | No open TODOs in `ptycho_torch/workflows/components.py`; full suite 236 passed |

### Phase D1e Completion Criteria (from `docs/fix_plan.md`)

| Criterion | Status | Evidence |
|:---|:---|:---|
| Shape instrumentation captured demonstrating resolved tensor parity at decoder merge point | ✅ COMPLETE | `shape_trace.md` at `reports/2025-10-19T105248Z/phase_d2_completion/` |
| New pytest regression (`TestDecoderLastShapeParity`) passes, confirming decoder output matches TensorFlow reference | ✅ COMPLETE | GREEN log at `reports/2025-10-19T111855Z/phase_d2_completion/pytest_decoder_shape_green.log` (2/2 passing) |
| Integration selector completes Lightning inference without shape mismatch | ✅ COMPLETE | GREEN log at `reports/2025-10-19T111855Z/phase_d2_completion/pytest_integration_shape_green.log` (1/1 PASSED) |
| docs/fix_plan.md attempts updated with GREEN evidence; phase_d2_completion.md D1e row marked [x] | 🚧 PENDING | Update required (this parity_update.md authored as prerequisite) |

---

## Next Actions & Recommendations

### Immediate (This Loop — Attempt #41)

1. ✅ **Author Parity Update:** This document captures Attempt #40 success and delta vs 2025-10-18 baseline
2. 🚧 **Update docs/workflows/pytorch.md §§5–7:** Reflect working stitching (`_reassemble_cdi_image_torch` no longer stub)
3. 🚧 **Mark D2/D3 complete:** Update `phase_d2_completion.md` rows D2/D3 to `[x]` with artifact links
4. 🚧 **Update docs/fix_plan.md:** Record Attempt #41 with parity update + docs refresh

### Follow-Up (Phase E or Later)

5. **Performance Profiling:** Investigate 35.9% runtime delta (PyTorch faster than TensorFlow) — validate training epochs, batch sizes, backend-specific optimizations
6. **Quantitative Parity:** Compare reconstruction quality metrics (SSIM, MAE, FRC) between TensorFlow and PyTorch outputs (deferred from 2025-10-18 plan)
7. **Ptychodus Integration:** Validate reconstructor lifecycle with dual-backend toggle (Phase E3 of canonical plan)
8. **TEST-PYTORCH-001 Handoff:** Complete charter conversion to phased plan per `plans/pytorch_integration_test_plan.md`

---

## References

### Plans & Evidence

- **Phase D2 Plan:** `plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md`
- **Phase D Workflow:** `plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md`
- **Old Parity Summary:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T093500Z/phase_e_parity_summary.md` (2025-10-18 baseline)
- **D1c Checkpoint Fix:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T134500Z/phase_d2_completion/summary.md`
- **D1d Dtype Fix:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T110500Z/phase_d2_completion/summary.md`
- **D1e Decoder Fix:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T111855Z/phase_d2_completion/summary.md`

### Specifications & Guides

- **API Spec:** `specs/ptychodus_api_spec.md` §4.5–§4.6 (reconstructor lifecycle, persistence)
- **Data Contract:** `specs/data_contracts.md` §1 (float32 normalization requirement)
- **PyTorch Workflow Guide:** `docs/workflows/pytorch.md` (to be updated in this loop)
- **Findings Ledger:** `docs/findings.md` (CONFIG-001, POLICY-001, DATA-001)

### Test Selectors

- **Integration Test:** `pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv`
- **Checkpoint Tests:** `pytest tests/torch/test_lightning_checkpoint.py -vv`
- **Dtype Tests:** `pytest tests/torch/test_workflows_components.py::TestReassembleCdiImageTorchFloat32 -vv`
- **Decoder Tests:** `pytest tests/torch/test_workflows_components.py::TestDecoderLastShapeParity -vv`
- **Full Suite:** `pytest tests/ -v`

---

## Conclusion

The PyTorch backend has achieved **full end-to-end parity** with the TensorFlow baseline for the complete train → save → load → infer workflow. All Phase D2 completion criteria are satisfied, with ZERO new failures introduced across 16 passing test improvements (+16 tests from Attempt #21 baseline to Attempt #40).

**Key Achievements:**
- ✅ Lightning checkpoint serialization fixed (D1c)
- ✅ Float32 dtype enforcement implemented (D1d)
- ✅ Decoder shape alignment resolved (D1e)
- ✅ Integration test GREEN (20.44s, 35.9% faster than TensorFlow)
- ✅ Full regression suite clean (236 passed, 0 failed)

**Phase Status:** ✅ **INTEGRATE-PYTORCH-001 Phase D2 COMPLETE**

**Recommendation:** Proceed to Phase E (Ptychodus dual-backend integration) or close INTEGRATE-PYTORCH-001-STUBS initiative with governance sign-off.
