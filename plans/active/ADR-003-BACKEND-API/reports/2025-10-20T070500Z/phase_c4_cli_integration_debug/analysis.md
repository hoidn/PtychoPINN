# ADR-003 C4.D3 Factory Channel Sync — Analysis

**Date:** 2025-10-20
**Loop:** Ralph Attempt #23 (TDD implementation loop)
**Focus:** Factory C_forward/C_model synchronization with data channel count
**Artifacts:** plans/active/ADR-003-BACKEND-API/reports/2025-10-20T070500Z/phase_c4_cli_integration_debug/

---

## Problem Statement

From supervisor debug (Attempt #22):
- Integration test (`test_run_pytorch_train_save_load_infer`) failed with coords_relative shape mismatch
- Error: `RuntimeError: shape '[16, 2, 1]' is invalid for input of size 8` in `ptycho_torch/helper.py:425`
- Root cause: `create_training_payload()` sets `pt_data_config.C = 1` (correct) but leaves `pt_model_config.C_forward = 4` (dataclass default)
- PyTorch helpers (`reassemble_patches_position_real`) assume `C_forward` channels, producing tensor dimension mismatch

**Spec requirement:**
Per `specs/data_contracts.md` §1, grouped coordinate data must have shape `(N, C, 1, 2)` where `C = gridsize²`. PyTorch model config must synchronize channel counts with data config to maintain parity with TensorFlow pipeline behavior.

**Architecture ADR:**
ADR-003-BACKEND-API mandates factory functions produce payload configs with consistent channel dimensions across DataConfig, ModelConfig, and TensorFlow bridge outputs.

---

## TDD Implementation (RED → GREEN)

### 1. Test Authorship (RED Phase)

Added `test_gridsize_sets_channel_count` to `tests/torch/test_config_factory.py:199-248`:
- **Case 1:** `gridsize=1` → asserts `C=1`, `C_forward=1`, `C_model=1`
- **Case 2:** `gridsize=2` → asserts `C=4`, `C_forward=4`, `C_model=4`
- **Case 3:** No override → asserts `C_forward == C` and `C_model == C` (any C ≥ 1)

**RED outcome:**
```
FAILED tests/torch/test_config_factory.py::TestTrainingPayloadStructure::test_gridsize_sets_channel_count
AssertionError: ModelConfig.C_forward should match DataConfig.C
assert 4 == 1
```

Confirmed mismatch: `pt_data_config.C = 1` but `pt_model_config.C_forward = 4` (dataclass default per `ptycho_torch/config_params.py:82`).

**Artifact:** `pytest_config_factory.log` (RED run, 1 FAILED in 3.82s)

### 2. Factory Implementation (GREEN Phase)

Modified `ptycho_torch/config_factory.py` in two locations:

**Training factory** (`create_training_payload`, lines 210-223):
Set `C_forward=C` and `C_model=C` when instantiating `PTModelConfig`.

**Inference factory** (`create_inference_payload`, lines 406-415):
Set `C_forward=C` and `C_model=C` when instantiating `PTModelConfig`.

**GREEN outcome:**
```
tests/torch/test_config_factory.py::TestTrainingPayloadStructure::test_gridsize_sets_channel_count PASSED [100%]
======================== 1 passed, 2 warnings in 3.66s =========================
```

All three test cases pass.

---

## Validation Results

### Config Factory Parity Test (Targeted)

**Selector:**
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py::TestTrainingPayloadStructure::test_gridsize_sets_channel_count -vv
```

**Result:** ✅ **PASSED in 3.66s**

### Integration Workflow Progress

**Selector:**
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
```

**Result:** ⚠️ **FAILED (different error) in 17.34s**

**Key Finding:** The coords_relative shape mismatch **is now resolved** — training subprocess completed successfully without tensor dimension errors.

**New failure mode:**
```
ValueError: Model archive not found: .../training_outputs/wts.h5.zip
```

**Interpretation:**
- Training saves `checkpoints/last.ckpt` (Lightning checkpoint format)
- Inference expects `wts.h5.zip` (TensorFlow archive format per spec §4.8)
- This is a **separate persistence issue**, NOT related to C4.D3 channel sync bug
- ADR-003 C4.D3 goal (factory channel synchronization) is **ACHIEVED**

**Artifact:** `pytest_integration.log` (17.34s runtime, training succeeded, inference failed on different issue)

### CLI Guardrail Tests

**Training CLI:** ✅ **6/6 PASSED in 4.95s**
**Inference CLI:** ✅ **4/4 PASSED in 4.57s**

---

## Exit Criteria Validation

Per `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md` C4.D3:

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Targeted factory test GREEN | ✅ | `test_gridsize_sets_channel_count` PASSED (3.66s) |
| Factory sync regression fixed | ✅ | Training subprocess completed without coords_relative error |
| CLI guardrails remain GREEN | ✅ | Training CLI 6/6 PASSED, Inference CLI 4/4 PASSED |
| Integration test coords error resolved | ✅ | Different failure mode (persistence), not tensor shape mismatch |

**C4.D3 Completion Status:** ✅ **COMPLETE**

---

## Summary

**What was implemented:**
1. Added regression test `test_gridsize_sets_channel_count` validating C_forward/C_model synchronization (TDD RED)
2. Updated `create_training_payload()` to set `C_forward=C` and `C_model=C` (lines 221-222)
3. Updated `create_inference_payload()` to set `C_forward=C` and `C_model=C` (lines 413-414)
4. Verified GREEN outcome: factory test PASSED, CLI tests PASSED, training subprocess no longer fails with coords_relative error

**What remains:**
- Integration test checkpoint format mismatch (separate issue, deferred)
- Phase C4.E documentation updates
- Phase C4.F ledger wrap-up

**Compliance:**
- ✅ CONFIG-001: Factory maintains update_legacy_dict() ordering
- ✅ DATA-001: Channel counts align with grouped coordinate shape `(N, C, 1, 2)`
- ✅ ADR-003: Factory payload configs internally consistent
- ✅ TDD discipline: RED test authored before implementation, GREEN validation completed
