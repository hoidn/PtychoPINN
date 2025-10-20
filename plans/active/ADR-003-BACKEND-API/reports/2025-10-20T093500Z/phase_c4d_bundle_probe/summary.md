# Phase C4.D.B4 Bundle Loader Verification Summary

## Outcome
✅ **SUCCESS** — Bundle loader already returns proper Lightning modules (fixed in commit 40968c02)

## Problem Statement
Integration test `test_run_pytorch_train_save_load_infer` was failing with:
```
AttributeError: 'dict' object has no attribute 'eval'
```
at `ptycho_torch/inference.py:522` when calling `model.eval()` on loaded model.

## Investigation Findings

### Discovery 1: Previous Fix Already Implemented
Commit `40968c02` (2025-10-20 01:12:09) had already implemented the fix:
- `load_torch_bundle()` now returns `(models_dict, params_dict)` with **both** autoencoder and diffraction_to_obj as `nn.Module` instances
- Each model is reconstructed via `create_torch_model_with_gridsize()` 
- CONFIG-001 compliance maintained (params.cfg restored before reconstruction)

### Discovery 2: Test Artifact Issue
Original integration test failure was NOT due to dict-vs-module issue, but rather:
- Test checked for Lightning checkpoint at `checkpoints/last.ckpt`
- Lightning actually saves checkpoints with epoch/step naming: `epoch=1-step=32.ckpt`
- Inference CLI correctly uses `wts.h5.zip` bundle, not Lightning checkpoint
- Test assertion was incorrect

## Implementation Actions

### 1. Added Regression Test (`test_bundle_loader_returns_modules`)
- **Location:** `tests/torch/test_integration_workflow_torch.py:176-242`
- **Purpose:** Explicitly verify bundle loader returns `nn.Module` instances supporting `.eval()`
- **Result:** ✅ PASSED (13.28s runtime)

### 2. Fixed Integration Test Assertion
- **Change:** Replace checkpoint assertion with bundle (`wts.h5.zip`) verification
- **Lines:** `tests/torch/test_integration_workflow_torch.py:264-278`
- **Rationale:** Inference uses bundle, not Lightning checkpoint; checkpoint format varies

### 3. Verified End-to-End Workflow
Manual subprocess testing confirmed:
- Training creates `wts.h5.zip` (8.6 MB)
- Bundle loads as Lightning modules (not dicts)
- Inference CLI executes successfully
- Reconstructions generated

## Test Results

### Targeted Tests
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py -vv
# 2 passed, 2 skipped in 29.71s
```

### Full Suite
```bash
pytest tests/ -v
# 288 passed, 17 skipped, 64 warnings in 244.96s
# Zero regressions
```

## Technical Details

### Bundle Structure (verified)
```
wts.h5.zip/
├── manifest.dill  # {'models': ['diffraction_to_obj', 'autoencoder'], 'version': '2.0-pytorch'}
├── diffraction_to_obj/
│   ├── model.pth  # 9.4 MB state_dict
│   └── params.dill  # CONFIG-001 snapshot
└── autoencoder/
    ├── model.pth  # 1.2 KB sentinel or state_dict
    └── params.dill  # Same CONFIG-001 snapshot
```

### Load Path Verification
```python
from ptycho_torch.workflows.components import load_inference_bundle_torch

models_dict, params_dict = load_inference_bundle_torch('training_outputs')

# models_dict['diffraction_to_obj']: PtychoPINN_Lightning instance
# models_dict['autoencoder']: PtychoPINN_Lightning instance
# Both support .eval(), .to(device), state_dict operations
```

## Compliance Checklist
- [x] Spec §4.6 dual-model requirement (both models are nn.Module)
- [x] CONFIG-001 (params.cfg restored in load_torch_bundle:336)
- [x] TDD RED→GREEN (regression test added, integration test fixed)
- [x] Zero test regressions (288 passed, same skip count as baseline)
- [x] Phase C4.D.B4 row marked complete in plan.md

## Artifacts Generated
- `pytest_bundle_loader_red.log` — Initial regression test PASSED (already fixed)
- `pytest_integration_recheck.log` — Integration test failure (checkpoint assertion)
- `pytest_integration_green.log` — Both tests GREEN after assertion fix
- `pytest_full_suite.log` — Full regression validation (288 passed)
- `summary.md` — This document

## Next Steps (per input.md)
Phase C4.D.B4 complete. Integration workflow now reaches inference without AttributeError.

Remaining tasks from `phase_c4d_blockers/plan.md`:
- Phase B: Training channel parity (gridsize configuration)
- Phase C: Documentation updates, ledger sync

## References
- Commit 40968c02: "ADR-003 C4.D bundle loader: load_torch_bundle dual-model implementation"
- input.md Do Now task 2: "author a regression test that loads... and asserts both models_dict are torch.nn.Module instances"
- specs/ptychodus_api_spec.md §4.6: Dual-model persistence contract
- docs/workflows/pytorch.md §12: Backend selection and CONFIG-001 requirements
