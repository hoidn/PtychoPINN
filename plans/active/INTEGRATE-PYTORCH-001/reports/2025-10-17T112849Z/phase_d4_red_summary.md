# Phase D4.B Red-Phase Summary — PyTorch Persistence & Orchestration Regression Tests

## Execution Date
2025-10-17

## Objective
Author torch-optional failing regression tests for PyTorch persistence (Phase D4.B1) and workflow orchestration (Phase D4.B2) that document expected behaviors and serve as exit criteria for Phase D4.C implementation work.

## Tests Authored

### D4.B1 — Persistence Round-Trip Test

**Test:** `tests/torch/test_model_manager.py::TestLoadTorchBundle::test_load_round_trip_returns_model_stub`

**Selector:**
```bash
export CUDA_VISIBLE_DEVICES="" && export MLFLOW_TRACKING_URI=memory && \
pytest tests/torch/test_model_manager.py::TestLoadTorchBundle::test_load_round_trip_returns_model_stub -vv
```

**Status:** XFAIL (expected)

**Failure Mode:**
```python
NotImplementedError("load_torch_bundle model reconstruction not yet implemented")
```

**What It Tests:**
- Round-trip persistence: `save_torch_bundle` → `load_torch_bundle` → model + params restoration
- CONFIG-001 compliance: `params.cfg` must be restored during load before model reconstruction
- Sentinel model handling (torch-optional: works without actual torch.nn.Module instances)

**Red-Phase Rationale:**
`load_torch_bundle` currently raises `NotImplementedError` because Phase D3.C model reconstruction logic is pending. The test uses `pytest.xfail` to document this expected state and will pass once the loader implementation is complete.

**Log Path:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T112849Z/phase_d4_red_persistence.log`

---

### D4.B2.1 — Orchestration Persistence Test

**Test:** `tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_persists_models`

**Selector:**
```bash
export CUDA_VISIBLE_DEVICES="" && export MLFLOW_TRACKING_URI=memory && \
pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_persists_models -vv
```

**Status:** FAILED (expected)

**Failure Mode:**
```python
AttributeError: 'module' object at ptycho_torch.workflows.components has no attribute 'save_torch_bundle'
```

**What It Tests:**
- Orchestration persistence wiring: `run_cdi_example_torch` must call `save_torch_bundle` when `config.output_dir` is set
- Dual-model bundle requirement: persistence function receives both `autoencoder` and `diffraction_to_obj` models
- TensorFlow baseline parity per `ptycho/workflows/components.py:709-723`

**Red-Phase Rationale:**
The test fails during monkeypatch setup because `ptycho_torch/workflows/components.py` does not yet import or invoke `save_torch_bundle`. Phase D4.C1 will add the persistence call after training completes successfully.

**Log Path:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T112849Z/phase_d4_red_workflows.log` (lines 1-135)

---

### D4.B2.2 — Inference Bundle Loader Test

**Test:** `tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_load_inference_bundle_handles_bundle`

**Selector:**
```bash
export CUDA_VISIBLE_DEVICES="" && export MLFLOW_TRACKING_URI=memory && \
pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_load_inference_bundle_handles_bundle -vv
```

**Status:** FAILED (expected)

**Failure Mode:**
```python
AttributeError: 'module' object at ptycho_torch.workflows.components has no attribute 'load_torch_bundle'
```

**What It Tests:**
- Inference loader delegation: `load_inference_bundle_torch` must delegate to `load_torch_bundle` shim
- CONFIG-001 compliance: `params.cfg` must be restored during load (via spy validation)
- Return signature: `(models_dict, params_dict)` tuple matching TensorFlow baseline per `ptycho/workflows/components.py:94-174`

**Red-Phase Rationale:**
The test fails during monkeypatch setup because `ptycho_torch/workflows/components.py` does not yet import or define `load_inference_bundle_torch`. Phase D4.C2 will add the loader function with delegation to `load_torch_bundle`.

**Log Path:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T112849Z/phase_d4_red_workflows.log` (lines 136-276)

---

## Environment Configuration

All tests executed with torch-optional CPU-only configuration to avoid GPU/network dependencies:

```bash
export CUDA_VISIBLE_DEVICES=""        # Hide GPUs from CUDA
export MLFLOW_TRACKING_URI=memory     # Disable MLflow network calls
```

**Torch Availability:** PyTorch 2.8.0+cu128 detected but tests run in fallback mode (sentinel dicts instead of nn.Module instances for persistence tests).

---

## Failing Assertions Summary

| Test ID | Assertion Type | Current Behavior | Expected Behavior (Phase D4.C) |
|---------|----------------|------------------|-------------------------------|
| D4.B1 | NotImplementedError handling | `load_torch_bundle` raises NotImplementedError | Returns `(model, params)` tuple with CONFIG-001 restoration |
| D4.B2.1 | Attribute existence | `save_torch_bundle` not imported in workflows.components | Import and invoke after training when `output_dir` set |
| D4.B2.2 | Attribute existence | `load_torch_bundle` not imported in workflows.components | Import and delegate from `load_inference_bundle_torch` |

---

## Follow-Up Actions for Phase D4.C (Green Phase)

### D4.C1 — Persistence Wiring

**Implementation Targets:**
1. `ptycho_torch/workflows/components.py`:
   - Import `save_torch_bundle` from `ptycho_torch.model_manager`
   - Add persistence call in `run_cdi_example_torch` after training completes
   - Guard with `if config.output_dir is not None`
   - Pass `results['models']` dict + `config.output_dir / "wts.h5"` base path

**Validation:**
```bash
pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_persists_models -vv
```
Expected: PASSED

### D4.C2 — Loader Delegation

**Implementation Targets:**
1. `ptycho_torch/workflows/components.py`:
   - Import `load_torch_bundle` from `ptycho_torch.model_manager`
   - Define `load_inference_bundle_torch(bundle_dir)` function
   - Delegate to `load_torch_bundle(bundle_dir / "wts.h5", model_name='diffraction_to_obj')`
   - Return `(models_dict, params_dict)` tuple

**Validation:**
```bash
pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_load_inference_bundle_handles_bundle -vv
```
Expected: PASSED

### D4.C3 — Full Regression Check

**After D4.C1 + D4.C2 complete:**
```bash
# Run all Phase D4 regression tests
export CUDA_VISIBLE_DEVICES="" && export MLFLOW_TRACKING_URI=memory && \
pytest tests/torch/test_model_manager.py::TestLoadTorchBundle::test_load_round_trip_returns_model_stub \
       tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_persists_models \
       tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_load_inference_bundle_handles_bundle \
       -vv
```
Expected: 3/3 PASSED (or 1 XFAIL if D3.C still pending)

---

## Blockers & Dependencies

### No Blocking Issues Identified

All tests are properly torch-optional and fail gracefully with expected errors. No missing dependencies or environment issues observed.

### Dependencies for Green Phase

1. **Phase D3.C Completion (Optional):** `load_torch_bundle` model reconstruction implementation. If not complete, D4.B1 test will remain XFAIL (acceptable interim state).

2. **Persistence Imports:** Phase D4.C1/C2 require adding imports to `ptycho_torch/workflows/components.py`:
   ```python
   from ptycho_torch.model_manager import save_torch_bundle, load_torch_bundle
   ```

3. **Training Results Schema:** D4.C1 assumes `train_cdi_model_torch` returns `results['models']` dict with dual-model structure. Current stub implementation may need adjustment if not already present.

---

## Artifact Storage

All Phase D4.B artifacts stored under:
```
plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T112849Z/
├── phase_d4_red_persistence.log       (6 lines, XFAIL output)
├── phase_d4_red_workflows.log         (276 lines, 2 FAILED outputs)
└── phase_d4_red_summary.md            (this file)
```

---

## Next Steps

1. **Update Checklist:** Mark D4.B1, D4.B2, D4.B3 rows as `[P]` in `plans/active/INTEGRATE-PYTORCH-001/phase_d4_regression.md`
2. **Update Ledger:** Log Attempt #56 in `docs/fix_plan.md` for INTEGRATE-PYTORCH-001 with artifact references
3. **Execute D4.C:** Implement persistence/loader wiring per follow-up actions above
4. **Coordinate TEST-PYTORCH-001:** Once D4.C complete, activate subprocess integration test plan per Phase D4.A alignment strategy

---

## Test Statistics

- **Total Tests:** 3
- **XFAIL:** 1 (expected, documented)
- **FAILED:** 2 (expected, AttributeError during setup)
- **Runtime:** ~10s total (CPU-only, no GPU overhead)
- **Torch-Optional:** ✅ All tests importable without PyTorch runtime

---

## References

- **Spec Contracts:** `specs/ptychodus_api_spec.md:192-202` (persistence), `specs/ptychodus_api_spec.md:94-174` (inference loader)
- **TensorFlow Baseline:** `ptycho/workflows/components.py:676-723` (orchestration), `ptycho/workflows/components.py:94-174` (loader)
- **Phase D3 Evidence:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T104700Z/phase_d3_callchain/static.md`
- **Phase D4 Plan:** `plans/active/INTEGRATE-PYTORCH-001/phase_d4_regression.md`
- **Selector Map:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T111700Z/phase_d4_selector_map.md`
