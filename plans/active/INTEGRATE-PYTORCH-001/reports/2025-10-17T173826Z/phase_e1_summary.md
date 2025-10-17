# Phase E1.A+B Evidence Summary (INTEGRATE-PYTORCH-001)

## Loop Summary

**Phase**: E1.A+B (Evidence-only)
**Date**: 2025-10-17
**Mode**: Parity (callchain + red tests)
**Status**: COMPLETE (evidence-only, no implementation)

---

## Objectives Completed

### E1.A: Callchain Analysis ✅

**Goal**: Map how the TensorFlow backend is currently invoked and document PyTorch parallel structure.

**Deliverables**:
1. `phase_e_callchain/static.md` — Comprehensive callgraph analysis with file:line anchors
2. `phase_e_callchain/summary.md` — Executive summary of findings
3. `phase_e_callchain/pytorch_workflow_comparison.md` — Feature parity matrix

**Key Findings**:
- **No explicit backend selection** currently exists; TensorFlow is default via module import
- **PyTorch workflows provide API-identical entry points** (`run_cdi_example_torch`, `train_cdi_model_torch`, `load_inference_bundle_torch`)
- **CONFIG-001 gates** present in both backends (TensorFlow: line 706; PyTorch: lines 161 + 265)
- **Phase D complete**: Config bridge, data adapters, persistence (save) all green
- **Phase D2.B/C blockers**: Training orchestration and inference/stitching still stubs

**Recommended Implementation Strategy**:
- Add `backend='tensorflow'` field to `TrainingConfig`/`InferenceConfig` (default: 'tensorflow')
- Import dispatcher reads config flag and imports appropriate workflow module
- Fail-fast with actionable error if PyTorch unavailable when `backend='pytorch'`

### E1.B: Red Tests ✅

**Goal**: Author torch-optional failing pytest cases documenting expected backend selection behavior.

**Deliverables**:
1. `tests/torch/test_backend_selection.py` — 6 red tests (all XFAIL)
2. `phase_e_red_backend_selection.log` — pytest output (6 xfailed in 3.16s)
3. Updated `tests/conftest.py` — added `test_backend_selection` to torch-optional whitelist

**Test Coverage**:
1. **Default behavior**: System defaults to TensorFlow when `backend` unspecified
2. **PyTorch selection**: `backend='pytorch'` routes to PyTorch workflows
3. **CONFIG-001 compliance**: `update_legacy_dict()` called before workflow dispatch
4. **Torch unavailability**: Actionable error raised when PyTorch selected but unavailable
5. **Inference support**: `InferenceConfig` also supports `backend` parameter
6. **API parity**: Both backends accept identical config signatures

**Test Status**: All 6 tests marked `@pytest.mark.xfail(strict=True)` with reason "Backend selection not yet implemented (Phase E1.C pending)"

---

## Artifacts Generated

| File | Purpose | Location |
|------|---------|----------|
| `static.md` | Full callchain analysis | `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T173826Z/phase_e_callchain/` |
| `summary.md` | Executive summary | Same directory |
| `pytorch_workflow_comparison.md` | TensorFlow vs PyTorch feature matrix | Same directory |
| `test_backend_selection.py` | Red tests (6 XFAIL) | `tests/torch/` |
| `phase_e_red_backend_selection.log` | pytest output | `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T173826Z/` |
| `phase_e1_summary.md` | This summary | Same directory |

---

## Phase E1 Exit Criteria

- [x] Phase E1.A: Callchain evidence captured with CONFIG-001 gates documented
- [x] Phase E1.B: Red tests authored and logged (6/6 XFAIL as expected)
- [ ] Phase E1.C: Design blueprint (deferred to next loop per input.md scope)

---

## Next Steps for Phase E1.C

### Implementation Blueprint Roadmap

1. **Add `backend` field to config dataclasses** (`ptycho/config/config.py`):
   ```python
   @dataclass
   class TrainingConfig:
       # ... existing fields ...
       backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'
   ```

2. **Create workflow dispatcher helper** (new module or in existing orchestration):
   ```python
   def get_workflow_module(config):
       """Import appropriate workflow module based on backend selection."""
       if config.backend == 'pytorch':
           try:
               from ptycho_torch.workflows import components
               return components
           except ImportError:
               raise RuntimeError(
                   "PyTorch backend selected but ptycho_torch unavailable. "
                   "Install with: pip install torch"
               )
       else:  # backend == 'tensorflow'
           from ptycho.workflows import components
           return components
   ```

3. **Update tests to pass** (make red → green):
   - Remove `@pytest.mark.xfail` markers
   - Verify default behavior (backward compatibility)
   - Validate PyTorch selection routes correctly
   - Confirm CONFIG-001 gates triggered
   - Test error handling for unavailable backends

4. **Documentation updates**:
   - `docs/workflows/pytorch.md` — Add backend selection instructions
   - `specs/ptychodus_api_spec.md` — Document dual-backend behavior (§4.1)
   - `docs/findings.md` — Log CONFIG-XXX finding for backend selection pattern

---

## Open Questions for Phase E1.C

1. **Config field location**: Add to existing dataclasses or create `BackendConfig`?
2. **Import dispatch layer**: Implement in Ptychodus or within PtychoPINN?
3. **Environment variable fallback**: Support `PTYCHOPINN_BACKEND=pytorch` override?
4. **Probe initialization**: Resolve architectural decision (PyTorch-specific vs shared)?

---

## Test Results

### Backend Selection Red Tests (Phase E1.B)

```
============================= test session starts ==============================
platform linux -- Python 3.11.13, pytest-8.4.1, pluggy-1.6.0
PyTorch: available
PyTorch version: 2.8.0+cu128

tests/torch/test_backend_selection.py::TestBackendSelection::test_defaults_to_tensorflow_backend XFAIL [ 16%]
tests/torch/test_backend_selection.py::TestBackendSelection::test_selects_pytorch_backend XFAIL [ 33%]
tests/torch/test_backend_selection.py::TestBackendSelection::test_pytorch_backend_calls_update_legacy_dict XFAIL [ 50%]
tests/torch/test_backend_selection.py::TestBackendSelection::test_pytorch_unavailable_raises_error XFAIL [ 66%]
tests/torch/test_backend_selection.py::TestBackendSelection::test_inference_config_supports_backend_selection XFAIL [ 83%]
tests/torch/test_backend_selection.py::TestBackendSelection::test_backend_selection_preserves_api_parity XFAIL [100%]

============================== 6 xfailed in 3.16s ==============================
```

**Status**: All expected XFAIL ✅ (Phase E1.B red phase complete)

---

## Time Budget

- **E1.A Callchain Analysis**: ~25 minutes
- **E1.B Red Tests**: ~10 minutes
- **Documentation**: ~5 minutes
- **Total**: ~40 minutes (within estimated budget)

---

## Phase E1 Status

- **E1.A**: ✅ COMPLETE — Callchain evidence captured
- **E1.B**: ✅ COMPLETE — Red tests authored and logged
- **E1.C**: ⏸️ PENDING — Design blueprint deferred to next loop

**Overall Phase E1 Status**: Evidence phase complete; ready for Phase E1.C implementation design.
