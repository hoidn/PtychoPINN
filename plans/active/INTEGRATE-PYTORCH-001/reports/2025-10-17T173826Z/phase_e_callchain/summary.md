# Phase E1.A Callchain Analysis Summary

## Analysis Question

**How does Ptychodus select and invoke the TensorFlow backend, and what would a PyTorch backend selection mechanism require?**

---

## One-Page Executive Summary

### Current State: No Explicit Backend Selection

The PtychoPINN system currently defaults to TensorFlow by importing workflow functions from `ptycho.workflows.components`. There is **no backend selection mechanism** — the choice is implicit based on which module is imported.

### TensorFlow Entry Flow (Existing)

```
External Caller
  ↓
run_cdi_example(train_data, test_data, config, ...)
  ↓
[CONFIG-001 GATE] update_legacy_dict(params.cfg, config)
  ↓
train_cdi_model() → train_pinn.train_eval() [TensorFlow/Keras]
  ↓
(Optional) reassemble_cdi_image() → nbutils + tf_helper
  ↓
ModelManager.save_multiple_models() → wts.h5.zip
```

**Critical Points**:
- CONFIG-001 gate at `ptycho/workflows/components.py:706`
- Single entry point: `run_cdi_example()`
- Implicit TensorFlow selection via module import

### PyTorch Parallel Implementation (Phase D Complete)

The PyTorch backend provides **API-identical entry points** in `ptycho_torch.workflows.components`:

| Component | TensorFlow | PyTorch | Status |
|-----------|-----------|---------|--------|
| Entry signatures | `run_cdi_example()` | `run_cdi_example_torch()` | ✅ Identical |
| CONFIG-001 gates | Line 706 (training) | Lines 161 + 265 (training + inference) | ✅ Complete |
| Data adapters | RawData → PtychoDataContainer | RawData → RawDataTorch → PtychoDataContainerTorch | ✅ Phase C complete |
| Training | `train_pinn.train_eval()` | `_train_with_lightning()` | 🔴 Stub (D2.B) |
| Inference | `reassemble_cdi_image()` | `_reassemble_cdi_image_torch()` | 🔴 NotImplementedError (D2.C) |
| Persistence | `ModelManager.save/load` | `save_torch_bundle/load_torch_bundle` | ✅ D3.B complete, D3.C partial |

**Torch-Optional Design**: PyTorch workflows are importable without torch via guarded imports; errors raised only when torch-specific functions invoked.

---

## First Implementation Steps for Phase E1.C

### Recommended Backend Selection Strategy

**Option: Configuration Flag + Import Dispatch**

1. **Add `backend` field to config dataclasses**:
   ```python
   # ptycho/config/config.py
   @dataclass
   class TrainingConfig:
       # ... existing fields ...
       backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'

   @dataclass
   class InferenceConfig:
       # ... existing fields ...
       backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'
   ```

2. **Ptychodus reconstructor imports appropriate workflow**:
   ```python
   # Pseudo-code for Ptychodus integration
   if config.backend == 'pytorch':
       try:
           from ptycho_torch.workflows.components import run_cdi_example_torch as run_cdi_example
       except ImportError:
           raise RuntimeError("PyTorch backend selected but ptycho_torch unavailable")
   else:
       from ptycho.workflows.components import run_cdi_example

   # Rest of code uses run_cdi_example() transparently
   ```

3. **Fallback behavior**: Raise immediate, actionable error if backend unavailable (fail-fast)

### Alternative Strategies (Deferred to E1.C Design)

- **Environment variable**: `PTYCHOPINN_BACKEND=pytorch` (system-wide default)
- **Separate reconstructor classes**: `PtychoPINNTorchReconstructor` vs `PtychoPINNTFReconstructor` (heavier weight)
- **Runtime autodetection**: Check torch availability, fall back to TensorFlow (too magical)

---

## Phase E1.B Next Steps

The Phase E1.B red tests should document:

1. **Default behavior**: System defaults to TensorFlow when `backend` unspecified
2. **PyTorch selection**: `backend='pytorch'` routes to `ptycho_torch.workflows.components`
3. **CONFIG-001 compliance**: `update_legacy_dict()` called before workflow dispatch
4. **Torch unavailability**: Actionable error raised when PyTorch selected but unavailable
5. **API parity**: Both backends accept identical function signatures

### Test Structure (Torch-Optional)

```python
# tests/torch/test_backend_selection.py
class TestBackendSelection:
    def test_selects_pytorch_backend(self, params_cfg_snapshot):
        """Verify config flag routes to PyTorch workflow."""
        config = TrainingConfig(backend='pytorch', ...)
        # Assert: imports ptycho_torch.workflows.components
        # Assert: update_legacy_dict called before dispatch
        # Expected: FAIL until Phase E1.C implementation

    def test_defaults_to_tensorflow_backend(self):
        """Verify backward compatibility when backend unspecified."""
        config = TrainingConfig(...)  # backend defaults to 'tensorflow'
        # Assert: imports ptycho.workflows.components
        # Expected: PASS (existing behavior)

    def test_pytorch_unavailable_raises_error(self):
        """Verify actionable error when torch unavailable."""
        # Mock: TORCH_AVAILABLE = False
        config = TrainingConfig(backend='pytorch', ...)
        # Assert: raises RuntimeError with installation guidance
        # Expected: FAIL until error handling implemented
```

---

## Open Questions for Phase E1.C Design

1. **Config field location**: Add `backend` to `TrainingConfig`/`InferenceConfig` or create separate `BackendConfig`?
2. **Import dispatch layer**: Implement in Ptychodus or within PtychoPINN library?
3. **TEST-PYTORCH-001 coordination**: What fixtures/selectors required for integration harness?
4. **Probe initialization**: Resolve architectural decision (PyTorch-specific vs shared implementation)?

---

## Artifacts Generated

| File | Purpose | Location |
|------|---------|----------|
| `static.md` | Full callchain analysis with file:line anchors | `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T173826Z/phase_e_callchain/` |
| `summary.md` | This executive summary | Same directory |
| `pytorch_workflow_comparison.md` | Detailed TensorFlow vs PyTorch feature matrix | `/home/ollie/Documents/PtychoPINN2/` (will be moved to reports/) |

---

## Status Update

- **Phase E1.A**: ✅ **COMPLETE** — Callchain evidence captured with CONFIG-001 gates documented
- **Phase E1.B**: 🔄 **IN PROGRESS** — Red tests pending (next task in this loop)
- **Phase E1.C**: ⏸️ **PENDING** — Design blueprint awaiting E1.B completion

**Time Spent**: ~25 minutes (within 30min budget)

**Next Action**: Author Phase E1.B torch-optional failing tests per red-test template above.
