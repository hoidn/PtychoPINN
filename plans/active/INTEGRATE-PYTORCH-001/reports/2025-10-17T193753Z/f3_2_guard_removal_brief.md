# Phase F3.2 Guard Removal — Execution Brief

**Initiative:** INTEGRATE-PYTORCH-001  
**Phase:** F3.2 — Remove guarded imports & flags  
**Context:** PyTorch is now a mandatory dependency (Phase F3.1 ✅). This loop retires the `TORCH_AVAILABLE` pattern across production modules so PyTorch backend code paths become first-class citizens without NumPy fallbacks.

---

## Objectives
- Eliminate `try/except ImportError` guards in PyTorch production modules.
- Remove `TORCH_AVAILABLE` flags and NumPy fallback code paths.
- Ensure code assumes PyTorch is installed, while retaining clear RuntimeError messaging where guard code previously masked missing functionality.
- Maintain API parity with TensorFlow counterparts; update docstrings/comments accordingly.

## Target Modules (per F2.1 inventory)
1. `ptycho_torch/config_params.py`
   - Drop optional import block; import `torch` unconditionally.
   - Remove `TORCH_AVAILABLE` export and fallback `TensorType = Any` alias.
   - Keep `TensorType = torch.Tensor` alias for backwards compatibility.

2. `ptycho_torch/config_bridge.py`
   - Stop importing `TORCH_AVAILABLE`.
   - Simplify `probe_mask` translation: treat non-`None` probe mask as enabled without torch check.
   - Remove any defensive branches depending on `TORCH_AVAILABLE`.

3. `ptycho_torch/data_container_bridge.py`
   - Inline torch import at module top; delete guard + NumPy fallback branch.
   - Keep dtype validation but enforce torch tensor conversions only.
   - Ensure `__repr__` no longer references `TORCH_AVAILABLE`—detect torch tensors via `hasattr(attr, 'dtype')`.

4. `ptycho_torch/memmap_bridge.py`
   - No torch usage beyond guard; remove optional import entirely and rely on downstream modules for torch usage.

5. `ptycho_torch/model_manager.py`
   - Remove guard around torch/nn imports.
   - Delete sentinel dictionary fallback logic; require `nn.Module` instances.
   - Replace RuntimeError messaging to instruct users to install torch if import somehow fails.

6. `ptycho_torch/workflows/components.py`
   - Remove TORCH guard scaffolding; import RawDataTorch, PtychoDataContainerTorch, save/load helpers directly.
   - Replace fallback `None` assignments with explicit RuntimeError raising during function execution when dependencies missing (should not occur given torch requirement, but keep defensive messaging for clarity).
   - Update module preamble to reflect mandatory torch assumption.

## Testing Expectations
- `pytest tests/torch/test_config_bridge.py -q`
- `pytest tests/torch/test_data_pipeline.py -q`
- `pytest tests/torch/test_model_manager.py -q`
- `pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_not_implemented -q`

(Per migration blueprint §6: run module-level tests after each major edit; bundle final verification under Phase F3.4.)

## Artifact Capture
- Document code touchpoints and test evidence in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T193753Z/guard_removal_summary.md` during implementation loop (Ralph).
- Update `phase_f_torch_mandatory.md` (F3.2 row) with per-module completion notes and relevant test logs.

## Notes & Pitfalls
- Ensure torch import failures raise actionable RuntimeError with install guidance (no silent fallback).
- Confirm any remaining type aliases or dataclass defaults reference `torch.Tensor` symbols correctly.
- Preserve comments referencing historical guard behavior only when summarizing migration rationale; remove live code relying on it.
- Be mindful of circular import risk when simplifying `workflows/components.py`; verify import order still succeeds.

---

**Ready for Implementation:** This brief, combined with the Phase F blueprint, should guide Ralph through guard removal while keeping parity and contract compliance in view.
