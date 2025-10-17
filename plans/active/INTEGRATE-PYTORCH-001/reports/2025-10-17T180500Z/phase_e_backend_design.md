# Phase E1.C Backend Selection Blueprint

## Context
- Initiative: INTEGRATE-PYTORCH-001 — Phase E1 (Backend Selection & Orchestration Bridge).
- Goal: Define the implementation blueprint that allows Ptychodus to select the PyTorch backend while retaining TensorFlow parity, satisfying `specs/ptychodus_api_spec.md` §4.1–§4.6.
- Inputs reviewed: `phase_e_callchain/summary.md` (callchain diff), `tests/torch/test_backend_selection.py` (red tests), `ptycho/config/config.py` (dataclasses), `ptycho/workflows/components.py:700-760` (TensorFlow entry), `ptycho_torch/workflows/components.py:150-240` (PyTorch entry), `ptycho_torch/config_bridge.py` (Phase B adapters), `plans/ptychodus_pytorch_integration_plan.md` (canonical deliverables).
- Dependencies: CONFIG-001 finding (legacy params.cfg gate), Phase D handoff (`reports/2025-10-17T121930Z/phase_d4c_summary.md`), TEST-PYTORCH-001 fixture coordination (Phase D4 selector map).

## Architectural Summary
1. **Configuration Flag** — Add a `backend: Literal['tensorflow', 'pytorch']` field to both `TrainingConfig` and `InferenceConfig` with default `'tensorflow'` to preserve backward compatibility. The field must be torch-agnostic and survive YAML loading.
2. **PyTorch Bridge Awareness** — Ensure `ptycho_torch.config_bridge.to_training_config()` / `.to_inference_config()` set `backend='pytorch'` on returned dataclasses so CONFIG-001 updates propagate the selected backend when the PyTorch stack spins up without involving Ptychodus.
3. **Selection Layer** — Introduce a dispatcher in the Ptychodus reconstructor surface (`ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py` per spec §4.1) that inspects `config.backend` and imports the correct workflow module (`ptycho.workflows.components` vs `ptycho_torch.workflows.components`). Fall back to TensorFlow when the field is missing or `None`.
4. **Fail-Fast Guard** — When `backend='pytorch'` but `ptycho_torch` (or `torch`) is unavailable, raise a `RuntimeError` with actionable installation guidance before attempting any workflow invocation.
5. **Parity Assurance** — Maintain identical function signatures and return types, log the active backend once per invocation, and document the chosen backend in results metadata (`results['backend']`) to aid regression analysis.

## Phase E1.C Implementation Tasks

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| E1.C1 | Extend dataclass schema with `backend` field | [ ] | Modify `ptycho/config/config.py` (`TrainingConfig`, `InferenceConfig`) to accept `backend: Literal['tensorflow','pytorch']='tensorflow'`. Update docstrings, validation helpers (no-op), and ensure `dataclass_to_legacy_dict` passes the value through. Verify YAML loader + `update_legacy_dict` keep defaults intact (`specs/ptychodus_api_spec.md:213-273`). |
| E1.C2 | Propagate backend in PyTorch adapters | [ ] | Update `ptycho_torch/config_bridge.{to_training_config,to_inference_config}` to set `backend='pytorch'` on emitted dataclasses unless overridden. Maintain torch-optional imports; adjust tests (`tests/torch/test_config_bridge.py`) to assert backend field translation once green. Reference `reports/2025-10-17T032218Z/config_schema_map.md` for field coverage. |
| E1.C3 | Implement reconstructor dispatcher | [ ] | In `ptychodus` reconstructor library (`model/ptychopinn/reconstructor.py` + helpers), add a backend selection shim that calls TensorFlow workflows by default and PyTorch workflows when `backend='pytorch'`. Ensure CONFIG-001 gate (`update_legacy_dict`) executes before dispatch, and patch persistence hooks to leverage Phase D3 save/load functions when PyTorch selected. Capture import errors and raise descriptive `RuntimeError`. |
| E1.C4 | Instrument selection + align tests | [ ] | Update logging (`logger.info`) and results dict to indicate active backend (aids D4/E2 parity). Make `tests/torch/test_backend_selection.py` green: remove `xfail`, assert dispatcher path via monkeypatch, cover fallback error. Run targeted selectors: `pytest tests/torch/test_backend_selection.py -vv`, `pytest tests/torch/test_workflows_components.py -k backend -vv`. Document outputs under `plans/active/INTEGRATE-PYTORCH-001/reports/<ts>/phase_e_backend_green.log`. |

## Decision Details

### Configuration Plumbing
- **Default Behavior:** Keep `'tensorflow'` default to honour existing callers and pass `test_defaults_to_tensorflow_backend`.
- **Legacy Bridge:** Allow `params.cfg['backend']` to reflect the selection (no KEY_MAPPINGS change needed). Downstream legacy modules ignore the key, but making it visible aids debugging.
- **Validation:** No additional validation beyond literal type required; selection is enforced at dispatcher layer.

### Dispatcher Surface
- **Entry Points:** Wrap backend selection inside `run_cdi_example` / `load_inference_bundle` wrappers used by Ptychodus `PtychoPINNReconstructorLibrary`. Avoid touching stable physics modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`) per CLAUDE directive.
- **Import Strategy:** Use local imports to keep torch optional:
  ```python
  if config.backend == 'pytorch':
      try:
          from ptycho_torch.workflows import components as torch_components
      except ImportError as exc:
          raise RuntimeError("PyTorch backend selected but unable to import ptycho_torch; install PyTorch extras.") from exc
      workflow = torch_components.run_cdi_example_torch
  else:
      from ptycho.workflows import components as tf_components
      workflow = tf_components.run_cdi_example
  ```
- **Shared Signature:** Delegate with identical kwargs; propagate any kwargs (flip_x, etc.) transparently.

### Results & Logging
- Log `logger.info("Using %s backend", config.backend)` at dispatcher entry.
- Add `results["backend"] = config.backend` before returning to aid parity summaries (Phase E2.D).
- Capture backend choice in persistence metadata when PyTorch bundler invoked (Phase D3 already stores version tag `2.0-pytorch`).

### Error Handling
- Raise `RuntimeError` when PyTorch backend requested but `ptycho_torch` import fails; message must include installation guidance (`pip install .[torch]`) per test expectations.
- Retain existing exception paths for TensorFlow (no change).

## Testing & Validation Plan
- **Unit/Parity Tests:** Turn `tests/torch/test_backend_selection.py` green (remove `xfail`). Update mocks to verify dispatcher invoked correct module. Add targeted regression to `tests/test_pinn_reconstructor.py` (TensorFlow) if necessary to confirm defaults unaffected.
- **Integration Tests:** After wiring, extend Phase E2 tests to exercise backend selection end-to-end (subprocess). Blueprint sets stage; implementation loop should coordinate with TEST-PYTORCH-001 for fixture reuse (`phase_d4_selector_map.md`).
- **Manual Verification:** Run `pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_persists_models -vv` under both backends to ensure persistence unaffected.

## Risks & Mitigations
- **Torch Optionality:** Dispatcher imports must remain guarded to avoid ImportError during docs-only loops. Use inner imports + try/except.
- **Config Drift:** Ensure YAML configs that lack `backend` still parse (default). Add release note to `phase_e_docs_update.md` (Phase E3) describing new field.
- **Test Fragility:** Keep backend logs deterministic to avoid brittle assertions; rely on monkeypatch/spy in tests rather than string matching when possible.

## Exit Criteria for E1.C
- Blueprint documented (this file) and linked from plan + ledger.
- Implementation table above marked `[ ]` until Ralph delivers code; this document defines acceptance for their loop.
- Next supervisor loop should instruct Ralph to execute E1.C1–E1.C4 (can be bundled if feasible) following TDD (tests already failing per Phase E1.B).

