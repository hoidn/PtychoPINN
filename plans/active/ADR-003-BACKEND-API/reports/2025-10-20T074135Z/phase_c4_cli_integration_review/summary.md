# ADR-003 Phase C4.D3 Supervisor Review (2025-10-20T074135Z)

## Context
- **Do Now reference:** input.md (2025-10-20T070610Z) — Poisson support fix + integration validation.
- **Focus:** Confirm RED/GREEN coverage for `test_lightning_poisson_count_contract`, ensure plan ledger reflects completion, and capture remaining gaps blocking C4.D gate.

## Findings
- ✅ `tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_poisson_count_contract` RED/GREEN sequence completed; logs captured at `reports/2025-10-20T070610Z/phase_c4_cli_integration_debug/{pytest_poisson_red.log,pytest_poisson_green.log}`.
- ✅ `ptycho_torch/model.py` now squares predicted amplitudes and observed amplitudes before Poisson log-prob (parity with `ptycho/model.py:506-511`), resolving the support violation highlighted in `poisson_failure_summary.md`.
- ⚠️ Integration selector rerun exists only inside `pytest_full_suite.log`; expected `pytest_integration.log` artifact absent. Failure remains the known `load_torch_bundle` `NotImplementedError`.
- ⚠️ `train_debug.log` reintroduced at repo root; violates C4.F4 hygiene rule.
- ⚠️ Plan row `C4.D3` still describes pre-fix Poisson blocker; needs refresh to reflect current status + remaining dependency on Phase D3 bundle loader.

## Next Actions for Engineer
1. Capture targeted integration rerun at `plans/active/ADR-003-BACKEND-API/reports/<next-ts>/phase_c4_cli_integration_debug/pytest_integration.log` (Command: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv`).
2. Update plan row `C4.D3` to show Poisson parity ✅ and document that the remaining failure is the expected `load_torch_bundle` gap (Phase D3 dependency). Adjust verification checklist note accordingly.
3. Relocate or delete `train_debug.log` (keep copy under timestamped hub if needed) to satisfy C4.F4 hygiene.
4. Proceed to manual CLI smoke test (C4.D4) once targeted log captured.

## References
- Do Now logs: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T070610Z/phase_c4_cli_integration_debug/test_suite_summary.md`
- Poisson fix commit: `e10395e7`
- Outstanding bundle loader gap: `ptycho_torch/model_manager.py:267`
