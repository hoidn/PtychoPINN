Summary: Scaffold torch workflows module with update_legacy_dict parity guard
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 / Phase D2.A — Scaffold orchestration module
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_workflows_components.py::test_run_cdi_example_calls_update_legacy_dict -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T091450Z/{phase_d2_scaffold.md,pytest_scaffold.log}
Do Now:
- INTEGRATE-PYTORCH-001 (D2.A) — Author failing parity test ensuring `run_cdi_example_torch` invokes `update_legacy_dict` @ plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md (tests: pytest tests/torch/test_workflows_components.py::test_run_cdi_example_calls_update_legacy_dict -vv)
- INTEGRATE-PYTORCH-001 (D2.A) — Implement torch-optional `ptycho_torch/workflows/components.py` scaffold + export update @ plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md (tests: pytest tests/torch/test_workflows_components.py::test_run_cdi_example_calls_update_legacy_dict -vv)
If Blocked: Capture the failure (stack trace, environment notes) in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T091450Z/blocker.md`, log Attempt in docs/fix_plan.md, and halt implementation pending supervisor review.
Priorities & Rationale:
- specs/ptychodus_api_spec.md:187 mandates `run_cdi_example` lifecycle parity, so the torch entry point must align before adding behaviour.
- plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md:24-33 keeps Phase D2 scoped; completing D2.A unblocks training/inference adapters.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T085431Z/phase_d_decision.md:1 commits us to Option B shims, making scaffold design the next dependency.
- docs/workflows/pytorch.md:9 highlights parity expectations and torch-optional constraints we must respect in new modules.
How-To Map:
1. Create `tests/torch/test_workflows_components.py` defining `test_run_cdi_example_calls_update_legacy_dict`; use `monkeypatch` to spy on `ptycho.config.config.update_legacy_dict`, construct minimal dummy config via `TrainingConfig(...)`, and xfail-skip guard when torch unavailable (`pytest.importorskip("torch", reason="torch backend unavailable")`). Document rationale and inputs in `phase_d2_scaffold.md`.
2. Run `pytest tests/torch/test_workflows_components.py::test_run_cdi_example_calls_update_legacy_dict -vv` to confirm red state; save console output to `pytest_scaffold.log` within the artifact directory.
3. Add new module `ptycho_torch/workflows/components.py` exposing `run_cdi_example_torch`, `train_cdi_model_torch`, and `load_inference_bundle_torch` stubs. Ensure module imports guard torch availability, call `update_legacy_dict(params.cfg, config)` before delegating, and raise `NotImplementedError` for paths that Phase D2.B/C will fill. Update `ptycho_torch/__init__.py` to export the new helpers.
4. Re-run the targeted pytest command; expect pass via captured flag. Update `phase_d2_scaffold.md` with design notes (entry signatures, placeholder decisions, follow-on TODOs) and reference Option B doc sections.
5. Record outcomes in docs/fix_plan.md Attempt log and keep artifacts under the timestamped directory.
Pitfalls To Avoid:
- Do not import `torch` at module top level; rely on guarded availability checks to preserve optional backend (ANTIPATTERN-001).
- Avoid invoking heavy data loaders inside tests; keep fixtures minimal to preserve speed and torch-optional behaviour.
- Ensure `update_legacy_dict` receives the dataclass config object, not a dict, to stay aligned with CONFIG-001.
- Leave training/inference implementations raising `NotImplementedError` (or delegating to stubs) so later phases can replace them cleanly.
- Do not mutate global state besides the sanctioned `params.cfg` update; no singletons or MLflow calls yet.
- Keep new tests pure pytest style (no unittest mixins) and honour existing skip markers for missing torch.
- Document each decision in the artifact markdown so Phase D2.B/C have traceability.
Pointers:
- specs/ptychodus_api_spec.md:187
- plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md:24-33
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T085431Z/phase_d_decision.md:1
- ptycho/workflows/components.py:676
- docs/workflows/pytorch.md:9
Next Up: D2.B — implement training path adapter and add targeted Lightning orchestration tests once scaffold is stable.
