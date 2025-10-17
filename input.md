Summary: Author a minimal failing PyTorch↔spec config bridge test covering the 9 MVP fields.
Mode: TDD
Focus: INTEGRATE-PYTORCH-001 — Prepare for PyTorch Backend Integration with Ptychodus
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_config_bridge.py -k mvp -v (expected FAIL)
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T033500Z/{failing_test.md,pytest.log}
Do Now: INTEGRATE-PYTORCH-001 Attempt #6 — Phase B.B2 failing test; author `tests/torch/test_config_bridge.py::test_mvp_config_bridge_populates_params_cfg` that exercises the PyTorch configs via the expected adapter stubs, then run `pytest tests/torch/test_config_bridge.py -k mvp -v` to confirm it fails.
If Blocked: Capture decisions + partial attempts in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T033500Z/failing_test.md`, update scope_notes.md with the blocker, and log it under Attempt #6 in docs/fix_plan.md before stopping.
Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/implementation.md:44-49 — Phase B.B2 mandates a failing test before bridge implementation, including the new adapter API.
- specs/ptychodus_api_spec.md:61-149 — Defines how `update_legacy_dict` must populate `params.cfg`; the test locks in that contract.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T032218Z/scope_notes.md:38-170 — Source of the 9-field MVP checklist and open questions to document if assertions need clarification.
- docs/TESTING_GUIDE.md:153-161 — Reinforces the Red-Green-Refactor loop we must follow.
- docs/findings.md:9 — CONFIG-001 highlights why params.cfg population is critical; the test should currently fail on that gap.
How-To Map:
- Add `tests/torch/test_config_bridge.py` and import at module scope: `Path` (pathlib), `pytest`, `ptycho.params`, `update_legacy_dict`, `ModelConfig`, `TrainingConfig`, `InferenceConfig` (from `ptycho.config.config`), plus the PyTorch configs `DataConfig`, `ModelConfig as PTModelConfig`, `TrainingConfig as PTTrainingConfig`, `InferenceConfig as PTInferenceConfig` (from `ptycho_torch.config_params`). Keep adapter imports inside the test body to avoid module-level ImportError.
- Within `test_mvp_config_bridge_populates_params_cfg()`, instantiate PyTorch configs with MVP-aligned values: `pt_data = DataConfig(N=128, grid_size=(2, 2), nphotons=1e9, K=7)`, `pt_model = PTModelConfig(mode='Unsupervised')`, `pt_train = PTTrainingConfig(epochs=1)`, `pt_infer = PTInferenceConfig(batch_size=1)`.
- Inside the test, attempt to import the adapter module: `from ptycho_torch import config_bridge` (wrap in try/except so the failure happens under xfail). Expect helper functions `to_model_config`, `to_training_config`, `to_inference_config`; call them as follows:
  - `spec_model = config_bridge.to_model_config(pt_data, pt_model)`
  - `spec_train = config_bridge.to_training_config(spec_model, pt_data, pt_train, overrides=dict(train_data_file=Path('train.npz'), n_groups=512, neighbor_count=7, nphotons=1e9))`
  - `spec_infer = config_bridge.to_inference_config(spec_model, pt_data, pt_infer, overrides=dict(model_path=Path('model_dir'), test_data_file=Path('test.npz'), n_groups=512, neighbor_count=7))`
- Use `params_snapshot = dict(params.cfg)` and `try/finally` to restore global state after assertions. Call `update_legacy_dict(params.cfg, spec_train)` followed by `update_legacy_dict(params.cfg, spec_infer)`.
- Assert that `params.cfg` matches MVP expectations (values and types) for keys: `'N'`, `'gridsize'`, `'model_type'`, `'train_data_file_path'`, `'test_data_file_path'`, `'model_path'`, `'n_groups'`, `'neighbor_count'`, `'nphotons'`.
- Decorate the test with `@pytest.mark.xfail(strict=True, reason="PyTorch config bridge missing MVP translations")`; inside the `except ModuleNotFoundError` / `AttributeError` block call `pytest.xfail` with the caught error to register the current failure pathway.
- Run `pytest tests/torch/test_config_bridge.py -k mvp -v` from repo root and pipe output to `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T033500Z/pytest.log` (e.g., `pytest ... | tee plans/.../pytest.log`).
- Summarize the failure mode (ImportError or assertion diffs), command, and exit code in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T033500Z/failing_test.md`.
Pitfalls To Avoid:
- Don’t implement the adapter functions yet; this loop must remain red.
- Avoid mutating existing PyTorch or TensorFlow config definitions.
- Do not leave `params.cfg` dirty—restore the snapshot in the finally block.
- Skip broad pytest runs; execute only the targeted selector.
- Don’t bypass the adapter by instantiating TensorFlow configs directly—the test must exercise the PyTorch entry path.
- Keep artifact names and timestamp directory exactly as specified.
- Capture the full traceback in the report instead of paraphrasing it.
- Avoid GPU/device assumptions; keep the test CPU-only.
- Ensure the test lives under `tests/torch/` and follows naming conventions for discovery.
- Leave the xfail in place until implementation completes in a later loop.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/implementation.md:44-49
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T032218Z/config_schema_map.md:1-307
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T032218Z/scope_notes.md:38-170
- specs/ptychodus_api_spec.md:61-147
- docs/DEVELOPER_GUIDE.md:603-640
Next Up: Phase B.B3 — implement the MVP adapter + bridge once the failing test is documented.
