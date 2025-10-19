Summary: Restore Lightning checkpoint hyperparameters via TDD so load_from_checkpoint works without kwargs
Mode: TDD
Focus: INTEGRATE-PYTORCH-001-STUBS — Phase D1c Lightning hyperparameter serialization
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_lightning_checkpoint.py -vv; pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T134500Z/phase_d2_completion/{pytest_checkpoint_red.log,pytest_checkpoint_green.log,pytest_integration_checkpoint_green.log,summary.md}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS D1c @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Author `tests/torch/test_lightning_checkpoint.py::TestLightningCheckpointSerialization` with red cases for missing `hyper_parameters` payload and failing `load_from_checkpoint`, then run `pytest tests/torch/test_lightning_checkpoint.py -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T134500Z/phase_d2_completion/pytest_checkpoint_red.log` (tests: targeted selector)
2. INTEGRATE-PYTORCH-001-STUBS D1c @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Implement `PtychoPINN_Lightning` serialization fix (call `self.save_hyperparameters()` with serialisable payload + adjust inference loader if needed) and rerun `pytest tests/torch/test_lightning_checkpoint.py -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T134500Z/phase_d2_completion/pytest_checkpoint_green.log` (tests: targeted selector)
3. INTEGRATE-PYTORCH-001-STUBS D1c @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Re-run integration workflow `pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T134500Z/phase_d2_completion/pytest_integration_checkpoint_green.log` to confirm checkpoint load succeeds (tests: targeted selector)
4. INTEGRATE-PYTORCH-001-STUBS D1c @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Summarize results in `summary.md`, mark plan row `[x]`, update `docs/fix_plan.md` Attempts (include log paths), and note lingering risks or follow-up items (tests: none)

If Blocked: Capture stdout/stderr from the failing command into the artifact directory, leave D1c `[P]`, and document the blocker with reproduction details in `summary.md` plus docs/fix_plan.md before pausing.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:73 — D1c is the next open checklist gate after confirming missing `hyper_parameters`.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T123000Z/phase_d2_completion/checkpoint_inspection.md:64 — Evidence shows Lightning checkpoint omits hyperparameters; remediation must follow Hypothesis 1.
- specs/ptychodus_api_spec.md:205 — Reconstructor lifecycle demands checkpoints reload without supplemental kwargs.
- docs/workflows/pytorch.md:260 — Troubleshooting section calls out the current TypeError; this loop resolves it.
- docs/TESTING_GUIDE.md:15 — TDD mandate: add failing tests before modifying production code.

How-To Map:
- Test scaffolding: import canonical configs via `from ptycho.config.config import TrainingConfig, ModelConfig, DataConfig, InferenceConfig, update_legacy_dict` and bridge them with `ptycho_torch.config_bridge.bridge_training_config`. Use small deterministic values (N=64, gridsize=1) and CPU device. Persist checkpoints with `Trainer(max_epochs=0, enable_checkpointing=True)` pointed at `tmp_path / "ckpt.ckpt"`.
- Red tests: A) After `trainer.save_checkpoint`, load with `torch.load` and assert `checkpoint['hyper_parameters'] is None` (expected failure). B) Call `PtychoPINN_Lightning.load_from_checkpoint(ckpt_path)` without kwargs and assert it raises the current `TypeError`. Document expected failure messages in test docstrings so the red log is easy to interpret.
- Implementation: Add `from dataclasses import asdict` if needed. Place `self.save_hyperparameters(model_config=model_config, data_config=data_config, training_config=training_config, inference_config=inference_config)` immediately after `super().__init__()`. If Lightning rejects dataclass instances, convert to serialisable dicts (`asdict`) and rebuild dataclasses in `__init__` when kwargs arrive via checkpoint load (`if isinstance(model_config, dict): model_config = ModelConfig(**model_config)`). Ensure inference loader handles restored configs without duplicating params.cfg updates.
- Green validation: After fix, assert `checkpoint['hyper_parameters']` contains four entries and `load_from_checkpoint` returns a module; add assertions in the test. Then run the integration selector to verify the inference CLI succeeds. Remove temporary checkpoints after tests to avoid large git artifacts.
- Reporting: Update `summary.md` with red/green selectors, key code references (line numbers for save_hyperparameters call), and note any follow-up (e.g., legacy checkpoints missing metadata). Record docs/fix_plan Attempt #34 with artifact paths.

Pitfalls To Avoid:
- Do not commit `.ckpt` binaries; delete them after inspection.
- Keep Lightning imports torch-optional (no top-level torch usage outside guarded code).
- Maintain CONFIG-001 discipline: call `update_legacy_dict` before loading data in tests/fixtures.
- Ensure tests use pytest style (no unittest subclasses) and remain deterministic (set `Trainer(deterministic=True)`).
- Avoid modifying TensorFlow core modules; limit changes to PyTorch backend and inference loader.
- Don’t bypass TDD—capture failing log before implementing fix.
- Guard against mutating dataclass instances in place when converting to dicts; use copies to avoid shared-state bugs.
- Keep artifact logs under the specified timestamp directory; no stray files at repo root.
- Verify the integration log shows the checkpoint load path now succeeds (look for absence of the TypeError).

Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:66
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T123000Z/phase_d2_completion/checkpoint_inspection.md:68
- specs/ptychodus_api_spec.md:205
- docs/workflows/pytorch.md:260
- ptycho_torch/model.py:923
- tests/torch/test_workflows_components.py:713

Next Up:
- D2 parity summary & docs refresh once integration test is green.
