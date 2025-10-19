# Phase D1c Planning — Lightning Hyperparameter Serialization

## Context
- Initiative: INTEGRATE-PYTORCH-001-STUBS Phase D2 completion
- Recent evidence (`2025-10-19T123000Z`) confirmed Lightning checkpoints omit the `hyper_parameters` payload, preventing state-free reloads and blocking the integration test.
- Objective: Design the TDD loop that restores Lightning hyperparameter serialization and verifies `load_from_checkpoint()` works without manual kwargs.

## Planned Loop (D1c)
1. **Author Red Tests**
   - New module `tests/torch/test_lightning_checkpoint.py` with class `TestLightningCheckpointSerialization`.
   - Red case A: instantiate `PtychoPINN_Lightning` with canonical dataclass configs, call `trainer.save_checkpoint`, and assert `torch.load(path)['hyper_parameters'] is not None`.
   - Red case B: call `PtychoPINN_Lightning.load_from_checkpoint(path)` without kwargs and assert it returns a module whose configs match the originals.
   - Capture selectors + failure log at `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T134500Z/phase_d2_completion/pytest_checkpoint_red.log`.

2. **Implement Serialization Fix**
   - Add `self.save_hyperparameters()` immediately after `super().__init__()` in `ptycho_torch/model.py`.
   - Provide sanitized payload helper if Lightning rejects dataclasses containing `Path`/`Tensor` fields (e.g., convert to primitives before saving, rebuild dataclasses in `__init__` when `hparams` dict is provided).
   - Ensure inference loader (`ptycho_torch/inference.py`) uses the restored configs; add guardrails for backward-compat checkpoints lacking metadata.

3. **Green Validation**
   - Re-run targeted selector (`pytest tests/torch/test_lightning_checkpoint.py -vv`) and full integration workflow to prove `load_from_checkpoint` now succeeds. Store green logs as `pytest_checkpoint_green.log` and `pytest_integration_checkpoint_green.log` under the 2025-10-19T134500Z artifact hub.
   - Update `reports/2025-10-19T134500Z/phase_d2_completion/summary.md` with remediation notes and remaining risks (e.g., legacy checkpoints).

## Dependencies & References
- Contract: `specs/ptychodus_api_spec.md` §4.6 (state-free checkpoint reload requirement)
- Workflow guide: `docs/workflows/pytorch.md` §6 (Lightning checkpoint management)
- Evidence: `reports/2025-10-19T095900Z/phase_d2_completion/diagnostics.md`, `reports/2025-10-19T123000Z/phase_d2_completion/checkpoint_inspection.md`

## Exit Criteria for D1c
- New tests cover hyperparameter serialization and state-free reload, passing after implementation.
- Lightning checkpoint contains non-null `hyper_parameters` with four config objects.
- Integration workflow test passes the checkpoint load stage (subsequent stitching steps may still fail for unrelated reasons, but load must succeed).
- Plan checklist row `D1c` marked `[x]` with artifact references, docs/fix_plan Attempts updated accordingly.
