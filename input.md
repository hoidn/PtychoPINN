Summary: Implement `_train_with_lightning` so Lightning orchestration tests pass
Mode: TDD
Focus: INTEGRATE-PYTORCH-001-STUBS — Finish PyTorch workflow stubs deferred from Phase D2
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T031500Z/phase_d2_completion/{summary.md,pytest_train_green.log}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS B2.1-B2.7 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Implement `_train_with_lightning` per phase_b2_implementation.md (tests: none)
2. INTEGRATE-PYTORCH-001-STUBS B2.8 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Run pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T031500Z/phase_d2_completion/pytest_train_green.log (tests: targeted)
3. INTEGRATE-PYTORCH-001-STUBS B2 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Update summary.md + docs/fix_plan Attempts with new artifacts, mark checklist B2 accordingly (tests: none)

If Blocked: Capture partial implementation notes + pytest output in plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T031500Z/phase_d2_completion/summary.md, leave checklist B2 `[P]`, and log blocker details (selector, traceback, hypothesis) in docs/fix_plan.md Attempts.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md#L30 — Phase B exit criteria require `_train_with_lightning` to delegate Lightning execution with deterministic results.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T020940Z/phase_d2_completion/phase_b2_implementation.md#L1 — B2 blueprint enumerates tasks B2.1–B2.8 and artifact expectations.
- specs/ptychodus_api_spec.md:191 — Training workflow contract demands API parity with TensorFlow orchestration and persistence hooks.
- docs/workflows/pytorch.md:52 — PyTorch workflow guide outlines Lightning + MLflow expectations for training.
- tests/torch/test_workflows_components.py:713 — `TestTrainWithLightningRed` encodes acceptance criteria for module instantiation, Trainer.fit, and result payload.

How-To Map:
- Use `_ensure_container` outputs directly; add a helper (e.g., `_build_lightning_dataloaders`) that wraps train/test containers into `TensorDictDataLoader` objects respecting `config.batch_size`, `config.sequential_sampling`, and deterministic seeds (`lightning.pytorch.seed_everything(config.subsample_seed or 42)`).
- Import `PtychoPINN_Lightning` and `lightning.pytorch.Trainer` inside `_train_with_lightning`; if import fails, raise `RuntimeError` echoing POLICY-001 guidance.
- Instantiate Lightning module with `(config.model, config.data, config.training, config.inference)` equivalents, then call `save_hyperparameters()` so checkpoints capture configs.
- Configure `Trainer` with `max_epochs=config.nepochs`, `accelerator='auto'`, `devices=1 if config.device is None else config.device`, `log_every_n_steps=1`, and `default_root_dir=config.output_dir`. Respect `config.debug` by enabling progress bars and deterministic flags.
- Collect training metrics via `trainer.callback_metrics`; build results dict containing the original containers, `history = {'train_loss': ..., 'val_loss': ...}`, and `models = {'lightning_module': module, 'trainer': trainer}` so downstream persistence can reuse handles.
- Command: `pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T031500Z/phase_d2_completion/pytest_train_green.log`
- After the run, append a short outcome paragraph to `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T031500Z/phase_d2_completion/summary.md` and update docs/fix_plan.md Attempts with the timestamped artifacts.

Pitfalls To Avoid:
- Do not import Lightning or torch at module scope; keep `_train_with_lightning` torch-optional.
- Avoid loading datasets from disk; operate on provided containers only.
- Preserve deterministic seeds (`config.sequential_sampling`, `config.subsample_seed`) when constructing dataloaders.
- Ensure Trainer writes checkpoints into `config.output_dir` and does not leak files elsewhere.
- Keep new helpers private to `ptycho_torch.workflows.components`; no cross-module refactors this loop.
- Return a `'models'` dict — omitting it will keep tests red and break persistence follow-up phases.
- Capture pytest output with `tee`; missing logs will revert plan checklist status.
- Do not run the full test suite; stick to the targeted selector unless debugging requires otherwise.
- Leave `_reassemble_cdi_image_torch` untouched; stitching work happens in Phase C.
- Ensure RuntimeErrors for missing torch reference POLICY-001 verbatim to remain findings-compliant.

Pointers:
- ptycho_torch/workflows/components.py:265
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md#L30
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T020940Z/phase_d2_completion/phase_b2_implementation.md
- tests/torch/test_workflows_components.py:713
- specs/ptychodus_api_spec.md:191
- docs/workflows/pytorch.md:52

Next Up: B3 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md (surface determinism + MLflow controls) once B2 is green.
