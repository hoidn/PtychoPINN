Summary: Prepare `_train_with_lightning` to satisfy the Lightning orchestration tests and unblock Phase D2.B
Mode: TDD
Focus: INTEGRATE-PYTORCH-001-STUBS — Finish PyTorch workflow stubs deferred from Phase D2
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T031500Z/phase_d2_completion/{summary.md,pytest_train_green.log}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS B2.1-B2.7 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Implement `_train_with_lightning` per phase_b2_implementation.md (tests: none)
2. INTEGRATE-PYTORCH-001-STUBS B2.8 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Run `pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T031500Z/phase_d2_completion/pytest_train_green.log` (tests: targeted)
3. INTEGRATE-PYTORCH-001-STUBS B2 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Update `summary.md`, plan checklist, and docs/fix_plan Attempts with new evidence (tests: none)

If Blocked: Capture partial implementation notes and pytest output in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T031500Z/phase_d2_completion/summary.md`, leave checklist B2 `[P]`, and document the blocker (selector, traceback, hypothesis) in docs/fix_plan.md Attempts.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:35 — Phase B exit criteria require `_train_with_lightning` to drive Lightning end-to-end.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T020940Z/phase_d2_completion/phase_b2_implementation.md:23 — Blueprint enumerates helper structure and artifact expectations.
- specs/ptychodus_api_spec.md:191 — Reconstructor lifecycle mandates trained module handles + serialized configs for persistence.
- docs/workflows/pytorch.md:69 — PyTorch workflow guide defines Lightning trainer behaviour and config knobs.
- tests/torch/test_workflows_components.py:713 — `TestTrainWithLightningRed` encodes acceptance criteria for module instantiation, Trainer.fit, and result payload.

How-To Map:
- Read the B2 blueprint, then add `_build_lightning_dataloaders(train_container, test_container, config)` inside `ptycho_torch/workflows/components.py`; keep it private and duck-type friendly so dict-based fixtures from the red tests work.
- Within `_train_with_lightning`, import `lightning.pytorch.Trainer` and `PtychoPINN_Lightning` lazily; on ImportError, raise `RuntimeError` citing POLICY-001 with install guidance.
- Seed deterministically via `lightning.pytorch.seed_everything(config.subsample_seed or 42)` before building loaders; construct loaders with `TensorDictDataLoader` (train/test) respecting `config.batch_size`, `config.sequential_sampling`, and allowing `None` for validation when `test_container` is absent.
- Instantiate `PtychoPINN_Lightning(config.model, config.data, config, config.inference)` (adjust for actual dataclasses), immediately call `save_hyperparameters()`, and preserve handles for the results payload.
- Configure `Trainer(max_epochs=config.nepochs, accelerator='auto', devices=1 if not config.device else config.device, log_every_n_steps=1, default_root_dir=config.output_dir, enable_progress_bar=config.debug)` and run `trainer.fit(model, train_loader, val_loader)` (val loader may be `None`).
- Collect losses from `trainer.callback_metrics`, build the results dict with `history`, original containers, and `models={'lightning_module': model, 'trainer': trainer}`, then write outcomes to `summary.md` alongside the green pytest log before refreshing plan + ledger entries.

Pitfalls To Avoid:
- Do not import Lightning or torch at module scope; keep `_train_with_lightning` torch-optional with actionable RuntimeError messaging.
- Avoid disk I/O beyond the configured output directory; no new logs at repo root (cleanup the existing `train_debug.log` only after confirming plan allows it).
- Preserve deterministic behaviour—respect `config.sequential_sampling`, `config.subsample_seed`, and avoid random shuffles without explicit seeds.
- Ensure dataloaders work with the red-test dict fixtures; guard attribute access with `getattr`/`dict.get` as needed.
- Return a `'models'` dict; omitting it keeps red tests failing and blocks Phase D persistence tasks.
- Keep new helper functions private to `ptycho_torch/workflows/components.py`; no cross-module refactors this loop.
- Capture pytest output with `tee`; missing logs will force us to rollback checklist status.
- Do not touch `_reassemble_cdi_image_torch`; stitching is Phase C.

Pointers:
- ptycho_torch/workflows/components.py:265
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:35
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T020940Z/phase_d2_completion/phase_b2_implementation.md:23
- specs/ptychodus_api_spec.md:191
- tests/torch/test_workflows_components.py:713

Next Up: B3 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — integrate determinism + MLflow toggles once Lightning orchestration is green.
