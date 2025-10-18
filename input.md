Summary: Wire `_train_with_lightning` to Lightning so TestTrainWithLightningRed turns green
Mode: TDD
Focus: INTEGRATE-PYTORCH-001-STUBS — Finish PyTorch workflow stubs deferred from Phase D2
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T014317Z/phase_d2_completion/{summary.md,pytest_train_green.log}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS B2.1-B2.7 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Implement `_train_with_lightning` per phase_b2_implementation.md (tests: none)
2. INTEGRATE-PYTORCH-001-STUBS B2.8 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Run pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T014317Z/phase_d2_completion/pytest_train_green.log (tests: targeted)
3. INTEGRATE-PYTORCH-001-STUBS B2 checklist @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Record summary.md, update checklist states, and add docs/fix_plan Attempts entry with artifact paths (tests: none)

If Blocked: Capture partial implementation notes + pytest output in plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T014317Z/phase_d2_completion/summary.md, leave checklist B2 tasks `[P]`, and log the blocker (selector, traceback, hypothesis) in docs/fix_plan.md Attempts before exiting.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md#L30 — Phase B exit criteria require `_train_with_lightning` to delegate Lightning execution with deterministic results.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T020940Z/phase_d2_completion/phase_b2_implementation.md — B2 blueprint enumerates tasks B2.1–B2.8 and evidence expectations.
- specs/ptychodus_api_spec.md:187 — Reconstructor lifecycle demands Lightning training yields persistence-ready handles and serialized configs.
- docs/workflows/pytorch.md:69 — PyTorch workflow guide calls out TensorDict loaders + Lightning Trainer integration.
- tests/torch/test_workflows_components.py:713 — `TestTrainWithLightningRed` codifies acceptance criteria for module instantiation, Trainer.fit, and result payload.

How-To Map:
- Instantiate PyTorch config objects (DataConfig, ModelConfig, TrainingConfig, InferenceConfig from `ptycho_torch.config_params`) using values from the incoming TensorFlow dataclasses; lean on the field mapping at plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T032218Z/config_schema_map.md to keep parity.
- Call `lightning.pytorch.seed_everything(config.subsample_seed or 0)` before building dataloaders so shuffling respects deterministic seeds; mirror sequential sampling flags when `config.sequential_sampling` is True.
- Build train/val loaders from the Phase C containers with `TensorDictDataLoader` (`ptycho_torch.dset_loader_pt_mmap`) and any required collate helper; prefer a private `_build_lightning_dataloaders` that accepts the container and batch configuration.
- Import `PtychoPINN_Lightning` and `lightning.pytorch.Trainer` inside `_train_with_lightning`; on ImportError, raise `RuntimeError` citing POLICY-001 (torch>=2.2 requirement).
- Instantiate the Lightning module with `(model_cfg, data_cfg, train_cfg, infer_cfg)` and call `save_hyperparameters()` so checkpoints embed the configuration state required by Phase D persistence.
- Configure `Trainer` with `max_epochs=config.nepochs`, `accelerator='auto'`, `devices=1` unless `config.device` specifies otherwise, `log_every_n_steps=1`, and `default_root_dir=str(config.output_dir)`. Respect `config.debug` by toggling deterministic/progress settings.
- Execute `trainer.fit(lightning_module, train_loader, val_loader)` and harvest losses from `trainer.callback_metrics`; assemble the results dict with `history`, original containers, and `models = {'lightning_module': lightning_module, 'trainer': trainer}` so downstream save/load steps have handles.
- Command: `pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T014317Z/phase_d2_completion/pytest_train_green.log`
- After the run, replace README.md in the artifact directory with `summary.md` capturing implementation notes + selector outcome, then update docs/fix_plan.md Attempts and the plan checklist IDs touched.

Pitfalls To Avoid:
- Do not import Lightning or torch at module scope; keep `_train_with_lightning` torch-optional with actionable RuntimeError messaging.
- Avoid reloading datasets from disk; operate on provided containers only.
- Respect deterministic flags (`config.sequential_sampling`, `config.subsample_seed`); no hard-coded RNG seeds.
- Ensure Trainer checkpoints land under `config.output_dir`; no stray outputs elsewhere in the repo.
- Keep helper functions private to `ptycho_torch.workflows.components`; defer refactors or module moves.
- Return a `'models'` mapping; skipping it will leave persistence paths broken and tests red.
- Capture pytest output with `tee`; missing logs will force plan rollback.
- Leave `_reassemble_cdi_image_torch` untouched (Phase C focus) and avoid incidental edits outside `_train_with_lightning`/helper scope.
- Update params.cfg only through existing bridges; do not mutate globals manually inside the helper.
- Run only the targeted selector unless debugging demands additional evidence.

Pointers:
- ptycho_torch/workflows/components.py:265
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md#L30
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T020940Z/phase_d2_completion/phase_b2_implementation.md
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T032218Z/config_schema_map.md
- tests/torch/test_workflows_components.py:713
- specs/ptychodus_api_spec.md:187
- docs/workflows/pytorch.md:69

Next Up: B3 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md (surface determinism + MLflow controls) once B2 artifacts are green.
