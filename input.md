Summary: Implement `_train_with_lightning` so the new Lightning orchestration tests pass
Mode: TDD
Focus: INTEGRATE-PYTORCH-001-STUBS — Finish PyTorch workflow stubs deferred from Phase D2
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T014317Z/phase_d2_completion/{summary.md,pytest_train_green.log}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS B2.1-B2.7 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Implement `_train_with_lightning` per phase_b2_implementation.md blueprint (tests: none)
2. INTEGRATE-PYTORCH-001-STUBS B2.8 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Run pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T014317Z/phase_d2_completion/pytest_train_green.log (tests: targeted)
3. INTEGRATE-PYTORCH-001-STUBS B2 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Update docs/fix_plan.md Attempt entry + plan checklist with new artifacts (tests: none)

If Blocked: Capture partial pytest output and implementation notes under the artifact directory (summary.md + failing log), leave B2 checklist `[P]`, and document the blocker + reproduction in docs/fix_plan.md Attempts.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md#L37 — Phase B checklist expects Lightning orchestration implementation next.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T020940Z/phase_d2_completion/phase_b2_implementation.md — Execution blueprint detailing tasks B2.1–B2.8.
- specs/ptychodus_api_spec.md §4.5 — Reconstructor lifecycle requires serialized configs and trained module handles.
- docs/workflows/pytorch.md §§5–7 — Lightning training expectations and MLflow hooks.
- tests/torch/test_workflows_components.py:713 — TestTrainWithLightningRed encodes acceptance criteria.

How-To Map:
- Read the B2 blueprint (`phase_b2_implementation.md`) and mirror the helper patterns used in `ptycho_torch/train.py` when wiring configs, dataloaders, and Trainer options.
- Keep imports torch-optional: import `PtychoPINN_Lightning` and `Trainer` inside `_train_with_lightning`, raise RuntimeError referencing POLICY-001 if torch is unavailable.
- Build train/val dataloaders from provided containers (reuse existing TensorDict adapters or add `_build_lightning_dataloaders` helper); propagate `batch_size`, `sequential_sampling`, and seeds from `TrainingConfig`.
- Instantiate the Lightning module with `(model_config, data_config, training_config, inference_config)` and call `save_hyperparameters()` so checkpoints capture configs.
- Configure `Trainer` with deterministic defaults (max_epochs, accelerator='auto', devices=1 unless overridden) and respect `config.output_dir` for checkpoints; call `trainer.fit` using constructed dataloaders.
- Return a results dict containing history, original containers, and a `'models'` entry with the Lightning module (and optionally trainer) to satisfy persistence expectations.
- Command: `pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T014317Z/phase_d2_completion/pytest_train_green.log`
- Summarize the implementation + test outcome in `summary.md` within the same artifact directory before updating docs/fix_plan.md.

Pitfalls To Avoid:
- Do not move `_train_with_lightning` out of `ptycho_torch.workflows.components`; keep helper functions private to the module.
- Avoid hardcoding GPU devices; rely on Lightning defaults unless `config.device` specifies otherwise.
- Keep torch imports guarded to preserve meaningful RuntimeError messaging when torch is missing.
- Ensure dataloaders respect existing container shapes; avoid loading datasets from disk inside the helper.
- Do not alter the newly added red tests beyond minimal fixture tweaks needed for passing.
- Preserve deterministic seeds and `config.sequential_sampling`; no random shuffling without honoring config flags.
- Keep logs and summaries confined to the timestamped artifact directory; no outputs at repo root.
- Update docs/fix_plan.md Attempts with the exact artifact paths used.
- Leave subsequent B3/B4 tasks untouched unless explicitly instructed.
- Run only the targeted pytest selector; defer broader suites until later phases.

Pointers:
- ptycho_torch/workflows/components.py:250
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T020940Z/phase_d2_completion/phase_b2_implementation.md
- specs/ptychodus_api_spec.md:187
- tests/torch/test_workflows_components.py:713

Next Up: B3 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md (surface determinism + MLflow controls) once B2 turns the Lightning tests green.
