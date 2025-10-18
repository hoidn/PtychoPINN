Summary: Author red Lightning orchestration tests for `_train_with_lightning`
Mode: TDD
Focus: INTEGRATE-PYTORCH-001-STUBS — Finish PyTorch workflow stubs deferred from Phase D2
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T000606Z/phase_d2_completion/{phase_b_test_design.md,summary.md,pytest_train_red.log}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS B1 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Add `TestTrainWithLightningRed` tests per phase_b_test_design.md (tests: none)
2. INTEGRATE-PYTORCH-001-STUBS B1 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Run pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T000606Z/phase_d2_completion/pytest_train_red.log (tests: targeted)
3. INTEGRATE-PYTORCH-001-STUBS B1 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Record Attempt entry in docs/fix_plan.md referencing new tests + log (tests: none)

If Blocked: Capture the partial pytest output under the artifact directory, summarize the failure in a short README alongside status in docs/fix_plan.md, and halt before modifying production code.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:40 — Phase B checklist defines Lightning TDD contract.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T000606Z/phase_d2_completion/phase_b_test_design.md — Detailed test expectations for `_train_with_lightning`.
- ptycho_torch/workflows/components.py:265 — Current stub lacking Lightning orchestration.
- specs/ptychodus_api_spec.md:187 — Reconstructor lifecycle requires trained module handles for persistence.
- docs/TESTING_GUIDE.md:1 — TDD discipline and targeted pytest usage.

How-To Map:
- Edit `tests/torch/test_workflows_components.py`: add class `TestTrainWithLightningRed` with three tests (`test_train_with_lightning_instantiates_module`, `test_train_with_lightning_runs_trainer_fit`, `test_train_with_lightning_returns_models_dict`). Use monkeypatch to spy on `ptycho_torch.model.PtychoPINN_Lightning` and `lightning.pytorch.Trainer` while keeping execution torch-optional.
- Use existing fixtures (`minimal_training_config`, `dummy_raw_data`) for config + containers; create lightweight sentinel containers if additional shape control is needed.
- Expected failure signature: AssertionErrors complaining that Lightning module/trainer were not invoked and missing `'models'` key in results. Do not “fix” the stub this loop.
- Command: `pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T000606Z/phase_d2_completion/pytest_train_red.log`
- After run, update docs/fix_plan.md Attempts and mention the log path; leave plan checklist B1 marked `[P]` until implementation turns tests green.

Pitfalls To Avoid:
- Do not modify `_train_with_lightning` or other production code yet; this loop is red tests only.
- Keep tests importable without a live torch runtime; rely on monkeypatch instead of instantiating tensors.
- Ensure the pytest selector matches exactly; avoid running full `tests/torch` suite.
- Capture full pytest output via `tee`; no truncated logs or clipboard artifacts.
- Reference findings by ID (POLICY-001, CONFIG-001) if cited in comments—do not restate prose.
- Maintain existing helper fixtures; do not duplicate config creation logic.
- Leave plan checklist items in `[P]` state; completion requires green implementation loop.
- Skip committing generated logs outside the timestamped reports directory.
- Observe two-message loop discipline: one engineer execution per supervisor brief.
- No MLflow or Lightning environment reconfiguration in tests; keep mocks local.

Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:40
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T000606Z/phase_d2_completion/phase_b_test_design.md
- ptycho_torch/workflows/components.py:265
- tests/torch/test_workflows_components.py:1
- specs/ptychodus_api_spec.md:187

Next Up: Phase B.B2 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md (implement Lightning orchestration once red tests land)
