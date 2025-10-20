Summary: Implement the Phase D training CLI thin wrapper and turn RED tests GREEN using the new helper module.
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase D.B (Training CLI thin wrapper, B3 implementation)
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_shared.py -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py -k train_cli -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T111500Z/phase_d_cli_wrappers_training_impl/{summary.md,pytest_cli_shared_green.log,pytest_cli_train_green.log,pytest_workflows_train_cli_green.log}

Do Now:
1. ADR-003-BACKEND-API B3.a @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:32 — Create `ptycho_torch/cli/__init__.py` and `ptycho_torch/cli/shared.py` implementing `resolve_accelerator`, `build_execution_config_from_args`, and `validate_paths` per `training_refactor.md`; tests: none
2. ADR-003-BACKEND-API B3.b+B3.c @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:33+34 — Extend `PyTorchExecutionConfig.__post_init__()` with new validation rules and refactor `ptycho_torch/train.py` CLI to use helpers (`--quiet` alias, deprecated `--device` messaging, helper calls); tests: none
3. ADR-003-BACKEND-API B3.d @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:35 — Run and capture GREEN selectors (tee to artifact hub):
   - `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_shared.py -vv`
   - `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -vv`
   - `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py -k train_cli -vv`
4. ADR-003-BACKEND-API B3.e @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:36 — Update plan checklist states, refresh `summary.md`, append Attempt entry in docs/fix_plan.md, and write loop summary (`summary.md`) referencing GREEN logs and code deltas; tests: none

If Blocked: Capture the blocker in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T111500Z/phase_d_cli_wrappers_training_impl/blocker.md`, revert any premature checklist changes, and note the issue inside docs/fix_plan.md before pausing.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:32-36 — B3 checklist defines helper module scope, validation updates, CLI refactor, and test evidence required for Phase D.B completion.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T105408Z/phase_d_cli_wrappers_training/training_refactor.md:120-356 — Blueprint captures exact helper semantics, warning behaviour, and RawData ownership choices driving the implementation.
- tests/torch/test_cli_shared.py:1 — Newly added RED tests encode the thin-wrapper acceptance criteria and must pass in this loop.
- specs/ptychodus_api_spec.md:191-210 — CLI contract and CONFIG-001 requirements demand legacy bridge order and backend routing remain intact.
- docs/workflows/pytorch.md:387-421 — Workflow guide documents accelerator flag usage and deprecation messaging that must remain consistent post-refactor.

How-To Map:
- Create artifact hub: `mkdir -p plans/active/ADR-003-BACKEND-API/reports/2025-10-20T111500Z/phase_d_cli_wrappers_training_impl`.
- Implement helpers in `ptycho_torch/cli/shared.py`: rehouse accelerator/device resolution, path validation (with `Path.mkdir(parents=True, exist_ok=True)`), and execution config construction using `PyTorchExecutionConfig`. Ensure warnings align with blueprint (DeprecationWarning for `--device`, UserWarning for deterministic+num_workers>0) and keep functions pure.
- Update `PyTorchExecutionConfig.__post_init__()` (`ptycho/config/config.py`) to enforce accelerator whitelist (`{"auto","cpu","gpu","cuda","tpu","mps"}`), non-negative workers, positive learning rates/batch sizes, and raise `ValueError` with clear messages when violated.
- Refactor `ptycho_torch/train.py`: add `--quiet` argparse flag (alias for legacy `--disable_mlflow`), annotate `--device` help text with deprecation notice, replace inline accelerator/path logic with helper calls, ensure CONFIG-001 ordering (`update_legacy_dict` before data loading) remains untouched, retain RawData loading (Option A), and route helper outputs into factory payload.
- After implementation, run each mapped pytest command with `CUDA_VISIBLE_DEVICES=""` and `tee` into artifact hub (`pytest_cli_shared_green.log`, `pytest_cli_train_green.log`, `pytest_workflows_train_cli_green.log`).
- Update `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md` B3.a–B3.d to `[x]`, log outcomes in the new `summary.md`, and append Attempt entry in docs/fix_plan.md referencing artifact paths.

Pitfalls To Avoid:
- Do not remove RawData loading yet—Option A keeps it inside the CLI for Phase D.
- Keep helper functions side-effect free beyond warnings; no direct logging or filesystem writes.
- Preserve CONFIG-001 order (`update_legacy_dict` call sequence) and backend routing logic.
- Maintain pytest style in existing tests—avoid introducing `unittest.TestCase` or shared state mutations.
- Ensure help text/deprecation warnings match blueprint wording to keep docs aligned.
- Avoid broad try/except around helper imports; let failures surface for GREEN tests.
- Don’t run full pytest suite; capture only the mapped selectors.
- Store all artifacts under the specified timestamped directory—no repo-root logs.
- Keep CLI default behaviours identical (accelerator default `cpu`, deterministic `True`).

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:32-36
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T105408Z/phase_d_cli_wrappers_training/training_refactor.md:120-356
- tests/torch/test_cli_shared.py:1
- ptycho/config/config.py:105
- ptycho_torch/train.py:380

Next Up: Begin Phase D.B4 documentation updates once the training CLI implementation is green.
