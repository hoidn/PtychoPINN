Summary: Draft the Phase D training CLI thin-wrapper blueprint and capture RED pytest coverage before refactoring.
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase D.B (Training CLI thin wrapper)
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -vv (expected RED after new tests)
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T105408Z/phase_d_cli_wrappers_training/{training_refactor.md,pytest_cli_train_thin_red.log,summary.md}

Do Now:
1. ADR-003-BACKEND-API B1 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:24 — Author `training_refactor.md` (store in the new artifact hub) detailing helper/module layout, delegation flow, RawData ownership, accelerator warning strategy, and `disable_mlflow` handling; tests: none
2. ADR-003-BACKEND-API B2 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:25 — Extend `tests/torch/test_cli_train_torch.py` with RED coverage for the thin wrapper (new helper dispatch, legacy warning, payload handoff) and run `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -vv || true`, teeing output to `pytest_cli_train_thin_red.log`; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -vv
3. ADR-003-BACKEND-API plan sync @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:24+25 — Update checklist states (B1→`[x]`, B2→`[P]` or `[x]` per outcomes), capture loop notes in `summary.md`, and append the Attempt entry in docs/fix_plan.md referencing the new artifact hub; tests: none

If Blocked: Capture the blocker in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T105408Z/phase_d_cli_wrappers_training/blocker.md`, roll back any premature checklist changes, and log the issue in docs/fix_plan.md before pausing.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:22-27 — Phase D.B requires a blueprint and RED tests before code edits.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T131500Z/phase_d_cli_wrappers_baseline/baseline.md:10-76 — Baseline call graph highlights duplicated RawData loading and execution-config logic the blueprint must address.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T131500Z/phase_d_cli_wrappers_baseline/design_notes.md:9-78 — Legacy flag decisions (D1–D8) must be encoded in the plan and exercised by new tests.
- specs/ptychodus_api_spec.md:191-201 — Training workflow contract demands the CLI continue delegating to canonical workflows post-refactor.
- docs/workflows/pytorch.md:387-405 — Backend routing rules guide accelerator/CONFIG-001 handling for the redesigned entry points.

How-To Map:
- `mkdir -p plans/active/ADR-003-BACKEND-API/reports/2025-10-20T105408Z/phase_d_cli_wrappers_training`
- For B1, structure `training_refactor.md` with: overview, helper/module diagram, call flow (legacy vs new), hand-off to `run_cdi_example_torch`, accelerator resolution helper plan, MLflow toggle mapping, open questions list.
- For B2, add pytest coverage (native pytest style) to `tests/torch/test_cli_train_torch.py`, relying on mocks/fixtures to assert helper dispatch and warnings. Keep implementation TODOs minimal.
- Run the selector via `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T105408Z/phase_d_cli_wrappers_training/pytest_cli_train_thin_red.log || true` so failing tests remain visible.
- Summarise outputs and open issues in `summary.md` (include decisions, failing assertion highlights, follow-up actions).
- Update plan checklist cells directly in `phase_d_cli_wrappers/plan.md` and log the attempt in `docs/fix_plan.md` with artifact references.

Pitfalls To Avoid:
- Do not modify production CLI code in this loop—focus on docs and tests only.
- Keep new tests RED; avoid temporary hotfixes that accidentally make them pass.
- Store every artifact (md, log) in the specified timestamped directory; no repo-root debris.
- Maintain pytest style (no unittest mix-ins, no shared state across tests).
- Ensure new tests remain deterministic (set fixtures/seeds explicitly if needed).
- Reference helper names consistently with the blueprint to avoid churn later.
- Document any lingering legacy dependencies instead of removing them prematurely.
- Honour CONFIG-001 ordering in design—call it out where the blueprint expects `update_legacy_dict`.
- Record deprecation warnings in tests without asserting on exact strings unless stabilised.
- Clean tmp outputs (if any) before finishing the loop.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:22-52
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T131500Z/phase_d_cli_wrappers_baseline/baseline.md:10-76
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T131500Z/phase_d_cli_wrappers_baseline/design_notes.md:20-118
- specs/ptychodus_api_spec.md:191-205
- docs/workflows/pytorch.md:387-421

Next Up: Phase D.B3 implementation (thin wrapper refactor + GREEN tests) once the blueprint and RED coverage are in place.
