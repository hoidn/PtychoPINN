Summary: Clean up Phase C2 artifacts and document Phase C3 evidence for the PyTorch integration pytest.
Mode: Docs
Focus: TEST-PYTORCH-001 — Phase C pytest modernization
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
Artifacts: plans/active/TEST-PYTORCH-001/reports/2025-10-19T130900Z/phase_c_modernization/{artifact_audit.md,pytest_modernization_rerun.log,summary.md}

Do Now:
1. TEST-PYTORCH-001 C2 cleanup @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T120415Z/phase_c_modernization/plan.md — `git mv train_debug.log plans/active/TEST-PYTORCH-001/reports/2025-10-19T122449Z/phase_c_modernization/train_debug.log`, refresh `summary.md` in that directory to note the log, update the in-module comment in `tests/torch/test_integration_workflow_torch.py:188` (remove “currently raises NotImplementedError”), and flip the C2 row in `plans/active/TEST-PYTORCH-001/implementation.md` to `[x]` with a completion note (tests: none).
2. TEST-PYTORCH-001 C3.A @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T120415Z/phase_c_modernization/plan.md — Re-run `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv | tee plans/active/TEST-PYTORCH-001/reports/2025-10-19T130900Z/phase_c_modernization/pytest_modernization_rerun.log`, record the returned tmp_path in notes, and inspect the generated training/inference folders for artifact details (tests: pytest … -vv).
3. TEST-PYTORCH-001 C3.B+C3.C @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T120415Z/phase_c_modernization/plan.md — Capture findings in `artifact_audit.md`, add a fresh `summary.md` for this loop, update `plans/active/TEST-PYTORCH-001/implementation.md` (note C3 progress), and log Attempt #7 in `docs/fix_plan.md` with paths to the new artifacts (tests: none).

If Blocked: If the rerun fails or artifacts are missing, keep the new directory, store the failing log and any stdout/stderr dumps there, and document the failure mode + hypotheses in `artifact_audit.md` before updating docs/fix_plan.md.

Priorities & Rationale:
- train_debug.log currently sits at repo root; plan & ledger require it under `plans/active/TEST-PYTORCH-001/reports/2025-10-19T122449Z/phase_c_modernization/`.
- tests/torch/test_integration_workflow_torch.py:188 still claims the helper raises `NotImplementedError`, contradicting C2 implementation.
- plans/active/TEST-PYTORCH-001/implementation.md:50 shows C2 `[ ]` even though GREEN run succeeded; ledger accuracy depends on flipping it.
- Phase C3 plan (plan.md §C3) expects artifact audit + documentation; no evidence exists yet for those rows.
- docs/findings.md#POLICY-001 requires PyTorch selectors to remain targeted; rerun log documents compliance.

How-To Map:
- Move log: `git mv train_debug.log plans/active/TEST-PYTORCH-001/reports/2025-10-19T122449Z/phase_c_modernization/train_debug.log`.
- Update existing summary: append a short bullet noting the relocated log and its relevance.
- Create new artifact hub: `mkdir -p plans/active/TEST-PYTORCH-001/reports/2025-10-19T130900Z/phase_c_modernization`.
- Targeted rerun: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv | tee plans/active/TEST-PYTORCH-001/reports/2025-10-19T130900Z/phase_c_modernization/pytest_modernization_rerun.log`.
- After pytest, note `tmp_path` (print from log or inspect `pytest-of-*` directories) and record artifact sizes/checkpoints in `artifact_audit.md`.
- Update plan + ledger: edit `plans/active/TEST-PYTORCH-001/implementation.md` C2 to `[x]`, add C3 notes; append Attempt #7 in `docs/fix_plan.md` with new artifact paths.
- New loop summary: add runtime + findings to `plans/active/TEST-PYTORCH-001/reports/2025-10-19T130900Z/phase_c_modernization/summary.md`.

Pitfalls To Avoid:
- Do not leave train_debug.log at repo root after this loop.
- Keep CUDA disabled via `cuda_cpu_env`; avoid mutating global `os.environ`.
- Treat pytest rerun artifacts as transient—capture sizes/paths immediately before cleanup.
- Do not adjust subprocess CLI parameters; maintain Phase C2 settings.
- Avoid touching PyTorch workflow source files; focus on tests and documentation.
- Ensure artifact filenames match exactly (case-sensitive) when recording in audit.
- Update all referenced documents in the same loop; no stale checkboxes or ledger omissions.
- Retain pytest log in the new timestamped directory even when green.
- Use `git add` on moved log to prevent deletion.
- Keep Do Now scope tight—no extra refactors.

Pointers:
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T120415Z/phase_c_modernization/plan.md#phase-c3 — Checklist for this work.
- tests/torch/test_integration_workflow_torch.py:1-210 — Pytest harness & comments to update.
- plans/active/TEST-PYTORCH-001/implementation.md:35-60 — Phase C tracking table.
- docs/fix_plan.md:140-170 — TEST-PYTORCH-001 attempts history (append #7).
- docs/workflows/pytorch.md:120-160 — Artifact expectations for train/infer pipeline.

Next Up: Phase C3 completion unlocks Phase D hardening (CI markers + documentation updates).
