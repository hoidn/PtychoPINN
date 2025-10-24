Summary: Sync normative docs and ledger for logger backend before moving to governance close-out.
Mode: Docs
Focus: [ADR-003-BACKEND-API] Phase EB3 — Logger backend implementation (Phase C docs)
Branch: feature/torchapi
Mapped tests: none — evidence-only
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/docs/2025-10-24T041500Z/{spec_redline.md,summary.md}
Do Now:
- [ADR-003-BACKEND-API] EB3.C1 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md — update specs/ptychodus_api_spec.md §4.9 + §7.1 with logger defaults/options, DeprecationWarning text, and capture diff to docs/spec_redline.md; tests: none.
- [ADR-003-BACKEND-API] EB3.C2 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md — refresh docs/workflows/pytorch.md §12 (training execution flags + prose) to document `--logger` behaviour and MLflow deprecation; produce docs/summary.md; tests: none.
- [ADR-003-BACKEND-API] EB3.C3+C4 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md — add CONFIG-LOGGER-001 finding, update implementation plan + fix_plan Attempt #69, and note MLflow logger backlog link; tests: none.
If Blocked: Record blockers in docs/summary.md and docs/fix_plan.md Attempt log, leaving checklist rows `[P]`; include follow-up questions for supervisor review.
Priorities & Rationale:
- plans/.../plan.md Phase C rows — authoritative checklist; must flip `[x]` before Phase EB3.D.
- specs/ptychodus_api_spec.md:281 — currently states logger backend is pending; must reflect approved CSV default and CLI flag.
- docs/workflows/pytorch.md:312 — training table lacks `--logger` entry; needs parity with spec and CLI.
- docs/findings.md:11 — add policy entry documenting logger default + options for future regressions.
- decision/approved.md — governance record mandates documenting MLflow backlog and DeprecationWarning.
How-To Map:
- Spec update: edit specs/ptychodus_api_spec.md to (1) rewrite the `logger_backend` bullet in §4.9 with CSV default + allowed values + fallback behaviour and dependency notes, (2) remove logger from the “programmatic-only” backlog note near line 424, (3) add `--logger` row to §7.1 table (include default `'csv'`, field `PyTorchExecutionConfig.logger_backend`, behaviour summary including `'none'` option and DeprecationWarning for `--disable_mlflow`), then run `git diff specs/ptychodus_api_spec.md > plans/.../docs/2025-10-24T041500Z/spec_redline.md`.
- Workflow guide: update docs/workflows/pytorch.md §12 training flag table with `--logger` entry and adjust prose to describe CSV default, enabling TensorBoard/MLflow, disabling via `'none'`, and the deprecation path for `--disable_mlflow`; append notes about storing manual run outputs in existing evidence; summarize changes in `plans/.../docs/2025-10-24T041500Z/summary.md`.
- Findings & ledgers: add a new row (e.g., `CONFIG-LOGGER-001`) to docs/findings.md referencing decision/approved.md and the new documentation; update plans/active/ADR-003-BACKEND-API/implementation.md Phase E narrative + docs/fix_plan.md Attempt #69 with artifact paths and backlog pointer; mention upcoming fix_plan entry for MLflow logger migration per plan row C4.
- Hygiene: ensure all new artifacts live under `plans/.../docs/2025-10-24T041500Z/`; keep files ASCII; verify no stray TODOs remain.
Pitfalls To Avoid:
- Do not modify production code—docs + ledgers only this loop.
- Preserve markdown tables alignment; keep pipes vertically aligned.
- Document DeprecationWarning verbiage verbatim to avoid drift with CLI output.
- Ensure spec and workflow guide reference identical option sets/order (`csv`, `tensorboard`, `mlflow`, `none`).
- Update findings with stable ID formatting (`| ID | Date | Tags | Summary | Evidence | Status |`).
- When exporting `git diff`, overwrite (not append) spec_redline.md to avoid stale diffs.
- Reference artifact paths relative to repo root in fix_plan/summary.
- Avoid introducing new `<doc-ref>` tags without verifying they resolve.
- Keep summary.md concise (<1k words) but include table of artifacts + checklist status.
- Double-check plan checklist rows flipped `[x]` before staging.
Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md
- specs/ptychodus_api_spec.md:271
- docs/workflows/pytorch.md:302
- docs/findings.md:11
- plans/active/ADR-003-BACKEND-API/implementation.md:120
- docs/fix_plan.md#ADR-003-BACKEND-API
Next Up: Phase EB3.D optional smoke/CI guidance once documentation sync is complete.
