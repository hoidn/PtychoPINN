Summary: Document the new checkpoint CLI knobs and sync EB1 planning artifacts.
Mode: Docs
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase E.B1 (checkpoint controls)
Branch: feature/torchapi
Mapped tests: none — docs-only
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-20T163500Z/{summary.md,spec_updates.md,workflow_table_diff.md}

Do Now:
1. EB1.A (spec refresh) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md — update `specs/ptychodus_api_spec.md` §4.9 + §7 training CLI table for the checkpoint flags (`checkpoint_mode`, enable/disable checkpointing, save-top-k, monitor, early-stop patience); document deltas under `spec_updates.md`; tests: none.
2. EB1.F (workflow+ledger sync) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md — extend `docs/workflows/pytorch.md` §12 training table with the checkpoint knobs, mark EB1.A/F rows `[x]`, add Attempt #60 in `docs/fix_plan.md`, and capture a loop `summary.md`; tests: none.

If Blocked: Capture the issue + context in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-20T163500Z/blockers.md`, leave EB1.F `[P]`, and log the blocker in docs/fix_plan.md.

Priorities & Rationale:
- specs/ptychodus_api_spec.md:270 — checkpoint knobs still tagged “CLI backlog”; must reflect shipped flags.
- docs/workflows/pytorch.md:322 — training CLI table omits new checkpoint flags added in commit 496a8ce3.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md:27 — EB1.A/F remain open; plan needs closing evidence.
- docs/fix_plan.md:168 — ledger needs Attempt #60 to document doc updates and evidence location.

How-To Map:
- Edit `specs/ptychodus_api_spec.md` §4.9 “Checkpoint/Logging Knobs” list: add `checkpoint_mode`, remove “CLI backlog” wording, note defaults/validation, and reference Lightning callbacks; update §7.1 training CLI table to include `--enable-checkpointing/--disable-checkpointing`, `--checkpoint-save-top-k`, `--checkpoint-monitor`, `--checkpoint-mode`, `--early-stop-patience` with default values and config links.
- Update `docs/workflows/pytorch.md` §12 training flags table to mirror the new spec entries; add short guidance on when to disable checkpointing and how monitor/mode interact with validation data.
- After docs edits, set EB1.A/EB1.F rows to `[x]` in `phase_e_execution_knobs/plan.md`, append Attempt #60 (Mode: Docs) to the fix-plan entry summarizing doc changes + evidence paths, and note “tests: not run”.
- Store a concise `summary.md` and supporting `spec_updates.md` / `workflow_table_diff.md` under the `2025-10-20T163500Z` directory describing edits and linking to doc sections.

Pitfalls To Avoid:
- Don’t alter production Python code—this loop is documentation only.
- Preserve Markdown tables’ alignment and ASCII characters.
- Keep terminology consistent with spec (e.g., `enable_checkpointing`, `checkpoint_mode`).
- Ensure CLI defaults in docs mirror actual argparse defaults (`'auto'`, `True`, `1`, etc.).
- Note when monitor defaults require validation data; avoid promising behaviour not implemented.
- Update plan and ledger in the same loop; no deferred bookkeeping.
- Keep evidence, logs, and summaries inside the timestamped reports directory.
- Mention “tests: not run” explicitly in Attempt #60.

Pointers:
- specs/ptychodus_api_spec.md:268
- docs/workflows/pytorch.md:320
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md:27
- docs/fix_plan.md:168

Next Up: EB2 scheduler + accumulation knobs once EB1 documentation closes.
