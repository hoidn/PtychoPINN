Summary: Sync EB2 documentation—spec and workflow tables must describe dynamic monitor aliasing and scheduler/accum knobs.
Mode: Docs
Focus: [ADR-003-BACKEND-API] Phase EB2 — Scheduler & Gradient Accumulation
Branch: feature/torchapi
Mapped tests: none — docs-only
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T103000Z/{summary.md,spec_redline.md}
Do Now:
- EB2.C1 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-23T081500Z/eb2_plan.md — revise specs/ptychodus_api_spec.md §§4.9 & 7.1 so default `checkpoint_monitor_metric='val_loss'` explicitly maps to `model.val_loss_name` and note scheduler/accum defaults; tests: none.
- EB2.C2 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-23T081500Z/eb2_plan.md — update docs/workflows/pytorch.md §12 training flag table + narrative with the same monitor alias guidance and accumulation cautions; tests: none.
- EB2.C3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-23T081500Z/eb2_plan.md — capture spec/workflow diffs into spec_redline.md + loop summary, set EB2.C rows to [x], and add Attempt #64 in docs/fix_plan.md referencing the new artifact hub; tests: none.
If Blocked: Document unresolved questions + exact doc passages in plans/.../2025-10-23T103000Z/summary.md and leave EB2.C rows [P]; note blockers in docs/fix_plan.md.
Priorities & Rationale:
- specs/ptychodus_api_spec.md:270-283 — normative execution-config contract must match monitor/scheduler semantics after Attempt #63.
- docs/workflows/pytorch.md:302-330 — public CLI guide needs to describe new flags and dynamic monitor alias so users can reproduce GREEN evidence.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-23T081500Z/eb2_plan.md — EB2.C1–C3 remain open; closing them unblocks Phase EB3.
- docs/fix_plan.md:18-80 — Attempt log requires #64 entry summarizing documentation sync to keep ledger authoritative.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md — EB2.C currently [P]; needs to move to [x] once docs captured.
How-To Map:
- Update spec §4.9 bullet for `checkpoint_monitor_metric` and CLI table row (line ~278 and ~391) to state `'val_loss'` becomes `model.val_loss_name` when validation is present, including fallback to train metrics; mention scheduler/accum defaults referencing `PyTorchExecutionConfig` definitions.
- Refresh docs/workflows/pytorch.md training flag table row texts for `--checkpoint-monitor`, `--scheduler`, and `--accumulate-grad-batches`; add short paragraph explaining effective batch size and monitor aliasing, linking back to the spec.
- From repo root, run `git diff specs/ptychodus_api_spec.md docs/workflows/pytorch.md > plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T103000Z/spec_redline.md` after edits, and capture narrative summary in the same directory.
- Mark EB2.C1–C3 `[x]` in plans/.../eb2_plan.md and set EB2.C `[x]` in phase_e_execution_knobs/plan.md once docs/logs saved.
- Append Attempt #64 (Mode: Docs) to docs/fix_plan.md citing new artifact paths and reiterating that EB3 now next.
Pitfalls To Avoid:
- Do not touch production Python files; this loop is documentation-only.
- Keep spec/workflow tables column-aligned and ASCII; avoid smart quotes.
- Ensure spec/workflow wording stays in sync (no conflicting defaults or terminology).
- Include artifact paths in the new summary and ledger entry; missing paths break traceability.
- Do not delete prior timestamp directories (`2025-10-23T081500Z`, `2025-10-23T094500Z`).
- Avoid inventing new flag names—reuse existing CLI spelling exactly.
- When generating spec_redline.md, overwrite the file (no append of prior diffs) so the diff reflects this loop only.
- Double-check that Attempt numbering increments (use #64) to keep history ordered.
Pointers:
- specs/ptychodus_api_spec.md:270-283,391-400
- docs/workflows/pytorch.md:302-334
- plans/active/ADR-003-BACKEND-API/reports/2025-10-23T081500Z/eb2_plan.md
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md
- docs/fix_plan.md:18-80
Next Up: EB3 logger governance blueprint @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md once EB2 docs land.
