### Turn Summary
Extended `save_stats()` so `stats.json` now records the applied `prediction_scale`, keeping Phase C4e instrumentation aligned with the tracing spec.
Updated docs/fix_plan.md (new attempts entry) and galph_memory with the persistence fix, then reran the SIM-LINES CLI smoke guard for evidence (`pytest_cli_smoke.log`).
Next: rerun gs1_ideal/gs2_ideal under the new scaling hook and refresh the analyzer outputs per the active C4e checklist.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/ (pytest_cli_smoke.log)

### Turn Summary
Documented reviewer hygiene by fixing the gridsize guide / arch_writer links and logging DOC-HYGIENE-20260120 so future authors land on the right docs.
Synced the DEBUG plan (Phase A0/A1b/A2 + C4d) with existing evidence, recorded the A1b waiver, and scoped the Phase C4e amplitude-rescale prototype with a fresh artifacts hub.
Next: implement the `prediction-scale-source` hook in the runner + sim_lines pipeline, rerun gs1/gs2 with least-squares scaling, and re-run the CLI smoke/analyzer selectors.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/ (planning_notes.md, summary.md)
