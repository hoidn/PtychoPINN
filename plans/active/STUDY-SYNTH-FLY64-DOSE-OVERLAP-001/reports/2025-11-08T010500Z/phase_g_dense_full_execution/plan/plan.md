# Phase G Dense Full Execution — Plan (2025-11-08T010500Z)

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Focus:** Phase G comparison & analysis (dense evidence)  
**Action Type:** Planning (supervisor loop)  
**Target State:** ready_for_implementation

---

## Context

- `test_run_phase_g_dense_collect_only_generates_commands` shipped in the previous loop; CLI wiring is now guarded by a dry-run smoke test.  
- `validate_phase_c_metadata` enforces both `_metadata` presence and `transpose_rename_convert` tracking, with tests covering RED/GREEN paths.  
- The dense Phase C→G pipeline has not yet been executed end-to-end on a clean hub; no recent manifest/metrics artifacts exist for dose=1000, view=dense.  
- `summarize_phase_g_outputs` produces per-job JSON/Markdown, but the summary lacks top-level aggregates that make Turn Summary authoring faster.

## Objectives

1. Extend `summarize_phase_g_outputs` to compute and log aggregate MS-SSIM/MAE statistics (per model, per metric) alongside the existing per-job tables.  
2. Update pytest coverage so `test_summarize_phase_g_outputs` asserts the new aggregates in both JSON and Markdown outputs.  
3. Run the dense Phase G orchestrator with `--clobber` to regenerate artifacts under a fresh hub, capturing CLI logs and resulting manifest/metrics.  
4. Invoke `validate_phase_c_metadata` and the enhanced `summarize_phase_g_outputs` on the new outputs, archiving logs plus the enriched summaries.  
5. Refresh documentation (`summary.md`, `docs/TESTING_GUIDE.md`, `docs/development/TEST_SUITE_INDEX.md`, `docs/fix_plan.md`) with the new evidence and selector inventory.

## Deliverables

- Updated `summarize_phase_g_outputs` implementation emitting aggregate statistics (e.g., best/worst MS-SSIM amplitude & phase per model).  
- Strengthened pytest expectations for `test_summarize_phase_g_outputs` verifying aggregate fields and Markdown headers.  
- CLI transcript `cli/phase_g_dense_pipeline.log` plus regenerated Phase C/G artifacts under the new hub.  
- Validation log `analysis/validate_and_summarize.log` showing guard success and aggregate summary emission.  
- Turn summary + metrics snapshot recorded in `summary/summary.md` and mirrored in supervisor reply; docs/test registries synced.

## Task Breakdown

1. **Implementation — Aggregate Metrics**
   - Modify `plans/active/.../bin/run_phase_g_dense.py::summarize_phase_g_outputs` to compute per-model aggregates (mean & best MS-SSIM amplitude/phase, mean MAE amplitude/phase) and include them in both JSON and Markdown summaries.
   - Ensure aggregates retain deterministic ordering and clearly labeled sections (e.g., `## Aggregate Metrics`).

2. **TDD — Strengthen Summary Tests**
   - Update `tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs` to assert aggregate fields in the JSON payload and presence of the new Markdown section.
   - Keep failure messages actionable; maintain TYPE-PATH-001 and DATA-001 compliance in fixtures.

3. **Dense Pipeline Execution**
   - Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`.  
   - Run `python plans/.../bin/run_phase_g_dense.py --hub <this hub> --dose 1000 --view dense --splits train test --clobber | tee .../cli/phase_g_dense_pipeline.log`.  
   - Preserve manifest/metrics outputs under `analysis/`; on failure, capture traceback to `analysis/blocker.log`.

4. **Post-run Validation**
   - Execute helper shim to call `validate_phase_c_metadata(hub)` and `summarize_phase_g_outputs(hub)`; tee logs to `analysis/validate_and_summarize.log`.  
   - Verify `metrics_summary.json` now contains `aggregate_metrics`, and Markdown summary includes the aggregate section.

5. **Documentation & Ledger Updates**
   - Update `summary/summary.md` with Turn Summary, aggregate highlights, and log pointers.  
   - Append Attempts History entry to `docs/fix_plan.md`, refresh findings references, and sync test registries once GREEN.

## Findings & Guardrails Reinforced

- POLICY-001 — Dense pipeline may spawn PyTorch baselines; environment must retain torch>=2.2.  
- CONFIG-001 — CLI helper already bridges legacy params; avoid altering initialization order.  
- DATA-001 — Guard remains read-only; aggregate computation must not mutate NPZ artifacts.  
- TYPE-PATH-001 — Normalize all paths in script/tests.  
- OVERSAMPLING-001 — Dense overlap parameters stay unchanged; aggregates should not reinterpret datasets.

## Pitfalls

- Avoid floating-point drift by rounding aggregates only for presentation (keep raw floats in JSON).  
- Long-running CLI: abort on first failure, log blocker, and stop the loop.  
- Do not overwrite previous hubs; rely on `prepare_hub` and the new timestamped path.  
- Update documentation only after aggregates & CLI run succeed; otherwise, mark the attempt blocked with rationale.
