Summary: Extend the grouping instrumentation so B3 can log per-axis stats and capture gs1/gs2 A/B runs (including the neighbor-count failure) to pin whether grouping—not probe scaling—is the root cause.
Focus: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy
Branch: paper (sync with origin/paper)
Mapped tests: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T041700Z/

Do Now (hard validity contract)
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/grouping_summary.py::describe_array,main — add per-axis stats (min/max/mean/std per coordinate axis plus nn_indices min/max), update the Markdown output accordingly, then rerun the CLI for gs1 default, gs2 default, and gs2 neighbor-count=1 so the refreshed summaries (and CLI log) land under `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T041700Z/` as required by Phase B3.
- Pytest: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T041700Z/{grouping_gs1_custom_default.json, grouping_gs1_custom_default.md, grouping_gs2_custom_default.json, grouping_gs2_custom_default.md, grouping_gs2_custom_neighbor1.json, grouping_gs2_custom_neighbor1.md, grouping_cli.log, pytest_sim_lines_pipeline_import.log}

How-To Map
1. export ARTIFACT_DIR=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T041700Z && mkdir -p "$ARTIFACT_DIR".
2. Update `describe_array()` to compute overall min/max/mean/std plus `axis_stats` whenever the array’s penultimate dimension is 2 (coords axes), and add nn_indices min/max reporting inside `summarize_subset()`; ensure Markdown tables include the new numbers.
3. Run (identical seeds across all invocations):
   - python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/grouping_summary.py --scenario gs1_custom --label gs1_custom_default --output-json "$ARTIFACT_DIR/grouping_gs1_custom_default.json" --output-markdown "$ARTIFACT_DIR/grouping_gs1_custom_default.md" | tee "$ARTIFACT_DIR/grouping_cli.log"
   - python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/grouping_summary.py --scenario gs2_custom --label gs2_custom_default --output-json "$ARTIFACT_DIR/grouping_gs2_custom_default.json" --output-markdown "$ARTIFACT_DIR/grouping_gs2_custom_default.md" | tee -a "$ARTIFACT_DIR/grouping_cli.log"
   - python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/grouping_summary.py --scenario gs2_custom --label gs2_custom_neighbor1 --neighbor-count 1 --output-json "$ARTIFACT_DIR/grouping_gs2_custom_neighbor1.json" --output-markdown "$ARTIFACT_DIR/grouping_gs2_custom_neighbor1.md" | tee -a "$ARTIFACT_DIR/grouping_cli.log"
4. pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee "$ARTIFACT_DIR/pytest_sim_lines_pipeline_import.log"

Pitfalls To Avoid
- Do not touch production modules under ptycho/ or scripts/; keep instrumentation in plans/active.
- Run `update_legacy_dict(params.cfg, config)` before any RawData or probe helper so CONFIG-001 is satisfied.
- Preserve deterministic seeds (object, sim, neighbor) so A/B comparisons are meaningful; capture them in the metadata block.
- Only append fields to the JSON schema; do not rename/remove existing keys consumed by earlier artifacts.
- Handle the neighbor-count failure by recording a clean `status:error` block—no stack traces to stdout.
- Keep commands device-neutral (CPU-friendly) and avoid writing large raw arrays outside the artifacts hub.
- Continue to invoke Python via `python` per PYTHON-ENV-001; no virtualenv-specific paths.
- Archive every CLI + pytest log in the artifacts directory before exiting the loop.

If Blocked
- If grouping fails for all scenarios or the snapshot file is missing, record the error string in `$ARTIFACT_DIR/blocker.log`, add the signature to docs/fix_plan.md Attempts History, set `<status>blocked</status>` in input.md, and stop after capturing the pytest output.

Findings Applied (Mandatory)
- CONFIG-001 — synchronize `params.cfg` via `update_legacy_dict` before hitting legacy grouping/probe helpers so gridsize and neighbor_count match the TrainingConfig inputs.
- MODULE-SINGLETON-001 — avoid depending on implicit module-level state; instantiate probes/RawData inside the CLI and don’t let global singletons leak between runs.
- NORMALIZATION-001 — we already proved probe normalization aligned; keep the grouping runs focused on KDTree behavior so physics/statistical normalization bindings stay untouched.
- BUG-TF-001 — by holding seeds/config consistent we minimize surprise shape mismatches in TF grouping paths when toggling gridsize.

Pointers
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:114 — Phase B3 checklist describing the richer telemetry + three scenarios required this loop.
- docs/fix_plan.md:38-57 — Attempts History and new 2026-01-16T041700Z entry documenting why B3 instrumentation is next.
- docs/specs/spec-ptycho-workflow.md:12-29 — Normative grouping + normalization contracts we must honor while instrumenting.
- docs/DATA_GENERATION_GUIDE.md:5-120 — Grid vs nongrid simulation overview that explains why kd-tree needs ≥gridsize² neighbors.
- plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T031500Z/probe_stats_gs1_custom.md — Prior evidence showing probe scaling parity; cite when writing the summary.

Next Up (optional)
- Once B3 telemetry is in place, start sketching the B4 reassembly experiment that reuses a fixed synthetic container to compare stitch math.

Doc Sync Plan — not needed (no new tests added or renamed).
Mapped Tests Guardrail: the CLI smoke selector above already collects/passes (>0) and remains mandatory for every loop.
Normative Math/Physics: cite `docs/specs/spec-ptycho-workflow.md` §2–3 for grouping/normalization math whenever referencing equations; do not restate the formulas inside the CLI.
