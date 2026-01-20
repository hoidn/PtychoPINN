Summary: Bake the reduced-load profiles into the Phase C2 runner so ideal-scenario reruns no longer rely on manual CLI overrides.
Focus: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy
Branch: paper
Mapped tests: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T063500Z/

Do Now (hard validity contract)
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py::main — add baked-in “stable profile” overrides for `gs1_ideal` (base_total_images=512, group_count=256, batch_size=8) and `gs2_ideal` (base_total_images=256, group_count=128, batch_size=4, neighbor_count=4) that trigger automatically, record the applied profile in `run_metadata.json`, warn when a caller overrides those knobs, then rerun both scenarios without manual CLI flags to regenerate amplitude/phase dumps, PNGs, stats, inspection notes, and reassembly telemetry under the new artifacts hub.
- Pytest: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T063500Z/{gs1_ideal_runner.log,gs2_ideal_runner.log,gs1_ideal/inference_outputs/*,gs2_ideal/inference_outputs/*,gs1_ideal_notes.md,gs2_ideal_notes.md,reassembly_cli.log,reassembly_gs1_ideal.json,reassembly_gs1_ideal.md,reassembly_gs2_ideal.json,reassembly_gs2_ideal.md,pytest_cli_smoke.log}

How-To Map
1. export ARTIFACT_DIR=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T063500Z && mkdir -p "$ARTIFACT_DIR".
2. Edit `bin/run_phase_c2_scenario.py`:
   - Add a `STABLE_PROFILES` dict mapping scenario names to overrides (base_total_images, group_count, batch_size, neighbor_count, label).
   - When a requested scenario appears in the dict, apply the overrides before derive_counts/TrainingConfig creation, store `profile` + applied overrides in `run_metadata`, and emit a warning if the user passed conflicting CLI flags.
   - Ensure batch_size overrides propagate through `TrainingConfig`, but still respect explicit CLI values when provided (log that the manual override disabled the profile default).
3. Re-run the runner without manual load flags:
   - `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py --scenario gs1_ideal --output-dir "$ARTIFACT_DIR/gs1_ideal" --snapshot plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json --group-limit 64 | tee "$ARTIFACT_DIR/gs1_ideal_runner.log"`
   - `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py --scenario gs2_ideal --output-dir "$ARTIFACT_DIR/gs2_ideal" --snapshot plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json --group-limit 64 | tee "$ARTIFACT_DIR/gs2_ideal_runner.log"`
4. Update `$ARTIFACT_DIR/gs1_ideal_notes.md` and `$ARTIFACT_DIR/gs2_ideal_notes.md` with manual inspection commentary referencing the refreshed PNGs.
5. Refresh reassembly telemetry off the new runs:
   - `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/reassembly_limits_report.py --scenario gs1_ideal --group-limit 64 --output-json "$ARTIFACT_DIR/reassembly_gs1_ideal.json" --output-markdown "$ARTIFACT_DIR/reassembly_gs1_ideal.md" | tee "$ARTIFACT_DIR/reassembly_cli.log"`
   - `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/reassembly_limits_report.py --scenario gs2_ideal --group-limit 64 --output-json "$ARTIFACT_DIR/reassembly_gs2_ideal.json" --output-markdown "$ARTIFACT_DIR/reassembly_gs2_ideal.md" | tee -a "$ARTIFACT_DIR/reassembly_cli.log"`
6. Run the pytest guard and record the log: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee "$ARTIFACT_DIR/pytest_cli_smoke.log"`.
7. Verify the artifacts hub contains all logs/PNGs/JSON (`ls "$ARTIFACT_DIR" "$ARTIFACT_DIR"/gs*_ideal`) before finishing.

Pitfalls To Avoid
- Confine edits to the plan-local runner; do not touch `ptycho/` modules this loop.
- Preserve CONFIG-001 compliance by reusing workflow helpers for training/inference.
- Keep the baked-in profile additive—manual CLI overrides should still work but log that the profile defaults were bypassed.
- Do not alter probe normalization or physics parameters; only workload knobs change.
- Maintain consistent PNG color scaling (reuse stats-based vmax) for comparability.
- Run each scenario via a fresh CLI invocation to avoid stale params state.
- Capture stdout/stderr via tee even on failures; stash logs under the hub.
- Honor PYTHON-ENV-001 (call `python`, no interpreter shims) and keep execution on GPU (reduce workloads rather than dropping to CPU if resources tighten).
- Ensure `run_metadata.json` clearly documents the applied profile + overrides for auditing.
- Keep reassembly CLI logs appended to the same file to show both scenarios in order.

If Blocked
- If NaNs or OOMs persist even with the baked-in profile, tee the failing command output into `$ARTIFACT_DIR/blocker.log`, summarize the failure in docs/fix_plan.md Attempts History, and mark the focus blocked before proceeding.

Findings Applied (Mandatory)
- CONFIG-001 — rely on canonical helpers so legacy params sync happens before grouping/inference.
- MODULE-SINGLETON-001 — instantiate models via factories and never touch module-level singletons directly.
- NORMALIZATION-001 — keep physics/statistical/display normalization paths separate; workload tweaks must not mix them.
- BUG-TF-REASSEMBLE-001 — reassembly telemetry must keep padded sizes integer/consistent to avoid the mixed-type crash.

Pointers
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:148 — C2b checklist describing the baked-in profile requirements and overrides.
- docs/specs/spec-ptycho-workflow.md:46 — Reassembly requirement `M ≥ N + 2·max(|dx|,|dy|)` we continue to validate with the telemetry.
- docs/TESTING_GUIDE.md:1 — Pytest guard conventions and evidence logging policy for CLI tests.

Next Up (optional)
- After C2b lands, pivot to C3 (root-cause recap + doc updates) using the refreshed stats.

Doc Sync Plan — N/A (existing selector, no registry updates).
Mapped Tests Guardrail — `pytest --collect-only tests/scripts/test_synthetic_helpers_cli_smoke.py -q` must show 1 collected test; fix the module if collection fails before coding.
Normative Math/Physics — Reference `docs/specs/spec-ptycho-workflow.md §Reassembly Requirements` whenever reasoning about padded size vs offsets.
