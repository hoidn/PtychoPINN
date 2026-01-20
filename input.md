Summary: Add intensity-normalization telemetry to the sim_lines runner and rerun the gs1/gs2 ideal scenarios so we can see where amplitude scaling drifts.
Focus: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy (Phase C4 intensity stats)
Branch: paper
Mapped tests: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T113000Z/

Do Now:
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py::main (add intensity-scale stats for raw/grouped/container tensors + recorded `legacy_params['intensity_scale']` and thread the new JSON/markdown paths into run_metadata) so each rerun emits an `intensity_stats.json` alongside the bias outputs.
- Validate: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T113000Z/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export RUN_HUB=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T113000Z && mkdir -p "$RUN_HUB/gs1_ideal" "$RUN_HUB/gs2_ideal"
3. python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py --scenario gs1_ideal --output-dir "$RUN_HUB/gs1_ideal" --group-limit 64 --nepochs 5 (stable profile will fill in base images / batch); archive updated run_metadata.json, stats JSON, intensity_stats.json, amplitude/phase PNGs, and training logs into the hub.
4. python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py --scenario gs2_ideal --output-dir "$RUN_HUB/gs2_ideal" --group-limit 64 --nepochs 5; confirm the new intensity telemetry emits scalar stats only (no raw arrays) and that summary markdown links are updated.
5. pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee "$RUN_HUB/pytest_cli_smoke.log"

Pitfalls To Avoid:
- Do not reuse the 2026-01-20T093000Z directories; every rerun must write into the 2026-01-20T113000Z hub so evidence stays grouped.
- Keep intensity_stats limited to scalar summaries (min/max/mean/std); never dump full tensors to JSON/Markdown.
- Make sure the new telemetry leaves existing run_metadata keys untouched (additive fields only) so downstream parsers stay stable.
- Continue exporting AUTHORITATIVE_CMDS_DOC before invoking the runner or pytest to satisfy governance checks.
- Stable profiles must remain in effect—avoid overriding base_total_images, group_count, or batch_size unless explicitly instructed.
- Remember CONFIG-001: the runner must continue to build configs via the existing helpers so params.cfg stays in sync; do not short-circuit that flow while adding stats.
- Capture a fresh pytest log even if the selector feels redundant; attach it to the new hub with the other evidence.
- Clean up matplotlib figures in the stats helper to avoid leaking file handles (use `plt.close`).

If Blocked:
- Document the blocker in docs/fix_plan.md under DEBUG-SIM-LINES-DOSE-001 (Attempts History) and drop a short note in plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md describing what failed.
- Tag the exact command + stderr and leave the partial artifacts in the RUN_HUB with a `_blocked.txt` note so we can pick up next loop.
- Ping me only after the blocker is recorded; the next supervisor loop will decide whether to pivot or escalate.

Findings Applied (Mandatory):
- CONFIG-001 — ensure every call to `run_phase_c2_scenario.py` keeps using update_legacy_dict via the Training/InferenceConfig factories before touching grouped data.
- NORMALIZATION-001 — keep physics/statistical/display scaling separate when logging new telemetry (record raw diffraction stats plus `intensity_scale` without mutating data).
- POLICY-001 — workflows assume TensorFlow dependencies are present; do not try to downshift to a CPU-only or torch-free path when rerunning the scenarios.

Pointers:
- specs/spec-ptycho-core.md:20 (Normalization invariants + intensity_scale contract)
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:170 (Phase C4 checklist)
- docs/fix_plan.md:90 (Latest Attempts and C4 scope)
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py:260 (stats helpers and runner entry point)
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/inspect_intensity_scaler.py:1 (reference for recent telemetry style)

Next Up (optional): If time remains, start comparing the logged stats vs recorded `intensity_scale` to decide whether C5 needs config tweaks or workflow math changes.
Doc Sync Plan (Conditional): Not needed this loop (no new/renamed tests).
Mapped Tests Guardrail: Selector above already collects (>0) and must be kept up to date.
Normative Math/Physics: Use specs/spec-ptycho-core.md §Normalization Invariants and docs/DEVELOPER_GUIDE.md §3.5 to keep the telemetry grounded in the official intensity-scaling math.
