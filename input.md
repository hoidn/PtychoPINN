Summary: Instrument the Phase D runner/analyzer so every scenario records train/test dataset intensity stats and exposes their ratios versus the bundle scale, giving us concrete evidence for Phase D5.
Focus: DEBUG-SIM-LINES-DOSE-001 — Phase D5 train/test intensity-scale parity instrumentation
Branch: paper
Mapped tests: pytest tests/scripts/test_analyze_intensity_bias.py::TestDatasetStats::test_reports_train_test -v; pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T024500Z/
Do Now (DEBUG-SIM-LINES-DOSE-001 / D5 — see implementation.md §Phase D5):
- Implement: `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py::run_scenario`. Before training, call `compute_dataset_intensity_stats()` on both `train_raw.diff3d` and `test_raw.diff3d`, derive the per-split dataset scale (`sqrt(nphotons / batch_mean_sum_intensity)`), and persist a `split_intensity_stats` block inside `run_metadata.json`. Keep computations NumPy-only so PINN-CHUNKED-001 remains intact and cite `specs/spec-ptycho-core.md §Normalization Invariants` in the metadata note.
- Implement: `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py::summarize_results` (and supporting helpers) to parse the new `split_intensity_stats` payload, report train vs test dataset scales alongside the bundle value in both JSON + Markdown, and flag any >5% deviation per the spec. Include the new table in the Markdown summary for each scenario with clear labels.
- Implement: `tests/scripts/test_analyze_intensity_bias.py::TestDatasetStats::test_reports_train_test`. Build a minimal fixture (temp directory with stub `run_metadata.json` + required analyzer inputs) that proves the analyzer surfaces the new train/test stats and tolerates missing metadata gracefully.
- Validate via: (1) `pytest tests/scripts/test_analyze_intensity_bias.py::TestDatasetStats::test_reports_train_test -v`; (2) rerun both stable profiles with enriched telemetry —
  `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py --scenario gs1_ideal --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T024500Z/gs1_ideal --prediction-scale-source least_squares`
  and
  `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py --scenario gs2_ideal --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T024500Z/gs2_ideal --prediction-scale-source least_squares`;
  (3) regenerate the summary `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py --scenario gs1_ideal=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T024500Z/gs1_ideal --scenario gs2_ideal=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T024500Z/gs2_ideal --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T024500Z/`;
  (4) `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`.
How-To Map:
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`.
2. Edit `bin/run_phase_c2_scenario.py` to compute and stash train/test stats; format them as Python floats in `run_metadata.json`.
3. Update `bin/analyze_intensity_bias.py` to load the new metadata, extend the JSON payload, and add the Markdown table.
4. Author `tests/scripts/test_analyze_intensity_bias.py` (or extend the module) with the new regression test; keep fixtures under `tests/data/` or use `tmp_path`.
5. Run `pytest tests/scripts/test_analyze_intensity_bias.py::TestDatasetStats::test_reports_train_test -v | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T024500Z/logs/pytest_analyzer_dataset_stats.log`.
6. Execute the gs1_ideal + gs2_ideal runners with the commands above; archive each `*_runner.log`, `intensity_stats.json`, and `run_metadata.json` under the new hub.
7. Generate the refreshed `bias_summary` using the analyzer command, capturing its stdout to `bias_summary.log` in the hub.
8. Finish with `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T024500Z/logs/pytest_cli_smoke.log`.
Pitfalls To Avoid:
- Don’t touch `.X`/`.Y` — use the raw `diff3d` arrays so lazy containers stay CPU-bound (PINN-CHUNKED-001).
- Ensure train/test stats use the same nphotons as the bundle; mismatched params.cfg values will poison the comparison (SIM-LINES-CONFIG-001).
- Keep new metadata keys backward compatible (analyzer should tolerate scenarios created before this change).
- Use float64 for reductions but cast to Python floats before JSON so logs stay human-readable.
- The analyzer regression test must not depend on GPU; stub arrays < 4×4 to keep runtime trivial.
- When rerunning scenarios, reuse the baked stable profiles (do not bump workloads or neighbor counts) so comparisons stay apples-to-apples.
- Record every CLI invocation under the artifacts hub; missing evidence violates TEST-CLI-001.
- Update docs only after tests pass; no premature edits to specs.
- Leave `docs/DATA_GENERATION_GUIDE.md` untouched this loop (D4f docs already landed).
- Avoid mutating existing reports; write new outputs under the 2026-01-21T024500Z hub.
If Blocked: capture the failure signature (stack trace or mismatched stats) in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T024500Z/blocked.md`, update docs/fix_plan.md Attempts History with the repro command, and stop work.
Findings Applied (Mandatory):
- NORMALIZATION-001 — enforce the separation between normalize_data’s `(N/2)^2` target and the dataset-derived intensity scale when interpreting split stats.
- PINN-CHUNKED-001 — keep dataset statistic computation on NumPy buffers so lazy loading and chunked inference remain viable.
- SIM-LINES-CONFIG-001 — always run `update_legacy_dict(params.cfg, config)` before loader/training so recorded stats line up with the active params.
Pointers:
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:387 (Phase D5 checklist)
- specs/spec-ptycho-core.md:80 (Normalization invariants formula)
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py:330 (record_intensity_stage helpers)
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py:70 (scenario parsing logic)
- docs/fix_plan.md:401 (DEBUG-SIM-LINES-DOSE-001 attempts history)
Next Up (optional): With train/test stats captured, we can pivot to D5b instrumenting forward-pass energy (IntensityScaler output vs prediction) if evidence points past normalization.
Doc Sync Plan: After code and tests pass, run `pytest --collect-only tests/scripts/test_analyze_intensity_bias.py -q` and archive the log under the new hub, then update `docs/TESTING_GUIDE.md §2` and `docs/development/TEST_SUITE_INDEX.md` to reference the new analyzer regression before closing the loop.
