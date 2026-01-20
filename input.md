Summary: Capture ground-truth comparison metrics in the Phase C2 runner so we can quantify how far gs1_ideal diverges from gs2_ideal despite matching telemetry.
Focus: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy
Branch: paper
Mapped tests: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T083000Z/

Do Now (hard validity contract)
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py::main — extend the runner with helpers that (a) save the simulated `object_guess` amplitude/phase (NPY + PNG), (b) center-crop stitched reconstructions to `RunParams.object_size`, (c) emit amplitude/phase diff metrics (MAE, RMSE, max abs error, Pearson r) plus diff PNGs, and (d) record the new artifact paths + metrics in `run_metadata.json`; rerun gs1_ideal and gs2_ideal so the comparison bundles live under the new hub.
- Pytest: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T083000Z/{gs1_ideal_runner.log,gs2_ideal_runner.log,gs1_ideal/ground_truth_amp.npy,gs1_ideal/ground_truth_phase.npy,gs1_ideal/ground_truth_amp.png,gs1_ideal/amplitude_diff.png,gs1_ideal/comparison_metrics.json,gs2_ideal/comparison_metrics.json,reassembly_cli.log,reassembly_gs1_ideal.json,reassembly_gs2_ideal.json,pytest_collect_cli_smoke.log,pytest_cli_smoke.log}

How-To Map
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md; export ARTIFACT_DIR=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T083000Z; mkdir -p "$ARTIFACT_DIR" "$ARTIFACT_DIR"/{gs1_ideal,gs2_ideal}` to stage the new hub.
2. In `bin/run_phase_c2_scenario.py` add utilities: `save_ground_truth(object_guess)` (write amp/phase NPY + PNG), `center_crop(array, size)` (symmetric crop with parity guard), and `write_diff_artifacts(pred, truth, prefix, out_dir)` that stores diff arrays/PNGs + MAE/RMSE/max/pearson stats.
3. After stitching (`amp, phase = ...`) but before PNG saves, crop both amplitude and phase to `params.object_size` and call the diff helper against the stored ground truth; persist metrics to `<scenario>/comparison_metrics.json` and include diff PNGs alongside amplitude/phase PNGs.
4. Update `run_metadata.json` to include `ground_truth_amp_path`, `ground_truth_phase_path`, diff metric JSON path, and pearson/max/MAE/RMSE scalars so downstream analysis can read them without re-loading arrays.
5. Rerun gs1_ideal with the baked profile: `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py --scenario gs1_ideal --snapshot plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json --output-dir "$ARTIFACT_DIR/gs1_ideal" --group-limit 64 |& tee "$ARTIFACT_DIR/gs1_ideal_runner.log"`.
6. Repeat for gs2_ideal (swap scenario/output dir) and confirm both runs emit ground-truth metrics + diff PNGs.
7. Refresh the padded-size telemetry: `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/reassembly_limits_report.py --scenario gs1_ideal --group-limit 64 --output-json "$ARTIFACT_DIR/reassembly_gs1_ideal.json" --output-markdown "$ARTIFACT_DIR/reassembly_gs1_ideal.md" |& tee "$ARTIFACT_DIR/reassembly_cli.log"` and append the gs2 call with `|& tee -a`.
8. Guard the CLI via testing: `pytest --collect-only tests/scripts/test_synthetic_helpers_cli_smoke.py -q | tee "$ARTIFACT_DIR/pytest_collect_cli_smoke.log"` then `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee "$ARTIFACT_DIR/pytest_cli_smoke.log"`.

Pitfalls To Avoid
- Keep cropping symmetric; if padded_size - object_size is odd, document the handling and don’t silently drop pixels on one side.
- Use JSON-safe scalars only (no numpy types) when writing diff metrics or run_metadata.
- Ensure diff PNGs use consistent color ranges so gs1 vs gs2 can be compared visually.
- Never mutate production modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`); instrumentation lives inside the plan-local runner/CLI.
- Do not reintroduce randomness—reuse the seeded snapshot and baked profile so outputs remain comparable to earlier hubs.
- Don’t skip the reassembly_limits rerun; we must prove padded-size math still reports `fits_canvas=true` after the runner changes.
- Respect PYTHON-ENV-001: invoke `python` from PATH, no conda/venv wrappers.
- Avoid accidental overwrites of prior artifacts; only write inside `2026-01-20T083000Z`.
- Validate that Pearson correlation is computed on flattened arrays with finite masks; guard against NaNs before writing metrics.
- Keep matplotlib usage headless (`Agg`) and close figures to avoid memory leaks during repeated runs.

If Blocked
- If cropping or metric computation fails (shape mismatch, NaNs), capture the exception, write a short note to `$ARTIFACT_DIR/blocked_<timestamp>.log`, and log the failure plus command in docs/fix_plan.md Attempts before marking the focus blocked.

Findings Applied (Mandatory)
- CONFIG-001 — `simulate_nongrid_raw_data` already calls `update_legacy_dict`; do not bypass that flow when adding new helpers.
- MODULE-SINGLETON-001 — continue to rely on `train_cdi_model_with_backend` factories so runner instrumentation doesn’t touch module singletons directly.
- NORMALIZATION-001 — comparison metrics must use the same normalized amplitude/phase outputs; never mix physics/display scales.
- BUG-TF-REASSEMBLE-001 — keep padded-size inputs integer when logging stats to avoid regressing the mixed-type crash.

Pointers
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:160 — Phase C3b checklist describing the new ground-truth comparison scope.
- docs/specs/spec-ptycho-workflow.md:46 — Reassembly requirements; ensure diff metrics respect the mandated canvas sizing.
- plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T071800Z/gs1_ideal_training_summary.md:1 — Latest telemetry showing gs1 history is finite but visually degraded; use it as the baseline for comparison notes.

Next Up (optional)
- If gs1 metrics remain much worse than gs2, scope Phase C4 fixes (e.g., adjust training dataset counts or investigate coordinate scaling).

Doc Sync Plan — N/A (existing selectors; no renames planned).
Mapped Tests Guardrail — Run `pytest --collect-only tests/scripts/test_synthetic_helpers_cli_smoke.py -q` before executing the selector to prove it still collects >0 tests.
Normative Math/Physics — Reference `docs/specs/spec-ptycho-workflow.md §Reassembly Requirements` for any reasoning on cropping or padded-size math; do not paraphrase formulas.
