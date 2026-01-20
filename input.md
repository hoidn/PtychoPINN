Summary: Capture dataset-derived vs fallback intensity scales so we can confirm whether a gain mismatch explains the ≈6.7× amplitude bias before changing normalization code.
Focus: DEBUG-SIM-LINES-DOSE-001 — Phase D4 architecture/loss diagnostics (dataset intensity-scale telemetry)
Branch: paper
Mapped tests: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T231745Z/

Do Now (hard validity contract):
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py::{run_inference_and_reassemble,write_intensity_stats_outputs} — compute the dataset-derived intensity scale (`s = sqrt(nphotons / E_batch[Σ_xy |Ψ|²])` from `specs/spec-ptycho-core.md §Normalization Invariants`), retain the existing closed-form fallback, and persist both values plus their deltas/ratios inside `intensity_stats.json`, `run_metadata.json`, and the Markdown summary for each scenario.
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py::{load_intensity_stats,render_markdown} — parse the new dataset/fallback fields, expose them in the JSON payload, and render a “Intensity Scale Comparison” table ahead of the stage-ratio section so reviewers can see whether sim_lines is stuck on the fallback gain (988.21).
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py::main — rerun `gs2_ideal` (stable profile, 5 epochs) and `gs2_ideal` with `--nepochs 60` under `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T231745Z/{gs2_ideal,gs2_ideal_nepochs60}/`, regenerate `bias_summary.{json,md}` with the updated analyzer, and archive the CLI smoke pytest log.
- Validate: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T231745Z/

How-To Map:
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T231745Z`.
2. Update `run_phase_c2_scenario.py`:
   - In `run_inference_and_reassemble`, compute `dataset_intensity_scale = sqrt(params.nphotons / np.mean(np.sum(test_raw.diff3d**2, axis=(1,2))))`, stash it in `intensity_info`, and pass it into `write_intensity_stats_outputs`.
   - Extend `write_intensity_stats_outputs` so `intensity_stats.json`/Markdown (and `run_metadata.json`) include dataset scale, fallback scale, bundle scale, deltas, and ratios.
3. Update `bin/analyze_intensity_bias.py` to read the new fields and render a dataset-vs-recorded table for each scenario in both JSON and Markdown before the stage-ratio narrative.
4. Re-run the scenarios:
   - Baseline: `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py --scenario gs2_ideal --group-limit 64 --prediction-scale-source least_squares --output-dir "$HUB"/gs2_ideal | tee "$HUB"/gs2_ideal_runner.log`
   - 60-epoch: `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py --scenario gs2_ideal --nepochs 60 --group-limit 64 --prediction-scale-source least_squares --output-dir "$HUB"/gs2_ideal_nepochs60 | tee "$HUB"/gs2_ideal_nepochs60_runner.log`
5. Regenerate the analyzer bundle with the updated script: `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py --scenario gs2_base="$HUB"/gs2_ideal --scenario gs2_ne60="$HUB"/gs2_ideal_nepochs60 --output-dir "$HUB" | tee "$HUB"/analyze_dataset_scale.log`.
6. Guard selector: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee "$HUB"/pytest_cli_smoke.log`.

Pitfalls To Avoid:
- Stay plan-local — do not touch `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` (CLAUDE.md §2.6).
- Keep CONFIG-001 bridging in place; dataset telemetry must not reorder `update_legacy_dict`.
- Serialize new fields with plain Python scalars (`_serialize_scalar`) so `json.dump` works.
- Cite the normative spec sections verbatim in Markdown rather than paraphrasing math.
- Ensure analyzer changes degrade gracefully when older hubs lack the new fields.
- Training must remain on GPU; stop if CUDA is unavailable rather than falling back to CPU (policy §12).
- Write artifacts exclusively under the new hub; do not overwrite prior evidence.
- Capture stdout/stderr with `tee` for every CLI/pytest invocation per TEST-CLI-001.
- Reuse the baked gs2 profiles so diffs reflect instrumentation changes only.
- Do not alter prediction-scale behavior beyond telemetry — we are measuring gain mismatches, not patching them yet.

If Blocked:
- If either scenario rerun OOMs or crashes, stop immediately, dump the command + stack trace into `$HUB/blocker.md`, and record the blocker (with log path) in docs/fix_plan.md Attempts History before retrying smaller workloads or switching focus.

Findings Applied (Mandatory):
- CONFIG-001 — Always sync `params.cfg` before legacy modules touch grouped data (docs/debugging/QUICK_REFERENCE_PARAMS.md).
- SIM-LINES-CONFIG-001 — Maintain the plan-local CONFIG-001 bridge so NaN fixes remain active.
- NORMALIZATION-001 — Treat dataset vs fallback scales as physics normalization only; cite `specs/spec-ptycho-core.md §Normalization Invariants`.
- H-NEPOCHS-001 — Training-length hypothesis already rejected; telemetry must isolate gain mismatches instead of tweaking epochs/batch size.
- TEST-CLI-001 — Archive CLI + pytest logs beside scenario outputs so reviewers can audit the guard selector.

Pointers:
- docs/fix_plan.md:180-320 — Phase D attempts history + new dataset-scale entry.
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:320 — Phase D checklist (see D4a instrumentation task).
- plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md:1-15 — Latest supervisor note describing this increment.
- specs/spec-ptycho-core.md:70-110 — Normative intensity-scale math to cite in Markdown.
- specs/spec-ptycho-workflow.md:1-80 — Loss/IntensityScaler architecture references for analyzer commentary.

Next Up (optional):
1. If dataset vs fallback scales match, pivot to gs1_ideal instrumentation to see whether gridsize=1 diverges post-normalization.
2. If a mismatch appears, prep follow-up probing of `normalize_data` vs `IntensityScaler_inv` wiring in the training graph before proposing fixes.
