Summary: Capture IntensityScaler + training-container telemetry so Phase D4 can prove whether the architecture is double-scaling normalized inputs before we touch production physics.
Focus: DEBUG-SIM-LINES-DOSE-001 — Phase D4 architecture/loss diagnostics
Branch: paper
Mapped tests: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T173500Z/

Do Now (hard validity contract):
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py::{run_scenario,write_intensity_stats_outputs} — add helpers that snapshot the trained IntensityScaler state (params.cfg value, trainable flag, exp(log_scale)) plus the training-container X stats into `run_metadata.json` and `intensity_stats.{json,md}`, then re-run the gs2 baseline (5-epoch) and gs2_ideal_nepochs60 scenarios under the new hub so both runs emit the enriched telemetry alongside their existing inference artifacts.
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py::render_markdown — parse the new scaler/train-container fields, fold them into the JSON payload, and surface an “IntensityScaler state” section in the Markdown before regenerating the gs2_base vs gs2_ne60 comparison for the fresh evidence hub.
- Validate: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T173500Z/

How-To Map:
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T173500Z`.
2. After coding, run the baseline scenario: `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py --scenario gs2_ideal --output-dir "$HUB"/gs2_ideal --group-limit 64 --prediction-scale-source least_squares | tee "$HUB"/gs2_ideal_runner.log`.
3. Run the 60-epoch variant: `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py --scenario gs2_ideal --nepochs 60 --output-dir "$HUB"/gs2_ideal_nepochs60 --group-limit 64 --prediction-scale-source least_squares | tee "$HUB"/gs2_ideal_nepochs60_runner.log`.
4. Regenerate the analyzer report with the new metadata: `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py --scenario gs2_base="$HUB"/gs2_ideal --scenario gs2_ne60="$HUB"/gs2_ideal_nepochs60 --output-dir "$HUB" | tee "$HUB"/analyze_intensity_scaler.log`.
5. Guard selector: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee "$HUB"/pytest_cli_smoke.log`.

Pitfalls To Avoid:
- Stay plan-local: do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` (Directive #6).
- Keep CONFIG-001 bridging intact; never move the `update_legacy_dict` calls or reuse stale params when rerunning the scenarios.
- Avoid double-writing artifacts — write only under the new hub and leave prior evidence untouched for diffability.
- Training must stay on GPU; if CUDA is unavailable, stop and report instead of attempting CPU runs.
- Ensure new JSON fields are serializable scalars (use `_serialize_scalar`) to prevent NumPy types from leaking into metadata.
- Reference the exact spec clauses (`specs/spec-ptycho-core.md §Normalization Invariants`, `specs/spec-ptycho-workflow.md §Loss and Optimization`) in Markdown so reviewers can trace requirements.
- Rerun commands with deterministic seeds (reuse the snapshot and baked profiles) so diffs isolate the instrumentation, not random noise.
- Don’t forget to include the new scaler metadata inside both `run_metadata.json` and `intensity_stats.json`; analyzer relies on both for cross-checks.
- Keep analyzer extensions backwards compatible — existing scenario bundles without the new fields should still render without exploding.
- Capture stdout/stderr via `tee` so we have logs ready for the artifacts folder per TEST-CLI-001.

If Blocked:
- If either scenario rerun OOMs or TensorFlow crashes, stop immediately, copy the failing command + stack trace into `$HUB/blocker.md`, and note the blocker (with log paths) in docs/fix_plan.md Attempts History so we can decide whether to shrink workloads before retrying.

Findings Applied (Mandatory):
- CONFIG-001 — Bridge params.cfg before every training/inference call to keep legacy modules in sync.
- SIM-LINES-CONFIG-001 — The sim_lines runner must continue calling `update_legacy_dict` inside the helper so NaN fixes remain active.
- NORMALIZATION-001 — Treat normalization telemetry as physics-normalization only; do not mix statistical/display scaling while summarizing stage ratios.
- TEST-CLI-001 — Archive the CLI/pytest logs under the hub and cite the selector so reviewers can audit the guard evidence.

Pointers:
- docs/fix_plan.md:1-200 — Active focus description + Phase D Attempts History for context.
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:320 — Phase D checklist (D4 scope + exit criteria).
- plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T162500Z/summary.md:1 — Latest D4 findings (loss composition + next steps).
- specs/spec-ptycho-core.md:86 — Normalization invariant contract (defines how stage ratios should behave).
- specs/spec-ptycho-workflow.md:1-80 — Loss & intensity-scaler architecture references that the new Markdown section must cite.

Next Up (optional):
1. If scaler telemetry still shows symmetry, pivot to gs1_ideal to see whether gridsize=1 diverges before prediction.
2. If a mismatch appears between bundle vs layer scale, prep a follow-up loop to bisect `_get_log_scale()` usage inside `ptycho/train_pinn.prepare_inputs`.
