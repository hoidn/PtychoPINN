Summary: Compute prediction↔truth scaling diagnostics so we can prove whether a single scalar explains the ≈12× amplitude gap before touching loader/workflow code.
Focus: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy (Phase C4d scaling diagnostics)
Branch: paper
Mapped tests: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T143000Z/

Do Now:
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py::gather_scenario_data — load `inference_outputs/amplitude.npy` and `ground_truth_amp.npy` for each scenario, compute best-fit prediction↔truth scalars (mean, median, p05/p95, least-squares), derive the rescaled MAE/RMSE after applying the best scalar, and persist these metrics under `derived['scaling_analysis']` plus a new Markdown section.
- Validate:
  1. python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py --scenario gs1_ideal=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T113000Z/gs1_ideal --scenario gs2_ideal=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T113000Z/gs2_ideal --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T143000Z > plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T143000Z/analyze_intensity_bias.log
  2. pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T143000Z/pytest_cli_smoke.log
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T143000Z/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export REPORT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports && export OUT_HUB=$REPORT_ROOT/2026-01-20T143000Z && mkdir -p "$OUT_HUB"
3. Modify `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py` to:
   - Extend `ScenarioInput` so it resolves `inference_outputs/amplitude.npy` and `ground_truth_amp.npy` for each scenario.
   - Load those arrays inside `gather_scenario_data`, compute truth/pred ratios (mean, median, p05, p95), the least-squares scalar `c = Σ(pred*truth)/Σ(pred²)`, and rescaled MAE/RMSE (`mae_scaled`, `rmse_scaled`) after multiplying predictions by `c`.
   - Attach these values under `derived['scaling_analysis']` and render them in Markdown (new “Prediction ↔ Truth Scaling” table with scalar + ratio stats + rescaled error callouts).
   - Ensure JSON serialization uses floats (no numpy types) and guard against division by zero (skip ratios where prediction mean or pixel values are zero).
4. python "$REPORT_ROOT/bin/analyze_intensity_bias.py" --scenario gs1_ideal="$REPORT_ROOT/2026-01-20T113000Z/gs1_ideal" --scenario gs2_ideal="$REPORT_ROOT/2026-01-20T113000Z/gs2_ideal" --output-dir "$OUT_HUB" > "$OUT_HUB/analyze_intensity_bias.log"
5. Inspect `$OUT_HUB/bias_summary.json` + `.md` to verify the new scaling section reports a consistent scalar for gs1_ideal and highlights gs2_ideal’s NaN collapse (expect ratios to report NaN/— when predictions are zero).
6. pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee "$OUT_HUB/pytest_cli_smoke.log"

Pitfalls To Avoid:
- Do not mutate existing JSON keys; append data under `derived['scaling_analysis']` so older consumers keep working.
- Guard against zero/NaN predictions: skip ratios or emit `null` instead of dividing by zero.
- Keep the analyzer plan-local; do not import heavy ML libraries or touch `ptycho/*`.
- Ensure Markdown stays compact (tables, bullet callouts) to keep diffs reviewable.
- Apply CONFIG-001 hygiene: analyzer can read params snapshots but must never mutate `params.cfg`.
- Use float conversions (`float(...)`) and `.tolist()` when serializing numpy scalars/arrays.
- Device-neutral: no TensorFlow/PyTorch imports.
- Preserve log hygiene by teeing CLI + pytest output into the artifacts hub.
- Respect the existing ratio “largest drop” logic; do not regress previous derived stats.
- Keep `AUTHORITATIVE_CMDS_DOC` pointing to docs/TESTING_GUIDE.md for every command invocation.

If Blocked:
- Capture the failing command + traceback in $OUT_HUB/blocker.log, note the blocker in docs/fix_plan.md Attempts + summary.md, and stop after updating galph_memory so we can reassess before touching production modules.

Findings Applied (Mandatory):
- POLICY-001 (docs/findings.md:12) — plan-local CLIs must remain portable and compatible with PyTorch-enabled environments.
- CONFIG-001 (docs/findings.md:14) — analyzer may read params snapshots but must not mutate params.cfg or rely on stale gridsize state.
- NORMALIZATION-001 (docs/findings.md:22) — derived scaling metrics must respect the three-stage normalization separation (physics/statistics/display) and avoid mixing units.

Pointers:
- docs/fix_plan.md:119 — latest Attempts entry describing Phase C4d scope and artifacts hub.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py:1 — analyzer CLI you’ll extend with scaling diagnostics.
- plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md:1 — supervisor notes for the new scaling-analysis plan and artifact path.
- specs/spec-ptycho-core.md:86 — Normative Normalization Invariants (intensity scaling symmetry) referenced when interpreting ratios.

Next Up (optional):
1. If a constant scalar aligns gs1_ideal, scope a quick experiment applying that scalar inside the analyzer to quantify residual MAE before touching workflow code.
2. If gs2_ideal still collapses after scaling, plan a runner probe that captures intermediate prediction tensors to trace the NaN path.

Doc Sync Plan (Conditional): Not needed (no test additions or selector changes).
Mapped Tests Guardrail: The CLI smoke selector above already collects; rerun it after analyzer changes to keep the initiative guard green.
Normative Math/Physics: Reference specs/spec-ptycho-core.md §Normalization Invariants when describing ratios or scalars; do not paraphrase equations into pseudo-code.
