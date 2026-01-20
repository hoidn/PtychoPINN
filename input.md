Summary: Add stage-by-stage intensity ratio diagnostics to the bias analyzer so we can pinpoint where the ≈2.5 amplitude drop occurs before touching shared workflows.
Focus: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy (Phase C4 ratio diagnostics)
Branch: paper
Mapped tests: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T132500Z/

Do Now:
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py::gather_scenario_data — extend the analyzer to capture per-stage amplitude means/ratios (raw_diffraction → grouped_diffraction → grouped_X_full → container_X → reconstructed amplitude) plus prediction-vs-truth ratios, emit them in the JSON payload, and surface the highlights in the Markdown summary.
- Validate:
  1. python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py --scenario gs1_ideal=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T113000Z/gs1_ideal --scenario gs2_ideal=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T113000Z/gs2_ideal --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T132500Z > plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T132500Z/analyze_intensity_bias.log
  2. pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T132500Z/pytest_cli_smoke.log
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T132500Z/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export REPORT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports && export OUT_HUB=$REPORT_ROOT/2026-01-20T132500Z && mkdir -p "$OUT_HUB"
3. Update analyze_intensity_bias.py to:
   - Pull per-stage means from intensity_stats (raw_diffraction, grouped_diffraction, grouped_X_full, container_X) and snapshot them in a new `derived['stage_means']` structure along with ratios (raw→grouped, grouped→normalized, normalized→prediction, prediction→truth).
   - Record which stage introduces the steepest drop (max absolute ratio delta) and include that indicator in both JSON and Markdown (e.g., a new bullet under each scenario).
   - Use existing `pred_stats`/`truth_stats` fields to compute prediction/truth means so no new data files are required.
4. python "$REPORT_ROOT/bin/analyze_intensity_bias.py" --scenario ... (as above) --output-dir "$OUT_HUB" > "$OUT_HUB/analyze_intensity_bias.log" and confirm the JSON contains `derived` entries with the new ratios plus a Markdown table describing the stage means.
5. pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee "$OUT_HUB/pytest_cli_smoke.log"
6. Verify `$OUT_HUB/bias_summary.json` + `.md` have the derived ratio data and note any anomalies (e.g., stage where ratio≈0.34) in galph notes/input.

Pitfalls To Avoid:
- Do not assume every scenario has all four stages; handle missing names gracefully and label the ratios as `null` instead of crashing.
- Keep the analyzer pure-Python with no heavy imports; the script is plan-local and must stay portable.
- Leave existing telemetry untouched; only append new JSON fields/Markdown sections so downstream consumers remain compatible.
- Use safe float math (check division by zero when computing ratios) and clamp outputs to scalars for readability.
- Device/dtype neutral: never hard-code TensorFlow/PyTorch assumptions in the analyzer.
- Avoid editing production modules (ptycho/*); all work stays under plans/active/DEBUG-SIM-LINES-DOSE-001/bin/.
- Preserve the Markdown formatting so existing reviewers can diff the new sections easily (tables + bullet list, no giant dumps).
- Keep command logs (`analyze_intensity_bias.log`, `pytest_cli_smoke.log`) in the artifacts hub for traceability.
- Respect CONFIG-001 bridging rules; analyzer must not mutate params.cfg even when reading run_metadata.
- When flagging the worst stage drop, tie-break deterministically (e.g., earliest stage) to simplify comparisons.

If Blocked:
- Capture the failing command + traceback in $OUT_HUB/blocker.log, note the blocker in docs/fix_plan.md Attempts + summary.md, and stop after updating galph_memory so we can reassess before touching production modules.

Findings Applied (Mandatory):
- POLICY-001 (docs/findings.md:12) — keep scripts PyTorch-ready even if we don’t touch torch this loop.
- CONFIG-001 (docs/findings.md:14) — analyzer must read existing params snapshots but never mutate params.cfg.
- NORMALIZATION-001 (docs/findings.md:22) — maintain clear separation between physics/stats/display scaling when reporting stage ratios.

Pointers:
- docs/fix_plan.md:119 — latest Attempts entry describing the ratio-diagnostics increment and artifacts hub.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py:148 — existing scenario aggregation hook you’ll extend.
- plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md:4 — supervisor notes summarizing this loop’s focus and artifact path.
- specs/spec-ptycho-core.md:86 — Normative Normalization Invariants to cite when interpreting ratio results.

Next Up (optional):
1. If the derived ratios show the drop occurs after normalization, plan a runner probe that dumps model outputs before/after IntensityScaler.
2. If the drop is already visible by grouped_X_full, revisit Phase B loaders to confirm scaling inputs (A2 backlog) before changing loss weights.

Doc Sync Plan (Conditional): Not needed (no test additions or selector changes).
Mapped Tests Guardrail: The CLI smoke selector above already collects; rerun it after analyzer changes to keep the initiative guard green.
Normative Math/Physics: Reference specs/spec-ptycho-core.md §Normalization for any equations mentioned in comments/logs; avoid inventing new math in the analyzer.
