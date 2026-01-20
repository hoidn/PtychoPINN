Summary: Build a telemetry analyzer for the SIM-LINES gs1/gs2 runs so we can correlate amplitude bias vs intensity stats and capture the NaN onset before touching shared workflows.
Focus: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy (Phase C4 intensity stats)
Branch: paper
Mapped tests: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121500Z/

Do Now:
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py::main — add a plan-local CLI that ingests one or more scenario directories (each containing run_metadata.json, intensity_stats.json, inference_outputs/stats.json, comparison_metrics.json, train_outputs/history_summary.json) and emits aggregated `bias_summary.json`/`.md` capturing amplitude/phase bias deltas, intensity-scale comparisons, normalization stage stats, and training NaN indicators. Accept repeated `--scenario name=path` arguments plus `--output-dir`, validate inputs exist, and keep outputs device/dtype agnostic.
- Validate: 
  1. python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py --scenario gs1_ideal=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T113000Z/gs1_ideal --scenario gs2_ideal=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T113000Z/gs2_ideal --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121500Z > plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121500Z/analyze_intensity_bias.log
  2. pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121500Z/pytest_cli_smoke.log
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121500Z/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export REPORT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports && export OUT_HUB=$REPORT_ROOT/2026-01-20T121500Z && mkdir -p "$OUT_HUB"
3. python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py --scenario gs1_ideal=$REPORT_ROOT/2026-01-20T113000Z/gs1_ideal --scenario gs2_ideal=$REPORT_ROOT/2026-01-20T113000Z/gs2_ideal --output-dir "$OUT_HUB" > "$OUT_HUB/analyze_intensity_bias.log"
4. pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee "$OUT_HUB/pytest_cli_smoke.log"
5. Verify `$OUT_HUB/bias_summary.json` and `.md` exist and capture bias/intensity/NaN fields referenced in docs/fix_plan.md; mention any anomalies in the log.

Pitfalls To Avoid:
- Don’t read from stale hubs; use the explicit 2026-01-20T113000Z directories for scenario inputs and write only into $OUT_HUB.
- Keep analyzer outputs JSON/Markdown human-readable with scalar stats only—never dump full tensors.
- Maintain CONFIG-001 compliance when scripts touch params (loader helpers already do this; analyzer must not mutate params.cfg).
- Avoid hard-coding gridsize or backend assumptions; expect both gs1 (gridsize=1) and gs2 (gridsize=2) inputs.
- Gracefully flag missing files with actionable errors instead of stack traces.
- Do not alter production modules (ptycho/*); all code lives under plans/active/DEBUG-SIM-LINES-DOSE-001/bin/.
- Ensure matplotlib remains offscreen (Agg) if plots are added; this script should not emit plots by default.
- Preserve existing run_metadata fields—append new analyzer outputs rather than rewriting scenario directories.
- Keep CLI exit status non-zero on validation failures so supervisors can detect issues quickly.
- Archive every command’s stdout/stderr in the artifacts hub as shown in the How-To map.

If Blocked:
- Capture the failing command and traceback in $OUT_HUB/blocker.log, note the blocker in docs/fix_plan.md Attempts + summary.md, and stop after updating galph_memory so we can reassess next loop.

Findings Applied (Mandatory):
- CONFIG-001 — analyzer must not bypass the existing Training/InferenceConfig bridge; use saved run metadata and params snapshots without mutating params.cfg.
- NORMALIZATION-001 — keep physics/statistical/display scaling contexts separate when reporting intensity stats; label each stage explicitly.
- POLICY-001 — keep workflows TensorFlow-first; PyTorch isn’t involved here but any helper must continue to assume torch>=2.2 availability when routed via backend_selector.

Pointers:
- specs/spec-ptycho-core.md §Normalization Invariants — authoritative math for intensity_scale semantics.
- docs/fix_plan.md: DEBUG-SIM-LINES-DOSE-001 entry (latest Attempts) — defines Phase C4 scope + artifacts.
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md §Phase C4 — checklist + deliverables.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py:360-460 — reference format for telemetry payloads your analyzer must read.
- plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T113000Z/* — concrete gs1/gs2 inputs for the analyzer.

Next Up (optional):
1. Compare analyzer output against expected spec math (intensity_scale, amplitude bias) to determine whether we need to adjust loss weights or preprocessing.
2. If analyzer shows loss-weight imbalance, craft a minimal config patch (e.g., realspace_mae_weight) and plan regression evidence.

Doc Sync Plan (Conditional): Not needed (no new pytest selectors).
Mapped Tests Guardrail: The CLI smoke selector above already collects (>0); keep it as the validation guard.
Normative Math/Physics: Cite specs/spec-ptycho-core.md §Normalization + docs/DEVELOPER_GUIDE.md §3.5 in analyzer docs/comments when referencing intensity math; do not restate derivations inline.
