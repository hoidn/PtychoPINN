# Phase G Dense Full Execution — Real Run Plan (2025-11-08T030500Z)

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Focus:** Phase G comparison & analysis (dense evidence)  
**Action Type:** Planning (supervisor loop)  
**Target State:** ready_for_implementation

---

## Context

- Aggregate metric support shipped in 2025-11-08T010500Z+exec; tests cover JSON/Markdown structure.
- Dense Phase C→G orchestrator has not been executed end-to-end since aggregations landed; no real metrics exist for dose=1000, view=dense under the new logic.
- Turn summaries still require manual aggregation; no helper exists to distill `metrics_summary.json` into deltas against baselines.

## Objectives

1. Run the dense Phase C→G pipeline (dose=1000, view=dense, splits=train/test) with `--clobber` on a fresh hub, capturing CLI logs and regenerated artifacts.
2. Design and ship a reusable reporting helper that parses `metrics_summary.json` and emits concise delta tables for Turn Summary authoring.
3. Validate the regenerated outputs (Phase C metadata guard + metric summarizer) and archive logs under this hub.
4. Sync documentation/ledger entries once evidence is in place.

## Deliverables

- Hub `plans/active/.../reports/2025-11-08T030500Z/phase_g_dense_full_execution_real_run/` populated with Phase C→G outputs, CLI transcripts, guard logs, and summary artifacts.
- Script `plans/active/.../bin/report_phase_g_dense_metrics.py` (T2) that reads `metrics_summary.json` and writes a Markdown/console digest of aggregate deltas (PtychoPINN vs Baseline vs Ptychi) with explicit amplitude/phase comparisons.
- Pytest coverage that exercises the reporting helper with fixture data (ensuring deterministic ordering and formatted output).
- Updated `summary/summary.md` describing dense run metrics, plus ledger/doc updates referencing this hub.

## Task Breakdown

1. **TDD — Reporting Helper**
   - Author `tests/study/test_phase_g_dense_metrics_report.py::test_report_phase_g_dense_metrics` capturing a fixture `metrics_summary.json` (PtychoPINN, Baseline, PtyChi) and asserting the helper emits:
     - Tabulated aggregate metrics per model.
     - Delta section comparing PtychoPINN against Baseline/PtyChi for MS-SSIM amplitude/phase and MAE amplitude/phase.
     - Deterministic ordering and 3-decimal formatting.
   - Mark expected failure (helper not yet implemented).

2. **Implementation — Reporting Helper**
   - Implement `plans/active/.../bin/report_phase_g_dense_metrics.py::main`, accepting `--metrics` (path to metrics_summary.json) and `--output` (optional Markdown path).
   - Compute aggregate tables + deltas (PtychoPINN minus Baseline/PtyChi) and emit both stdout and Markdown (if requested) with 3-decimal floats.
   - Return non-zero if required models/metrics missing.

3. **Dense Pipeline Execution**
   - Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`.
   - Run `python plans/.../bin/run_phase_g_dense.py --hub <hub> --dose 1000 --view dense --splits train test --clobber | tee .../cli/phase_g_dense_pipeline.log`.
   - On failure, capture blocker log and stop.

4. **Post-run Validation & Reporting**
   - Reuse helper shim to run `validate_phase_c_metadata` and `summarize_phase_g_outputs`; tee to `analysis/validate_and_summarize.log`.
   - Execute `report_phase_g_dense_metrics.py --metrics analysis/metrics_summary.json --output analysis/aggregate_report.md` and archive stdout.

5. **Documentation & Ledger Sync**
   - Update `summary/summary.md` with Turn Summary + key metrics.
   - Refresh `docs/TESTING_GUIDE.md`, `docs/development/TEST_SUITE_INDEX.md`, and `docs/fix_plan.md` with selector inventory and aggregate highlights.

## Findings & Guardrails Reinforced

- POLICY-001 — Dense pipeline invokes PyTorch baselines; ensure torch dependency present.
- CONFIG-001 — Orchestrator must keep `update_legacy_dict` ordering intact; reporting helper treats outputs read-only.
- DATA-001 — Guard validates regenerated NPZ metadata; reporting helper consumes CSV-derived aggregates only.
- TYPE-PATH-001 — Normalize all paths and keep reports under initiative hub.
- OVERSAMPLING-001 — Dense overlap parameters remain fixed; report should flag if aggregates contradict expectations.

## Pitfalls

- CLI may take >20 minutes; abort after first failure and log context.
- Reporting helper must degrade gracefully if optional models absent (exit with message, not silent failure).
- Keep Markdown output idempotent (overwrite previous report deterministically).
- Do not reuse 2025-11-08T010500Z hub; always operate in this new timestamped directory.
