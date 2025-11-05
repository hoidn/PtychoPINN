### Turn Summary
Implemented test_analyze_dense_metrics_flags_failures regression test validating analyze_dense_metrics.py exits with code 1 and emits failure banner when n_failed > 0; test passed immediately (GREEN).
Test validates existing behavior (lines 254-277 in analyze_dense_metrics.py) to prevent future regression; no code changes needed.
Launched dense Phase C→G pipeline in background (shell 21953b); Phase C generation progressing through dose=1000/10000/100000 datasets.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T010500Z/phase_g_dense_full_execution_real_run/ (green/pytest logs, pipeline running in background shell 21953b)

---

# Phase G Dense Metrics Rerun + Pipeline Launch (2025-11-09T010500Z)

## Objective
Execute the full Phase C→G pipeline for dose=1000, view=dense with fresh artifacts after metadata coverage landed, then generate the metrics digest inside the same hub. Capture RED→GREEN pytest evidence for the new analyze script regression test and archive long-running CLI logs.

## Scope
- Add a regression test ensuring `analyze_dense_metrics.py` returns exit code 1 when `n_failed > 0` and emits the failure banner.
- Run targeted pytest selectors for the reporting/analyze helpers and highlight preview guard.
- Execute `run_phase_g_dense.py --clobber` for dose 1000 dense train/test and persist CLI logs.
- Run `analyze_dense_metrics.py` once metrics_summary.json + aggregate_highlights.txt exist to produce `analysis/metrics_digest.md`.
- Update `summary.md` + `docs/fix_plan.md` with MS-SSIM/MAE deltas and digest path.

## Deliverables
1. `tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_flags_failures` (new test).
2. RED→GREEN pytest logs under `reports/2025-11-09T010500Z/.../{red,green}/`.
3. Fresh Phase C→G CLI transcripts under `.../cli/` and metrics artifacts under `.../analysis/`.
4. `analysis/metrics_digest.md` plus digest CLI log.
5. Turn Summary appended to `summary/summary.md`.
6. docs/fix_plan.md Attempts History update referencing this rerun.

## Acceptance Criteria
- New pytest test fails if `analyze_dense_metrics.py` stops flagging `n_failed > 0`. Selector added to registry.
- Phase G pipeline exits 0 with `metrics_summary.json`, `aggregate_report.md`, `aggregate_highlights.txt`, and updated `comparison_manifest.json` in hub.
- `analyze_dense_metrics.py` exits 0 when `n_failed == 0` (real run) and produces digest referencing Phase G outputs.
- Summary/docs capture MS-SSIM + MAE amplitude/phase deltas for PtychoPINN vs Baseline and PtyChi.
- Artifacts recorded in `docs/fix_plan.md` + `galph_memory.md` with timestamped path.
