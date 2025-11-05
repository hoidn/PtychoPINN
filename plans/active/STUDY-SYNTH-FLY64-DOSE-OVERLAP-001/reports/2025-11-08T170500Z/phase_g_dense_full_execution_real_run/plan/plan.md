# Phase G Dense Execution Evidence Plan (2025-11-08T170500Z)

## Context
- Phase C validator hotfix (2025-11-08T150500Z+exec) restored compatibility with the refactored in-memory DATA-001 contract.
- Dense Phase C→G pipeline run (hub 130500Z) aborted during Stage 5 validation due to the signature drift; the hotfix landed with new regression coverage and GREEN tests.
- All orchestrator enhancements (prepare_hub, metadata guard, reporting helper, highlights preview) are GREEN with coverage and doc sync complete.
- We now need a clean, end-to-end dense run to capture real MS-SSIM/MAE evidence and archive highlights/aggregate reports under a fresh hub.

## Findings
- POLICY-001 — PyTorch dependency enforced downstream (Phase F/G comparison utilities rely on torch).
- CONFIG-001 — AUTHORITATIVE_CMDS_DOC must be exported before invoking orchestrator/tests.
- DATA-001 — Phase C validator ensures canonical NPZ keys/dtypes; new signature regression test must remain GREEN.
- TYPE-PATH-001 — Orchestrator/reporting helpers expect Path objects for IO; avoid raw string math in any follow-up tooling.
- OVERSAMPLING-001 — Neighbor count K must remain ≥ gridsize² when analyzing metadata.

## Actions for Ralph
1. **Regression sanity (fast):** Re-run the highlights preview regression to ensure stdout preview and reporting helper wiring stay GREEN after the validator patch.
2. **Pipeline execution:** Launch `bin/run_phase_g_dense.py --clobber` for dose 1000, view dense, splits train/test with AUTHORITATIVE_CMDS_DOC exported. Capture stdout via tee into `cli/run_phase_g_dense_cli.log`; retain phase-specific logs under `cli/`.
3. **Post-run analysis script (new T2 tooling):** Author `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py` to parse `{hub}/phase_g_dense_full_execution_real_run/analysis/metrics_summary.json` and `{hub}/.../aggregate_highlights.txt`, emitting a concise Markdown digest (`analysis/metrics_digest.md`) summarizing MS-SSIM (phase/amplitude) and MAE deltas (PtychoPINN vs Baseline / PtyChi). Script should support CLI args for `--metrics-json`, `--highlights`, and `--output` and print the digest to stdout as well.
4. **Evidence capture:** Run the new analysis script to generate `analysis/metrics_digest.md`, then update `summary/summary.md` with the Turn Summary block plus bullet list of key deltas. Archive pytest log, pipeline logs, aggregate report, highlights text, metrics JSON, and digest paths.

## Exit Criteria
- `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv` GREEN with log saved.
- `bin/run_phase_g_dense.py --clobber` command exits 0; CLI and phase logs archived under the 170500Z hub.
- `analysis/metrics_digest.md` produced by `analyze_dense_metrics.py` matches highlights content (deltas for MS-SSIM phase/amplitude and MAE) and is referenced in summary.
- `summary/summary.md` updated with MS-SSIM/MAE results, command outcomes, and artifact links; docs/fix_plan.md attempts ledger records success or blockers.

## References
- Prior hub (validator hotfix): `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T150500Z/phase_g_dense_full_execution_real_run/`
- CLI instructions: `docs/TESTING_GUIDE.md` §2 (AUTHORITATIVE_CMDS_DOC guard)
- Metrics helper: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py`
- Regression tests: `tests/study/test_phase_g_dense_orchestrator.py`
