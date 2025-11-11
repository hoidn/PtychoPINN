# Phase G Dense Delta Preview & Evidence Plan (2025-11-11T003351Z)

## Reality Check
- Code audit (`rg -n "metrics_delta_highlights_preview"`) shows **run_phase_g_dense.py never creates** `analysis/metrics_delta_highlights_preview.txt`, yet the hardened validator (`verify_dense_pipeline_artifacts.py::validate_metrics_delta_highlights`, lines 327-481) now **requires** the preview file for RED/GREEN coverage. Any dense run will therefore fail validation even if every other artifact is present.
- Highlight formatting still uses the legacy helper (`compute_delta -> f"{delta:+.3f}"`) for **all metrics** (plans/.../run_phase_g_dense.py:985-1035). The validator + findings (TEST-CLI-001, STUDY-001) expect **metric-specific precision** (MS-SSIM: ±0.000, MAE: ±0.000000); otherwise new preview/validator checks cannot pass once real artifacts exist.
- Hub `plans/active/.../reports/2025-11-11T001033Z/phase_g_dense_full_execution_real_run/` remains empty (no Phase D–G CLI/analysis outputs). Reusing it risks stale instructions; created **fresh hub 2025-11-11T003351Z** to stage the corrected run plus summary artifacts.

## Objectives for Ralph (single loop)
1. **TDD helper for delta previews**
   - Introduce a dedicated helper (e.g., `persist_delta_highlights()`) inside `plans/.../bin/run_phase_g_dense.py` that accepts the Phase G analysis root + aggregate metrics, formats deltas with MS-SSIM/MAE precision rules, writes both `metrics_delta_highlights.txt` and the missing `metrics_delta_highlights_preview.txt`, and returns structured metadata (`delta_summary` dict). Keep all path handling TYPE-PATH-001 compliant.
   - Add a focused pytest (`tests/study/test_phase_g_dense_orchestrator.py::test_persist_delta_highlights_creates_preview`) that feeds synthetic aggregate metrics (PtychoPINN/Baseline/PtyChi) through the helper and asserts:
     - Highlight text lines include amplitude+phase entries with the correct sign + precision (`+0.015`, `-0.000025`, etc.).
     - Preview file exists with 4 lines referencing the phase deltas and matching precision, so the validator will find the expected strings.
     - Returned `delta_summary['vs_Baseline']['mae']['phase']` equals the numeric difference (not formatted string).
2. **Wire helper into orchestrator**
   - Replace the inline delta/preview logic (run_phase_g_dense.py:985-1058) with the new helper so `main()` both prints highlights to stdout and writes preview/JSON artifacts using a single source of truth for formatting. Ensure `metrics_delta_summary.json` still carries provenance metadata and that the helper returns the ready-to-serialize `deltas` payload.
   - Update stdout banners to mention the new preview path so operators know where to look.
3. **Execute dense run with fresh hub**
   - Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T003351Z/phase_g_dense_full_execution_real_run` (directories already staged: analysis/ cli/ collect/ green/ red/ summary/ plan/).
   - Rerun the dense orchestrator with `python plans/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log` and wait for `[8/8]` + SUCCESS.
4. **Verification + evidence bundle**
   - Run `python plans/.../bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/pipeline_verification.json`.
   - Execute `python plans/.../bin/check_dense_highlights_match.py --hub "$HUB" |& tee "$HUB"/analysis/highlights_check.log` to confirm preview/highlights/JSON parity.
   - Capture pytest collect-only evidence (`pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_highlights.log`).
   - Summarize MS-SSIM/MAE deltas (phase emphasis), highlight guard signals, CLI validation result, and artifact_inventory counts in `$HUB/summary/summary.md`; update docs/fix_plan.md + Turn Summary accordingly.

## Required Tests / Evidence
- `pytest tests/study/test_phase_g_dense_orchestrator.py::test_persist_delta_highlights_creates_preview -vv`
- `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`
- `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv`
- `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv`
- Dense orchestrator run + verifier/checker logs archived under `$HUB`

## Findings to Reaffirm
- **POLICY-001** — PyTorch dependency remains mandatory (Phase G comparisons + Pty-chi LSQML still require torch>=2.2).
- **CONFIG-001** — Do not bypass `AUTHORITATIVE_CMDS_DOC`; orchestrator ensures config bridge before legacy consumers.
- **DATA-001** — Trust Phase C validators; no manual rewrites of NPZs during run.
- **TYPE-PATH-001** — Maintain POSIX-relative paths inside hub (artifact inventory + CLI validator rely on this).
- **STUDY-001** — Report MS-SSIM/MAE deltas (phase emphasis) with sign conventions in summary.
- **TEST-CLI-001** — Preview + highlights must contain spec-formatted deltas so validator + checker stay RED→GREEN.
