# Phase G Dense Highlights Metadata & Evidence Plan (2025-11-11T001033Z)

## Reality Check
- `plans/.../bin/verify_dense_pipeline_artifacts.py::validate_metrics_delta_highlights` already loads `analysis/metrics_delta_summary.json` and enforces preview parity (see lines 309-460), so the 193500Z gap statement about "4 lines only" is stale. Tests covering missing preview/preview mismatch/delta mismatch exist in `tests/study/test_phase_g_dense_artifacts_verifier.py` (lines 1738-2184), but they only assert on the error string and never inspect the structured metadata (`checked_models`, `missing_preview_values`, etc.).
- No artifact hub after 2025-11-10T193500Z contains Phase D–G CLI logs or metrics: `ls plans/.../reports/2025-11-10T193500Z/phase_g_dense_full_execution_real_run/{analysis,cli}` shows nothing. We still need a real `run_phase_g_dense.py --clobber` execution that captures `[1/8]→[8/8]`, produces metrics/highlights files, and feeds the verifier/checker.
- Early-return paths in `validate_metrics_delta_highlights` (e.g., missing preview file) do not populate metadata fields, so pytest cannot assert on `missing_preview_values` when a file is absent. We need consistent result shapes for regression coverage.

## Objectives for Ralph (single loop)
1. **TDD: Strengthen metadata assertions**
   - Update the RED fixtures in `tests/study/test_phase_g_dense_artifacts_verifier.py` for `test_verify_dense_pipeline_highlights_missing_preview`, `..._preview_mismatch`, and `..._delta_mismatch` to assert on the structured metadata (e.g., `missing_preview_values`, `mismatched_highlight_values`, `checked_models`). Failures should capture the exact formatted delta (`+0.015`, `-0.000025`, etc.).
   - Extend the GREEN fixture `test_verify_dense_pipeline_highlights_complete` to assert that the validation result includes `checked_models == ['vs_Baseline','vs_PtyChi']` and empty `missing_*` lists so regressions surface immediately.
   - Keep log capture under `$HUB/red/` before code edits and `$HUB/green/` after implementation.
2. **Implementation: Normalize highlight metadata output**
   - Patch `plans/.../bin/verify_dense_pipeline_artifacts.py::validate_metrics_delta_highlights` so it always attaches `checked_models`, `missing_models`, `missing_metrics`, `missing_preview_values`, and `mismatched_highlight_values`, even when bailing out early (missing preview/JSON). Introduce a helper to initialize these keys upfront so pytest assertions never see missing fields.
   - While touching the helper, document the precision rules (MS-SSIM → 3 decimals, MAE → 6) inline to guide future contributors.
3. **Execute dense pipeline with the new hub**
   - Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T001033Z/phase_g_dense_full_execution_real_run`.
   - Ensure the hub directories `{analysis,cli,collect,green,red,summary}` exist (already staged) and rerun `python plans/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log`, waiting for `[8/8]` + SUCCESS.
4. **Verification + doc sync**
   - Run `python plans/.../bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/pipeline_verification.json` and `python plans/.../bin/check_dense_highlights_match.py --hub "$HUB"`, capturing logs under `analysis/`.
   - Collect pytest evidence (`pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_highlights.log`).
   - Summarize MS-SSIM/MAE deltas, highlight guard status, CLI validation outcome, and artifact inventory line counts in `$HUB/summary/summary.md`; update `docs/fix_plan.md` Attempts History and log durable lessons (if any) in `docs/findings.md` (likely tie-ins to TEST-CLI-001 / STUDY-001).

## Required Tests / Evidence
- RED → GREEN: `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_missing_preview -vv`
- RED → GREEN: `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_mismatch -vv`
- RED → GREEN: `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_delta_mismatch -vv`
- GREEN: `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv`
- Regression: `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv`
- Regression: `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`
- Doc-sync: `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv`

## Findings to Reaffirm
- **POLICY-001** PyTorch dependency remains mandatory even though this loop is TF-heavy.
- **CONFIG-001** Run the orchestrator CLI so `update_legacy_dict` executes inside `run_phase_g_dense.py` before any legacy consumers.
- **DATA-001** Trust only validator-approved NPZ layouts; do not short-circuit Phase C metadata guard.
- **TYPE-PATH-001** Keep all hub paths POSIX-relative (artifact inventory + CLI validation rely on this).
- **OVERSAMPLING-001** Dense view overlap (0.7) must remain unchanged; surface any deviations in summary.md.
- **STUDY-001** Always report MS-SSIM/MAE deltas versus both Baseline and PtyChi with sign conventions.
- **PHASEC-METADATA-001** Carry Phase C metadata compliance results from `summarize_phase_g_outputs()` into the final summary.
- **TEST-CLI-001** Maintain explicit RED/GREEN fixtures for CLI/log validation; per-phase log filenames must include dose/view suffixes.
