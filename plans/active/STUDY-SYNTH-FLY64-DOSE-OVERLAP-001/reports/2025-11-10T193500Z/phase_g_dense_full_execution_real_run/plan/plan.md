# Phase G Dense Highlights & Pipeline Evidence Plan (2025-11-10T193500Z)

## Reality Check
- `verify_dense_pipeline_artifacts.py::validate_metrics_delta_highlights` only checks for four lines + prefixes; it never inspects `metrics_delta_summary.json` or the preview text file, so highlight drift can slip by even when the JSON deltas disagree.
- The helper script `check_dense_highlights_match.py` still expects a legacy JSON layout (`baseline/ptychi` keys) that does not match the current `run_phase_g_dense.py` output (`deltas → vs_Baseline/vs_PtyChi → ms_ssim/mae → amplitude/phase`).
- No dense hub after 2025-11-10T173500Z contains Phase D–G artifacts; `analysis/` directories are empty and the orchestrator logs stop after Phase C.

## Objectives for Ralph (single loop)
1. **TDD: Harden highlight validation**
   - Extend `tests/study/test_phase_g_dense_artifacts_verifier.py` with RED fixtures that cover:
     - Missing preview file (`metrics_delta_highlights_preview.txt`).
     - Preview values that do not match the JSON deltas.
     - Highlight TXT values that do not match the JSON deltas (while preview is correct).
   - Update the existing GREEN fixture to assert that the validation result includes `checked_models`, `missing_models`, `missing_fields`, and `missing_preview_values` metadata.
   - Drive the new tests RED→GREEN by enhancing `validate_metrics_delta_highlights` to:
     - Load `analysis/metrics_delta_summary.json` (current schema with `deltas → vs_Baseline/vs_PtyChi → ms_ssim/mae → amplitude/phase`).
     - Require both `metrics_delta_highlights.txt` **and** `metrics_delta_highlights_preview.txt` to exist.
     - Compute the expected formatted deltas (`±0.000` for MS-SSIM, `±0.000000` for MAE) and ensure they appear in both highlight files.
     - Populate structured failure details (`missing_models`, `missing_metrics`, `missing_preview_values`, `mismatched_highlight_values`).
   - Ensure `verify_dense_pipeline_artifacts.py::main` surfaces these failures in its JSON report for pytest assertions.
2. **Align the standalone checker**
   - Update `plans/active/.../bin/check_dense_highlights_match.py` to reuse the enhanced validation logic (or re-implement it with the same JSON schema & formatting rules) so the CLI guard matches the new expectations.
3. **Execute dense pipeline with evidence capture**
   - Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T193500Z/phase_g_dense_full_execution_real_run`.
   - Rerun `python plans/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log` and wait for `[8/8]` + SUCCESS.
   - Invoke the enhanced verifier and the updated highlights checker, teeing logs into `$HUB/analysis/`.
   - Capture pytest RED/Green logs under `$HUB/red/` and `$HUB/green/`, and record the pytest `--collect-only` output under `$HUB/collect/` once GREEN.
4. **Documentation & ledger**
   - Summarize MS-SSIM/MAE deltas, highlight guard status, and metadata compliance in `$HUB/summary/summary.md` (reference STUDY-001, PHASEC-METADATA-001, TEST-CLI-001).
   - Update `docs/fix_plan.md` Attempts History with this loop’s execution results and add any durable lessons to `docs/findings.md` if new guard rules emerge.

## Required Tests / Evidence
- RED: `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_missing_preview -vv`
- RED: `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_mismatch -vv`
- RED: `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_delta_mismatch -vv`
- GREEN: `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv`
- Regression: `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv`
- Regression: `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`
- Doc-sync: `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv`

## Findings to Reaffirm
- **POLICY-001** torch>=2.2 enforced for verifier imports.
- **CONFIG-001** Run official CLIs so legacy bridge ordering remains intact.
- **DATA-001** Verify Phase C NPZ contract via orchestrator validators before trusting deltas.
- **TYPE-PATH-001** Keep artifact_inventory entries POSIX-relative and expose JSON provenance via relative paths.
- **OVERSAMPLING-001** Dense view overlap must remain at 0.7; report deviations in summary if metrics drift.
- **STUDY-001** Summaries must report MS-SSIM/MAE deltas against Baseline and PtyChi.
- **PHASEC-METADATA-001** Carry Phase C metadata compliance status forward to dense summaries.
- **TEST-CLI-001** Maintain explicit RED/GREEN fixtures for CLI + highlight validation rules.
