# Phase G Dense Evidence Capture Plan (2025-11-10T173500Z)

## Current Status
- CLI log validator now matches dose/view-specific filenames and enforces completion sentinels (commit 27398b6d). RED/GREEN evidence stored in 2025-11-10T153500Z hub.
- Dense Phase C→G pipeline has **not** been executed since the validator updates; hubs 133500Z/153500Z contain no `analysis/` or `cli/` outputs.
- Highlights consistency guard exists (`bin/check_dense_highlights_match.py`) but is not yet integrated with the primary verifier, so highlight/preview drift could slip through automated checks.
- `docs/fix_plan.md` still lists the filename-pattern guard as planning-only; Attempts History needs a 2025-11-10T153500Z+exec record capturing the shipped validator work.

## Risks & Gaps
1. **Highlight drift:** `metrics_delta_highlights.txt` and `_preview.txt` can diverge from `metrics_delta_summary.json` without failing the verifier.
2. **Evidence debt:** Dense pipeline artifacts (metrics deltas, highlights, inventories) are still missing under a post-guard hub.
3. **Ledger lag:** Fix plan lacks the implementation attempt data for the latest guard update, obscuring progress.

## Objectives for Ralph (single loop)
1. **TDD guard upgrade**
   - Add RED fixtures to `tests/study/test_phase_g_dense_artifacts_verifier.py` that exercise highlight/preview mismatches (missing model, wrong sign/precision) and expect the verifier to report failures via `validate_metrics_delta_highlights`.
   - Extend the GREEN fixture to cover the new success path once implementation lands.
   - Update `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py::validate_metrics_delta_highlights` so it:
     - Loads `analysis/metrics_delta_summary.json` and checks each expected model/metric pair.
     - Verifies both `metrics_delta_highlights.txt` **and** `metrics_delta_highlights_preview.txt` include the correctly formatted deltas (`±0.000` for MS-SSIM, `±0.000000` for MAE).
     - Surfaces structured failure details (`missing_models`, `missing_fields`, `missing_preview_values`) for pytest assertions.
2. **Execute dense pipeline under new hub**
   - Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and set `HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T173500Z/phase_g_dense_full_execution_real_run`.
   - Ensure no lingering orchestrators (`pgrep -af run_phase_g_dense.py || true`), then run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log` and wait for `[8/8]` + SUCCESS banner.
   - After completion, run the upgraded verifier and the highlights matcher:
     - `python plans/.../bin/verify_dense_pipeline_artifacts.py --hub "$PWD/$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log`
     - `python plans/.../bin/check_dense_highlights_match.py --hub "$PWD/$HUB" |& tee "$HUB"/analysis/highlights_check.log`
3. **Evidence & documentation**
   - Archive pytest RED/GREEN logs under `$HUB/red` and `$HUB/green` (include new mismatch fixtures and updated GREEN run).
   - Confirm `analysis/` contains: comparison_manifest.json, metrics_summary.{json,md}, metrics_delta_{summary.json,highlights.txt,highlights_preview.txt}, metrics_digest.md, artifact_inventory.txt, pipeline_verification.json, highlights_check.log.
   - Update `$HUB/summary/summary.md` with MS-SSIM/MAE deltas vs Baseline & PtyChi, metadata compliance results, and guard outcomes, referencing STUDY-001 and PHASEC-METADATA-001.
   - Refresh `docs/fix_plan.md` Attempts History (add 2025-11-10T153500Z+exec entry for the validator shipped this loop and log the new 2025-11-10T173500Z attempt) plus any durable findings.

## Required Tests / Evidence
- `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_missing_model -vv`
- `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_mismatched_value -vv`
- `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv`
- `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv` (regression)
- `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`
- `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv` (doc sync after GREEN)

## Findings to Reaffirm
- **POLICY-001** — torch>=2.2 required for verifier imports.
- **CONFIG-001** — Use official CLIs so legacy bridge executes before data/model loading.
- **DATA-001** — Ensure Phase C outputs remain amplitude/complex64 via verifier checks.
- **TYPE-PATH-001** — Keep inventory entries POSIX-relative inside the hub.
- **OVERSAMPLING-001** — Dense overlap (0.7) invariant; note deviations in summary if metrics drift.
- **STUDY-001** — Report MS-SSIM/MAE deltas vs Baseline & PtyChi.
- **PHASEC-METADATA-001** — Surface metadata compliance status in summary & verifier output.
- **TEST-CLI-001** — Maintain explicit RED/GREEN fixtures for CLI/highlight validation.
