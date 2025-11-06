# Phase G Dense Pipeline Sentinel Guard Plan (2025-11-10T153500Z)

## Current Status
- Per-phase CLI log presence checks landed in commits e45821c3/db06f775; `validate_cli_logs()` now asserts that the orchestrator log exists, emits `[1/8]…[8/8]`, includes the `"SUCCESS: All phases completed"` sentinel, and that eight `phase_*.log` files exist.
- The guard still assumes generic filenames (`phase_e_baseline.log`, `phase_e_dense.log`, …) and does **not** match the actual log naming produced by `run_phase_g_dense.py` (`phase_e_baseline_gs1_dose1000.log`, `phase_e_dense_gs2_dose1000.log`, etc.), so a real pipeline run will fail verification even when all logs are present.
- No dense Phase C→G execution has been captured under the new guard: `reports/2025-11-10T133500Z/phase_g_dense_full_execution_real_run/{cli,analysis}` remain empty and `docs/fix_plan.md` still lists the initiative as `ready_for_implementation`.

## Risks & Gaps
1. **Filename pattern mismatch** — The verifier’s hard-coded names diverge from the orchestrator’s dose/view-specific filenames. Without pattern-aware matching the guard produces false negatives, blocking evidence collection.
2. **Sentinel blind spot** — Individual per-phase logs are not inspected for completion markers; truncated logs would still pass.
3. **Evidence debt** — We still lack a full dense pipeline run producing `metrics_delta_summary.json`, highlights, artifact inventory, and verifier JSON under a controlled hub.
4. **Ledger/document lag** — `docs/fix_plan.md` and `docs/findings.md` do not yet record the filename-pattern issue or the eventual dense-run evidence.

## Objectives for Ralph (single loop)
1. **TDD guard: enforce real log patterns and sentinels**
   - Add RED fixtures to `tests/study/test_phase_g_dense_artifacts_verifier.py`:
     - `test_verify_dense_pipeline_cli_phase_logs_wrong_pattern` — create logs using the actual filenames _minus_ the dose/view suffix (current verifier expectation) and assert the check fails, capturing `$HUB/red/pytest_cli_phase_logs_pattern_fail.log`.
     - `test_verify_dense_pipeline_cli_phase_logs_incomplete` — create dose/view-specific filenames but omit completion sentinels (e.g., `"Completed Phase"`/`"SUCCESS"` variants) so the guard fails, capturing `$HUB/red/pytest_cli_phase_logs_incomplete.log`.
   - Extend the existing GREEN fixture to generate realistic log names (`phase_e_baseline_gs1_dose1000.log`, `phase_e_dense_gs2_dose1000.log`, `phase_f_dense_train.log`, etc.) with completion markers, then assert `missing_phase_logs` is empty and `incomplete_phase_logs` is empty after implementation; capture GREEN evidence in `$HUB/green/pytest_cli_phase_logs_fix.log`.
2. **Implementation: pattern-aware + sentinel validation**
   - Update `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py::validate_cli_logs` to:
     - Accept pattern lists for required logs: e.g., `phase_e_baseline_*.log`, `phase_e_{view}_*.log`, `phase_f_*_train.log`, `phase_f_*_test.log`, `phase_g_*_train.log`, `phase_g_*_test.log`, `aggregate_report_cli.log`, `metrics_digest_cli.log`.
     - Record both `missing_phase_logs` (no match) and `incomplete_phase_logs` (missing sentinel strings). Treat `"Completed Phase"`, `"Training finished"`, or `"SUCCESS"` as acceptable sentinels; document the allowed set in the docstring.
     - Surface structured error messages listing which patterns are missing vs which files lacked sentinels, and expose counts in the result dict so tests can assert on them.
3. **Execution: dense Phase C→G run under new guard**
   - Export guards (`AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`, `HUB=$PWD/plans/active/.../2025-11-10T153500Z/...`) and ensure no stale orchestrators are running (`pgrep -af run_phase_g_dense.py || true`).
   - Run `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log` and wait for `[8/8]` + `"SUCCESS"` banners.
   - Execute the verifier with the tightened guard (`python plans/active/.../bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log`) and confirm the CLI check passes with the pattern-aware logic.
4. **Evidence & documentation**
   - Preserve `analysis/metrics_delta_summary.json`, `analysis/metrics_delta_highlights.txt`, `analysis/metrics_summary.json`, `analysis/artifact_inventory.txt`, and verifier JSON/logs.
   - Summarize MS-SSIM/MAE deltas vs Baseline & PtyChi plus metadata compliance in `$HUB/summary/summary.md`, referencing STUDY-001 and PHASEC-METADATA-001.
   - Update `docs/fix_plan.md` Attempts History with the dense-run outcome and note the resolved filename-pattern issue; add a `docs/findings.md` entry if the pattern mismatch constitutes a durable lesson.

## Required Tests / Evidence
- `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_wrong_pattern -vv`
- `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_incomplete -vv`
- `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_complete -vv`
- `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv`
- `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`
- After implementation, `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_cli_logs.log` (Doc Sync guard once selectors succeed).

## Artifacts to Capture (`$HUB`)
- `red/pytest_cli_phase_logs_pattern_fail.log`
- `red/pytest_cli_phase_logs_incomplete.log`
- `green/pytest_cli_phase_logs_fix.log`
- `green/pytest_cli_logs_fix.log` (existing orchestrator guard)
- `green/pytest_orchestrator_dense_exec_cli_guard.log`
- `cli/run_phase_g_dense.log`, `cli/phase_*.log`, `cli/aggregate_report_cli.log`, `cli/metrics_digest_cli.log`
- `analysis/pipeline_verification.json`, `analysis/verifier_cli.log`, `analysis/artifact_inventory.txt`
- `analysis/metrics_delta_summary.json`, `analysis/metrics_delta_highlights.txt`, `analysis/metrics_summary.json`, `analysis/metrics_digest.md`
- `summary/summary.md` documenting metrics, verifier status, and findings references

## Findings to Reaffirm
- **POLICY-001** — PyTorch dependency required for verifier imports (`torch>=2.2`).
- **CONFIG-001** — Ensure CLI entry points run with the legacy bridge before data/model construction.
- **DATA-001** — Confirm Phase C NPZs maintain amplitude/complex64 contract via verifier output.
- **TYPE-PATH-001** — Keep hub-relative POSIX paths in logs and inventories.
- **OVERSAMPLING-001** — Dense overlap (0.7) remains invariant; note in summary if metrics drift.
- **STUDY-001** — Record MS-SSIM/MAE deltas vs Baseline & PtyChi.
- **PHASEC-METADATA-001** — Surface metadata compliance block in metrics summary.
- **TEST-CLI-001** — Maintain explicit RED/GREEN fixtures for CLI validation guards with realistic log content.
