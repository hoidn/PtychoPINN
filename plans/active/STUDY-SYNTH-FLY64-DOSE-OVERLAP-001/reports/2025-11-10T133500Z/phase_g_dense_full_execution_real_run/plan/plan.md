# Phase G Dense Pipeline Evidence Plan (2025-11-10T133500Z)

## Current Status
- CLI log validation helper `validate_cli_logs()` now fails when the orchestrator log is missing or lacks `[1/8]…[8/8]` banners and the `"SUCCESS: All phases completed"` sentinel (cf. 2025-11-10T113500Z hub).
- No dense pipeline run has been executed with the enhanced verifier; `cli/` under the 113500Z hub is empty, `analysis/` has no metrics bundle, and docs/fix_plan.md still lists the initiative as `ready_for_implementation`.
- Prior Phase C runs (e.g., 2025-11-10T093500Z) produced canonicalized NPZs, but Phases D→G artifacts remain absent; MS-SSIM/MAE deltas versus Baseline/PtyChi have never been captured.

## Risks & Gaps
1. **Per-phase log coverage blind spot** — `validate_cli_logs()` records which `phase_*.log` files exist but does not fail if a per-phase log is missing or truncated. A partial run could pass verification so long as the orchestrator log is intact.
2. **Evidence debt** — We still need a clean dense Phase C→G execution with metrics delta JSON/TXT populated, verifier evidence GREEN, and summary/docs updated.
3. **Ledger staleness** — Attempts history lacks an entry for the post-CLI-validation run; findings ledger has no note covering per-phase log enforcement or runtime metrics parity.

## Objectives for Ralph (single loop)
1. **TDD guard: per-phase CLI log enforcement**
   - Extend `tests/study/test_phase_g_dense_artifacts_verifier.py` with fixtures covering:
     - Missing Phase E log (`phase_e_dense_gs2_dose1000.log`) → expect verifier failure referencing the absent file.
     - Complete per-phase log bundle (Phase D/E/F/G logs populated with minimal “Completed” sentinel lines) → expect pass.
   - Capture RED evidence in `$HUB/red/pytest_cli_phase_logs_fail.log`, then GREEN in `$HUB/green/pytest_cli_phase_logs_fix.log`.
2. **Implementation: tighten `validate_cli_logs()`**
   - Update `plans/active/.../bin/verify_dense_pipeline_artifacts.py::validate_cli_logs` to:
     - Require the presence of all expected `phase_*.log` files for Phases D, E (baseline + dense gs2), F (per split), G (per split), plus reporting helpers (`aggregate_report_cli.log`, `metrics_digest_cli.log`).
     - Parse each log and ensure at least one completion sentinel appears (e.g., `"Completed Phase"` or `"SUCCESS"` lines emitted by `run_phase_g_dense.py::run_command`); aggregate missing files/sentinels into actionable error messages.
     - Return structured details (`missing_logs`, `incomplete_logs`) that tests can assert on.
   - Wire failures into the verifier’s JSON report so `valid` is false when any per-phase log is missing/incomplete.
3. **Execute dense Phase C→G pipeline & verifier**
   - Export environment guard (`AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`) and run:
     ```bash
     export HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T133500Z/phase_g_dense_full_execution_real_run
     python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py \
       --hub "$PWD/$HUB" \
       --dose 1000 \
       --view dense \
       --splits train test \
       --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
     ```
   - Once `[8/8]` appears, execute the verifier with the tightened CLI checks:
     ```bash
     python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py \
       --hub "$PWD/$HUB" \
       --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log
     ```
   - Preserve `analysis/artifact_inventory.txt`, `metrics_delta_summary.json`, `metrics_delta_highlights.txt`, `metrics_digest.md`, and the CLI logs.
4. **Document metrics & ledger updates**
   - Summarize MS-SSIM/MAE deltas vs Baseline/PtyChi in `$HUB/summary/summary.md`, referencing `metrics_delta_summary.json` and `metrics_summary.json`.
   - Update `docs/fix_plan.md` with this attempt, including verifier status and artifact paths.
   - If per-phase log enforcement exposes new lessons, append an entry to `docs/findings.md` (linking to HUB evidence).

## Required Tests / Evidence
- `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_missing -vv`
- `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv`
- `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_missing -vv`
- `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_complete -vv`
- `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`

## Artifacts to Capture (`$HUB`)
- `red/pytest_cli_phase_logs_fail.log`
- `green/pytest_cli_phase_logs_fix.log`
- `green/pytest_cli_logs_fix.log` (existing orchestrator guard) & `green/pytest_orchestrator_dense_exec_cli_guard.log`
- `cli/*.log` from pipeline run, including orchestrator and per-phase logs
- `analysis/pipeline_verification.json`, `analysis/verifier_cli.log`, `analysis/artifact_inventory.txt`
- `analysis/metrics_delta_summary.json`, `analysis/metrics_delta_highlights.txt`, `analysis/metrics_summary.json`, `analysis/metrics_digest.md`
- `summary/summary.md` documenting metrics, verifier status, and findings references

## Findings to Reaffirm
- **POLICY-001** — Ensure PyTorch remains installed for verifier/test imports.
- **CONFIG-001** — Maintain CONFIG-001 bridge before legacy components during pipeline execution.
- **DATA-001** — Confirm Phase C NPZs retain amplitude/complex64 compliance (verifier logs).
- **TYPE-PATH-001** — Keep hub-relative POSIX paths in artifact inventory and CLI logs.
- **OVERSAMPLING-001** — Dense overlap remains 0.7; cite in summary if metrics deviate.
- **STUDY-001** — Record MS-SSIM/MAE deltas vs Baseline/PtyChi.
- **PHASEC-METADATA-001** — Highlight metadata compliance results captured in metrics summary.
