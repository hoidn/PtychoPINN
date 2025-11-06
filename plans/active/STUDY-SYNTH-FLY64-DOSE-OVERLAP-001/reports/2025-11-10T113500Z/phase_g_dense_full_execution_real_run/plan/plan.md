# Phase G Dense Pipeline Evidence Plan (2025-11-10T113500Z)

## Current Status Snapshot
- Existing hub `2025-11-10T093500Z` contains RED/GREEN pytest logs and the ongoing dense pipeline run; `run_phase_g_dense.py` reached `[1/8] Phase C` and the generation process is still active (`pgrep -af studies.fly64_dose_overlap.generation` shows PID 2675688 under `/home/ollie/Documents/PtychoPINN2`).
- Phase C outputs are streaming into `/home/ollie/Documents/PtychoPINN2/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T093500Z/phase_g_dense_full_execution_real_run/data/phase_c`, but no Phase D–G artifacts or analysis bundle exists yet.
- The verifier now enforces artifact inventory coverage, but it does **not** confirm the CLI logs captured every phase’s completion banners; regressions could slip through if a phase aborts silently after emitting partial outputs.

## Gaps / Risks
1. **CLI log blind spot** — `verify_dense_pipeline_artifacts.py` ignores the `cli/` logs, so the verifier cannot detect truncated runs (violates STUDY-001 evidence expectations).
2. **Pipeline completion risk** — Phase C is still running; if it fails overnight we need actionable blocker notes, not silent retries.
3. **Documentation lag** — Ledger/summary still reflect “pipeline running”; we must capture the completion verdict, verifier results, and MS-SSIM/MAE deltas once available.

## Objectives for Ralph (single loop)
1. **TDD guard for CLI log validation**
   - Extend `tests/study/test_phase_g_dense_artifacts_verifier.py` with fixtures covering:
     - Missing `cli/run_phase_g_dense.log` → expect the verifier to fail with a descriptive error.
     - Complete CLI bundle (synthetic logs containing `[run_phase_g_dense] SUCCESS` and per-phase sentinels) → expect the verifier to pass.
   - Capture RED log at `$HUB/red/pytest_cli_logs_fail.log`, then GREEN at `$HUB/green/pytest_cli_logs_fix.log`.
2. **Implement CLI log validation in verifier**
   - Add `validate_cli_logs()` (TYPE-PATH-001 compliant) to `plans/active/.../bin/verify_dense_pipeline_artifacts.py` ensuring:
     - `cli/run_phase_g_dense.log` exists and contains `SUCCESS: All phases completed` plus expected `[1/8]`…`[8/8]` banners.
     - Each per-phase log (`cli/phase_c_generation.log`, `cli/phase_d_<view>.log`, `cli/phase_f_<view>_<split>.log`, etc.) exists and contains a completion sentinel (e.g., “Completed Phase …” or final timestamp).
     - Errors aggregate into the JSON report with actionable messages and offending filenames.
   - Integrate the helper into `main()` so the verifier fails when any CLI log is missing or incomplete.
3. **Finish the dense pipeline run and verification**
   - Monitor `pgrep -af run_phase_g_dense.py` / `studies.fly64_dose_overlap.generation`. Allow the in-flight 093500Z run to finish to avoid GPU contention; capture its exit status in the previous hub notes.
   - After the existing run reaches `[8/8]`, launch a fresh end-to-end execution into this loop’s hub (clobber enabled so we capture artifacts under 113500Z):
     ```bash
     export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
     export HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T113500Z/phase_g_dense_full_execution_real_run
     python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py \
       --hub "$PWD/$HUB" \
       --dose 1000 \
       --view dense \
       --splits train test \
       --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
     ```
   - Once `[8/8]` appears in the new hub’s CLI log, run the updated verifier:
     ```bash
     python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py \
       --hub "$PWD/$HUB" \
       --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log
     ```
4. **Document outcomes**
   - Record MS-SSIM/MAE deltas and metadata compliance in `summary/summary.md` (pull numbers from `metrics_delta_summary.json` and `metrics_summary.json`).
   - Update `docs/fix_plan.md` Attempts History (timestamped entry for 2025-11-10T113500Z) and log CLI validation lesson(s) in `docs/findings.md` if new failure modes appear.

## Required Tests (mapped selectors)
- `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_missing -vv`
- `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv`
- `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`

## Artifacts to Capture (under `$HUB`)
- `red/pytest_cli_logs_fail.log`
- `green/pytest_cli_logs_fix.log`
- `green/pytest_orchestrator_dense_exec_cli_guard.log`
- `cli/run_phase_g_dense.log` and per-phase logs copied from the active hub once run completes
- `analysis/pipeline_verification.json`, `analysis/verifier_cli.log`, `analysis/artifact_inventory.txt`
- `analysis/metrics_delta_summary.json`, `analysis/metrics_delta_highlights.txt`, `analysis/metrics_summary.json`
- `summary/summary.md` with updated deltas + verification notes

## Findings to Reaffirm
- **POLICY-001** — PyTorch dependency remains installed for verifier/tests.
- **CONFIG-001** — Ensure `update_legacy_dict` executes via orchestrator CLI before Phase C loads legacy modules.
- **DATA-001** — Phase C outputs must keep amplitude/complex64 guarantees; reference during verifier run.
- **TYPE-PATH-001** — New CLI validation must normalize to POSIX paths, no absolute leakage.
- **OVERSAMPLING-001** — Dense overlap settings (0.7) remain unchanged; confirm in logs.
- **STUDY-001** — MS-SSIM/MAE deltas captured and reported.
- **PHASEC-METADATA-001** — Validate metadata compliance passes in `metrics_summary.json` and surface in summary.
