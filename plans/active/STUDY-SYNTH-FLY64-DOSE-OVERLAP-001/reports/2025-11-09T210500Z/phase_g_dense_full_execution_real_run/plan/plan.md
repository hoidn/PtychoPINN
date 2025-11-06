# Phase G Dense Pipeline Completion Plan (2025-11-09T210500Z)

## Current Status Snapshot
- Prior hub `2025-11-09T170500Z` captured a partial Phase C→G run: Phase C NPZ outputs exist, but there are no Phase D–G CLI logs and the `analysis/` directory only contains the placeholder `artifact_inventory_partial.txt`; no metrics/summary artifacts were produced.
- No live orchestrator or study processes detected (`pgrep -fl run_phase_g_dense.py` and `pgrep -fl studies.fly64_dose_overlap` return empty), so the long pipeline is idle.
- The orchestrator regression selector (`test_run_phase_g_dense_exec_runs_analyze_digest`) last ran green in the 170500Z hub; rerunning it after the upcoming code change will confirm guards stay intact.
- Goal for this loop: (1) add automated artifact inventory generation to the orchestrator script under TDD, (2) execute the dense Phase C→G pipeline end-to-end in the new hub (`2025-11-09T210500Z`), (3) verify the metrics bundle with the enhanced verifier, and (4) document MS-SSIM/MAE deltas plus provenance in summary/docs.

## Scope for Ralph
1. **Test-first guard for artifact inventory**
   - Update `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest` (or add a focused helper test) so it asserts the orchestrator writes `analysis/artifact_inventory.txt` containing relative paths for key artifacts (at minimum the reporting helper outputs and metrics bundle placeholders).
   - Run the selector (`pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`) to capture the RED failure before implementation; store log under `$HUB/red/pytest_orchestrator_dense_exec_inventory_fail.log`.
2. **Implement automated artifact inventory**
   - Extend `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main` to create an `analysis/artifact_inventory.txt` file at the end of successful execution. Inventory should list files relative to the hub root, covering `analysis/`, `cli/`, and `summary/` outputs; preserve deterministic ordering (e.g., sorted POSIX paths) to keep tests stable.
   - Ensure the new helper respects TYPE-PATH-001 (Path normalization) and writes UTF-8 text.
3. **GREEN the updated regression selector**
   - Re-run `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv` after implementation; archive log at `$HUB/green/pytest_orchestrator_dense_exec_inventory_fix.log`.
4. **Pre-flight sanity + bridge check**
   - Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and confirm no stray orchestrator processes (`pgrep -fl run_phase_g_dense.py` empty).
   - Set `HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T210500Z/phase_g_dense_full_execution_real_run` from repo root.
5. **Dense Phase C→G pipeline execution**
   - Run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log`.
   - Expected duration 2–4 hours; ensure `[1/8]` through `[8/8]` complete and Phase D–G logs appear under `$HUB/cli/`.
6. **Artifact verification**
   - Execute `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$PWD/$HUB" --report "$HUB"/analysis/pipeline_verification.json`; the report must show all checks valid.
   - Spot-check `analysis/artifact_inventory.txt` for the metrics bundle entries (`metrics_summary.json`, `metrics_delta_summary.json`, `metrics_delta_highlights.txt`, `metrics_digest.md`, `aggregate_report.md`, `aggregate_highlights.txt`).
7. **Documentation + ledger updates**
   - Update `$HUB/summary/summary.md` with pipeline runtime, MS-SSIM/MAE delta values (PtychoPINN vs Baseline/PtyChi), metadata compliance status, verifier results, and log references.
   - Append attempts/observations plus artifact path to `docs/fix_plan.md` and note lessons (if any) in `docs/findings.md`.

## Required Tests
- `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`

## Artifacts to Produce
- `$HUB/red/pytest_orchestrator_dense_exec_inventory_fail.log`
- `$HUB/green/pytest_orchestrator_dense_exec_inventory_fix.log`
- `$HUB/cli/run_phase_g_dense.log` plus per-phase CLI logs (`phase_c_generation.log`, `phase_d_dense.log`, `phase_e_train.log`, `phase_f_reconstruct.log`, `phase_g_compare.log`, etc.)
- `$HUB/analysis/{artifact_inventory.txt,metrics_summary.json,metrics_delta_summary.json,metrics_delta_highlights.txt,metrics_digest.md,aggregate_report.md,aggregate_highlights.txt,pipeline_verification.json}`
- `$HUB/summary/summary.md` with MS-SSIM/MAE delta narrative

## Exit Criteria
- Updated regression selector passes and confirms artifact inventory generation.
- Dense Phase C→G pipeline run completes successfully with full metrics bundle present in `$HUB/analysis/` and listed in `artifact_inventory.txt`.
- Verifier report indicates all checks valid; highlights text matches delta JSON values.
- Summary and ledger reflect real metrics, provenance (CONFIG-001 bridge, DATA-001 compliance), and artifact references.
- Any failures are logged, summarized, and blockers recorded in docs/fix_plan.md.
