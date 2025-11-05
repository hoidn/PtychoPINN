# Phase G Dense Real-Run Evidence — Supervisor Plan (2025-11-05T115706Z)

## Current Status Snapshot
- Previous relaunch hub `2025-11-05T111247Z` is staged but still empty: no CLI logs under `cli/` and `analysis/` contains no metrics bundle, so the dense Phase C→G pipeline has not yet produced evidence.
- No new MS-SSIM/MAE deltas are recorded in docs/fix_plan.md or summary.md; last successful metrics came from earlier collect-only dry runs.
- Orchestrator regression selector `test_run_phase_g_dense_exec_runs_analyze_digest` last passed on 2025-11-09 but needs to be re-run before any long execution, per TDD guardrails.
- Objective for this loop: execute the dense Phase C→G pipeline end-to-end in the fresh `2025-11-05T115706Z` hub, confirm artifact completeness, and record MS-SSIM/MAE deltas plus provenance in summary/docs.

## Scope for Ralph
1. **Workspace sanity + guard exports**
   - `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` (must precede any orchestrator commands to satisfy CONFIG-001 policy).
   - Ensure no stale orchestrator processes: run `pgrep -fl run_phase_g_dense.py` and `pgrep -fl studies.fly64_dose_overlap`; terminate any PIDs before continuing.
2. **Pre-flight regression**
   - Set `HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run`.
   - Execute `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv` and tee output into `$HUB/green/pytest_orchestrator_dense_exec_recheck.log`.
   - If the selector fails, capture the RED log under `$HUB/red/` and fix before proceeding.
3. **Dense pipeline execution**
   - Launch from repo root: `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log`.
   - Expect multi-hour runtime covering Phase C→G (dataset regeneration, baseline + PtychoPINN inference, digest/report helpers). Keep the session attached so logs stream into the hub.
4. **Artifact verification + delta checks**
   - Confirm the following files exist under `$HUB/analysis/`:
     - `metrics_summary.json`
     - `metrics_delta_summary.json`
     - `metrics_delta_highlights.txt`
     - `metrics_delta_highlights_preview.txt`
     - `metrics_digest.md`
     - `aggregate_report.md`
     - `aggregate_highlights.txt`
   - Generate `$HUB/analysis/artifact_inventory.txt` via `find "$HUB" -maxdepth 3 -type f | sort`.
   - Verify highlights vs JSON deltas: `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$PWD/$HUB" | tee "$HUB"/analysis/highlights_consistency_check.log`.
5. **Documentation updates**
   - Populate `$HUB/summary/summary.md` with: guard test result, pipeline runtime, CONFIG-001/DATA-001/TYPE-PATH-001 adherence notes, MS-SSIM & MAE deltas (PtychoPINN vs Baseline & PtyChi), and artifact links.
   - Update `docs/fix_plan.md` Attempts History with this execution attempt, citing the new hub path and recorded metrics.
   - If failures occur, capture logs under `$HUB/cli/` or `$HUB/red/`, document blockers in summary.md + docs/fix_plan.md, and mark the ledger status appropriately.

## Required Tests
- `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`

## Artifacts to Produce
- `$HUB/green/pytest_orchestrator_dense_exec_recheck.log`
- `$HUB/cli/run_phase_g_dense.log`
- `$HUB/analysis/{metrics_summary.json,metrics_delta_summary.json,metrics_delta_highlights.txt,metrics_delta_highlights_preview.txt,metrics_digest.md,aggregate_report.md,aggregate_highlights.txt,artifact_inventory.txt,highlights_consistency_check.log}`
- `$HUB/summary/summary.md` (runtime, provenance, MS-SSIM/MAE deltas, artifact links)

## Exit Criteria
- Dense pipeline completes successfully (exit code 0) with all expected analysis artifacts present and verified.
- Highlights text/preview values match `metrics_delta_summary.json` for both Baseline and PtyChi comparisons.
- Summary.md + docs/fix_plan.md capture runtime, provenance confirmations, MS-SSIM/MAE deltas, and artifact references.
- Any failures are documented with logs and mitigation steps; no silent gaps remain in Attempts History.
