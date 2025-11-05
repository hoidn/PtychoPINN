# Phase G Dense Real-Run Evidence — Supervisor Plan (2025-11-09T190500Z)

## Current Status Snapshot
- Previous hub `2025-11-09T170500Z` captured the latest pipeline launch. Phase C dataset generation completed (train/test splits present under `data/phase_c/`), but the orchestrator did not advance to Phase D or later — no `phase_d_*.log`, no metrics artifacts, and `analysis/` only contains the helper-generated inventory stub.
- No live `run_phase_g_dense.py` or `studies.fly64_dose_overlap.*` processes are currently running (`pgrep` returns empty), so the long pipeline is not in-flight.
- Pre-flight regression selector (`test_run_phase_g_dense_exec_runs_analyze_digest`) remains GREEN (log archived at `2025-11-09T170500Z/.../green/pytest_orchestrator_dense_exec_recheck.log`).
- Goal for this loop: execute the dense Phase C→G pipeline end-to-end in a fresh hub (`2025-11-09T190500Z`) with `--clobber`, produce the full metrics bundle, and update docs/fix_plan with real MS-SSIM/MAE deltas.

## Scope for Ralph
1. **Workspace + process sanity check**
   - Confirm no leftover orchestrator processes: `pgrep -fl run_phase_g_dense.py` and `pgrep -fl studies.fly64_dose_overlap` should both return empty before starting.
   - If any Phase C artifacts are needed for comparison, note they remain under the 170500Z hub; this run will use a new hub and will not reuse the partial outputs.
2. **Pre-flight regression**
   - Re-run `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv` with `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` exported. Capture log to `$HUB/green/pytest_orchestrator_dense_exec_recheck.log`.
3. **Dense pipeline execution**
   - Set `HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T190500Z/phase_g_dense_full_execution_real_run` from the repo root.
   - Run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log`.
   - Expected duration: 2–4 hours (Phase E training + Phase F recon dominate). Pipeline should emit per-phase CLI logs under `$HUB/cli/` and generate artifacts in `$HUB/analysis/` automatically (summarize, report, digest, highlights).
4. **Evidence capture & verification**
   - After pipeline exits 0, verify the following files exist: `metrics_summary.json`, `metrics_delta_summary.json`, `metrics_delta_highlights.txt`, `metrics_digest.md`, `aggregate_report.md`, `aggregate_highlights.txt`, and `metrics_delta_highlights_preview.txt`.
   - Produce `$HUB/analysis/artifact_inventory.txt` via `find "$HUB" -maxdepth 3 -type f | sort`.
   - Spot-check that highlights text matches the delta JSON (both Baseline and PtyChi comparisons). If mismatched, capture diff in summary.md and investigate before sign-off.
5. **Documentation updates**
   - Update `$HUB/summary/summary.md` with: pre-flight test status, pipeline runtime (elapsed wall-clock), key MS-SSIM/MAE deltas (PtychoPINN vs Baseline/PtyChi), provenance confirmations (CONFIG-001 bridge, DATA-001 compliance), and artifact links.
   - Append the same delta/key notes to `docs/fix_plan.md` Attempts History (linking to the 190500Z hub).
   - If any command fails, archive the failing log under `$HUB/cli/` (or `$HUB/red/` for pytest), summarize the error signature in summary.md, and mark the ledger entry blocked with next actions.

## Required Tests
- `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`

## Artifacts to Produce
- `$HUB/green/pytest_orchestrator_dense_exec_recheck.log`
- `$HUB/cli/run_phase_g_dense.log` plus per-phase CLI logs (`phase_c_generation.log`, `phase_d_dense.log`, etc.)
- `$HUB/analysis/{metrics_summary.json,metrics_delta_summary.json,metrics_delta_highlights.txt,metrics_digest.md,aggregate_report.md,aggregate_highlights.txt,metrics_delta_highlights_preview.txt,artifact_inventory.txt}`
- `$HUB/summary/summary.md` with metrics narrative and links

## Exit Criteria
- Dense pipeline completes successfully with all eight commands executed and summary/report/digest artifacts present under the new hub.
- MS-SSIM/MAE delta values recorded for both Baseline and PtyChi comparisons, and highlights text matches the JSON values.
- Summary.md and docs/fix_plan.md updated with runtime, deltas, provenance, and artifact links.
- Any failures documented with logs + mitigation plan; no silent TODOs.
