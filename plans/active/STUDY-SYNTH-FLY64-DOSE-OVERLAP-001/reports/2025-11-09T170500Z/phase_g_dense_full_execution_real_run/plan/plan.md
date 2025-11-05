# Phase G Dense Real-Run Evidence — Supervisor Plan (2025-11-09T170500Z)

## Objective
Run the dense Phase C→G pipeline with `--clobber` to capture true MS-SSIM/MAE deltas now that the highlights artifact is automated. Archive the full evidence bundle (CLI log, metrics JSON, delta summary, highlights text, digest, aggregate Markdown, inventory) and propagate the results into summary.md plus docs/fix_plan.md.

## Scope for Ralph
1. **Pre-flight validation**
   - Re-run `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv` to ensure the highlights regression stays GREEN before launching the long pipeline.
   - If the selector fails, triage using existing hub logs before proceeding.
2. **Dense pipeline execution**
   - Invoke `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber` with `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` exported.
   - Monitor runtime (expected 2–4 hours). Preserve stdout/stderr by teeing into `$HUB/cli/run_phase_g_dense.log`.
   - On success, confirm the new artifacts under `$HUB/analysis/`:
     - `metrics_summary.json`
     - `metrics_delta_summary.json`
     - `metrics_delta_highlights.txt`
     - `metrics_digest.md`
     - `aggregate_report.md`
     - `aggregate_highlights.txt`
3. **Evidence capture**
   - Generate `$HUB/analysis/artifact_inventory.txt` via `find ... | sort`.
   - Preview highlights via `cat metrics_delta_highlights.txt` into `$HUB/analysis/metrics_delta_highlights_preview.txt`.
   - Extract MS-SSIM/MAE deltas (PtychoPINN vs Baseline/PtyChi) for summary + ledger. Note photon dose, overlap setting, training snapshots used.
4. **Doc and ledger updates**
   - Update `$HUB/summary/summary.md` with RED→GREEN validation, runtime duration, MS-SSIM/MAE deltas, provenance metadata check, and artifact links.
   - Record the same deltas and artifact path in `docs/fix_plan.md` Attempts History.
   - If any failure occurs, log the error signature in summary.md and mark the ledger entry blocked with next steps.

## Required Tests
- `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`

## Artifacts to Produce
- `$HUB/green/pytest_orchestrator_dense_exec_recheck.log` — GREEN proof for the regression selector.
- `$HUB/cli/run_phase_g_dense.log` — full pipeline stdout/stderr.
- `$HUB/analysis/{metrics_summary.json,metrics_delta_summary.json,metrics_delta_highlights.txt,metrics_digest.md,aggregate_report.md,aggregate_highlights.txt,artifact_inventory.txt,metrics_delta_highlights_preview.txt}` — refreshed evidence bundle.
- `$HUB/summary/summary.md` — curated narrative with extracted metrics and provenance validation.

## Exit Criteria
- Dense pipeline completes successfully with fresh artifacts under the new hub.
- Highlights text matches JSON delta values (spot-check both Baseline and PtyChi comparisons).
- Summary.md + docs/fix_plan.md document the real MS-SSIM/MAE deltas and link to artifacts.
- No pending TODOs; any blockers captured with logs and mitigation plan.
