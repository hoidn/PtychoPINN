# Phase G Dense Evidence Loop — 2025-11-09T090500Z

## Objective
Run the dense Phase C→G pipeline end-to-end with the new digest banner improvements, capture real MS-SSIM/MAE deltas, and surface the key delta values directly in stdout so the evidence bundle is easy to audit.

## Scope
- Extend the orchestrator success banner to print MS-SSIM and MAE deltas (PtychoPINN vs Baseline / PtyChi) derived from `analysis/metrics_summary.json`.
- Guard the new output with pytest (RED→GREEN).
- Execute `run_phase_g_dense.py --clobber` for dose=1000, view=dense, splits train/test, archiving CLI + digest artifacts under this hub.
- Extract MS-SSIM/MAE deltas from the new stdout block and record them in summary/docs.

## Tasks
1. **TDD — Delta summary guard (RED)**
   - Update `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest` to
     - Seed `metrics_summary.json` in the stubbed summarizer with aggregate metrics for Baseline/PtyChi/PtychoPINN.
     - Assert stdout includes four lines:
       1. `MS-SSIM Δ vs Baseline: amplitude …, phase …`
       2. `MS-SSIM Δ vs PtyChi: amplitude …, phase …`
       3. `MAE Δ vs Baseline: amplitude …, phase …`
       4. `MAE Δ vs PtyChi: amplitude …, phase …`
     - Store RED log at `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T090500Z/phase_g_dense_full_execution_real_run/red/pytest_orchestrator_delta.log`.
2. **Implementation — Banner delta summary (GREEN)**
   - Modify `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main` to call a new helper (e.g., `print_key_metric_deltas`) after `analyze_digest_cmd` completes.
   - Helper responsibilities:
     - Load `{hub}/analysis/metrics_summary.json`.
     - Compute mean MS-SSIM and MAE deltas (PtychoPINN - Baseline, PtychoPINN - PtyChi) using aggregate metrics.
     - Print a formatted block with the four delta lines listed above; fall back to `N/A` when values are missing.
     - Keep TYPE-PATH-001 compliance (Paths + safe formatting).
   - Run GREEN for the orchestrator test + guard selectors:
     - `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest`
     - `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands`
     - `pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest`
     - Log outputs under `green/` with descriptive filenames.
3. **Evidence Run — Dense pipeline execution**
   - Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`.
   - Execute:
     ```
     python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py \
       --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T090500Z/phase_g_dense_full_execution_real_run \
       --dose 1000 --view dense --splits train test --clobber
     ```
   - Tee stdout to `cli/run_phase_g_dense.log` (store stderr too, or redirect `2>&1`).
   - Verify `analysis/metrics_summary.json`, `analysis/metrics_digest.md`, `cli/metrics_digest_cli.log`, `analysis/aggregate_highlights.txt`, and the new stdout delta block exist.
4. **Digest + Highlights Review**
   - Run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --metrics <hub>/analysis/metrics_summary.json --highlights <hub>/analysis/aggregate_highlights.txt --output <hub>/analysis/manual_metrics_digest.md` only if orchestrator did not already generate digest (expect skip if present).
   - Capture MS-SSIM/MAE delta values (Baseline vs PtyChi) from stdout block and digest, storing quick notes in `analysis/metrics_highlights.txt`.
5. **Documentation & Ledger Sync**
   - Update `summary/summary.md` (prepend Turn Summary later via supervisor instructions) with:
     - RED/GREEN status, selector counts, and key delta numbers.
     - CLI log + digest file inventory.
   - Update `docs/fix_plan.md` Attempts History with execution details, delta values, and artifact path `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T090500Z/phase_g_dense_full_execution_real_run/`.
   - No doc registry changes unless selectors are renamed (expect none).

## Success Criteria
- Orchestrator delta summary test fails before implementation and passes after helper is added.
- Guard selectors remain GREEN (collect-only + analyze digest success).
- Dense pipeline command completes with exit code 0 and new stdout delta block present.
- Metrics digest + CLI log captured; key deltas recorded in summary and docs.

## Risks / Mitigations
- **Long runtime (2–4 h):** Kick off pipeline immediately after GREEN tests; monitor `cli/run_phase_g_dense.log`. If interruption occurs, capture partial log and record blocker in fix_plan.
- **Missing metrics_summary.json:** Helper should emit clear error message and return early; document in Attempts if encountered.
- **Delta computation missing fields:** Helper should guard each lookup and print `N/A` instead of raising.

## Artifacts
Store all evidence under `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T090500Z/phase_g_dense_full_execution_real_run/` with subdirs for red/green/collect/cli/analysis/summary.
