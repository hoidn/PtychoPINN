# Phase G Dense Real-Run Evidence — Supervisor Plan (2025-11-09T130500Z)

## Objective
Run the dense Phase C→G pipeline with `--clobber` to capture real MS-SSIM/MAE deltas while enriching `metrics_delta_summary.json` with provenance metadata (`generated_at`, `source_metrics`). Ensure tests assert the metadata schema and documentation reflects the update.

## Scope for Ralph
1. **Delta metadata TDD**
   - Extend `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest` to expect a `generated_at` ISO8601Z timestamp and `source_metrics` (relative path to `metrics_summary.json`) within the JSON.
   - Confirm RED by running the selector before modifying the orchestrator.
2. **Orchestrator metadata persistence**
   - Update `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main` to emit the new metadata fields when writing `metrics_delta_summary.json`. Timestamp should be generated with `datetime.now(timezone.utc)` and serialized via `.isoformat()`, and `source_metrics` should stay TYPE-PATH-001 compliant (relative to hub).
3. **Docs refresh**
   - Amend `docs/TESTING_GUIDE.md` Phase G section to document the enriched JSON schema and how to verify the metadata during evidence runs.
4. **Evidence run**
   - Execute `run_phase_g_dense.py --clobber` for the dense view, capture CLI logs, and archive fresh artifacts: `metrics_summary.json`, `metrics_delta_summary.json`, `metrics_digest.md`, `aggregate_report.md`, highlights, inventory.
   - Extract MS-SSIM/MAE deltas from the JSON into the hub summary + docs/fix_plan.md Attempts History.

## Required Tests
- `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`
- `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv`
- `pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv`

## Artifacts to Produce
- `$HUB/red/pytest_orchestrator_delta_metadata_red.log` — RED failure for missing metadata.
- `$HUB/green/pytest_orchestrator_delta_metadata_green.log` — GREEN success for metadata selector.
- `$HUB/green/pytest_collect_only.log`, `$HUB/green/pytest_analyze_success.log` — regression guards.
- `$HUB/cli/run_phase_g_dense.log` — full pipeline execution with banner.
- `$HUB/analysis/{metrics_summary.json,metrics_delta_summary.json,metrics_digest.md,aggregate_report.md,aggregate_highlights.txt,metrics_delta_highlights.txt}` — refreshed evidence set.
- `$HUB/analysis/artifact_inventory.txt` — sorted file list for traceability.
- `$HUB/summary/summary.md` — loop summary capturing deltas, tests, and artifacts.

## Exit Criteria
- Metadata fields (`generated_at`, `source_metrics`) present in `metrics_delta_summary.json` and covered by tests/docs.
- All mapped selectors GREEN with logs archived in hub.
- Dense pipeline completes successfully with fresh artifacts under `analysis/`.
- Summary and ledger updated with MS-SSIM/MAE deltas and artifact pointers.
