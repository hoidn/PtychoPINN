# Phase G Dense Pipeline Evidence Plan (2025-11-10T093500Z)

## Current Status Snapshot
- Reviewed prior hub `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T210500Z/phase_g_dense_full_execution_real_run/`; only RED/GREEN pytest logs and planning documents exist. `analysis/` is empty (no metrics bundle, no artifact inventory) and no CLI logs were produced.
- Earlier hub `2025-11-09T170500Z` still holds Phase C NPZs plus `analysis/artifact_inventory_partial.txt`, confirming the dense Phase C→G orchestration never completed end-to-end in any hub.
- `plans/active/.../bin/verify_dense_pipeline_artifacts.py` validates metrics/highlights but does **not** confirm that `analysis/artifact_inventory.txt` exists or contains the required bundle paths, leaving a blind spot for the upcoming evidence run.

## Gaps / Risks
1. **Verifier coverage gap** — Missing validation for `analysis/artifact_inventory.txt` could allow silent regressions in artifact provenance, undermining ledger traceability (violates TYPE-PATH-001 + STUDY-001 expectations).
2. **Evidence deficit** — No real dense Phase G metrics or highlights have been captured post-inventory change; pipeline runtime remains unverified in practice.

## Objectives for Ralph (single loop)
1. **TDD guard for artifact inventory validation**
   - Add a targeted pytest (new module `tests/study/test_phase_g_dense_artifacts_verifier.py`) that exercises `plans/active/.../bin/verify_dense_pipeline_artifacts.py::main` against a synthetic hub.
   - First scenario: inventory missing key entries (`metrics_summary.json`, `aggregate_report.md`) → expect non-zero exit and error in JSON report. Capture RED log at `$HUB/red/pytest_artifact_inventory_fail.log`.
   - Second scenario (GREEN): complete inventory with required paths → expect success. Capture GREEN log at `$HUB/green/pytest_artifact_inventory_fix.log`.
2. **Implement inventory validation in verifier**
   - Extend `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py` with a helper (e.g., `validate_artifact_inventory`) ensuring the file exists, uses POSIX relative paths, and includes the metrics/highlights bundle entries. Integrate the check into `main()` alongside existing validations and surface detailed diagnostics (missing entries, empty file, absolute paths).
3. **Run dense Phase C→G pipeline and verify outputs**
   - Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T093500Z/phase_g_dense_full_execution_real_run`.
   - Execute `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log` (2–4 hours expected). Confirm `[1/8]`…`[8/8]` banners, Phase D–G CLI logs, and metrics bundle under `$HUB/analysis/`.
   - Run the enhanced verifier to produce `$HUB/analysis/pipeline_verification.json` and ensure the new inventory checks pass.
4. **Document outcomes**
   - Update `$HUB/summary/summary.md` with runtime, MS-SSIM/MAE deltas vs Baseline/PtyChi, metadata compliance status, and verifier summary. Refresh `docs/fix_plan.md` Attempts History with this loop’s results and record any new lessons in `docs/findings.md` (if applicable).

## Required Tests (mapped selectors)
- `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_artifact_inventory_blocks_missing_entries -vv`
- `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_artifact_inventory_passes_with_complete_bundle -vv`
- `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`

## Artifacts to Capture (under `$HUB`)
- `red/pytest_artifact_inventory_fail.log`
- `green/pytest_artifact_inventory_fix.log`
- `green/pytest_orchestrator_dense_exec_inventory_fix.log` (rerun to ensure orchestration guard still GREEN)
- `cli/run_phase_g_dense.log` plus per-phase logs (`cli/phase_c_generation.log`, `cli/phase_d_dense.log`, …, `cli/phase_g_compare.log`)
- `analysis/{artifact_inventory.txt,comparison_manifest.json,metrics_summary.json,metrics_summary.md,metrics_delta_summary.json,metrics_delta_highlights.txt,metrics_digest.md,aggregate_report.md,aggregate_highlights.txt,pipeline_verification.json}`
- `summary/summary.md` with MS-SSIM/MAE deltas and provenance references

## Findings to Reaffirm
- POLICY-001 (PyTorch dependency policy)
- CONFIG-001 (Legacy bridge guard)
- DATA-001 (Dataset contract compliance)
- TYPE-PATH-001 (Path normalization + inventory POSIX paths)
- OVERSAMPLING-001 (Dense overlap definition)
- STUDY-001 (Metrics/summary coverage expectations)
- PHASEC-METADATA-001 (Metadata compliance visibility in Phase G)

