# Dense Phase G Run + ssim_grid Integration (2025-11-11T235500Z)

## Reality Check
- 2025-11-11T013612Z/ssim_grid_mvp now holds RED/GREEN pytest logs, collect-only evidence, and `analysis/ssim_grid_summary.md`, so the helper + smoke test are complete.
- The most recent dense hubs (2025-11-11T012044Z and 2025-11-11T005802Z) still contain only plan/analysis notes—no `{cli,analysis}` outputs, metrics deltas, or verifier reports—so no counted Phase D–G run exists post-preview guard fix.
- `run_phase_g_dense.py` does not yet invoke `bin/ssim_grid.py`, which means the new summary/preview guard is not enforced automatically and manual steps are needed after every run.
- `docs/TESTING_GUIDE.md` (§Phase G Delta Metrics Persistence) still claims all deltas use ±0.000 precision and never mentions the preview-only artifact or `ssim_grid.py`; `docs/development/TEST_SUITE_INDEX.md` likewise lacks the new test selector.

## Objectives for Ralph (single loop)
1. **Orchestrator integration** — Extend `plans/active/.../bin/run_phase_g_dense.py::main` to invoke `bin/ssim_grid.py --hub <hub>` after `analyze_dense_metrics.py`, logging to `cli/ssim_grid_cli.log` and surfacing the generated markdown path in the success banner (TYPE-PATH-001 compliant).
2. **Test coverage** — Update `tests/study/test_phase_g_dense_orchestrator.py` so both `test_run_phase_g_dense_collect_only_generates_commands` and `test_run_phase_g_dense_exec_runs_analyze_digest` assert the new helper command/log ordering; stub `ssim_grid.py` invocation to create `analysis/ssim_grid_summary.md` in the exec test.
3. **Counted dense run** — Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and run `python plans/active/.../bin/run_phase_g_dense.py --hub $HUB --dose 1000 --view dense --splits train test --clobber`, pointing `$HUB` to `plans/active/.../reports/2025-11-11T235500Z/phase_g_dense_run_with_ssim_grid/`. Allow the orchestrator to produce full Phase C→G artifacts plus the new `ssim_grid` summary/log.
4. **Verification & digests** — Execute `python plans/active/.../bin/verify_dense_pipeline_artifacts.py --hub $HUB --report $HUB/analysis/verification_report.json --dose 1000 --view dense`, rerun `plans/active/.../bin/check_dense_highlights_match.py` if needed, and archive all CLI/stdout logs under `$HUB/{cli,analysis}`. Capture MS-SSIM/MAE deltas, preview guard results, and the generated `analysis/ssim_grid_summary.md` in `$HUB/summary/summary.md`.
5. **Doc/test registry sync** — Once code/tests pass, refresh `docs/TESTING_GUIDE.md` §Phase G Delta Metrics Persistence and `docs/development/TEST_SUITE_INDEX.md` to document the preview-only artifact, MAE ±0.000000 precision, the new `ssim_grid.py` helper, and `tests/study/test_ssim_grid.py::test_smoke_ssim_grid`.

## Execution Steps
1. Set env + hub: `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `export HUB=$PWD/plans/active/.../reports/2025-11-11T235500Z/phase_g_dense_run_with_ssim_grid`.
2. Modify `run_phase_g_dense.py`:
   - Add helper to build `ssim_grid.py` command + log path.
   - Ensure CLI banner lists `analysis/ssim_grid_summary.md` + `cli/ssim_grid_cli.log` (relative paths).
3. Update orchestrator tests:
   - `test_run_phase_g_dense_collect_only_generates_commands`: assert printed command list includes `ssim_grid.py` + `ssim_grid_cli.log` reference.
   - `test_run_phase_g_dense_exec_runs_analyze_digest`: capture `run_command` calls, assert `ssim_grid.py` executes after `analyze_dense_metrics.py`, stub file creation for `ssim_grid_summary.md`, and include new log expectation.
4. Run targeted pytest (RED/GREEN):
   - `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv`
   - `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`
5. Execute the dense pipeline run with `--clobber`, letting new helper fire automatically; archive orchestrator stdout under `$HUB/cli/phase_g_dense_run.log` if helpful.
6. Run verifier + helper checks (`verify_dense_pipeline_artifacts.py`, `check_dense_highlights_match.py`, optional `analyze_dense_metrics.py` rerun) and copy resulting reports into `$HUB/analysis/`.
7. Capture MS-SSIM/MAE deltas and table snippet in `$HUB/summary/summary.md`, referencing PREVIEW-PHASE-001 + STUDY-001 + TYPE-PATH-001 compliance.
8. Update docs (`docs/TESTING_GUIDE.md`, `docs/development/TEST_SUITE_INDEX.md`) and note log locations in summary.

## Acceptance Criteria
- Orchestrator code emits `ssim_grid.py` command + CLI log, and success banner lists `analysis/ssim_grid_summary.md` with POSIX-relative paths.
- Updated tests pass (collect-only + exec) and prove command ordering: reporting helper → analyze digest → ssim_grid summary.
- Full pipeline run completes under the new hub with populated `{analysis,cli}` directories, including metrics deltas, preview.txt, digest, `ssim_grid_summary.md`, and UTC-stamped logs.
- `verify_dense_pipeline_artifacts.py` report shows all validations passing; report JSON stored under `$HUB/analysis/`.
- Summary captures MS-SSIM/MAE deltas (phase emphasis) and references the helper log + verifier outputs; docs/test index mention the helper/test.

## Evidence & Artifacts
- `$HUB/cli/*.log` (Phase C–G CLI logs + `ssim_grid_cli.log`)
- `$HUB/analysis/{metrics_summary.json,metrics_delta_summary.json,metrics_delta_highlights.txt,metrics_delta_highlights_preview.txt,metrics_digest.md,ssim_grid_summary.md,verification_report.json}`
- `$HUB/green/pytest_phase_g_dense_exec.log`, `$HUB/green/pytest_phase_g_dense_collect_only.log`, plus matching RED logs if failures precede fixes.
- `$HUB/summary/summary.md` with MS-SSIM/MAE table, preview guard status, doc/test diff notes.

## Risks & Mitigations
- **Runtime length:** Dense run is long; if it times out, capture partial logs + blocker reason, then rerun after addressing failure.
- **Preview guard regressions:** If the new helper surfaces amplitude contamination, retain the failing preview file in `$HUB/analysis/` for triage.
- **Doc drift:** Ensure doc updates happen after evidence is captured; mention new helper/test in both TESTING_GUIDE and TEST_SUITE_INDEX to avoid future confusion.

## Findings Applied
- POLICY-001 (PyTorch required for baseline recon; orchestrator already enforces)
- CONFIG-001 (params.cfg bridge via AUTHORITATIVE_CMDS_DOC)
- DATA-001 (NPZ/JSON structure validated by verifier)
- TYPE-PATH-001 (relative paths in banners + markdown)
- STUDY-001 (phase-focused MS-SSIM/MAE reporting)
- TEST-CLI-001 (CLI logs + pytest red/green evidence under hub)
- PREVIEW-PHASE-001 (ssim_grid helper enforces phase-only preview)

## Exit Criteria
- Code + tests merged for `run_phase_g_dense.py` + orchestrator tests.
- Dense pipeline run artifacts + verifier report + ssim_grid summary archived in `$HUB`.
- Docs + test registry updated; input.md rewritten with next focus if additional work remains.
