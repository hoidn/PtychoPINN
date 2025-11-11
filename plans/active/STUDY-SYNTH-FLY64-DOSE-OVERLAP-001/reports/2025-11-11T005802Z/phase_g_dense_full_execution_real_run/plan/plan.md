# Phase G Dense Preview Guard & Evidence Plan (2025-11-11T005802Z)

## Reality Check
- Helper + tests for `persist_delta_highlights()` landed in commit d6029656, so the orchestrator now emits `metrics_delta_highlights_preview.txt` with phase-only lines and metric-specific precision (3 decimals for MS-SSIM, 6 for MAE).
- `verify_dense_pipeline_artifacts.py::validate_metrics_delta_highlights` (`plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py`:309-481) currently ensures the preview file exists and that the formatted phase deltas appear in both highlights + preview, but it **never asserts that preview lines stay phase-only**; a regression that reintroduces `amplitude` text would still pass.
- `docs/TESTING_GUIDE.md` §“Phase G Delta Metrics Persistence” still states “Signed 3-decimal formatting for all deltas” and never documents the preview artifact, so the spec is out of sync with the new helper.
- Hub `plans/active/.../reports/2025-11-11T003351Z/phase_g_dense_full_execution_real_run/` is still empty (`analysis/`, `cli/`, `green/`, `red/` contain only placeholders): no Phase D–G CLI logs, metrics summary, highlights, or verifier outputs exist yet.
- Created fresh hub `HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T005802Z/phase_g_dense_full_execution_real_run/` with `{analysis,cli,collect,green,red,summary}` to stage the hardened validator + dense run evidence; AUTHORITATIVE_CMDS_DOC remains `./docs/TESTING_GUIDE.md`.

## Objectives for Ralph (single loop)
1. **TDD preview-phase guard**
   - Extend `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py::validate_metrics_delta_highlights` to reject preview lines containing `amplitude` (case-insensitive) or missing the phase-only prefix. Capture offending lines in new metadata (`preview_phase_only` bool + `preview_format_errors` list) so reports stay actionable.
   - Add a new RED test `tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_contains_amplitude` that constructs a hub with valid highlights + JSON but injects `amplitude` into the preview; assert the validator fails with the new metadata populated.
   - Update the existing GREEN test `...::test_verify_dense_pipeline_highlights_complete` to assert `preview_format_errors == []` and `preview_phase_only is True`, keeping MODE=TDD.
2. **Doc sync for new artifact rules**
   - Amend `docs/TESTING_GUIDE.md` Phase G delta section so MAE precision is ±0.000000, document the preview artifact (phase-only lines, 4 entries), and cite TYPE-PATH-001 requirements.
   - Update `docs/development/TEST_SUITE_INDEX.md` (Phase G section) to mention the new preview-phase-only pytest selector so other engineers can discover it.
3. **Dense pipeline execution (new hub)**
   - Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T005802Z/phase_g_dense_full_execution_real_run`.
   - Run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log` and let it proceed through `[1/8]`→`[8/8]` to SUCCESS; archive per-phase logs under `$HUB/cli/`.
4. **Verification, highlights parity, and evidence bundle**
   - `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log`
   - `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" |& tee "$HUB"/analysis/highlights_check.log`
   - `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_highlights.log`
   - Summarize MS-SSIM/MAE (phase emphasis) deltas, CLI guard status, verifier results, artifact inventory counts, and highlight preview evidence in `$HUB/summary/summary.md`; update docs/fix_plan.md + Turn Summary with artifact links.

## Required Tests / Evidence
- `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_contains_amplitude -vv`
- `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv`
- `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`
- `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv`
- Dense pipeline run + verifier/checker logs archived under `$HUB`

## Findings to Reaffirm
- **POLICY-001** — PyTorch is mandatory for PtyChi reconstruction inside the dense pipeline.
- **CONFIG-001** — Export AUTHORITATIVE_CMDS_DOC before invoking orchestrator/verifier to keep legacy modules in sync.
- **DATA-001** — Do not mutate generated Phase C NPZ artifacts; rely on validators instead.
- **TYPE-PATH-001** — Preview/highlights paths in inventory + JSON must be POSIX-relative; keep helper + validator path handling consistent.
- **STUDY-001** — Report MS-SSIM/MAE deltas with explicit ± signs, phase emphasis.
- **TEST-CLI-001** — CLI/verifier tests must cover real filename patterns and block incomplete log bundles.
