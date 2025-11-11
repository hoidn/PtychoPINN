# SSIM Grid Smoke Driver (2025-11-11T013612Z)

## Reality Check
- Retrospective hub `2025-11-11T213000Z` showed four consecutive documentation-only loops with empty `{cli,red,green}` folders; we must re-establish momentum with a focused code+test deliverable before running another dense pipeline.
- `verify_dense_pipeline_artifacts.py::validate_metrics_delta_highlights` (plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py:309-481) now enforces phase-only previews, but no lightweight tool summarizes MS-SSIM/MAE deltas per dose/view for quick sharing.
- `docs/TESTING_GUIDE.md` and input.md still steer engineers toward heavy end-to-end runs; we need a minimal `plans/.../bin/ssim_grid.py` helper plus pytest coverage so future loops can gate on a cheap smoke test.

## Objectives for Ralph (single loop)
1. **Implement ssim_grid helper**
   - File: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/ssim_grid.py`.
   - Behavior: scan one or more study hubs (default: `$HUB`), load `analysis/metrics_delta_summary.json`, and emit `analysis/ssim_grid_summary.md` with a markdown table containing MS-SSIM (phase & amplitude, ±0.000 precision) and MAE (phase & amplitude, ±0.000000 precision) for `vs_Baseline` and `vs_PtyChi` within the selected dose/view.
   - Enforce preview guard: read `analysis/metrics_delta_highlights_preview.txt`; fail with actionable error/exit code if any line contains the substring `amplitude` (reasserting PREVIEW-PHASE-001) and report the offending lines.
   - Add `--hub` argument (required) and `--output` optional path (defaults to `<hub>/analysis/ssim_grid_summary.md`). Follow scriptization Tier-2 header template (docstring with inputs/outputs per AGENTS.md).

2. **Add pytest smoke test**
   - File: `tests/study/test_ssim_grid.py` (new) with function `test_smoke_ssim_grid(tmp_path)`.
   - Test steps: create a fake hub directory with `analysis/metrics_delta_summary.json` + `metrics_delta_highlights_preview.txt`; ensure preview contains only phase labels for GREEN case, then run the script and assert the markdown table includes ± signs with correct precision. Add RED coverage by injecting `amplitude` into preview and asserting the script exits non-zero with the expected message (can run sequentially inside the same test using subprocess or helper).
   - Use pytest tmp_path fixtures; no external files.

3. **Evidence expectations**
   - Before running anything, export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and set `HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T013612Z/ssim_grid_mvp` (already provisioned) so artifacts land under this loop.
   - Store pytest stdout/stderr under `$HUB/green/pytest_ssim_grid_smoke.log` (green run) and `$HUB/red/pytest_ssim_grid_preview_amplitude.log` if you split RED/GREEN executions.
   - Summarize outcomes, preview guard status, and table sample inside `$HUB/summary/summary.md` before handing back.

## Acceptance Criteria
- `plans/.../bin/ssim_grid.py` exists with executable header, argparse, preview guard, and summary writer referencing TYPE-PATH-001 (POSIX relative paths) and PREVIEW-PHASE-001.
- `tests/study/test_ssim_grid.py::test_smoke_ssim_grid` fails (RED) before implementation and passes afterward, capturing logs in artifacts.
- Pytest selector `pytest tests/study/test_ssim_grid.py::test_smoke_ssim_grid -vv` collects and passes locally with evidence under `$HUB`.
- `docs/fix_plan.md` Attempts History updated with this loop’s timestamp/artifact path and reference to the new helper/test.
- `galph_memory.md` records state `ready_for_implementation`, dwell incremented (2), and next action pointing Ralph at the new Do Now.
