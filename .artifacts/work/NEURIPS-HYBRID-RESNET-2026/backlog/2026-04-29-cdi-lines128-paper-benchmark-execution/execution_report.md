# Execution Report

## Completed In This Pass

- Fixed complete-table finalization so promoted recovered PyTorch rows are treated as reuse-backed evidence during `complete_table` runs, even though the top-level CLI does not set `--reuse-existing-recons`.
- Passed `launcher_stdout.log` into recovery finalization so rows promoted from mixed source roots can synthesize `launcher_completion.json` from the wrapper eval markers when copied `launcher_stderr.log` is not authoritative for that row.
- Added an end-to-end complete-table regression in `tests/studies/test_lines128_paper_benchmark.py` that exercises promoted rows from both the minimum-subset root and the separate FFNO prerequisite root, then asserts `launcher_completion.json` is present in both `metrics.json` and `paper_benchmark_manifest.json`.

## Completed Current-Scope Work

- The blocking review issue is resolved: complete-table repaired bundles no longer rely on `paper_complete` status without row-local `launcher_completion.json` for promoted recovered PyTorch rows.
- Focused and required verification for the approved backlog contract passed again after the fix.

## Follow-Up Work

- Top-level `launcher_stdout.log` / `launcher_stderr.log` in repaired roots can still originate from copied promoted-source artifacts before the recovery pass writes its own wrapper logs. That did not block the current fix because row-local `launcher_completion.json` now comes from the wrapper recovery markers, but the bundle would be cleaner if repair-root launcher logs were always emitted fresh or source-root logs were kept strictly row-scoped.

## Verification

- Focused regression:
  - `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_paper_provenance.py tests/test_grid_lines_compare_wrapper.py`
  - result: `80 passed, 23 warnings in 18.11s`
  - log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/pytest_focused_20260430T143750Z.log`
- Required deterministic gate:
  - `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  - result: `173 passed, 47 warnings in 300.80s (0:05:00)`
  - log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/pytest_required_20260430T143750Z.log`
- Compile gate:
  - `python -m compileall -q ptycho_torch scripts/studies`
  - result: exit `0`
  - log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/compileall_required_20260430T143750Z.log`

## Residual Risks

- The authoritative paper-facing root remains `complete_table_20260430T141325Z_repair`; this pass repaired the code path and regression coverage, but it did not rewrite historical claim-bearing artifacts.
- The remaining cleanliness issue is top-level launcher-log provenance in repaired roots, as noted above under Follow-Up Work.
