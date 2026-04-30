# Execution Report

## Completed In This Pass

- Implemented a complete-table execution path in `scripts/studies/lines128_paper_benchmark.py` that can promote audited prerequisite rows, rerun only still-missing rows, and finalize the six-row Lines128 paper bundle.
- Added complete-table recovery and provenance regressions in:
  - `tests/studies/test_lines128_paper_benchmark.py`
  - `tests/studies/test_paper_provenance.py`
- Wrote the deterministic pre-launch audit surfaces:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/execution/row_audit_manifest.json`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/execution/complete_execution_manifest.json`
- Launched the initial complete-table root at:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T134500Z`
  - this root completed the fresh `pinn_spectral_resnet_bottleneck_net` training/inference pass with tracked `exit_code=0`
  - it remained `benchmark_incomplete` because promoted-row recovery still omitted TensorFlow parameter-count source files and FFNO completion-proof recovery
- Repaired the promotion/finalization path, emitted:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/execution/complete_execution_manifest_repair.json`,
  and published the authoritative repaired root at:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T141325Z_repair`
- Wrote the durable summary and study-index updates for the final six-row `paper_complete` bundle.

## Completed Plan Tasks

- Tranche 1: complete-table execution authority and row-audit artifacts are frozen and recorded with final per-row promote/rerun decisions.
- Tranche 2: the harness/execution surface now supports full six-row complete-table assembly without regressing existing `preflight` or `minimum_subset` behavior.
- Tranche 3: the spectral row was rerun once in the initial complete-table root, then the final authoritative bundle was rebuilt in a fresh repaired root with all six rows present and paper-grade complete.
- Tranche 4: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`, `docs/studies/index.md`, and `docs/index.md` now point to the same authoritative repaired root and claim boundary.
- Tranche 5: focused regression, required deterministic pytest gates, and `compileall` were rerun with archived logs under this item’s verification directory.

## Remaining Required Plan Tasks

- None for the approved current scope.
- The earlier root `complete_table_20260430T134500Z` remains preserved as the fresh spectral-rerun source, but it is superseded and not the claim-bearing bundle.

## Verification

- Final authoritative root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T141325Z_repair`
  - `invocation.json`: `status=completed`, `exit_code=0`
  - `metrics.json`: `benchmark_status=paper_complete`, `selected_fno_comparator=fno_vanilla`, no missing bundle artifacts, no missing row fields
  - `model_manifest.json` and `paper_benchmark_manifest.json`: `benchmark_status=paper_complete`
- Focused execution-surface regression:
  - `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_paper_provenance.py tests/test_grid_lines_compare_wrapper.py`
  - result: `79 passed, 23 warnings in 18.99s`
  - log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/pytest_focused_20260430T141325Z.log`
- Required deterministic gate:
  - `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  - result: `173 passed, 47 warnings in 301.74s (0:05:01)`
  - log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/pytest_required_20260430T141325Z.log`
- Compile gate:
  - `python -m compileall -q ptycho_torch scripts/studies`
  - result: exit `0`
  - log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/compileall_required_20260430T141325Z.log`
- Repair command logs:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/lines128_complete_bundle_regen_20260430T070700Z.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/lines128_complete_table_repair_20260430T140954Z.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/lines128_complete_table_repair_20260430T141325Z.log`

## Residual Risks

- The authoritative bundle is the repaired root `complete_table_20260430T141325Z_repair`; consumers must not use the earlier `complete_table_20260430T134500Z` root for paper-facing claims.
- FFNO row provenance still depends on backfilled row invocation records from the prerequisite repair pass, but the authoritative repaired root now includes the row-local completion proof required by the current complete-table bundle contract.
