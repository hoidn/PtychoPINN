## Completed In This Pass

- completed the minimum CDI paper-table bundle by finishing honest same-root
  recovery in `runs/minimum_subset_20260429T213028Z`
- patched the compare-wrapper reuse path so recovered TensorFlow and PyTorch row
  artifacts can rebuild the paper row payloads required for final collation
- added benchmark-entry reuse plumbing plus NumPy metric-pair serialization
  hardening, then covered those behaviors with focused regression tests
- refreshed the durable audit trail with the chosen `same_root_recovery` path
  and wrote the minimum-table summary

## Completed Plan Tasks

- Task 1 complete:
  reconfirmed the authority surfaces, preserved the readiness-only versus
  launch-authority split, and selected `minimum_subset_20260429T213028Z` as the
  honest same-root recovery candidate
- Task 2 complete:
  closed the remaining recovery gap in
  `scripts/studies/grid_lines_compare_wrapper.py`,
  `scripts/studies/lines128_paper_benchmark.py`, and
  `scripts/studies/metrics_tables.py`, with regression coverage in
  `tests/test_grid_lines_compare_wrapper.py`,
  `tests/studies/test_lines128_paper_benchmark.py`, and
  `tests/studies/test_metrics_tables.py`
- Task 3 complete:
  reran and archived the focused recovery suite, the required deterministic
  pytest gates, and the required `compileall` gate before the final write-side
  recovery
- Task 4 complete:
  reran only the final benchmark collation path with
  `--reuse-existing-recons` in the chosen root and verified that the root now
  emits `metrics.json`, `metric_schema.json`, `model_manifest.json`,
  `metrics_table.csv`, `metrics_table.tex`, and `metrics_table_best.tex`
- Task 7 closeout complete:
  wrote the durable summary, updated `docs/index.md`, and published the
  `COMPLETED` implementation-state bundle

## Remaining Required Plan Tasks

- none for this backlog item
- later complete-table CDI rows `pinn_spectral_resnet_bottleneck_net` and
  `pinn_ffno` remain intentionally out of scope for the later complete
  `lines128` benchmark item

## Verification

- focused recovery regression suite:
  `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py`
  -> `189 passed, 45 warnings in 29.86s`
- required deterministic gate:
  `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  -> `169 passed, 45 warnings in 285.72s`
- required compile gate:
  `python -m compileall -q ptycho_torch scripts/studies`
  -> exit `0`
- same-root recovery command:
  `python scripts/studies/lines128_paper_benchmark.py --mode minimum_subset --decision-artifact .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json --execution-authority-note docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md --execution-manifest .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json --output-dir .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T213028Z --reuse-existing-recons`
  -> exit `0`
- archived verification logs:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/pytest_focused_recovery.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/pytest_deterministic_gates_recovery.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/compileall_recovery.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/lines128_same_root_recovery.log`
- final bundle standard:
  `benchmark_status=paper_complete` with empty `missing_fields_by_row` for all
  four required rows

## Residual Risks

- this completed bundle is the minimum draftable CDI subset, not the later
  complete `lines128` paper table
- the authoritative root is recovered from existing row-local artifacts rather
  than from a single uninterrupted original collation pass, so the final bundle
  correctly records `recovered_from_existing_artifacts` caveats on the rows
- any future fresh rerun remains sensitive to local disk availability because
  `minimum_subset_20260429T224103Z` failed under `OSError: [Errno 28] No space
  left on device`
