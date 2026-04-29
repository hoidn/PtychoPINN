# NeurIPS Lines128 Minimum Paper Table Summary

- Date: `2026-04-29`
- Backlog item: `2026-04-29-cdi-lines128-minimum-paper-table`
- State: `paper_complete`
- Chosen execution path: `same_root_recovery`
- Authoritative root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T213028Z`

## Completed In This Pass

- completed the previously failed four-row minimum CDI paper bundle in the
  existing root `minimum_subset_20260429T213028Z` without retraining any row
- extended the reuse path in `scripts/studies/grid_lines_compare_wrapper.py` so
  same-root recovery can rebuild paper-row payloads from existing TensorFlow and
  PyTorch row artifacts, including parameter counts, final epoch/loss fields,
  hardware summaries, and recovered runtime metadata
- threaded `--reuse-existing-recons` through
  `scripts/studies/lines128_paper_benchmark.py` so the benchmark entry point can
  drive honest same-root completion instead of forcing a new write-side rerun
- hardened `scripts/studies/metrics_tables.py` so bundle collation accepts
  NumPy scalar / array metric pairs during JSON emission
- added regression coverage for recovered row payloads, CLI reuse plumbing, and
  NumPy metric-pair serialization

## Final Bundle Contract

- fixed runtime roster:
  `baseline`, `pinn`, `pinn_hybrid_resnet`, `pinn_fno_vanilla`
- paper-facing row labels:
  `CDI CNN + supervised`, `CDI CNN + PINN`, `Hybrid ResNet + PINN`,
  `FNO Vanilla + PINN`
- selected FNO comparator:
  `fno_vanilla`
- claim boundary:
  `minimum_draftable_cdi_subset`
- status:
  `paper_complete`
- fixed sample ids:
  `0`, `1`
- shared visual-scale policy:
  stitched numeric arrays for amplitude/phase and derived shared absolute-error
  scales

## Bundle Outputs

- merged bundle artifacts now exist in the authoritative root:
  - `metrics.json`
  - `metric_schema.json`
  - `model_manifest.json`
  - `metrics_table.csv`
  - `metrics_table.tex`
  - `metrics_table_best.tex`
- recovered row metadata in the final bundle:
  - `baseline`: `parameter_count=4612418`, `final_completed_epoch=40`,
    `final_train_loss=0.079`
  - `pinn`: `parameter_count=4661212`, `final_completed_epoch=40`,
    `final_train_loss=6.773`
  - `pinn_hybrid_resnet`: `parameter_count=18006600`,
    `final_completed_epoch=40`,
    `final_train_loss=0.027469921857118607`
  - `pinn_fno_vanilla`: `parameter_count=1272902`,
    `final_completed_epoch=40`,
    `final_train_loss=0.0719698816537857`

## Verification

- focused recovery regression suite:
  `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py`
  -> `189 passed, 45 warnings in 29.86s`
- required deterministic gates:
  `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  -> `169 passed, 45 warnings in 285.72s`
- required compile gate:
  `python -m compileall -q ptycho_torch scripts/studies`
  -> exit `0`
- same-root recovery command:
  `python scripts/studies/lines128_paper_benchmark.py --mode minimum_subset --decision-artifact .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json --execution-authority-note docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md --execution-manifest .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json --output-dir .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T213028Z --reuse-existing-recons`
  -> exit `0`
- archived logs:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/pytest_focused_recovery.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/pytest_deterministic_gates_recovery.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/compileall_recovery.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/lines128_same_root_recovery.log`

## Boundary And Remaining Scope

- this item closes only the minimum draftable CDI subset and does not widen the
  result to the later complete `lines128` paper table
- `pinn_spectral_resnet_bottleneck_net` and `pinn_ffno` remain later
  out-of-scope rows
- the harness preflight note remains readiness-only authority; launch authority
  for this completed bundle remains the checked-in minimum-subset execution note
  plus its machine-readable execution manifest
