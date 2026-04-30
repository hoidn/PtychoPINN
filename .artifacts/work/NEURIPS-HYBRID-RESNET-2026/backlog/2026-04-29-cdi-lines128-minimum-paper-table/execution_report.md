## Completed In This Pass

- fixed the review-blocking provenance contract bug so paper-grade validation now
  requires referenced row-local files to exist, not just non-empty path strings
- tightened same-root recovery so recovered rows only promote to `paper_grade`
  when `invocation.sh` and row visual PNGs exist alongside the prior config,
  history, metrics, and recon artifacts
- added final bundle validation for wrapper/root artifacts and reran the
  authoritative same-root collation in
  `runs/minimum_subset_20260429T235811Z` with `--reuse-existing-recons`

## Completed Current-Scope Work

- authoritative root remains:
  `runs/minimum_subset_20260429T235811Z`
- final bundle state after the rerun:
  `benchmark_status=paper_complete`,
  `claim_boundary=minimum_draftable_cdi_subset`,
  `missing_bundle_artifacts=[]`
- required rows remain complete with empty `missing_fields_by_row`:
  `baseline`, `pinn`, `pinn_hybrid_resnet`, `pinn_fno_vanilla`
- updated bundle artifacts in the authoritative root:
  `metrics.json`, `model_manifest.json`, `paper_benchmark_manifest.json`
- verification:
  - `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py`
    -> `196 passed, 47 warnings in 39.96s`
  - `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
    -> `171 passed, 47 warnings in 299.93s`
  - `python -m compileall -q ptycho_torch scripts/studies`
    -> exit `0`
  - same-root recovery command:
    `python scripts/studies/lines128_paper_benchmark.py --mode minimum_subset --decision-artifact .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json --execution-authority-note docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md --execution-manifest .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json --output-dir .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T235811Z --reuse-existing-recons`
    -> exit `0`

## Follow-Up Work

- later complete-table CDI rows `pinn_spectral_resnet_bottleneck_net` and
  `pinn_ffno` remain intentionally out of scope for this minimum-subset item
- wrapper dirty-state provenance is still derived from the current workspace at
  manifest-write time; persisting that value directly at invocation time remains
  a separate follow-up if the project wants that field to be immutable

## Residual Risks

- this item still closes only the minimum draftable CDI subset, not the later
  complete `lines128` paper table
- paper-grade validity still depends on the recovered authoritative root
  remaining intact; deleting row-local or visual artifacts from that root would
  correctly downgrade the bundle on the next validation pass
- reproduction still depends on the recorded local runtime stack
  (`ptycho311`, local GPU, local TF/Torch installs) even though the path-based
  provenance gate is now enforced honestly
