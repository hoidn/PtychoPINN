# NeurIPS Lines128 Minimum Paper Table Summary

- Date: `2026-04-29`
- Backlog item: `2026-04-29-cdi-lines128-minimum-paper-table`
- State: `implementation_fixed_fresh_rerun_required`
- Chosen execution path: `fresh_rerun_required_after_tf_row_log_fix`
- Authoritative root: none in this pass

## Completed In This Pass

- fixed the review-blocking provenance path in code so TensorFlow rows now
  write real row-scoped workflow logs and the compare wrapper no longer stamps
  one shared TF transcript onto multiple required rows
- tightened paper-grade validation so required rows are downgraded when they
  reuse identical `stdout.log` payloads
- added regression coverage for both the wrapper-overwrite bug and the
  duplicate-log validator failure mode
- reran the focused and required backlog verification gates with fresh passing
  evidence
- reclassified the earlier historical completion claim for
  `minimum_subset_20260430T035104Z` as non-authoritative under the stricter
  validator because its TensorFlow `baseline` and `pinn` logs are duplicated
  shared-workflow captures
- started a fresh post-fix rerun in `minimum_subset_20260430T051928Z`, then
  stopped it after confirming the full benchmark rerun is follow-up execution
  work rather than part of this review-fix commit

## Current Contract State

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
  `fresh_rerun_required_after_tf_row_log_fix`
- bundle completeness:
  no current authoritative paper-grade root
- fixed sample ids:
  `0`, `1`
- shared visual-scale policy:
  stitched numeric arrays for amplitude/phase and derived shared absolute-error
  scales

## Root Status

- historical root downgraded under the current validator:
  `runs/minimum_subset_20260430T035104Z`
  - reason: TensorFlow `baseline` and `pinn` `stdout.log` files are duplicated
    shared-workflow captures, so the root no longer satisfies the stricter
    paper-grade TF provenance contract
- interrupted follow-up rerun:
  `runs/minimum_subset_20260430T051928Z`
  - reason: the fresh post-fix benchmark rerun was started successfully but
    stopped before completion because the full execution extends beyond a normal
    review-fix pass

## Verification

- focused review-fix regression suite:
  `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py tests/test_grid_lines_workflow.py tests/torch/test_grid_lines_torch_runner.py`
  -> `254 passed, 53 warnings in 44.74s`
- required deterministic gates:
  `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  -> `172 passed, 47 warnings in 303.97s`
- required compile gate:
  `python -m compileall -q ptycho_torch scripts/studies`
  -> exit `0`
- archived logs:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/focused_pytest_tf_provenance_fix_20260429.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/backlog_required_pytest_tf_provenance_fix_20260429.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/compileall_tf_provenance_fix_20260429.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/lines128_fresh_rerun_tf_provenance_fix_20260430T051928Z.log`

## Boundary And Remaining Scope

- this item does not yet restore a paper-grade minimum CDI subset root; the
  code fix is complete, but the benchmark rerun remains follow-up execution
  work
- `pinn_spectral_resnet_bottleneck_net` and `pinn_ffno` remain later
  out-of-scope rows
- the harness preflight note remains readiness-only authority; the required
  follow-up rerun remains governed by the checked-in minimum-subset execution
  note plus its machine-readable execution manifest
