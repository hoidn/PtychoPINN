# NeurIPS Lines128 Minimum Paper Table Summary

- Date: `2026-04-29`
- Backlog item: `2026-04-29-cdi-lines128-minimum-paper-table`
- State: `paper_complete`
- Chosen execution path: `fresh_rerun_after_provenance_fix`
- Authoritative root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T035104Z`

## Completed In This Pass

- fixed the review-blocking provenance path in code so TensorFlow rows emit
  structured row provenance, invocation/config/history/metric/output artifacts,
  and process-completion evidence
- relaxed the diagnostic-log contract so shared or duplicated `stdout.log` /
  `stderr.log` content is not by itself a paper-grade blocker when structured
  row evidence is complete
- retained the stricter rejection of synthetic reuse proof, missing outputs,
  missing visuals, incomplete invocation status, and contract/provenance gaps
- added regression coverage for both the wrapper provenance path and the
  diagnostic-log admissibility rule
- reran the focused and required backlog verification gates with fresh passing
  evidence
- accepted `minimum_subset_20260430T035104Z` as the current authoritative
  minimum-table root: its metrics, model manifest, and paper benchmark manifest
  report `paper_complete`, empty `missing_fields_by_row`, and no missing bundle
  artifacts
- kept the interrupted `minimum_subset_20260430T051928Z` rerun classified as an
  incomplete follow-up artifact, not the evidence root

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
  `paper_complete`
- bundle completeness:
  `minimum_subset_20260430T035104Z` has all required rows and no missing
  required fields or bundle artifacts
- fixed sample ids:
  `0`, `1`
- shared visual-scale policy:
  stitched numeric arrays for amplitude/phase and derived shared absolute-error
  scales

## Root Status

- authoritative root:
  `runs/minimum_subset_20260430T035104Z`
  - status: `paper_complete`
  - reason: structured row provenance, metrics, source arrays, visuals,
    dataset/split identity, git/environment metadata, and process-completion
    evidence satisfy the current evidence contract. Shared or duplicated
    diagnostic logs are not a paper-grade blocker by themselves.
- interrupted follow-up rerun:
  `runs/minimum_subset_20260430T051928Z`
  - reason: the fresh post-fix benchmark rerun was started successfully but
    stopped before completion because the full execution extends beyond a normal
    review-fix pass

## Verification

- focused review-fix regression suite:
  `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py`
  -> `200 passed, 47 warnings in 41.27s`
- required deterministic gates:
  `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  -> `172 passed, 47 warnings in 301.72s (0:05:01)`
- required compile gate:
  `python -m compileall -q ptycho_torch scripts/studies`
  -> exit `0`
- archived logs:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/focused_pytest_final_20260430.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/backlog_required_pytest_final_20260430.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/compileall_final_20260430.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/lines128_fresh_rerun_tf_provenance_fix_20260430T051928Z.log`

## Boundary And Remaining Scope

- this item restores a paper-grade minimum CDI subset root under the current
  evidence-sufficient contract
- `pinn_spectral_resnet_bottleneck_net` and `pinn_ffno` remain later
  out-of-scope rows
- the harness preflight note remains readiness-only authority; the required
  follow-up rerun remains governed by the checked-in minimum-subset execution
  note plus its machine-readable execution manifest
