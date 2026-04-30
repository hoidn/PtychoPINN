# NeurIPS Lines128 Minimum Paper Table Summary

- Date: `2026-04-29`
- Backlog item: `2026-04-29-cdi-lines128-minimum-paper-table`
- State: `paper_complete`
- Chosen execution path: `same_root_recovery`
- Authoritative root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T235811Z`

## Completed In This Pass

- fixed the review-blocking provenance gate so paper-grade validation now
  requires referenced row-local files to exist, not just non-empty path strings
- tightened same-root recovery so recovered rows promote to `paper_grade` only
  when `invocation.sh` and row visual PNGs exist alongside the earlier config,
  history, metrics, and reconstruction artifacts
- added wrapper/root bundle-artifact validation before the final merged result
  can remain `paper_complete`
- reused the existing authoritative root
  `minimum_subset_20260429T235811Z` and reran only the same-root bundle
  collation path with `--reuse-existing-recons`
- regenerated the merged bundle so every required row now reports complete
  provenance blocks, emitted validation loss, `row_status=paper_grade`, and an
  empty `missing_bundle_artifacts` list

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
- bundle completeness:
  `missing_bundle_artifacts=[]`
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
  - `paper_benchmark_manifest.json`
- row-local provenance now exists for every required row:
  - `runs/baseline/`: `invocation.json`, `invocation.sh`, `config.json`,
    `history.json`, `metrics.json`, `stdout.log`, `stderr.log`
  - `runs/pinn/`: `invocation.json`, `invocation.sh`, `config.json`,
    `history.json`, `metrics.json`, `stdout.log`, `stderr.log`
  - `runs/pinn_hybrid_resnet/` and `runs/pinn_fno_vanilla/` now also include
    recovered `config.json` alongside their Torch invocation, history, metrics,
    checkpoint, and randomness artifacts
- final row metadata in the completed bundle:
  - `baseline`: `parameter_count=4612418`, `final_completed_epoch=40`,
    `final_train_loss=0.07928212732076645`,
    `validation_loss=0.09419603645801544`
  - `pinn`: `parameter_count=4661212`, `final_completed_epoch=40`,
    `final_train_loss=10.780830383300781`,
    `validation_loss=10.800644874572754`
  - `pinn_hybrid_resnet`: `parameter_count=18006600`,
    `final_completed_epoch=40`,
    `final_train_loss=0.027469921857118607`,
    `validation_loss=0.037633031606674194`
  - `pinn_fno_vanilla`: `parameter_count=1272902`,
    `final_completed_epoch=40`,
    `final_train_loss=0.0719698816537857`,
    `validation_loss=0.07295490801334381`
- final required visuals in the authoritative root:
  - `compare_amp_phase.png`
  - per-row `amp_phase_*.png`
  - per-row `amp_phase_error_*.png`
  - `frc_curves.png`

## Verification

- focused review-fix regression suite:
  `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py`
  -> `196 passed, 47 warnings in 39.96s`
- required deterministic gates:
  `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  -> `171 passed, 47 warnings in 299.93s`
- required compile gate:
  `python -m compileall -q ptycho_torch scripts/studies`
  -> exit `0`
- same-root recovery command:
  `python scripts/studies/lines128_paper_benchmark.py --mode minimum_subset --decision-artifact .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json --execution-authority-note docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md --execution-manifest .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json --output-dir .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T235811Z --reuse-existing-recons`
  -> exit `0`
- archived logs:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/pytest_focused_review_fix_current.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/pytest_deterministic_gates_review_fix_current.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/compileall_review_fix_current.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/lines128_same_root_recovery_20260429T235811Z.log`

## Boundary And Remaining Scope

- this item closes only the minimum draftable CDI subset and does not widen the
  result to the later complete `lines128` paper table
- `pinn_spectral_resnet_bottleneck_net` and `pinn_ffno` remain later
  out-of-scope rows
- the harness preflight note remains readiness-only authority; the completed
  fresh rerun is governed by the checked-in minimum-subset execution note plus
  its machine-readable execution manifest
