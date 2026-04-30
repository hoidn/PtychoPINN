# NeurIPS Lines128 Minimum Paper Table Summary

- Date: `2026-04-29`
- Backlog item: `2026-04-29-cdi-lines128-minimum-paper-table`
- State: `paper_complete`
- Chosen execution path: `fresh_rerun_required`
- Authoritative root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T235811Z`

## Completed In This Pass

- revalidated the implementation review findings and rejected the earlier
  `minimum_subset_20260429T213028Z` same-root promotion because it did not meet
  the required row-local provenance or visual-bundle contract
- tightened the paper-bundle status contract so only `paper_grade` rows can
  produce `benchmark_status=paper_complete`; recovered rows are now labeled
  `decision_support`
- extended the TensorFlow grid-lines workflow to emit row-local
  `invocation`, `config`, `history`, and `metrics` artifacts for `baseline` and
  `pinn`
- expanded the final visual bundle to include per-row absolute-error panels and
  `frc_curves.png`
- reran the full four-row minimum subset into a brand-new root and verified the
  tracked tmux shell PID exited `0`

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
- row-local provenance now exists for every required row:
  - `runs/baseline/`: `invocation.json`, `invocation.sh`, `config.json`,
    `history.json`, `metrics.json`, `stdout.log`, `stderr.log`
  - `runs/pinn/`: `invocation.json`, `invocation.sh`, `config.json`,
    `history.json`, `metrics.json`, `stdout.log`, `stderr.log`
  - `runs/pinn_hybrid_resnet/` and `runs/pinn_fno_vanilla/` retain their Torch
    invocation, history, metrics, checkpoint, and randomness artifacts
- final row metadata in the completed bundle:
  - `baseline`: `parameter_count=4612418`, `final_completed_epoch=40`,
    `final_train_loss=0.07900754362344742`
  - `pinn`: `parameter_count=4661212`, `final_completed_epoch=40`,
    `final_train_loss=6.772934436798096`
  - `pinn_hybrid_resnet`: `parameter_count=18006600`,
    `final_completed_epoch=40`,
    `final_train_loss=0.027469921857118607`
  - `pinn_fno_vanilla`: `parameter_count=1272902`,
    `final_completed_epoch=40`,
    `final_train_loss=0.0719698816537857`
- final required visuals in the authoritative root:
  - `compare_amp_phase.png`
  - per-row `amp_phase_*.png`
  - per-row `amp_phase_error_*.png`
  - `frc_curves.png`

## Verification

- focused review-fix regression suite:
  `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py`
  -> `190 passed, 45 warnings in 36.44s`
- required deterministic gates:
  `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  -> `169 passed, 45 warnings in 296.75s`
- required compile gate:
  `python -m compileall -q ptycho_torch scripts/studies`
  -> exit `0`
- fresh rerun command:
  `python scripts/studies/lines128_paper_benchmark.py --mode minimum_subset --decision-artifact .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json --execution-authority-note docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md --execution-manifest .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json --output-dir .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T235811Z`
  -> tracked tmux shell PID `1211720`, exit `0`
- archived logs:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/pytest_focused_review_fix.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/pytest_deterministic_gates_review_fix.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/compileall_review_fix.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/lines128_fresh_rerun_20260429T235811Z.log`

## Boundary And Remaining Scope

- this item closes only the minimum draftable CDI subset and does not widen the
  result to the later complete `lines128` paper table
- `pinn_spectral_resnet_bottleneck_net` and `pinn_ffno` remain later
  out-of-scope rows
- the harness preflight note remains readiness-only authority; the completed
  fresh rerun is governed by the checked-in minimum-subset execution note plus
  its machine-readable execution manifest
