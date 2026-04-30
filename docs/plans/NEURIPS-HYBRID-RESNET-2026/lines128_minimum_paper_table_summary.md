# NeurIPS Lines128 Minimum Paper Table Summary

- Date: `2026-04-29`
- Backlog item: `2026-04-29-cdi-lines128-minimum-paper-table`
- State: `paper_complete`
- Chosen execution path: `fresh_rerun_after_review_fix`
- Authoritative root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T035104Z`

## Completed In This Pass

- fixed the review-blocking provenance gate so paper-grade validation now
  rejects synthetic reused-row proofs and requires completed invocation
  metadata plus real row-local `stdout.log` and `stderr.log`
- updated fresh-run collation so TensorFlow and Torch rows emit honest
  row-local logs and exit-code proof payloads, and TensorFlow row provenance
  records completed invocation state instead of placeholder outputs
- rejected the earlier same-root recovery claim for
  `minimum_subset_20260429T235811Z` because it depended on fabricated
  `--reuse-existing-recons` artifacts
- completed a fresh rerun in `minimum_subset_20260430T035104Z`, then validated
  the accepted root as `paper_complete` with empty `missing_fields_by_row` and
  empty `missing_bundle_artifacts`

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
- root-level provenance artifacts now exist:
  - `invocation.json`
  - `invocation.sh`
  - `dataset_identity_manifest.json`
  - `split_manifest.json`
  - `live_stdout.log`
  - `live_stderr.log`
- row-local provenance now exists for every required row:
  - `runs/baseline/`: `invocation.json`, `invocation.sh`, `config.json`,
    `history.json`, `metrics.json`, `stdout.log`, `stderr.log`,
    `exit_code_proof.json`
  - `runs/pinn/`: `invocation.json`, `invocation.sh`, `config.json`,
    `history.json`, `metrics.json`, `stdout.log`, `stderr.log`,
    `exit_code_proof.json`
  - `runs/pinn_hybrid_resnet/`: `invocation.json`, `invocation.sh`,
    `config.json`, `history.json`, `metrics.json`, `stdout.log`,
    `stderr.log`, `exit_code_proof.json`, `model.pt`,
    `randomness_contract.json`
  - `runs/pinn_fno_vanilla/`: `invocation.json`, `invocation.sh`,
    `config.json`, `history.json`, `metrics.json`, `stdout.log`,
    `stderr.log`, `exit_code_proof.json`, `model.pt`,
    `randomness_contract.json`
- final row metadata in the completed bundle:
  - `baseline`: `parameter_count=4612418`, `final_completed_epoch=40`,
    `final_train_loss=0.08847878873348236`, `validation_loss=0.10440725088119507`
  - `pinn`: `parameter_count=4661212`, `final_completed_epoch=40`,
    `final_train_loss=10.244312286376953`, `validation_loss=10.230887413024902`
  - `pinn_hybrid_resnet`: `parameter_count=8962149`,
    `final_completed_epoch=40`,
    `final_train_loss=0.027469921857118607`,
    `validation_loss=0.037633031606674194`
  - `pinn_fno_vanilla`: `parameter_count=636452`,
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
  `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py tests/test_grid_lines_workflow.py tests/torch/test_grid_lines_torch_runner.py`
  -> `252 passed, 53 warnings in 46.35s`
- required deterministic gates:
  `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  -> `171 passed, 47 warnings in 301.98s`
- required compile gate:
  `python -m compileall -q ptycho_torch scripts/studies ptycho/workflows`
  -> exit `0`
- accepted fresh rerun command:
  `python scripts/studies/lines128_paper_benchmark.py --mode minimum_subset --decision-artifact .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json --execution-authority-note docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md --execution-manifest .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json --output-dir .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T035104Z`
  -> exit `0`
- archived logs:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/focused_pytest_review_fix_20260430b.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/backlog_required_pytest_review_fix_20260430b.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/compileall_review_fix_20260430b.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/lines128_fresh_rerun_20260430T035104Z.log`

## Boundary And Remaining Scope

- this item closes only the minimum draftable CDI subset and does not widen the
  result to the later complete `lines128` paper table
- `pinn_spectral_resnet_bottleneck_net` and `pinn_ffno` remain later
  out-of-scope rows
- the harness preflight note remains readiness-only authority; the accepted
  fresh rerun is governed by the checked-in minimum-subset execution note plus
  its machine-readable execution manifest
