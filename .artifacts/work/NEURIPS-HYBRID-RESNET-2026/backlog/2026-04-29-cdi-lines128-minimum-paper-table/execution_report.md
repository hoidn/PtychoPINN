## Completed In This Pass

- fixed the review-blocking bundle contract gaps in the checked-in code:
  paper-grade provenance gating, emitted validation-loss propagation,
  wrapper-level `paper_benchmark_manifest.json`, and schema-stable Torch
  row-local `config.json` / `metrics.json`
- reran the minimum-subset collation in place against the existing
  authoritative root
  `runs/minimum_subset_20260429T235811Z` with
  `--reuse-existing-recons`, without retraining any row
- regenerated the merged bundle so every required row now carries complete
  invocation/config/git/environment/dataset/split/randomness/output/visual
  provenance and emitted validation loss in the authoritative root

## Completed Current-Scope Work

- current authoritative root:
  `runs/minimum_subset_20260429T235811Z`
- final bundle state:
  `benchmark_status=paper_complete`,
  `claim_boundary=minimum_draftable_cdi_subset`
- required rows now all report `row_status=paper_grade` with empty
  `missing_fields_by_row`:
  `baseline`, `pinn`, `pinn_hybrid_resnet`, `pinn_fno_vanilla`
- required merged artifacts are freshly written in the authoritative root:
  `metrics.json`, `metric_schema.json`, `model_manifest.json`,
  `metrics_table.csv`, `metrics_table.tex`, `metrics_table_best.tex`,
  `paper_benchmark_manifest.json`
- required visuals now exist in the authoritative root:
  `compare_amp_phase.png`, per-row `amp_phase_*.png`,
  per-row `amp_phase_error_*.png`, and `frc_curves.png`
- row-local provenance now exists for every required row, including the
  synthesized Torch `runs/pinn_hybrid_resnet/config.json` and
  `runs/pinn_fno_vanilla/config.json` recovered honestly from the saved
  invocation artifacts in the same authoritative root
- emitted validation loss now propagates into the merged paper bundle:
  - `baseline`: `0.09419603645801544`
  - `pinn`: `10.800644874572754`
  - `pinn_hybrid_resnet`: `0.037633031606674194`
  - `pinn_fno_vanilla`: `0.07295490801334381`
- verification:
  - `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_workflow.py`
    -> `248 passed, 53 warnings in 41.74s`
  - `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
    -> `171 passed, 47 warnings in 298.38s`
  - `python -m compileall -q ptycho_torch scripts/studies`
    -> exit `0`
  - same-root recovery command:
    `python scripts/studies/lines128_paper_benchmark.py --mode minimum_subset --decision-artifact .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json --execution-authority-note docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md --execution-manifest .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json --output-dir .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T235811Z --reuse-existing-recons`
    -> exit `0`

## Follow-Up Work

- later complete-table CDI rows `pinn_spectral_resnet_bottleneck_net` and
  `pinn_ffno` remain intentionally out of scope for this minimum-subset item
- if the project wants recovery-path parameter counts to preserve the original
  fresh-run semantics rather than the recovered row-local model payload values,
  that contract should be pinned explicitly in a later follow-up; it is not
  needed for this minimum-subset approval because the current authoritative
  bundle and durable summary are internally consistent again

## Residual Risks

- this item still closes only the minimum draftable CDI subset, not the later
  complete `lines128` paper table
- the authoritative bundle now depends on honest same-root recovery of the
  existing fresh root; deleting those row-local artifacts would require another
  full rerun under the frozen contract
- future reproductions still depend on the recorded local runtime stack
  (`ptycho311`, local GPU, local TF/Torch installs); the new manifest reduces
  ambiguity but does not remove host dependency
