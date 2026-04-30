## Completed In This Pass

- rejected the prior `same_root_recovery` promotion after verifying the review's
  blocking findings against the current root and switched the item back to the
  plan-authorized `fresh_rerun_required` path
- fixed the paper-bundle status contract so only `paper_grade` rows can produce
  `benchmark_status=paper_complete`, while recovered rows are labeled
  `decision_support`
- extended TensorFlow row execution to emit row-local `invocation`,
  `config`, `history`, and `metrics` artifacts, and expanded the final visual
  bundle to include absolute-error panels plus `frc_curves.png`
- reran the full four-row minimum subset into
  `runs/minimum_subset_20260429T235811Z` under the frozen contract, with the
  tracked tmux shell PID exiting `0`

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
  `metrics_table.csv`, `metrics_table.tex`, `metrics_table_best.tex`
- required visuals now exist in the authoritative root:
  `compare_amp_phase.png`, per-row `amp_phase_*.png`,
  per-row `amp_phase_error_*.png`, and `frc_curves.png`
- implementation-boundary deviation recorded:
  TensorFlow row-local provenance had to be emitted from
  `ptycho/workflows/grid_lines_workflow.py` because that workflow owns the
  actual TF histories, saved models, and row-local metrics needed for honest
  paper-grade artifacts; the compare wrapper alone did not have enough
  information to synthesize those files truthfully
- verification:
  - `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py`
    -> `190 passed, 45 warnings in 36.44s`
  - `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
    -> `169 passed, 45 warnings in 296.75s`
  - `python -m compileall -q ptycho_torch scripts/studies`
    -> exit `0`
  - fresh rerun command:
    `python scripts/studies/lines128_paper_benchmark.py --mode minimum_subset --decision-artifact .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json --execution-authority-note docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md --execution-manifest .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json --output-dir .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T235811Z`
    -> tracked PID `1211720`, exit `0`

## Follow-Up Work

- later complete-table CDI rows `pinn_spectral_resnet_bottleneck_net` and
  `pinn_ffno` remain intentionally out of scope for this minimum-subset item
- standalone Torch-run best-effort visual rendering still logs a non-fatal FRC
  parsing warning because row-local `metrics.json` currently serializes complex
  numeric payloads with `default=str`; the final compare-wrapper bundle is
  unaffected because it regenerates the authoritative visuals from the merged
  metrics root

## Residual Risks

- this item closes only the minimum draftable CDI subset, not the later
  complete `lines128` paper table
- the fresh rerun depends on the current local host/runtime stack
  (`ptycho311`, local GPU, local TF/Torch installs); future reproductions still
  need the recorded invocation/environment artifacts rather than assumptions
- disk pressure is lower than before but still relevant for future fresh roots;
  the earlier `minimum_subset_20260429T224103Z` failure remains the local proof
  that write-side reruns can fail under `OSError: [Errno 28] No space left on
  device`
