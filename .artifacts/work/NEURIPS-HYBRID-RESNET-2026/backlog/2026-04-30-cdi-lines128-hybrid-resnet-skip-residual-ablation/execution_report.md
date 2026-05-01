# Execution Report

## Completed In This Pass

- Added fixed-residual-scale support for Hybrid ResNet bottlenecks and threaded the new controls through config, model construction, workflow overrides, and the grid-lines Torch runner CLI.
- Added the dedicated `lines128_hybrid_resnet_skip_residual_ablation.py` study helper, prepared the execution scaffold, launched or recovered the required rows, and collated the final same-contract ablation bundle.
- Added focused generator/runner/study-helper tests, archived verification logs, and synchronized the durable summary and evidence indexes for the completed backlog item.

## Completed Plan Tasks

- Implement residual-scale and row-override plumbing required for the skip/residual ablation.
- Create the study helper and emit the scaffold artifacts, comparison bundle, and row-local provenance files.
- Execute the required fresh rows: `pinn_hybrid_resnet_skip_add`, `pinn_hybrid_resnet_residual_fixed`, and `pinn_hybrid_resnet_skip_add_residual_fixed`, while reusing the existing `pinn_hybrid_resnet` anchor.
- Collate the comparison outputs into `metrics.json`, `model_manifest.json`, `metrics_table.csv`, `metrics_table.tex`, and `comparison_summary.json`.
- Update the durable summary/index authorities for this decision-support family.

## Remaining Required Plan Tasks

- None for the current approved scope.
- Optional deferred follow-up only: `pinn_hybrid_resnet_skip_gated_add` if a later bounded plan wants to reopen this family.

## Verification

- `pytest -q tests/torch/test_fno_generators.py -k "hybrid_resnet or resnet_decoder_block or skip_style"`
- `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_skip or hybrid_resnet_blocks or resnet_width"`
- `pytest -q tests/studies/test_lines128_hybrid_resnet_skip_residual_ablation.py`
- `python -m compileall -q ptycho_torch scripts/studies`
- `pytest -v -m integration`
- Same-contract bundle collation completed successfully; the bundle intentionally remains `decision_support` with `benchmark_status: "benchmark_incomplete"` because none of the fresh rows are promoted to `paper_grade`.
- Archived logs live under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation/verification/`.

## Residual Risks

- The conclusions are based on the frozen `lines128`, `seed=3`, `40`-epoch contract with a two-image test split, so they should be treated as local decision-support rather than broad architecture claims.
- Skip-add and fixed residual scale pull in different directions: skip-add helps the phase side most, while fixed residual scale helps the amplitude side most. The family does not supply a clean replacement for the existing paper-grade Hybrid anchor.
