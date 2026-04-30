# Lines128 Supervised-Equivalent Protocol Compatibility Audit

- Date: `2026-04-30`
- Backlog item: `2026-04-29-cdi-lines128-supervised-equivalent-rows`
- Audit status: `executed_same_contract_parity`

## Contract Check

- Locked dataset/split/probe/sample/metric contract remains the frozen `lines128` `N=128`,
  `gridsize=1`, `set_phi=True`, custom Run1084 probe, `pad_extrapolate`, fixed `seed=3`,
  `40`-epoch MAE Torch recipe already used by the authoritative minimum-subset and complete-table
  roots.
- The new row id is fixed as `supervised_ffno` with paper label `FFNO + supervised`.
- The reused same-architecture comparator is fixed as `pinn_ffno` from the authoritative
  complete-table root.
- The reused supervised CNN reference is fixed as `baseline` from the authoritative
  minimum-subset root.

## Original Gap

Before this pass, the row-local Torch execution path was not protocol-compatible for the required
same-contract supervised FFNO row because:

- `scripts/studies/grid_lines_torch_runner.py` hard-wired Torch rows to `training_procedure=pinn`
  and `model_type='pinn'`
- the grid-lines Torch study path did not bridge the locked dataset’s `Y_I` / `Y_phi` tensors into
  the supervised dataloader contract keys `label_amp` / `label_phase`
- `scripts/studies/grid_lines_compare_wrapper.py` could not route a distinct supervised FFNO model
  id while preserving the existing `ffno` PINN row
- metrics-table display labels did not have a stable `supervised_ffno` paper label

## Narrow Fix Applied

- Added explicit `training_procedure` support to the Torch runner and compare-wrapper routing.
- Added the deterministic `Y_I` / `Y_phi` -> `label_amp` / `label_phase` bridge for the locked
  grid-lines dataset path.
- Preserved legacy architecture-mode behavior as PINN-only while adding explicit
  `supervised_ffno` model-id routing.
- Added `supervised_ffno` display-label support to the paper metrics-table surface.

## Verification Status

- Focused regression surface is green after the narrow fix and final label
  update:
  - `tests/torch/test_grid_lines_torch_runner.py`
  - `tests/test_grid_lines_compare_wrapper.py`
  - `tests/studies/test_metrics_tables.py`
  - `tests/studies/test_lines128_paper_benchmark.py`
  - archived log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_focused_20260430T163023Z.log`
- Required deterministic gate is green:
  - archived log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_required_20260430T163124Z.log`
- Compile gate is green:
  - archived log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/compileall_required_20260430T163124Z.log`
- Launch completed successfully under tmux:
  - archived log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/supervised_ffno_launch_20260430T160218Z.log`
- Pairwise extension bundle is `paper_complete`:
  - authoritative root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/supervised_ffno_extension_20260430T160218Z`
  - bundle regeneration log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/supervised_ffno_bundle_20260430T162807Z.log`
- Exact parity between `supervised_ffno` and preserved `pinn_ffno` is archived
  at:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/execution/supervised_ffno_parity_audit.json`

## Final Outcome

`executed_same_contract_parity`

The supervised FFNO row ran successfully under the frozen `lines128` contract,
the adjacent extension root validates as `paper_complete`, and the resulting
`FFNO + supervised` artifacts are exactly identical to the preserved
`FFNO + PINN` row under the comparison standard recorded in the parity audit.
