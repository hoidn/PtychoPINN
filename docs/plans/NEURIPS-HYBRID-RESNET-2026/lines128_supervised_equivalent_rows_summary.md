# NeurIPS Lines128 Supervised-Equivalent Rows Summary

- Date: `2026-04-30`
- Backlog item: `2026-04-29-cdi-lines128-supervised-equivalent-rows`
- State: `paper_complete`
- Claim boundary: `lines128_supervised_ffno_extension`
- Authoritative extension root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/supervised_ffno_extension_20260430T160218Z`

## Completed In This Pass

- extended the Lines128 Torch runner and compare-wrapper surfaces so the frozen
  `ffno` row can execute under an explicit `training_procedure=supervised`
  contract without regressing legacy architecture-mode PINN behavior
- added the deterministic `Y_I` / `Y_phi` to `label_amp` / `label_phase`
  bridge required by the existing supervised Lightning path on the locked
  `lines128` dataset split
- made the paper metrics-table surface label the pair explicitly as
  `FFNO + PINN` versus `FFNO + supervised`
- launched the required same-contract supervised FFNO run in tmux under the
  fixed `N=128`, `seed=3`, `40`-epoch, custom-probe contract and archived the
  launcher log at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/supervised_ffno_launch_20260430T160218Z.log`
- promoted the preserved `pinn_ffno` row into the new extension root,
  regenerated the pairwise table/visual bundle, and repaired the top-level
  wrapper invocation metadata so the final extension root validates as
  `paper_complete`

## Final Row Roster

- bundle rows:
  `pinn_ffno` -> `FFNO + PINN`
  `supervised_ffno` -> `FFNO + supervised`
- reference-only same-contract supervised row:
  `baseline` -> `CDI CNN + supervised`
  source:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T084339Z`
- preserved primary CDI benchmark root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`

## Extension Outcome

- the supervised FFNO row executed successfully under the frozen contract and
  produced a `paper_complete` adjacent extension root
- the extension’s central result is exact supervised/PINN FFNO parity on the
  locked Lines128 contract:
  - `recon.npz`, `history.json`, and `model.pt` are byte-for-byte identical
    between `supervised_ffno` and the preserved `pinn_ffno` row
  - the parity comparison standard is stronger than tolerance-based equality:
    SHA-256 identity plus `numpy.allclose(..., rtol=0.0, atol=0.0)` on the
    reconstruction arrays
- parity evidence is archived at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/execution/supervised_ffno_parity_audit.json`

## Verification

- bundle regeneration log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/supervised_ffno_bundle_20260430T162807Z.log`
- focused regression suite:
  `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py`
  -> `221 passed, 47 warnings in 42.13s`
  log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_focused_20260430T163023Z.log`
- required deterministic gate:
  `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  -> `177 passed, 47 warnings in 301.19s (0:05:01)`
  log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_required_20260430T163124Z.log`
- compile gate:
  `python -m compileall -q ptycho_torch scripts/studies`
  -> exit `0`
  log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/compileall_required_20260430T163124Z.log`

## Remaining Caveats

- the promoted `pinn_ffno` row remains recovered prerequisite evidence inside
  this extension root rather than a fresh rerun from this pass
- the extension is adjacent evidence only; it does not replace the preserved
  six-row primary CDI benchmark claim authority
- the verification logs still carry the known non-fatal `tight_layout`,
  `skimage` SSIM, and FRC warning set already seen on the related Lines128
  study surfaces
