# NeurIPS Lines128 Supervised-Equivalent Rows Summary

- Date: `2026-04-30`
- Backlog item: `2026-04-29-cdi-lines128-supervised-equivalent-rows`
- State: `paper_complete`
- Claim boundary: `lines128_supervised_ffno_extension`
- Authoritative extension root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/supervised_ffno_extension_20260430T180217Z`

## Completed In This Pass

- fixed the supervised FFNO correctness bug by wiring supervised Lightning
  runs to the configured generator module instead of always instantiating the
  legacy `Autoencoder`
- fixed the checkpoint persistence gap by rebuilding generator-backed Lightning
  modules from saved config state during `load_from_checkpoint()`, then added
  checkpoint round-trip regressions for supervised FFNO and the other
  generator-backed architectures
- fixed the compare-wrapper recovery path so current-root Torch rows can be
  rebuilt as `paper_grade` evidence even when the original direct-runner launch
  did not emit row-local `stdout.log` / `stderr.log`
- added targeted regressions for the supervised generator wiring, fresh
  current-root recovery semantics, recovered row-log enrichment, and direct
  wrapper import bootstrap
- refreshed the canonical `checkpoints/last.ckpt` in the authoritative
  extension root from the valid rerun checkpoint after verifying the review's
  exact reload path was still pointing at a stale pre-fix checkpoint file
- reran the required same-contract `supervised_ffno` row under the frozen
  `lines128` contract and archived the launcher log at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/supervised_ffno_launch_20260430T180217Z.log`
- promoted the preserved `pinn_ffno` comparator into the corrected root,
  replayed the compare-wrapper recovery path in tmux, and rebuilt the adjacent
  bundle, manifest, and comparison audit from the corrected artifacts

## Final Row Roster

- bundle rows:
  `pinn_ffno` -> `FFNO-local proxy + PINN`
  `supervised_ffno` -> `FFNO-local proxy + supervised`
- reference-only same-contract supervised row:
  `baseline` -> `CDI CNN + supervised`
  source:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T084339Z`
- preserved primary CDI benchmark root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`

## Extension Outcome

- the corrected `supervised_ffno` row is genuinely supervised:
  `lightning_logs/version_0/hparams.yaml` in the authoritative extension root
  records `mode: Supervised`, `architecture: ffno`, and
  `generator_output: real_imag`
- the authoritative checkpoint reload path is now also truthful again:
  `checkpoints/last.ckpt` in the authoritative extension root reloads as
  `FfnoGeneratorModule` under `PtychoPINN_Lightning.load_from_checkpoint()`
- the adjacent two-row extension validates as `paper_complete`, but the
  corrected comparison is not exact parity
- the comparison audit at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/execution/supervised_ffno_parity_audit.json`
  records `comparison_outcome: non_identical_same_contract_comparison`
- on the corrected bundle metrics, `supervised_ffno` is materially worse on
  amplitude reconstruction than the preserved `pinn_ffno` comparator while
  slightly improving phase MAE:
  - amplitude MAE: `0.38641318678855896` vs `0.0627724751830101`
  - phase MAE: `0.046562865303675434` vs `0.08283866878648244`
  - amplitude SSIM: `0.24842735671882815` vs `0.934830339980703`
  - phase SSIM: `0.9371787178129534` vs `0.9815915191210483`

## Verification

- bundle regeneration log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/supervised_ffno_bundle_20260430T180217Z.log`
- checkpoint reload regression log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_checkpoint_reload_20260430.log`
- real-artifact checkpoint reload proof:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/checkpoint_reload_real_artifact_20260430.log`
- targeted review regressions:
  `pytest -q tests/torch/test_generator_registry.py::test_ffno_generator_builds_supervised_lightning_model tests/test_grid_lines_compare_wrapper.py::test_recover_torch_row_payload_marks_current_root_rows_as_fresh tests/test_grid_lines_compare_wrapper.py::test_enrich_paper_row_payload_recovers_missing_direct_runner_logs tests/test_grid_lines_compare_wrapper.py::test_compare_wrapper_script_path_bootstraps_repo_imports`
  -> `4 passed in 7.88s`
  log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_targeted_20260430_supervised_equivalent_rows.log`
- required deterministic gate:
  `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_20260430_supervised_equivalent_rows_checkpoint_fix.log`
- compile gate:
  `python -m compileall -q ptycho_torch scripts/studies`
  -> exit `0`
  log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/compileall_20260430_supervised_equivalent_rows_checkpoint_fix.log`
- repo integration marker:
  `pytest -q -m integration`
  -> `5 passed, 4 skipped, 1748 deselected in 302.70s`
  log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_integration_20260430_supervised_equivalent_rows_checkpoint_fix.log`

## Remaining Caveats

- the extension remains adjacent evidence and does not replace the preserved
  six-row primary CDI benchmark claim authority
- the promoted `pinn_ffno` row is reused accepted evidence, not a fresh rerun
- post-hoc 2026-05-06 caveat: both FFNO rows used `fno_cnn_blocks=2`.
  Correct pure-FFNO objective-control evidence requires
  `2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun` and
  `2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun`.
  from this pass
- older non-canonical checkpoint files remain in the authoritative root for
  auditability, but `checkpoints/last.ckpt` is now the intended reload path
- verification logs still contain the known non-fatal `tight_layout`,
  `skimage` SSIM, FRC, and TensorFlow Addons warning set already seen on
  related Lines128 study and repo integration surfaces
