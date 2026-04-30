# NeurIPS Lines128 Supervised-Equivalent Rows Summary

- Date: `2026-04-30`
- Backlog item: `2026-04-29-cdi-lines128-supervised-equivalent-rows`
- State: `paper_complete`
- Claim boundary: `lines128_supervised_ffno_extension`
- Authoritative extension root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/supervised_ffno_extension_20260430T170808Z`

## Completed In This Pass

- fixed the blocking supervised-mode handoff by mapping training-payload
  `model_type` overrides onto `PTModelConfig.mode`
- fixed the supervised FFNO forward path so the supervised loss call accepts
  `experiment_ids` once the corrected mode handoff activates the real
  supervised path
- added targeted regressions for the training-factory/workflow boundary and
  the supervised loss path so future `supervised_ffno` launches cannot silently
  fall back to `mode: Unsupervised`
- reran the required same-contract `supervised_ffno` row under the frozen
  `lines128` contract and archived the launcher log at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/supervised_ffno_launch_20260430T170808Z.log`
- promoted the preserved `pinn_ffno` comparator into the corrected root,
  replayed the compare-wrapper recovery path in tmux, and rebuilt the adjacent
  bundle, manifest, and comparison audit from the corrected artifacts

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

- the corrected `supervised_ffno` row is genuinely supervised:
  `lightning_logs/version_0/hparams.yaml` in the authoritative extension root
  records `mode: Supervised`, `loss_function: MAE`, and `architecture: ffno`
- the adjacent two-row extension validates as `paper_complete`, but the
  corrected comparison is not exact parity
- the comparison audit at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/execution/supervised_ffno_parity_audit.json`
  records `comparison_outcome: non_identical_same_contract_comparison`
- on the corrected bundle metrics, `supervised_ffno` is materially worse on
  amplitude reconstruction than the preserved `pinn_ffno` comparator while
  slightly improving phase MAE:
  - amplitude MAE: `0.5208796262741089` vs `0.062772475`
  - phase MAE: `0.04091365368331462` vs `0.08283866878648244`
  - amplitude SSIM: `0.23552695072779553` vs `0.934830339980703`
  - phase SSIM: `0.9460098057751931` vs `0.9815915191210483`

## Verification

- bundle regeneration log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/supervised_ffno_bundle_20260430T173326Z.log`
- targeted review regressions:
  `pytest -q tests/torch/test_loss_modes.py::test_supervised_compute_loss_accepts_experiment_ids tests/torch/test_config_factory.py::TestTrainingPayloadStructure::test_training_payload_maps_model_type_override_to_pt_mode tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_train_with_lightning_builds_supervised_model_config tests/scripts/test_training_backend_selector.py::TestTrainingCliBackendDispatch::test_supervised_mode_enforces_mae_loss`
  -> `4 passed, 4 warnings in 5.09s`
  log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_targeted_20260430T173424Z.log`
- focused regression suite:
  `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py`
  -> `221 passed, 47 warnings in 42.03s`
  log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_focused_20260430T173446Z.log`
- required deterministic gate:
  `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  -> `177 passed, 47 warnings in 300.83s (0:05:00)`
  log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_required_20260430T173548Z.log`
- compile gate:
  `python -m compileall -q ptycho_torch scripts/studies`
  -> exit `0`
  log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/compileall_required_20260430T174105Z.log`

## Remaining Caveats

- the extension remains adjacent evidence and does not replace the preserved
  six-row primary CDI benchmark claim authority
- the promoted `pinn_ffno` row is reused accepted evidence, not a fresh rerun
  from this pass
- verification logs still contain the known non-fatal `tight_layout`,
  `skimage` SSIM, and FRC warning set already seen on related Lines128 study
  surfaces
