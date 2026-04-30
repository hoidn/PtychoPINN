# Lines128 Supervised-Equivalent Protocol Compatibility Audit

- Date: `2026-04-30`
- Backlog item: `2026-04-29-cdi-lines128-supervised-equivalent-rows`
- Audit status: `executed_same_contract_comparison`

## Contract Check

- the locked dataset/split/probe/sample/metric contract remains the frozen
  `lines128` `N=128`, `gridsize=1`, `set_phi=True`, custom Run1084 probe,
  `pad_extrapolate`, fixed `seed=3`, `40`-epoch MAE Torch recipe already used
  by the authoritative minimum-subset and complete-table roots
- the new row id remains `supervised_ffno` with paper label
  `FFNO + supervised`
- the reused same-architecture comparator remains `pinn_ffno` from the
  authoritative complete-table root
- the reused supervised CNN reference remains `baseline` from the authoritative
  minimum-subset root

## Reviewed Failure Mode

The delivered extension was not protocol-correct because the claimed
`supervised_ffno` row silently trained in unsupervised PINN mode.

- `_train_with_lightning()` passed `model_type='Supervised'` into the training
  payload build
- `create_training_payload()` updated `PTModelConfig` directly, but
  `PTModelConfig` exposes `mode`, not `model_type`, so the override was
  ignored and the effective mode stayed `Unsupervised`
- once the mode handoff was corrected, the real supervised path exposed a
  second compatibility defect: `Ptycho_Supervised.forward(...)` did not accept
  the `experiment_ids` argument passed by the supervised loss path

## Narrow Fix Applied

- mapped training-payload override aliases from `model_type` to
  `PTModelConfig.mode` in `ptycho_torch/config_factory.py`
- updated `Ptycho_Supervised.forward(...)` in `ptycho_torch/model.py` to accept
  the supervised-path `experiment_ids` argument
- added regressions covering:
  - the training-factory alias bridge
  - the `_train_with_lightning()` supervised mode/loss handoff
  - the supervised FFNO loss path with `experiment_ids`

## Verification Status

- targeted review regressions are green:
  - `tests/torch/test_loss_modes.py::test_supervised_compute_loss_accepts_experiment_ids`
  - `tests/torch/test_config_factory.py::TestTrainingPayloadStructure::test_training_payload_maps_model_type_override_to_pt_mode`
  - `tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_train_with_lightning_builds_supervised_model_config`
  - `tests/scripts/test_training_backend_selector.py::TestTrainingCliBackendDispatch::test_supervised_mode_enforces_mae_loss`
  - archived log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_targeted_20260430T173424Z.log`
- focused regression surface is green:
  - archived log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_focused_20260430T173446Z.log`
- required deterministic gate is green:
  - archived log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_required_20260430T173548Z.log`
- compile gate is green:
  - archived log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/compileall_required_20260430T174105Z.log`
- supervised launch completed successfully under the frozen contract:
  - archived log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/supervised_ffno_launch_20260430T170808Z.log`
  - corrected run evidence:
    `lightning_logs/version_0/hparams.yaml` in the authoritative extension
    root records `mode: Supervised` and `loss_function: MAE`
- the rebuilt adjacent extension bundle is `paper_complete`:
  - authoritative root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/supervised_ffno_extension_20260430T170808Z`
  - bundle regeneration log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/supervised_ffno_bundle_20260430T173326Z.log`
- the corrected comparison audit is archived at:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/execution/supervised_ffno_parity_audit.json`

## Final Outcome

`executed_same_contract_comparison`

The supervised FFNO row now runs truthfully under the frozen `lines128`
contract, the adjacent extension root validates as `paper_complete`, and the
rebuilt comparison audit shows the corrected supervised row is not identical to
the preserved `FFNO + PINN` comparator.
