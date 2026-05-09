# PDEBench CNS Rollout Checkpoint Refresh Summary

- Date: `2026-05-08`
- Backlog item: `2026-05-08-cns-rollout-checkpoint-refresh`
- State: `completed`
- Claim boundary: `bounded_capped_decision_support_only`
- Item root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-08-cns-rollout-checkpoint-refresh/`
- Governing plan:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-08-cns-rollout-checkpoint-refresh/execution_plan.md`

## Objective

Recover real trained checkpoints for the current manuscript-facing PDEBench CNS
matched-condition rows and export deterministic density rollout GIFs from those
exact checkpoints without replacing the frozen matched-condition table
authority.

## Fixed Contract

- task: `2d_cfd_cns`
- spatial size: `128x128`
- dataset:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- `history_len=5`
- split counts: `512 / 64 / 64`
- `max_windows_per_trajectory=8`
- `epochs=40`
- `batch_size=4`
- training loss: `mse`
- metric family:
  `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`,
  `fRMSE_high`

## Claim-Boundary Reconciliation

The live CNS pilot producer still emits
`comparison_summary.json:evidence_scope=capped_decision_support_only`, while
the frozen matched-condition table authority is documented as
`bounded_capped_decision_support_only`. This implementation pass uses the
plan-approved normalization rule for audit only: producer-side
`capped_decision_support_only` is treated as the exact synonym of
authority-side `bounded_capped_decision_support_only` for same-contract
matched-condition checkpoint recovery. No producer string was changed in this
backlog item.

## Rollout Window Decision

The locked matched-condition dataset reports `time_steps=21` in the inspect
manifest. Under `history_len=5`, the maximum valid autoregressive rollout span
from the earliest valid start is therefore `16` steps, not the planned `20`.
This summary records the narrow implementation deviation used for all rollout
exports:

- split: `test`
- sample id: `0`
- trajectory id: `7823`
- start time: `5`
- steps: `16`
- field: `density`
- include error panels: `true`
- device: `cpu`

This deviation changes only the rollout length needed to satisfy the real
dataset guard; it does not change the training contract or the frozen table
authority.

## Completed In This Pass

- Ran the required preflight checks:
  - `pytest -q tests/studies/test_pdebench_image128_rollout_video.py`
  - `pytest -q tests/studies/test_pdebench_image128_runner.py -k "cfd_cns or model_state or matched_condition"`
  - `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
- Ran the required exact-contract inspect pass at
  `tmp/cns_rollout_checkpoint_refresh_inspect/`, confirming the locked
  `history_len=5`, `512 / 64 / 64`, `batch_size=4`,
  `max_windows_per_trajectory=8` contract.
- Confirmed by code audit and tests that the CNS runner writes
  `model_state_<row_id>.pt` plus `model_state_<row_id>.json`, and that the
  rollout loader rejects one-step NPZ-only roots.
- Hit and recovered one narrow launcher failure on the first author-FFNO
  attempt: precreating a `logs/` directory under the target run root caused the
  runner to reject the non-empty output root. The row was relaunched on a fresh
  timestamped run root with launcher logs stored outside the run root.
- Completed the fresh same-contract rerun for `author_ffno_cns_base` at:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-08-cns-rollout-checkpoint-refresh/runs/author_ffno_cns_base_20260508T223449Z`
- Exported the corresponding density rollout GIF and manifests at:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-08-cns-rollout-checkpoint-refresh/rollouts/author_ffno_cns_base/`
- Completed the fresh same-contract rerun for
  `spectral_resnet_bottleneck_base` at:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-08-cns-rollout-checkpoint-refresh/runs/spectral_resnet_bottleneck_base_20260508T235241Z`
- Exported the corresponding density rollout GIF and manifests at:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-08-cns-rollout-checkpoint-refresh/rollouts/spectral_resnet_bottleneck_base/`
- Completed the fresh same-contract rerun for `fno_base` at:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-08-cns-rollout-checkpoint-refresh/runs/fno_base_20260509T001339Z`
- Exported the corresponding density rollout GIF and manifests at:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-08-cns-rollout-checkpoint-refresh/rollouts/fno_base/`

## Row Audit

### `author_ffno_cns_base`

- Run root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-08-cns-rollout-checkpoint-refresh/runs/author_ffno_cns_base_20260508T223449Z`
- Launch proof:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-08-cns-rollout-checkpoint-refresh/launch/author_ffno_cns_base_20260508T223449Z/`
- Audit classification: `metric_compatible_with_current_authority`
- Claim-boundary reconciliation path:
  normalized producer-side `capped_decision_support_only` to authority-side
  `bounded_capped_decision_support_only`
- Delta versus frozen authority:
  - `err_RMSE`: `0.0`
  - `err_nRMSE`: `0.0`
  - `relative_l2`: `0.0`
  - `fRMSE_low`: `0.0`
  - `fRMSE_mid`: `0.0`
  - `fRMSE_high`: `0.0`
- Rounded table compatibility:
  exact numeric match and therefore exact compatibility with the current paper
  table rounding.

### `spectral_resnet_bottleneck_base`

- Run root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-08-cns-rollout-checkpoint-refresh/runs/spectral_resnet_bottleneck_base_20260508T235241Z`
- Launch proof:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-08-cns-rollout-checkpoint-refresh/launch/spectral_resnet_bottleneck_base_20260508T235241Z/`
- Audit classification: `visualization_lineage_only`
- Claim-boundary reconciliation path:
  normalized producer-side `capped_decision_support_only` to authority-side
  `bounded_capped_decision_support_only`
- Contract audit:
  profile config, split manifest, history length, split ids, training loss,
  optimizer, scheduler, learning rate, normalization, and horizon all matched
  the frozen authority inputs.
- Delta versus frozen authority:
  - `err_RMSE`: `+0.3758467436`
  - `err_nRMSE`: `+0.0155590586`
  - `relative_l2`: `+0.0155590586`
  - `fRMSE_low`: `+0.9103852510`
  - `fRMSE_mid`: `+0.0100345165`
  - `fRMSE_high`: `+0.0217032731`
- Rounded table compatibility:
  incompatible with the current paper table; every published metric column
  would change after rounding.

### `fno_base`

- Run root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-08-cns-rollout-checkpoint-refresh/runs/fno_base_20260509T001339Z`
- Launch proof:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-08-cns-rollout-checkpoint-refresh/launch/fno_base_20260509T001339Z/`
- Audit classification: `visualization_lineage_only`
- Claim-boundary reconciliation path:
  normalized producer-side `capped_decision_support_only` to authority-side
  `bounded_capped_decision_support_only`
- Contract audit:
  split manifest, history length, split ids, profile config, training loss,
  optimizer, scheduler, learning rate, normalization, and horizon all matched
  the frozen authority inputs.
- Delta versus frozen authority:
  - `err_RMSE`: `+0.0527966022`
  - `err_nRMSE`: `+0.0021856390`
  - `relative_l2`: `+0.0021856390`
  - `fRMSE_low`: `+0.1259384155`
  - `fRMSE_mid`: `+0.0017085224`
  - `fRMSE_high`: `+0.0139194429`
- Rounded table compatibility:
  incompatible with the current paper table; the rounded `nRMSE`,
  `Low RMSE`, `Mid RMSE`, and `High RMSE` values all change.

## Result And Manuscript Recommendation

This backlog item succeeded as checkpoint-and-rollout recovery, not as a new
matched-condition table refresh.

- `author_ffno_cns_base` reproduced the frozen authority exactly and is both
  checkpoint-valid and table-compatible.
- `spectral_resnet_bottleneck_base` preserved the same contract and split ids
  but drifted materially enough that it is rollout-lineage evidence only.
- `fno_base` preserved the same contract and split ids but still drifted enough
  to change rounded table values, so it is also rollout-lineage evidence only.

The manuscript recommendation therefore remains unchanged:

- keep
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.{json,csv,tex}`
  as the current frozen headline table
- use this backlog item's run roots and rollout roots as manuscript-adjacent
  checkpoint/visual lineage only
- leave `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  unchanged because no current downstream consumer needs machine-readable
  row-to-checkpoint or row-to-rollout lookup for this backlog item

## Table Authority Decision

The current paper-local CNS table remains unchanged in this backlog item. Fresh
same-contract reruns are used for checkpoint lineage and rollout export first;
they do not automatically replace:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.{json,csv,tex}`

## Verification

- Author row launcher exit code: `0`
- Author row required artifacts present:
  `model_state_author_ffno_cns_base.pt`,
  `model_state_author_ffno_cns_base.json`,
  `model_profile_author_ffno_cns_base.json`,
  `metrics_author_ffno_cns_base.json`,
  `split_manifest.json`,
  `normalization_stats_state.json`,
  `invocation.json`,
  `invocation.sh`
- Author rollout verification:
  - GIF exists
  - frame count `16`
  - first frame nonblank
  - rollout manifest names the correct run root, checkpoint, row id, split,
    sample id, trajectory id, start time, and step count
- Spectral row launcher exit code: `0`
- Spectral row required artifacts present:
  `model_state_spectral_resnet_bottleneck_base.pt`,
  `model_state_spectral_resnet_bottleneck_base.json`,
  `model_profile_spectral_resnet_bottleneck_base.json`,
  `metrics_spectral_resnet_bottleneck_base.json`,
  `split_manifest.json`,
  `normalization_stats_state.json`,
  `invocation.json`,
  `invocation.sh`
- Spectral rollout verification:
  - GIF exists
  - frame count `16`
  - first frame nonblank
  - rollout manifest names the correct run root, checkpoint, row id, split,
    sample id, trajectory id, start time, and step count
- FNO row launcher exit code: `0`
- FNO row required artifacts present:
  `model_state_fno_base.pt`,
  `model_state_fno_base.json`,
  `model_profile_fno_base.json`,
  `metrics_fno_base.json`,
  `split_manifest.json`,
  `normalization_stats_state.json`,
  `invocation.json`,
  `invocation.sh`
- FNO rollout verification:
  - GIF exists
  - frame count `16`
  - first frame nonblank
  - rollout manifest names the correct run root, checkpoint, row id, split,
    sample id, trajectory id, start time, and step count

## Residual Risks

- The spectral rerun preserved the expected contract but diverged materially
  from the frozen authority metrics, and the FNO rerun also drifted enough to
  change rounded table values. This summary records both rows honestly as
  rollout-lineage evidence only; this backlog item does not attempt another
  authority-refresh loop.
- The rollout exporter currently requires module-style invocation with
  `PYTHONPATH=/home/ollie/Documents/PtychoPINN python -m ...` in this
  environment because direct script execution did not resolve the
  `scripts.studies...` package path. This is an execution-path quirk, not yet a
  code patch.
