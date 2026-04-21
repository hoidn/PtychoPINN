# PDEBench Author FFNO Equal-Footing CNS Compare Plan

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Date: `2026-04-21`
- Status: queued backlog plan
- Scope: run the authors' actual FFNO model on the same capped PDEBench `2d_cfd_cns` dataset slice and epoch budgets already used for `spectral_resnet_bottleneck_base`, `fno_base`, and `unet_strong`, so the comparison is on equal footing

## Objective

Add one official-author FFNO comparison row for PDEBench CNS with the same local
data slice, training-loss contract, epoch counts, and reporting surfaces as the
current capped CNS baselines.

The purpose is not to compare against the repo's FFNO-close bottleneck proxy.
The purpose is to compare against the actual author FFNO implementation under a
fair local protocol.

## Equal-Footing Contract

The target comparison contract is the existing capped CNS setup:

- task: `2d_cfd_cns`
- dataset:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- split:
  `512 / 64 / 64` trajectories
- per-trajectory cap:
  `8` windows
- history contract:
  `history_len=2`, `concat u[t-2:t] -> u[t]`
- training loss:
  `mse`
- batch size:
  `4`
- epoch budgets:
  `10` and `40`
- reporting:
  `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low/mid/high`,
  sample prediction PNGs, and comparison galleries

If the authors' implementation cannot be run under this contract without a
materially different data or optimization regime, the work should stop and
record the incompatibility explicitly rather than quietly claiming a fair
compare.

## In Scope

- identify the authoritative author FFNO code path to use
- document what "actual model from the authors" means in this repo context
- adapt or wrap the author code enough to accept the local CNS supervision unit
- run capped `10`-epoch and `40`-epoch CNS jobs on the equal-footing contract
- compare against the existing local rows:
  - `spectral_resnet_bottleneck_base`
  - `fno_base`
  - `unet_strong`
  - optionally the earlier `hybrid_resnet_base` row for continuity
- publish metrics and PNG galleries into the same discoverable docs surfaces

## Out Of Scope

- replacing the current local `fno_base`
- redefining the PDEBench CNS benchmark gate
- claiming paper-faithful reproduction beyond the exact local contract used here
- changing the data split or loss just to suit the imported FFNO code

## Main Risks

1. The authors' code may assume a different PDEBench preprocessing or rollout
   contract than the local one-step CNS setup.
2. The official implementation may depend on a package or repo state that is
   not currently vendored here.
3. The author code may use a different optimizer/scheduler recipe, creating a
   fairness dispute unless the deviation is documented explicitly.
4. The model may be substantially heavier or lighter than the local FNO row,
   which is fine, but that difference needs to be recorded rather than blurred.

## Acceptance Criteria

1. The official-author FFNO source and version are recorded in a durable doc.
2. A local run exists at `10` epochs on the equal-footing CNS slice.
3. A local run exists at `40` epochs on the same slice.
4. The result is compared directly against `spectral_resnet_bottleneck_base`,
   `fno_base`, and `unet_strong` with the same reported metric family.
5. Any incompatibility or fairness caveat is written explicitly in the summary.

## Suggested Checks

- `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py`
- `python -m compileall -q scripts/studies/pdebench_image128`

