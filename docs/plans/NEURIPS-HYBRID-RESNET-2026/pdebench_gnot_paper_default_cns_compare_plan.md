# PDEBench GNOT Paper-Default CNS Compare Plan

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Date: `2026-04-22`
- Status: queued backlog plan
- Scope: rerun `gnot_cns_base` on the existing local PDEBench `2d_cfd_cns`
  contract using the patched paper-style GNOT recipe, then compare it directly
  against the current spectral anchor

## Objective

Produce the second local GNOT baseline row for PDEBench CNS, this time using the
paper-style GNOT training recipe that is already patched into the repo, so the
comparison can answer whether the first fairness-probe result was mostly a
recipe mismatch or a deeper architecture/task mismatch.

## Fixed Local Contract

Keep the current local CNS compare contract fixed:

- task: `2d_cfd_cns`
- dataset:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- resolution: `128x128`
- history contract: `history_len=2`
- capped split: `512 / 64 / 64` trajectories
- `max_windows_per_trajectory=8`
- batch size: `4`
- reporting:
  - `comparison_summary.json`
  - `metrics_gnot_cns_base.json`
  - sample prediction/residual PNGs
  - train-split eval + held-out eval

## What Changes Relative To The First GNOT Compare

Only the GNOT training recipe changes.

The next row must use the patched `gnot_cns_base` defaults:

- `training_loss=relative_l2`
- `optimizer=AdamW`
- `learning_rate=1e-3`
- `weight_decay=5e-5`
- `scheduler=OneCycleLR`
- `gnot_hidden=128`

This is the core purpose of the rerun. Do not reintroduce the earlier local
MSE fairness recipe.

## Environment Contract

Use the already validated environment:

- conda env: `ptycho311_2`
- `torch 2.4.1+cu124`
- `dgl 2.4.0+cu124`

The active default repo env `ptycho311` is not sufficient for CUDA GNOT and
should not be used for this run.

## Primary Comparison Surface

The mandatory comparison target is:

- `spectral_resnet_bottleneck_base`

Direct numerical comparison should answer:

1. did paper-default GNOT improve materially over the first local fairness-probe run
2. is GNOT still clearly behind the local spectral row on aggregate error
3. is the failure mode still mainly low-frequency/global structure error

Secondary anchors such as `fno_base`, `unet_strong`, or `hybrid_resnet_cns` may
be mentioned only if the summary pins their exact artifact roots and epoch
budget explicitly.

## In Scope

- preflight the GNOT runtime environment in `ptycho311_2`
- run `gnot_cns_base` for `40` epochs on the capped CNS slice
- compare against the current spectral anchor on the same local contract
- render prediction and residual galleries
- update the durable summary and findings if the result changes the current read

## Out Of Scope

- new GNOT adapter work
- changing the CNS dataset contract
- switching to Markov `history_len=1`
- changing resolution to match the paper's non-canonical CFD-2D setup
- claiming paper-faithful GNOT reproduction beyond the fixed local contract

## Acceptance Criteria

1. a `40`-epoch `gnot_cns_base` run exists under a fresh timestamped output root
2. the run completed in `ptycho311_2` with exit code `0`
3. standard CNS metric artifacts exist and are fresh
4. the summary states whether paper-default GNOT materially changed the first conclusion
5. the result is discoverable from durable docs

## Suggested Checks

- `python -m pytest tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py -q`
- `python -m compileall -q scripts/studies/pdebench_image128`

