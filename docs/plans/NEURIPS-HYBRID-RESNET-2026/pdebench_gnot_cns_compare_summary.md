# NeurIPS Hybrid ResNet PDEBench GNOT CNS Compare Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-22-pdebench-gnot-paper-default-cns-compare`
- Date: `2026-04-23`
- Status: implementation complete; local GNOT integration, fairness-probe compare, fresh paper-default smoke gate, and paper-default `40`-epoch follow-up are all recorded
- Governing design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- Governing plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-22-pdebench-gnot-paper-default-cns-compare/execution_plan.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/`

This summary records four linked outcomes on the fixed local PDEBench
`2d_cfd_cns` capped contract:

1. official GNOT local CUDA hosting in `ptycho311_2`
2. the first capped equal-footing fairness probe against the local spectral row
3. a fresh paper-default smoke rerun proving the current checkout still emits
   the intended GNOT recipe
4. the same-day completed paper-default `40`-epoch follow-up anchored against
   the pinned spectral `40`-epoch row

All evidence here remains capped decision-support evidence only. It does not
create a benchmark-complete CNS ranking or a paper-faithful GNOT reproduction.

## Environment And Source

The active repo environment (`ptycho311`, `torch 2.9.1+cu128`) was not usable
for CUDA GNOT because no matching working CUDA DGL path was available there.

The working local compare environment is:

- env: `ptycho311_2`
- python: `3.11`
- `torch 2.4.1+cu124`
- `dgl 2.4.0+cu124`

The official GNOT source is pinned externally at:

- repo: `https://github.com/HaoZhongkai/GNOT`
- commit: `5ee2e6925a43f9a340a6016bad4da2c82a452cbe`
- local clone: [gnot](/home/ollie/Documents/PtychoPINN/.artifacts/external/gnot)
- provenance artifact:
  [gnot_source.json](/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/gnot_source.json)

## Implemented Surface

Local GNOT support for the PDEBench CNS runner lives in:

- [gnot_adapter.py](/home/ollie/Documents/PtychoPINN/scripts/studies/pdebench_image128/gnot_adapter.py)
- [models.py](/home/ollie/Documents/PtychoPINN/scripts/studies/pdebench_image128/models.py)
- [run_config.py](/home/ollie/Documents/PtychoPINN/scripts/studies/pdebench_image128/run_config.py)
- [cfd_cns.py](/home/ollie/Documents/PtychoPINN/scripts/studies/pdebench_image128/cfd_cns.py)

Required deterministic checks on the current checkout:

- `python -m pytest tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py -q`
- `python -m pytest tests/studies/test_pdebench_image128_runner.py::test_cfd_cns_gnot_profile_uses_paper_default_training_recipe -q`
- `python -m compileall -q scripts/studies/pdebench_image128`

Observed results in this pass:

- combined runner/model selector: `50 passed in 33.50s`
- targeted paper-default recipe selector: `1 passed in 5.57s`
- `compileall`: exit code `0`

## Fixed Local Contract

The fairness boundary stayed fixed throughout the fairness probe and the
paper-default follow-up:

- task: `2d_cfd_cns`
- dataset:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- resolution: `128x128`
- `history_len=2`
- trajectories: `512 / 64 / 64`
- train / val / test windows: `4096 / 512 / 512`
- `max_windows_per_trajectory=8`
- batch size: `4`
- reported metrics:
  `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`,
  `fRMSE_high`

Pinned comparison anchors:

- fairness-probe root:
  [cns-gnot-vs-spectral-cap512-10ep-20260422T200900Z](/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cns-gnot-vs-spectral-cap512-10ep-20260422T200900Z)
- paper-default smoke root:
  [gnot-paper-default-smoke-20260423T015239Z](/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/gnot-paper-default-smoke-20260423T015239Z)
- paper-default `40`-epoch run root:
  [cns-gnot-paper-default-40ep-20260422T214016Z](/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cns-gnot-paper-default-40ep-20260422T214016Z)
- pinned spectral `40`-epoch anchor:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`

## Fresh Paper-Default Smoke Gate

The fresh smoke rerun in this pass completed at:

- smoke root:
  [gnot-paper-default-smoke-20260423T015239Z](/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/gnot-paper-default-smoke-20260423T015239Z)

The smoke proved the current checkout still emits the intended paper-default
recipe:

- `training_loss=relative_l2`
- `optimizer=AdamW`
- `scheduler=OneCycleLR`
- `learning_rate=1e-3`
- `weight_decay=5e-5`

Observed smoke metrics:

- `err_nRMSE=1.3825286627`
- `relative_l2=1.3825286627`
- `fRMSE_low=39.5250282288`
- `fRMSE_high=0.0568131991`

Interpretation boundary:

- this smoke is a recipe and dependency-health gate only
- it does not support performance interpretation or model ranking

## First Equal-Footing Fairness Probe

The first completed GNOT row intentionally held the local CNS training recipe
fixed for fairness:

- run root:
  [cns-gnot-vs-spectral-cap512-10ep-20260422T200900Z](/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cns-gnot-vs-spectral-cap512-10ep-20260422T200900Z)
- epochs: `10`
- recipe:
  local `mse` / `ReduceLROnPlateau` CNS setup, not the paper-default GNOT
  recipe

Observed fairness-probe metrics:

- `gnot_cns_base`
  - `relative_l2=0.2456480712`
  - `err_RMSE=5.9365601540`
  - `fRMSE_low=14.1862068176`
  - `fRMSE_high=0.0177509338`
  - params: `551,815`
- `spectral_resnet_bottleneck_base`
  - `relative_l2=0.0859752074`
  - `err_RMSE=2.0777571201`
  - `fRMSE_high=0.6988957524`
  - params: `8,186,726`

Fairness-probe read:

- spectral beat GNOT badly on aggregate denormalized error
- GNOT kept much lower high-frequency error on the saved sample
- the dominant GNOT gap was low-frequency/global structure error

## Paper-Default `40`-Epoch Follow-Up

The same-day completed paper-default follow-up is recorded at:

- run root:
  [cns-gnot-paper-default-40ep-20260422T214016Z](/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cns-gnot-paper-default-40ep-20260422T214016Z)
- environment provenance:
  [preflight_env.json](/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cns-gnot-paper-default-40ep-20260422T214016Z/preflight_env.json)
- anchored compare sidecar:
  [compare_40ep_against_existing.json](/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cns-gnot-paper-default-40ep-20260422T214016Z/compare_40ep_against_existing.json)
  and
  [compare_40ep_against_existing.csv](/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cns-gnot-paper-default-40ep-20260422T214016Z/compare_40ep_against_existing.csv)

Verified paper-default recipe:

- `gnot_hidden=128`
- `training_loss=relative_l2`
- `optimizer=AdamW`
- `scheduler=OneCycleLR`
- `learning_rate=1e-3`
- `weight_decay=5e-5`

Observed `40`-epoch paper-default GNOT metrics:

- `err_RMSE=4.2522983551`
- `err_nRMSE=0.1759552360`
- `relative_l2=0.1759552360`
- `fRMSE_low=10.1596441269`
- `fRMSE_mid=0.1153646782`
- `fRMSE_high=0.0415550619`
- params: `2,184,967`
- peak CUDA memory: `6,372,947,456` bytes

Pinned spectral `40`-epoch anchor metrics:

- `err_RMSE=1.4877649546`
- `err_nRMSE=0.0615620054`
- `relative_l2=0.0615620054`
- `fRMSE_low=3.4756414890`
- `fRMSE_mid=0.2800448835`
- `fRMSE_high=0.4349334538`
- params: `8,186,726`

Relative to the first fairness-probe GNOT row:

- aggregate denormalized error improved materially:
  `relative_l2 0.24565 -> 0.17596`
- low-frequency error improved but remained large:
  `fRMSE_low 14.18621 -> 10.15964`
- high-frequency error stayed very low in absolute terms, but increased
  slightly:
  `fRMSE_high 0.01775 -> 0.04156`

Relative to the pinned spectral `40`-epoch anchor:

- GNOT still trails badly on aggregate denormalized error:
  `relative_l2 0.17596` vs `0.06156`
- the biggest remaining gap is still low-frequency/global structure:
  `fRMSE_low 10.15964` vs `3.47564`
- GNOT still keeps much lower saved-sample high-frequency error:
  `fRMSE_high 0.04156` vs `0.43493`

## Interpretation

The paper-default rerun sharpens the earlier local read rather than reversing
it:

- the first equal-footing fairness probe understated GNOT by using the local
  `mse` readiness recipe
- switching to the paper-default GNOT recipe materially improved the GNOT row
  on the fixed capped CNS slice
- even after that improvement, the pinned spectral anchor still wins clearly on
  the main aggregate denormalized metrics
- the remaining GNOT failure mode is still mostly low-frequency/global
  structure error rather than high-frequency shock capture

Decision boundary:

- treat the paper-default result as a stronger directional read than the first
  fairness probe
- do **not** promote it to benchmark-performance evidence because the run is
  still capped at `512 / 64 / 64`, `8` windows per trajectory
- keep the authored FFNO lane separate; the authored `40`-epoch FFNO result is
  currently much stronger on this capped local contract

## Current Status

- local GNOT integration: **working**
- paper-default recipe under the current checkout: **verified by fresh smoke**
- paper-default `40`-epoch follow-up on the fixed capped CNS contract:
  **completed**
- current local read:
  paper-default GNOT improved materially over the first fairness probe, but it
  still trails the pinned spectral anchor on aggregate error while preserving a
  much lower `fRMSE_high`
