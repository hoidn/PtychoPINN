# NeurIPS Hybrid ResNet PDEBench GNOT CNS Compare Summary

## Scope

This summary records the first successful local integration of the official GNOT source into the PDEBench `2d_cfd_cns` image-suite runner, plus the first capped equal-footing comparison against `spectral_resnet_bottleneck_base`.

It covers two distinct outcomes from the same session:

1. the **environment and adapter gate** required to run official GNOT locally on CUDA, and
2. the first **equal-footing capped CNS compare** using the repo's existing local readiness contract.

This summary does **not** claim paper-faithful GNOT performance. The first compare intentionally held the local training recipe fixed for fairness, then the code was updated so that **subsequent** GNOT runs use the paper-style defaults.

## Environment Gate

The active repo environment (`ptycho311`, `torch 2.9.1+cu128`) was not usable for CUDA GNOT because no matching working CUDA DGL path was available there.

The working local compare environment is:

- env: `ptycho311_2`
- `torch 2.4.1+cu124`
- `dgl 2.4.0+cu124`

The official GNOT source is pinned externally at:

- repo: `https://github.com/HaoZhongkai/GNOT`
- commit: `5ee2e6925a43f9a340a6016bad4da2c82a452cbe`
- local clone: [gnot](/home/ollie/Documents/PtychoPINN/.artifacts/external/gnot)

Provenance artifact:

- [gnot_source.json](/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/gnot_source.json)

## Implementation

Added local GNOT support to the PDEBench CNS runner:

- adapter: [gnot_adapter.py](/home/ollie/Documents/PtychoPINN/scripts/studies/pdebench_image128/gnot_adapter.py)
- model builder wiring: [models.py](/home/ollie/Documents/PtychoPINN/scripts/studies/pdebench_image128/models.py)
- profile contract: [run_config.py](/home/ollie/Documents/PtychoPINN/scripts/studies/pdebench_image128/run_config.py)
- runner task-metadata and training-recipe wiring: [cfd_cns.py](/home/ollie/Documents/PtychoPINN/scripts/studies/pdebench_image128/cfd_cns.py)

Focused verification:

- `python -m pytest tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py -q`
- result: `42 passed`

## First Equal-Footing Compare

Run root:

- [cns-gnot-vs-spectral-cap512-10ep-20260422T200900Z](/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cns-gnot-vs-spectral-cap512-10ep-20260422T200900Z)

Protocol:

- task: `2d_cfd_cns`
- `history_len=2`
- trajectories: `512 / 64 / 64`
- `max_windows_per_trajectory=8`
- train/val/test windows: `4096 / 512 / 512`
- epochs: `10`
- batch size: `4`
- profiles:
  - `gnot_cns_base`
  - `spectral_resnet_bottleneck_base`

Important caveat:

- this first compare used the **local equal-footing readiness recipe**, not the paper-default GNOT training recipe:
  - MSE-style local CNS setup for the in-flight run
  - intended as a fairness probe against the local spectral baseline

### Result

From [comparison_summary.json](/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cns-gnot-vs-spectral-cap512-10ep-20260422T200900Z/comparison_summary.json):

- `gnot_cns_base`
  - `relative_l2 = 0.2456480712`
  - `err_RMSE = 5.9365601540`
  - `fRMSE_high = 0.0177509338`
  - params: `551,815`

- `spectral_resnet_bottleneck_base`
  - `relative_l2 = 0.0859752074`
  - `err_RMSE = 2.0777571201`
  - `fRMSE_high = 0.6988957524`
  - params: `8,186,726`

So on this first equal-footing capped run:

- spectral beat GNOT on the main aggregate denormalized error metrics by a wide margin
- GNOT's high-frequency error was much lower, but its low-frequency/global field error was much worse

GNOT per-profile artifact:

- [metrics_gnot_cns_base.json](/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cns-gnot-vs-spectral-cap512-10ep-20260422T200900Z/metrics_gnot_cns_base.json)

Spectral per-profile artifact:

- [metrics_spectral_resnet_bottleneck_base.json](/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cns-gnot-vs-spectral-cap512-10ep-20260422T200900Z/metrics_spectral_resnet_bottleneck_base.json)

Prediction PNGs:

- [comparison_gnot_cns_base_sample0.png](/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cns-gnot-vs-spectral-cap512-10ep-20260422T200900Z/comparison_gnot_cns_base_sample0.png)
- [comparison_spectral_resnet_bottleneck_base_sample0.png](/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cns-gnot-vs-spectral-cap512-10ep-20260422T200900Z/comparison_spectral_resnet_bottleneck_base_sample0.png)

## Interpretation

This result should be read narrowly:

- it proves the official GNOT source can be hosted locally under the CNS image-suite contract
- it gives a first strict apples-to-apples comparison under the local readiness recipe
- it does **not** say that paper-default GNOT is weak on this task

Why not:

- the public GNOT recipe uses `rel2`, `AdamW`, `lr=1e-3`, `cycle` scheduling, and `epochs=500`
- the first capped compare was deliberately run under the local fairness recipe instead

## Follow-On Decision

After this first compare, the local `gnot_cns_base` profile was updated so that **subsequent** GNOT runs use the paper-style defaults in the CNS runner:

- `gnot_hidden=128`
- `training_loss=relative_l2`
- `optimizer=AdamW`
- `learning_rate=1e-3`
- `scheduler=OneCycleLR`
- `weight_decay=5e-5`

That means the next GNOT row answers a different question:

- **first run:** equal-footing local recipe
- **next run:** paper-default GNOT recipe on the same CNS contract

## Current Status

- GNOT local integration: **working**
- equal-footing capped compare: **completed**
- paper-default GNOT rerun: **next required step**
