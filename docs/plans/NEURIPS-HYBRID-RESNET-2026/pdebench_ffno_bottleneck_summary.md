# PDEBench FFNO-Close Bottleneck CNS Compare Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Tranche: `phase-2-pdebench-ffno-bottleneck-cns-compare`
- Date: `2026-04-21`
- Status: implementation complete; capped CNS compare complete
- Scope: add a bottleneck-only `ffno_bottleneck_net` family, host the local, spectral, and FFNO-close rows in the same canonical CNS skip-add shell, and compare them on a capped `2d_cfd_cns` slice
- Governing design: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_design.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`

This summary records implementation and capped-readiness evidence only. It does not create a benchmark-complete claim or a manuscript-ready FFNO result.

## Implemented Surface

New code:

- `ptycho_torch/generators/ffno_bottleneck.py`
- `tests/torch/test_ffno_bottleneck.py`

Modified code:

- `scripts/studies/pdebench_image128/models.py`
- `scripts/studies/pdebench_image128/run_config.py`
- `tests/studies/test_pdebench_image128_models.py`
- `tests/studies/test_pdebench_image128_runner.py`

Implemented contract:

- `ffno_bottleneck_net` is a separate family, not a `hybrid_resnet_*` alias.
- The FFNO-close row keeps the same canonical CNS shell as `hybrid_resnet_cns`:
  - same `SpatialLifter`
  - same encoder stack
  - same `hybrid_downsample_steps=2`
  - same skip-add fusion policy
  - same decoder / upsampler path
  - same output head
- The only intended architectural change across the three compare rows is the bottleneck:
  - `hybrid_resnet_cns`: local ResNet bottleneck
  - `spectral_resnet_bottleneck_base`: shared spectral-bypass ResNet bottleneck
  - `ffno_bottleneck_base`: FFNO-close factorized spectral plus feedforward bottleneck
- `ffno_bottleneck_base` remains manual opt-in and does not enter the primary or readiness-required profile bundles.

## FFNO-Close Bottleneck Contract

The implemented FFNO-close bottleneck is shape-preserving and stackable:

- factorized spectral mixing at fixed latent resolution
- optional normalization
- `1x1` expand -> `GELU` -> `1x1` project feedforward path
- outer residual add
- explicit control for:
  - `ffno_bottleneck_blocks`
  - `ffno_bottleneck_modes`
  - `ffno_bottleneck_share_weights`
  - `ffno_bottleneck_mlp_ratio`
  - `ffno_bottleneck_gate_init`
  - `ffno_bottleneck_norm`

Base comparison profile:

- `profile_id`: `ffno_bottleneck_base`
- `hidden_channels=32`
- `hybrid_downsample_steps=2`
- `hybrid_skip_connections=True`
- `hybrid_skip_style="add"`
- `ffno_bottleneck_blocks=6`
- `ffno_bottleneck_modes=12`
- `ffno_bottleneck_share_weights=True`
- `ffno_bottleneck_mlp_ratio=2.0`
- `ffno_bottleneck_gate_init=0.1`
- `ffno_bottleneck_norm="instance"`
- `evidence_scope="readiness-only"`

## Verification

Commands run from `/home/ollie/Documents/PtychoPINN`:

```bash
python -m pytest tests/torch/test_ffno_bottleneck.py -q
python -m pytest tests/studies/test_pdebench_image128_models.py -q
python -m pytest tests/studies/test_pdebench_image128_runner.py -q
python -m compileall -q scripts/studies/pdebench_image128 ptycho_torch/generators/ffno_bottleneck.py
```

Observed results:

- `tests/torch/test_ffno_bottleneck.py`: `4 passed in 5.03s`
- `tests/studies/test_pdebench_image128_models.py`: `23 passed in 8.25s`
- `tests/studies/test_pdebench_image128_runner.py`: `9 passed in 25.39s`
- `compileall`: passed with no output

## Capped CNS Comparison

Command:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode readiness \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep \
  --profiles hybrid_resnet_cns,spectral_resnet_bottleneck_base,ffno_bottleneck_base \
  --history-len 2 \
  --epochs 10 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8 \
  --device cuda \
  --num-workers 0
```

Run contract:

- task: `2d_cfd_cns`
- mode: `readiness`
- training loss: `mse`
- history contract: `concat u[t-2:t] -> u[t]`
- train / val / test trajectories: `512 / 64 / 64`
- train / val / test windows: `4096 / 512 / 512`
- evidence scope: `smoke_feasibility_only`

Dataset:

- `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`

## Results

From `comparison_summary.json`:

- `hybrid_resnet_cns`
  - `err_RMSE=2.2814`
  - `err_nRMSE=0.09440`
  - `relative_l2=0.09440`
  - `fRMSE_low=5.3005`
  - `fRMSE_mid=0.4186`
  - `fRMSE_high=0.8000`
  - params: `7,793,509`
  - epoch-10 train loss: `0.03216`

- `spectral_resnet_bottleneck_base`
  - `err_RMSE=2.1024`
  - `err_nRMSE=0.08699`
  - `relative_l2=0.08699`
  - `fRMSE_low=4.8862`
  - `fRMSE_mid=0.4221`
  - `fRMSE_high=0.6955`
  - params: `8,186,726`
  - epoch-10 train loss: `0.02551`

- `ffno_bottleneck_base`
  - `err_RMSE=2.6842`
  - `err_nRMSE=0.11107`
  - `relative_l2=0.11107`
  - `fRMSE_low=6.3083`
  - `fRMSE_mid=0.3836`
  - `fRMSE_high=0.7277`
  - params: `6,809,701`
  - epoch-10 train loss: `0.02627`

Training-loss traces:

- `hybrid_resnet_cns`: `[0.18262, 0.09488, 0.07030, 0.06109, 0.05148, 0.04490, 0.04215, 0.03545, 0.03474, 0.03216]`
- `spectral_resnet_bottleneck_base`: `[0.16655, 0.07622, 0.05886, 0.04854, 0.04030, 0.03788, 0.03287, 0.03027, 0.02876, 0.02551]`
- `ffno_bottleneck_base`: `[0.17516, 0.07974, 0.05968, 0.04966, 0.04487, 0.04014, 0.03240, 0.03376, 0.03106, 0.02627]`

Practical read:

- On this capped 10-epoch CNS slice, `spectral_resnet_bottleneck_base` is clearly best.
- `hybrid_resnet_cns` is second on aggregate denormalized error.
- `ffno_bottleneck_base` did not beat the canonical local `hybrid_resnet_cns` on aggregate denormalized error.
- FFNO-close is still useful as a distinct architecture probe because it reached a final train loss close to the spectral row with fewer parameters, but that did not translate into better eval metrics on this slice.

## Relation To Prior Vanilla FNO Baseline

For context, the earlier capped `fno_base` 10-epoch CNS run at
`.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`
reported:

- `err_RMSE=2.5700`
- `err_nRMSE=relative_l2=0.10634`
- `fRMSE_low=6.0062`
- `fRMSE_mid=0.2311`
- `fRMSE_high=0.9280`

Compared against that vanilla FNO row:

- `spectral_resnet_bottleneck_base` is better on aggregate error, low-band error,
  and high-band error, but worse on mid-band error.
- `hybrid_resnet_cns` is also better on aggregate error, low-band error, and
  high-band error, but worse on mid-band error.
- `ffno_bottleneck_base` is worse than vanilla FNO on aggregate error and
  low-band error, slightly better on `fRMSE_high`, and still worse on
  `fRMSE_mid`.

This is still not a paper-grade FNO versus FFNO claim because the rows come
from separate capped runs, but the split and protocol are aligned enough to use
this as local decision-support evidence.

## Artifacts

Core artifacts:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep/comparison_summary.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep/comparison_summary.csv`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep/metrics_hybrid_resnet_cns.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep/metrics_spectral_resnet_bottleneck_base.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep/metrics_ffno_bottleneck_base.json`

Prediction PNGs:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep/comparison_hybrid_resnet_cns_sample0.png`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep/comparison_spectral_resnet_bottleneck_base_sample0.png`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep/comparison_ffno_bottleneck_base_sample0.png`

Rendered galleries:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep/gallery_sample0.png`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep/gallery_sample0_error.png`

## Boundary On Claims

This run is capped readiness evidence only.

It supports these claims:

- the FFNO-close family is implemented and test-covered
- the three rows now share the same canonical CNS skip-add shell
- the spectral bottleneck is the strongest of the three on this capped slice
- the FFNO-close bottleneck did not yet improve over the local Hybrid bottleneck on capped CNS evaluation metrics

It does not support these claims:

- benchmark-complete CNS ranking
- paper-quality FFNO comparison
- full-dataset convergence behavior
- superiority or inferiority under a broader hyperparameter search
