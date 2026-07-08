# PDEBench CNS Spectral Modes-32 Compare Plan

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Date: `2026-04-21`
- Status: queued backlog plan
- Scope: increase the spectral CNS row from `12` to `32` Fourier modes in both
  the encoder spectral path and the spectral bottleneck, then rerun the capped
  CNS compare to see whether metrics improve

## Objective

Test whether the current strongest local CNS direction,
`spectral_resnet_bottleneck_base`, improves when both:

- encoder spectral modes: `fno_modes`
- bottleneck spectral modes: `spectral_bottleneck_modes`

are raised from `12` to `32`.

This is a targeted capacity ablation, not a general hyperparameter sweep.

## Equal-Footing Contract

Keep the current capped CNS protocol fixed except for the mode counts.

Fixed contract:

- task: `2d_cfd_cns`
- dataset:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- split: `512 / 64 / 64` trajectories
- `max_windows_per_trajectory=8`
- `history_len=2`
- loss: `mse`
- batch size: `4`
- epoch budgets:
  - `10`
  - `40`
- reported metrics:
  - `err_RMSE`
  - `err_nRMSE`
  - `relative_l2`
  - `fRMSE_low`
  - `fRMSE_mid`
  - `fRMSE_high`

Changed variables only:

- `fno_modes: 12 -> 32`
- `spectral_bottleneck_modes: 12 -> 32`

Leave these unchanged:

- `hidden_channels=32`
- `fno_blocks=4`
- `hybrid_downsample_steps=2`
- `hybrid_resnet_blocks=6`
- `hybrid_skip_connections=True`
- `hybrid_skip_style=add`
- spectral weight sharing mode
- upsampler choice

## Why This Is Worth Testing

The capped 40-epoch spectral row is currently the strongest local CNS result.
One plausible remaining gain is that `12` modes underuse the `32x32` latent
resolution. Raising both encoder and bottleneck modes to `32` is the most direct
test of whether the current model is frequency-limited.

## Risks

1. `32` modes may increase memory and runtime enough to make the comparison less
   practical on the RTX 3090.
2. More modes may overfit the capped slice or simply not help if the current
   limitation is elsewhere.
3. If only one of the two mode knobs is changed, the result becomes ambiguous.

## Acceptance Criteria

1. A manual spectral profile exists with `fno_modes=32` and
   `spectral_bottleneck_modes=32`.
2. A `10`-epoch capped CNS run exists for that profile.
3. A `40`-epoch capped CNS run exists for that profile.
4. The result is compared against the current `12/12` spectral row and the
   existing local `fno_base` / `unet_strong` anchors.
5. The summary states clearly whether the higher-mode spectral row improved the
   capped metrics enough to justify further attention.

## Suggested Checks

- `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
