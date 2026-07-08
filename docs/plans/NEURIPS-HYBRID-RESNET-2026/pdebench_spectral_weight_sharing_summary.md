# PDEBench Spectral Weight-Sharing CNS Compare Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Tranche: `phase-2-pdebench-spectral-weight-sharing-cns-compare`
- Date: `2026-04-21`
- Status: implementation complete; capped CNS compare complete
- Scope: add a manual opt-in non-shared spectral bottleneck row and compare it against the existing shared-weight spectral row under the same canonical CNS skip-add shell
- Governing designs:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_resnet_bottleneck_design.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_design.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-spectral-weight-sharing-cns-compare/cns-spectral-share-vs-noshare-10ep`

This summary records implementation and capped-readiness evidence only. It does not create a benchmark-complete ranking or justify promoting the non-shared row into any default bundle.

## Implemented Surface

Modified code:

- `scripts/studies/pdebench_image128/run_config.py`
- `tests/studies/test_pdebench_image128_models.py`

Reused without code changes:

- `scripts/studies/pdebench_image128/models.py`
- `ptycho_torch/generators/spectral_resnet_bottleneck.py`
- `scripts/studies/pdebench_image128/render_hybrid_upsampler_gallery.py`

Implemented contract:

- Added manual profile `spectral_resnet_bottleneck_noshare`.
- Kept the profile in the `spectral_resnet_bottleneck_*` namespace.
- Kept the profile out of:
  - `PRIMARY_CFD_CNS_PROFILE_IDS`
  - `READINESS_CFD_CNS_PROFILE_IDS`
  - `PRIMARY_DARCY_PROFILE_IDS`
- Left generator/model implementation unchanged because the existing spectral bottleneck already supported `share_spectral_weights=False`.

## Fairness Boundary

The compare keeps the same canonical `hybrid_resnet_cns` shell for both rows:

- `base_model="spectral_resnet_bottleneck_net"`
- `hidden_channels=32`
- `fno_modes=12`
- `fno_blocks=4`
- `hybrid_downsample_steps=2`
- `hybrid_resnet_blocks=6`
- `hybrid_skip_connections=True`
- `hybrid_skip_style="add"`
- `spectral_bottleneck_blocks=6`
- `spectral_bottleneck_modes=12`
- `spectral_bottleneck_gate_init=0.1`
- `spectral_bottleneck_gate_mode="shared"`
- same decoder, output head, training loss, scheduler, data split, and run mode

The only intended architectural change is:

- `spectral_resnet_bottleneck_base`: `spectral_bottleneck_share_weights=True`
- `spectral_resnet_bottleneck_noshare`: `spectral_bottleneck_share_weights=False`

## Verification

Commands run from `/home/ollie/Documents/PtychoPINN`:

```bash
python -m pytest tests/studies/test_pdebench_image128_models.py -k 'spectral_noshare' -v
python -m pytest tests/studies/test_pdebench_image128_models.py -q
python -m pytest tests/studies/test_pdebench_image128_runner.py -q
python -m compileall -q scripts/studies/pdebench_image128 ptycho_torch/generators/spectral_resnet_bottleneck.py
```

Observed implementation results before the capped run:

- Red phase: `2 failed, 25 deselected in 4.91s` because `spectral_resnet_bottleneck_noshare` did not exist yet
- Green focused rerun: `2 passed, 26 deselected in 19.38s`
- `tests/studies/test_pdebench_image128_models.py -q`: `28 passed in 38.23s`
- `tests/studies/test_pdebench_image128_runner.py -q`: `9 passed in 44.63s`
- `compileall`: passed with no output

Fresh final verification after the summary/index/ledger updates:

- `tests/studies/test_pdebench_image128_models.py -q`: `28 passed in 8.31s`
- `tests/studies/test_pdebench_image128_runner.py -q`: `9 passed in 24.59s`
- `compileall`: passed with no output
- required run-root artifacts confirmed present:
  - `comparison_spectral_resnet_bottleneck_base_sample0.npz`
  - `comparison_spectral_resnet_bottleneck_noshare_sample0.npz`
  - `metrics_spectral_resnet_bottleneck_base.json`
  - `metrics_spectral_resnet_bottleneck_noshare.json`
  - `comparison_summary.json`
  - `gallery_sample0.png`
  - `gallery_sample0_error.png`

## Capped CNS Comparison

Command:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode readiness \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-spectral-weight-sharing-cns-compare/cns-spectral-share-vs-noshare-10ep \
  --profiles spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_noshare \
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

- `spectral_resnet_bottleneck_base`
  - `err_RMSE=2.7317`
  - `err_nRMSE=0.11304`
  - `relative_l2=0.11304`
  - `fRMSE_low=6.4021`
  - `fRMSE_mid=0.3999`
  - `fRMSE_high=0.8170`
  - params: `8,186,726`
  - epoch-10 train loss: `0.02647`

- `spectral_resnet_bottleneck_noshare`
  - `err_RMSE=2.0493`
  - `err_nRMSE=0.08480`
  - `relative_l2=0.08480`
  - `fRMSE_low=4.7782`
  - `fRMSE_mid=0.3940`
  - `fRMSE_high=0.6321`
  - params: `10,546,022`
  - epoch-10 train loss: `0.02789`

Train-loss traces:

- `spectral_resnet_bottleneck_base`:
  `[0.17065, 0.07739, 0.06081, 0.04959, 0.04138, 0.03775, 0.03335, 0.03183, 0.02802, 0.02647]`
- `spectral_resnet_bottleneck_noshare`:
  `[0.17017, 0.07744, 0.05850, 0.05025, 0.04149, 0.03562, 0.03353, 0.03127, 0.02751, 0.02789]`

Practical read:

- On this capped 10-epoch CNS slice, the non-shared row is better on every tracked eval metric:
  - `err_RMSE`
  - `err_nRMSE`
  - `relative_l2`
  - `fRMSE_low`
  - `fRMSE_mid`
  - `fRMSE_high`
- The non-shared row achieves those gains with a larger parameter count:
  - `10,546,022` vs `8,186,726`
- Final train loss is slightly lower for the shared row, so the non-shared gain is not just "lower training loss"; it is translating into better held-out denormalized metrics despite a near-tied training endpoint.

## Qualitative Read

Rendered galleries for sample `0` do not show a new gross failure mode in the non-shared row relative to the shared row:

- density, `Vx`, and `Vy` residuals are generally lower in the non-shared row on the rendered sample
- pressure remains visibly striped in both rows, so the improvement is not a full pressure-quality fix
- the capped compare therefore supports the non-shared row as a serious follow-up candidate, but not as a solved spectral artifact story

## Artifacts

Core artifacts:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-spectral-weight-sharing-cns-compare/cns-spectral-share-vs-noshare-10ep/comparison_summary.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-spectral-weight-sharing-cns-compare/cns-spectral-share-vs-noshare-10ep/comparison_summary.csv`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-spectral-weight-sharing-cns-compare/cns-spectral-share-vs-noshare-10ep/metrics_spectral_resnet_bottleneck_base.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-spectral-weight-sharing-cns-compare/cns-spectral-share-vs-noshare-10ep/metrics_spectral_resnet_bottleneck_noshare.json`

Per-profile outputs:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-spectral-weight-sharing-cns-compare/cns-spectral-share-vs-noshare-10ep/comparison_spectral_resnet_bottleneck_base_sample0.npz`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-spectral-weight-sharing-cns-compare/cns-spectral-share-vs-noshare-10ep/comparison_spectral_resnet_bottleneck_base_sample0.png`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-spectral-weight-sharing-cns-compare/cns-spectral-share-vs-noshare-10ep/comparison_spectral_resnet_bottleneck_noshare_sample0.npz`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-spectral-weight-sharing-cns-compare/cns-spectral-share-vs-noshare-10ep/comparison_spectral_resnet_bottleneck_noshare_sample0.png`

Rendered galleries:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-spectral-weight-sharing-cns-compare/cns-spectral-share-vs-noshare-10ep/gallery_sample0.png`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-spectral-weight-sharing-cns-compare/cns-spectral-share-vs-noshare-10ep/gallery_sample0_error.png`

Quick-copy inspection paths:

- `tmp/cns_spectral_share_vs_noshare_sample0.png`
- `tmp/cns_spectral_share_vs_noshare_sample0_error.png`

## Claim Boundary

This is:

- a capped 10-epoch readiness compare
- decision-support evidence only
- not a benchmark-complete architecture ranking
- not enough evidence to replace the shared spectral row as the default reference automatically

The justified next step is a longer, still-controlled rerun or a broader compare that includes the canonical local Hybrid row under the same protocol.
