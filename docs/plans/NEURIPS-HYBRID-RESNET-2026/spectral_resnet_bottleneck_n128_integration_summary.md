# Spectral ResNet Bottleneck N128 Integration Prototype Summary

- Date: 2026-04-20
- Tranche: `phase-2-spectral-resnet-bottleneck-n128-integration-prototype`
- Scope: implement `spectral_resnet_bottleneck_net`, verify the torch/grid-lines path on the existing `N=128` integration dataset contract, and wire a manual PDEBench image-suite profile without changing the required PDEBench bundles

## Implementation Result

Implemented:

- New generator family: `spectral_resnet_bottleneck_net`
- New module: `ptycho_torch/generators/spectral_resnet_bottleneck.py`
- Torch registry/config/runner plumbing for:
  - `spectral_bottleneck_blocks`
  - `spectral_bottleneck_modes`
  - `spectral_bottleneck_share_weights`
  - `spectral_bottleneck_gate_init`
  - `spectral_bottleneck_gate_mode`
- Manual PDEBench profile: `spectral_resnet_bottleneck_base`
- PDEBench supervised wrapper: `SpectralResnetBottleneckImageModel`

Preserved gates:

- the new family is exposed outside `hybrid_resnet`
- `spectral_resnet_bottleneck_base` is manual opt-in only
- no changes were made to `PRIMARY_DARCY_PROFILE_IDS`, `PRIMARY_CFD_CNS_PROFILE_IDS`, or `READINESS_CFD_CNS_PROFILE_IDS`

## Verification

Passing targeted verification:

- `python -m pytest tests/torch/test_spectral_resnet_bottleneck.py tests/torch/test_grid_lines_torch_runner.py tests/studies/test_pdebench_image128_models.py -q`
  - result: `120 passed`
- `python -m pytest tests/torch/test_grid_lines_hybrid_resnet_integration.py -q`
  - result: `4 passed in 266.00s`

## N128 Smoke Comparison

Dataset contract reused directly from `tests/torch/test_grid_lines_hybrid_resnet_integration.py`:

- dataset root: `.artifacts/integration/grid_lines_hybrid_resnet/datasets/N128/gs1/`
- `N=128`
- `gridsize=1`
- probe: `datasets/Run1084_recon3_postPC_shrunk_3.npz`
- `nimgs_train=2`
- `nimgs_test=1`
- `nphotons=1e9`
- `probe_source=custom`
- `probe_smoothing_sigma=0.5`
- `probe_scale_mode=pad_extrapolate`
- `set_phi=True`
- `probe_mask=False`
- seed: `3`

Smoke artifact root:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-spectral-resnet-bottleneck-n128-integration-prototype/smoke/`

Executed rows:

1. `hybrid_resnet`
2. `spectral_resnet_bottleneck_net`

Common run budget:

- epochs: `1`
- batch size: `16`
- infer batch size: `16`
- optimizer: `adam`
- learning rate: `2e-4`
- scheduler: `ReduceLROnPlateau`
- loss: `mae`
- output mode: `real_imag`

Observed metrics:

| Row | Params | Train loss | Amp MAE | Phase MAE | Amp SSIM | Phase SSIM | Amp PSNR | Phase PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `hybrid_resnet` | 18,006,600 | 0.17475011944770813 | 0.15146497 | 0.13174885870005545 | 0.6102832736813287 | 0.8943025804277306 | 62.65638866852762 | 63.765577848650054 |
| `spectral_resnet_bottleneck_net` | 23,429,454 | 0.1693844348192215 | 0.15282492 | 0.14333466301592376 | 0.6161865740637941 | 0.9007643286721325 | 62.545326940767765 | 63.07779248552278 |

Randomness contract:

- both rows recorded `requested_seed=3`
- both rows recorded `effective_subsample_seed=3`
- both rows recorded `effective_lightning_seed=3`

## Interpretation

This prototype did what it needed to do:

- the new family is buildable through the full torch/grid-lines path
- the direct runner path works on the existing `N=128` integration dataset
- the PDEBench image-suite can build the model manually without changing default bundles

The one-epoch `N=128` result is only plumbing/prototype evidence. It is too small to support a performance claim:

- the spectral variant had slightly lower one-epoch train loss
- amplitude metrics were roughly tied
- phase metrics slightly favored the baseline on MAE/PSNR while slightly favoring the spectral variant on SSIM
- the spectral variant is larger (`23.43M` vs `18.01M` params), so this first row should still be treated as a same-shell changed-bottleneck comparison, not a fairness result

## Artifact Pointers

- Hybrid smoke run:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-spectral-resnet-bottleneck-n128-integration-prototype/smoke/hybrid_resnet_base/`
- Spectral smoke run:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-spectral-resnet-bottleneck-n128-integration-prototype/smoke/spectral_resnet_bottleneck_base/`

## Next Decision

Recommended next step:

- keep this family manual opt-in
- do not promote it into PDEBench default bundles yet
- if further work is warranted, add a more controlled follow-up compare such as:
  - longer `N=128` prototype training on the same integration dataset, or
  - a manual PDEBench Darcy ablation row after the required primary suite evidence is complete
