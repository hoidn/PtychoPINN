# Model Baselines

This document defines the current **project-recommended training baselines** for
major model families.

Use this document when you need the recommended starting point for a real run or
study. Do not infer "best practice" from a library default, an older run
artifact, or a prompt.

## Authority Boundary

- `docs/model_baselines.md` is the authority for **recommended project
  baselines**.
- `docs/CONFIGURATION.md` is the authority for **parameter definitions and raw
  defaults**.
- Study docs may override these baselines, but they must say so explicitly.

If a study doc does not explicitly override a baseline family, inherit the
relevant baseline from this document.

## Hybrid ResNet

### Status

Current status: `recommended baseline`

This baseline is the recommended starting point for grid-lines Torch studies and
single-dataset Hybrid ResNet experiments unless a study doc explicitly says
otherwise.

### Baseline Schedule

- optimizer: `adam`
- learning rate: `2e-4`
- scheduler: `ReduceLROnPlateau`
- plateau factor: `0.5`
- plateau patience: `2`
- plateau min lr: `1e-4`
- plateau threshold: `0.0`
- weight decay: `0.0`
- `beta1`: `0.9`
- `beta2`: `0.999`
- loss mode: `mae`
- `torch_mae_pred_l2_match_target`: `on`
- `probe_mask`: `off`

### Baseline Architecture

- architecture: `hybrid_resnet`
- `fno_modes=12`
- `fno_width=32`
- `fno_blocks=4`
- `hybrid_skip_connections=off`
- `hybrid_downsample_steps=2`
- `hybrid_downsample_op=stride_conv`
- `hybrid_encoder_conv_hidden_scale=2.0`
- `hybrid_encoder_spectral_hidden_scale=1.0`
- `hybrid_resnet_blocks=6`
- `hybrid_skip_style=add`

### Conventions

- For `N=128` grid-lines integration-style checks, keep the integration-test
  schedule above unless the baseline is intentionally being reevaluated.
- For study loops at a different resolution or epoch budget, inherit this
  schedule and only override the study-specific parts that must differ, such as
  `epochs`, dataset path, or fixed wrapper output locations.
- If a wrapper or runbook intentionally diverges from this baseline, document
  the override in the study doc that owns that wrapper.

### Evidence

This baseline is currently grounded in:

- [test_grid_lines_hybrid_resnet_integration.py](/home/ollie/Documents/tmp/PtychoPINN/tests/torch/test_grid_lines_hybrid_resnet_integration.py)
- [workflows/pytorch.md](/home/ollie/Documents/tmp/PtychoPINN/docs/workflows/pytorch.md)
- [CONFIGURATION.md](/home/ollie/Documents/tmp/PtychoPINN/docs/CONFIGURATION.md)

The integration test is not itself the authority. It is evidence for the
recommended baseline recorded here.

## How To Maintain This File

When a new study or regression test establishes a better recommended baseline:

1. update this document first
2. update any study docs that inherit or override that baseline
3. update fixed wrapper constants or runner defaults that are meant to match it

Do not leave the recommended baseline implicit in test files alone.
