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

## Shared Torch CDI Geometry And Scaling

These values are the canonical starting point for Torch CDI training and
inference. They are geometry/physics invariants, not tuning suggestions.

| Context | `object_big` | `probe_big` | `amplitude_physics_gain` |
| --- | --- | --- | --- |
| Single-patch (`gridsize=1`) CNN | `False` | `False` | Derived for legacy amplitude data; `1.0` for CI/rectangular scaling |
| Grouped/reassembled (`gridsize>1`) CNN | `True` | `True` | Derived for legacy amplitude data; `1.0` for CI/rectangular scaling |
| Grouped/reassembled non-CNN study | `True` | `True` in shared manifests, even when the generator does not consume the CNN support branch | Same physics rule as above |

For an `object_big=True` CNN, `probe_big=True` means that each semantic
component head learns the complementary outer support needed for the full
patch. It is required for the normal Torch path. Set it to `False` only for an
explicitly named historical-checkpoint or zero-border diagnostic; record that
reason beside the override. Training and inference for one checkpoint must
resolve the same support policy.

Each CNN component head emits one outer latent channel per semantic output
channel: `C_model` for the real/amplitude head and `C_model` for the
imaginary/phase head. Thus `C_model=4` has eight outer latent channels in
total, not eight per head. The legacy asymmetric/channel-fraction override is
checkpoint-diagnostic only.

`amplitude_physics_gain` is not a quality hyperparameter. For a sealed legacy
normalized-amplitude training split, derive it once from the exact forward:

```text
P_eff = normalize_probe_like_tf(P_stored, probe_scale) / probe_scale
A0 = fftshift(abs(sum_p FFT(O * P_eff[p]))) / N
r = sqrt(N^2 / mean_samples(sum_hw(Y^2)))
G_phys = r * sqrt(sum(Y^2) / sum(A0^2))
```

Use that one value across architectures and legacy MAE/NLL profiles. CI's
rectangular/count path requires `amplitude_physics_gain=1.0` because its
trainable `s1`/`s2` own the training scale. For the sealed Task 30 v3 lines
training input, the derived legacy value is `12.452229360013307`; it is not a
universal default. The historical value `16` reproduces the accidental
batch-size broadcast conditioning at batch size 16, but is not the physical
normalization and must not be selected by a gain sweep.

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
- For PDE, OpenFWI, or other supervised adapters that are used to judge Hybrid
  ResNet competitiveness, inherit this recipe unless the study plan explicitly
  records a justified override. A one-epoch smoke run is only data/adapter
  feasibility evidence, not a meaningful competitiveness result.
- Task-local override: the canonical PDEBench `2d_cfd_cns` Hybrid row is
  `hybrid_resnet_cns`, not `hybrid_resnet_base`. It keeps the same width,
  modes, depth, and downsample schedule but enables
  `hybrid_skip_connections=on` with `hybrid_skip_style=add` based on the capped
  CNS skip-study evidence. As of the post-skip-add upsampler compare, the
  canonical CNS row also defaults to `hybrid_upsampler=pixelshuffle`. The
  earlier transpose decoder remains available only as the explicit manual study
  profile `hybrid_resnet_cns_transpose`. This override is CNS-specific; Darcy
  and generic supervised adapters still inherit `hybrid_resnet_base` unless
  their owning study doc says otherwise.
- If a wrapper or runbook intentionally diverges from this baseline, document
  the override in the study doc that owns that wrapper.

### Evidence

This baseline is currently grounded in:

- [test_grid_lines_hybrid_resnet_integration.py](../tests/torch/test_grid_lines_hybrid_resnet_integration.py)
- [workflows/pytorch.md](workflows/pytorch.md)
- [CONFIGURATION.md](CONFIGURATION.md)

The integration test is not itself the authority. It is evidence for the
recommended baseline recorded here.

## How To Maintain This File

When a new study or regression test establishes a better recommended baseline:

1. update this document first
2. update any study docs that inherit or override that baseline
3. update fixed wrapper constants or runner defaults that are meant to match it

Do not leave the recommended baseline implicit in test files alone.
