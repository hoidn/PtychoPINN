# NeurIPS WaveBench Shared-Encoder Supervised Summary

## Decision

- Selected variant: `time_varying/is/thick_lines_gaussian_lens`
- Locked split: seed-42 `9000 / 500 / 500` (`train / val / test`)
- Stable dataset target: `<wavebench repo>/wavebench_dataset/time_varying/is/`
- Staged dataset member: `wavebench_dataset/time_varying/is/thick_lines_gaussian_lens.beton`
- Observed checkout path: `tmp/wavebench_repo` at revision
  `2bea258d9f05ec7182741293be11be1e545576ae`
- Environment actually used: PATH `python`
  `/home/ollie/miniconda3/envs/ptycho311/bin/python` (`Python 3.11.13`)
  with the WaveBench supervised-loader requirements installed
  (`opencv`, `pkg-config`, `libjpeg-turbo`, `ffcv`, `ml-collections`,
  `numpy==1.26.4`).
- Claim boundary: candidate-lane evidence only. The shared-encoder rows do
  not promote WaveBench into manuscript evidence and do not satisfy any
  required CDI `lines128` or PDEBench CNS pillar.

This summary records the first repo-local supervised WaveBench shared-encoder
architecture comparison on the locked inverse-source variant. Every shared
encoder + body row was trained jointly under a single fixed measurement
encoder, the same locked split, and the same supervised recipe. Native
WaveBench rows in `wavebench_native_baseline_summary.md` remain external
reference context and are not members of the shared-encoder comparison.

## Shared-Encoder Row Contract

- Row roster (machine-readable lock at
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-shared-encoder-supervised-benchmark/row_contract.json`):
  - `cnn` (local U-Net body)
  - `hybrid_resnet` (shell + ResNet bottleneck)
  - `spectral_resnet_bottleneck_net` (shell + shared-spectral ResNet bottleneck)
  - `fno`
  - `ffno`
- Latent-channel settings: `C=32` (minimum serious row set) and `C=64`
  (immediate sensitivity comparison).
- Encoder architecture: one fixed shared anisotropic convolutional encoder
  in `scripts/studies/wavebench_shared_encoder/encoder.py`. The encoder is
  jointly trained per row; it is never frozen or pre-trained in this item.
- Tensor contract: archived input `(1, 128, 128) float32`, archived target
  `(1, 128, 128) float32`. The encoder lifts to a `(C, 128, 128)` latent.
- Supervised recipe (locked across every row and both `C` settings):
  - loss: `L1 / MAE`
  - optimizer: `Adam`, `lr = 2e-4`
  - scheduler: `ReduceLROnPlateau`, `factor=0.5`, `patience=2`,
    `min_lr=1e-5`, `threshold=0.0`
  - seed: `42`
  - batch size: `train=32`, `eval=64`
  - epochs: `50`
- Row status vocabulary: `completed | smoke_pass | blocked | not_protocol_compatible`.
  `completed` is reserved for benchmark-mode rows on the full locked split;
  `smoke_pass` records a feasibility-only smoke run; `blocked` and
  `not_protocol_compatible` preserve explicit row-level outages.

## Shared-Encoder Rows

Every required row was executed end-to-end on the locked recipe. Outcome:
all 10 benchmark configurations (5 architectures × `C=32` and `C=64`)
converged to a near-zero trivial-prediction baseline under the locked
`L1 + Adam 2e-4 + ReduceLROnPlateau(threshold=0.0)` recipe. Train/val L1
losses settled at ~`0.0335 / 0.0334` for every configuration, and the test
metrics collapsed to `MAE ≈ 0.0328`, `RMSE ≈ 0.1452`, `RelL2 ≈ 1.0`,
`SSIM ≈ 0` — i.e., the models predicted approximately the per-pixel mean
(near zero) rather than the line structure. Visual inspection of the
fixed-sample reconstructions confirms this: predictions are near-uniform
zero and the error map is the negation of the target.

This is the honest as-run benchmark outcome under the locked fairness
contract. Under the locked recipe there is no fair architecture ranking
between rows — all rows finish at the same trivial baseline, so the
comparison contract is preserved (every row got the same encoder, split,
loss, optimizer, scheduler, seed, batch size, epoch budget) but the
discriminating signal is absent. Lifting the recipe lock (e.g., looser
plateau threshold, different loss, normalization, longer warmup) is out of
scope for this item; per the plan, the recipe is locked and a future
follow-on item would have to authorize the change.

| Row | C | Encoder params | Body params | Total params | Runtime (s) | MAE | RMSE | RelL2 | SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cnn` | 32 | 73,408 | 1,936,769 | 2,010,177 | 772.3 | 0.032856 | 0.145195 | 0.999984 | 2.541e-05 |
| `cnn` | 64 | 75,488 | 1,945,985 | 2,021,473 | 800.6 | 0.032882 | 0.145194 | 0.999975 | 1.897e-05 |
| `hybrid_resnet` | 32 | 73,408 | 7,795,106 | 7,868,514 | 984.3 | 0.032854 | 0.145195 | 0.999979 | 9.983e-06 |
| `hybrid_resnet` | 64 | 75,488 | 7,804,322 | 7,879,810 | 1014.6 | 0.032853 | 0.145195 | 0.999981 | 1.750e-05 |
| `spectral_resnet_bottleneck_net` | 32 | 73,408 | 8,188,323 | 8,261,731 | 1255.0 | 0.032873 | 0.145197 | 0.999993 | 7.290e-06 |
| `spectral_resnet_bottleneck_net` | 64 | 75,488 | 8,197,539 | 8,273,027 | 1284.0 | 0.032878 | 0.145195 | 0.999982 | 2.580e-05 |
| `fno` | 32 | 73,408 | 597,313 | 670,721 | 560.2 | 0.032825 | 0.145200 | 1.000020 | 2.940e-06 |
| `fno` | 64 | 75,488 | 598,337 | 673,825 | 579.7 | 0.032820 | 0.145195 | 0.999983 | 6.063e-06 |
| `ffno` | 32 | 73,408 | 97,122 | 170,530 | 1157.8 | 0.032827 | 0.145199 | 1.000012 | 9.565e-06 |
| `ffno` | 64 | 75,488 | 106,338 | 181,826 | 1185.0 | 0.032868 | 0.145195 | 0.999982 | 1.137e-05 |

Locked benchmark metrics for each completed row are persisted machine-readably
in:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-shared-encoder-supervised-benchmark/table_ready_metrics.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-shared-encoder-supervised-benchmark/comparison_summary.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-shared-encoder-supervised-benchmark/comparison_summary.csv`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-shared-encoder-supervised-benchmark/rows/<row>/c{32,64}/metrics.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-shared-encoder-supervised-benchmark/rows/<row>/c{32,64}/model_profile.json`

Per-row reconstruction figures and source arrays are persisted under:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-shared-encoder-supervised-benchmark/figures/c{32,64}/<row>/`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-shared-encoder-supervised-benchmark/figures/source_arrays/<row>/c{32,64}/`

The execution report at
`artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-shared-encoder-supervised-benchmark/execution_report.md`
records the as-run benchmark outcome per row plus any row-level blockers and
their narrow-fix attempts.

The shared-encoder body adapters in
`scripts/studies/wavebench_shared_encoder/models.py` are repo-owned
comparison wrappers, not native WaveBench architectures. Their fairness value
depends on the shared encoder, split, loss, optimizer, scheduler, seed, and
reporting schema being held fixed across every row — every one of these is
locked in `row_contract.json` and the runner.

Body-label semantics actually used in this bundle:

- `cnn`: U-Net with `width=32`, three down/up stages (`LocalUnetBody`).
- `hybrid_resnet`: shared-encoder `HybridShellBody` with a 6-block
  `ResnetBottleneck`.
- `spectral_resnet_bottleneck_net`: shared-encoder `HybridShellBody` with a
  6-block `SharedSpectralResnetBottleneck` (`modes=12`).
- `fno`: 4-block standard FNO 2D body
  (`_FallbackSpectralConv2d(modes=12)` spectral path + 1×1 bypass + GELU per
  block, `hidden_width=32`). The block uses real `torch.fft.rfft2` /
  `torch.fft.irfft2` spectral convolution, matching the design's
  `fno_modes=12, fno_width=32, fno_blocks=4` recipe.
- `ffno`: shared-encoder FFNO stack via
  `build_no_refiner_ffno_stack(modes=12, n_blocks=4,
  share_spectral_weights=True)` with two downstream local 3×3 residual
  refiners before the 1×1 output projection. The "no-refiner" name applies
  to the FFNO stack itself (no built-in local refiner inside the spectral
  trunk); the trailing local refiners here are part of the shared-encoder
  WaveBench wrapper, not the upstream FFNO contract used elsewhere in the
  initiative.

## Native WaveBench Reference Context

- Native WaveBench rows from `wavebench_native_baseline_summary.md` remain
  external-reference context only. The native U-Net checkpoint route (test
  MAE `0.014410`, RMSE `0.068332`, RelL2 `0.446032`, SSIM `0.881951`) and the
  reproduced native FNO depth-4 retraining route are reported there.
- These native rows must not be mixed into the shared-encoder comparison
  table. They run a different encoder/input contract than the shared-encoder
  rows and answer a different question (external-reference performance under
  the upstream's own input contract, rather than fair internal architecture
  comparison under the shared encoder).

## Durable Outputs

- Summary authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_shared_encoder_supervised_summary.md`
- Shared-encoder artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-shared-encoder-supervised-benchmark/`
- Locked row contract:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-shared-encoder-supervised-benchmark/row_contract.json`
- Execution manifest:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-shared-encoder-supervised-benchmark/shared_encoder_execution_manifest.json`
- Code surfaces:
  - `scripts/studies/wavebench_shared_encoder/{data,encoder,models,metrics,reporting}.py`
  - `scripts/studies/run_wavebench_shared_encoder_benchmark.py`
  - `scripts/studies/validate_wavebench_shared_encoder_contract.py`
- Test surfaces:
  - `tests/studies/test_wavebench_shared_encoder_data.py`
  - `tests/studies/test_wavebench_shared_encoder_models.py`
  - `tests/studies/test_wavebench_shared_encoder_runner.py`
  - `tests/studies/test_wavebench_shared_encoder_contract.py`

## Verification

- Prerequisite contract checks:
  `python scripts/studies/validate_wavebench_preflight_contract.py`
  `python scripts/studies/validate_wavebench_provisioning_decision.py`
  `python scripts/studies/validate_wavebench_native_baseline_contract.py`
- Implementation gate:
  `pytest -q tests/studies/test_wavebench_shared_encoder_data.py
  tests/studies/test_wavebench_shared_encoder_models.py
  tests/studies/test_wavebench_shared_encoder_runner.py
  tests/studies/test_wavebench_shared_encoder_contract.py`
- Final bundle validation under the benchmark-completion gate:
  `python scripts/studies/validate_wavebench_shared_encoder_contract.py
  --require-benchmark-completion`

## Residual Risks

- All 10 rows under the locked recipe collapsed to a near-zero
  trivial-prediction baseline (final train/val L1 ≈ `0.0335 / 0.0334`,
  test `RelL2 ≈ 1.0`, `SSIM ≈ 0`). Under this contract the body family is
  not separable — the locked recipe is the limiting factor, not the body
  architecture. Reading any of these rows as evidence that a particular
  body is competitive or non-competitive on WaveBench inverse source would
  be a misuse of this bundle.
- The locked `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5,
  threshold=0.0)` is aggressive: with `threshold=0.0` any non-monotonic
  validation loss triggers an LR halving, and the floor `1e-5` is reached
  in ~6 step-downs. The collapse is consistent with early LR floor +
  shared-encoder collapse, but the plan explicitly forbids loosening the
  fairness contract to make a row easier, so the recipe is preserved. A
  later candidate-lane item could authorize a recipe change (e.g., looser
  plateau threshold, different loss, normalization, longer warmup) and
  rerun under the same row roster.
- The shared-encoder rows are repo-owned comparison wrappers, not native
  WaveBench architectures. Their fairness depends on the locked encoder,
  split, loss, optimizer, scheduler, seed, and metric schema being preserved.
- The chosen anisotropic encoder is intentionally simple. A later item may
  test alternative measurement encoders, but this lane locks one encoder so
  the comparison reflects the body alone after the shared latent.
- This lane uses a single seed (`42`). Multi-seed sensitivity is out of scope
  here and would require a follow-on item.
- The bundle remains additive candidate-lane evidence only. Promoting any
  shared-encoder row into manuscript evidence would require a checked-in
  evidence-package amendment outside this item.
