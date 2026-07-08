# 2026-02-12 Session Findings: Hybrid-ResNet, AMP, SHARP vs LSQML

## Scope
This note summarizes the concrete performance and solver-comparison findings from the interactive debugging/performance session on 2026-02-12.

## 1) Hybrid-ResNet forward cost (local benchmark)
- Context: `HybridResnetGeneratorModule.forward` benchmarked on RTX 3090 with `N=128`, `B=16`, `C=4` (fallback spectral path).
- Measured forward time: about `8.43 ms` per forward pass.
- Main hotspot: ResNet bottleneck stage, about `42.6%` of forward runtime.
- Observed scaling: moving from `N=64` to `N=128` was about `3x` slower (not idealized `4x`) due to fixed overheads and kernel efficiency effects.

### Layer-level timings (explicit)
Benchmarked with CUDA events on RTX 3090, `N=128`, `C=4`, default `hybrid_resnet` config (`hidden_channels=32`, `fno_blocks=4`, `resnet_blocks=6`, `modes=12`), inference mode.

#### Primary profile (`B=16`)
- End-to-end forward mean: `8.461 ms` (`std=0.007 ms`)
- Stage aggregates:
  - `lifter`: `0.559 ms` (`6.6%`)
  - `encoder path` (4 spectral blocks + 2 downsamples): `3.276 ms` (`38.7%`)
  - `resnet bottleneck` (6 blocks @ 32x32): `3.691 ms` (`43.6%`)
  - `decoder path` (`up1`, `up2`, `output_proj`, reshape): `1.007 ms` (`11.9%`)

- Per-block breakdown:
  - `encoder_block0`: `1.158 ms`
  - `downsample0`: `0.147 ms`
  - `encoder_block1`: `0.672 ms`
  - `downsample1`: `0.130 ms`
  - `encoder_block2`: `0.584 ms`
  - `encoder_block3`: `0.585 ms`
  - `resnet_block0`: `0.616 ms`
  - `resnet_block1`: `0.615 ms`
  - `resnet_block2`: `0.616 ms`
  - `resnet_block3`: `0.615 ms`
  - `resnet_block4`: `0.616 ms`
  - `resnet_block5`: `0.614 ms`
  - `up1`: `0.326 ms`
  - `up2`: `0.596 ms`
  - `output_proj`: `0.081 ms`

#### Reference profile (`B=8`)
- End-to-end forward mean: `4.390 ms` (`std=0.011 ms`)
- ResNet bottleneck share: `~41.7%` of total.
- Confirms same bottleneck dominance pattern as `B=16`.

### Convolution vs spectral breakdown (fine-grained)
This decomposition times sub-ops (spectral, local conv, pads/norms/activations, etc.) inside blocks. Because of extra instrumentation overhead, component sums are slightly above end-to-end forward time; use shares as approximate.

- End-to-end forward (`B=16`): `8.449 ms`
- Decomposed totals:
  - `conv` ops total: `4.994 ms`
  - `spectral` ops total: `1.439 ms`
  - `other` ops (padding, norms, activations, residual adds, reshape): `2.272 ms`
  - Decomposed component sum: `8.705 ms`

- Share of decomposed sum (normalized):
  - `conv`: `57.4%`
  - `spectral`: `16.5%`
  - `other`: `26.1%`

- Spectral implementation in this environment:
  - `_FallbackSpectralConv2d` (FFT/einsum path), i.e. neuraloperator spectral kernel was not active.

- Major contributors:
  - ResNet convs (`rb*_conv1/conv2`) total: `2.636 ms`
  - Encoder spectral (`enc*_spectral`) total: `1.439 ms`
  - Encoder local conv (`enc*_local_conv`) total: `0.933 ms`
  - Largest single conv kernels: `up2_convT` (`0.390 ms`), `enc0_local_conv` (`0.296 ms`), `lifter_conv2` (`0.296 ms`)

## 2) AMP and inference acceleration constraints
- ResNet blocks themselves are AMP-friendly in principle.
- End-to-end AMP is currently blocked by the hybrid complex FFT/einsum fallback path (ComplexHalf/BF16 support limitations in the relevant ops/path).
- AMP does not need to be all-or-nothing: selective autocast around AMP-safe subgraphs remains viable while keeping numerically/compatibility-sensitive complex ops in FP32.

## 3) SHARP throughput evidence (paper-derived)
- SHARP paper evidence indicates 10,000 frames of `128x128` reconstructed in under 2 seconds on 16 GTX Titan GPUs.
- Derived throughput:
  - Cluster-level: `>5000 frames/s`.
  - Linear per-GPU back-of-envelope: `>312 frames/s` (estimate only; real scaling depends on implementation and workload details).

## 4) Which LSQML path pty-chi is using here
- Local wrapper uses pty-chi LSQML options via `api.LSQMLOptions()` with Gaussian noise model and `batch_size=96`.
- Installed `ptychi` package (`v1.2.0`) documents LSQ-ML (Odstrcil 2018), with autodiff gradients and analytical step-size solving.
- Default batching mode is random unless overridden.

## 5) Is “LSQML/rPIE are 2x+ slower than SHARP” always true?
- Not as a universal claim.
- Based on the reviewed benchmark tables (normalized runtime metric), SHARP is often faster, but optimized LSQML/PIE-family implementations can be near parity to moderately slower depending on dataset and settings (roughly parity to ~1.6x in the cited comparisons, not always `2x+`).

## 6) Clarification of the `0.5 ns` metric
- `0.5 ns` in the cited benchmarking context is a **normalized** performance quantity, not wall-clock full reconstruction latency.
- It is reported as nanoseconds per normalized work unit (iteration/position/probe-pixel style normalization in that benchmark framework), so it should not be read as “0.5 ns per full image reconstruction”.

## Source pointers used during session
- Local integration script: `scripts/reconstruction/ptychi_reconstruct_tike.py:107`
- Local installed package:
  - `/home/ollie/miniconda3/envs/ptycho311/lib/python3.11/site-packages/ptychi/reconstructors/lsqml.py:30`
  - `/home/ollie/miniconda3/envs/ptycho311/lib/python3.11/site-packages/ptychi/api/options/base.py:936`
- Benchmark artifacts generated in this session:
  - `tmp/benchmarks/hybrid_resnet_layer_profile_n128_b16_c4_events.json`
  - `tmp/benchmarks/hybrid_resnet_layer_profile_n128_b8_c4_events.json`
  - `tmp/benchmarks/hybrid_resnet_conv_vs_spectral_n128_b16_c4.json`
- External references:
  - SHARP paper: `https://arxiv.org/abs/1602.01448`
  - PtychoShelves performance discussion/tables: `https://pmc.ncbi.nlm.nih.gov/articles/PMC7133065/`
