# Hybrid-Spectral To FFNO Parameter-Space Backlog Plan

## Goal

Explore intermediate architecture points between Hybrid-spectral and FFNO, with
controlled comparisons on both PDEBench CNS and CDI/ptycho.

## Motivation

The current repo has several endpoints:

- Hybrid ResNet with local convolutional shell and spectral pieces
- Hybrid-spectral bottleneck variants
- local FFNO-close and authored FFNO baselines

The useful question is not just which endpoint wins, but which architectural
ingredients matter: encoder/downsampling, decoder, bottleneck, local
convolutional features, and factorized Fourier depth.

## Scope

Run this as a staged architecture study, not a Cartesian sweep.

Required axes:

- encoder/downsampling:
  - Hybrid-style downsampled shell
  - reduced or removed downsampling where feasible
- decoder:
  - canonical Hybrid decoder shell
  - FFNO-like direct projection or lighter decoder where feasible
- bottleneck:
  - Hybrid-spectral bottleneck
  - FFNO-style bottleneck
  - intermediate local-plus-factorized variants

Required domains:

- PDEBench `2d_cfd_cns` under the local capped comparison contract.
- CDI/ptycho using the best study-indexed lines configuration.

## Boundaries

- Do not mix CNS and CDI results into one scalar ranking. Report each domain
  separately, then identify architecture patterns that transfer or fail to
  transfer.
- Do not start this before the narrower CNS Hybrid-spectral architecture
  ablation and FFNO-convolutional-feature item have clarified the immediate
  design space.
- Keep each row attributable: change one architecture-family axis at a time
  within a staged matrix.

## Success Criteria

- The final summary distinguishes domain-specific winners from architecture
  ingredients that appear robust across both CNS and CDI/ptycho.
- Encoder/downsampling, decoder, and bottleneck effects are each reported
  separately.
- The study records which intermediate points are worth promoting into future
  default or paper-facing comparisons.
