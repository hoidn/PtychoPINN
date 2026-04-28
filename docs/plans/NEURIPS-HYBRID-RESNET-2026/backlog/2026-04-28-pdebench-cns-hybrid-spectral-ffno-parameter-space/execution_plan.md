# CNS Hybrid-Spectral To FFNO Parameter-Space Backlog Plan

## Goal

Explore intermediate architecture points between Hybrid-spectral and FFNO on
PDEBench `2d_cfd_cns` only, under the local capped comparison contract.

## Motivation

The completed CNS Hybrid-spectral architecture ablation and FFNO
local-convolutional-feature item define useful endpoints, but they do not yet
separate which architecture-family ingredients matter most: downsampling,
decoder shape, spectral bottleneck structure, local convolutional features, and
factorized Fourier depth.

This plan keeps the Phase 2 work honest by answering that question for CNS
without pulling in Phase 3 CDI evidence.

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

Required domain:

- PDEBench `2d_cfd_cns` under the local capped comparison contract.

## Boundaries

- Do not run CDI/ptycho rows from this item.
- Do not mix this item with full-training benchmark claims. Capped rows are
  decision-support evidence unless a later full-training item reruns them on the
  full available training split.
- Do not start this before the narrower CNS Hybrid-spectral architecture
  ablation and FFNO-convolutional-feature item have clarified the immediate
  design space.
- Keep each row attributable: change one architecture-family axis at a time
  within a staged matrix.

## Success Criteria

- The final summary distinguishes encoder/downsampling, decoder, and bottleneck
  effects on CNS.
- Each reported row records its anchor, cap, model contract, and changed
  architecture axis.
- The study records which intermediate points are worth promoting into later
  full-training CNS or Phase 3 CDI comparisons.
