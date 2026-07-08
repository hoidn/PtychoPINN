# PDEBench CNS FFNO Convolutional Features Backlog Plan

## Goal

Test whether adding local convolutional features to the FFNO baseline improves
PDEBench `2d_cfd_cns` performance on the local capped comparison contract.

## Motivation

The authored FFNO baseline and the local FFNO-close bottleneck isolate
factorized Fourier behavior, while Hybrid-spectral keeps explicit local
convolutional processing. This item asks whether FFNO's local performance is
partly limited by missing local feature extraction rather than by the Fourier
operator itself.

## Scope

- Keep the current local CNS contract fixed:
  - `128x128`
  - `history_len=2`
  - capped `512 / 64 / 64` trajectories
  - `8` windows per trajectory
  - batch size `4`
  - existing CNS metrics and galleries
- Add one or more FFNO variants with local convolutional features, such as:
  - pre-FFNO convolutional stem
  - per-block local convolutional residual branch
  - shallow decoder-side convolutional refinement
- Compare against authored FFNO, local FFNO-close, and Hybrid-spectral anchors.

## Boundaries

- Do not change the CNS dataset, split, normalization, metric family, or epoch
  budget to make the variant look better.
- Do not fold this into the Hybrid-spectral architecture ablation; this is an
  FFNO-family extension.
- Keep the local convolutional addition explicit in profile names and model
  provenance.

## Success Criteria

- The new row can be built and profiled without changing existing FFNO,
  Hybrid-spectral, FNO, or U-Net profiles.
- The comparison reports whether convolutional features improve `relative_l2`
  and whether they help or hurt `fRMSE_high`.
- The summary states whether any gain is worth the extra parameter and
  implementation cost.
