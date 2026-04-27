# Backlog: PDEBench FFNO-Style Bottleneck Variant

**Created:** 2026-04-20
**Status:** Paused
**Priority:** Medium
**Related:** `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_resnet_bottleneck_design.md`, `scripts/studies/pdebench_image128/models.py`, `ptycho_torch/generators/hybrid_resnet.py`, `ptycho_torch/generators/resnet_components.py`
**Impacts:** PDEBench architecture experiments, CNS-focused operator design, benchmark-table clarity

## Summary

The original follow-on idea after the Darcy adapter work was a bottleneck-only F-FNO-style variant for the PDEBench image-suite path:

- keep the current supervised PDEBench shell,
- replace only the constant-resolution `ResnetBottleneck`,
- add a deep factorized Fourier stack,
- use shared spectral weights across bottleneck depth,
- expose the result as a separate model family rather than `hybrid_resnet`.

That idea remains scientifically interesting, especially for 2D Compressible Navier-Stokes, but it is no longer the active design direction. The active design has narrowed to a simpler and more interpretable variant:

- keep the current ResNet `3x3` conv bottleneck body,
- add a shared factorized spectral residual branch,
- expose it as `spectral_resnet_bottleneck_net`.

See `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_resnet_bottleneck_design.md` for the current active design.

## Why This Is Paused

The full FFNO-style bottleneck is a larger step than the current campaign needs for the next experiment. In particular, it would require:

1. a new factorized/separable Fourier operator implementation,
2. an FFNO-style bottleneck block design rather than reuse of the current local ResNet bottleneck body,
3. explicit design choices around residual placement relative to the local branch,
4. a broader config surface and larger interpretation burden.

That makes it harder to isolate what changed. The spectral-ResNet variant is currently a better next move because it preserves the repo's existing local bottleneck behavior and changes only the added global-mixing path.

## What This Variant Would Contain

If resumed later, the FFNO-style bottleneck variant should include:

- a `FactorizedSpectralConv2d` operator,
- a deep bottleneck stack with shared spectral weights across layers,
- a distinct model family name outside `hybrid_resnet_*`,
- the same PDEBench image-suite data, split, normalization, and reporting protocol used by the primary local comparisons,
- clear caveats that it is still a bottleneck-only adaptation, not a full paper-faithful full-resolution F-FNO baseline.

Two plausible forms were discussed:

1. pure FFNO-style bottleneck stack replacing the local ResNet bottleneck entirely;
2. a hybrid bottleneck with FFNO-style spectral sublayers plus the existing local ResNet branch.

The second form was later rejected in favor of keeping the local branch purely ResNet `3x3` conv-based for the active design.

## Resume Conditions

Resume this backlog only if at least one of these becomes true:

1. the spectral-ResNet bottleneck variant underperforms in a way that suggests the missing FFNO-style bottleneck structure is the limiting factor;
2. CNS benchmarking becomes the top PDEBench priority and a deeper spectral bottleneck is worth the extra engineering cost;
3. the repo grows a reusable factorized Fourier layer that reduces implementation risk for a fuller FFNO-style bottleneck.

## Acceptance Criteria If Resumed

1. The new model family remains clearly named outside `hybrid_resnet_*`.
2. The implementation provides explicit tests for factorized spectral shape preservation and cross-layer weight sharing.
3. The PDEBench image-suite model builder can instantiate the variant without breaking existing `hybrid_resnet_base`, `fno_base`, or `unet_strong` rows.
4. Any reported comparison against published F-FNO CNS numbers carries the caveat that this is a bottleneck-only adaptation unless a full encoder rewrite also lands.

## Suggested Next Step

Do not execute this now. Revisit it only after the active `spectral_resnet_bottleneck_net` path has shape tests, builder tests, and at least one Darcy readiness result.
