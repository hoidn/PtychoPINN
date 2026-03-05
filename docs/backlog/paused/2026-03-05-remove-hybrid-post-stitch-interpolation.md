# Backlog: Remove Post-Stitch Interpolation That Fills Unsupported Hybrid Canvas Regions

**Created:** 2026-03-05
**Status:** Open
**Priority:** High
**Related:** `scripts/studies/grid_lines_torch_runner.py`, `ptycho/image/harmonize.py`, `scripts/studies/grid_lines_compare_wrapper.py`
**Impacts:** hybrid reconstruction support semantics, cross-model visual parity, metric trustworthiness

## Summary
The current hybrid reconstruction path can interpolate stitched outputs to match ground-truth shape, which introduces nonzero values outside physically supported regions. This produces visually misleading full-canvas signal and weakens interpretation of cross-model comparisons.

Create a support-preserving compare contract so post-stitch harmonization does not invent signal in unsupported regions.

## Why
- Position stitching has a finite support footprint determined by probe-position span and effective patch size.
- Interpolation-based harmonization can spread nonzero values across the full target canvas, masking true support boundaries.
- This behavior causes apples-to-oranges comparisons against methods that retain explicit zero support outside sampled regions.

## Proposed Contract
1. Preserve stitched support semantics by default:
   - no interpolation that creates nonzero values outside occupied support.
2. If shape harmonization is required for compare/metrics:
   - apply a canonical support mask after harmonization, or
   - compute metrics in a common native-support window.
3. Emit support metadata in manifests:
   - stitched support bounds/mask stats,
   - harmonization mode used,
   - nonzero fraction outside support (should be zero under strict mode).
4. Keep legacy interpolation path only as an explicit opt-in compatibility mode.

## Acceptance Criteria
1. Hybrid outputs in strict mode have zero-valued regions outside stitched support (numerically tolerant threshold).
2. Compare visuals for `gt`, `pinn_hybrid_resnet`, and `pinn_ptychovit` use a documented common-support policy.
3. Metrics are either:
   - computed inside shared support only, or
   - reported with explicit support policy labels.
4. Regression tests cover:
   - support preservation after stitching/harmonization,
   - no unintended nonzero spread to full canvas,
   - manifest/runtime-contract support fields.

## Risks / Open Questions
1. Whether strict support masking should be default globally or only for study/compare flows.
2. How to handle legacy artifacts that relied on interpolation-smoothed boundaries.
3. Exact policy for combining different native canvases (e.g., 512 vs 1024) without biasing any model arm.

## Suggested Next Step
Prototype a compare-boundary support policy first (without changing core model/reassembly internals), then promote to default behavior after side-by-side metric and visual validation on the NERSC scan807+cameraman study artifacts.
