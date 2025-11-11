# Overlap Metrics Specification (Phase D)

This specification defines overlap-driven sampling and reporting for Phase D of the fly64 dose/overlap study. It replaces spacing/packing acceptance gates and the legacy “dense/sparse” labels with explicit controls over sampling and three normative overlap metrics.

## Scope and Goals

- Control sampling via two explicit inputs: image subsampling fraction (`s_img`) and number of sampled groups (`n_groups`).
- Compute and report measured overlap fractions; do not fail runs based on geometric packing limits. Abort only for degenerate inputs (e.g., empty after subsampling).
- Support both `gridsize=1` and `gridsize=2`; prefer `gridsize=1` first for stability. Grouping semantics follow the unified `n_groups` policy in `docs/GRIDSIZE_N_GROUPS_GUIDE.md`.
- Provide both a Python API and a thin CLI layer; Python API is preferred for programmatic workflows.

## Parameters

- `gridsize` (int): 1 or 2 for this study.
- `s_img` (float, 0<≤1]: Fraction of images retained after deterministic subsampling.
- `n_groups` (int): Number of groups to produce. Interpretation follows the unified policy:
  - `gridsize=1`: one image per group; effectively couples to the number of retained images. Sweeping `n_groups` is optional and typically equals the retained sample count.
  - `gridsize=2`: each group contains `gridsize^2` images formed by KNN; duplication across groups is allowed and follows existing behavior.
- `neighbor_count` (int): K for neighbor-based averages. Default K=6 (not counting the central/seed image). Parameterized.
- `probe_diameter_px` (float): Nominal probe diameter in pixels used for overlap. Default derived from probe FWHM when available; otherwise a documented constant fallback (e.g., `0.6 × N`). Must be recorded in outputs.
- `rng_seed_subsample` (int): RNG seed for deterministic `s_img` subsampling; default sourced from the study design (same seed policy as existing code paths).

## 2D Overlap Definition

For two axis-aligned discs in the object plane with common diameter `D = probe_diameter_px` and centers separated by distance `d`:

- If `d ≥ D`, overlap area = 0.
- Else, with radius `R = D/2`, the overlap area is:

  `A_overlap(d) = 2 R^2 cos⁻¹(d / (2R)) − (d/2) √(4 R^2 − d^2)`

- The normalized overlap fraction is:

  `f_overlap(d) = A_overlap(d) / (π R^2)`

Distances are computed in the same pixel coordinate system as Phase C `xcoords/ycoords`. When object/world scaling metadata is available, the pixel geometry must remain internally consistent (i.e., use the same units for coordinates and diameter).

## Metrics

All metrics are computed per split (train/test) and reported as a global average plus configuration parameters. Additional distributions (e.g., per-sample/per-group means or histograms) are optional add-ons.

1) Metric 1 — Group-based (gs=2 only)
   - For each sample (seed) within a group, compute the mean of `f_overlap(d)` to its `neighbor_count` group neighbors (central-to-neighbors only; do not compute all pairwise neighbor pairs).
   - Average this per-sample mean across all samples/groups.
   - Not applicable for `gridsize=1`.

2) Metric 2 — Image-based (global)
   - Deduplicate images by exact `(x, y)` equality (no tolerance).
   - For each unique image, compute the mean of `f_overlap(d)` to its `neighbor_count` nearest neighbors in the global set (exclude self).
   - Average across all unique images.

3) Metric 3 — Group ↔ Group (COM-based)
   - Compute a center-of-mass (COM) coordinate for each group (mean of member `(x, y)` in pixel coordinates).
   - For each group, find all other groups whose COM distance `d` satisfies `d < probe_diameter_px` (disc overlap > 0). Compute the mean of `f_overlap(d)` over this neighbor set.
   - Average across groups. If a group has zero overlapping neighbors, its contribution is 0 by definition.

Notes
- Group formation for `gridsize=2` follows the existing KNN-based approach (sample-then-group). Duplication of images across different groups is allowed and unchanged by this spec.
- For `gridsize=1`, `n_groups` couples to `n_samples` after subsampling and Metric 1 is skipped.

## Behavior and Failure Policy

- No geometry/spacing acceptance gating or greedy spacing fallback must be used.
- Pipelines must proceed and report measured overlaps even when the average overlap is near zero.
- Only abort early for degenerate inputs (e.g., zero images after `s_img` subsampling, missing keys, or impossible parameter combinations).

## Outputs (per split)

Writers must record:
- `metrics_version`: semantic version string (e.g., `"1.0"`).
- `gridsize`, `s_img`, `n_groups`, `neighbor_count`, `probe_diameter_px`, `rng_seed_subsample`.
- `metric_1_group_based_avg` (gs=2 only; omit for gs=1)
- `metric_2_image_based_avg`
- `metric_3_group_to_group_avg`
- Optional: size counts (images, unique images, groups), and brief summaries (e.g., neighbor coverage rates).

## API and CLI

Python API (preferred)
- `compute_overlap_metrics(coords, gridsize, s_img, n_groups, neighbor_count=6, probe_diameter_px=None, rng_seed_subsample=None) -> dict`
  - Returns a dict containing global averages for Metric 2 and Metric 3; includes Metric 1 when `gridsize=2` and grouping is provided/constructed.
  - Must not mutate global configuration or rely on legacy params.

CLI (thin wrapper; parity with API)
- Required: `--gridsize`, `--simg`, `--ngroups`
- Optional: `--neighbor-count`, `--probe-diameter-px`, `--rng-seed-subsample`
- Output: per-split metrics JSON and an aggregated bundle JSON.

## Compatibility and Documentation

- `docs/GRIDSIZE_N_GROUPS_GUIDE.md`: dense/sparse labels are deprecated for this study; inter-group control is explicitly via `s_img` and `n_groups`. Keep unified `n_groups` meaning across gridsizes. Link to this spec.
- `docs/index.md`: add entry under Specifications linking to this file.
- `docs/TESTING_GUIDE.md` and `docs/COMMANDS_REFERENCE.md`: when updated, reference explicit `s_img`/`n_groups` controls and measured-overlap reporting.

## Testing Strategy (to be implemented in code/tests)

- Unit: disc-overlap function vs analytically known cases (d=0 → 1.0, d=R → ~0.391..., d≥D → 0.0).
- Unit: Metric 1 aggregation on synthetic groups (`gridsize=2`).
- Unit: Metric 2 with deduplication behavior (exact equality).
- Unit: Metric 3 neighbor detection via COM thresholds.
- Integration: gs=1 skip Metric 1; gs=2 compute all three; no spacing gating; degenerate cases handled.

