# Overlap Metrics Spec — Phase D (Normative)

Overview (Normative)
- Purpose: Define 2D disc-overlap metrics for overlap-driven sampling analysis and the API/outputs consumed by the dose/overlap study. Metrics quantify spatial redundancy at three levels with explicit controls via `s_img` (image subsampling), `n_groups` (number of solution regions), and `neighbor_count` (K nearest neighbors).
- Scope: Geometry-only metrics over scan coordinates in object-pixel units; independent of diffraction values. Supports `gridsize ∈ {1,2}`. Metric 1 applies to `gridsize=2` only.

Inputs & Parameters (Normative)
- Coordinates: `coords ∈ ℝ^{N×2}` with axis order `[x, y]` in pixels on the object grid; duplicates allowed.
- Gridsize: `gridsize ∈ {1,2}`.
- Subsampling: `s_img ∈ (0,1]` fraction of images kept; `rng_seed_subsample` seeds the deterministic subsampling.
- Group count: `n_groups ≥ 1` target number of solution regions produced by the grouping routine.
- Neighbor count: `neighbor_count = K ≥ 1` number of nearest neighbors for per-image/group averages.
- Probe diameter: `probe_diameter_px > 0` used as nominal disc diameter for overlap computations.

Definitions (Normative)
- Disc overlap fraction `f_overlap(d, D)` between two discs of diameter `D` at center distance `d`:
  - Area overlap: `A(d,D) = 2R²·arccos(d/2R) − (d/2)·√(4R²−d²)` for `d < D`, else `0`.
  - Fraction: `f_overlap(d,D) = A(d,D) / (πR²)`, with `R = D/2`.

Metrics (Normative)
- Metric 1 — Group‑based (gs=2 only):
  - For each sample (seed) within a group, compute distances `d` to other members, take up to `K` nearest, and average `f_overlap(d, D)` over that set; then average across all seeds/groups.
  - Undefined for `gridsize=1`; report `null`.

- Metric 2 — Image‑based (global):
  - Deduplicate images by exact `(x, y)` equality; for each unique image, take up to `K` nearest neighbors among the other unique images; compute mean `f_overlap(d, D)`; average across all unique images.

- Metric 3 — Group↔Group COM:
  - Compute per‑group centers of mass (COM). For each group, consider neighbor groups with COM distance `d < D`; compute mean `f_overlap(d, D)` over that neighbor set; average across groups. If a group has no overlapping neighbors, its contribution is `0`.

Subsampling & Grouping (Normative)
- Subsampling: Select `⌊s_img · N⌋` images without replacement using `rng_seed_subsample`; if result is `0`, raise an error.
- Grouping:
  - For `gridsize=1`: Assign each subsampled image as its own group until `n_groups` is reached (truncate if needed).
  - For `gridsize=2`: Produce `n_groups` groups via neighbor‑based grouping (e.g., K‑NN) allowing duplication of coordinates so that per‑group membership matches the required `C = gridsize²` structure.

Outputs (Normative)
- Per‑split metrics JSON:
  - Fields: `metrics_version`, `gridsize`, `s_img`, `n_groups`, `neighbor_count`, `probe_diameter_px`, `rng_seed_subsample`,
    `metric_1_group_based_avg|null`, `metric_2_image_based_avg`, `metric_3_group_to_group_avg`,
    `n_images_total`, `n_images_subsampled`, `n_unique_images`, `n_groups_actual`,
    `geometry_acceptance_bound`, `effective_min_acceptance`.
- Bundle JSON (train/test): `{ "train": <metrics JSON>, "test": <metrics JSON> }`.
- NPZ augmentation (optional): When writing filtered or baseline NPZs, `_metadata` SHOULD include the parameters above plus `metrics_version`. Implementations MAY preserve original arrays and attach metadata only (metrics still reflect subsampling/grouping applied internally).
    - `geometry_acceptance_bound` is the theoretical acceptance limit computed as `(bounding_box_area / (n_positions · disc_area))` capped at 10 %.
    - `effective_min_acceptance` clamps the bound to a small positive epsilon to avoid zero-floor downstream logic.

Error Conditions (Normative)
- Invalid parameters: `gridsize ∉ {1,2}`, `s_img ∉ (0,1]`, `n_groups < 1`, or `probe_diameter_px ≤ 0` SHALL raise.
- Empty subsample: if `⌊s_img · N⌋ = 0`, SHALL raise.

API & CLI Mapping (Informative)
- Python API: `studies.fly64_dose_overlap.overlap.compute_overlap_metrics(coords, gridsize, s_img, n_groups, neighbor_count, probe_diameter_px, rng_seed_subsample)` returns the metrics dataclass documented above.
- CLI: `python -m studies.fly64_dose_overlap.overlap ...` writes per‑split metrics JSON, a metrics bundle, and NPZs with `_metadata` containing the parameters.

Notes (Informative)
- Metrics are geometry‑only; they do not inspect diffraction values.
- Grouping policy is intentionally decoupled from Core KD‑tree grouping; for study parity, ensure neighbor selection (distance metric, seed handling) is documented and stable.
