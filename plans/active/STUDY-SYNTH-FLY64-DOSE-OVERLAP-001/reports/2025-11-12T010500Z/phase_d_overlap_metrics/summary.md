# Phase D Overlap Metrics Implementation — Loop Summary

**Loop timestamp:** 2025-11-11  
**Agent:** Ralph (implementation)  
**Focus:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase D overlap metrics adoption  
**Status:** Implementation complete — GREEN tests, ready for CLI execution

---

## Acceptance Focus & Module Scope

**Acceptance focus:** AT-OVERLAP-METRICS-001 (implement Metric 1/2/3 per specs/overlap_metrics.md)  
**Module scope:** algorithms/numerics (disc overlap math, metric aggregation)

**Verification:** Stayed within single module category ✅

---

## SPEC Compliance (specs/overlap_metrics.md)

**Implemented sections:**
- §2D Overlap Definition (disc overlap area and normalized fraction)
- §Metrics (Metric 1 group-based gs=2, Metric 2 image-based, Metric 3 group↔group COM)
- §Parameters (gridsize, s_img, n_groups, neighbor_count, probe_diameter_px, rng_seed_subsample)
- §Outputs (metrics_version, sampling parameters, Metric 1/2/3 averages, size counts)
- §API and CLI (Python API + thin CLI wrapper)

**File pointers:**
- studies/fly64_dose_overlap/overlap.py:77-143 (disc overlap functions)
- studies/fly64_dose_overlap/overlap.py:258-417 (Metric 1/2/3 implementations)
- studies/fly64_dose_overlap/overlap.py:420-550 (compute_overlap_metrics)
- studies/fly64_dose_overlap/overlap.py:691-830 (CLI)

---

## Implementation Changes

**File:** studies/fly64_dose_overlap/overlap.py

**Added:**
- disc_overlap_area, disc_overlap_fraction (2D analytical formulas)
- subsample_images (deterministic RNG-based subsampling)
- form_groups_gs1, form_groups_gs2 (grouping with duplication tracking)
- compute_metric_1_group_based (per-sample mean overlap to K group neighbors)
- compute_metric_2_image_based (global image-based with deduplication)
- compute_metric_3_group_to_group (group↔group COM-based)
- compute_overlap_metrics (primary Python API)
- generate_overlap_views (integration function)
- CLI with explicit parameters

**Removed:**
- MIN_ACCEPTANCE_RATE and greedy spacing fallback
- compute_spacing_matrix, build_acceptance_mask, greedy_min_spacing_selection
- compute_geometry_aware_acceptance_floor
- All spacing/packing acceptance gates

**Delta:** ~959 insertions, ~1048 deletions (complete overhaul)

---

## Testing

**Full suite:** 18 passed in 1.63s ✅

**Evidence:** `green/pytest_phase_d_overlap.log`

**Primary selector:** tests/study/test_dose_overlap_overlap.py::test_overlap_metrics_bundle

**Coverage:**
- Unit: disc overlap (perfect/half/no overlap, area symmetry/monotonicity)
- Unit: Metric 1 (synthetic groups)
- Unit: Metric 2 (deduplication, single-image edge case)
- Unit: Metric 3 (overlapping/non-overlapping COMs)
- Integration: compute_overlap_metrics (gs=1/gs=2, degenerate inputs)
- Integration: generate_overlap_views (basic execution, bundle validation)

---

## Version Control

**Commits:**
- d94f24f7 (implementation + tests, 18 passed)
- b547bb60 (GREEN evidence + artifact inventory)

**Push:** ✅ to origin/feature/torchapi-newprompt

---

## Next Steps

Execute CLI with real Phase C data:

```bash
HUB="$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_d_overlap_metrics"

python -m studies.fly64_dose_overlap.overlap \
  --phase-c-root data/phase_c/dose_1000 \
  --output-root tmp/phase_d_overlap \
  --artifact-root "$HUB" \
  --gridsize 2 \
  --s-img 0.8 \
  --n-groups 512 \
  --neighbor-count 6 \
  --probe-diameter-px 38.4 \
  --rng-seed-subsample 456 \
  |& tee "$HUB"/cli/phase_d_overlap_metrics.log
```

Then validate Metric 1/2/3 in metrics_bundle.json and update ledgers.
