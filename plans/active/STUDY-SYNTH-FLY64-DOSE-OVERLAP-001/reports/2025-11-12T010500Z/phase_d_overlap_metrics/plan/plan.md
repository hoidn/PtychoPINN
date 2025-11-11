# Phase D Overlap Metrics Implementation Hub (2025-11-12T010500Z)

## Reality Check — 2025-11-13
- `studies/fly64_dose_overlap/overlap.py` still performs dense/sparse spacing enforcement (`build_acceptance_mask`, `greedy_min_spacing_selection`, `compute_spacing_metrics`) and aborts when acceptance < geometry-aware minimum. No code exists for the spec-mandated overlap metrics (Metric 1/2/3) or explicit `s_img`/`n_groups` controls.
- CLI (`python -m studies.fly64_dose_overlap.overlap`) only exposes `--phase-c-root/--output-root/--views`; there are no knobs for gridsize, sampling fraction, group counts, neighbor count, or probe diameter inputs.
- Metrics bundle currently contains only spacing stats (min/mean spacing, acceptance) per split. There are no `metric_1_group_based_avg`, `metric_2_image_based_avg`, or `metric_3_group_to_group_avg` fields, nor do manifests/metadata record the explicit sampling parameters from specs/overlap_metrics.md.
- Tests in `tests/study/test_dose_overlap_overlap.py` reinforce the old behavior (dense vs sparse thresholds, spacing acceptance, geometry guard). There are zero tests covering disc overlap math, deduplication, COM grouping, or the new CLI arguments; the existing acceptance-floor test merely guards against the Phase G geometry bug.

## Do Now — Hand-off to Ralph
1. **Implement overlap metrics API** in `studies/fly64_dose_overlap/overlap.py`:
   - Add a reusable disc-overlap helper that matches specs/overlap_metrics.md (“two discs diameter D, distance d”).
   - Build Metric 1 (gs=2 only, per-sample mean overlap to neighbor_count group members), Metric 2 (global unique-image KNN), and Metric 3 (group COM overlap) with parameterized `neighbor_count` (default 6) and logged `probe_diameter_px`. Ensure Metric 1 is skipped for `gridsize=1`.
   - Use deterministic subsampling controlled by `s_img` + `rng_seed_subsample`, then honor the unified `n_groups` policy from docs/GRIDSIZE_N_GROUPS_GUIDE.md (gs=1 → one image per group; gs=2 → n_groups × gridsize² images with allowed duplication).

2. **Deprecate spacing gates/dense+sparse labels**:
   - Remove MIN_ACCEPTANCE_RATE, greedy fallback, and spacing-threshold failures. Runs should only abort for degenerate inputs (empty after subsample, invalid params).
   - Metadata (`_metadata`, metrics bundle) must record `gridsize`, `s_img`, `n_groups`, `neighbor_count`, `probe_diameter_px`, RNG seeds, and the Metric 1/2/3 averages. Retain historical spacing stats only if useful for regression but mark them as non-blocking.

3. **Refresh CLI and manifests**:
   - Replace `--views dense/sparse` with explicit flags: `--gridsize`, `--s-img`, `--n-groups`, optional `--neighbor-count`, `--probe-diameter-px`, `--rng-seed-subsample`. Keep `--phase-c-root`, `--output-root`, `--artifact-root` for hub copies.
   - CLI should iterate over requested dose × (gridsize, s_img, n_groups) tuples and drop per-split metrics JSON + aggregated `metrics_bundle.json` plus a manifest describing the sampling parameters and resulting overlap stats.

4. **Update tests + documentation**:
   - Rewrite `tests/study/test_dose_overlap_overlap.py` to cover the disc overlap helper, Metric 1/2/3 calculations (gs=1 skip Metric 1), parameter plumbing, and bundle schema. Remove spacing-threshold assertions and dense/sparse semantics; add fixtures for deterministic subsampling and group COM neighbors.
   - Extend the module’s CLI test(s) so the new arguments round-trip and the metrics bundle contains the new fields with expected values, including the omission of Metric 1 for gs=1.
   - Update `plans/active/.../test_strategy.md` if selectors or artifacts change once tests exist.

5. **Evidence expectations for this hub** (root = `$HUB = plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_d_overlap_metrics/`):
   - `green/pytest_phase_d_overlap.log` capturing `pytest tests/study/test_dose_overlap_overlap.py -vv` with the new selectors passing.
   - CLI run logs demonstrating the new arguments (place under `cli/phase_d_overlap_metrics.log`). Include command lines with `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` + `HUB` exports per policy.
   - Updated metrics artifacts: per-split JSON (`train_metrics.json`, `test_metrics.json`) and `metrics_bundle.json` showing Metric 1/2/3, sampling parameters, and RNG seeds.
   - If sampling/generation helpers emit additional summaries, record them under `analysis/` and add an `artifact_inventory.txt` listing key files.

## References
- `docs/index.md` — authoritative doc map (docs/prompt_sources_map.json is still missing).
- `docs/findings.md` — POLICY-001, CONFIG-001, DATA-001, OVERSAMPLING-001, ACCEPTANCE-001 guardrails.
- `specs/overlap_metrics.md` — canonical metric definitions + CLI expectations.
- `docs/GRIDSIZE_N_GROUPS_GUIDE.md` — unified `n_groups` policy.
- `plans/active/.../implementation.md` §Phase D — spec-adopted objectives.
- `plans/active/.../test_strategy.md` — selectors + bundle requirements.
- `docs/TESTING_GUIDE.md` & `docs/development/TEST_SUITE_INDEX.md` — command references for pytest selectors.
