Summary: Launch Phase D by implementing dense/sparse overlap filtering and tests for the fly64 dose study.

Mode: TDD

Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase D (Group-Level Overlap Views)

Branch: feature/torchapi-newprompt

Mapped tests: pytest tests/study/test_dose_overlap_overlap.py -k spacing_filter -vv; pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_paths -vv

Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.D:
  - Implement: studies/fly64_dose_overlap/overlap.py::generate_overlap_views — build dense/sparse filtering pipeline (with helper `compute_spacing_matrix`/`build_acceptance_mask`) that consumes Phase C outputs, enforces StudyDesign spacing thresholds, preserves DATA-001 contract, and emits metrics + manifests under the Phase D artifact hub.
  - Test: tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_paths — begin RED expecting spacing enforcement failure, then drive to GREEN so `pytest tests/study/test_dose_overlap_overlap.py -k spacing_filter -vv` passes; capture red/green/collect logs in `phase_d_overlap_filtering/{red,green,collect}/`.
  - Document: Refresh `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/{implementation.md,test_strategy.md}` with Phase D implementation details, update `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/summary.md` (spacing stats + findings), and append Attempt #7 to `docs/fix_plan.md` noting artifact paths, CLI log, and spacing metrics JSON.
  - Validating selector: pytest tests/study/test_dose_overlap_overlap.py -k spacing_filter -vv
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/

Priorities & Rationale:
  - docs/GRIDSIZE_N_GROUPS_GUIDE.md:154 — spacing formula S ≈ (1 − f_overlap) × N defines dense/sparse acceptance thresholds.
  - docs/SAMPLING_USER_GUIDE.md:118 — OVERSAMPLING-001 mandates neighbor_count ≥ gridsize² when forming grouped views.
  - specs/data_contracts.md:207 — DATA-001 NPZ contract must hold after filtering to avoid downstream loader regressions.
  - studies/fly64_dose_overlap/design.py:37 — StudyDesign spacing thresholds + seeds are authoritative for deterministic filtering.
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/plan.md:15 — Tasks D1–D4 set deliverables, logs, and metrics routing for this loop.

How-To Map:
  - export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
  - mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/{red,green,collect,metrics}
  - pytest tests/study/test_dose_overlap_overlap.py -k spacing_filter -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/red/pytest.log
  - After implementing overlap.py + tests, rerun the selector with tee → `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/green/pytest.log`
  - pytest tests/study/test_dose_overlap_overlap.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/collect/pytest_collect.log
  - python -m studies.fly64_dose_overlap.overlap --phase-c-root data/studies/fly64_dose_overlap --output-root data/studies/fly64_dose_overlap_views 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/dense_sparse_generation.log
  - Move spacing metrics JSON (per dose/view) into `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/metrics/` and summarize results in summary.md.

Pitfalls To Avoid:
  - Do not mutate params.cfg or call legacy bridges inside overlap utilities (CONFIG-001 boundary).
  - Avoid loading full-resolution NPZs inside pytest unless necessary; use synthetic arrays for speed.
  - Keep RNG seeding deterministic (`StudyDesign.rng_seeds['grouping']`) to stabilize expectations.
  - Ensure validator invoked with `view` argument so spacing thresholds are enforced.
  - Do not overwrite Phase C source NPZs; write dense/sparse outputs to new directories.
  - Record red/green/collect logs; missing evidence blocks completion.
  - Preserve DATA-001 keys/dtypes when writing filtered NPZs; no ad-hoc fields beyond documented metadata.
  - Capture acceptance/rejection counts for dense/sparse views and include them in metrics JSON.

If Blocked:
  - Log failing command output to the artifact directory (e.g., `dense_sparse_generation.log`), note blocker + minimal stack trace in summary.md, and record Attempt #7 in docs/fix_plan.md as blocked with return conditions.

Findings Applied (Mandatory):
  - CONFIG-001 — Keep overlap utilities pure (no params.cfg mutation or legacy module side effects).
  - DATA-001 — Validate filtered NPZs so downstream loaders ingest canonical keys/dtypes.
  - OVERSAMPLING-001 — Maintain neighbor_count ≥ gridsize² when forming grouped views.

Pointers:
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/plan.md:15
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:96
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:70
  - docs/GRIDSIZE_N_GROUPS_GUIDE.md:141
  - specs/data_contracts.md:207

Next Up (optional):
  - Prep Phase E training plan once dense/sparse datasets validate.

Doc Sync Plan:
  - After GREEN, append the new selector to docs/TESTING_GUIDE.md §Study suite and update docs/development/TEST_SUITE_INDEX.md; archive collect-only output under `phase_d_overlap_filtering/collect/pytest_collect.log`.

Mapped Tests Guardrail:
  - Confirm `pytest tests/study/test_dose_overlap_overlap.py --collect-only -vv` reports ≥1 test; if collection fails, pause implementation, capture the log, and mark Attempt #7 as blocked pending fix.
