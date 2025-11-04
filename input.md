Summary: Enable sparse Phase D overlap generation by adding greedy spacing selection so Phase F LSQML can consume emitted NPZs.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.D7 — Phase D sparse overlap downsampling rescue
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_sparse_downsamples -vv; pytest tests/study/test_dose_overlap_overlap.py -k overlap -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T034500Z/phase_d_sparse_downsampling_fix/

Do Now:
- Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.D7): studies/fly64_dose_overlap/overlap.py::generate_overlap_views — wire a deterministic greedy spacing fallback + metadata and author RED test tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_sparse_downsamples covering the 64 px spacing scenario.
- Validate: pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_sparse_downsamples -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T034500Z/phase_d_sparse_downsampling_fix/green/pytest_sparse_downsamples_green.log
- Collect: pytest tests/study/test_dose_overlap_overlap.py --collect-only -k sparse_downsamples -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T034500Z/phase_d_sparse_downsampling_fix/collect/pytest_sparse_downsamples_collect.log
- Artifacts: Capture RED log for the pre-fix failure (`.../red/pytest_sparse_downsamples_red.log`), summarize acceptance metrics + selection strategy in `.../docs/summary.md`, and update docs/fix_plan.md Attempt #85 with links.

Priorities & Rationale:
- docs/fix_plan.md:31 — Active focus still lacks sparse LSQML because Phase D overlap rejects all sparse positions; unblock by emitting salvageable sparse NPZs.
- studies/fly64_dose_overlap/overlap.py:330 — Current guard raises when acceptance <10%; introduce selective downsampling instead of immediate abort when a valid subset exists.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/summary.md — Training CLI evidence flagged sparse overlap absence as root blocker; fixing Phase D removes that bottleneck.
- docs/GRIDSIZE_N_GROUPS_GUIDE.md:143 — Spacing formula (S ≈ (1 − f) × N) mandates 102.4 px for sparse view; greedy sampler must honor this constraint while selecting a subset.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- RED: pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_sparse_downsamples -vv --maxfail=1 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T034500Z/phase_d_sparse_downsampling_fix/red/pytest_sparse_downsamples_red.log
- Implement helper (e.g., greedy_min_spacing_mask) and metadata note, then re-run GREEN validate command (tee to green log).
- Run pytest tests/study/test_dose_overlap_overlap.py -k overlap -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T034500Z/phase_d_sparse_downsampling_fix/green/pytest_overlap_suite_green.log for regression coverage.
- Populate docs/summary.md with accepted counts, acceptance rate, MIN_ACCEPTANCE_RATE check, and note CLI unblocker.
- After GREEN, refresh docs/TESTING_GUIDE.md and docs/development/TEST_SUITE_INDEX.md entries per Doc Sync Plan, then append Attempt #85 in docs/fix_plan.md.

Pitfalls To Avoid:
- Keep overlap module pure (no params.cfg touch) per CONFIG-001.
- Ensure greedy selection is deterministic (stable ordering) to prevent flaky manifests.
- Do not weaken MIN_ACCEPTANCE_RATE guard; only bypass via fallback when acceptance rises above threshold.
- Preserve DATA-001 validation by reusing validate_dataset_contract after filtering.
- Record selection strategy in metadata without overwriting existing fields.
- Avoid writing artifacts outside the reserved reports hub.
- Maintain numpy dtype fidelity (float32 diffraction, complex64 guesses).
- Keep scipy dependency usage unchanged; no new external packages.
- Do not downgrade skip instrumentation assertions added in Attempt #84.

If Blocked:
- If greedy fallback still yields <10% acceptance, keep ValueError, capture log + metrics JSON under docs/, and mark Attempt #85 blocked in docs/fix_plan.md with acceptance stats and next-step hypothesis (e.g., regenerate Phase C coordinates).

Findings Applied (Mandatory):
- POLICY-001 — Respect PyTorch dependency expectations (no torch removal; CLI assumptions unchanged).
- CONFIG-001 — Greedy selector must remain side-effect free and avoid params.cfg.
- DATA-001 — Filtered NPZs validated via existing contract to avoid silent dtype/key drift.
- OVERSAMPLING-001 — Preserve neighbor_count metadata so Phase E grouping stays valid.

Pointers:
- docs/fix_plan.md:31
- studies/fly64_dose_overlap/overlap.py:318
- tests/study/test_dose_overlap_overlap.py:246
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T034500Z/phase_d_sparse_downsampling_fix/plan/plan.md:1
- docs/GRIDSIZE_N_GROUPS_GUIDE.md:143

Next Up (optional):
- Execute sparse/train and sparse/test LSQML CLI runs once Phase D emits NPZs.

Doc Sync Plan:
- After tests pass, add new selector entry + command snippet to docs/TESTING_GUIDE.md §Phase D and register it in docs/development/TEST_SUITE_INDEX.md with artifact pointer `plans/.../034500Z/phase_d_sparse_downsampling_fix/green/pytest_sparse_downsamples_green.log`.
- Attach collect-only log (`.../collect/pytest_sparse_downsamples_collect.log`) and confirm selector collects >0 before updating docs.
