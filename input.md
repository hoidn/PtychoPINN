Summary: Align Phase D overlap metrics with plan requirements and capture CLI evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase D (Group-Level Overlap Views)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_metrics_manifest -vv; pytest tests/study/test_dose_overlap_overlap.py -k spacing_filter -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T041900Z/phase_d_metrics_alignment/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.D:
  - Implement: studies/fly64_dose_overlap/overlap.py::generate_overlap_views and studies/fly64_dose_overlap/overlap.py::main — emit per-view metrics JSON paths (train/test) and surface them in results so tests can assert `metrics/<dose>/<view>.json`, and extend the CLI to accept `--artifact-root` for copying metrics/manifests under the Phase D reports hub.
  - Test: tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_metrics_manifest — start RED by asserting missing metrics files before code change, then drive GREEN after implementation; keep `pytest tests/study/test_dose_overlap_overlap.py -k spacing_filter -vv` as validating selector for regression.
  - Document: Refresh `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/{implementation.md,test_strategy.md}` Phase D sections, update `reports/2025-11-04T034242Z/phase_d_overlap_filtering/summary.md` with metrics/CLI evidence, archive CLI log + copied metrics under the new artifact hub, and append Attempt #8 outcome to `docs/fix_plan.md`.
  - Validating selector: pytest tests/study/test_dose_overlap_overlap.py -k spacing_filter -vv
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T041900Z/phase_d_metrics_alignment/

Priorities & Rationale:
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/plan.md:16 — D2 calls for metrics stored under reports/…/metrics/<dose>/<view>.json; current pipeline writes a single spacing_metrics.json.
  - studies/fly64_dose_overlap/overlap.py:304 — generate_overlap_views currently returns metrics objects but not file artifacts, so CLI consumers cannot trace evidence.
  - specs/data_contracts.md:207 — DATA-001 enforcement must continue post-filtering; updated tests guard regression.
  - docs/TESTING_GUIDE.md:52 — mandates recording new selectors + collect-only evidence once tests land.

How-To Map:
  - export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
  - mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T041900Z/phase_d_metrics_alignment/{red,green,collect,cli,metrics}
  - pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_metrics_manifest -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T041900Z/phase_d_metrics_alignment/red/pytest.log
  - Update overlap.py + test module; rerun `pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_metrics_manifest -vv` teeing to `green/pytest.log`, then run `pytest tests/study/test_dose_overlap_overlap.py -k spacing_filter -vv` teeing to `green/pytest_spacing.log`.
  - pytest tests/study/test_dose_overlap_overlap.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T041900Z/phase_d_metrics_alignment/collect/pytest_collect.log
  - python -m studies.fly64_dose_overlap.overlap --phase-c-root data/studies/fly64_dose_overlap --output-root tmp/phase_d_overlap_views --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T041900Z/phase_d_metrics_alignment 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T041900Z/phase_d_metrics_alignment/cli/dense_sparse_generation.log
  - find plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T041900Z/phase_d_metrics_alignment/metrics -maxdepth 3 -type f -print > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T041900Z/phase_d_metrics_alignment/metrics/metrics_inventory.txt
  - rm -rf tmp/phase_d_overlap_views

Pitfalls To Avoid:
  - Do not mutate `params.cfg`; keep overlap utilities pure (CONFIG-001).
  - Preserve existing return structure; extend with metrics paths without breaking call sites/tests.
  - Keep tests lightweight with synthetic NPZs; no disk writes to shared datasets.
  - Ensure new CLI flag defaults maintain backward compatibility (artifact flag optional).
  - Verify the CLI wrote metrics into the reports hub before deleting tmp outputs; document locations in summary.md.
  - Update docs/TESTING_GUIDE.md and TEST_SUITE_INDEX only after GREEN evidence.
  - Remove temporary tmp/phase_d_overlap_views artifacts after copying metrics.
  - Avoid hardcoding absolute paths; use Path APIs to maintain portability.

If Blocked:
  - Capture failing command output into the artifact hub (e.g., `cli/blocker.log`), summarize the issue/stack trace in summary.md, and log the blocker + return conditions in docs/fix_plan.md.

Findings Applied (Mandatory):
  - CONFIG-001 — overlap pipeline must stay params.cfg-neutral and callable without legacy bridge.
  - DATA-001 — validator enforces canonical keys/dtypes after filtering; tests assert compliance.
  - OVERSAMPLING-001 — neighbor_count (7) ≥ gridsize² (4); verify metrics keep that invariant.

Pointers:
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/plan.md:16
  - studies/fly64_dose_overlap/overlap.py:304
  - studies/fly64_dose_overlap/overlap.py:476
  - tests/study/test_dose_overlap_overlap.py:260
  - specs/data_contracts.md:207

Next Up (optional):
  - Prepare Phase E training run checklist once metrics artifacts verified.

Doc Sync Plan:
  - After GREEN, append the new selector to docs/TESTING_GUIDE.md §Study suite and update docs/development/TEST_SUITE_INDEX.md; archive collect-only output under the artifacts directory before committing.

Mapped Tests Guardrail:
  - Verify `pytest tests/study/test_dose_overlap_overlap.py --collect-only -vv` lists the new metrics manifest test; if it fails to collect, stop, capture the log, and mark the attempt blocked in docs/fix_plan.md.
