Summary: Launch Phase E by defining the training job matrix and builder via RED→GREEN TDD so dense/sparse overlap datasets can feed gs1/gs2 runs safely.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase E (Train PtychoPINN)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_build_training_jobs_matrix -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T060200Z/phase_e_training_e1/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E1:
  - Test: tests/study/test_dose_overlap_training.py::test_build_training_jobs_matrix — author the RED case that asserts 9 jobs (3 doses × {dense,sparse,gs1}) with correct dataset/model metadata and log paths derived from StudyDesign; capture failed run to `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T060200Z/phase_e_training_e1/red/pytest_red.log`.
  - Implement: studies/fly64_dose_overlap/training.py::build_training_jobs — introduce `TrainingJob` dataclass and builder that enumerates baseline (Phase C train/test) plus dense/sparse gs2 jobs per dose, validating required files and preparing artifact directories; rerun the selector to go GREEN and log to `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T060200Z/phase_e_training_e1/green/pytest_green.log`.
  - Doc: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md::Phase E — add the Phase E test design (selectors, fixtures, dependency-injection plan) before implementation, then append Attempt #13 to docs/fix_plan.md with artifact links.
  - Validate: pytest tests/study/test_dose_overlap_training.py --collect-only -vv (tee to `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T060200Z/phase_e_training_e1/collect/pytest_collect.log`) once GREEN succeeded; update docs/TESTING_GUIDE.md and docs/development/TEST_SUITE_INDEX.md with the new selector after code passes.

Priorities & Rationale:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T053500Z/phase_e_training_plan/plan.md:8-31 defines tasks E1–E2 requiring the job builder as first implementation deliverable.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:133 marks Phase E as the next active milestone, so we need executable structure before launching full training runs.
- docs/DEVELOPER_GUIDE.md:68-104 reiterates CONFIG-001 ordering; the builder must avoid touching params.cfg until execution helpers run `update_legacy_dict`.
- specs/data_contracts.md:190-260 codifies DATA-001 NPZ expectations, which the test should verify when job builder inspects Phase C/D outputs.
- docs/GRIDSIZE_N_GROUPS_GUIDE.md:154-172 provides dense/sparse spacing context that the job matrix must preserve for gs2 variants.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T060200Z/phase_e_training_e1/{red,green,collect,docs}
- Update test_strategy.md Phase E section before creating tests so TDD guardrail is satisfied.
- For the RED test, use pytest `tmp_path` to fabricate Phase C (`dose_{n}/patched_{split}.npz`) and Phase D (`dose_{n}/{view}_{split}.npz`) trees plus metrics bundle placeholders; assert job metadata (dose, view, gridsize, dataset paths, artifact target).
- After implementing `build_training_jobs`, rerun the targeted selector and capture GREEN log; include dry-run logging via dataclass repr for future CLI use.
- Run collect-only selector and copy log; stage doc updates (test strategy, TESTING_GUIDE, TEST_SUITE_INDEX) under `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T060200Z/phase_e_training_e1/docs/`.
- Append Attempt #13 summary to docs/fix_plan.md referencing RED/GREEN logs and doc updates.

Pitfalls To Avoid:
- Do not call `update_legacy_dict` inside `build_training_jobs`; reserve CONFIG-001 bridge for execution helper.
- Avoid hard-coding filesystem paths—parameterize Phase C/D roots via function arguments for testability.
- Keep tests lightweight; use dependency injection and synthetic files rather than invoking `ptycho_train`.
- Ensure counts stay deterministic (3 doses × 3 variants) even if datasets missing—prefer raising descriptive errors.
- Maintain ASCII formatting in docs and tests; no smart quotes or tabs.
- Store all logs under the new artifact hub; do not leave files in repo root or tmp/.
- Reference metrics bundle paths but do not load large JSON payloads in tests.
- Update doc registries only after tests pass to avoid documenting selectors that fail.
- Preserve CONFIG-001/DATA-001/OVERSAMPLING-001 references in new docs/tests.

If Blocked:
- Record the failure in plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T060200Z/phase_e_training_e1/docs/summary.md, note missing dependency (e.g., Phase D outputs absent), stash RED log, and update docs/fix_plan.md Attempt #13 with unblock criteria (e.g., regenerate Phase D assets).

Findings Applied (Mandatory):
- CONFIG-001 — Ensure legacy bridge remains outside builder logic; document plan to call `update_legacy_dict` in runner.
- DATA-001 — Validate that job builder references canonical NHW NPZs; tests must assert dataset paths align with spec.
- OVERSAMPLING-001 — Preserve neighbor_count = 7 for gs2 jobs and reflect spacing constraints in test assertions.
- POLICY-001 — Training plan must keep PyTorch backend optional (default TensorFlow) while acknowledging torch dependency per project policy.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T053500Z/phase_e_training_plan/plan.md:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:133
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:64
- docs/DEVELOPER_GUIDE.md:68
- specs/data_contracts.md:190

Next Up (optional):
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E2 — Implement `run_training_job` helper + CLI dry-run once the builder and tests are green.

Doc Sync Plan:
- After GREEN, update docs/TESTING_GUIDE.md §2 and docs/development/TEST_SUITE_INDEX.md with the new selector; archive diffs under `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T060200Z/phase_e_training_e1/docs/`.
