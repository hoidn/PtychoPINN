Summary: Build the Phase B dataset validation harness for the synthetic fly64 study and certify it with red/green pytest evidence plus doc updates.

Mode: TDD

Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase B (Test Infrastructure Design)

Branch: feature/torchapi-newprompt

Mapped tests: pytest tests/study/test_dose_overlap_dataset_contract.py::test_validate_dataset_contract -vv

Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T025541Z/phase_b_test_infra/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001:
  - Implement: studies/fly64_dose_overlap/validation.py::validate_dataset_contract — add reusable checks for DATA-001 keys/dtypes, amplitude requirement, spacing thresholds from StudyDesign, y-axis split validation, and oversampling preconditions with descriptive ValueErrors.
  - Test: tests/study/test_dose_overlap_dataset_contract.py::test_validate_dataset_contract — create pass/fail fixtures for the validator, run the selector once pre-implementation (expect FAIL) and once post-implementation (PASS), tee logs to {red,green}/pytest.log, and run --collect-only to collect/pytest_collect.log.
  - Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/{implementation.md,test_strategy.md} Phase B sections plus reports/2025-11-04T025541Z/phase_b_test_infra/summary.md with validator scope, findings references, and execution proof.
  - Validating selector: pytest tests/study/test_dose_overlap_dataset_contract.py::test_validate_dataset_contract -vv
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T025541Z/phase_b_test_infra/

Priorities & Rationale:
  - specs/data_contracts.md:1-152 — normative NPZ schema the validator must enforce (keys, dtypes, amplitude requirement).
  - docs/GRIDSIZE_N_GROUPS_GUIDE.md:143-151 — spacing heuristic S ≈ (1 − f_group) × N drives dense/sparse overlap checks.
  - docs/SAMPLING_USER_GUIDE.md:112-123 — oversampling preconditions (neighbor_count ≥ gridsize², n_groups > n_subsample) inform validator guardrails.
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:48-60 — Phase B responsibilities mandate validator + pytest coverage.
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:63-69 — working plan deliverables and artifact hub for Phase B.

How-To Map:
  - export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
  - mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T025541Z/phase_b_test_infra/{red,green,collect}
  - After authoring the new test but before implementing the validator, run `pytest tests/study/test_dose_overlap_dataset_contract.py::test_validate_dataset_contract -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T025541Z/phase_b_test_infra/red/pytest.log`
  - Implement `validate_dataset_contract`, then rerun the selector with the same tee path for GREEN evidence.
  - Capture collection proof: `pytest tests/study/test_dose_overlap_dataset_contract.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T025541Z/phase_b_test_infra/collect/pytest_collect.log`
  - Update docs (`implementation.md`, `test_strategy.md`, summary.md) to align with validator checks and reference findings CONFIG-001/DATA-001/OVERSAMPLING-001; keep artifact paths consistent.
  - Run `pytest tests/study/test_dose_overlap_dataset_contract.py -vv` once after docs to ensure selector still green; summarize pass/fail counts in summary.md.

Pitfalls To Avoid:
  - Do not skip the RED log—capture the failing selector before implementing the validator.
  - Keep validator pure (no filesystem or params.cfg access); rely only on provided arrays/dicts.
  - Avoid broad pytest runs; stay on the targeted selector to reduce noise.
  - Do not reuse production datasets; craft small in-memory fixtures for tests.
  - Make ValueError messages actionable (field, expected vs actual) to aid future debugging.
  - Keep artifacts inside the designated timestamped directory; nothing at repo root or tmp leftovers.
  - Do not change runtime scripts for dataset generation yet—Focus is validation harness only.
  - Ensure new test module follows pytest style (no unittest mixins) to honor project guidance.
  - Avoid introducing non-ASCII characters in new files; stick to ASCII per repo policy.

If Blocked:
  - If numpy fixtures require additional helpers, log the gap in summary.md and mark Attempt INCOMPLETE in docs/fix_plan.md before exiting.
  - If tests fail due to unexpected dependency/import error, capture the traceback in the red log and annotate docs/fix_plan.md with the minimal error signature; halt further implementation.
  - Should validator design conflict with spec interpretation, record the question in summary.md and flag the task as blocked in docs/fix_plan.md Attempts History.

Findings Applied (Mandatory):
  - CONFIG-001 — Keep validator independent of params.cfg mutations so CONFIG-001 bridge can occur before legacy loaders execute.
  - DATA-001 — Enforce canonical dataset keys/dtypes and amplitude requirement described by the data contract.
  - OVERSAMPLING-001 — Verify neighbor_count ≥ gridsize² so oversampling paths remain feasible.

Pointers:
  - specs/data_contracts.md:1
  - docs/GRIDSIZE_N_GROUPS_GUIDE.md:143
  - docs/SAMPLING_USER_GUIDE.md:112
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:48
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:63
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T025541Z/phase_b_test_infra/plan.md:1

Next Up (optional):
  - Phase B: integrate the validator into dataset generation CLI hooks ahead of Phase C runs.
  - Phase C kickoff: generate single-dose dataset and exercise the new validator against actual NPZ output.

Doc Sync Plan:
  - After GREEN run, update docs/TESTING_GUIDE.md (section referencing study tests) and docs/development/TEST_SUITE_INDEX.md to list `tests/study/test_dose_overlap_dataset_contract.py::test_validate_dataset_contract`; note artifact path in summary.md.
  - Commit summary of new selector coverage in summary.md and ensure collect-only log is stored under phase_b_test_infra/collect/pytest_collect.log before wrapping.

Mapped Tests Guardrail:
  - Confirm `pytest tests/study/test_dose_overlap_dataset_contract.py --collect-only -vv` reports ≥1 test; if collection breaks, revert the selector addition or mark as Planned with rationale in summary.md before closing the loop.
