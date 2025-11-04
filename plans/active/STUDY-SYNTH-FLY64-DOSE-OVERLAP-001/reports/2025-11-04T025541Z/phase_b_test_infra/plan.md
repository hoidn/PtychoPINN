# Phase B Plan — Test Infrastructure Design

## Context
- Initiative: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Phase Goal: Establish automated validation for synthetic fly64 datasets before generation work begins.
- Dependencies: Phase A design constants (`studies/fly64_dose_overlap/design.py`), DATA-001 contract (`specs/data_contracts.md`), overlap heuristics (`docs/GRIDSIZE_N_GROUPS_GUIDE.md:141-151`), oversampling preconditions (`docs/SAMPLING_USER_GUIDE.md:112-123`).

### Phase B — Dataset Contract Validation Harness
Goal: Provide reusable validation logic and pytest coverage that enforce dataset structure, dtype, and overlap invariants for the study.
Prereqs: Phase A design module complete; test strategy Section 6 enumerates Phase B checks; no datasets generated yet.
Exit Criteria: Validator accepts well-formed mock dataset derived from design constants, rejects malformed variants, pytest selector passes (GREEN) with logs captured, documentation (implementation.md, test_strategy.md) updated with validation coverage, summary.md records decision points.

| ID | Task Description | State | How/Why & Guidance (API/doc/artifact/source refs) |
| --- | --- | --- | --- |
| B1 | Implement `studies/fly64_dose_overlap/validation.py::validate_dataset_contract` to check DATA-001 keys/dtypes, amplitude vs intensity, train/test split axis, and spacing thresholds against design constants. | [ ] | Use StudyDesign from `design.py` plus `numpy`. Reference `specs/data_contracts.md:1-152` for normative fields and `docs/GRIDSIZE_N_GROUPS_GUIDE.md:141-151` for spacing heuristics. Raise descriptive `ValueError` with offending field. |
| B2 | Author pytest module `tests/study/test_dose_overlap_dataset_contract.py` with parametrized tests covering pass/fail cases (missing key, wrong dtype, overlap threshold breach). Capture RED log before implementation and GREEN log after. | [ ] | RED: `pytest tests/study/test_dose_overlap_dataset_contract.py::test_validate_dataset_contract -vv` (expect failure before validator). GREEN: same selector after implementation. Store logs under `reports/2025-11-04T025541Z/phase_b_test_infra/{red,green}/pytest.log`; collect-only run to `collect/pytest_collect.log`. |
| B3 | Update documentation: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/{implementation.md,test_strategy.md}` Phase B sections + new `summary.md` capturing validator scope, edge cases, and findings ledger references (CONFIG-001, DATA-001, OVERSAMPLING-001). | [ ] | Align doc narratives with implemented checks; mention validator usage in future phases. Record findings adherence and decisions in `summary.md`. |
| B4 | Archive artifacts and sync ledger: add Attempt entry to `docs/fix_plan.md`, ensure `input.md` references artifact hub `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T025541Z/phase_b_test_infra/`, and upload pytest summary counts. | [ ] | Capture summary under `reports/2025-11-04T025541Z/phase_b_test_infra/summary.md`; update `docs/findings.md` only if new durable lessons emerge. |

