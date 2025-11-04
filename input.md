Summary: Encode the Phase A design constants for the synthetic fly64 study in code and lock a TDD harness for the spacing heuristic before updating the design docs.

Mode: TDD

Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase A (Design & Constraints)

Branch: feature/torchapi-newprompt

Mapped tests: pytest tests/study/test_dose_overlap_design.py::test_study_design_constants -vv

Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T021500Z/

Do Now:
  - Implement: studies/fly64_dose_overlap/design.py::get_study_design — create a canonical data structure exporting the Phase A constants (dose list [1e3, 1e4, 1e5], gridsizes {1, 2}, neighbor_count=7 for gs2, spacing heuristic S ≈ (1 − f_group) × N with explicit dense/sparse thresholds, and fixed RNG seeds for simulation/grouping).
  - Test: tests/study/test_dose_overlap_design.py::test_study_design_constants — author RED test asserting the exported design includes the required keys/values and spacing thresholds; run it (expect fail), then make it pass after implementing the design module; capture red/green logs under the artifacts path.
  - Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/{implementation.md,test_strategy.md} Phase A sections with the concrete numbers and spacing rule derivation, plus summary.md capturing rationale.
  - Validating selector: pytest tests/study/test_dose_overlap_design.py::test_study_design_constants -vv
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T021500Z/

Priorities & Rationale:
  - specs/data_contracts.md §2 — dataset amplitude/complex contract must be satisfied by any generated synthetic set; design constants must not violate the contract.
  - docs/GRIDSIZE_N_GROUPS_GUIDE.md — authoritative overlap/neighbor heuristics for gridsize/K spacing.
  - docs/SAMPLING_USER_GUIDE.md §3 — oversampling rules (K ≥ C, n_groups vs n_subsample) inform the spacing heuristic and dense/sparse splits.
  - docs/findings.md (CONFIG-001, OVERSAMPLING-001) — guardrails for config initialization and oversampling constraints we must respect when codifying the study design.

How-To Map:
  - export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
  - mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T021500Z/{red,green,collect} and log pytest outputs accordingly.
  - Run `pytest tests/study/test_dose_overlap_design.py::test_study_design_constants -vv` once pre-implementation (expect FAIL) and once post-implementation (PASS), then `pytest tests/study/test_dose_overlap_design.py --collect-only -vv` for guardrail evidence.
  - Update the Phase A narrative in implementation.md and PAS criteria in test_strategy.md to match the code constants; record decisions in summary.md under the same artifact hub.
  - Keep seeds documented in both code and docs; align RNG handling with existing simulations (see scripts/simulation/simulate_and_save.py usage).

Pitfalls To Avoid:
  - Do not introduce datasets or heavy runs this loop—only constants/tests.
  - Avoid duplicate sources of truth; the new design module becomes canonical.
  - Keep spacing thresholds unit-consistent (pixels) and document derivation.
  - Do not hardcode torch/tensorflow imports inside the design module; it should be pure data.
  - Prevent accidental commits of generated data; only metadata/docs go in git.
  - Don’t skip red/green logging; capture pytest output for both runs.

If Blocked:
  - If spacing heuristics require data that is not yet available, log the gap in docs/fix_plan.md and adjust summary.md with the open question; leave the test marked xfail with explanation only if necessary and recorded.

Findings Applied (Mandatory):
  - CONFIG-001 — mirrors future requirement to bridge params.cfg before using legacy code; document in summary to avoid omissions when datasets are generated.
  - DATA-001 — ensures the design constants anticipate amplitude/complex dtype requirements.
  - OVERSAMPLING-001 — spacing heuristic must respect K ≥ C to avoid impossible grouping.

Pointers:
  - specs/data_contracts.md:1
  - docs/GRIDSIZE_N_GROUPS_GUIDE.md:1
  - docs/SAMPLING_USER_GUIDE.md:1
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:20
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:24

Next Up (optional):
  - Implement Phase C single-dose dataset generation once constants and tests are approved.

Doc Sync Plan:
  - After the new test lands, run `pytest tests/study/test_dose_overlap_design.py --collect-only -vv` (log to collect/pytest_collect.log) and update docs/TESTING_GUIDE.md §PyTorch Backend Tests + docs/development/TEST_SUITE_INDEX.md Torch table with the new selector once GREEN.

Mapped Tests Guardrail:
  - Ensure the new selector collects >0 tests during `--collect-only`; if not, keep the test as xfail with rationale and log in summary before closing the loop.
