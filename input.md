Summary: Kick off Phase C by wiring the fly64 dose sweep dataset generator (code + tests) and capture staged artifacts for all doses.

Mode: TDD

Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase C (Dataset Generation)

Branch: feature/torchapi-newprompt

Mapped tests: pytest tests/study/test_dose_overlap_generation.py -k generation_pipeline -vv; pytest tests/study/test_dose_overlap_generation.py::test_build_simulation_plan -vv

Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T032018Z/phase_c_dataset_generation/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.C:
  - Implement: studies/fly64_dose_overlap/generation.py::generate_dataset_for_dose — create the orchestration entrypoint (with helper `build_simulation_plan`) that consumes StudyDesign, invokes simulate→canonicalize→patch→split→validate, and returns paths for downstream phases.
  - Test: tests/study/test_dose_overlap_generation.py::test_generate_dataset_pipeline — start red with monkeypatched stubs (expect NotImplementedError), then make it green ensuring each stage is invoked, dose→nphotons mapping holds, and validator runs on final train/test paths; log red/green/collect evidence under the Phase C artifact hub.
  - Document: Update plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/{implementation.md,test_strategy.md} Phase C sections plus reports/2025-11-04T032018Z/phase_c_dataset_generation/summary.md with pipeline steps, artifact locations, and findings references; record dataset_manifest/run logs per dose.
  - Validating selector: pytest tests/study/test_dose_overlap_generation.py -k generation_pipeline -vv
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T032018Z/phase_c_dataset_generation/

Priorities & Rationale:
  - specs/data_contracts.md:207 — DATA-001 mandates canonical NHW diffraction arrays that our pipeline must produce before validation.
  - docs/GRIDSIZE_N_GROUPS_GUIDE.md:143 — spacing heuristic informs how generated coordinates must remain compatible with later overlap filtering.
  - docs/SAMPLING_USER_GUIDE.md:112 — oversampling preconditions (K ≥ C) must be preserved in generated metadata for Phase D.
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:74 — Phase C deliverable requires simulate→split workflow per dose.
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T032018Z/phase_c_dataset_generation/plan.md:15 — Working plan specifies functions, artifact layout, and log capture expectations for C1–C4.

How-To Map:
  - export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
  - mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T032018Z/phase_c_dataset_generation/{red,green,collect}
  - Create the red log: `pytest tests/study/test_dose_overlap_generation.py -k generation_pipeline -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T032018Z/phase_c_dataset_generation/red/pytest.log`
  - Implement generation.py helpers + pipeline, then rerun the selector for GREEN evidence with the same tee path but `.../green/pytest.log`.
  - Collect proof: `pytest tests/study/test_dose_overlap_generation.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T032018Z/phase_c_dataset_generation/collect/pytest_collect.log`
  - After tests pass, run the CLI entry (once) to generate all doses, capturing output: `python -m studies.fly64_dose_overlap.generation --base-npz datasets/fly64/fly64_shuffled.npz --output-root data/studies/fly64_dose_overlap 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T032018Z/phase_c_dataset_generation/dataset_generation.log`
  - Move per-dose run_manifest.json files (or a consolidated manifest) into the artifact directory and summarize validator results in summary.md.

Pitfalls To Avoid:
  - Do not commit generated NPZs; leave them under data/ and reference paths in artifacts instead.
  - Keep simulation seeds deterministic (design.rng_seeds['simulation']) so downstream comparisons stay reproducible.
  - Ensure validator executes on both train and test splits—skipping validation violates DATA-001 guardrails.
  - Avoid direct np.savez writes that bypass transpose_rename_convert; rely on canonical tool to enforce NHW layout.
  - Do not hardcode absolute paths; use pathlib Paths relative to repo root for portability.
  - Do not call simulate_and_save before update_legacy_dict via TrainingConfig (CONFIG-001) in helper logic.
  - Keep monkeypatched tests lightweight; never invoke the real simulation in pytest.
  - Capture CLI output with tee; missing logs will block summary/ledger updates.

If Blocked:
  - If simulate_and_save import errors occur, log the traceback in dataset_generation.log and mark Attempt INCOMPLETE in docs/fix_plan.md with the minimal error signature.
  - If validator fails on generated data, keep failing dataset under data/ (do not delete), summarize the error in summary.md, and flag the attempt as blocked pending investigation.
  - If pytest selector refuses to collect (0 tests), record the failure in the red log, update summary.md, and pause implementation for supervisor follow-up.

Findings Applied (Mandatory):
  - CONFIG-001 — Maintain bridge boundaries: configs update params.cfg before legacy modules, keeping the generator safe to run pre-initialization.
  - DATA-001 — Enforce canonical keys/dtypes so downstream loaders honor the contract without manual fixes.
  - OVERSAMPLING-001 — Preserve neighbor_count ≥ gridsize² metadata to ensure later K-choose-C operations remain valid.

Pointers:
  - specs/data_contracts.md:207
  - docs/GRIDSIZE_N_GROUPS_GUIDE.md:143
  - docs/SAMPLING_USER_GUIDE.md:112
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:74
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T032018Z/phase_c_dataset_generation/plan.md:15

Next Up (optional):
  - Once datasets exist, prepare Phase D plan for overlap filtering (dense vs sparse acceptance metrics).

Doc Sync Plan:
  - After GREEN, append the new selector to docs/TESTING_GUIDE.md (study section) and docs/development/TEST_SUITE_INDEX.md with artifact path references; store collect-only output under phase_c_dataset_generation/collect/pytest_collect.log.

Mapped Tests Guardrail:
  - Confirm `pytest tests/study/test_dose_overlap_generation.py --collect-only -vv` reports ≥1 test; if not, downgrade selector status to Planned in summary.md and halt implementation until collection succeeds.
