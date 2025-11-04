**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001

## 1. Framework & Style
- Use pytest-style tests and lightweight validation scripts (no unittest mix-in) to keep parity with project guidance.
- Tests focus on data contracts, selection invariants, and execution proofs; heavy training is validated via logs/CSVs captured as artifacts.

## 2. CI/CD & Optional Dependencies
- Pty-chi is optional and may be unavailable in CI; mark pty-chi validation as “evidence via artifact” (manual run) with justification in summary.md.
- Torch is required in project policy but not exercised directly in this TensorFlow-based study; ensure tests skip gracefully if torch-related paths are unneeded.

## 3. Test Tiers
- Unit: dataset contract checks (keys/dtypes), dose normalization ranges, spacing calculations.
- Integration: end-to-end dataset creation + split + grouping + filtering (no training), ensuring artifacts produced.
- Smoke: minimal training invocation (short epochs) to verify plumbing, not performance.

## 4. Execution Proof
- For any test-related task, collect:
  - pytest log at `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/<timestamp>/pytest.log`
  - results summary at `.../summary.md` with pass/fail counts and any skips justified
  - evidence CSVs from compare_models (MS-SSIM etc.)
- Acceptable SKIP reasons: pty-chi unavailable; GPU-specific long tests; long benchmarks.

## 5. Mock/Stub Strategy
- For pty-chi steps in CI, rely on recorded outputs (aligned NPZs) when available; otherwise skip with justification.
- For simulator inputs, use small synthetic objects/probes when sanity checking structure.

## 6. Specific Checks

### Phase A — Design Constants (COMPLETE)
**Test module:** `tests/study/test_dose_overlap_design.py`
**Status:** 3/3 PASSED

Validates:
- Dose list [1e3, 1e4, 1e5]
- Gridsizes {1, 2} with neighbor_count=7
- K ≥ C constraint (K=7 ≥ C=4 for gridsize=2)
- Overlap views: dense=0.7, sparse=0.2
- Derived spacing thresholds: dense ≈ 38.4px, sparse ≈ 102.4px
- Spacing formula S = (1 − f_group) × N with N=128
- Train/test split axis='y'
- RNG seeds: simulation=42, grouping=123, subsampling=456
- MS-SSIM config: sigma=1.0, emphasize_phase=True

**Execution proof:**
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T021500Z/green/pytest_green.log`
- All tests PASSED; no SKIPs

### Phase B — Dataset Contract Validation (COMPLETE)
**Status:** 11/11 tests PASSED — RED/GREEN/collect evidence captured
- Working Plan: `reports/2025-11-04T025541Z/phase_b_test_infra/plan.md`
- Validator: `studies/fly64_dose_overlap/validation.py::validate_dataset_contract` checks ✅
  - DATA-001 NPZ keys/dtypes and amplitude requirement (`diffraction` as amplitude float32, complex64 fields for object/probe).
  - Train/test split axis `'y'` using StudyDesign constants.
  - Spacing thresholds `S ≈ (1 - f_group) × N` derived from `design.get_study_design()`; enforce dense/sparse minima (GRIDSIZE_N_GROUPS_GUIDE.md:143-151).
  - Oversampling preconditions (neighbor_count ≥ gridsize²) aligned with OVERSAMPLING-001.
- Tests: `tests/study/test_dose_overlap_dataset_contract.py` ✅
  - 11 tests covering happy path, missing keys, dtype violations, shape mismatches, spacing thresholds, oversampling constraints
  - All tests PASSED in GREEN phase (0.96s runtime)
  - RED phase: stub implementation → 1 FAILED (NotImplementedError)
  - GREEN phase: full implementation → 11 PASSED
  - Logs: RED and GREEN runs stored under `reports/2025-11-04T025541Z/phase_b_test_infra/{red,green}/pytest.log`; collect-only log under `collect/pytest_collect.log`.
- Documentation: Updated `summary.md`, `implementation.md`, and this strategy with validator coverage; recorded findings adherence (CONFIG-001, DATA-001, OVERSAMPLING-001). ✅

### Phase D — Group-Level Overlap Views (COMPLETE)
**Test module:** `tests/study/test_dose_overlap_overlap.py` ✅
**Selectors (Active):**
- `pytest tests/study/test_dose_overlap_overlap.py -k spacing_filter -vv` (RED→GREEN: 2 tests, spacing math validation)
- `pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_metrics_manifest -vv` (metrics bundle guard)
- `pytest tests/study/test_dose_overlap_overlap.py --collect-only -vv` (10 tests total)
**Coverage Delivered:**
- Spacing utilities (`compute_spacing_matrix`, `build_acceptance_mask`) validated against dense (0.7 → S≈38.4 px) and sparse (0.2 → S≈102.4 px) thresholds using synthetic coordinates (`test_compute_spacing_matrix_dense`, `test_compute_spacing_matrix_sparse`).
- `generate_overlap_views` orchestration verified via monkeypatched loaders ensuring correct NPZ outputs, validator invocation with `view` parameter, and spacing metrics JSON emission (`test_generate_overlap_views_paths`, `test_generate_overlap_views_calls_validator_with_view`).
- Metrics bundle structure validated: `test_generate_overlap_views_metrics_manifest` asserts bundle contains `train`/`test` keys with `output_path` and `spacing_stats_path` entries; paths exist on disk.
- Failure path exercised when Phase C data missing (`test_generate_overlap_views_missing_phase_c_data` expects FileNotFoundError).
- RED→GREEN evidence: Attempt #7 spacing filter tests (2 PASSED), Attempt #10 metrics manifest test (1 PASSED RED phase, 1 PASSED GREEN phase after bundle implementation).
**Execution Proof:**
- Spacing filter RED/GREEN: `reports/2025-11-04T034242Z/phase_d_overlap_filtering/{red,green}/pytest.log` (2 tests)
- Metrics bundle RED/GREEN: `reports/2025-11-04T045500Z/phase_d_cli_validation/{red,green}/pytest_metrics_bundle.log` (1 test)
- Regression suite: `reports/2025-11-04T045500Z/phase_d_cli_validation/green/pytest_spacing.log` (2 tests)
- Collection proof: `reports/2025-11-04T045500Z/phase_d_cli_validation/collect/pytest_collect.log` (10 collected)
- CLI execution: `reports/2025-11-04T045500Z/phase_d_cli_validation/cli/phase_d_overlap.log` (metrics bundle copied to artifact hub)
**Findings Alignment:** CONFIG-001 (pure functions; no params.cfg mutation in overlap.py), DATA-001 (validator ensures NPZ contract per specs/data_contracts.md), OVERSAMPLING-001 (neighbor_count=7 ≥ C=4 for gridsize=2 enforced in group construction and validated in tests).

### Phase E — Train PtychoPINN (In Progress)
**Test module:** `tests/study/test_dose_overlap_training.py`
**Selectors (Planned):**
- `pytest tests/study/test_dose_overlap_training.py::test_build_training_jobs_matrix -vv` (RED→GREEN: job enumeration with correct counts and metadata)
- `pytest tests/study/test_dose_overlap_training.py --collect-only -vv` (collection proof)

**Coverage Plan:**
- Job matrix enumeration: Validate that `build_training_jobs()` produces exactly 9 jobs per dose (3 doses × 3 variants):
  - Baseline: gs1 using Phase C patched_{train,test}.npz
  - Dense overlap: gs2 using Phase D dense_{train,test}.npz
  - Sparse overlap: gs2 using Phase D sparse_{train,test}.npz
- Job metadata validation: Each `TrainingJob` dataclass must contain:
  - dose (float, from StudyDesign.dose_list)
  - view (str, one of {"baseline", "dense", "sparse"})
  - gridsize (int, 1 or 2)
  - train_data_path (str, validated existence)
  - test_data_path (str, validated existence)
  - artifact_dir (Path, deterministic from dose/view/gridsize)
  - log_path (Path, derived from artifact_dir)
- Dependency injection: Tests use `tmp_path` fixture to fabricate Phase C and Phase D NPZ trees with minimal valid datasets (DATA-001 keys only, small arrays).
- Failure handling: Test behavior when Phase C or Phase D artifacts are missing (expect FileNotFoundError with descriptive message).
- CONFIG-001 guard: Ensure `build_training_jobs()` does NOT call `update_legacy_dict`; that bridge remains in execution helper for E3.

**Execution Proof Requirements:**
- RED log: Capture initial failure (NotImplementedError or ImportError) in `reports/2025-11-04T060200Z/phase_e_training_e1/red/pytest_red.log`
- GREEN log: Capture passing test after implementation in `reports/2025-11-04T060200Z/phase_e_training_e1/green/pytest_green.log`
- Collection log: Verify test collection in `reports/2025-11-04T060200Z/phase_e_training_e1/collect/pytest_collect.log`

**Findings Alignment:**
- CONFIG-001: Job builder remains pure (no params.cfg mutation); legacy bridge deferred to `run_training_job()` in E3.
- DATA-001: Test fixtures must create NPZs with canonical keys (diffraction, objectGuess, probeGuess, Y, coords, filenames) per specs/data_contracts.md:190-260.
- OVERSAMPLING-001: Grouped jobs (gs2) assume neighbor_count=7 from Phase D outputs; tests validate gridsize field matches view expectation.

### Future Phases (Pending)
1) Dose sanity per dataset
   - Confirm expected scaling behavior across doses (qualitative/log statistics)
2) Train/test separation
   - Validate y-axis split with non-overlapping regions
3) Comparison outputs
   - Confirm CSV present with MS-SSIM (phase and amplitude), plots saved, aligned NPZs exist

## 7. PASS Criteria
- All contract checks pass for generated datasets
- Filtering invariants satisfied; logs record spacing stats
- Comparison CSVs exist with non-empty records for each condition
- No unexpected SKIPs, and any expected SKIPs are justified in summary.md
