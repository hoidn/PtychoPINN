# Phase E5 Path Alignment Summary

**Objective:** Fix Phase D path mismatch in `build_training_jobs` and add `allow_missing_phase_d` parameter to handle missing overlap views gracefully.

**Problem:** Attempt #20 revealed that `build_training_jobs:171` expected Phase D datasets at `dose_1000/dense_train.npz`, but `overlap.py:490,366` actually creates `dose_1000/dense/dense_train.npz` (subdirectories per view).

**Solution:** Updated path construction in `build_training_jobs` to match actual Phase D layout (`dose_{dose}/{view}/{view}_{split}.npz`) and added `allow_missing_phase_d` parameter to gracefully skip overlap jobs when NPZs are missing (expected when Phase D filtering rejects views due to spacing threshold).

---

## Changes Implemented

### 1. Path Alignment (training.py:187-210)

**Before:**
```python
train_data_path=str(phase_d_root / dose_suffix / f"{view}_train.npz"),
test_data_path=str(phase_d_root / dose_suffix / f"{view}_test.npz"),
```

**After:**
```python
train_data_path = phase_d_root / dose_suffix / view / f"{view}_train.npz"
test_data_path = phase_d_root / dose_suffix / view / f"{view}_test.npz"
```

**Rationale:** Match actual Phase D output from `overlap.py:490` (creates `dose/view/` subdirectories) and `overlap.py:366` (writes `view_split.npz` under that subdirectory).

---

### 2. Missing View Handling (training.py:95, 192-203)

**New Parameter:**
```python
def build_training_jobs(
    ...
    allow_missing_phase_d: bool = False,
) -> List[TrainingJob]:
```

**Logic:**
- If `allow_missing_phase_d=False` (default, strict mode): `TrainingJob.__post_init__` validation raises `FileNotFoundError` when NPZ paths don't exist
- If `allow_missing_phase_d=True` (non-strict mode): Log skip message and `continue` to next view; overlap job not added to job list

**CLI Integration (training.py:615):**
```python
all_jobs = build_training_jobs(
    phase_c_root=args.phase_c_root,
    phase_d_root=args.phase_d_root,
    artifact_root=args.artifact_root,
    allow_missing_phase_d=True,  # Non-strict mode for CLI robustness
)
```

**Rationale:** CLI must proceed with baseline training even when Phase D overlap filtering rejects some views (e.g., sparse view with too few positions). Tests retain strict mode (default `False`) to ensure deterministic job matrices.

---

### 3. Test Coverage

#### Updated Tests
- **`mock_phase_d_datasets` fixture (test_dose_overlap_training.py:63-124):** Now creates subdirectory structure matching actual Phase D layout
- **CLI test fixtures (test_dose_overlap_training.py:481,638):** Updated to create `dose/view/view_split.npz` structure instead of flat `dose/view_split.npz`

#### New Test
- **`test_build_training_jobs_skips_missing_view` (test_dose_overlap_training.py:1045-1177):**
  - Validates `allow_missing_phase_d=False` raises `FileNotFoundError` (strict mode)
  - Validates `allow_missing_phase_d=True` logs skip message and returns only baseline + dense jobs (6 jobs instead of 9)
  - Captures log output to verify skip messaging

---

## Test Results

### RED Phase
- `test_build_training_jobs_matrix`: `FileNotFoundError: Training dataset not found: .../phase_d/dose_1000/dense_train.npz`
  - **Expected:** Fixture creates subdirectories but implementation expects flat structure
- `test_build_training_jobs_skips_missing_view`: `TypeError: build_training_jobs() got an unexpected keyword argument 'allow_missing_phase_d'`
  - **Expected:** Parameter not yet added to function signature

**Artifacts:** `red/pytest_build_training_jobs_red.log`

---

### GREEN Phase
- `test_build_training_jobs_matrix`: **PASSED** (1/1 in 4.37s)
  - All 9 jobs enumerated correctly (3 doses × [baseline + dense + sparse])
  - Path validation confirms `dose_X/view/view_split.npz` structure
- `test_build_training_jobs_skips_missing_view`: **PASSED** (1/1 in 4.22s)
  - Strict mode correctly raises `FileNotFoundError` mentioning "sparse"
  - Non-strict mode returns 6 jobs (baseline + dense only)
  - Log output confirms skip message: "Skipping sparse view for dose=1e+03: NPZ files not found"
- `training_cli` selector: **PASSED** (3/3 in 3.79s)
  - `test_training_cli_filters_jobs`: All 9 jobs filtered correctly
  - `test_training_cli_manifest_and_bridging`: Manifest emission validated
  - `test_training_cli_invokes_real_runner`: Real runner integration confirmed

**Artifacts:**
- `green/pytest_build_training_jobs_green.log`
- `green/pytest_training_cli_suite_green_rerun.log`

---

### Collection Proof
- **8 tests collected** (test_dose_overlap_training.py:1-1177)
- All selectors functional:
  - `test_build_training_jobs_matrix`
  - `test_build_training_jobs_skips_missing_view` (NEW)
  - `test_run_training_job_invokes_runner`
  - `test_run_training_job_dry_run`
  - `test_training_cli_filters_jobs`
  - `test_training_cli_manifest_and_bridging`
  - `test_execute_training_job_delegates_to_pytorch_trainer`
  - `test_training_cli_invokes_real_runner`

**Artifacts:** `collect/pytest_collect.log`

---

### Comprehensive Test Suite
- **384 passed** / 17 skipped / 1 known failure (`test_interop_h5_reader`) / 104 warnings
- **No regressions** introduced by path alignment changes
- **Time:** 246.94s (4:06)

**Artifacts:** `green/pytest_full_suite.log`

---

## Findings Applied

- **CONFIG-001:** `build_training_jobs` remains pure (no `params.cfg` mutation); bridge deferred to `run_training_job`
- **DATA-001:** Dataset paths validated for existence; Phase D structure aligns with `overlap.py` output
- **OVERSAMPLING-001:** Gridsize=2 jobs preserve neighbor_count=7 assumption from Phase D filtering
- **POLICY-001:** PyTorch backend wiring unaffected (all tests pass)

---

## Next Actions

1. **Update documentation:**
   - `test_strategy.md`: Mark Phase E5 selector active; add skip-aware test reference
   - `docs/TESTING_GUIDE.md` §2: Add `test_build_training_jobs_skips_missing_view` selector
   - `docs/development/TEST_SUITE_INDEX.md`: Register new test with rationale

2. **CLI real-run execution:**
   - Regenerate Phase C/D fixtures with deterministic knobs (`--deterministic`, `--accelerator cpu`, `--num-workers 0`)
   - Execute `python -m studies.fly64_dose_overlap.training` with baseline dose=1000 view=baseline
   - Archive CLI logs, Lightning outputs, and manifest to `real_run/` subdirectory

3. **Update ledger:**
   - Record Attempt #21 in `docs/fix_plan.md` with artifact hub, metrics summary, and next actions
   - Mark Phase E5 `[P] → [x]` once CLI real-run evidence lands

---

## References

- **Implementation:** `studies/fly64_dose_overlap/training.py:90-216,604-617`
- **Tests:** `tests/study/test_dose_overlap_training.py:63-124,1045-1177,481,638`
- **Spec:** `specs/data_contracts.md:190-260` (DATA-001 NPZ requirements)
- **Phase D Actual Output:** `studies/fly64_dose_overlap/overlap.py:490,366`
- **Prior Attempt:** `docs/fix_plan.md` Attempt #20 (CLI failure on path mismatch + sparse rejection)
