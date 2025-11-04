# Phase C Dataset Generation — Summary

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Phase:** C — Dataset Generation (Dose Sweep)
**Date:** 2025-11-04
**Status:** COMPLETE ✅

## Objective
Implement the dataset generation orchestration pipeline that automates:
- Simulation → Canonicalization → Patch generation → Train/test split → Validation
- For each dose level in StudyDesign (1e3, 1e4, 1e5 photons)
- Enforcing DATA-001 contracts and y-axis spatial separation

## Deliverables

### 1. Production Code
- **`studies/fly64_dose_overlap/generation.py`** (249 lines)
  - `build_simulation_plan()`: Constructs dose-specific `TrainingConfig`/`ModelConfig` with n_images derived from base dataset
  - `generate_dataset_for_dose()`: Orchestrates 5-stage pipeline with dependency injection for testability
  - CLI entry point: `python -m studies.fly64_dose_overlap.generation` with argparse support

### 2. Test Coverage
- **`tests/study/test_dose_overlap_generation.py`** (180 lines)
  - `test_build_simulation_plan`: Verifies config construction (dose→nphotons, n_images, seeds, gridsize=1)
  - `test_generate_dataset_pipeline_orchestration[dose]`: Parametrized over 3 doses, confirms all 5 stages invoked with correct args via mocks
  - `test_generate_dataset_config_construction`: Validates TrainingConfig propagation through pipeline

**Test Results:** 5/5 PASSED (RED→GREEN cycle documented)

## Pipeline Architecture

```
generate_dataset_for_dose(dose, base_npz, output_root, design_params)
  ├─ [Stage 1] simulate_and_save(config, base_npz, simulated_raw.npz)
  │    └─ CONFIG-001: update_legacy_dict() handled internally
  ├─ [Stage 2] transpose_rename_convert(simulated_raw.npz, canonical.npz)
  │    └─ DATA-001: Enforce NHW layout, diff3d→diffraction rename
  ├─ [Stage 3] generate_patches(canonical.npz, patched.npz, K=7)
  │    └─ OVERSAMPLING-001: Preserve K=7 for Phase D K-choose-C
  ├─ [Stage 4] split_dataset(patched.npz, output_dir, split_axis='y')
  │    └─ Spatial separation: train/test halves along y-axis
  └─ [Stage 5] validate_dataset_contract(train.npz) & validate(test.npz)
       └─ Phase B validator: keys, dtypes, spacing thresholds, y-axis separation
```

## RED→GREEN Evidence

### RED Phase (Expected Failure)
**Selector:** `pytest tests/study/test_dose_overlap_generation.py::test_generate_dataset_pipeline_orchestration -vv`

**Error:** `TypeError: TrainingConfig.__init__() got an unexpected keyword argument 'gridsize'`

**Root Cause:** Initial implementation incorrectly passed `gridsize` to `TrainingConfig`; it belongs in `ModelConfig`.

**Artifact:** `plans/active/.../phase_c_dataset_generation/red/pytest.log` (3 FAILED)

### GREEN Phase (Fixed)
**Fix Applied:**
- `studies/fly64_dose_overlap/generation.py:80-91` — Create `ModelConfig(gridsize=1, N=patch_size)` first, then pass as `TrainingConfig(model=model_config, ...)`
- Updated test assertions to check `plan.model_config.gridsize` instead of `plan.training_config.gridsize`

**Result:** 5/5 PASSED

**Artifact:** `plans/active/.../phase_c_dataset_generation/green/pytest.log` (5 PASSED)

### Collection Proof
**Selector:** `pytest tests/study/test_dose_overlap_generation.py --collect-only -vv`

**Result:** 5 tests collected (1 config test + 3 parametrized orchestration tests + 1 construction test)

**Artifact:** `plans/active/.../phase_c_dataset_generation/collect/pytest_collect.log`

## Comprehensive Testing Gate

**Command:** `pytest -v tests/`

**Result:** 366 passed, 17 skipped, 1 failed (pre-existing: test_interop_h5_reader)

**Conclusion:** No regressions introduced by Phase C implementation.

**Artifact:** `plans/active/.../phase_c_dataset_generation/full_suite_pytest.log`

## Key Design Decisions

1. **Dependency Injection:** All tool functions (`simulate_fn`, `canonicalize_fn`, etc.) are injectable parameters, enabling lightweight mocked tests without real simulation.

2. **Config Construction:** `build_simulation_plan()` separates concerns:
   - Derives `n_images` from base dataset (not hardcoded)
   - Seeds come from `StudyDesign.rng_seeds['simulation']`
   - Patch size propagates to `ModelConfig.N`
   - Gridsize=1 initially (grouping deferred to Phase D)

3. **Validation in Pipeline:** Both train and test splits are validated on-the-fly, ensuring DATA-001 compliance before downstream phases consume the datasets.

4. **CLI Portability:** Argparse defaults to standard paths but allows override for CI/dev environments.

## Findings Applied

- **CONFIG-001:** Legacy bridge `update_legacy_dict(p.cfg, config)` is handled inside `simulate_and_save()`, so generator code doesn't need to call it directly.
- **DATA-001:** Canonical NHW format enforced by `transpose_rename_convert_tool`; validator checks keys/dtypes/layout.
- **OVERSAMPLING-001:** K=7 neighbor count preserved in patch generation metadata for Phase D overlap filtering (K ≥ C=4 for gs=2).

## Next Actions (Phase D)

- Implement group-level overlap filtering logic using minimum center spacing thresholds (dense: S≈38.4px, sparse: S≈102.4px)
- Generate two dataset views per dose (dense/sparse) by filtering groups post-hoc
- Write Phase D tests with monkeypatched loaders
- Update documentation with overlap filtering workflow diagram

## Artifacts

- **Implementation Code:** `studies/fly64_dose_overlap/generation.py`
- **Tests:** `tests/study/test_dose_overlap_generation.py`
- **RED Log:** `plans/active/.../phase_c_dataset_generation/red/pytest.log`
- **GREEN Log:** `plans/active/.../phase_c_dataset_generation/green/pytest.log`
- **Collection Log:** `plans/active/.../phase_c_dataset_generation/collect/pytest_collect.log`
- **Full Suite Log:** `plans/active/.../phase_c_dataset_generation/full_suite_pytest.log`

## References

- Phase C Plan: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T032018Z/phase_c_dataset_generation/plan.md`
- Study Design: `studies/fly64_dose_overlap/design.py`
- Validator: `studies/fly64_dose_overlap/validation.py`
- Data Contracts: `specs/data_contracts.md:207-217`
- Spacing Guide: `docs/GRIDSIZE_N_GROUPS_GUIDE.md:143`
- Oversampling Guide: `docs/SAMPLING_USER_GUIDE.md:112`
