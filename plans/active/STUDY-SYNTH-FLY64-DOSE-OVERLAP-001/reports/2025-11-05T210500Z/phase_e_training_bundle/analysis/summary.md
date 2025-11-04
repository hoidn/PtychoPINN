# Phase E Training Bundle Persistence - Summary

**Loop ID:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.G2 (Phase E5)
**Date:** 2025-11-05
**Timestamp:** 2025-11-05T210500Z
**Mode:** TDD (RED → GREEN)
**Focus:** Bundle persistence in execute_training_job
**Branch:** feature/torchapi-newprompt

## Objective

Implement bundle persistence in `execute_training_job` to emit spec-compliant `wts.h5.zip` archives after successful training, unblocking Phase G comparison runs that require real PINN/baseline model bundles.

## Implementation Summary

### Changes Made

1. **Import Addition** (`studies/fly64_dose_overlap/training.py:30`)
   - Added `from ptycho_torch.model_manager import save_torch_bundle`

2. **Bundle Persistence Logic** (`studies/fly64_dose_overlap/training.py:484-516`)
   - After successful training, check if `training_results['models']` exists
   - Call `save_torch_bundle` with dual-model dict (autoencoder + diffraction_to_obj)
   - Create bundle at `{output_dir}/wts.h5.zip` per Ptychodus naming convention
   - Populate `result['bundle_path']` for manifest emission
   - Handle persistence failures gracefully (log warning, continue with bundle_path=None)

3. **Result Dict Update** (`studies/fly64_dose_overlap/training.py:523`)
   - Added `'bundle_path': bundle_path` to result dict

4. **Test Addition** (`tests/study/test_dose_overlap_training.py:987-1152`)
   - Added `test_execute_training_job_persists_bundle` to validate:
     - `save_torch_bundle` called with correct models_dict
     - `result['bundle_path']` populated after success
     - Bundle archive exists on disk
     - Manifest metadata includes bundle_path

### SPEC Alignment

**Quoted SPEC Lines Implemented:**
> "Checkpoint persistence MUST produce `wts.h5.zip` archives compatible with the TensorFlow persistence contract (§4.6), containing both Lightning `.ckpt` state and bundled hyperparameters for state-free reload."
> — specs/ptychodus_api_spec.md:239

**ADR References:**
- specs/ptychodus_api_spec.md §4.6: Model persistence requirements (wts.h5.zip contract)
- docs/architecture.md: Persistence layer integration
- docs/pytorch_runtime_checklist.md: Bundle persistence best practices

## Test Results

### RED Phase (Expected Failure)
**Command:** `pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv`
**Result:** ❌ FAILED
**Error:** `AttributeError: <module 'studies.fly64_dose_overlap.training'> has no attribute 'save_torch_bundle'`
**Artifact:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T210500Z/phase_e_training_bundle/red/pytest_execute_training_job_bundle_red.log`

### GREEN Phase (Implementation Success)
**Command:** `pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv`
**Result:** ✅ PASSED (1 passed in 3.76s)
**Artifact:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T210500Z/phase_e_training_bundle/green/pytest_execute_training_job_bundle_green.log`

### Regression Suite
**Command:** `pytest tests/study/test_dose_overlap_training.py -k training_cli -vv`
**Result:** ✅ PASSED (3 passed, 6 deselected in 3.64s)
**Tests:**
- `test_training_cli_filters_jobs`
- `test_training_cli_manifest_and_bridging`
- `test_training_cli_invokes_real_runner`
**Artifact:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T210500Z/phase_e_training_bundle/green/pytest_training_cli_suite_green.log`

### Collection Validation
**Command:** `pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv`
**Result:** ✅ 3/9 tests collected (6 deselected)
**Artifact:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T210500Z/phase_e_training_bundle/collect/pytest_training_cli_collect.log`

### Full Test Suite
**Command:** `pytest -v tests/`
**Result:** ✅ 395 passed, 17 skipped, 1 failed (pre-existing) in 249.08s (0:04:09)
**Failed Test:** `tests/io/test_ptychodus_interop_h5.py::test_interop_h5_reader`
**Failure Cause:** Pre-existing `ModuleNotFoundError: No module named 'ptychodus'` (unrelated to bundle persistence changes)
**Artifact:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T210500Z/phase_e_training_bundle/green/pytest_full_suite.log`

## Acceptance Criteria Met

✅ **AT-49 (Bundle Persistence):** `execute_training_job` emits `wts.h5.zip` bundles after successful training
✅ **AT-50 (Manifest Fields):** Result dict includes `bundle_path` for downstream consumers
✅ **AT-51 (Spec Compliance):** Bundles follow §4.6 dual-model structure (autoencoder + diffraction_to_obj)
✅ **AT-52 (Graceful Degradation):** Persistence failures logged but don't crash training
✅ **AT-53 (Test Coverage):** TDD test validates bundle creation and manifest population

## Outstanding Gaps

1. **Real Training Bundles Missing:** Phase E executor has not been invoked with real datasets yet. The CLI command in `input.md:13` requires Phase C/D datasets to be present at `tmp/phase_c_f2_cli` and `tmp/phase_d_f2_cli`.

2. **Sparse View Bundles:** Only dense training tested so far. Sparse view training (with acceptance metadata) needs separate execution once datasets exist.

3. **Intensity Scale Handling:** Currently passing `intensity_scale=None` to `save_torch_bundle`. May need to extract from `params.cfg` or model attributes for downstream inference parity.

## Next Actions

1. **Regenerate Phase C/D Datasets (if missing):**
   ```bash
   # Verify dataset existence
   ls -la tmp/phase_c_f2_cli/dose_1000/
   ls -la tmp/phase_d_f2_cli/dose_1000/dense/

   # If missing, rerun Phase C/D generators and record commands in summary
   ```

2. **Execute Real Training (Phase E5):**
   ```bash
   export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
   python -m studies.fly64_dose_overlap.training \
     --phase-c-root tmp/phase_c_f2_cli \
     --phase-d-root tmp/phase_d_f2_cli \
     --artifact-root tmp/phase_e_training_gs2 \
     --dose 1000 --view dense --gridsize 2 \
     | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T210500Z/phase_e_training_bundle/cli/dose1000_dense_train.log
   ```

3. **Copy Artifacts to Hub:**
   ```bash
   cp -r tmp/phase_e_training_gs2/pinn tmp/phase_e_training_gs2/baseline \
     plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T210500Z/phase_e_training_bundle/cli/
   ```

4. **Verify Bundle Contents:**
   ```bash
   unzip -l tmp/phase_e_training_gs2/pinn/wts.h5.zip
   # Validate manifest.dill contains 'models': ['autoencoder', 'diffraction_to_obj']
   ```

5. **Update Phase G Comparison Inventory:**
   - Document bundle paths in `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T162500Z/phase_g_inventory/analysis/inventory.md`
   - Unblock Phase G comparisons (dose_1000 dense/test) with real bundles

## Module Scope

**Declared:** `Module scope: { CLI/config }`
**Rationale:** Changes confined to training runner helper and test suite; no cross-module dependencies.

## Files Changed

- `studies/fly64_dose_overlap/training.py` (import + bundle persistence logic)
- `tests/study/test_dose_overlap_training.py` (new test: `test_execute_training_job_persists_bundle`)

## Artifacts Directory

All loop artifacts saved to:
```
plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T210500Z/phase_e_training_bundle/
├── red/
│   └── pytest_execute_training_job_bundle_red.log
├── green/
│   ├── pytest_execute_training_job_bundle_green.log
│   ├── pytest_training_cli_suite_green.log
│   └── pytest_full_suite.log
├── collect/
│   └── pytest_training_cli_collect.log
├── cli/
│   └── (pending: real training output)
├── analysis/
│   └── summary.md (this file)
└── docs/
    └── (pending: doc sync updates)
```

## Commit Message Template

```
STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 training: add bundle persistence (tests: test_execute_training_job_persists_bundle)

Implement bundle persistence in execute_training_job per specs/ptychodus_api_spec.md §4.6.
After successful training, call save_torch_bundle to emit wts.h5.zip archives with dual-model
structure (autoencoder + diffraction_to_obj). Populate result['bundle_path'] for manifest emission.

Changes:
- studies/fly64_dose_overlap/training.py: import save_torch_bundle, add persistence logic
- tests/study/test_dose_overlap_training.py: add test_execute_training_job_persists_bundle

Test results:
- RED: AttributeError (expected) → GREEN: PASSED (1 passed in 3.76s)
- Regression suite: 3 passed (training_cli tests)
- Full suite: 395 passed, 1 pre-existing failure (test_interop_h5_reader)

Unblocks: Phase G comparison runs requiring real training bundles

Acceptance IDs: AT-49, AT-50, AT-51, AT-52, AT-53
```

## Doc Sync Checklist

- [ ] Update `docs/TESTING_GUIDE.md` §Phase E with new test selector
- [ ] Update `docs/development/TEST_SUITE_INDEX.md` with test metadata
- [ ] Record bundle path contract in `docs/workflows/pytorch.md` §12
- [ ] Add bundle persistence example to `docs/DEVELOPER_GUIDE.md`
