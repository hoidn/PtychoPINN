### Turn Summary
Implemented `compute_dataset_intensity_stats` helper in `ptycho/loader.py` and updated `scripts/inspect_ptycho_data.py::load_ptycho_data` to preserve dataset_intensity_stats from NPZ keys or fallback-compute from X array.
Added regression tests proving the helper handles raw/normalized/rank-3 data and that NPZ loaders preserve stats without touching `_tensor_cache` (PINN-CHUNKED-001).
Next: remaining manual constructors (dose_response_study, data_preprocessing) may need similar updates; amplitude bias investigation continues in D5.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T013900Z/ (pytest logs, collection logs)

## Implementation Details

**SPEC lines implemented (specs/spec-ptycho-core.md §Normalization Invariants lines 87-89):**
> 1) Dataset-derived mode (preferred): `s = sqrt(nphotons / E_batch[Σ_xy |Ψ|²])` computed from illuminated objects over the dataset.
> 2) Closed-form fallback: `s ≈ sqrt(nphotons) / (N/2)` when dataset statistics are unavailable at runtime.

### Files Changed

1. **ptycho/loader.py** — Added `compute_dataset_intensity_stats()` helper function
   - Shared NumPy reducer that accepts raw diffraction or normalized data + intensity_scale
   - Returns `{'batch_mean_sum_intensity', 'n_samples'}` dict
   - Never touches TensorFlow tensors (PINN-CHUNKED-001 compliant)

2. **scripts/inspect_ptycho_data.py** — Updated `load_ptycho_data()`
   - Checks for stored NPZ keys (`dataset_intensity_stats_batch_mean`, `dataset_intensity_stats_n_samples`)
   - Falls back to computing from X array when keys are missing
   - Always attaches `dataset_intensity_stats` to returned container

3. **tests/test_loader_normalization.py** — Added `test_manual_dataset_stats_helper`
   - Tests raw data path, normalized data path, error handling, and rank-3 tensors

4. **tests/scripts/test_inspect_ptycho_data.py** — New file with 2 tests
   - `test_load_preserves_dataset_stats`: Tests both stored-keys and fallback paths
   - `test_load_preserves_tensor_cache_empty`: Verifies PINN-CHUNKED-001 compliance

### Test Results (All Pass)

```
tests/test_loader_normalization.py::TestNormalizeData::test_dataset_stats_attachment PASSED
tests/test_loader_normalization.py::TestNormalizeData::test_manual_dataset_stats_helper PASSED
tests/test_train_pinn.py::TestIntensityScale::test_uses_dataset_stats PASSED
tests/scripts/test_inspect_ptycho_data.py::TestInspectPtychoData::test_load_preserves_dataset_stats PASSED
tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke PASSED
```
