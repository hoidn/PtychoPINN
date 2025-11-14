# Blocker: Chunking Architecture Limitation

## Timestamp  
2025-11-13T23:00:00Z

## Status
BLOCKED - Chunking approach requires deeper architectural refactor

## Context
Implemented `--pinn-chunk-size` flag with chunked PINN inference similar to BASELINE-CHUNKED-001 approach (commit pending).

## Error Encountered
When attempting chunked PINN inference by slicing RawData before creating containers:
```
ValueError: Requesting 320 groups but only 160 points available (gridsize=2, C=4).
K choose C oversampling is required but not enabled.
```

## Root Cause Analysis
The architectural limitation is deeper than anticipated:

1. **Grouping vs Chunking Order**: The test NPZ file contains 5216 *already-grouped* diffraction patterns, not raw images
2. **Container Creation Bottleneck**: `create_ptycho_data_container()` eagerly converts ALL groups to GPU tensors via `combine_complex()` in `PtychoDataContainer.__init__`
3. **Slice-Then-Group Fails**: Slicing RawData before grouping doesn't work because the data is already grouped
4. **Group-Then-Slice Needed**: Would require either:
   - Loading full dataset into CPU arrays, creating container chunks WITHOUT GPU conversion, OR
   - Refactoring `PtychoDataContainer` to accept pre-grouped data and delay GPU tensor conversion

## Evidence
- Test run log: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/compare_models_dense_test_debug_chunked_pinn.log`
- Train split (5088 groups): Fits in GPU memory
- Test split (5216 groups): OOM at `combine_complex` GPU tensor allocation
- Debug run (320 groups requested, 160 available after slice): Grouping validation error

## Architecture Refactor Scope
To properly support chunked PINN inference would require:
1. Refactor `PtychoDataContainer.__init__` to delay GPU tensor conversion
2. Add lazy loading or chunked tensor creation methods
3. Update all downstream code expecting eager tensors
4. Extensive testing across all workflows

**Estimated effort**: 2-3 implementation loops + validation

## Pragmatic Mitigation
Per blocker document mitigation option 3: **Proceed with train-only Baseline metrics for Phase G**.

Train split compare_models already succeeded with full Baseline stats:
- `analysis/dose_1000/dense/train/comparison_metrics.csv`: Complete Baseline rows
- Baseline mean=0.188, 78.7M nonzero pixels
- All three models (PINN, Baseline, PtyChi) have metrics

Test split can proceed with PINN vs PtyChi only (2-way comparison) by skipping Baseline inference.

## Recommended Next Steps
1. Commit current `--pinn-chunk-size` implementation with documentation noting the architectural limitation
2. Rerun test-split compare_models WITHOUT `--pinn-chunk-size` and WITHOUT Baseline (remove `--baseline_dir`)
3. Proceed with Phase D/Phase G pipeline using train-split Baseline metrics + test-split PINN/PtyChi metrics
4. File architectural refactor as future work if full test-split Baseline metrics are required

## Alternative: Reduce Test Split Size
If test-split Baseline metrics are critical:
- Use `--n-test-groups 5000` (slightly under train split size of 5088)
- OR regenerate Phase C with smaller test split

