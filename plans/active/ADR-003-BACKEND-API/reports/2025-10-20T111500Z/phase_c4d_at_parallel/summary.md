# Phase C4.D.B3 Evidence Collection Summary

## Objective
Validate PyTorch backend parity by rerunning targeted regression tests and executing a gridsize=2 CLI smoke test to capture fresh GREEN evidence for Phase C4.D completion.

## Test Execution Results

### 1. Gridsize Regression Test
**Selector**: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize -vv`

**Result**: ✅ PASSED
**Runtime**: 4.99s
**Log**: `pytest_gridsize_green.log`

**Key Observations**:
- Test validates that Lightning training correctly handles the coords_relative tensor layout for gridsize configurations
- Confirms the axis permutation fix from Phase C4.D.B2 remains stable
- 4 warnings (expected): test_data_file override suggestion, params.cfg double-population, checkpoint directory exists, num_workers bottleneck

### 2. Bundle Loader Test
**Selector**: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_bundle_loader_returns_modules -vv`

**Result**: ✅ PASSED
**Runtime**: 13.02s
**Log**: `pytest_bundle_loader_green.log`

**Key Observations**:
- Confirms `load_torch_bundle()` successfully reconstructs model instances from persisted checkpoints
- Validates Phase A implementation: bundle persistence and restoration works end-to-end
- No regressions detected

### 3. Integration Workflow Test
**Selector**: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv`

**Result**: ✅ PASSED
**Runtime**: 16.77s
**Log**: `pytest_integration_green.log`

**Key Observations**:
- Full train → save → load → infer cycle completes successfully
- Checkpoint persistence and restoration verified
- Inference phase produces valid amplitude/phase reconstructions
- Validates end-to-end workflow parity with TensorFlow baseline

### 4. CLI Smoke Test (gridsize=2)
**Command**:
```bash
python -m ptycho_torch.train \
  --train_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz \
  --test_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz \
  --output_dir /tmp/cli_smoke \
  --n_images 64 \
  --gridsize 2 \
  --batch_size 4 \
  --max_epochs 1 \
  --disable_mlflow \
  --device cpu \
  --accelerator cpu \
  --learning-rate 1e-3 \
  --num-workers 0 \
  --deterministic
```

**Result**: ✅ SUCCESS
**Runtime**: ~4.5s (training phase)
**Log**: `manual_cli_smoke_gs2.log`

**Key Observations**:
- Training completed successfully with gridsize=2 configuration
- Model bundle saved to `/tmp/cli_smoke/wts.h5.zip`
- Correct tensor shapes observed:
  - Input diffraction: `(64, 64, 64, 4)` (batch, H, W, gridsize²)
  - coords_relative correctly permuted and contiguous
- Lightning model summary: 2.3M trainable params (expected for gridsize=2)
- Warnings observed (expected):
  - `--device` deprecation notice (Phase D migration)
  - `test_data_file` override suggestion (non-blocking)
  - TensorFlow XLA registration warnings (environment noise)
  - Lightning logger not configured (expected for CLI smoke)

## Dataset & Environment Details

**Test Dataset**: `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz`
- 64 scan positions (stratified sampling from canonical dataset)
- Canonical (N,H,W) format per DATA-001
- Size: 1.9M
- SHA256: `6c2fbea0dcadd950385a54383e6f5f731282156d19ca4634a5a19ba3d1a5899c`

**Environment**:
- Python 3.11.13
- PyTorch 2.8.0+cu128
- Lightning 2.5.5
- CUDA_VISIBLE_DEVICES="" (CPU-only execution)
- Deterministic mode enabled

## Phase C4.D.B3 Exit Criteria Assessment

✅ **All exit criteria met**:
1. Gridsize regression test GREEN (pytest_gridsize_green.log)
2. Bundle loader test GREEN (pytest_bundle_loader_green.log)
3. Integration workflow test GREEN (pytest_integration_green.log)
4. CLI smoke test with gridsize=2 completed successfully (manual_cli_smoke_gs2.log)
5. No blockers encountered

## Remaining Risks & Follow-Up Actions

### Low-Risk Items (Phase C Close-Out)
1. **Documentation Updates** (Phase C.C1–C3):
   - Update parent plan `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md` to mark C4.D rows complete
   - Update `docs/workflows/pytorch.md` §12 with gridsize=2 validation evidence
   - Append Attempt #35 to `docs/fix_plan.md` with artifact links

### No Critical Blockers Detected
- All targeted selectors passed without errors
- CLI smoke test completed end-to-end with expected warnings only
- Checkpoint persistence and restoration verified across all test tiers

## Artifacts Generated

All artifacts stored under:
`plans/active/ADR-003-BACKEND-API/reports/2025-10-20T111500Z/phase_c4d_at_parallel/`

1. `pytest_gridsize_green.log` (4.99s runtime, 1 passed)
2. `pytest_bundle_loader_green.log` (13.02s runtime, 1 passed)
3. `pytest_integration_green.log` (16.77s runtime, 1 passed)
4. `manual_cli_smoke_gs2.log` (CLI smoke test full output)
5. `summary.md` (this document)

## Conclusion

Phase C4.D.B3 evidence collection is **COMPLETE**. All regression tests passed, CLI smoke test with gridsize=2 succeeded, and no blockers were encountered. Ready to proceed with Phase C close-out documentation updates.

**Recommendation**: Mark plan row B3 as `[x]` and transition to Phase C documentation tasks (C1–C3).
