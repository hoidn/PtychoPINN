# Phase C3.A Artifact Audit — TEST-PYTORCH-001

**Date:** 2025-10-19
**Artifact Hub:** `plans/active/TEST-PYTORCH-001/reports/2025-10-19T130900Z/phase_c_modernization/`
**Phase:** C3 (Validation & Documentation)

## Objective

Inspect training and inference artifacts produced during the Phase C2 GREEN rerun to validate:
1. Checkpoint persistence format and location
2. Reconstruction output file formats and sizes
3. Artifact consistency with PyTorch workflow specifications

## Test Execution Context

**Pytest selector:**
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
```

**Runtime:** 35.98s (within 120s budget per Phase A baseline)
**Result:** 1 PASSED ✅
**Log:** `pytest_modernization_rerun.log` (captured via tee)

## Artifact Inspection

### Training Outputs

**Location:** `tmp_path / "training_outputs"` (pytest-managed temporary directory)

Expected artifacts per `_run_pytorch_workflow` implementation:
- `checkpoints/last.ckpt` — Lightning checkpoint with model state + hyperparameters
- Training logs (Lightning default location)

**Validation:**
- Checkpoint existence verified via assertion at `test_integration_workflow_torch.py:193`
- No permanent artifacts retained (pytest cleans up `tmp_path` after test)

### Inference Outputs

**Location:** `tmp_path / "pytorch_output"` (pytest-managed temporary directory)

Expected artifacts per `_run_pytorch_workflow` implementation:
- `reconstructed_amplitude.png` — Amplitude reconstruction visualization
- `reconstructed_phase.png` — Phase reconstruction visualization

**Validation:**
- Both PNG files verified to exist via assertions at lines 194-195
- File sizes confirmed >1KB via assertions at lines 198-199
- No permanent artifacts retained (pytest cleans up `tmp_path` after test)

## Artifact Format Specifications

### Lightning Checkpoint (`last.ckpt`)

**Format:** PyTorch Lightning checkpoint (`.ckpt`)
**Contents (per Phase D1c implementation):**
- `state_dict`: Model weights
- `hyper_parameters`: Serialized dataclass configs (model_config, data_config, training_config, inference_config)
- `epoch`, `global_step`: Training metadata
- `optimizer_states`: Optimizer state for resumption
- `callbacks`, `lr_schedulers`, `loops`: Lightning internal state

**Contract:** Must be loadable via `PtychoPINN_Lightning.load_from_checkpoint()` without explicit config kwargs (per Phase D1c fix)

### Reconstruction Images (`*.png`)

**Format:** PNG image files
**Contract:**
- Non-empty (>1KB per assertions)
- Generated via `ptycho_torch.inference` CLI module
- Produced by `_reassemble_cdi_image_torch` workflow (Phase D2.C implementation)

## Artifact Lifecycle

**Creation:** Artifacts generated during `_run_pytorch_workflow` subprocess calls
**Validation:** Assertions in test function verify existence and basic properties
**Cleanup:** Pytest automatically removes `tmp_path` contents after test completion
**Persistence:** No artifacts retained post-test (transient validation only)

## Findings

### Checkpoint Persistence ✅
- Lightning checkpoint successfully created at expected path (`checkpoints/last.ckpt`)
- Checkpoint loading now functional per Phase D1c (hyperparameter serialization fix)
- No size checks performed (relying on Lightning's internal validation)

### Reconstruction Outputs ✅
- Both amplitude and phase PNGs generated successfully
- File size assertions (>1KB) validate non-empty outputs
- Format compatible with `ptycho_torch.inference` output contract

### Performance ✅
- Runtime 35.98s (35.86s in C2 GREEN, <1% variance)
- Well within 120s budget established in Phase A baseline
- No performance regression vs. legacy unittest baseline (32.54s in Phase A)

## Recommendations

### Artifact Retention (Optional Enhancement)
Current implementation uses transient `tmp_path` with automatic cleanup. For debugging/analysis, consider:
- Optional `--keep-artifacts` pytest flag to preserve outputs
- Symlink pattern from `tmp_path` to known location (e.g., `plans/active/TEST-PYTORCH-001/artifacts/latest/`)

### Checkpoint Size Validation (Deferred)
Add assertion checking checkpoint file size if anomalies observed. Current implicit validation via successful load is sufficient for regression guard.

### Reconstruction Quality Metrics (Phase D+ Enhancement)
Current tests validate presence only. Future enhancements could add:
- Image dimension checks
- Pixel value range validation (amplitude non-negative, phase within [-π, π])
- Comparison to reference reconstruction (requires fixture groundtruth)

## References

- Helper implementation: `tests/torch/test_integration_workflow_torch.py:65-161`
- Test assertions: `tests/torch/test_integration_workflow_torch.py:188-199`
- Phase C2 summary: `plans/active/TEST-PYTORCH-001/reports/2025-10-19T122449Z/phase_c_modernization/summary.md`
- Phase D1c checkpoint fix: `docs/fix_plan.md` INTEGRATE-PYTORCH-001-STUBS Attempt #34
- Phase D2.C stitching implementation: `docs/fix_plan.md` INTEGRATE-PYTORCH-001-STUBS Attempt #28
- PyTorch workflow guide: `docs/workflows/pytorch.md` §§5-8

## Exit Criteria Validation

✅ **C3.A Complete:** Artifact audit documented with format specifications, lifecycle notes, and findings
✅ Checkpoint persistence verified (Lightning format, hyperparameter serialization per D1c)
✅ Reconstruction outputs verified (PNG format, non-empty file sizes)
✅ Runtime budget maintained (35.98s within 120s allocation)
✅ Recommendations captured for future enhancements (optional artifact retention, quality metrics)
