# Phase C4.D3/D4/F Integration Test Failure Logs — Ralph Loop

**Date:** 2025-10-20T081500Z
**Task:** ADR-003-BACKEND-API C4.D3/D4/F — Capture integration regression evidence and CLI smoke failure signatures
**Mode:** TDD (Evidence Capture)
**Loop Owner:** Ralph (Engineer)
**Branch:** feature/torchapi

---

## Executive Summary

Executed C4.D3 (integration test) and C4.D4 (manual CLI smoke) per `input.md` directive. Both commands captured expected failure signatures with comprehensive stdout/stderr logs stored under this artifact hub. C4.F hygiene completed (relocated `train_debug.log` from repo root). All three tasks COMPLETE per input.md checklist.

**Key Findings:**
1. **Integration Test (C4.D3):** Training completes successfully, but **inference fails with `NotImplementedError` in `load_torch_bundle`** (Phase D3.C blocker as documented).
2. **CLI Smoke (C4.D4):** Training fails earlier with **channel mismatch error** (`expected input[4, 4, 64, 64] to have 1 channels, but got 4 channels`), indicating dataloader is producing wrong tensor structure for gridsize=2.
3. **Artifact Hygiene (C4.F):** `train_debug.log` (119KB) successfully relocated from repo root to artifact directory.

---

## C4.D3: Integration Test Execution

### Command
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T081500Z/phase_c4_cli_integration_debug/pytest_integration.log
```

### Result
**FAILED** in 16.52s

### Failure Signature
```
NotImplementedError: load_torch_bundle model reconstruction not yet implemented.
Requires create_torch_model_with_gridsize helper from Phase D3.C.
params.cfg successfully restored: N=64, gridsize=1
```

### Analysis
- **Training Phase:** ✅ SUCCEEDS (checkpoint created at `checkpoints/last.ckpt`)
- **Inference Phase:** ❌ FAILS when attempting to load the bundle
- **Root Cause:** Phase D3.C (`load_torch_bundle` model reconstruction) is NOT YET IMPLEMENTED
- **Blocker Location:** `ptycho_torch/model_manager.py:267`
- **Call Chain:**
  ```
  inference.py:515 (cli_main)
    → workflows/components.py:995 (load_inference_bundle_torch)
      → model_manager.py:267 (load_torch_bundle)
        → NotImplementedError
  ```

### Notable Observations
1. **CONFIG-001 Warning:** `params.cfg already populated. Set force=True to overwrite existing values.` (config_factory.py:608)
2. **TensorFlow Warnings:** Multiple CUDA/cuFFT/cuDNN registration warnings (expected in dual TF+PyTorch environment)
3. **CUDA_VISIBLE_DEVICES="" Respected:** All CUDA diagnostics confirm CPU-only execution

### Artifacts
- **Log:** `pytest_integration.log` (full pytest -vv output with traceback)

---

## C4.D4: Manual CLI Smoke Test

### Command
```bash
CUDA_VISIBLE_DEVICES="" python -m ptycho_torch.train \
  --train_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz \
  --output_dir /tmp/cli_smoke \
  --n_images 64 \
  --max_epochs 1 \
  --accelerator cpu \
  --deterministic \
  --num-workers 0 \
  --learning-rate 1e-4
```

### Result
**FAILED** — RuntimeError: `Lightning training failed. See logs for details.`

### Failure Signature
```
RuntimeError: Given groups=1, weight of size [64, 1, 3, 3],
expected input[4, 4, 64, 64] to have 1 channels, but got 4 channels instead
```

### Analysis
- **Phase:** Training initialization succeeded, Lightning Trainer started, but **first forward pass FAILED**
- **Root Cause:** **Dataloader channel mismatch** — model expects 1-channel input (gridsize=1 semantics), but dataloader yields 4-channel tensors (gridsize=2 behavior)
- **Blocker Location:** `ptycho_torch/model.py:97` (conv1 layer in EncoderBlock)
- **Call Chain:**
  ```
  train.py:642 (cli_main)
    → workflows/components.py:155 (run_cdi_example_torch)
      → workflows/components.py:940 (train_cdi_model_torch)
        → workflows/components.py:690 (_train_with_lightning)
          → model.py:1229 (training_step)
            → model.py:1187 (compute_loss)
              → model.py:1144 (forward)
                → model.py:894 (autoencoder.forward)
                  → model.py:505 (encoder.forward)
                    → model.py:143 (EncoderBlock.forward)
                      → model.py:97 (conv1)
                        → RuntimeError
  ```

### Notable Observations
1. **Factory Config Discrepancy:**
   - Factory reports: `gridsize=(2, 2)` (tuple notation, implying 2×2=4 channels)
   - Fixture contains: `gridsize=1` metadata (minimal dataset generated with gridsize=1)
   - Model architecture: Built for 1-channel input (`weight of size [64, 1, 3, 3]`)
   - Dataloader output: Yields 4-channel tensors (`input[4, 4, 64, 64]`)

2. **Successful CLI Flag Parsing:**
   - Execution config correctly applied: `accelerator=cpu`, `deterministic=True`, `learning_rate=0.0001`
   - Config factory successfully created configs: `N=64`
   - `--device` deprecation warning issued (expected per Phase C design)

3. **Data Loading Success:**
   - RawData.from_file() succeeded: `diff3d shape: (64, 64, 64)`
   - Grouped data generated: `neighbor-sampled diffraction shape (64, 64, 64, 4)` ← **4 channels produced**
   - Ground truth patches generated from objectGuess (expected for minimal fixture without Y array)

4. **Model Instantiation Success:**
   - Lightning module initialized: `2.3 M Trainable params`
   - No decoder attention modules added (expected for default config)

### Critical Issue: GridSize Confusion
**Problem:** The CLI smoke test specifies `--n_images 64` (no explicit `--gridsize`), but the factory is interpreting this as `gridsize=(2, 2)` while the model was built for `gridsize=1`. This results in:
- Model architecture expecting **1 input channel** (gridsize=1 semantics)
- Dataloader producing **4 input channels** (gridsize² where gridsize=2)

**Likely Cause:** Factory default or inference logic not aligning with fixture metadata. The fixture was generated with `gridsize=1` (stratified sampling for minimal case), but factory/workflow is applying `gridsize=2` during reconstruction.

### Artifacts
- **Log:** `manual_cli_smoke.log` (full CLI stdout/stderr with traceback and diagnostic output)

---

## C4.F: Hygiene Artifact Cleanup

### Action
Relocated `train_debug.log` (119 KB) from repository root to artifact directory per input.md hygiene requirements.

### Command
```bash
mv train_debug.log plans/active/ADR-003-BACKEND-API/reports/2025-10-20T081500Z/phase_c4_cli_integration_debug/train_debug.log
```

### Verification
✅ File successfully moved, repo root now clean of stray artifacts.

---

## Artifacts Produced

All artifacts stored under: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T081500Z/phase_c4_cli_integration_debug/`

| File | Size | Description |
|------|------|-------------|
| `pytest_integration.log` | ~6 KB | C4.D3 integration test output (NotImplementedError in load_torch_bundle) |
| `manual_cli_smoke.log` | ~8 KB | C4.D4 CLI smoke test output (channel mismatch in first forward pass) |
| `train_debug.log` | 119 KB | Relocated training debug log from repo root (C4.F hygiene) |
| `summary.md` | This file | Comprehensive analysis of C4.D3/D4/F findings |

---

## Exit Criteria Validation

Per `input.md` checklist:

- [x] **C4.D3:** Run integration test with `CUDA_VISIBLE_DEVICES=""`, capture output to `pytest_integration.log`
  - ✅ Test executed, FAILED with documented NotImplementedError
  - ✅ Log captured with full traceback (16.52s runtime)

- [x] **C4.D4:** Run manual CLI smoke, capture output to `manual_cli_smoke.log`
  - ✅ Command executed with all specified flags
  - ✅ Log captured with full diagnostic output and error traceback
  - ✅ Temporary outputs left under `/tmp/cli_smoke` per input.md guidance (untracked)

- [x] **C4.F:** Relocate `train_debug.log` and update summary
  - ✅ File moved from repo root to artifact hub
  - ✅ Summary documents both failure modes comprehensively
  - ✅ Next actions identified (see below)

---

## Next Actions (Phase C4 Completion)

Based on captured evidence, the following blockers must be resolved before C4.D validation can proceed:

### 1. **Phase D3.C Model Loading (CRITICAL BLOCKER for C4.D3)**
- **Task:** Implement `load_torch_bundle` model reconstruction helper
- **File:** `ptycho_torch/model_manager.py:267`
- **Requirement:** Create `create_torch_model_with_gridsize()` function (analogous to TensorFlow `create_model_with_gridsize`)
- **Acceptance:** Integration test inference phase completes without NotImplementedError

### 2. **GridSize Configuration Reconciliation (CRITICAL BLOCKER for C4.D4)**
- **Task:** Fix factory gridsize inference mismatch
- **Root Cause:** Factory is producing `gridsize=(2, 2)` when fixture metadata specifies `gridsize=1`
- **Files to Investigate:**
  - `ptycho_torch/config_factory.py` (gridsize default/inference logic)
  - `ptycho_torch/workflows/components.py` (_build_lightning_dataloaders channel logic)
- **Acceptance:** CLI smoke test progresses past first forward pass with correct channel dimensions

### 3. **Fixture Validation (Optional Enhancement)**
- **Task:** Regenerate `minimal_dataset_v1.npz` with explicit gridsize metadata or update factory to respect fixture constraints
- **Rationale:** Ensure fixture CONTRACT-001 compliance includes gridsize parameter
- **Deferrable:** Can be addressed in Phase E fixture hardening

---

## Compliance Notes

- **CONFIG-001:** Factory correctly calls `update_legacy_dict()` before workflow execution (evidenced by UserWarning about populated params.cfg)
- **DATA-001:** Fixture correctly loaded (diff3d shape matches expectations)
- **POLICY-001:** PyTorch stack available and functional (no import errors)
- **FORMAT-001:** No legacy transpose issues observed (canonical (N,H,W) format loaded successfully)

---

## Supervisor Handoff

Ralph has completed C4.D3/D4/F per input.md directive. All requested evidence captured and stored. Next engineer loop should address the two critical blockers (D3.C model loading + gridsize reconciliation) before attempting C4.D validation again.

**Recommendation:** Mark C4.D3/D4/F rows in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md` as `[x]` with blocker notes, and create Phase D3.C execution plan for model loading implementation.
