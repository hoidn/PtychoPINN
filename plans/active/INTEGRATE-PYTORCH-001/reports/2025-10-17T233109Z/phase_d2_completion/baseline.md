# Phase D2 Baseline: PyTorch Workflow Stub Inventory (2025-10-17T233109Z)

## Executive Summary

This baseline documents the current state of PyTorch workflow orchestration stubs in `ptycho_torch/workflows/components.py` before Phase D2.B/C implementation begins. The integration test currently fails at Lightning checkpoint loading, revealing the need for complete training and inference orchestration.

**Key Findings:**
- Workflow entry points (signatures, CONFIG-001 gates) are complete
- Training orchestration (`_train_with_lightning`) is a stub returning placeholder results
- Stitching/inference (`_reassemble_cdi_image_torch`) raises `NotImplementedError`
- Integration test fails at Lightning checkpoint loading due to missing config serialization in checkpoint

**Status:** Phase A baseline complete; Phase B (Lightning training) and Phase C (inference/stitching) ready for TDD implementation.

---

## 1. Stub Inventory & Callsite Analysis

### 1.1. Complete Entry Points (Phase D2.A Scaffold)

**File:** `ptycho_torch/workflows/components.py`

#### `run_cdi_example_torch` (lines 83-192)
**Status:** ✅ Scaffold COMPLETE (CONFIG-001 gate present, delegates to stubs)

**Functionality:**
- Entry signature matches TensorFlow `ptycho.workflows.components.run_cdi_example`
- **CONFIG-001 compliance:** Calls `update_legacy_dict(params.cfg, config)` at line 150 before delegation
- Delegates to `train_cdi_model_torch` for training (line 155)
- Conditionally invokes `_reassemble_cdi_image_torch` when `do_stitching=True` (lines 162-170)
- Saves models via `save_torch_bundle` when `config.output_dir` provided (lines 177-188)
- Returns tuple `(recon_amp, recon_phase, train_results)` per spec §4.5

**Open TODOs:**
- Phase D2.B: Full Lightning training implementation
- Phase D2.C: Inference/stitching implementation
- MLflow disable flag handling (currently no `--disable_mlflow` CLI support)

#### `train_cdi_model_torch` (lines 361-418)
**Status:** ✅ Scaffold COMPLETE (data normalization via Phase C adapters)

**Functionality:**
- Normalizes `train_data` and `test_data` to `PtychoDataContainerTorch` via `_ensure_container` helper (lines 400-407)
- Delegates to `_train_with_lightning` stub (line 416)
- Probe initialization deferred to Lightning implementation (line 412)

**Dependencies:**
- Phase C adapters: `RawDataTorch`, `PtychoDataContainerTorch` (imported lines 68-70)
- Helper: `_ensure_container` (lines 195-262)

**Open TODOs:**
- Phase D2.B: Implement probe initialization for PyTorch
- Phase D2.B: Replace `_train_with_lightning` stub

#### `load_inference_bundle_torch` (lines 421-480)
**Status:** ✅ Scaffold COMPLETE (delegates to Phase D3 persistence module)

**Functionality:**
- Delegates to `load_torch_bundle` from `ptycho_torch.model_manager` (line 471)
- CONFIG-001 gate executed inside `load_torch_bundle` (params.cfg restoration)
- Returns `(models_dict, params_dict)` matching TensorFlow signature

**Dependencies:**
- `ptycho_torch.model_manager.load_torch_bundle` (imported line 70)

**Open TODOs:**
- Phase D3: Ensure `load_torch_bundle` correctly restores params.cfg from checkpoint metadata

---

### 1.2. Stub Functions (Phase D2.B/C Implementation Targets)

#### `_train_with_lightning` (lines 265-305)
**Status:** 🔶 STUB (returns placeholder dict without Lightning execution)

**Current Behavior:**
```python
return {
    "history": {
        "train_loss": [0.5, 0.3],
        "val_loss": [0.6, 0.4] if test_container is not None else None
    },
    "train_container": train_container,
    "test_container": test_container,
}
```

**Callsites:**
- `train_cdi_model_torch:416` (only caller)

**Phase D2.B Implementation Requirements:**
1. Import Lightning components (guarded for torch-optional compatibility)
2. Instantiate `PtychoPINN_Lightning` module from `ptycho_torch.model` with:
   - `model_config`, `data_config`, `training_config`, `inference_config` (4 required args per integration test error)
3. Configure `Trainer`:
   - `max_epochs` from `config.nepochs`
   - `devices` from CLI or config
   - `gradient_clip_val`, checkpoint callbacks
4. Create DataModule or dataloaders from containers:
   - Train: `TensorDictDataLoader(PtychoDataset(train_container))`
   - Val: `TensorDictDataLoader(PtychoDataset(test_container))` if test_container not None
5. Execute `trainer.fit(model, train_dataloader, val_dataloader)`
6. Extract training history from `trainer.callback_metrics`
7. Return structured dict with:
   - `history`: Dict of training/validation losses
   - `train_container`, `test_container`: Input containers
   - `models`: Dict with trained Lightning module for persistence

**Critical Findings from Integration Test Failure:**
- Lightning checkpoint requires config objects as `__init__` args (not inferred from checkpoint)
- Current checkpoint saving (via Lightning auto-checkpoint) does not serialize config metadata
- **Resolution:** Checkpoints must include `model_config`, `data_config`, `training_config`, `inference_config` in hyperparameters or metadata

**References:**
- Spec §4.5: Reconstructor lifecycle contract
- `docs/workflows/pytorch.md` §5: Lightning training knobs
- Integration test error (line 442-453): `PtychoPINN_Lightning.__init__() missing 4 required positional arguments`

#### `_reassemble_cdi_image_torch` (lines 308-358)
**Status:** ❌ STUB (raises `NotImplementedError`)

**Current Behavior:**
```python
raise NotImplementedError(
    "PyTorch inference/stitching path not yet implemented. "
    "Phase D2.C stub implementation in place for orchestration testing."
)
```

**Callsites:**
- `run_cdi_example_torch:166` (only caller, guarded by `if do_stitching and test_data is not None`)

**Phase D2.C Implementation Requirements:**
1. Normalize `test_data` → `PtychoDataContainerTorch` via `_ensure_container`
2. Load trained Lightning module from checkpoint (reuse `load_inference_bundle_torch` or direct load)
3. Run model inference to get reconstructed patches:
   - `trainer.predict(model, test_dataloader)` or direct `model.forward_predict()`
4. Convert outputs to NumPy (detach from PyTorch tensors)
5. Apply coordinate transformations:
   - `flip_x`, `flip_y` coordinate flips
   - `transpose` dimension swap
   - `coord_scale` from config or container metadata
6. Reassemble patches using PyTorch equivalent of `reassemble_position`:
   - Reference TensorFlow implementation: `ptycho/tf_helper.py:reassemble_position`
   - Reference PyTorch helper: `ptycho_torch/helper.py` (to be extended)
7. Extract amplitude + phase from complex reconstruction:
   - `recon_amp = np.abs(recon_complex)`
   - `recon_phase = np.angle(recon_complex)`
8. Return tuple `(recon_amp, recon_phase, results_dict)` where `results_dict` includes:
   - `obj_tensor_full`: Full reconstructed object
   - `global_offsets`: Coordinate offsets used for stitching
   - `coords_nominal`: Nominal scan coordinates

**References:**
- Spec §4.5: Reconstructor output contract
- `ptycho/workflows/components.py:714-721`: TensorFlow stitching baseline
- `ptycho/image/registration.py`: Phase alignment utilities (may need PyTorch port)

---

### 1.3. Helper Functions (Phase D2.A Complete)

#### `_ensure_container` (lines 195-262)
**Status:** ✅ COMPLETE (Phase C adapters integrated)

**Functionality:**
- Normalizes input data (`RawData`, `RawDataTorch`, or `PtychoDataContainerTorch`) to `PtychoDataContainerTorch`
- Case 1: Already container → return as-is (duck-type check lines 224-226)
- Case 2: TensorFlow `RawData` → wrap with `RawDataTorch` (lines 228-242)
- Case 3: `RawDataTorch` → call `generate_grouped_data` → create container (lines 245-257)
- Case 4: Unknown type → raise `TypeError` (lines 260-262)

**Dependencies:**
- `ptycho_torch.raw_data_bridge.RawDataTorch` (imported line 68)
- `ptycho_torch.data_container_bridge.PtychoDataContainerTorch` (imported line 69)
- `ptycho.raw_data.RawData` (imported line 64)

**No TODOs:** This helper is complete and ready for Phase D2.B/C use.

---

## 2. Integration Test Baseline Failure Analysis

**Test:** `tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle`

**Execution Command:**
```bash
pytest tests/torch/test_integration_workflow_torch.py -vv
```

**Log Artifact:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T233109Z/phase_d2_completion/pytest_integration_baseline.log`

### Failure Summary

**Phase:** Inference step (checkpoint loading)
**Exit Code:** 1 (subprocess failure)

**Error:**
```
TypeError: PtychoPINN_Lightning.__init__() missing 4 required positional arguments:
  'model_config', 'data_config', 'training_config', and 'inference_config'
```

**Stack Trace Location:**
```
File: ptycho_torch/inference.py:442 (cli_main)
Context: model = PtychoPINN_Lightning.load_from_checkpoint(checkpoint_path)
```

**Root Cause:**
Lightning's `load_from_checkpoint` instantiates the module class using only hyperparameters saved in the checkpoint. The current training workflow (via `ptycho_torch/train.py`) does not serialize config objects into checkpoint metadata, causing the deserialization to fail when it attempts to call `PtychoPINN_Lightning.__init__()` without the required config arguments.

**Training Output (Preceding Inference):**
- Training subprocess succeeded (returncode 0)
- Checkpoint created: `<output_dir>/checkpoints/last.ckpt` ✅
- **Missing:** Config serialization in checkpoint hyperparameters

### Planned Resolution (Phase D2.B)

**Option 1: Serialize Config in Checkpoint Hyperparameters (Recommended)**
1. In `_train_with_lightning`, pass config objects to Lightning module as hyperparameters:
   ```python
   model = PtychoPINN_Lightning(
       model_config=config.model,
       data_config=...,  # Derive from config
       training_config=config,
       inference_config=None,  # Or derive from config
   )
   model.save_hyperparameters()  # Auto-saves __init__ args to checkpoint
   ```
2. Lightning will serialize these configs into `checkpoint['hyper_parameters']`
3. `load_from_checkpoint` will reconstruct module with saved configs

**Option 2: Custom Load Hook**
1. Implement `PtychoPINN_Lightning.load_from_checkpoint` override
2. Manually extract configs from checkpoint metadata or external file
3. Instantiate module with reconstructed configs

**Recommendation:** Option 1 (Serialize Config in Hyperparameters) aligns with Lightning best practices and maintains checkpoint self-containment.

---

## 3. Findings Ledger Coverage Verification

### POLICY-001: PyTorch Mandatory Dependency ✅
**Reference:** `docs/findings.md:7`

**Status:** CONFIRMED in baseline

**Compliance in Workflows Module:**
- PyTorch imports are **mandatory** (lines 66-77)
- No torch-optional guards (imports raise RuntimeError if torch unavailable)
- Phase F3.1/F3.2 migration complete per POLICY-001

**Note:** This differs from original Phase D2.A scaffold design (which had torch-optional guards). Current implementation correctly enforces POLICY-001.

### FORMAT-001: Legacy NPZ Format Auto-Transpose ✅
**Reference:** `docs/findings.md:17`

**Status:** CONFIRMED in baseline

**Relevance to Workflows Module:**
- Auto-transpose heuristic exists in `ptycho_torch/dataloader.py` (per FORMAT-001)
- `_ensure_container` delegates to `RawDataTorch.generate_grouped_data`, which uses dataloader
- No additional handling needed in workflows layer (handled downstream)

**Integration Test Dataset:**
- `datasets/Run1084_recon3_postPC_shrunk_3.npz` is DATA-001 compliant (diffraction: (722, 64, 64))
- No FORMAT-001 transpose triggered during integration test

### CONFIG-001: params.cfg Initialization Order ✅
**Reference:** `docs/findings.md:9`

**Status:** CONFIRMED in baseline

**Compliance in Workflows Module:**
- `run_cdi_example_torch:150` calls `update_legacy_dict(params.cfg, config)` ✅
- Call occurs **before** delegating to `train_cdi_model_torch` (line 155)
- Call occurs **before** downstream data operations (grouping, model init)

**Downstream Consumers:**
- `RawDataTorch.generate_grouped_data` may read `params.cfg['gridsize']`, `params.cfg['N']`
- Legacy modules (if any) invoked during training/inference expect populated `params.cfg`

**Phase D2.B/C Requirement:**
- Ensure Lightning training does not bypass CONFIG-001 gate
- Verify inference path (via `load_inference_bundle_torch`) restores `params.cfg` before model reconstruction

---

## 4. Configuration Snapshot (Baseline Environment)

### Environment Knobs
**PyTorch Version:** 2.8.0+cu128 (CUDA-enabled)
**Device Used in Test:** CPU (via `--device cpu` flag)
**CUDA Available:** Yes (warnings in stderr about cuFFT/cuDNN registration, non-blocking)

### Dataset Configuration (Integration Test)
**File:** `datasets/Run1084_recon3_postPC_shrunk_3.npz`

**Array Shapes (from previous parity runs):**
- `diffraction`: (722, 64, 64) float32 [DATA-001 compliant]
- `probeGuess`: (64, 64) complex128
- `objectGuess`: (128, 128) complex128

**Training CLI Args (from integration test):**
```bash
--train_data_file datasets/Run1084_recon3_postPC_shrunk_3.npz
--test_data_file datasets/Run1084_recon3_postPC_shrunk_3.npz
--output_dir <temp>/training_outputs
--max_epochs 2
--n_images 64
--gridsize 1
--batch_size 4
--device cpu
--disable_mlflow
```

**Inference CLI Args (from integration test):**
```bash
--model_path <temp>/training_outputs
--test_data <same NPZ>
--output_dir <temp>/pytorch_output
--n_images 32
--device cpu
```

**Checkpoint Found:** `<output_dir>/checkpoints/last.ckpt` ✅

---

## 5. Phase D2.B/C Implementation Guidance

### Phase B.B1: Author Failing Lightning Regression Tests
**Location:** `tests/torch/test_workflows_components.py`

**Test Cases to Add:**
1. `test_train_with_lightning_instantiates_module` - Assert `PtychoPINN_Lightning` created with correct config args
2. `test_train_with_lightning_runs_trainer_fit` - Mock trainer, assert `fit()` called
3. `test_train_with_lightning_respects_deterministic_seed` - Assert seed from `config.subsample_seed` used
4. `test_train_with_lightning_returns_structured_results` - Assert dict includes `history`, `train_container`, `models`
5. `test_train_with_lightning_saves_config_in_checkpoint` - Assert checkpoint `hyper_parameters` includes configs

### Phase B.B2: Implement Lightning Orchestration
**Target Function:** `_train_with_lightning` (lines 265-305)

**Implementation Checklist:**
- [ ] Import `PtychoPINN_Lightning` from `ptycho_torch.model`
- [ ] Import `Trainer` from `lightning.pytorch`
- [ ] Import `TensorDictDataLoader`, `PtychoDataset` from `ptycho_torch.dset_loader_pt_mmap`
- [ ] Derive `model_config`, `data_config`, `training_config`, `inference_config` from input `config`
- [ ] Instantiate `PtychoPINN_Lightning` with 4 config args
- [ ] Call `model.save_hyperparameters()` to serialize configs into checkpoint
- [ ] Create dataloaders from `train_container`, `test_container`
- [ ] Configure `Trainer` with `max_epochs=config.nepochs`, `devices=...`, callbacks
- [ ] Execute `trainer.fit(model, train_dataloader, val_dataloader)`
- [ ] Extract `trainer.callback_metrics` for history
- [ ] Return dict with `history`, `train_container`, `test_container`, `models`

### Phase C.C1: Design Inference Data Flow
**Artifact:** `plans/active/INTEGRATE-PYTORCH-001/reports/<TS>/phase_d2_completion/inference_design.md`

**Design Questions:**
1. How to load Lightning module for inference? (via `load_from_checkpoint` or `load_inference_bundle_torch`?)
2. How to convert Lightning `predict()` outputs to NumPy?
3. Where does coordinate transformation happen? (before or after reassembly?)
4. Which PyTorch helper mirrors TensorFlow `reassemble_position`? (extend `ptycho_torch/helper.py`?)

### Phase C.C2: Add Failing Pytest Coverage
**Location:** `tests/torch/test_workflows_components.py`

**Test Cases to Add:**
1. `test_reassemble_cdi_image_torch_normalizes_container` - Assert `_ensure_container` called
2. `test_reassemble_cdi_image_torch_runs_inference` - Mock Lightning module, assert predict called
3. `test_reassemble_cdi_image_torch_applies_coordinate_transforms` - Assert flip_x/flip_y/transpose applied
4. `test_reassemble_cdi_image_torch_returns_amplitude_phase` - Assert tuple structure matches spec §4.5

### Phase C.C3: Implement Stitching Path
**Target Function:** `_reassemble_cdi_image_torch` (lines 308-358)

**Implementation Checklist:**
- [ ] Normalize `test_data` → `PtychoDataContainerTorch` via `_ensure_container`
- [ ] Load trained model (determine approach per C.C1 design)
- [ ] Create test dataloader from container
- [ ] Run `trainer.predict(model, test_dataloader)` or `model.forward_predict()`
- [ ] Convert outputs to NumPy (`.cpu().numpy()`)
- [ ] Apply coordinate transforms (`flip_x`, `flip_y`, `transpose`, `coord_scale`)
- [ ] Implement PyTorch `reassemble_position` equivalent in `ptycho_torch/helper.py`
- [ ] Call reassembly to stitch patches into full image
- [ ] Extract amplitude (`np.abs(recon)`) and phase (`np.angle(recon)`)
- [ ] Build `results_dict` with `obj_tensor_full`, `global_offsets`, `coords_nominal`
- [ ] Return `(recon_amp, recon_phase, results_dict)`

---

## 6. Next Steps (Immediate Loop Actions)

### This Loop (Phase A Baseline Documentation)
- ✅ Catalog stub inventory (Section 1)
- ✅ Reproduce integration failure (Section 2)
- ✅ Confirm findings ledger coverage (Section 3)
- ⏳ Update `docs/fix_plan.md` with baseline attempt entry
- ⏳ Update `plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md` checklist (A1-A3 → `[P]`)
- ⏳ Commit baseline artifacts and plan updates

### Next Loop (Phase B.B1)
- Author failing Lightning regression tests per Section 5 guidance
- Store red test log: `reports/<TS>/phase_d2_completion/pytest_train_red.log`
- Document TDD cycle in parity summary

---

## 7. Open Questions & Hypotheses

### Q1: Config Serialization Strategy
**Question:** Should configs be serialized as hyperparameters (Lightning automatic) or as external metadata (custom)?

**Hypothesis:** Hyperparameters are preferred because:
1. Lightning auto-serializes `__init__` args if `save_hyperparameters()` called
2. Checkpoints become self-contained (no external config files)
3. `load_from_checkpoint` reconstructs module without manual plumbing

**Validation:** Implement in Phase B.B2 and test roundtrip save/load.

### Q2: Probe Initialization Contract
**Question:** How should PyTorch probe initialization mirror TensorFlow's `probe.set_probe_guess()`?

**Hypothesis:** PyTorch Lightning module should:
1. Accept initial probe guess as constructor arg (from `container.probe`)
2. Store as module parameter or buffer (for checkpointing)
3. Optionally make trainable via `probe_trainable` config flag

**Validation:** Review TensorFlow implementation (`ptycho/probe.py`) and design PyTorch equivalent.

### Q3: Stitching Helper Placement
**Question:** Should PyTorch `reassemble_position` live in `ptycho_torch/helper.py` or new `ptycho_torch/stitching.py`?

**Hypothesis:** Extend `ptycho_torch/helper.py` because:
1. Mirrors TensorFlow's `tf_helper.py` organization
2. Keeps reassembly utilities colocated with other tensor ops
3. Simpler import path for workflows module

**Validation:** Prototype in Phase C.C3 and refactor if module becomes too large.

---

## Artifacts

**Generated in This Loop:**
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T233109Z/phase_d2_completion/baseline.md` (this file)
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T233109Z/phase_d2_completion/pytest_integration_baseline.log`

**Referenced Inputs:**
- `ptycho_torch/workflows/components.py` (lines 1-480)
- `plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md` (Phase A-D checklist)
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T231500Z/parity_summary.md` (probe size resolution)
- `docs/findings.md` (POLICY-001, FORMAT-001, CONFIG-001)
- `specs/ptychodus_api_spec.md` §4.5-4.6 (reconstructor contract)
- `docs/workflows/pytorch.md` §§5-7 (Lightning + MLflow knobs)

**Next Artifact (Phase B.B1):**
- `reports/<TS>/phase_d2_completion/pytest_train_red.log` (failing Lightning tests)

---

**Author:** Ralph (Codex Agent)
**Date:** 2025-10-17
**Task:** INTEGRATE-PYTORCH-001-STUBS (Phase A baseline documentation)
**Status:** ✅ BASELINE COMPLETE
