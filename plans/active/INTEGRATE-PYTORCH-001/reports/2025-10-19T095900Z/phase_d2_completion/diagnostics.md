# Phase D2.D1 Integration Test Diagnostics

**Test Run:** 2025-10-19T02:51:55 (UTC)
**Test Selector:** `pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv`
**Outcome:** FAILED (expected at Phase D2.D1)

---

## Executive Summary

Training phase succeeded; checkpoint created at `<output_dir>/checkpoints/last.ckpt`. Inference phase failed when attempting to load the Lightning checkpoint.

**Root Cause:** `PtychoPINN_Lightning.load_from_checkpoint()` failed with `TypeError` due to missing 4 required positional arguments: `model_config`, `data_config`, `training_config`, and `inference_config`.

**Impact:** Checkpoint persistence contract per `specs/ptychodus_api_spec.md` §4.6 violated. Lightning checkpoints must be loadable without extra constructor arguments, requiring hyperparameters to be serialized within the checkpoint payload.

---

## Failure Signature

### Error Stack Trace (Excerpt)

```
File "/home/ollie/miniconda3/envs/ptycho311/lib/python3.11/site-packages/lightning/pytorch/core/saving.py", line 165, in _load_state
  obj = instantiator(cls, _cls_kwargs) if instantiator else cls(**_cls_kwargs)
                                                            ^^^^^^^^^^^^^^^^^^
TypeError: PtychoPINN_Lightning.__init__() missing 4 required positional arguments: 'model_config', 'data_config', 'training_config', and 'inference_config'
```

### Checkpoint Location

Training created checkpoint at expected path:
```
training_output_dir / "checkpoints" / "last.ckpt"
```

Checkpoint file exists and is accessible by inference script, confirming persistence layer is operational. Issue is loading mechanism, not artifact creation.

---

## Comparison vs. 2025-10-17 Baseline

### Baseline Run (2025-10-17T233109Z)

From `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T233109Z/phase_d2_completion/pytest_integration_baseline.log`:
- **Same error:** `TypeError: PtychoPINN_Lightning.__init__() missing 4 required positional arguments`
- **Same checkpoint path:** `checkpoints/last.ckpt`
- **Same failure point:** Inference script line 442 (`PtychoPINN_Lightning.load_from_checkpoint`)

### Current Run (2025-10-19T095900Z)

**No behavioral change** — failure signature is identical. This is expected because:
1. Phase B.B2 (Attempt #10) implemented Lightning training orchestration with `save_hyperparameters()` call
2. Phase C.C3/C4 (Attempts #25, #28) implemented stitching path
3. **Neither phase addressed checkpoint loading** — that work is deferred to Phase D remediation

---

## Differences from Baseline

### Files Changed Since 2025-10-17T233109Z Baseline

1. **`ptycho_torch/workflows/components.py`:**
   - Lines 265-529: `_build_lightning_dataloaders()` + `_train_with_lightning()` implementation (Phase B)
   - Lines 375-447: `_build_inference_dataloader()` helper (Phase C)
   - Lines 607-743: `_reassemble_cdi_image_torch()` implementation (Phase C)

2. **`ptycho_torch/raw_data_bridge.py`:**
   - Line 187, 235: `dataset_path` kwarg forwarding to TensorFlow RawData (Phase C)

3. **`tests/torch/test_workflows_components.py`:**
   - Lines 713-1059: `TestTrainWithLightningRed` class (Phase B)
   - Lines 1076-1507: `TestReassembleCdiImageTorchRed` class (Phase C)

**Checkpoint creation code unchanged** from baseline — `PtychoPINN_Lightning.__init__` and `save_hyperparameters()` logic landed in earlier phases and remain unmodified since Attempt #10 (2025-10-18T014317Z).

### Test Execution Context

- **Training CLI:** Used `ptycho_torch.train` module (same as baseline)
- **Inference CLI:** Used `ptycho_torch.inference` module (same as baseline)
- **Dataset:** `datasets/Run1084_recon3_postPC_shrunk_3.npz` (canonical test fixture)
- **Device:** CPU-only execution (no CUDA required per POLICY-001)
- **Environment:** PyTorch 2.8.0+cu128, Lightning installed, `pip install -e .[torch]` confirmed

---

## Next Hypotheses for Remediation

### Hypothesis 1: Missing `save_hyperparameters()` Payload

**Theory:** `PtychoPINN_Lightning.__init__` called `self.save_hyperparameters()` during training (confirmed in `ptycho_torch/model.py`), but the checkpoint may not contain the expected `hyper_parameters` key or the payload is incomplete.

**Verification Steps:**
1. Inspect checkpoint with `torch.load(checkpoint_path)` and examine `checkpoint['hyper_parameters']`
2. Verify all four config objects (`model_config`, `data_config`, `training_config`, `inference_config`) are serialized
3. Check if Lightning's default hyperparameter serialization supports dataclass instances

**Remediation Options:**
- Add explicit hyperparameter dict construction in `__init__` before `save_hyperparameters()`
- Use `self.save_hyperparameters(ignore=['...'])` if some args should not be serialized
- Implement custom `on_save_checkpoint` hook to manually inject config metadata

### Hypothesis 2: Load Path Missing Hyperparameter Restoration

**Theory:** Inference script calls `PtychoPINN_Lightning.load_from_checkpoint(path)` without passing any constructor kwargs, relying solely on deserialized hyperparameters. If Lightning doesn't auto-populate `__init__` kwargs from checkpoint metadata, the load fails.

**Verification Steps:**
1. Review `ptycho_torch/inference.py` line 442 call signature
2. Test manual load: `model = PtychoPINN_Lightning.load_from_checkpoint(path, model_config=..., ...)` with configs extracted from somewhere (e.g., params.cfg snapshot)
3. Check Lightning documentation for hyperparameter restoration contract

**Remediation Options:**
- Store config metadata separately (e.g., `config.yaml` alongside checkpoint) and load via helper
- Modify inference script to reconstruct config objects before calling `load_from_checkpoint`
- Use Lightning's `LightningModule.load_from_checkpoint(path, hparams_file=...)` pattern

### Hypothesis 3: Dataclass Serialization Compatibility

**Theory:** PyTorch config dataclasses (`PTDataConfig`, `PTModelConfig`, etc.) may not be directly serializable by Lightning's hyperparameter system, causing incomplete checkpoint metadata.

**Verification Steps:**
1. Check if `save_hyperparameters()` correctly handles dataclass instances (may require `asdict()` conversion)
2. Inspect checkpoint `hyper_parameters` for dataclass representation (dict vs. object)
3. Test minimal reproduction: create Lightning module with dataclass args, save, reload

**Remediation Options:**
- Convert dataclasses to dicts before `save_hyperparameters(dataclasses.asdict(model_config), ...)`
- Implement custom `hparams` property returning serializable dict
- Use `@dataclass(frozen=True)` or `attr.s` if Lightning requires specific class structure

---

## Artifact Inventory

**Captured Logs:**
- `pytest_integration_current.log` (17KB, full pytest output with stack trace)
- `train_debug.log` (80KB, training subprocess debug output)

**Checkpoint Artifact:**
- Location: `<tempdir>/training_outputs/checkpoints/last.ckpt` (created by training subprocess, cleaned up by test teardown)
- **Preservation Note:** Checkpoint is ephemeral (temporary test directory). To preserve for analysis, modify test to copy artifact to `plans/active/.../reports/2025-10-19T095900Z/phase_d2_completion/sample_checkpoint.ckpt`

**Related Evidence:**
- Phase B.B2 implementation: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T014317Z/phase_d2_completion/summary.md`
- Baseline failure: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T233109Z/phase_d2_completion/pytest_integration_baseline.log`

---

## Recommended Next Actions

1. **Inspect Checkpoint Payload (immediate):**
   - Modify integration test to preserve `last.ckpt` artifact to artifact directory
   - Run `torch.load()` to examine `hyper_parameters` key contents
   - Document findings in new `checkpoint_inspection.md` under same report directory

2. **Review Lightning Documentation (research):**
   - Verify `save_hyperparameters()` contract for dataclass arguments
   - Identify recommended patterns for complex config objects
   - Document findings in `lightning_hyperparams_research.md`

3. **Implement Remediation (TDD loop):**
   - Author failing test: Load checkpoint and verify all configs restored
   - Implement fix based on Hypothesis 1, 2, or 3
   - Capture green run showing checkpoint load success
   - Update `phase_d2_completion.md` D1 checklist to `[x]`

4. **Update Documentation:**
   - Refresh `docs/workflows/pytorch.md` §10 (Checkpoint Management) with loading guidance
   - Add troubleshooting entry for "missing positional arguments" error
   - Cross-link to this diagnostics document for future reference

---

**Status:** Phase D2.D1 diagnostics COMPLETE. Integration test failure signature documented. Remediation hypotheses authored. Awaiting Hypothesis 1 verification loop (checkpoint payload inspection).
