# Lightning Checkpoint Inspection Report

**Date:** 2025-10-19
**Task:** Phase D1b - Inspect Lightning checkpoint payload to determine root cause of TypeError during checkpoint loading
**Checkpoint:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T123000Z/phase_d2_completion/checkpoint_run/checkpoints/last.ckpt`

---

## Training Command Used

```bash
python -m ptycho_torch.train \
  --train_data_file datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --test_data_file datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --output_dir plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T123000Z/phase_d2_completion/checkpoint_run \
  --max_epochs 2 \
  --n_images 64 \
  --gridsize 1 \
  --batch_size 4 \
  --device cpu \
  --disable_mlflow
```

**Result:** Training completed successfully (2 epochs), checkpoint created at expected path.

---

## Checkpoint Payload Inspection

### Checkpoint Keys Present

```python
['callbacks', 'epoch', 'global_step', 'loops', 'lr_schedulers',
 'optimizer_states', 'pytorch-lightning_version', 'state_dict']
```

### Critical Finding: Missing `hyper_parameters` Key

**Observation:** The `hyper_parameters` key is **ABSENT** from the checkpoint. When queried via `checkpoint.get('hyper_parameters')`, it returns `None`.

**Implication:** This confirms the root cause of the TypeError observed in the integration test failure:

```
TypeError: PtychoPINN_Lightning.__init__() missing 4 required positional
arguments: 'model_config', 'data_config', 'training_config', and 'inference_config'
```

When Lightning attempts to reload the checkpoint without hyperparameters, it cannot reconstruct the `PtychoPINN_Lightning` module because there are no saved config objects to pass to the constructor.

---

## Root Cause Analysis

### Expected Behavior (Phase D2.B2 Implementation)

Per Attempt #10 documentation (`reports/2025-10-18T014317Z/phase_d2_completion/summary.md`), the `_train_with_lightning` orchestrator was implemented with:

```python
# ptycho_torch/workflows/components.py:485-491
lightning_module = PtychoPINN_Lightning(
    model_config=pt_model_config,
    data_config=pt_data_config,
    training_config=pt_training_config,
    inference_config=pt_inference_config,
)
```

**Phase B.B2 checklist item B2.4** required calling `model.save_hyperparameters()` to embed the four config objects in the checkpoint for state-free reload.

### Actual Behavior

The checkpoint payload inspection reveals that **hyperparameters were NOT saved**, indicating one of the following:

1. `save_hyperparameters()` was **not called** in the `PtychoPINN_Lightning.__init__()` method
2. `save_hyperparameters()` was called but with incorrect arguments or conditions
3. `save_hyperparameters()` was called but the configs are not serializable via Lightning's mechanism

---

## Remediation Path

### Hypothesis 1: Missing `save_hyperparameters()` Call

**Next Action:** Inspect `ptycho_torch/model.py` `PtychoPINN_Lightning.__init__()` to verify whether `self.save_hyperparameters()` is present and correctly configured.

**Expected Code Pattern:**

```python
class PtychoPINN_Lightning(pl.LightningModule):
    def __init__(self, model_config, data_config, training_config, inference_config):
        super().__init__()
        # CRITICAL: This line must be present and execute successfully
        self.save_hyperparameters()
        # ... rest of initialization
```

**Diagnostic Command:**

```bash
grep -n "save_hyperparameters" ptycho_torch/model.py
```

### Hypothesis 2: Non-Serializable Config Objects

If `save_hyperparameters()` is present but failing silently, the PyTorch dataclass config objects may contain non-serializable attributes (e.g., `Path` objects, enums, or nested dataclasses that Lightning cannot pickle).

**Next Action:** Add explicit logging around `save_hyperparameters()` to capture exceptions, or convert configs to plain dicts before saving.

**Remediation Strategy:**

```python
# Option A: Convert to dict before saving
self.save_hyperparameters({
    'model_config': asdict(model_config),
    'data_config': asdict(data_config),
    'training_config': asdict(training_config),
    'inference_config': asdict(inference_config),
})

# Option B: Exclude non-serializable fields
self.save_hyperparameters(ignore=['problematic_field'])
```

### Hypothesis 3: Incorrect Calling Context

If `save_hyperparameters()` is called **after** other initialization steps that modify `self`, Lightning may not capture the constructor arguments correctly.

**Next Action:** Ensure `self.save_hyperparameters()` is the **first line** after `super().__init__()`.

---

## References

- **Phase D2.B2 Implementation:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T014317Z/phase_d2_completion/summary.md`
- **Integration Test Failure:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T095900Z/phase_d2_completion/diagnostics.md`
- **Lightning Checkpoint Docs:** https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#save-hyperparameters
- **PyTorch Workflow Guide:** `docs/workflows/pytorch.md` §6 (Checkpoint Management)

---

## Artifacts Generated

- **Checkpoint Dump:** `checkpoint_dump.txt` (4 lines, checkpoint keys + hyper_parameters inspection)
- **Checkpoint File:** `checkpoint_run/checkpoints/last.ckpt` (~45MB, will be deleted after inspection per artifact hygiene)
- **This Report:** `checkpoint_inspection.md`

---

## Next Steps

1. Inspect `ptycho_torch/model.py` for presence/correctness of `save_hyperparameters()` call
2. If missing: Add `self.save_hyperparameters()` as first line in `__init__()` after `super()`
3. If present but failing: Add logging/debugging to capture serialization errors
4. Re-run training with fixed implementation to verify checkpoint now contains `hyper_parameters` key
5. Update docs/fix_plan.md Attempt #32 with findings and remediation plan
6. Delete large checkpoint binary (`rm -rf checkpoint_run/checkpoints/`) to maintain artifact hygiene
