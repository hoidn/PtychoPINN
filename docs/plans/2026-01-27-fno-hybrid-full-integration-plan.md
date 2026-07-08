# FNO/Hybrid Full Pipeline Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable FNO and Hybrid generators to train and infer through the full PyTorch Lightning PINN pipeline (same physics loss/consistency path as CNN) so they can be compared against the existing PINN/CNN baseline.

**Architecture:** Add a generator-adapter path in `ptycho_torch/model.py` so non-CNN generators emit complex local object patches in channel-first format, then reuse the existing `ForwardModel` and Lightning training loop. Thread generator selection via the registry from `_train_with_lightning`, standardize generator `build_model()` signatures to accept a config dict, and pass architecture through the config factory. Reuse the existing TF reassembly helper for stitching; do not change stitching behavior.

**Tech Stack:** PyTorch 2.2+, Lightning, `neuraloperator` (when available), existing `ptycho_torch` modules, TF reassembly helper for stitching parity.

---

## Task 1: Add a complex-patch adapter path and generator plumbing in `ptycho_torch/model.py`

**Files:**
- Modify: `ptycho_torch/model.py`
- Create: `tests/torch/test_generator_adapter.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_generator_adapter.py
import torch


def test_real_imag_to_complex_channel_first():
    from ptycho_torch.model import _real_imag_to_complex_channel_first

    batch = 2
    H = 8
    W = 8
    C = 4
    x = torch.zeros(batch, H, W, C, 2)
    x[..., 0] = 1.0

    out = _real_imag_to_complex_channel_first(x)
    assert out.shape == (batch, C, H, W)
    assert out.is_complex()
    assert torch.allclose(out.real, torch.ones_like(out.real))
    assert torch.allclose(out.imag, torch.zeros_like(out.imag))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_generator_adapter.py::test_real_imag_to_complex_channel_first -v`
Expected: FAIL with `ImportError` or `AttributeError` for `_real_imag_to_complex_channel_first`.

**Step 3: Write minimal implementation**

```python
# ptycho_torch/model.py (near helper functions)

def _real_imag_to_complex_channel_first(real_imag: torch.Tensor) -> torch.Tensor:
    if real_imag.ndim != 5 or real_imag.shape[-1] != 2:
        raise ValueError(
            f"Expected real/imag tensor with shape (B, H, W, C, 2), got {tuple(real_imag.shape)}"
        )
    complex_last = torch.complex(real_imag[..., 0], real_imag[..., 1])  # (B, H, W, C)
    return complex_last.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
```

```python
# ptycho_torch/model.py (inside PtychoPINN.__init__ and forward)
class PtychoPINN(nn.Module):
    def __init__(
        self,
        model_config,
        data_config,
        training_config,
        generator_module=None,
        generator_output="amp_phase",
    ):
        ...
        self.generator_output = generator_output
        # Preserve existing attribute name to minimize refactor surface.
        self.autoencoder = Autoencoder(model_config, data_config) if generator_module is None else generator_module
        self.combine_complex = CombineComplex()
        self.forward_model = ForwardModel(model_config, data_config)

    def _predict_complex(self, x):
        if self.generator_output == "amp_phase":
            amp, phase = self.autoencoder(x)
            x_complex = self.combine_complex(amp, phase)
        elif self.generator_output == "real_imag":
            patches = self.autoencoder(x)
            x_complex = _real_imag_to_complex_channel_first(patches)
            amp = torch.abs(x_complex)
            phase = torch.angle(x_complex)
            return x_complex, amp, phase
        else:
            raise ValueError(f"Unsupported generator_output='{self.generator_output}'")
        return x_complex, amp, phase

    def forward(self, x, positions, probe, input_scale_factor, output_scale_factor, experiment_ids=None):
        x = self.scaler.scale(x, input_scale_factor)
        x_complex, amp, phase = self._predict_complex(x)
        x_out = self.forward_model.forward(
            x_complex, positions, probe / self.probe_scale, output_scale_factor, experiment_ids
        )
        return x_out, amp, phase

    def forward_predict(self, x, positions, probe, input_scale_factor):
        x = self.scaler.scale(x, input_scale_factor)
        x_complex, amp, phase = self._predict_complex(x)
        return x_complex
```

```python
# ptycho_torch/model.py (PtychoPINN_Lightning __init__ signature and wiring)
class PtychoPINN_Lightning(L.LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        training_config: TrainingConfig,
        inference_config: InferenceConfig,
        generator_module=None,
        generator_output="amp_phase",
    ):
        ...
        if model_config.mode == 'Unsupervised':
            self.model = PtychoPINN(
                model_config,
                data_config,
                training_config,
                generator_module=generator_module,
                generator_output=generator_output,
            )
        elif model_config.mode == 'Supervised':
            self.model = Ptycho_Supervised(model_config, data_config, training_config)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_generator_adapter.py::test_real_imag_to_complex_channel_first -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ptycho_torch/model.py tests/torch/test_generator_adapter.py
git commit -m "feat: add real/imag adapter for generator patches"
```

---

## Task 2: Wire generator registry into Lightning training (FNO/Hybrid build path)

**Files:**
- Modify: `ptycho_torch/workflows/components.py`
- Modify: `ptycho_torch/generators/fno.py`
- Modify: `ptycho_torch/generators/cnn.py`
- Modify: `tests/torch/test_fno_lightning_integration.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_fno_lightning_integration.py
import pytest


@pytest.mark.slow
@pytest.mark.parametrize("arch", ["fno", "hybrid"])
def test_train_history_collects_epochs_for_fno_hybrid(synthetic_ptycho_npz, tmp_path, arch):
    from ptycho.config.config import TrainingConfig, ModelConfig, PyTorchExecutionConfig
    from ptycho.raw_data import RawData
    from ptycho_torch.workflows.components import train_cdi_model_torch

    train_npz, _ = synthetic_ptycho_npz
    train_data = RawData.from_file(str(train_npz))

    cfg = TrainingConfig(
        model=ModelConfig(N=64, gridsize=1, architecture=arch),
        train_data_file=train_npz,
        test_data_file=None,
        nepochs=1,
        batch_size=2,
        backend="pytorch",
        output_dir=tmp_path,
        n_groups=4,
    )

    exec_cfg = PyTorchExecutionConfig(
        logger_backend=None,
        enable_checkpointing=False,
        strategy='auto',
    )

    results = train_cdi_model_torch(train_data, None, cfg, execution_config=exec_cfg)
    history = results["history"]["train_loss"]
    assert len(history) >= 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_fno_lightning_integration.py::test_train_history_collects_epochs_for_fno_hybrid -v`
Expected: FAIL (FNO/Hybrid not wired into Lightning)

**Step 3: Write minimal implementation**

```python
# ptycho_torch/workflows/components.py (_train_with_lightning)
from ptycho_torch.generators.registry import resolve_generator
...
factory_overrides = {
    ...
    'architecture': config.model.architecture,
}
...
pt_inference_config = PTInferenceConfig()

pt_configs = {
    "model_config": pt_model_config,
    "data_config": pt_data_config,
    "training_config": pt_training_config,
    "inference_config": pt_inference_config,
}

# Build Lightning module via registry (all generators use dict signature)
generator = resolve_generator(config)
model = generator.build_model(pt_configs)
```

```python
# ptycho_torch/generators/fno.py (standardize build_model signature to dict + wrap Lightning)
class HybridGenerator:
    ...
    def build_model(self, pt_configs):
        from ptycho_torch.model import PtychoPINN_Lightning
        data_config = pt_configs["data_config"]
        model_config = pt_configs["model_config"]
        training_config = pt_configs["training_config"]
        inference_config = pt_configs["inference_config"]
        core = HybridUNOGenerator(..., C=data_config.C)
        return PtychoPINN_Lightning(
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            inference_config=inference_config,
            generator_module=core,
            generator_output="real_imag",
        )

class FnoGenerator:
    ...
    def build_model(self, pt_configs):
        from ptycho_torch.model import PtychoPINN_Lightning
        data_config = pt_configs["data_config"]
        model_config = pt_configs["model_config"]
        training_config = pt_configs["training_config"]
        inference_config = pt_configs["inference_config"]
        core = CascadedFNOGenerator(..., C=data_config.C)
        return PtychoPINN_Lightning(
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            inference_config=inference_config,
            generator_module=core,
            generator_output="real_imag",
        )
```

```python
# ptycho_torch/generators/cnn.py (keep dict signature)
class CnnGenerator:
    ...
    def build_model(self, pt_configs):
        from ptycho_torch.model import PtychoPINN_Lightning
        return PtychoPINN_Lightning(**pt_configs)
```

```python
# ptycho_torch/model.py (PtychoPINN_Lightning __init__ signature)
class PtychoPINN_Lightning(L.LightningModule):
    def __init__(..., generator_module=None, generator_output="amp_phase"):
        ...
        if model_config.mode == 'Unsupervised':
            self.model = PtychoPINN(
                model_config,
                data_config,
                training_config,
                generator_module=generator_module,
                generator_output=generator_output,
            )
        elif model_config.mode == 'Supervised':
            self.model = Ptycho_Supervised(...)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_fno_lightning_integration.py::test_train_history_collects_epochs_for_fno_hybrid -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ptycho_torch/workflows/components.py ptycho_torch/generators/fno.py ptycho_torch/generators/cnn.py ptycho_torch/model.py tests/torch/test_fno_lightning_integration.py
git commit -m "feat: route Lightning training through generator registry"
```

---

## Task 3: Align inference/stitching path with generator outputs

**Files:**
- Modify: `ptycho_torch/workflows/components.py`
- Modify: `tests/torch/test_fno_lightning_integration.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_fno_lightning_integration.py
import numpy as np
import pytest


@pytest.mark.slow
def test_reassemble_cdi_image_torch_handles_real_imag_outputs(synthetic_ptycho_npz, tmp_path):
    from ptycho.config.config import TrainingConfig, ModelConfig, PyTorchExecutionConfig
    from ptycho.raw_data import RawData
    from ptycho_torch.workflows.components import train_cdi_model_torch, _reassemble_cdi_image_torch

    train_npz, test_npz = synthetic_ptycho_npz
    train_data = RawData.from_file(str(train_npz))
    test_data = RawData.from_file(str(test_npz))

    cfg = TrainingConfig(
        model=ModelConfig(N=64, gridsize=1, architecture="fno"),
        train_data_file=train_npz,
        test_data_file=test_npz,
        nepochs=1,
        batch_size=2,
        backend="pytorch",
        output_dir=tmp_path,
        n_groups=4,
    )

    exec_cfg = PyTorchExecutionConfig(
        logger_backend=None,
        enable_checkpointing=False,
        strategy='auto',
    )

    results = train_cdi_model_torch(train_data, test_data, cfg, execution_config=exec_cfg)
    amp, phase, _ = _reassemble_cdi_image_torch(test_data, cfg, M=64, train_results=results)

    assert amp.shape == phase.shape
    assert np.isfinite(amp).all()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_fno_lightning_integration.py::test_reassemble_cdi_image_torch_handles_real_imag_outputs -v`
Expected: FAIL (inference path does not handle real/imag outputs and Lightning forward signature).

**Step 3: Write minimal implementation**

```python
# ptycho_torch/workflows/components.py
# Move PtychoLightningDataset to module scope (private helper) so inference can reuse it
# Then in _build_inference_dataloader:
infer_dataset = PtychoLightningDataset(container)
return DataLoader(
    infer_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=execution_config.num_workers,
    pin_memory=execution_config.pin_memory,
)
```

```python
# ptycho_torch/workflows/components.py (_reassemble_cdi_image_torch)
for batch in infer_loader:
    batch_dict, probe, _ = batch
    X_batch = batch_dict['images'].to(torch.float32)
    positions = batch_dict['coords_relative']
    rms_scale = batch_dict.get('rms_scaling_constant', torch.ones(1, 1, 1))
    pred = lightning_module.forward_predict(X_batch, positions, probe, rms_scale)
    obj_patches.append(pred.cpu())
```

```python
# ptycho_torch/workflows/components.py (before TF reassembly)
from ptycho_torch.model import _real_imag_to_complex_channel_first
if obj_tensor_full.ndim == 5 and obj_tensor_full.shape[-1] == 2:
    obj_tensor_full = _real_imag_to_complex_channel_first(obj_tensor_full)

# Convert channel-first complex to channel-last complex for TF helper
if obj_tensor_full.ndim == 4:
    obj_tensor_full = obj_tensor_full.permute(0, 2, 3, 1).contiguous()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_fno_lightning_integration.py::test_reassemble_cdi_image_torch_handles_real_imag_outputs -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ptycho_torch/workflows/components.py tests/torch/test_fno_lightning_integration.py
git commit -m "feat: support FNO/Hybrid outputs in torch reassembly"
```

---

## Task 4: Documentation alignment

**Files:**
- Modify: `docs/backlog/FNO_HYBRID_FULL_INTEGRATION.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `ptycho_torch/generators/README.md`

**Step 1: Update backlog status and link this plan**

Add a “Plan” line in the backlog item referencing `docs/plans/2026-01-27-fno-hybrid-full-integration-plan.md`, keep Status: Open until tests pass.

**Step 2: Update PyTorch workflow docs**

Add a short note in `docs/workflows/pytorch.md` that `config.model.architecture` now routes through the generator registry and that FNO/Hybrid train via Lightning with physics loss (no stitching changes).

**Step 3: Update generator README**

Replace “Reserved / not yet implemented” language for `fno` and `hybrid` with the new Lightning integration contract.

**Step 4: Commit**

```bash
git add docs/backlog/FNO_HYBRID_FULL_INTEGRATION.md docs/workflows/pytorch.md ptycho_torch/generators/README.md
git commit -m "docs: document FNO/Hybrid Lightning integration"
```

---

## Notes & Constraints

- Do **not** modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` (stable core).
- Honor CONFIG-001: ensure `update_legacy_dict(params.cfg, config)` is invoked before data loading in any workflow you touch.
- Keep stitching behavior unchanged (no weighted averaging changes).
