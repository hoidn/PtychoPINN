# Torch Physics Scale Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align PyTorch physics scaling with TF semantics by deriving a dataset-level `intensity_scale` from normalized training amplitudes and using it only at the physics loss boundary and inference output mapping.

**Architecture:** Compute a single physics scale after sampling/normalization, persist it in the bundle/hparams, apply it to Poisson loss and output scaling, and honor metadata-first `nphotons` resolution. Keep RMS/statistical normalization unchanged.

**Tech Stack:** Python, PyTorch, Lightning, existing PtychoPINN configs and data pipeline.

---

### Task 1: Add a physics scale derivation helper + unit test

**Files:**
- Create: `tests/torch/test_physics_scale.py`
- Modify: `ptycho_torch/helper.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_physics_scale.py
import torch
from ptycho_torch import helper as hh


def test_derive_intensity_scale_from_amplitudes():
    # Two samples, 1 channel, 2x2
    x = torch.tensor([
        [[[1.0, 1.0], [1.0, 1.0]]],  # sum(x**2)=4
        [[[2.0, 0.0], [0.0, 0.0]]],  # sum(x**2)=4
    ])
    # mean(sum(x**2)) = 4
    nphotons = 100.0
    scale = hh.derive_intensity_scale_from_amplitudes(x, nphotons)
    assert torch.is_tensor(scale)
    assert torch.isclose(scale, torch.tensor(5.0), atol=1e-6)  # sqrt(100/4)=5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_physics_scale.py::test_derive_intensity_scale_from_amplitudes -v`
Expected: FAIL (helper missing)

**Step 3: Write minimal implementation**

```python
# ptycho_torch/helper.py
import torch

def derive_intensity_scale_from_amplitudes(x_norm: torch.Tensor, nphotons: float) -> torch.Tensor:
    """
    Derive dataset-level physics scale from normalized amplitudes.

    intensity_scale = sqrt(nphotons / mean(sum(x_norm**2)))
    """
    if not isinstance(x_norm, torch.Tensor):
        x_norm = torch.as_tensor(x_norm)
    if x_norm.ndim < 2:
        raise ValueError("x_norm must have at least 2 dims")
    if not isinstance(nphotons, (int, float)) or nphotons <= 0:
        raise ValueError("nphotons must be positive")

    # Sum over spatial dims (last two), then mean over remaining dims
    spatial = tuple(range(x_norm.ndim - 2, x_norm.ndim))
    mean_intensity = torch.mean(torch.sum(x_norm ** 2, dim=spatial))
    if mean_intensity.item() <= 0:
        raise ValueError("mean intensity must be positive")
    return torch.sqrt(torch.tensor(float(nphotons), dtype=mean_intensity.dtype) / mean_intensity)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_physics_scale.py::test_derive_intensity_scale_from_amplitudes -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/torch/test_physics_scale.py ptycho_torch/helper.py
git commit -m "test: add physics scale derivation helper"
```

---

### Task 2: Resolve nphotons with metadata precedence + unit test

**Files:**
- Create: `tests/torch/test_nphotons_resolution.py`
- Modify: `ptycho_torch/workflows/components.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_nphotons_resolution.py
from types import SimpleNamespace
from ptycho_torch.workflows import components


def test_resolve_nphotons_metadata_precedence():
    config = SimpleNamespace(nphotons=1e6)
    data = SimpleNamespace(metadata={"physics_parameters": {"nphotons": 1e9}})
    nphotons, source = components._resolve_nphotons(data, config)
    assert nphotons == 1e9
    assert source == "metadata"


def test_resolve_nphotons_fallback_to_config():
    config = SimpleNamespace(nphotons=1e6)
    data = SimpleNamespace(metadata=None)
    nphotons, source = components._resolve_nphotons(data, config)
    assert nphotons == 1e6
    assert source == "config"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_nphotons_resolution.py::test_resolve_nphotons_metadata_precedence -v`
Expected: FAIL (helper missing)

**Step 3: Write minimal implementation**

```python
# ptycho_torch/workflows/components.py
from ptycho.metadata import MetadataManager


def _resolve_nphotons(data, config):
    metadata = getattr(data, "metadata", None)
    if metadata is not None:
        return MetadataManager.get_nphotons(metadata), "metadata"
    return getattr(config, "nphotons", 1e9), "config"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_nphotons_resolution.py::test_resolve_nphotons_metadata_precedence -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/torch/test_nphotons_resolution.py ptycho_torch/workflows/components.py
git commit -m "test: add nphotons resolution helper"
```

---

### Task 3: Compute and attach derived physics scale in the training data path

**Files:**
- Modify: `ptycho_torch/workflows/components.py`
- Modify: `ptycho_torch/data_container_bridge.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_physics_scale_container.py
import torch
from types import SimpleNamespace
from ptycho_torch.data_container_bridge import PtychoDataContainerTorch
from ptycho_torch.workflows import components


def test_container_gets_physics_scale():
    grouped = {
        "X_full": (torch.ones(2, 4, 4, 1) * 2).numpy(),
        "coords_relative": torch.zeros(2, 1, 2, 1).numpy(),
        "coords_offsets": torch.zeros(2, 1, 2, 1).numpy(),
        "nn_indices": torch.zeros(2, 1, dtype=torch.int32).numpy(),
        "Y": None,
    }
    probe = (torch.ones(4, 4, dtype=torch.complex64)).numpy()
    container = PtychoDataContainerTorch(grouped, probe)

    config = SimpleNamespace(nphotons=100.0)
    components._attach_physics_scale(container, config, nphotons_source="config")
    assert hasattr(container, "physics_scaling_constant")
    assert torch.is_tensor(container.physics_scaling_constant)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_physics_scale_container.py::test_container_gets_physics_scale -v`
Expected: FAIL (helper missing)

**Step 3: Write minimal implementation**

```python
# ptycho_torch/workflows/components.py
import torch
from ptycho_torch import helper as hh


def _attach_physics_scale(container, config, nphotons_source):
    nphotons, source = _resolve_nphotons(container, config)
    if nphotons_source is not None:
        source = nphotons_source
    scale = hh.derive_intensity_scale_from_amplitudes(container.X, nphotons)
    container.physics_scaling_constant = scale.view(1, 1, 1)
    container.nphotons_source = source
    container.nphotons_resolved = nphotons
    return scale, source
```

Also update `_ensure_container` to call `_attach_physics_scale` after container creation, and copy metadata if input was RawData:

```python
# ptycho_torch/workflows/components.py (inside _ensure_container)
if isinstance(data, RawData):
    metadata = getattr(data, "metadata", None)
    ...
    container = PtychoDataContainerTorch(grouped_data, probe)
    if metadata is not None:
        container.metadata = metadata
    _attach_physics_scale(container, config, nphotons_source=None)
    return container
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_physics_scale_container.py::test_container_gets_physics_scale -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/torch/test_physics_scale_container.py ptycho_torch/workflows/components.py ptycho_torch/data_container_bridge.py
git commit -m "feat: attach derived physics scale to torch containers"
```

---

### Task 4: Use derived physics scale at the Poisson loss boundary

**Files:**
- Modify: `ptycho_torch/model.py`
- Modify: `tests/torch/test_physics_scale_loss.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_physics_scale_loss.py
import torch
from types import SimpleNamespace
from ptycho_torch.model import PtychoPINN_Lightning
from ptycho_torch.config_params import ModelConfig, DataConfig, TrainingConfig, InferenceConfig


def test_poisson_loss_uses_physics_scale():
    model_cfg = ModelConfig(loss_function="Poisson")
    data_cfg = DataConfig()
    train_cfg = TrainingConfig()
    infer_cfg = InferenceConfig()
    model = PtychoPINN_Lightning(model_cfg, data_cfg, train_cfg, infer_cfg)

    # Dummy batch: input amplitudes all ones
    x = torch.ones(1, 1, 2, 2)
    batch = (
        {
            "images": x,
            "coords_relative": torch.zeros(1, 1, 1, 2),
            "rms_scaling_constant": torch.ones(1, 1, 1, 1),
            "physics_scaling_constant": torch.full((1, 1, 1, 1), 10.0),
            "experiment_id": torch.zeros(1, dtype=torch.long),
        },
        torch.ones(1, 1, 1, 2, 2, dtype=torch.complex64),
        torch.ones(1),
    )

    loss = model.compute_loss(batch)
    assert torch.isfinite(loss)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_physics_scale_loss.py::test_poisson_loss_uses_physics_scale -v`
Expected: FAIL until loss path uses physics scale

**Step 3: Write minimal implementation**

```python
# ptycho_torch/model.py (inside compute_loss)
physics_scale = batch[0]["physics_scaling_constant"]

pred, amp, phase = self(
    x, positions, probe,
    input_scale_factor=rms_scale,
    output_scale_factor=rms_scale,
    experiment_ids=experiment_ids,
)

if self.model_config.mode == "Unsupervised":
    pred_physics = pred * physics_scale
    obs_physics = x * physics_scale
    total_loss += self.Loss(pred_physics, obs_physics).mean()
    total_loss /= intensity_norm_factor
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_physics_scale_loss.py::test_poisson_loss_uses_physics_scale -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/torch/test_physics_scale_loss.py ptycho_torch/model.py
git commit -m "feat: apply derived physics scale at poisson loss"
```

---

### Task 5: Persist and restore intensity_scale in bundles and inference

**Files:**
- Modify: `ptycho_torch/model_manager.py`
- Modify: `ptycho_torch/inference.py`
- Modify: `ptycho_torch/workflows/components.py`
- Modify: `tests/torch/test_physics_scale_bundle.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_physics_scale_bundle.py
from types import SimpleNamespace
from ptycho_torch.model_manager import save_torch_bundle


def test_bundle_persists_intensity_scale(tmp_path):
    # Minimal stub model dict
    models = {"autoencoder": {"_sentinel": True}, "diffraction_to_obj": {"_sentinel": True}}
    config = SimpleNamespace(model=SimpleNamespace(N=64, gridsize=1), nphotons=1e9)
    base_path = tmp_path / "wts.h5"
    save_torch_bundle(models, str(base_path), config, intensity_scale=123.0)
    assert (tmp_path / "wts.h5.zip").exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_physics_scale_bundle.py::test_bundle_persists_intensity_scale -v`
Expected: FAIL until persist path includes scale

**Step 3: Write minimal implementation**

- Ensure `train_cdi_model_torch` passes derived `intensity_scale` to `save_torch_bundle`.
- Ensure `create_torch_model_with_gridsize` reads `intensity_scale` from `params_dict` and sets `ModelConfig.intensity_scale`.
- Ensure `inference.py` uses stored scale from bundle instead of deriving from inference data.

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_physics_scale_bundle.py::test_bundle_persists_intensity_scale -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/torch/test_physics_scale_bundle.py ptycho_torch/model_manager.py ptycho_torch/inference.py ptycho_torch/workflows/components.py
git commit -m "feat: persist and reuse derived intensity scale"
```

---

### Task 6: Update docs and run focused tests

**Files:**
- Modify: `docs/DATA_NORMALIZATION_GUIDE.md`

**Step 1: Update doc**

Add a short PyTorch note describing that derived physics scale is computed from normalized amplitudes and persisted in bundles (no metadata write-back).

**Step 2: Run focused tests**

Run:
- `pytest tests/torch/test_physics_scale.py -v`
- `pytest tests/torch/test_nphotons_resolution.py -v`
- `pytest tests/torch/test_physics_scale_container.py -v`
- `pytest tests/torch/test_physics_scale_loss.py -v`
- `pytest tests/torch/test_physics_scale_bundle.py -v`

**Step 3: Commit**

```bash
git add docs/DATA_NORMALIZATION_GUIDE.md
git commit -m "docs: note torch physics scale derivation and persistence"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-02-05-torch-physics-scale-implementation-plan.md`.

Two execution options:
1. Subagent-Driven (this session)
2. Parallel Session (separate)

Which approach?
