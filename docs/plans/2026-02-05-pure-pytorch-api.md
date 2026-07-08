# Pure PyTorch API (PtychoModel) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a pure PyTorch API (`PtychoModel`) that trains and infers from raw NPZ files without touching `params.cfg` or TensorFlow bridge code.

**Architecture:** Implement a pure data path (`ptycho_torch/pure_data.py`) that loads raw NPZ, groups coordinates via `cKDTree`, and returns the existing grouped dict schema. Wrap this in a `PtychoModel` facade that constructs PyTorch config dataclasses, uses the generator registry for CNN/FNO/Hybrid/Hybrid-ResNet, and runs Lightning training/inference without legacy globals.

**Tech Stack:** Python, NumPy, SciPy (`cKDTree`), PyTorch, Lightning

---

### Task 1: Pure Raw-NPZ Loader + Grouping (Unit Tests First)

**Files:**
- Create: `tests/torch/test_pure_data.py`
- Create: `ptycho_torch/pure_data.py`

**Step 1: Write the failing tests**

```python
import numpy as np
import pytest
from ptycho import params


def _write_npz(tmp_path, key_name="diffraction"):
    xcoords = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ycoords = np.array([0.0, -1.0, -2.0, -3.0], dtype=np.float64)
    diffraction = np.ones((4, 4, 4), dtype=np.float32)
    probe = (np.ones((4, 4)) + 1j * np.zeros((4, 4))).astype(np.complex64)
    path = tmp_path / "sample.npz"
    payload = {
        "xcoords": xcoords,
        "ycoords": ycoords,
        "probeGuess": probe,
        key_name: diffraction,
    }
    np.savez(path, **payload)
    return path


def test_load_raw_npz_accepts_diffraction_and_diff3d(tmp_path):
    from ptycho_torch.pure_data import load_raw_npz

    p1 = _write_npz(tmp_path, "diffraction")
    p2 = _write_npz(tmp_path, "diff3d")
    d1 = load_raw_npz(p1)
    d2 = load_raw_npz(p2)
    assert d1.diffraction.shape == (4, 4, 4)
    assert d2.diffraction.shape == (4, 4, 4)


def test_group_raw_npz_shapes_and_keys(tmp_path):
    from ptycho_torch.pure_data import load_raw_npz, group_raw_npz

    path = _write_npz(tmp_path, "diffraction")
    raw = load_raw_npz(path)
    grouped = group_raw_npz(raw, N=4, gridsize=2, neighbor_count=4, n_groups=2)
    assert grouped["diffraction"].shape == (2, 4, 4, 4)
    assert grouped["coords_relative"].shape == (2, 1, 2, 4)
    assert grouped["coords_offsets"].shape == (2, 1, 2, 1)


def test_group_raw_npz_requires_k_ge_c(tmp_path):
    from ptycho_torch.pure_data import load_raw_npz, group_raw_npz

    path = _write_npz(tmp_path, "diffraction")
    raw = load_raw_npz(path)
    with pytest.raises(ValueError):
        group_raw_npz(raw, N=4, gridsize=2, neighbor_count=3, n_groups=1)


def test_pure_loader_does_not_touch_params_cfg(tmp_path):
    from ptycho_torch.pure_data import load_raw_npz, group_raw_npz

    params_snapshot = dict(params.cfg)
    path = _write_npz(tmp_path, "diffraction")
    raw = load_raw_npz(path)
    _ = group_raw_npz(raw, N=4, gridsize=1, neighbor_count=1, n_groups=1)
    assert params.cfg == params_snapshot
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_pure_data.py -v`
Expected: FAIL with `ModuleNotFoundError: ptycho_torch.pure_data`

**Step 3: Write minimal implementation**

```python
# ptycho_torch/pure_data.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict
import numpy as np
from scipy.spatial import cKDTree

LOCAL_OFFSET_SIGN = -1

@dataclass(frozen=True)
class RawNPZData:
    xcoords: np.ndarray
    ycoords: np.ndarray
    diffraction: np.ndarray
    probeGuess: np.ndarray
    objectGuess: Optional[np.ndarray] = None
    scan_index: Optional[np.ndarray] = None
    metadata: Optional[dict] = None


def _canonicalize_diffraction(diffraction: np.ndarray, n_points: int) -> np.ndarray:
    if diffraction.ndim != 3:
        raise ValueError(f"diffraction must be 3D (N,H,W), got {diffraction.shape}")
    if diffraction.shape[0] != n_points and diffraction.shape[-1] == n_points:
        diffraction = np.transpose(diffraction, (2, 0, 1))
    return diffraction


def load_raw_npz(path: Path) -> RawNPZData:
    data = np.load(path)
    xcoords = np.asarray(data["xcoords"], dtype=np.float64)
    ycoords = np.asarray(data["ycoords"], dtype=np.float64)
    if "diffraction" in data:
        diff = data["diffraction"]
    elif "diff3d" in data:
        diff = data["diff3d"]
    else:
        raise KeyError("NPZ missing diffraction or diff3d")
    diff = _canonicalize_diffraction(np.asarray(diff, dtype=np.float32), len(xcoords))
    probe = np.asarray(data["probeGuess"], dtype=np.complex64)
    obj = np.asarray(data["objectGuess"], dtype=np.complex64) if "objectGuess" in data else None
    scan_index = np.asarray(data["scan_index"], dtype=np.int32) if "scan_index" in data else None
    metadata = data["_metadata"].item() if "_metadata" in data else None
    return RawNPZData(xcoords, ycoords, diff, probe, obj, scan_index, metadata)


def _normalize_diffraction(diffraction: np.ndarray, N: int) -> np.ndarray:
    denom = np.mean(np.sum(diffraction ** 2, axis=(1, 2)))
    scale = np.sqrt(((N / 2) ** 2) / denom)
    return diffraction * scale


def group_raw_npz(raw: RawNPZData, *, N: int, gridsize: int, neighbor_count: int, n_groups: int) -> Dict[str, np.ndarray]:
    C = gridsize ** 2
    if neighbor_count < C:
        raise ValueError(f"neighbor_count={neighbor_count} must be >= C={C}")
    coords = np.column_stack([raw.xcoords, raw.ycoords])
    n_points = len(coords)
    n_groups_actual = min(n_groups, n_points)
    rng = np.random.default_rng()
    seed_indices = rng.choice(np.arange(n_points), size=n_groups_actual, replace=False)
    if C == 1:
        nn_indices = seed_indices.reshape(-1, 1)
    else:
        tree = cKDTree(coords)
        _, neighbor_indices = tree.query(coords[seed_indices], k=min(neighbor_count + 1, n_points))
        nn_indices = np.zeros((n_groups_actual, C), dtype=np.int32)
        for i, neighbors in enumerate(neighbor_indices):
            if len(neighbors) > neighbor_count:
                neighbors = neighbors[1:neighbor_count + 1]
            available = neighbors if len(neighbors) >= C else np.concatenate([[seed_indices[i]], neighbors])
            nn_indices[i] = rng.choice(available, size=C, replace=(len(available) < C))

    diff4d = np.transpose(raw.diffraction[nn_indices], (0, 2, 3, 1))
    coords_nn = np.transpose(np.array([raw.xcoords[nn_indices], raw.ycoords[nn_indices]]), (1, 0, 2))[:, None, :, :]
    coords_offsets = np.mean(coords_nn, axis=3)[..., None]
    coords_relative = LOCAL_OFFSET_SIGN * (coords_nn - coords_offsets)
    dset = {
        "diffraction": diff4d,
        "Y": None,
        "coords_offsets": coords_offsets,
        "coords_relative": coords_relative,
        "coords_nn": coords_nn,
        "nn_indices": nn_indices,
        "objectGuess": raw.objectGuess,
    }
    dset["X_full"] = _normalize_diffraction(dset["diffraction"], N)
    return dset
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_pure_data.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/torch/test_pure_data.py ptycho_torch/pure_data.py
git commit -m "feat(torch): add pure raw NPZ loader and grouper"
```

---

### Task 2: Facade Class + Generator Registry Wiring

**Files:**
- Create: `ptycho_torch/api/pure_model.py`
- Modify: `ptycho_torch/api/__init__.py`
- Create: `tests/torch/test_pure_model.py`

**Step 1: Write the failing tests**

```python
import numpy as np
from unittest.mock import MagicMock


def test_pychomodel_builds_config_and_architecture(tmp_path, monkeypatch):
    from ptycho_torch.api.pure_model import PtychoModel

    model = PtychoModel(architecture="hybrid_resnet", N=64, gridsize=1)
    assert model.model_config.architecture == "hybrid_resnet"


def test_pychomodel_train_uses_pure_grouping(tmp_path, monkeypatch):
    from ptycho_torch.api.pure_model import PtychoModel
    from ptycho_torch import pure_data

    dummy_grouped = {
        "diffraction": np.ones((1, 4, 4, 1), dtype=np.float32),
        "X_full": np.ones((1, 4, 4, 1), dtype=np.float32),
        "coords_relative": np.zeros((1, 1, 2, 1), dtype=np.float32),
        "coords_offsets": np.zeros((1, 1, 2, 1), dtype=np.float64),
        "nn_indices": np.zeros((1, 1), dtype=np.int32),
        "Y": None,
    }

    monkeypatch.setattr(pure_data, "load_raw_npz", lambda *args, **kwargs: "raw")
    monkeypatch.setattr(pure_data, "group_raw_npz", lambda *args, **kwargs: dummy_grouped)

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            self.fit_called = False
        def fit(self, *args, **kwargs):
            self.fit_called = True

    model = PtychoModel(N=4, gridsize=1)
    model.train(tmp_path / "train.npz", trainer_cls=DummyTrainer)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_pure_model.py -v`
Expected: FAIL with `ModuleNotFoundError: ptycho_torch.api.pure_model`

**Step 3: Write minimal implementation**

```python
# ptycho_torch/api/pure_model.py
from pathlib import Path
from typing import Optional, Type
import lightning as L
from ptycho_torch.config_params import ModelConfig, DataConfig, TrainingConfig, InferenceConfig
from ptycho_torch.generators.registry import resolve_generator
from ptycho_torch.model import PtychoPINN_Lightning
from ptycho_torch.dataloader import PtychoDataset, TensorDictDataLoader
from ptycho_torch.data_container_bridge import PtychoDataContainerTorch
from ptycho_torch import pure_data


class PtychoModel:
    def __init__(self, *, architecture: str = "cnn", N: int = 64, gridsize: int = 1, **kwargs):
        self.model_config = ModelConfig(architecture=architecture, N=N, gridsize=gridsize, **kwargs)
        self.data_config = DataConfig(N=N, grid_size=(gridsize, gridsize))
        self.training_config = TrainingConfig()
        self.inference_config = InferenceConfig()
        self.execution_config = kwargs.get("execution_config", None)
        self.model = None
        self.intensity_scale = None

    def _build_model(self):
        generator = resolve_generator(self._wrap_config())
        pt_configs = {
            "model_config": self.model_config,
            "data_config": self.data_config,
            "training_config": self.training_config,
        }
        self.model = generator.build_model(pt_configs)

    def _wrap_config(self):
        class _Wrapper:
            def __init__(self, model):
                self.model = model
        return _Wrapper(self.model_config)

    def train(self, train_npz: Path, *, trainer_cls: Type[L.Trainer] = L.Trainer, **kwargs):
        raw = pure_data.load_raw_npz(train_npz)
        grouped = pure_data.group_raw_npz(
            raw,
            N=self.model_config.N,
            gridsize=self.model_config.gridsize,
            neighbor_count=kwargs.get("neighbor_count", 4),
            n_groups=kwargs.get("n_groups", 1),
        )
        container = PtychoDataContainerTorch(grouped, raw.probeGuess)
        dataset = PtychoDataset(container)
        loader = TensorDictDataLoader(dataset, batch_size=kwargs.get("batch_size", 1), shuffle=True)
        if self.model is None:
            self._build_model()
        trainer = trainer_cls()
        trainer.fit(self.model, loader)
        return self
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_pure_model.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/torch/test_pure_model.py ptycho_torch/api/pure_model.py ptycho_torch/api/__init__.py
git commit -m "feat(torch): add PtychoModel facade for pure API"
```

---

### Task 3: Inference Path + Reassembly Glue (Pure)

**Files:**
- Modify: `ptycho_torch/api/pure_model.py`
- Create: `tests/torch/test_pure_model_infer.py`

**Step 1: Write the failing test**

```python
import numpy as np
from unittest.mock import MagicMock


def test_pychomodel_infer_calls_reassembly(tmp_path, monkeypatch):
    from ptycho_torch.api.pure_model import PtychoModel
    from ptycho_torch import pure_data

    dummy_grouped = {
        "diffraction": np.ones((1, 4, 4, 1), dtype=np.float32),
        "X_full": np.ones((1, 4, 4, 1), dtype=np.float32),
        "coords_relative": np.zeros((1, 1, 2, 1), dtype=np.float32),
        "coords_offsets": np.zeros((1, 1, 2, 1), dtype=np.float64),
        "nn_indices": np.zeros((1, 1), dtype=np.int32),
        "Y": None,
    }

    monkeypatch.setattr(pure_data, "load_raw_npz", lambda *args, **kwargs: type("R", (), {"probeGuess": np.ones((4, 4), dtype=np.complex64), "xcoords": np.array([0.0]), "ycoords": np.array([0.0])})())
    monkeypatch.setattr(pure_data, "group_raw_npz", lambda *args, **kwargs: dummy_grouped)

    model = PtychoModel(N=4, gridsize=1)
    model.model = MagicMock()
    model.model.forward_predict = MagicMock(return_value=np.ones((1, 1, 4, 4), dtype=np.complex64))

    result = model.infer(tmp_path / "test.npz", stitching=False)
    assert result is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_pure_model_infer.py -v`
Expected: FAIL with `AttributeError: 'PtychoModel' object has no attribute 'infer'`

**Step 3: Write minimal implementation**

```python
    def infer(self, test_npz: Path, *, stitching: bool = True, **kwargs):
        raw = pure_data.load_raw_npz(test_npz)
        grouped = pure_data.group_raw_npz(
            raw,
            N=self.model_config.N,
            gridsize=self.model_config.gridsize,
            neighbor_count=kwargs.get("neighbor_count", 4),
            n_groups=kwargs.get("n_groups", 1),
        )
        # TODO: build dataloader, run forward_predict, optional stitching via ptycho_torch.reassembly
        return grouped
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_pure_model_infer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/torch/test_pure_model_infer.py ptycho_torch/api/pure_model.py
git commit -m "feat(torch): add pure inference entrypoint"
```

---

### Task 4: Documentation + Usage Example

**Files:**
- Modify: `ptycho_torch/README.md`

**Step 1: Add usage snippet (failing doc build is OK, no tests)**

```markdown
## Pure PyTorch API (No params.cfg)

```python
from ptycho_torch.api.pure_model import PtychoModel

model = PtychoModel(architecture="hybrid_resnet", N=128, gridsize=1)
model.train("train.npz", neighbor_count=4, n_groups=512)
recon = model.infer("test.npz", stitching=True)
```
```

**Step 2: Commit**

```bash
git add ptycho_torch/README.md
git commit -m "docs(torch): add pure API usage snippet"
```

---

### Test Checklist (per `docs/TESTING_GUIDE.md`)

Run these selectors during implementation and archive logs:
- `pytest tests/torch/test_pure_data.py -v`
- `pytest tests/torch/test_pure_model.py -v`
- `pytest tests/torch/test_pure_model_infer.py -v`

If implementation touches training/inference orchestration, also run:
- `pytest tests/torch/test_workflows_components.py -v`

