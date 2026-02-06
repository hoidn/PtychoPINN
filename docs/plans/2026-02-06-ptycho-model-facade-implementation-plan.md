# PtychoModel Facade Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a minimal OO façade (`PtychoModel`) that accepts either dataset objects or paths, preserves memmap data flow, and delegates to existing PyTorch workflows while honoring legacy guardrails.

**Architecture:** A thin façade in `ptycho_torch/api/` builds configs, resolves data/probe, and calls a single legacy bridge before any legacy usage. It delegates training/inference to `ptycho_torch/workflows/components.py` and persistence to `ptycho_torch/model_manager.py`.

**Tech Stack:** Python, pytest, PyTorch (existing), Lightning (existing).

---

### Task 1: Add legacy bridge guard (RED)

**Files:**
- Test: `tests/torch/test_ptycho_model_facade.py`

**Step 1: Write failing test for legacy bridge call**

```python
# tests/torch/test_ptycho_model_facade.py
from unittest import mock
from pathlib import Path

def test_facade_calls_legacy_bridge_on_path_inputs(tmp_path, monkeypatch):
    from ptycho_torch.api import model as facade

    spy = mock.Mock()
    monkeypatch.setattr("ptycho_torch.legacy_bridge.populate_legacy_params", spy)

    # Minimal model params
    m = facade.PtychoModel(arch="cnn", model_params={}, training_params={}, execution_params={})
    m._resolve_data(Path("fake.npz"))

    assert spy.called, "Facade must call legacy bridge before path-based data loading"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_ptycho_model_facade.py::test_facade_calls_legacy_bridge_on_path_inputs -v`
Expected: FAIL (legacy_bridge module missing)

**Step 3: Commit failing test**

```bash
git add tests/torch/test_ptycho_model_facade.py
git commit -m "test: require legacy bridge call in facade"
```

---

### Task 2: Implement legacy bridge + data resolver (GREEN)

**Files:**
- Create: `ptycho_torch/legacy_bridge.py`
- Create: `ptycho_torch/api/data_resolver.py`
- Modify: `ptycho_torch/api/model.py`

**Step 1: Implement legacy bridge**

```python
# ptycho_torch/legacy_bridge.py
from ptycho import params
from ptycho.config import config as ptycho_config

def populate_legacy_params(config):
    """Centralized mutation of params.cfg required for legacy modules."""
    ptycho_config.update_legacy_dict(params.cfg, config)
```

**Step 2: Add data resolver (path vs object)**

```python
# ptycho_torch/api/data_resolver.py
from pathlib import Path
from typing import Any, Tuple
from ptycho.raw_data import RawData
from ptycho_torch.dataloader import PtychoDataset
from ptycho_torch.workflows.components import _ensure_container
from ptycho_torch.legacy_bridge import populate_legacy_params


def resolve_data(data: Any, config, use_memmap: bool) -> Tuple[Any, str]:
    """Return (data_handle, mode) where mode is 'memmap' or 'container'."""
    if isinstance(data, (str, Path)):
        populate_legacy_params(config)
        path = Path(data)
        if use_memmap:
            return PtychoDataset(str(path), probe_dir=None), "memmap"
        raw = RawData.from_file(str(path))
        return _ensure_container(raw, config), "container"
    return data, "object"
```

**Step 3: Update `PtychoModel` to use resolver**

```python
# ptycho_torch/api/model.py (skeleton)
from ptycho.config.config import TrainingConfig, ModelConfig, PyTorchExecutionConfig
from ptycho_torch.api.data_resolver import resolve_data

class PtychoModel:
    def __init__(self, arch, model_params, training_params, execution_params, probe_path=None, use_memmap=False):
        self.model_config = ModelConfig(architecture=arch, **model_params)
        self.training_config = TrainingConfig(model=self.model_config, **training_params)
        self.execution_config = PyTorchExecutionConfig(**execution_params)
        self.probe_path = probe_path
        self.use_memmap = use_memmap

    def _resolve_data(self, data):
        return resolve_data(data, self.training_config, self.use_memmap)
```

**Step 4: Run tests**

Run: `pytest tests/torch/test_ptycho_model_facade.py::test_facade_calls_legacy_bridge_on_path_inputs -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ptycho_torch/legacy_bridge.py ptycho_torch/api/data_resolver.py ptycho_torch/api/model.py
git commit -m "feat: add legacy bridge and data resolver for facade"
```

---

### Task 3: Train/infer/save/load façade (RED → GREEN)

**Files:**
- Modify: `ptycho_torch/api/model.py`
- Test: `tests/torch/test_ptycho_model_facade.py`

**Step 1: Add failing tests for train/infer/save/load contracts**

```python
# tests/torch/test_ptycho_model_facade.py
from unittest import mock

def test_facade_train_delegates(monkeypatch):
    from ptycho_torch.api import model as facade
    spy = mock.Mock(return_value={"history": {}})
    monkeypatch.setattr("ptycho_torch.workflows.components.train_cdi_model_torch", spy)

    m = facade.PtychoModel(arch="cnn", model_params={}, training_params={}, execution_params={})
    m.train("fake.npz")
    assert spy.called


def test_facade_save_load_delegates(monkeypatch, tmp_path):
    from ptycho_torch.api import model as facade
    save_spy = mock.Mock()
    load_spy = mock.Mock(return_value=({"diffraction_to_obj": object()}, {"intensity_scale": 1.0}))
    monkeypatch.setattr("ptycho_torch.model_manager.save_torch_bundle", save_spy)
    monkeypatch.setattr("ptycho_torch.model_manager.load_torch_bundle", load_spy)

    m = facade.PtychoModel(arch="cnn", model_params={}, training_params={}, execution_params={})
    m.save(tmp_path)
    facade.PtychoModel.load(tmp_path)
    assert save_spy.called
    assert load_spy.called
```

**Step 2: Run tests to confirm failure**

Run: `pytest tests/torch/test_ptycho_model_facade.py::test_facade_train_delegates -v`
Expected: FAIL (train not implemented)

**Step 3: Implement train/infer/save/load**

```python
# ptycho_torch/api/model.py
from ptycho_torch.workflows.components import train_cdi_model_torch, load_inference_bundle_torch
from ptycho_torch.model_manager import save_torch_bundle, load_torch_bundle

    def train(self, train_data, test_data=None):
        data, _ = self._resolve_data(train_data)
        test, _ = self._resolve_data(test_data) if test_data else (None, None)
        return train_cdi_model_torch(data, test, self.training_config, execution_config=self.execution_config)

    def infer(self, test_data):
        # Minimal façade: require a saved bundle path or loaded models in future iterations
        data, _ = self._resolve_data(test_data)
        return load_inference_bundle_torch(self.training_config.output_dir)

    def save(self, output_dir):
        # Save uses existing bundle format
        save_torch_bundle(models_dict=self._models, base_path=str(Path(output_dir) / "wts.h5"), config=self.training_config)

    @classmethod
    def load(cls, output_dir):
        models, params_dict = load_torch_bundle(str(Path(output_dir) / "wts.h5"))
        return models, params_dict
```

**Step 4: Run tests**

Run: `pytest tests/torch/test_ptycho_model_facade.py::test_facade_train_delegates -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ptycho_torch/api/model.py tests/torch/test_ptycho_model_facade.py
git commit -m "feat: add PtychoModel train/infer/save/load facade"
```

---

### Task 4: Memmap path support (RED → GREEN)

**Files:**
- Modify: `ptycho_torch/api/model.py`
- Test: `tests/torch/test_ptycho_model_facade.py`

**Step 1: Add failing test for memmap resolution**

```python
# tests/torch/test_ptycho_model_facade.py

def test_facade_uses_memmap_dataset(monkeypatch):
    from ptycho_torch.api import model as facade
    from ptycho_torch.dataloader import PtychoDataset

    m = facade.PtychoModel(arch="cnn", model_params={}, training_params={}, execution_params={}, use_memmap=True)
    data, mode = m._resolve_data("fake.npz")
    assert isinstance(data, PtychoDataset)
    assert mode in ("memmap", "object")
```

**Step 2: Run test to confirm failure**

Run: `pytest tests/torch/test_ptycho_model_facade.py::test_facade_uses_memmap_dataset -v`
Expected: FAIL (memmap path not wired)

**Step 3: Implement memmap handling**

Update `resolve_data` to build `PtychoDataset` for path inputs when `use_memmap=True`.

**Step 4: Run tests**

Run: `pytest tests/torch/test_ptycho_model_facade.py::test_facade_uses_memmap_dataset -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ptycho_torch/api/data_resolver.py tests/torch/test_ptycho_model_facade.py
git commit -m "feat: support memmap path resolution in facade"
```

---

### Task 5: Document façade usage

**Files:**
- Modify: `docs/workflows/pytorch.md`
- Modify: `docs/architecture_torch.md`

**Step 1: Add a short façade usage example to the PyTorch workflow guide**

Include a minimal snippet showing `PtychoModel(...).train(...).infer(...)` and mention memmap path support.

**Step 2: Add façade to architecture diagram text**

Note the façade as an optional API layer above workflows/components.

**Step 3: Commit**

```bash
git add docs/workflows/pytorch.md docs/architecture_torch.md
git commit -m "docs: add PtychoModel facade usage"
```

---

### Task 6: Focused verification

Run:
- `pytest tests/torch/test_ptycho_model_facade.py -v`

Expected: PASS

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-02-06-ptycho-model-facade-implementation-plan.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
