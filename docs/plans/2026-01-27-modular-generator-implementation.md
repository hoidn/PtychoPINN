# Modular Generator Registry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a minimal, config-driven generator registry so the CNN PINN can be swapped for other unsupervised architectures without rewiring the pipeline.

**Architecture:** Introduce a thin generator registry in both TF and Torch backends keyed by `model.architecture`. For now, only a CNN generator is implemented; unknown architectures raise clear errors. Workflows instantiate generators via the registry while leaving physics/consistency layers unchanged.

**Tech Stack:** Python, TensorFlow (Keras), PyTorch + Lightning, dataclasses config, pytest.

## Baseline Status (Preflight)
- Worktree: `.worktrees/plan-modular-generator`
- `poetry` not available (`/bin/bash: poetry: command not found`).
- `pytest` run (300s timeout) failed and timed out. Failures observed before timeout:
  - `tests/io/test_ptychodus_interop_h5.py`
  - `tests/study/test_dose_overlap_comparison.py`
  - `tests/study/test_phase_g_dense_orchestrator.py` (multiple failures)
- Do **not** treat full-suite failures as regressions for this change unless selectors touched by this work fail.

---

### Task 1: Add `model.architecture` to ModelConfig + validation + docs

**Files:**
- Modify: `ptycho/config/config.py`
- Modify: `docs/CONFIGURATION.md`
- Test: `tests/test_model_config_architecture.py`

**Step 1: Write the failing test**

```python
# tests/test_model_config_architecture.py
import pytest
from ptycho.config.config import ModelConfig, validate_model_config


def test_model_config_architecture_default_ok():
    cfg = ModelConfig()
    validate_model_config(cfg)


def test_model_config_architecture_invalid_raises():
    cfg = ModelConfig(architecture="not-a-real-arch")
    with pytest.raises(ValueError):
        validate_model_config(cfg)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_config_architecture.py -v`
Expected: FAIL because `ModelConfig` has no `architecture` field and validation ignores it.

**Step 3: Write minimal implementation**

```python
# ptycho/config/config.py (ModelConfig)
architecture: Literal['cnn', 'fno', 'hybrid'] = 'cnn'

# ptycho/config/config.py (validate_model_config)
valid_arches = {'cnn', 'fno', 'hybrid'}
if config.architecture not in valid_arches:
    raise ValueError(
        f"Invalid architecture '{config.architecture}'. "
        f"Expected one of {sorted(valid_arches)}."
    )
```

Update `docs/CONFIGURATION.md` to document `architecture` in the ModelConfig table.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_config_architecture.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ptycho/config/config.py docs/CONFIGURATION.md tests/test_model_config_architecture.py
git commit -m "feat: add model.architecture config field"
```

---

### Task 2: Bridge `architecture` through PyTorch config bridge + factory + spec

**Files:**
- Modify: `ptycho_torch/config_params.py`
- Modify: `ptycho_torch/config_bridge.py`
- Modify: `ptycho_torch/config_factory.py`
- Modify: `docs/specs/spec-ptycho-config-bridge.md`
- Test: `tests/torch/test_config_bridge.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_config_bridge.py (add test)
from ptycho_torch.config_params import DataConfig, ModelConfig
from ptycho_torch import config_bridge

def test_config_bridge_architecture_override(params_cfg_snapshot):
    pt_data = DataConfig(N=64, grid_size=(1, 1))
    pt_model = ModelConfig()

    tf_model = config_bridge.to_model_config(
        pt_data,
        pt_model,
        overrides={'architecture': 'cnn'}
    )

    assert tf_model.architecture == 'cnn'
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_config_bridge.py -k architecture -v`
Expected: FAIL (field missing in TF ModelConfig or not passed through)

**Step 3: Write minimal implementation**

```python
# ptycho_torch/config_params.py (ModelConfig)
architecture: Literal['cnn', 'fno', 'hybrid'] = 'cnn'

# ptycho_torch/config_bridge.py (to_model_config kwargs)
kwargs = {
    # existing fields...
    'architecture': model.architecture,
}

# ptycho_torch/config_factory.py (overrides)
factory_overrides = {
    # existing fields...
    'architecture': config.model.architecture,
}
```

Update `docs/specs/spec-ptycho-config-bridge.md` to include the new mapping rule:
- `ModelConfig.architecture` → `ModelConfig.architecture` (direct pass-through)

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_config_bridge.py -k architecture -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ptycho_torch/config_params.py ptycho_torch/config_bridge.py ptycho_torch/config_factory.py \
  docs/specs/spec-ptycho-config-bridge.md tests/torch/test_config_bridge.py
git commit -m "feat: bridge model.architecture through torch config"
```

---

### Task 3: Add generator registry (TF + Torch) with CNN implementation

**Files:**
- Create: `ptycho/generators/__init__.py`
- Create: `ptycho/generators/registry.py`
- Create: `ptycho/generators/cnn.py`
- Create: `ptycho_torch/generators/__init__.py`
- Create: `ptycho_torch/generators/registry.py`
- Create: `ptycho_torch/generators/cnn.py`
- Test: `tests/test_generator_registry.py`
- Test: `tests/torch/test_generator_registry.py`

**Step 1: Write the failing tests**

```python
# tests/test_generator_registry.py
import pytest
from ptycho.config.config import ModelConfig, TrainingConfig
from ptycho.generators.registry import resolve_generator


def test_resolve_generator_cnn():
    cfg = TrainingConfig(model=ModelConfig(architecture='cnn'))
    gen = resolve_generator(cfg)
    assert gen.name == 'cnn'


def test_resolve_generator_unknown_raises():
    cfg = TrainingConfig(model=ModelConfig(architecture='unknown'))
    with pytest.raises(ValueError):
        resolve_generator(cfg)
```

```python
# tests/torch/test_generator_registry.py
import pytest
from ptycho.config.config import ModelConfig, TrainingConfig
from ptycho_torch.generators.registry import resolve_generator


def test_resolve_generator_cnn():
    cfg = TrainingConfig(model=ModelConfig(architecture='cnn'))
    gen = resolve_generator(cfg)
    assert gen.name == 'cnn'
```

**Step 2: Run tests to verify they fail**

Run:
- `pytest tests/test_generator_registry.py -v`
- `pytest tests/torch/test_generator_registry.py -v`

Expected: FAIL (registry modules missing)

**Step 3: Write minimal implementation**

```python
# ptycho/generators/registry.py
from ptycho.generators.cnn import CnnGenerator

_REGISTRY = {
    'cnn': CnnGenerator,
}


def resolve_generator(config):
    arch = config.model.architecture
    if arch not in _REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[arch](config)
```

```python
# ptycho/generators/cnn.py
class CnnGenerator:
    name = 'cnn'
    def __init__(self, config):
        self.config = config

    def build_models(self):
        from ptycho import model
        return model.create_compiled_model()
```

```python
# ptycho_torch/generators/registry.py
from ptycho_torch.generators.cnn import CnnGenerator

_REGISTRY = {
    'cnn': CnnGenerator,
}


def resolve_generator(config):
    arch = config.model.architecture
    if arch not in _REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[arch](config)
```

```python
# ptycho_torch/generators/cnn.py
class CnnGenerator:
    name = 'cnn'
    def __init__(self, config):
        self.config = config

    def build_model(self, pt_configs):
        from ptycho_torch.model import PtychoPINN_Lightning
        return PtychoPINN_Lightning(**pt_configs)
```

**Step 4: Run tests to verify they pass**

Run:
- `pytest tests/test_generator_registry.py -v`
- `pytest tests/torch/test_generator_registry.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add ptycho/generators ptycho_torch/generators \
  tests/test_generator_registry.py tests/torch/test_generator_registry.py
git commit -m "feat: add generator registry with cnn implementation"
```

---

### Task 4: Wire generator selection into workflows (TF + Torch)

**Files:**
- Modify: `ptycho/workflows/components.py`
- Modify: `ptycho/train_pinn.py`
- Modify: `ptycho_torch/workflows/components.py`
- Modify: `docs/workflows/pytorch.md`
- Test: `tests/torch/test_workflows_components.py`
- Test: `tests/test_workflows_components.py` (new or existing)

**Step 1: Write failing tests**

```python
# tests/test_workflows_components.py (new or extend)
from unittest.mock import patch
from ptycho.config.config import ModelConfig, TrainingConfig
from ptycho.workflows import components


def test_train_cdi_model_uses_generator_registry():
    cfg = TrainingConfig(model=ModelConfig(architecture='cnn'))
    with patch('ptycho.generators.registry.resolve_generator') as mock_resolve:
        mock_resolve.return_value.build_models.return_value = (object(), object())
        components.train_cdi_model(train_data=object(), test_data=None, config=cfg)
        assert mock_resolve.called
```

```python
# tests/torch/test_workflows_components.py (add test)
from unittest.mock import patch
from ptycho.config.config import ModelConfig, TrainingConfig
from ptycho_torch.workflows import components


def test_train_with_lightning_uses_generator_registry():
    cfg = TrainingConfig(model=ModelConfig(architecture='cnn'))
    with patch('ptycho_torch.generators.registry.resolve_generator') as mock_resolve:
        mock_resolve.return_value.build_model.return_value = object()
        # call _train_with_lightning with minimal stubs/mocks for containers
        # Expect resolve_generator called
```

**Step 2: Run tests to verify they fail**

Run:
- `pytest tests/test_workflows_components.py -k generator -v`
- `pytest tests/torch/test_workflows_components.py -k generator -v`

Expected: FAIL (registry not wired)

**Step 3: Write minimal implementation**

```python
# ptycho/workflows/components.py
from ptycho.generators.registry import resolve_generator

# in train_cdi_model
generator = resolve_generator(config)
model_instance, diffraction_to_obj = generator.build_models()
results = train_pinn.train_eval(PtychoDataset(train_container, test_container), model_instance=model_instance)
```

```python
# ptycho/train_pinn.py
def train_eval(ptycho_dataset, model_instance=None):
    model_instance, history = train(ptycho_dataset.train_data, model_instance=model_instance)
    # rest unchanged
```

```python
# ptycho_torch/workflows/components.py (inside _train_with_lightning)
from ptycho_torch.generators.registry import resolve_generator

pt_configs = dict(
    model_config=pt_model_config,
    data_config=pt_data_config,
    training_config=pt_training_config,
    inference_config=pt_inference_config,
)
model = resolve_generator(config).build_model(pt_configs)
```

Update `docs/workflows/pytorch.md` to mention `model.architecture` selection.

**Step 4: Run tests to verify they pass**

Run:
- `pytest tests/test_workflows_components.py -k generator -v`
- `pytest tests/torch/test_workflows_components.py -k generator -v`

Expected: PASS

**Step 5: Commit**

```bash
git add ptycho/workflows/components.py ptycho/train_pinn.py \
  ptycho_torch/workflows/components.py docs/workflows/pytorch.md \
  tests/test_workflows_components.py tests/torch/test_workflows_components.py
git commit -m "feat: wire generator selection into workflows"
```

---

### Task 5: Verification & Evidence

**Required tests (per TESTING_GUIDE.md):**
- Unit tests added/modified in this plan.
- Integration marker: `pytest -m integration`

**Commands:**
```bash
pytest tests/test_model_config_architecture.py -v
pytest tests/torch/test_config_bridge.py -k architecture -v
pytest tests/test_generator_registry.py -v
pytest tests/torch/test_generator_registry.py -v
pytest tests/test_workflows_components.py -k generator -v
pytest tests/torch/test_workflows_components.py -k generator -v
pytest -m integration
```

**Evidence capture:**
- Save logs under `.artifacts/modular-generator/` (e.g., `pytest_integration.log`).
- Add a short note in this plan (top section) pointing to the log paths.

**Commit (if any doc/test-only tweaks happen during verification):**
```bash
git add <files>
git commit -m "test: verify modular generator wiring"
```

---

## Execution Handoff
Plan complete and saved to `docs/plans/2026-01-27-modular-generator-implementation.md`.

Two execution options:
1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks
2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
