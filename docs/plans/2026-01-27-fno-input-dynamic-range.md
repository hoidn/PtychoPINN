# FNO Input Dynamic-Range Compression Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an optional, config-gated input dynamic-range compressor for FNO/Hybrid generators to mitigate DC peak dominance while keeping default behavior unchanged.

**Architecture:** Introduce a small `InputTransform` module (none/sqrt/log1p/instancenorm) and wire it into `SpatialLifter` for FNO/Hybrid only. Expose the option via `ModelConfig.fno_input_transform` (default `"none"`) in both TF and PyTorch configs, and bridge through `config_bridge` and `config_factory`. Update docs and tests to ensure the option is propagated and behavior is correct.

**Tech Stack:** PyTorch, dataclasses, pytest

**Note:** User requested no worktree usage; plan executes directly on `fno2`.

---

### Task 1: Add failing unit tests for input transform

**Files:**
- Modify: `tests/torch/test_fno_generators.py`
- Modify (if needed): `tests/torch/test_config_bridge.py`

**Step 1: Write failing tests for transform behavior**

Add tests for `InputTransform` (to be added) and for config propagation:

```python
import math
import torch


def test_input_transform_sqrt_matches_expected():
    from ptycho_torch.generators.fno import InputTransform

    x = torch.tensor([[[[0.0, 4.0], [9.0, 16.0]]]])
    transform = InputTransform(mode="sqrt")
    out = transform(x)
    assert torch.allclose(out, torch.sqrt(x))


def test_input_transform_log1p_matches_expected():
    from ptycho_torch.generators.fno import InputTransform

    x = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
    transform = InputTransform(mode="log1p")
    out = transform(x)
    assert torch.allclose(out, torch.log1p(x))
```

Add a propagation test (optional but recommended) in `tests/torch/test_config_bridge.py`:

```python
from ptycho_torch.config_params import DataConfig, ModelConfig as PTModelConfig
from ptycho_torch import config_bridge


def test_config_bridge_fno_input_transform():
    pt_data = DataConfig(N=64, C=1)
    pt_model = PTModelConfig(architecture="fno")
    pt_model.fno_input_transform = "sqrt"

    tf_model = config_bridge.to_model_config(pt_data, pt_model)
    assert tf_model.fno_input_transform == "sqrt"
```

**Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/torch/test_fno_generators.py::test_input_transform_sqrt_matches_expected -v
pytest tests/torch/test_fno_generators.py::test_input_transform_log1p_matches_expected -v
pytest tests/torch/test_config_bridge.py::test_config_bridge_fno_input_transform -v
```
Expected: FAIL (InputTransform and config field missing).

---

### Task 2: Add config field + bridge mapping

**Files:**
- Modify: `ptycho/config/config.py`
- Modify: `ptycho_torch/config_params.py`
- Modify: `ptycho_torch/config_bridge.py`

**Step 1: Add field to TF ModelConfig**

```python
class ModelConfig:
    ...
    fno_input_transform: Literal['none', 'sqrt', 'log1p', 'instancenorm'] = 'none'
```

**Step 2: Add field to PyTorch ModelConfig**

```python
class ModelConfig:
    ...
    fno_input_transform: Literal['none', 'sqrt', 'log1p', 'instancenorm'] = 'none'
```

**Step 3: Bridge to TF ModelConfig**

Add in `ptycho_torch/config_bridge.py` kwargs:

```python
'fno_input_transform': model.fno_input_transform,
```

**Step 4: Re-run config bridge test**

Run:
```bash
pytest tests/torch/test_config_bridge.py::test_config_bridge_fno_input_transform -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add ptycho/config/config.py ptycho_torch/config_params.py ptycho_torch/config_bridge.py tests/torch/test_config_bridge.py
git commit -m "feat(config): add fno_input_transform option"
```

---

### Task 3: Implement InputTransform and wire into FNO/Hybrid lifter

**Files:**
- Modify: `ptycho_torch/generators/fno.py`

**Step 1: Implement InputTransform**

```python
class InputTransform(nn.Module):
    def __init__(self, mode: str = "none", channels: int = 1):
        super().__init__()
        self.mode = mode
        self.norm = None
        if mode == "instancenorm":
            self.norm = nn.InstanceNorm2d(channels, affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x
        if self.mode == "sqrt":
            return torch.sqrt(torch.clamp(x, min=0.0))
        if self.mode == "log1p":
            return torch.log1p(torch.clamp(x, min=0.0))
        if self.mode == "instancenorm":
            return self.norm(x)
        raise ValueError(f"Unknown input transform: {self.mode}")
```

**Step 2: Add optional transform to SpatialLifter**

```python
class SpatialLifter(nn.Module):
    def __init__(..., input_transform: str = "none"):
        ...
        self.input_transform = InputTransform(input_transform, channels=in_channels)

    def forward(self, x):
        x = self.input_transform(x)
        ...
```

**Step 3: Wire config into FNO/Hybrid generators**

In `HybridUNOGenerator` and `CascadedFNOGenerator`, pass the mode:

```python
self.lifter = SpatialLifter(in_channels * C, hidden_channels, input_transform=input_transform)
```

In `FnoGenerator.build_model` and `HybridGenerator.build_model`:

```python
input_transform = getattr(model_config, "fno_input_transform", "none")
core = CascadedFNOGenerator(..., input_transform=input_transform, ...)
```

**Step 4: Re-run unit tests**

Run:
```bash
pytest tests/torch/test_fno_generators.py::test_input_transform_sqrt_matches_expected -v
pytest tests/torch/test_fno_generators.py::test_input_transform_log1p_matches_expected -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add ptycho_torch/generators/fno.py tests/torch/test_fno_generators.py
git commit -m "feat(fno): add optional input dynamic-range transform"
```

---

### Task 4: Update docs and test index

**Files:**
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/architecture_torch.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `docs/TESTING_GUIDE.md`
- Modify: `docs/development/TEST_SUITE_INDEX.md`

**Step 1: Document new config field**

Add `fno_input_transform` to the ModelConfig table in `docs/CONFIGURATION.md`.

**Step 2: Note optional input transform in architecture docs**

Add a short note under FNO components in `docs/architecture_torch.md` and/or `docs/workflows/pytorch.md` about the optional transform and default `none`.

**Step 3: Update test docs**

Add selectors for the new tests (if new tests were added) in `docs/TESTING_GUIDE.md` and update the row in `docs/development/TEST_SUITE_INDEX.md` describing `test_fno_generators.py` coverage.

**Step 4: Commit**

```bash
git add docs/CONFIGURATION.md docs/architecture_torch.md docs/workflows/pytorch.md docs/TESTING_GUIDE.md docs/development/TEST_SUITE_INDEX.md
git commit -m "docs: document fno input transform option"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-01-27-fno-input-dynamic-range.md`.

Two execution options:
1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks
2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
