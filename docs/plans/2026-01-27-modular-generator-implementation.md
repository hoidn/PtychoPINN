# Modular Generator Registry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add minimal, config-driven generator selection in TF and integrate Torch-based FNO/hybrid generators (via `neuraloperator`) into the grid-lines harness without rewriting physics/consistency layers.

**Architecture:** Keep the TF workflow as the master harness (CNN PINN + baseline). Route non-`cnn` architectures to a Torch runner that uses `ptycho_torch` from `origin/torchapi-devel` for physics/consistency. TF uses a thin registry keyed by `model.architecture`; Torch selection is handled inside the runner.

**Tech Stack:** Python, TensorFlow (Keras), PyTorch + Lightning, dataclasses config, pytest.

## Mixed-Backend Amendments (2026-01-27)
- **Torch source of truth:** replace `ptycho_torch/` with the `origin/torchapi-devel` version before adding generators.
- **Registry scope:** TF registry remains; Torch registry tasks are superseded by a Torch runner + local factory.
- **Routing:** `cnn` runs in TF; `fno`/`hybrid` run in Torch and emit TF-compatible artifacts.
- **No stitching changes:** consistency layer behavior stays unchanged (out of scope).

## Baseline Status (Preflight)
- Worktree: `.worktrees/plan-modular-generator`
- `poetry` not available (`/bin/bash: poetry: command not found`).
- `pytest` run (300s timeout) failed and timed out. Failures observed before timeout:
  - `tests/io/test_ptychodus_interop_h5.py`
  - `tests/study/test_dose_overlap_comparison.py`
  - `tests/study/test_phase_g_dense_orchestrator.py` (multiple failures)
- Do **not** treat full-suite failures as regressions for this change unless selectors touched by this work fail.

## Alignment Notes (Grid-Lines Harness)
- Canonical experiment harness: `docs/plans/2026-01-27-grid-lines-workflow.md` (shared dataset, gridsize=1).
- Generator run labels must distinguish PINN generator vs supervised Baseline:
  - Generator outputs use `pinn_<arch>` (e.g., `pinn_cnn`, `pinn_fno`).
  - Supervised baseline uses `baseline` (do not alias with `cnn`).
- Grid-lines workflow is responsible for: dataset cache/manifest, running multiple architectures per call, baseline run, and emitting per-run artifacts + a JSON comparison report.

## FNO/Hybrid Lifter Decision (for future generator plans)
- Use a lightweight, spatially aware lifter before any Fourier layers: two 3x3 convs with GELU between (padding "same").
- Applies to both cascaded FNO (Arch A) and hybrid U-NO (Arch B).
- Avoid 1x1-only lifts (too weak for speckle geometry) and deep 4-6 layer stacks (memory-heavy, dilutes physics signal).
- Lifter must precede the first Fourier layer; avoid FFT of raw intensity directly.

## FNO/Hybrid Block Decision (PtychoBlock, for future generator plans)
- Standard FNO blocks use a 1x1 local path and activation after the spectral+local sum. For ptychography, this risks spectral bias and edge loss.
- Prefer a **PtychoBlock**: spectral conv + 3x3 local conv, wrapped by an outer residual.
- Suggested form: `y = x + GELU(Spectral(x) + Conv3x3(x))`.
- Rationale: the outer residual provides a high-frequency bypass; the 3x3 local path carries spatial gradients that a 1x1 path cannot.

---

### Task 0: Replace `ptycho_torch/` with torchapi-devel version

**Files:**
- Replace: `ptycho_torch/` (entire directory)

**Step 1: Path checkout from torchapi-devel (non-destructive in git history)**

```bash
git checkout origin/torchapi-devel -- ptycho_torch
```

**Step 2: Verify the replacement**

```bash
git status -sb
git diff --stat -- ptycho_torch
```

**Step 3: Note in plan summary why this version is required**
- torchapi-devel provides the Torch API/workflow surface used for physics/consistency.
- fno2’s `ptycho_torch/` diverges (removes generator registry, different workflow plumbing).

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

### Task 2 (Optional): Bridge `architecture` through PyTorch config bridge + factory + spec

**Note:** Only required if the Torch runner consumes the shared `TrainingConfig` objects. If the runner takes `--architecture` and its own Torch configs, skip this task.

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

### Task 3: Add generator registry (TF only) with CNN implementation

**Files:**
- Create: `ptycho/generators/__init__.py`
- Create: `ptycho/generators/registry.py`
- Create: `ptycho/generators/cnn.py`
- Test: `tests/test_generator_registry.py`

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

**Step 2: Run test to verify it fails**

Run:
- `pytest tests/test_generator_registry.py -v`

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

**Step 4: Run test to verify it passes**

Run:
- `pytest tests/test_generator_registry.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add ptycho/generators tests/test_generator_registry.py
git commit -m "feat: add TF generator registry with cnn implementation"
```

---

### Task 4: Wire generator selection into workflows (TF only)

**Files:**
- Modify: `ptycho/workflows/components.py`
- Modify: `ptycho/train_pinn.py`
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

**Step 2: Run tests to verify they fail**

Run:
- `pytest tests/test_workflows_components.py -k generator -v`

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

**Step 4: Run tests to verify they pass**

Run:
- `pytest tests/test_workflows_components.py -k generator -v`

Expected: PASS

**Step 5: Commit**

```bash
git add ptycho/workflows/components.py ptycho/train_pinn.py \
  tests/test_workflows_components.py
git commit -m "feat: wire generator selection into TF workflows"
```

---

### Task 5: Add generator README guidance (TF only)

**Files:**
- Create: `ptycho/generators/README.md`

**Step 1: Write the README content (TF)**

```markdown
# TensorFlow Generators

## Adding a Generator
1. Implement a generator class in `ptycho/generators/<name>.py`
2. Register it in `ptycho/generators/registry.py`
3. Ensure output contract: real/imag patches, shape `[B, N, N, C, 2]`
4. Add tests in `tests/test_generator_registry.py`

## Naming Conventions
- Generator runs are labeled `pinn_<arch>` in grid-lines outputs.
- The supervised baseline is labeled `baseline` (never `cnn`).
```

**Step 2: Commit**

```bash
git add ptycho/generators/README.md
git commit -m "docs: add TF generator README guidance"
```

---

### Task 6: Add Torch grid-lines runner (fno/hybrid)

**Files:**
- Create: `scripts/studies/grid_lines_torch_runner.py`
- Create: `ptycho_torch/workflows/grid_lines_torch_runner.py`
- Modify: `docs/plans/2026-01-27-grid-lines-workflow.md` (invoke runner + merge metrics)

**Step 1: Define the runner contract**
- Inputs: cached `train.npz`, `test.npz`, `output_dir`, `architecture` (`fno` or `hybrid`), `seed`, training hyperparams.
- Outputs: artifacts under `output_dir/runs/pinn_<arch>/` and metrics JSON compatible with the TF workflow.

**Step 2: Implement the Torch runner**
- Use torchapi-devel training/inference entrypoints to train and infer the requested architecture.
- Keep physics/consistency inside `ptycho_torch` (no TF reuse).
- Write metrics JSON with the same keys used by TF runs for merge.

**Step 3: Add a minimal smoke test**
- Validate that the runner can load cached NPZs and emits the metrics JSON (use tiny synthetic data).

---

### Task 7: Implement FNO/hybrid generators in Torch (neuraloperator)

**Files:**
- Create: `ptycho_torch/generators/fno.py` (Arch B: hybrid U-NO first)
- Create: `ptycho_torch/generators/fno_cascade.py` (Arch A: cascaded FNO → CNN)
- Modify: Torch runner to select generator class by `architecture`
- Modify: dependency management to include `neuraloperator`

**Step 1: Add dependency**
- Add `neuraloperator` to the Torch environment requirements (document the install step in the plan or workflow).

**Step 2: Implement Arch B (Hybrid U-NO, `architecture=hybrid`)**
- Use the lifter (2×3x3 convs + GELU) before spectral blocks.
- Use PtychoBlock (spectral + 3x3 local + outer residual).
- Keep decoder as CNN blocks with skip connections.

**Step 3: Implement Arch A (Cascaded FNO → CNN, `architecture=fno`)**
- FNO stage outputs a coarse patch; CNN refiner outputs final patch.
- Preserve the same output contract (real/imag patches).

**Step 4: Wire into the Torch runner**
- Map `architecture=hybrid` → Arch B (Hybrid U-NO).
- Map `architecture=fno` → Arch A (Cascaded FNO → CNN).

---

### Task 8: Verification & Evidence

**Required tests (per TESTING_GUIDE.md):**
- Unit tests added/modified in this plan.
- Integration marker: `pytest -m integration`

**Commands:**
```bash
pytest tests/test_model_config_architecture.py -v
pytest tests/test_generator_registry.py -v
pytest tests/test_workflows_components.py -k generator -v
pytest tests/torch/test_grid_lines_torch_runner.py -v
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
