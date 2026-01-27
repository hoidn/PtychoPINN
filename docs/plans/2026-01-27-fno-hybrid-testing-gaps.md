# FNO/Hybrid Testing Gaps Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the testing gaps in `docs/backlog/FNO_HYBRID_TESTING_GAPS.md` by adding end-to-end Torch tests for FNO/Hybrid generators, Lightning integration coverage, and neuraloperator validation.

**Architecture:** Add small synthetic NPZ fixtures, wire a minimal loss-history callback in `_train_with_lightning`, and implement integration tests that train + infer on tiny data. Add optional neuraloperator tests guarded by `HAS_NEURALOPERATOR`. Keep tests lightweight with `@pytest.mark.slow` where needed. **Note:** If `ptycho_torch` is replaced with `origin/torchapi-devel` (per modular generator plan), update imports to the runner-local factory (no registry) and keep test contracts unchanged.

**Tech Stack:** PyTorch + Lightning, pytest, numpy, neuraloperator (optional), `ptycho_torch`.

---

### Task 1: Add synthetic NPZ fixture for Torch FNO tests

**Files:**
- Modify: `tests/torch/conftest.py`
- Test: `tests/torch/test_fno_integration.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_fno_integration.py
import numpy as np


def test_synthetic_npz_fixture_contract(synthetic_ptycho_npz):
    train_npz, _ = synthetic_ptycho_npz
    data = dict(np.load(train_npz))
    for key in ("diffraction", "Y_I", "Y_phi", "coords_nominal", "coords_true"):
        assert key in data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_fno_integration.py::test_synthetic_npz_fixture_contract -v`  
Expected: FAIL (fixture missing).

**Step 3: Write minimal implementation**

```python
# tests/torch/conftest.py
@pytest.fixture
def synthetic_ptycho_npz(tmp_path):
    N = 32
    n_samples = 4
    gridsize = 1
    data = {
        "diffraction": np.random.rand(n_samples, N, N, gridsize**2).astype(np.float32),
        "Y_I": np.random.rand(n_samples, N, N, gridsize**2).astype(np.float32),
        "Y_phi": np.random.rand(n_samples, N, N, gridsize**2).astype(np.float32),
        "coords_nominal": np.random.rand(n_samples * gridsize**2, 2).astype(np.float32),
        "coords_true": np.random.rand(n_samples * gridsize**2, 2).astype(np.float32),
        "probeGuess": np.ones((N, N), dtype=np.complex64),
        "objectGuess": np.ones((N * 2, N * 2), dtype=np.complex64),
    }
    train_path = tmp_path / "train.npz"
    test_path = tmp_path / "test.npz"
    np.savez(train_path, **data)
    np.savez(test_path, **data)
    return train_path, test_path
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_fno_integration.py::test_synthetic_npz_fixture_contract -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/torch/conftest.py tests/torch/test_fno_integration.py
git commit -m "test(torch): add synthetic NPZ fixture for FNO integration"
```

---

### Task 2: Add loss history callback for Lightning training

**Files:**
- Modify: `ptycho_torch/workflows/components.py`
- Test: `tests/torch/test_fno_lightning_integration.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_fno_lightning_integration.py
import pytest
import numpy as np
from ptycho.config.config import TrainingConfig, ModelConfig, PyTorchExecutionConfig
from ptycho.raw_data import RawData
from ptycho_torch.workflows.components import train_cdi_model_torch


@pytest.mark.slow
def test_train_history_collects_epochs(synthetic_ptycho_npz, tmp_path):
    train_npz, _ = synthetic_ptycho_npz
    train_data = RawData.from_file(str(train_npz))
    cfg = TrainingConfig(
        model=ModelConfig(N=32, gridsize=1, architecture="hybrid"),
        train_data_file=train_npz,
        test_data_file=None,
        nepochs=2,
        batch_size=2,
        backend="pytorch",
        output_dir=tmp_path,
    )
    exec_cfg = PyTorchExecutionConfig(logger_backend=None, enable_checkpointing=False)
    results = train_cdi_model_torch(train_data, None, cfg, execution_config=exec_cfg)
    history = results["history"]["train_loss"]
    assert len(history) >= 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_fno_lightning_integration.py::test_train_history_collects_epochs -v`  
Expected: FAIL (history only contains latest metric).

**Step 3: Write minimal implementation**

```python
# ptycho_torch/workflows/components.py (inside _train_with_lightning)
class _LossHistoryCallback(L.Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "train_loss" in metrics:
            self.train_loss.append(float(metrics["train_loss"]))

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "val_loss" in metrics:
            self.val_loss.append(float(metrics["val_loss"]))

# Add to callbacks list before Trainer()
loss_history_cb = _LossHistoryCallback()
callbacks.append(loss_history_cb)

# After training, build history from callback:
history = {
    "train_loss": loss_history_cb.train_loss,
    "val_loss": loss_history_cb.val_loss if test_container is not None else None,
}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_fno_lightning_integration.py::test_train_history_collects_epochs -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add ptycho_torch/workflows/components.py tests/torch/test_fno_lightning_integration.py
git commit -m "feat(torch): add loss history callback for Lightning training"
```

---

### Task 3: End-to-end training test for FNO/Hybrid

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Test: `tests/torch/test_fno_integration.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_fno_integration.py
import numpy as np
import pytest
from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig, run_grid_lines_torch


@pytest.mark.slow
def test_fno_end_to_end_training_decreases_loss(synthetic_ptycho_npz, tmp_path):
    train_npz, test_npz = synthetic_ptycho_npz
    cfg = TorchRunnerConfig(
        train_npz=train_npz,
        test_npz=test_npz,
        output_dir=tmp_path,
        architecture="hybrid",
        epochs=2,
        batch_size=2,
        N=32,
        gridsize=1,
    )
    results = run_grid_lines_torch(cfg)
    history = results["history"]["train_loss"]
    assert len(history) >= 2
    assert history[-1] <= history[0]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_fno_integration.py::test_fno_end_to_end_training_decreases_loss -v`  
Expected: FAIL (runner uses scaffold path).

**Step 3: Write minimal implementation**

```python
# scripts/studies/grid_lines_torch_runner.py
from ptycho.raw_data import RawData
from ptycho_torch.workflows.components import train_cdi_model_torch

def run_torch_training(cfg, train_data, test_data):
    training_config, execution_config = setup_torch_configs(cfg)
    # Use RawData to stay aligned with Torch workflow
    train_raw = RawData.from_file(str(cfg.train_npz))
    test_raw = RawData.from_file(str(cfg.test_npz)) if cfg.test_npz else None
    results = train_cdi_model_torch(train_raw, test_raw, training_config, execution_config=execution_config)
    results["generator"] = cfg.architecture
    return results
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_fno_integration.py::test_fno_end_to_end_training_decreases_loss -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py tests/torch/test_fno_integration.py
git commit -m "test(torch): add end-to-end FNO training integration"
```

---

### Task 4: Reconstruction quality test vs CNN baseline

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Test: `tests/torch/test_fno_reconstruction_quality.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_fno_reconstruction_quality.py
import numpy as np
import pytest
from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig, run_grid_lines_torch


@pytest.mark.slow
def test_fno_quality_within_baseline(synthetic_ptycho_npz, tmp_path):
    train_npz, test_npz = synthetic_ptycho_npz
    base_cfg = TorchRunnerConfig(train_npz, test_npz, tmp_path, architecture="cnn", epochs=2, batch_size=2, N=32)
    fno_cfg = TorchRunnerConfig(train_npz, test_npz, tmp_path, architecture="hybrid", epochs=2, batch_size=2, N=32)
    base_results = run_grid_lines_torch(base_cfg)
    fno_results = run_grid_lines_torch(fno_cfg)
    assert fno_results["metrics"]["ssim"] >= 0.8 * base_results["metrics"]["ssim"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_fno_reconstruction_quality.py -v`  
Expected: FAIL (runner lacks cnn path + metrics).

**Step 3: Write minimal implementation**

```python
# scripts/studies/grid_lines_torch_runner.py
def compute_metrics(predictions, ground_truth, label):
    mse = float(np.mean(np.abs(predictions - ground_truth) ** 2))
    # Simple SSIM proxy for tiny tests (avoid heavy deps in unit tests)
    ssim = 1.0 / (1.0 + mse)
    return {"mse": mse, "ssim": ssim}

# Add cnn support in run_torch_training or route to existing torch CNN model.
# Ensure run_grid_lines_torch returns {"metrics": metrics, "history": ..., "architecture": ...}.
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_fno_reconstruction_quality.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py tests/torch/test_fno_reconstruction_quality.py
git commit -m "test(torch): add reconstruction quality comparison vs cnn baseline"
```

---

### Task 5: Neuraloperator integration test

**Files:**
- Create: `tests/torch/test_fno_neuraloperator.py`
- Modify: `setup.py` (extras_require)

**Step 1: Write the failing test**

```python
# tests/torch/test_fno_neuraloperator.py
import pytest
import torch
from ptycho_torch.generators.fno import HAS_NEURALOPERATOR, SpectralConv2d


@pytest.mark.optional
@pytest.mark.skipif(not HAS_NEURALOPERATOR, reason="neuraloperator not installed")
def test_spectral_conv_with_neuraloperator():
    layer = SpectralConv2d(in_channels=4, out_channels=4, modes1=4, modes2=4)
    x = torch.randn(2, 4, 16, 16)
    y = layer(x)
    assert y.shape == x.shape
```

**Step 2: Run test to verify it fails (if neuraloperator missing)**

Run: `pytest tests/torch/test_fno_neuraloperator.py -v`  
Expected: SKIP if neuraloperator missing; FAIL if import path is wrong.

**Step 3: Write minimal implementation**

```python
# setup.py
extras_require = {
    "torch": [
        "lightning",
        "mlflow",
        "tensordict",
        "neuraloperator",
    ],
}
```

**Step 4: Run test to verify it passes when neuraloperator is installed**

Run (with neuraloperator installed):  
`pytest tests/torch/test_fno_neuraloperator.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add setup.py tests/torch/test_fno_neuraloperator.py
git commit -m "test(torch): add neuraloperator coverage and dependency"
```

---

### Task 6: Verification & Evidence

**Required tests (per TESTING_GUIDE.md):**
- Unit tests added/modified in this plan.
- `pytest -m integration` (if workflow code is touched).

**Commands:**
```bash
pytest tests/torch/test_fno_integration.py -v
pytest tests/torch/test_fno_lightning_integration.py -v
pytest tests/torch/test_fno_reconstruction_quality.py -v
pytest tests/torch/test_fno_neuraloperator.py -v
pytest -m integration
```

**Evidence capture:**
- Save logs under `.artifacts/fno_hybrid_testing/` (e.g., `pytest_fno_integration.log`).
- Add a short note in this plan pointing to the log paths.

---

## Execution Handoff
Plan complete and saved to `docs/plans/2026-01-27-fno-hybrid-testing-gaps.md`.

Two execution options:
1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks
2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
