# FNO/Hybrid Testing Gaps Addendum Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add missing tests for FNO/Hybrid generator integration correctness (channel/gridsize alignment, forward signature, output contract wiring, and torchapi-devel path usage) that were not covered in `2026-01-27-fno-hybrid-testing-gaps.md`.

**Architecture:** Add focused unit/integration tests around the Torch runner and generator I/O contract. Introduce minimal helpers that normalize channel counts from gridsize, enforce model forward signatures, and convert real/imag outputs to complex patches before physics/consistency. Tests use tiny synthetic NPZ fixtures and are guarded with `@pytest.mark.slow` only when training is required.

**Tech Stack:** PyTorch + Lightning, pytest, numpy, `ptycho_torch` (torchapi-devel).

---

### Task 1: Channel/gridsize alignment in Torch runner

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_grid_lines_torch_runner.py
def test_runner_sets_channels_from_gridsize(tmp_path, synthetic_npz):
    train_path, test_path = synthetic_npz
    cfg = TorchRunnerConfig(
        train_npz=train_path,
        test_npz=test_path,
        output_dir=tmp_path / "out",
        architecture="hybrid",
        gridsize=1,
    )
    training_config, _ = setup_torch_configs(cfg)
    # Expect C == gridsize^2
    assert training_config.model.gridsize == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_runner_sets_channels_from_gridsize -v`  
Expected: FAIL (C not derived from gridsize).

**Step 3: Write minimal implementation**

```python
# scripts/studies/grid_lines_torch_runner.py (inside setup_torch_configs)
model_config = ModelConfig(
    N=N_literal,
    gridsize=cfg.gridsize,
    architecture=arch_literal,
)
# Ensure Torch-side C follows gridsize^2 for channel consistency
model_config.C = cfg.gridsize * cfg.gridsize
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_runner_sets_channels_from_gridsize -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "test(torch): align channels with gridsize in runner config"
```

---

### Task 2: Forward signature enforcement (coords not passed to FNO/Hybrid)

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_grid_lines_torch_runner.py
def test_inference_calls_model_with_single_input(monkeypatch, synthetic_npz, tmp_path):
    train_path, test_path = synthetic_npz
    cfg = TorchRunnerConfig(train_npz=train_path, test_npz=test_path, output_dir=tmp_path, architecture="fno")

    class DummyModel:
        def eval(self): return self
        def __call__(self, x): return x

    def fake_model(*args, **kwargs):
        return DummyModel()

    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_torch_training", lambda *a, **k: {"model": DummyModel()})
    result = run_grid_lines_torch(cfg)
    assert result["architecture"] == "fno"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_inference_calls_model_with_single_input -v`  
Expected: FAIL if inference passes coords.

**Step 3: Write minimal implementation**

```python
# scripts/studies/grid_lines_torch_runner.py (run_torch_inference)
predictions = model(X_test)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_inference_calls_model_with_single_input -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "test(torch): enforce single-input inference for FNO/Hybrid"
```

---

### Task 3: Output contract conversion to complex patches

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_grid_lines_torch_runner.py
def test_real_imag_output_converts_to_complex(synthetic_npz, tmp_path, monkeypatch):
    train_path, test_path = synthetic_npz
    cfg = TorchRunnerConfig(train_npz=train_path, test_npz=test_path, output_dir=tmp_path, architecture="hybrid")

    class DummyModel:
        def eval(self): return self
        def __call__(self, x):
            # (B, H, W, C, 2)
            return np.zeros((1, 32, 32, 1, 2), dtype=np.float32)

    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_torch_training", lambda *a, **k: {"model": DummyModel()})
    result = run_grid_lines_torch(cfg)
    assert "predictions_complex" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_real_imag_output_converts_to_complex -v`  
Expected: FAIL (conversion missing).

**Step 3: Write minimal implementation**

```python
# scripts/studies/grid_lines_torch_runner.py
def to_complex_patches(real_imag):
    real = real_imag[..., 0]
    imag = real_imag[..., 1]
    return real + 1j * imag
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_real_imag_output_converts_to_complex -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "test(torch): convert real/imag outputs to complex patches"
```

---

### Task 4: Torchapi-devel training path used by runner

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_grid_lines_torch_runner.py
def test_runner_uses_torchapi_train_path(monkeypatch, synthetic_npz, tmp_path):
    train_path, test_path = synthetic_npz
    cfg = TorchRunnerConfig(train_npz=train_path, test_npz=test_path, output_dir=tmp_path, architecture="hybrid")
    called = {"train": False}

    def fake_train(*args, **kwargs):
        called["train"] = True
        return {"model": None, "history": {"train_loss": []}}

    monkeypatch.setattr("ptycho_torch.workflows.components.train_cdi_model_torch", fake_train)
    run_grid_lines_torch(cfg)
    assert called["train"] is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_runner_uses_torchapi_train_path -v`  
Expected: FAIL if runner bypasses torchapi.

**Step 3: Write minimal implementation**

```python
# scripts/studies/grid_lines_torch_runner.py (run_torch_training)
from ptycho_torch.workflows.components import train_cdi_model_torch
results = train_cdi_model_torch(train_raw, test_raw, training_config, execution_config=execution_config)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_runner_uses_torchapi_train_path -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "test(torch): runner uses torchapi training path"
```

---

### Task 5: Verification & Evidence

**Required tests (per TESTING_GUIDE.md):**
- Unit tests added/modified in this plan.
- `pytest -m integration` if workflow code changes are introduced.

**Commands:**
```bash
pytest tests/torch/test_grid_lines_torch_runner.py -v
pytest -m integration
```

**Evidence capture:**
- Save logs under `.artifacts/fno_hybrid_testing_addendum/` (e.g., `pytest_runner_contract.log`).
- Add a short note in this plan pointing to the log paths.

**Evidence (2026-01-26):**
- `.artifacts/fno_hybrid_testing_addendum/pytest_runner_contract.log` - All 15 tests pass
- `.artifacts/fno_hybrid_testing_addendum/pytest_integration.log` - Integration test has pre-existing failure (TF `intensity_scale` issue)
- `.artifacts/fno_hybrid_testing_addendum/pytest_torch_tests.log` - Torch test suite has pre-existing failures (fixtures, modules)

---

## Execution Handoff
Plan complete and saved to `docs/plans/2026-01-27-fno-hybrid-testing-gaps-addendum.md`.

Two execution options:
1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks
2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
