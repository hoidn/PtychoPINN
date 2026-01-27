# FNO/Hybrid Correctness Fixes Plan (Revised)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make FNO/Hybrid generators and the Torch grid-lines runner correct and architecture-aligned: proper channel/gridsize propagation, true training via torchapi-devel workflows, stitched global reconstructions before metrics, and documentation updates reflecting the correct flow.

**Architecture:** Use the torchapi-devel training/inference stack: propagate `architecture` into Torch configs, instantiate the correct generator inside `PtychoPINN` (CNN vs FNO/Hybrid), run training via `train_cdi_model_torch`, and run inference via `reconstruct_image_barycentric` on a `PtychoDataset` built from the cached test NPZ directory. Compute metrics on stitched global objects. Remove reliance on registry if torchapi-devel is the base. Add tests that assert stitched metrics and correct channel propagation. Update backlog and grid-lines docs accordingly.

**Tech Stack:** PyTorch + Lightning, pytest, numpy, `ptycho_torch`.

---

### Task 1: Enforce channel/gridsize contract in torchapi config factory

**Files:**
- Modify: `ptycho_torch/config_factory.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write the failing test**

```python
def test_create_training_payload_sets_channels_from_gridsize(synthetic_ptycho_npz, tmp_path):
    from ptycho_torch.config_factory import create_training_payload
    train_npz, _ = synthetic_ptycho_npz
    payload = create_training_payload(
        train_data_file=train_npz,
        output_dir=tmp_path,
        overrides={"n_groups": 4, "gridsize": 1},
    )
    assert payload.pt_data_config.grid_size == (1, 1)
    assert payload.pt_data_config.C == 1
    assert payload.pt_model_config.C_forward == 1
    assert payload.pt_model_config.C_model == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_create_training_payload_sets_channels_from_gridsize -v`  
Expected: FAIL (C/grid_size remain defaults).

**Step 3: Write minimal implementation**

```python
# ptycho_torch/config_factory.py (create_training_payload)
grid_size = overrides.get('grid_size', (overrides.get('gridsize', 1), overrides.get('gridsize', 1)))
C = grid_size[0] * grid_size[1]
overrides['grid_size'] = grid_size
overrides['C'] = C
overrides['C_forward'] = C
overrides['C_model'] = C
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_create_training_payload_sets_channels_from_gridsize -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add ptycho_torch/config_factory.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "fix(torch): derive channels from gridsize in config factory"
```

---

### Task 2: Propagate architecture into Torch config factory

**Files:**
- Modify: `ptycho_torch/config_factory.py`
- Modify: `ptycho_torch/workflows/components.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write the failing test**

```python
def test_training_payload_receives_architecture(monkeypatch, tmp_path):
    from ptycho_torch.workflows import components
    from ptycho.config.config import TrainingConfig, ModelConfig
    cfg = TrainingConfig(model=ModelConfig(N=64, gridsize=1, architecture="hybrid"))
    called = {"arch": None}

    def spy_create_payload(*args, **kwargs):
        called["arch"] = kwargs["overrides"].get("architecture")
        raise RuntimeError("stop")

    monkeypatch.setattr("ptycho_torch.config_factory.create_training_payload", spy_create_payload)
    try:
        components._train_with_lightning(train_container=object(), test_container=None, config=cfg)
    except RuntimeError:
        pass
    assert called["arch"] == "hybrid"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_training_payload_receives_architecture -v`  
Expected: FAIL (architecture not forwarded).

**Step 3: Write minimal implementation**

```python
# ptycho_torch/workflows/components.py (factory_overrides)
factory_overrides = {
    # existing fields...
    "architecture": config.model.architecture,
}
```

```python
# ptycho_torch/config_factory.py
# Ensure update_existing_config sees architecture in overrides
# (PTModelConfig is instantiated then updated)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_training_payload_receives_architecture -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add ptycho_torch/workflows/components.py ptycho_torch/config_factory.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "fix(torch): propagate architecture into torch configs"
```

---

### Task 3: Integrate FNO/Hybrid generator into PtychoPINN

**Files:**
- Modify: `ptycho_torch/model.py`
- Test: `tests/torch/test_fno_integration.py`

**Step 1: Write the failing test**

```python
def test_pinn_uses_fno_generator_when_selected():
    from ptycho_torch.config_params import ModelConfig, DataConfig, TrainingConfig
    from ptycho_torch.model import PtychoPINN
    cfg = ModelConfig(architecture="fno")
    model = PtychoPINN(cfg, DataConfig(), TrainingConfig())
    assert hasattr(model, "generator")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_fno_integration.py::test_pinn_uses_fno_generator_when_selected -v`  
Expected: FAIL (no generator path).

**Step 3: Write minimal implementation**

```python
# ptycho_torch/model.py (inside PtychoPINN.__init__)
if model_config.architecture in ("fno", "hybrid"):
    from ptycho_torch.generators.fno import CascadedFNOGenerator, HybridUNOGenerator
    if model_config.architecture == "fno":
        self.generator = CascadedFNOGenerator(C=data_config.C)
    else:
        self.generator = HybridUNOGenerator(C=data_config.C)
    self.autoencoder = None
else:
    self.autoencoder = Autoencoder(model_config, data_config)
    self.generator = None
```

```python
# ptycho_torch/model.py (inside PtychoPINN.forward / forward_predict)
if self.generator is not None:
    x_pred = self.generator(x)  # (B,H,W,C,2)
    real = x_pred[..., 0]
    imag = x_pred[..., 1]
    x_combined = torch.complex(real, imag).permute(0, 3, 1, 2)  # (B,C,H,W)
    x_amp = torch.abs(x_combined)
    x_phase = torch.angle(x_combined)
else:
    x_amp, x_phase = self.autoencoder(x)
    x_combined = self.combine_complex(x_amp, x_phase)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_fno_integration.py::test_pinn_uses_fno_generator_when_selected -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add ptycho_torch/model.py tests/torch/test_fno_integration.py
git commit -m "feat(torch): integrate FNO/Hybrid generator into PtychoPINN"
```

---

### Task 4: Route training through torchapi Lightning path (no registry)

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write the failing test**

```python
def test_runner_uses_torchapi_lightning(monkeypatch, synthetic_npz, tmp_path):
    from unittest.mock import MagicMock
    train_npz, test_npz = synthetic_npz
    cfg = TorchRunnerConfig(train_npz=train_npz, test_npz=test_npz, output_dir=tmp_path, architecture="hybrid")
    called = {"train": False}

    def fake_train(train_container, test_container, config, execution_config=None, overrides=None):
        called["train"] = True
        assert "X" in train_container
        assert "coords_nominal" in train_container
        return {"history": {"train_loss": []}, "models": {"diffraction_to_obj": MagicMock(), "autoencoder": {}}}

    monkeypatch.setattr("ptycho_torch.workflows.components._train_with_lightning", fake_train)
    run_grid_lines_torch(cfg)
    assert called["train"] is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_runner_uses_torchapi_lightning -v`  
Expected: FAIL (runner still uses registry/scaffold path).

**Step 3: Write minimal implementation**

```python
# scripts/studies/grid_lines_torch_runner.py (run_torch_training)
from ptycho_torch.workflows.components import _train_with_lightning

def _reshape_coords(coords, n_samples, c_channels):
    if coords.ndim == 2:
        coords = coords.reshape(n_samples, c_channels, 2).transpose(0, 2, 1)
        coords = coords[:, None, :, :]
    return coords

X = train_data["diffraction"]
n_samples = X.shape[0]
c_channels = X.shape[-1] if X.ndim == 4 else 1
coords = _reshape_coords(train_data["coords_nominal"], n_samples, c_channels)
probe = train_data.get("probeGuess")
if probe is None:
    probe = np.ones((cfg.N, cfg.N), dtype=np.complex64)

train_container = {
    "X": X,
    "coords_nominal": coords,
    "probe": probe,
}
test_container = None
if test_data:
    X_te = test_data["diffraction"]
    n_te = X_te.shape[0]
    c_te = X_te.shape[-1] if X_te.ndim == 4 else 1
    coords_te = _reshape_coords(test_data["coords_nominal"], n_te, c_te)
    test_probe = test_data.get("probeGuess", probe)
    test_container = {
        "X": X_te,
        "coords_nominal": coords_te,
        "probe": test_probe,
    }

results = _train_with_lightning(train_container, test_container, training_config, execution_config=execution_config)
results["generator"] = cfg.architecture
```

**Step 3b: Update existing tests to match new training path**

- Remove/replace `TestTorchTrainingPath::test_runner_uses_generator_registry` and related scaffold expectations.
- Keep coverage by asserting `_train_with_lightning` is called and containers are shaped correctly.

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_runner_uses_torchapi_lightning -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "fix(torch): route runner training through torchapi workflow"
```

---

### Task 4b: Align inference signature + add batching (OOM guard)

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write the failing test**

```python
def test_runner_inference_uses_forward_predict_signature(monkeypatch, synthetic_npz, tmp_path):
    from unittest.mock import MagicMock
    train_npz, test_npz = synthetic_npz
    cfg = TorchRunnerConfig(train_npz=train_npz, test_npz=test_npz, output_dir=tmp_path, architecture="fno")
    called = {"args": None}

    class SpyModel:
        def eval(self): return self
        def forward_predict(self, x, positions, probe, input_scale_factor):
            called["args"] = (x, positions, probe, input_scale_factor)
            return x.new_zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

    test_data = load_cached_dataset(test_npz)
    run_torch_inference(SpyModel(), test_data, cfg)
    assert called["args"] is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_runner_inference_uses_forward_predict_signature -v`  
Expected: FAIL (runner calls model(X) without signature args).

**Step 3: Write minimal implementation**

```python
# scripts/studies/grid_lines_torch_runner.py (run_torch_inference)
# Build positions/probe + scale and call forward_predict in batches
```

**Step 4: Add batching**

- Add `infer_batch_size` to `TorchRunnerConfig`.
- Use a simple for‑loop over slices of X/coords and aggregate predictions.

**Step 5: Run test to verify it passes**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_runner_inference_uses_forward_predict_signature -v`  
Expected: PASS

---

### Task 4c: Expose FNO hyperparameters via config/runner

**Files:**
- Modify: `ptycho/config/config.py`
- Modify: `ptycho_torch/config_params.py`
- Modify: `ptycho_torch/config_factory.py`
- Modify: `ptycho_torch/model.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Test: `tests/torch/test_fno_integration.py`

**Step 1: Add config fields**
- `fno_modes`, `fno_width`, `fno_blocks`, `fno_cnn_blocks` (with safe defaults).

**Step 2: Thread through factory/runner**
- Ensure overrides propagate into PT configs.
- Runner CLI allows overrides for quick tuning.

**Step 3: Use config values when instantiating generators**
- Pass the values into `CascadedFNOGenerator`/`HybridUNOGenerator`.

**Step 4: Add a unit test**
- Instantiate `PtychoPINN` with non‑default fno config and assert generator attributes match.

### Task 5: Stitch patches before metrics (physics correctness)

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Test: `tests/torch/test_fno_integration.py`

**Step 1: Write the failing test**

```python
@pytest.mark.slow
def test_metrics_use_stitched_global_object(synthetic_npz, tmp_path):
    from scripts.studies.grid_lines_torch_runner import run_grid_lines_torch, TorchRunnerConfig
    train_npz, test_npz = synthetic_npz
    cfg = TorchRunnerConfig(train_npz=train_npz, test_npz=test_npz, output_dir=tmp_path, architecture="hybrid", epochs=1)
    result = run_grid_lines_torch(cfg)
    # Expect metrics to be computed on a stitched global object
    assert "stitched_object" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_fno_integration.py::test_metrics_use_stitched_global_object -v`  
Expected: FAIL (no stitching path).

**Step 3: Write minimal implementation**

```python
# scripts/studies/grid_lines_torch_runner.py
from tempfile import TemporaryDirectory
import shutil
from ptycho_torch.dataloader import PtychoDataset
from ptycho_torch.reassembly import reconstruct_image_barycentric
from ptycho_torch.config_params import DataConfig, ModelConfig as PTModelConfig, TrainingConfig as PTTrainingConfig, InferenceConfig as PTInferenceConfig

# Create torch configs aligned to gridsize/N
pt_data = DataConfig(N=cfg.N, grid_size=(cfg.gridsize, cfg.gridsize))
pt_model = PTModelConfig(architecture=cfg.architecture)
pt_train = PTTrainingConfig(epochs=cfg.epochs, batch_size=cfg.batch_size, learning_rate=cfg.learning_rate)
pt_infer = PTInferenceConfig()

# Build dataset from a temp dir containing only the test NPZ
with TemporaryDirectory() as tmp_dir:
    tmp_dir_path = Path(tmp_dir)
    shutil.copy(cfg.test_npz, tmp_dir_path / Path(cfg.test_npz).name)
    ptycho_dset = PtychoDataset(str(tmp_dir_path), pt_model, pt_data, remake_map=True)

    # Reconstruct global object via torchapi reassembly
    stitched_object, _ = reconstruct_image_barycentric(
        model=results["models"]["diffraction_to_obj"],
        ptycho_dset=ptycho_dset,
        training_config=pt_train,
        data_config=pt_data,
        model_config=pt_model,
        inference_config=pt_infer,
        gpu_ids=None,
        use_mixed_precision=False,
        verbose=False
    )

# Use stitched object for metrics
stitched_np = stitched_object.cpu().numpy()
if ground_truth.ndim == 3 and ground_truth.shape[0] == 1:
    ground_truth = ground_truth[0]
metrics = compute_metrics(stitched_np, ground_truth, f"pinn_{cfg.architecture}")
result_dict["stitched_object"] = stitched_np
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_fno_integration.py::test_metrics_use_stitched_global_object -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py tests/torch/test_fno_integration.py
git commit -m "fix(torch): stitch patches before metrics in runner"
```

---

### Task 6: Use real loss history callback in integration tests

**Files:**
- Modify: `tests/torch/test_fno_lightning_integration.py`
- Test: `tests/torch/test_fno_lightning_integration.py`

**Step 1: Write the failing test**

```python
def test_loss_history_callback_is_wired(monkeypatch):
    from ptycho_torch.workflows import components
    assert hasattr(components, "_train_with_lightning")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_fno_lightning_integration.py::test_loss_history_callback_is_wired -v`  
Expected: FAIL (test not yet real).

**Step 3: Write minimal implementation**

Update tests to exercise the actual callback wired in `_train_with_lightning` by running a 2‑epoch fit with small synthetic data (reuse existing fixture). Remove the local callback redefinition.

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_fno_lightning_integration.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/torch/test_fno_lightning_integration.py
git commit -m "test(torch): validate real loss history callback wiring"
```

---

### Task 7: Documentation updates

**Files:**
- Modify: `docs/backlog/FNO_HYBRID_TESTING_GAPS.md`
- Modify: `docs/plans/2026-01-27-grid-lines-workflow.md`

**Step 1: Update backlog status**
- Mark gaps 1–3 as “In Progress” or “Covered by runner stitching path” once tests pass.
- Add a note that metrics are now computed on stitched global reconstructions via `reconstruct_image_barycentric`.

**Step 2: Update grid-lines workflow doc**
- State that Torch runner now stitches before metrics and uses torchapi-devel `PtychoDataset` + `reconstruct_image_barycentric`.

**Step 3: Commit**

```bash
git add docs/backlog/FNO_HYBRID_TESTING_GAPS.md docs/plans/2026-01-27-grid-lines-workflow.md
git commit -m "docs: update FNO/hybrid testing status and grid-lines runner notes"
```

---

### Task 8: Verification & Evidence

**Required tests (per TESTING_GUIDE.md):**
- Unit tests added/modified in this plan.
- `pytest -m integration` if workflow code is touched.

**Commands:**
```bash
pytest tests/torch/test_grid_lines_torch_runner.py -v
pytest tests/torch/test_fno_integration.py -v
pytest tests/torch/test_fno_lightning_integration.py -v
pytest -m integration
```

**Evidence capture:**
- Save logs under `.artifacts/fno_hybrid_correctness/` (e.g., `pytest_runner_correctness.log`).
- Add a short note in this plan pointing to the log paths.

---

## Execution Handoff
Plan complete and saved to `docs/plans/2026-01-27-fno-hybrid-correctness-fixes.md`.

Two execution options:
1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks
2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
