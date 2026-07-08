# MLflow Recon Logging Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden MLflow recon logging with robust dataloader handling, correct scaling, best-effort stitching, and CLI plumbing for recon logging flags.

**Architecture:** Update the Lightning recon logging callback to be resilient to dataloader shapes, use RMS scaling for diffraction comparisons, and support best-effort stitched outputs via metadata-driven or fallback stitching. Wire recon logging flags through Torch CLI entry points so runs can enable logging without code changes.

**Tech Stack:** Python, PyTorch Lightning, NumPy, MLflow logger, pytest.

---

### Task 1: Patch Logging Robustness + Scaling Parity

**Files:**
- Modify: `ptycho_torch/workflows/recon_logging.py`
- Modify: `tests/torch/test_mlflow_recon_logging.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_mlflow_recon_logging.py

def test_handles_val_dataloader_list(self):
    dataset = FakeDataset(n=4, supervised=False)
    val_dl = FakeValDataloader(dataset)
    trainer = FakeTrainer(epoch=4, val_dl=[val_dl])
    module = FakeModule()

    cb = PtychoReconLoggingCallback(every_n_epochs=5, num_patches=1)
    cb.on_validation_epoch_end(trainer, module)

    assert trainer.logger.experiment.log_artifact.called


def test_uses_rms_scale_for_output(self):
    class CaptureModule(FakeModule):
        def __init__(self):
            super().__init__()
            self.calls = []
        def forward(self, x, positions, probe, input_scale, output_scale, experiment_ids):
            self.calls.append((input_scale, output_scale))
            return x, torch.abs(x), torch.zeros_like(x)

    dataset = FakeDataset(n=2, supervised=False)
    # override scales so rms != physics
    data, probe, scale = dataset[0]
    data["rms_scaling_constant"] = torch.tensor([2.0])
    data["physics_scaling_constant"] = torch.tensor([5.0])
    dataset.__getitem__ = lambda idx: (data, probe, scale)

    val_dl = FakeValDataloader(dataset)
    trainer = FakeTrainer(epoch=4, val_dl=val_dl)
    module = CaptureModule()

    cb = PtychoReconLoggingCallback(every_n_epochs=5, num_patches=1)
    cb.on_validation_epoch_end(trainer, module)

    assert module.calls
    input_scale, output_scale = module.calls[0]
    assert torch.allclose(input_scale, output_scale)


def test_multi_channel_patch_logging(self):
    class MultiChannelDataset(FakeDataset):
        def __getitem__(self, idx):
            data, probe, scale = super().__getitem__(idx)
            data["images"] = torch.randn(2, 8, 8)
            return data, probe, scale

    dataset = MultiChannelDataset(n=2, supervised=False)
    val_dl = FakeValDataloader(dataset)
    trainer = FakeTrainer(epoch=4, val_dl=val_dl)
    module = FakeModule()

    cb = PtychoReconLoggingCallback(every_n_epochs=5, num_patches=1)
    cb.on_validation_epoch_end(trainer, module)

    assert trainer.logger.experiment.log_artifact.called
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_mlflow_recon_logging.py::TestPatchLogging::test_handles_val_dataloader_list -v`
Expected: FAIL with AttributeError on `val_dataloaders` list

**Step 3: Write minimal implementation**

```python
# ptycho_torch/workflows/recon_logging.py

def _select_dataloader(self, dataloader):
    if isinstance(dataloader, (list, tuple)):
        return dataloader[0] if dataloader else None
    loaders = getattr(dataloader, "loaders", None)
    if isinstance(loaders, dict):
        return next(iter(loaders.values()), None)
    if isinstance(loaders, (list, tuple)):
        return loaders[0] if loaders else None
    return dataloader

# in on_validation_epoch_end
val_dl = self._select_dataloader(trainer.val_dataloaders) or self._select_dataloader(trainer.train_dataloader)

# in _log_patches
pred_diff, amp_pred, phase_pred = pl_module(images, coords, probe_t, rms_scale, rms_scale, exp_ids)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_mlflow_recon_logging.py::TestPatchLogging::test_handles_val_dataloader_list -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/torch/test_mlflow_recon_logging.py ptycho_torch/workflows/recon_logging.py
git commit -m "fix: harden recon patch logging (tests: test_mlflow_recon_logging)"
```

---

### Task 2: Best-effort Stitched Logging (Metadata + Fallback)

**Files:**
- Modify: `ptycho_torch/workflows/recon_logging.py`
- Modify: `tests/torch/test_mlflow_recon_logging.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_mlflow_recon_logging.py

def test_stitched_logging_uses_fallback_reassemble(monkeypatch):
    dataset = FakeDataset(n=4, supervised=False)
    val_dl = FakeValDataloader(dataset)
    trainer = FakeTrainer(epoch=4, val_dl=val_dl)
    module = FakeModule()

    called = {"count": 0}
    def fake_reassemble(patches, config, **kwargs):
        called["count"] += 1
        return np.zeros((1, 8, 8, 1), dtype=np.float32)

    monkeypatch.setattr("ptycho.image.stitching.reassemble_patches", fake_reassemble)

    cb = PtychoReconLoggingCallback(every_n_epochs=5, num_patches=1, log_stitch=True)
    cb.on_validation_epoch_end(trainer, module)

    assert called["count"] > 0
    assert trainer.logger.experiment.log_artifact.called
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_mlflow_recon_logging.py::test_stitched_logging_uses_fallback_reassemble -v`
Expected: FAIL because reassemble_patches not used

**Step 3: Write minimal implementation**

```python
# ptycho_torch/workflows/recon_logging.py
from ptycho.image.stitching import reassemble_patches
from ptycho_torch import helper as torch_helper

# inside _log_stitched
if offsets_xy is not None:
    amp_stitched = torch_helper.reassemble_patches_position_real(
        amp_pred_t, offsets_xy, data_cfg, model_cfg, padded_size=M
    )[0]
else:
    config = {
        "N": N,
        "gridsize": gridsize,
        "offset": offset,
        "outer_offset_test": outer_offset_test,
        "nimgs_test": nimgs_test,
    }
    amp_stitched = reassemble_patches(amp_pred[..., np.newaxis], config, norm_Y_I=norm_Y_I, part="amp")
    phase_stitched = reassemble_patches(phase_pred[..., np.newaxis], config, norm_Y_I=norm_Y_I, part="phase")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_mlflow_recon_logging.py::test_stitched_logging_uses_fallback_reassemble -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/torch/test_mlflow_recon_logging.py ptycho_torch/workflows/recon_logging.py
git commit -m "feat: best-effort stitched recon logging (tests: test_mlflow_recon_logging)"
```

---

### Task 3: CLI Plumbing for Recon Logging Flags

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `scripts/training/train.py`
- Modify: `ptycho_torch/cli/shared.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_grid_lines_torch_runner.py

def test_runner_propagates_recon_log_max_stitch_samples(self, tmp_path):
    cfg = TorchRunnerConfig(
        train_npz=tmp_path / "train.npz",
        test_npz=tmp_path / "test.npz",
        output_dir=tmp_path / "out",
        architecture="fno",
        recon_log_max_stitch_samples=8,
    )
    _, exec_cfg = setup_torch_configs(cfg)
    assert exec_cfg.recon_log_max_stitch_samples == 8
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_runner_propagates_recon_log_max_stitch_samples -v`
Expected: FAIL (field not present)

**Step 3: Write minimal implementation**

```python
# scripts/studies/grid_lines_torch_runner.py
parser.add_argument("--torch-logger", choices=["csv", "tensorboard", "mlflow", "none"], default=None)
parser.add_argument("--torch-recon-log-every-n-epochs", type=int, default=None)
parser.add_argument("--torch-recon-log-num-patches", type=int, default=4)
parser.add_argument("--torch-recon-log-fixed-indices", type=int, nargs='+', default=None)
parser.add_argument("--torch-recon-log-stitch", action="store_true", default=False)
parser.add_argument("--torch-recon-log-max-stitch-samples", type=int, default=None)

# TorchRunnerConfig fields + setup_torch_configs include recon_log_max_stitch_samples
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_runner_propagates_recon_log_max_stitch_samples -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py scripts/training/train.py ptycho_torch/cli/shared.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "feat: add recon logging CLI flags (tests: test_grid_lines_torch_runner)"
```

---

### Task 4: Required Validation

**Files:**
- None

**Step 1: Run recon logging unit tests**

Run: `pytest tests/torch/test_mlflow_recon_logging.py -v`
Expected: PASS

**Step 2: Run grid-lines runner unit tests**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_runner_propagates_recon_log_max_stitch_samples -v`
Expected: PASS

**Step 3: Run integration marker (required for workflow changes)**

Run: `pytest -m integration -v`
Expected: PASS

**Step 4: Commit (if needed for doc-only updates)**

```bash
git add docs/workflows/pytorch.md docs/TESTING_GUIDE.md docs/development/TEST_SUITE_INDEX.md
git commit -m "docs: update recon logging workflow docs"
```
