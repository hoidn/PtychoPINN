# Torch Runner Stitching Metrics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ensure `grid_lines_torch_runner.py` stitches patch predictions to full objects before calling `eval_reconstruction`, eliminating shape mismatch errors during metrics.

**Architecture:** Reuse the existing TF-compatible stitching helper (`ptycho.workflows.grid_lines_workflow.stitch_predictions`) and dataset metadata to set stitching params (`nimgs_test`, `outer_offset_test`, `N`, `gridsize`). Load NPZ metadata via `MetadataManager.load_with_metadata` and gate stitching based on prediction vs ground-truth spatial shape.

**Tech Stack:** Python, NumPy, PyTorch, pytest

**Note:** User requested no worktree usage; plan executes directly on `fno2`.

---

### Task 1: Add failing test for stitched metrics path

**Files:**
- Modify: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write failing test**

Add a test that forces metrics to use stitching and asserts `eval_reconstruction` sees matching shapes:

```python
import json
import numpy as np
import pytest
from ptycho.config.config import TrainingConfig, ModelConfig
from ptycho.metadata import MetadataManager


def test_metrics_stitch_predictions_to_ground_truth(monkeypatch, synthetic_npz, tmp_path):
    train_path, test_path = synthetic_npz

    # Overwrite test NPZ with metadata and norm_Y_I
    cfg = TrainingConfig(model=ModelConfig(N=64, gridsize=1))
    metadata = MetadataManager.create_metadata(
        cfg,
        script_name="test_grid_lines_torch_runner",
        nimgs_test=1,
        outer_offset_test=20,
    )
    data = dict(np.load(test_path, allow_pickle=True))
    data["norm_Y_I"] = np.array(1.0, dtype=np.float32)
    MetadataManager.save_with_metadata(str(test_path), data, metadata)

    # Spy on eval_reconstruction to assert shapes match
    called = {"ok": False}

    def fake_eval(stitched_obj, ground_truth_obj, label=""):
        assert stitched_obj.shape[0] == ground_truth_obj.shape[0]
        assert stitched_obj.shape[1] == ground_truth_obj.shape[1]
        called["ok"] = True
        return {"mse": 0.0}

    monkeypatch.setattr("ptycho.evaluation.eval_reconstruction", fake_eval)

    from scripts.studies.grid_lines_torch_runner import (
        TorchRunnerConfig,
        run_grid_lines_torch,
    )

    cfg = TorchRunnerConfig(
        train_npz=train_path,
        test_npz=test_path,
        output_dir=tmp_path,
        architecture="fno",
        epochs=1,
    )

    # Mock training + inference to return patch predictions
    def fake_train(*args, **kwargs):
        return {"model": None, "history": {}, "generator": "fno", "scaffold": True}

    def fake_infer(*args, **kwargs):
        # (B, H, W, C, 2) real/imag patches
        return np.random.rand(1, 64, 64, 1, 2).astype(np.float32)

    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_torch_training", fake_train)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_torch_inference", fake_infer)

    run_grid_lines_torch(cfg)
    assert called["ok"] is True
```

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/torch/test_grid_lines_torch_runner.py::test_metrics_stitch_predictions_to_ground_truth -v
```
Expected: FAIL (shape assertion from eval_reconstruction).

---

### Task 2: Load metadata + stitch predictions before metrics

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`

**Step 1: Add metadata-aware loader**

```python
from ptycho.metadata import MetadataManager


def load_cached_dataset_with_metadata(npz_path: Path):
    data, metadata = MetadataManager.load_with_metadata(str(npz_path))
    required_keys = ['diffraction', 'Y_I', 'Y_phi', 'coords_nominal']
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required key '{key}' in {npz_path}")
    return data, metadata
```

**Step 2: Add stitch helper**

```python
from ptycho import params as p
from ptycho.workflows.grid_lines_workflow import stitch_predictions


def _configure_stitching_params(cfg, metadata):
    if not metadata:
        raise ValueError("Missing metadata; cannot stitch predictions for metrics.")
    add = metadata.get("additional_parameters", {})
    nimgs_test = add.get("nimgs_test")
    outer_offset_test = add.get("outer_offset_test")
    if nimgs_test is None or outer_offset_test is None:
        raise ValueError("Metadata missing nimgs_test/outer_offset_test for stitching.")

    p.cfg["N"] = cfg.N
    p.cfg["gridsize"] = cfg.gridsize
    p.set("nimgs_test", nimgs_test)
    p.set("outer_offset_test", outer_offset_test)


def _stitch_for_metrics(pred_complex, cfg, metadata, norm_Y_I):
    _configure_stitching_params(cfg, metadata)
    return stitch_predictions(pred_complex, float(norm_Y_I), part="complex")
```

**Step 3: Apply stitching in run_grid_lines_torch**

- Load test data with metadata
- Convert real/imag to complex as already done
- If prediction spatial dims don’t match `ground_truth`, stitch

```python
train_data = load_cached_dataset(cfg.train_npz)

test_data, test_meta = load_cached_dataset_with_metadata(cfg.test_npz)
...

pred_for_metrics = predictions_complex if predictions_complex is not None else predictions
if pred_for_metrics.ndim >= 3:
    pred_h, pred_w = pred_for_metrics.shape[-3], pred_for_metrics.shape[-2]
    gt_h, gt_w = ground_truth.shape[-2], ground_truth.shape[-1]
    if (pred_h, pred_w) != (gt_h, gt_w):
        norm_Y_I = test_data.get("norm_Y_I", 1.0)
        pred_for_metrics = _stitch_for_metrics(pred_for_metrics, cfg, test_meta, norm_Y_I)
```

**Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/torch/test_grid_lines_torch_runner.py::test_metrics_stitch_predictions_to_ground_truth -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "fix(torch-runner): stitch predictions before metrics"
```

---

### Task 3: Update testing docs

**Files:**
- Modify: `docs/TESTING_GUIDE.md`
- Modify: `docs/development/TEST_SUITE_INDEX.md`

**Step 1: Document new test selector**

Add to PyTorch test list:
```
pytest tests/torch/test_grid_lines_torch_runner.py::test_metrics_stitch_predictions_to_ground_truth -v
```

**Step 2: Update test index entry**

Update `test_grid_lines_torch_runner.py` row description to mention stitched-metrics coverage.

**Step 3: Commit**

```bash
git add docs/TESTING_GUIDE.md docs/development/TEST_SUITE_INDEX.md
git commit -m "docs: document torch runner stitched-metrics test"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-01-27-torch-runner-stitching-metrics.md`.

Two execution options:
1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks
2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
