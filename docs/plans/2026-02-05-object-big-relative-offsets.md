# Object-Big Relative Offsets Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ensure PyTorch `object_big` uses TF-style relative offsets by making `coords_relative` explicit at the data boundary (grid-lines runner + metadata) and enforcing a hard guard that rejects ambiguous coords, with regression tests and integration verification.

**Architecture:** Introduce a small coords helper that reproduces TF `get_relative_coords` (mean over C + `local_offset_sign=-1`). Update the grid-lines workflow metadata to label coords as relative, and update the PyTorch grid-lines runner to select `coords_relative` explicitly (or derive it when metadata says coords are nominal). Add a hard guard in the Lightning dataloader: if `object_big=True` and `coords_relative` is missing, raise with actionable guidance. Keep physics code unchanged; normalize before `ForwardModel` is invoked. Add unit tests for the helper, runner selection logic, and guard behavior, then re-run integration and parity checks.

**Tech Stack:** Python, PyTorch, pytest.

---

### Task 1: Add failing unit tests for coords-relative contract

**Files:**
- Create: `tests/torch/test_coords_relative_contract.py`

**Step 1: Write the failing test**

```python
import numpy as np
import pytest

from ptycho_torch import coords as coords_mod


def test_coords_relative_from_nominal_c1_is_zero():
    coords = np.array([[[[10.0]], [[-5.0]]]], dtype=np.float32)
    coords = coords.reshape(1, 1, 2, 1)
    rel = coords_mod.coords_relative_from_nominal(coords)
    assert np.allclose(rel, 0.0)


def test_coords_relative_from_nominal_c4_matches_tf_sign():
    coords = np.array([
        [[[1.0, 3.0, 5.0, 7.0]], [[2.0, 0.0, -2.0, 4.0]]]
    ], dtype=np.float32)  # (1, 1, 2, 4)
    mean = coords.mean(axis=3, keepdims=True)
    expected = -(coords - mean)
    rel = coords_mod.coords_relative_from_nominal(coords)
    assert np.allclose(rel, expected)
```

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/torch/test_coords_relative_contract.py -v \
  | tee .artifacts/object_big_relative_offsets/pytest_coords_relative_contract.red.log
```
Expected: FAIL (`ptycho_torch.coords` or `coords_relative_from_nominal` missing).

---

### Task 2: Implement coords helper (TF contract)

**Files:**
- Create: `ptycho_torch/coords.py`

**Step 1: Implement the helper**

```python
import numpy as np


def coords_relative_from_nominal(coords: np.ndarray) -> np.ndarray:
    """Convert nominal coords to TF-style relative offsets.

    Expected input shape: (B, 1, 2, C) where C = gridsize**2.
    Uses local_offset_sign = -1 and centers per-group.
    """
    coords_np = np.asarray(coords, dtype=np.float32)
    if coords_np.ndim != 4 or coords_np.shape[1:3] != (1, 2):
        raise ValueError(f"coords must have shape (B, 1, 2, C); got {coords_np.shape}")
    mean = coords_np.mean(axis=3, keepdims=True)
    return -(coords_np - mean)
```

**Step 2: Run test to verify it passes**

Run:
```bash
pytest tests/torch/test_coords_relative_contract.py -v \
  | tee .artifacts/object_big_relative_offsets/pytest_coords_relative_contract.green.log
```
Expected: PASS.

**Step 3: Commit**

```bash
git add ptycho_torch/coords.py tests/torch/test_coords_relative_contract.py
git commit -m "test(torch): add coords_relative contract helper"
```

---

### Task 3: Make grid-lines workflow metadata explicit about coords type

**Files:**
- Modify: `ptycho/workflows/grid_lines_workflow.py`

**Step 1: Add coords-type metadata**

Update the metadata payload in `save_split_npz()` to include a `coords_type` flag indicating the TF contract (`relative`).

```python
metadata = MetadataManager.create_metadata(
    config,
    script_name="grid_lines_workflow",
    size=cfg.size,
    offset=cfg.offset,
    outer_offset_train=cfg.outer_offset_train,
    outer_offset_test=cfg.outer_offset_test,
    nimgs_train=cfg.nimgs_train,
    nimgs_test=cfg.nimgs_test,
    probe_mask_diameter=cfg.probe_mask_diameter,
    probe_source=cfg.probe_source,
    coords_type="relative",
)
```

**Step 2: Add a focused test for metadata**

Create a small unit test in `tests/torch/test_grid_lines_torch_runner.py` to assert the metadata flag is honored when selecting coords (see Task 4 for runner changes). If the file already exists, add a new test case there to avoid a new module.

**Step 3: Run the test**

```bash
pytest tests/torch/test_grid_lines_torch_runner.py -k coords_type -v \
  | tee .artifacts/object_big_relative_offsets/pytest_grid_lines_coords_type.red.log
```
Expected: FAIL until Task 4 is implemented.

---

### Task 4: Update grid-lines torch runner to use coords_relative explicitly

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Add helper to select/derive coords_relative**

In `grid_lines_torch_runner.py`, add a helper that:
- Reshapes coords to `(B, 1, 2, C)` using the existing `_reshape_coords`
- Uses `coords_relative` if present
- Else uses metadata `coords_type` to decide:
  - `relative` → pass through
  - `nominal` → call `coords_relative_from_nominal`
  - otherwise → raise a clear error when `object_big=True`

```python
from ptycho_torch.coords import coords_relative_from_nominal


def _select_coords_relative(data, metadata, n_samples, channels):
    coords_rel = data.get("coords_relative")
    if coords_rel is not None:
        return _reshape_coords(coords_rel, n_samples, channels)
    coords_nom = _reshape_coords(data.get("coords_nominal"), n_samples, channels)
    coords_type = (metadata or {}).get("additional_parameters", {}).get("coords_type")
    if coords_type == "relative" or coords_type is None:
        return coords_nom
    if coords_type == "nominal":
        return coords_relative_from_nominal(coords_nom)
    raise ValueError(f"Unknown coords_type='{coords_type}'.")
```

**Step 2: Use helper in training/inference container construction**

Replace direct `coords_nominal` usage in `run_torch_training()` and `run_torch_inference()` with `_select_coords_relative(...)`, and add `coords_relative` into the container dict.

**Step 3: Update tests**

In `tests/torch/test_grid_lines_torch_runner.py`, add tests to validate:
- `coords_relative` key wins over `coords_nominal` when both present
- `coords_type='nominal'` triggers normalization using the helper
- `coords_type='relative'` passes through unchanged

**Step 4: Run targeted tests**

```bash
pytest tests/torch/test_grid_lines_torch_runner.py -k coords_type -v \
  | tee .artifacts/object_big_relative_offsets/pytest_grid_lines_coords_type.green.log
```
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py tests/torch/test_grid_lines_torch_runner.py ptycho/workflows/grid_lines_workflow.py
git commit -m "fix(torch): use coords_relative for object_big inputs"
```

---

### Task 5: Add hard guard in Lightning dataloader for object_big

**Files:**
- Modify: `ptycho_torch/workflows/components.py`
- Create: `tests/torch/test_lightning_dataloader_coords_guard.py`

**Step 1: Implement the guard**

In `_build_lightning_dataloaders`, capture `model_config` from the payload (if provided) and pass it into `PtychoLightningDataset`. In the dataset initializer, if `model_config.object_big` is true and `coords_relative` is missing, raise a `ValueError` instructing callers to provide `coords_relative` (or set `object_big=False` for non-ptychographic runs).

Pseudo-patch:
```python
model_config = payload.pt_model_config if payload else None

class PtychoLightningDataset(Dataset):
    def __init__(self, container, model_config=None):
        ...
        self.coords_relative = _get_tensor(container, 'coords_relative')
        if self.coords_relative is None:
            if model_config and getattr(model_config, "object_big", False):
                raise ValueError(
                    "coords_relative is required when object_big=True. "
                    "Provide TF-style relative offsets or set object_big=False."
                )
            self.coords_relative = _get_tensor(container, 'coords_nominal')
```

**Step 2: Add unit tests**

Create `tests/torch/test_lightning_dataloader_coords_guard.py` with two cases:
- When `object_big=True` and container lacks `coords_relative`, `_build_lightning_dataloaders` raises.
- When `object_big=False`, fallback to `coords_nominal` is allowed.

**Step 3: Run tests**

```bash
pytest tests/torch/test_lightning_dataloader_coords_guard.py -v \
  | tee .artifacts/object_big_relative_offsets/pytest_lightning_coords_guard.log
```
Expected: PASS.

**Step 4: Commit**

```bash
git add ptycho_torch/workflows/components.py tests/torch/test_lightning_dataloader_coords_guard.py
git commit -m "fix(torch): hard guard coords_relative when object_big"
```

---

### Task 6: Integration + parity verification

**Files:**
- Test: `tests/torch/test_grid_lines_hybrid_resnet_integration.py`

**Step 1: Run the integration test**

```bash
pytest -v -m integration tests/torch/test_grid_lines_hybrid_resnet_integration.py \
  | tee .artifacts/object_big_relative_offsets/pytest_grid_lines_hybrid_resnet_integration.log
```
Expected: PASS.

**Step 2: Run integration marker (policy)**

```bash
pytest -v -m integration \
  | tee .artifacts/object_big_relative_offsets/pytest_integration_marker.log
```
Expected: PASS.

**Step 3: Patch-parity helper (backend parity evidence)**

```bash
python scripts/tools/patch_parity_helper.py \
  --out-dir tmp/patch_parity/object_big_relative_offsets \
  | tee .artifacts/object_big_relative_offsets/patch_parity_helper.log
```
Expected: PNGs under `tmp/patch_parity/object_big_relative_offsets/` for amplitude/phase inspection.

---

### Task 7: Update test registry + bug report

**Files:**
- Modify: `docs/development/TEST_SUITE_INDEX.md`
- Modify: `docs/TESTING_GUIDE.md`
- Modify: `docs/bugs/2026-02-05-object-big-coords-relative-regression.md`

**Step 1: Regenerate test index**

```bash
python scripts/tools/generate_test_index.py \
  | tee .artifacts/object_big_relative_offsets/generate_test_index.log
```

**Step 2: Update testing guide**

Add the new selector(s) for `test_coords_relative_contract.py` in the Torch tests section of `docs/TESTING_GUIDE.md` (if not already present after regeneration).

**Step 3: Append bug resolution note**

Update the bug report with fix summary and log paths from Tasks 4–5.

**Step 4: Commit**

```bash
git add docs/development/TEST_SUITE_INDEX.md docs/TESTING_GUIDE.md docs/bugs/2026-02-05-object-big-coords-relative-regression.md
git commit -m "docs(test): register coords-relative regression coverage"
```

---

Plan complete and saved to `docs/plans/2026-02-05-object-big-relative-offsets.md`.

Two execution options:

1. **Subagent-Driven (this session)** — I dispatch a fresh subagent per task, review between tasks.
2. **Parallel Session (separate)** — open a new session and use `superpowers:executing-plans` with checkpoints.

Which approach?
