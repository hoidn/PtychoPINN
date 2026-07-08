# Shared Center-Crop Utility Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make center-cropping behavior modular and reusable across production and study/data codepaths, so bridge stitching and non-bridge scripts use one canonical implementation.

**Architecture:** Introduce a public spatial center-crop utility in `ptycho/image/cropping.py` that works on arrays with spatial dims in the last two axes, plus a border-based helper for stitch crops. Keep backward compatibility by retaining `_center_crop` as a thin shim. Migrate bridge + selected production/study/data callsites to import the shared helper, then verify parity with focused regression tests.

**Tech Stack:** Python, NumPy, pytest

---

### Task 1: Add Canonical Crop API In `ptycho.image.cropping`

**Files:**
- Modify: `ptycho/image/cropping.py`
- Modify: `ptycho/image/__init__.py`
- Test: `tests/image/test_cropping.py`

**Step 1: Write the failing test**

```python
from ptycho.image.cropping import center_crop_spatial, center_crop_spatial_by_border


def test_center_crop_spatial_handles_rank3_stack_last_two_dims():
    arr = np.arange(2 * 6 * 6).reshape(2, 6, 6)
    out = center_crop_spatial(arr, 4, 4)
    assert out.shape == (2, 4, 4)


def test_center_crop_spatial_by_border_clamps_to_nonempty():
    arr = np.ones((1, 5, 5), dtype=np.float32)
    out = center_crop_spatial_by_border(arr, border=99)
    assert out.shape == (1, 1, 1)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/image/test_cropping.py -k "center_crop_spatial" -v`
Expected: FAIL with import or attribute errors for the new functions.

**Step 3: Write minimal implementation**

```python
def center_crop_spatial(array: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    ...

def center_crop_spatial_by_border(array: np.ndarray, border: int) -> np.ndarray:
    ...

# keep for backward compatibility
def _center_crop(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    return center_crop_spatial(img, target_h, target_w)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/image/test_cropping.py -k "center_crop_spatial or center_crop_exact_size" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add ptycho/image/cropping.py ptycho/image/__init__.py tests/image/test_cropping.py
git commit -m "refactor: add shared spatial center-crop utilities"
```

### Task 2: Use Shared Crop Utility In PtychoViT Bridge Stitching

**Files:**
- Modify: `scripts/studies/ptychovit_bridge_entrypoint.py`
- Test: `tests/test_ptychovit_bridge_entrypoint.py`

**Step 1: Write the failing test**

```python
def test_bridge_stitch_uses_shared_center_crop_helper(monkeypatch):
    called = {}

    def _fake_crop(patches, border):
        called["border"] = border
        return patches

    monkeypatch.setattr(
        "scripts.studies.ptychovit_bridge_entrypoint.center_crop_spatial_by_border",
        _fake_crop,
    )

    _stitch_complex_predictions(..., crop_border=8, pad=0)
    assert called["border"] == 8
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ptychovit_bridge_entrypoint.py -k "shared_center_crop_helper" -v`
Expected: FAIL because bridge still uses local crop helper.

**Step 3: Write minimal implementation**

```python
from ptycho.image.cropping import center_crop_spatial_by_border

# remove local crop implementation
patches = center_crop_spatial_by_border(patches, crop_border).astype(np.complex64)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ptychovit_bridge_entrypoint.py -k "stitch_param or center_crops or shared_center_crop_helper" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/ptychovit_bridge_entrypoint.py tests/test_ptychovit_bridge_entrypoint.py
git commit -m "refactor: route bridge stitch cropping through shared utility"
```

### Task 3: Migrate Study/Data Script Callsites Outside Bridge

**Files:**
- Modify: `scripts/studies/prepare_nersc_hybrid_dataset.py`
- Modify: `scripts/tools/downsample_data_tool.py`
- Modify: `scripts/simulation/simulate_and_save.py`
- Modify: `scripts/studies/dose_response_study.py`
- Test: `tests/studies/test_prepare_nersc_hybrid_dataset.py`
- Test: `tests/tools/test_downsample_data_tool.py` (create)

**Step 1: Write the failing test**

```python
def test_downsample_tool_crop_center_matches_shared_spatial_crop():
    src = np.arange(36).reshape(6, 6)
    expected = center_crop_spatial(src, 4, 4)
    assert np.array_equal(crop_center(src, 4, 4), expected)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_downsample_data_tool.py -v`
Expected: FAIL because test file/function does not yet exist.

**Step 3: Write minimal implementation**

```python
from ptycho.image.cropping import center_crop_spatial

# prepare_nersc_hybrid_dataset.py
return center_crop_spatial(arr, target_h, target_w)

# downsample_data_tool.py and simulate_and_save.py / dose_response_study.py
return center_crop_spatial(img, new_h, new_w)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/studies/test_prepare_nersc_hybrid_dataset.py tests/tools/test_downsample_data_tool.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/prepare_nersc_hybrid_dataset.py scripts/tools/downsample_data_tool.py scripts/simulation/simulate_and_save.py scripts/studies/dose_response_study.py tests/studies/test_prepare_nersc_hybrid_dataset.py tests/tools/test_downsample_data_tool.py
git commit -m "refactor: unify study/data script center-crop callsites"
```

### Task 4: Migrate Production Torch Helper To Shared Utility

**Files:**
- Modify: `ptycho_torch/helper.py`
- Test: `tests/torch/test_helper_center_crop.py` (create)

**Step 1: Write the failing test**

```python
def test_helper_center_crop_matches_shared_spatial_center_crop():
    arr = np.arange(64).reshape(8, 8)
    assert np.array_equal(
        center_crop(arr, 4),
        center_crop_spatial(arr, 4, 4),
    )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_helper_center_crop.py -v`
Expected: FAIL because new test module does not exist.

**Step 3: Write minimal implementation**

```python
from ptycho.image.cropping import center_crop_spatial


def center_crop(larger_img, target_size):
    return center_crop_spatial(larger_img, target_size, target_size)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_helper_center_crop.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add ptycho_torch/helper.py tests/torch/test_helper_center_crop.py
git commit -m "refactor: share torch helper center-crop implementation"
```

### Task 5: End-to-End Verification And Documentation Note

**Files:**
- Modify: `docs/findings.md` (optional note only if behavior or contracts changed)

**Step 1: Write verification checklist test selection**

```python
SELECTORS = [
    "tests/image/test_cropping.py",
    "tests/test_ptychovit_bridge_entrypoint.py",
    "tests/studies/test_prepare_nersc_hybrid_dataset.py",
    "tests/tools/test_downsample_data_tool.py",
    "tests/torch/test_helper_center_crop.py",
]
```

**Step 2: Run test suite subset**

Run: `pytest tests/image/test_cropping.py tests/test_ptychovit_bridge_entrypoint.py tests/studies/test_prepare_nersc_hybrid_dataset.py tests/tools/test_downsample_data_tool.py tests/torch/test_helper_center_crop.py -v`
Expected: PASS for all selectors.

**Step 3: Verify no remaining duplicate callsite helpers in scope**

Run: `rg -n "def (crop_center|_crop_center_2d|_center_crop_patches|center_crop)" scripts/studies scripts/tools scripts/simulation ptycho_torch/helper.py ptycho/image/cropping.py -S`
Expected: only canonical wrappers or required compatibility shims remain.

**Step 4: Stage all plan-related implementation changes**

```bash
git add ptycho/image/cropping.py ptycho/image/__init__.py scripts/studies/ptychovit_bridge_entrypoint.py scripts/studies/prepare_nersc_hybrid_dataset.py scripts/tools/downsample_data_tool.py scripts/simulation/simulate_and_save.py scripts/studies/dose_response_study.py ptycho_torch/helper.py tests/image/test_cropping.py tests/test_ptychovit_bridge_entrypoint.py tests/studies/test_prepare_nersc_hybrid_dataset.py tests/tools/test_downsample_data_tool.py tests/torch/test_helper_center_crop.py
```

**Step 5: Commit**

```bash
git commit -m "refactor: modularize center-crop behavior across production and study scripts"
```

---

## Notes For Execution

- Use `@test-driven-development` during implementation of each task.
- Use `@verification-before-completion` before claiming completion.
- Keep changes DRY/YAGNI: wrappers are allowed for backward compatibility, but avoid duplicate crop math at callsites.
