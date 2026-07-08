# Torch Stitching Collapse Debug Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Identify why Torch stitching collapses to a 64×64 square by tracing offsets through the inference/reassembly path and building a minimal failing test that reproduces the collapse.

**Architecture:** Capture evidence at each boundary (RawData → offsets → reassembly), add a multi‑patch parity test comparing TF vs Torch reassembly, and add opt‑in instrumentation to log offset statistics without changing default behavior. Use the evidence to localize whether offsets are identical, broadcast, or mis‑shaped before a fix is attempted.

**Tech Stack:** Python, PyTorch, TensorFlow, NumPy, pytest

---

## Task 1: Reproduce and Capture Baseline Evidence (No Code Changes)

**Files:**
- Create: `tmp/stitch_debug/baseline.md`

**Step 1: Reproduce the collapse**

Run:
```bash
RUN_LONG_INTEGRATION=1 python -m pytest \
  tests/torch/test_integration_manual_1000_512_torch.py::test_train_infer_cycle_1000_train_512_test -vv \
  | tee tmp/stitch_debug/pytest_manual_1000_512.log
```

**Step 2: Compute reconstructed bounding box**

Run:
```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image

root = Path("/tmp/pytest-of-ollie")
latest = sorted(root.glob("pytest-*"))[-1]
out = next(latest.glob("*/pytorch_output/reconstructed_amplitude.png"))
img = np.array(Image.open(out))
ys, xs = np.where(img > 0)
if ys.size == 0:
    print("No nonzero pixels found")
else:
    print(f"bbox: x[{xs.min()}, {xs.max()}], y[{ys.min()}, {ys.max()}], "
          f"w={xs.max()-xs.min()+1}, h={ys.max()-ys.min()+1}")
PY
```

**Step 3: Record baseline**

Write a short summary to `tmp/stitch_debug/baseline.md` with:
- pytest log path
- bbox size and location
- paths to reconstructed PNGs

---

## Task 2: Add Multi‑Patch Reassembly Parity Test (RED → GREEN)

**Files:**
- Create: `tests/torch/test_reassembly_multi_patch_parity.py`
- Modify (if needed): `ptycho_torch/helper.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_reassembly_multi_patch_parity.py
import numpy as np
import tensorflow as tf
import torch

from ptycho.tf_helper import reassemble_position
from ptycho_torch.helper import reassemble_patches_position_real
from ptycho_torch.config_params import DataConfig, ModelConfig


def test_reassembly_applies_distinct_offsets():
    N = 64
    patches = np.zeros((2, N, N, 1), dtype=np.complex64)
    patches[0, 8:12, 8:12, 0] = 1 + 0j
    patches[1, 20:24, 20:24, 0] = 1 + 0j

    # Two distinct offsets (pixels)
    offsets = np.array([
        [[[0.0, 0.0]]],
        [[[12.0, 10.0]]],
    ], dtype=np.float32)  # (B, 1, 1, 2)

    tf_out = reassemble_position(
        tf.convert_to_tensor(patches),
        tf.convert_to_tensor(offsets),
        M=20,
    ).numpy()

    data_cfg = DataConfig(N=N, grid_size=(1, 1), C=1)
    model_cfg = ModelConfig(C_forward=1, C_model=1, object_big=True)
    torch_out, _, _ = reassemble_patches_position_real(
        torch.from_numpy(patches).permute(0, 3, 1, 2),
        torch.from_numpy(offsets),
        data_cfg,
        model_cfg,
        crop_size=20,
    )
    torch_out = torch_out.detach().cpu().numpy()

    # Torch output should not collapse to a single patch region
    assert torch_out.shape[-1] > N
    assert np.count_nonzero(torch_out) > np.count_nonzero(patches[0])

    # Parity: torch and tf outputs should have similar nonzero support
    tf_mask = np.abs(tf_out[0]) > 0
    torch_mask = np.abs(torch_out[0]) > 0
    assert torch_mask.sum() == tf_mask.sum()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_reassembly_multi_patch_parity.py::test_reassembly_applies_distinct_offsets -vv`  
Expected: FAIL if offsets are being collapsed or broadcast.

**Step 3: If RED is confirmed, investigate cause before fixing**

Do not implement fixes yet; proceed to Task 3 for instrumentation.

---

## Task 3: Instrument Offset Flow (Evidence Only)

**Files:**
- Modify: `ptycho_torch/helper.py`
- Modify: `ptycho_torch/inference.py`
- Create: `ptycho_torch/debug.py`

**Step 1: Add debug helper**

```python
# ptycho_torch/debug.py
import torch

def summarize_offsets(name, offsets: torch.Tensor):
    if offsets is None:
        return f"{name}: None"
    flat = offsets.reshape(-1, offsets.shape[-1]).detach().cpu()
    return (
        f"{name}: shape={tuple(offsets.shape)} "
        f"min={flat.min(0).values.tolist()} "
        f"max={flat.max(0).values.tolist()} "
        f"std={flat.std(0).tolist()} "
        f"unique={torch.unique(flat, dim=0).shape[0]}"
    )
```

**Step 2: Gate logging by env var**

```python
# ptycho_torch/inference.py (before reassembly call)
import os
from ptycho_torch.debug import summarize_offsets

if os.getenv("PTYCHO_TORCH_STITCH_DEBUG") == "1":
    print(summarize_offsets("offsets_before_reassembly", offsets))
```

```python
# ptycho_torch/helper.py (inside reassemble_patches_position_real)
import os
from ptycho_torch.debug import summarize_offsets

if os.getenv("PTYCHO_TORCH_STITCH_DEBUG") == "1":
    print(summarize_offsets("offsets_input", offsets))
```

**Step 3: Run and capture logs**

Run:
```bash
PTYCHO_TORCH_STITCH_DEBUG=1 RUN_LONG_INTEGRATION=1 python -m pytest \
  tests/torch/test_integration_manual_1000_512_torch.py::test_train_infer_cycle_1000_train_512_test -vv \
  | tee tmp/stitch_debug/offsets_debug.log
```

**Step 4: Record findings**

Summarize in `tmp/stitch_debug/baseline.md`:
- Offsets shape at inference entry vs reassembly
- Unique counts and stds

---

## Task 4: Diagnose Root Cause (No Fix Yet)

**Files:**
- Create: `tmp/stitch_debug/analysis.md`

**Step 1: Analyze evidence**

Use the debug logs to decide:
- Are offsets identical before reassembly?
- Are offsets unique before reassembly but identical inside reassembly?
- Does shape conversion (permute/view) collapse them?

**Step 2: Document suspected root cause**

Write `tmp/stitch_debug/analysis.md` with:
- Which boundary collapses offsets
- Supporting log lines
- The minimal failing test from Task 2

---

## Task 5: Fix Plan (Follow‑up)

**Note:** Do not implement a fix until Task 4 identifies the failure boundary.

Once root cause is known, write a follow‑up fix plan that:
- Adds a failing regression test if needed
- Implements a single, minimal change at the failing boundary
- Verifies both the unit test (Task 2) and the long integration test pass

---

## Notes
- Store temporary artifacts under `tmp/stitch_debug/` and do not commit.
- If the failure is in data generation (offsets already identical), trace further upstream (RawData / coords_relative creation) before touching reassembly.

