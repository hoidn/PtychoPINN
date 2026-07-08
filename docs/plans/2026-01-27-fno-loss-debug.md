# FNO/U-FNO Loss Debugging Plan (Temporary Instrumentation)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Diagnose why the FNO/U‑FNO training loss appears nearly flat by inspecting intermediate inputs, activations, and gradients, using **temporary** gated instrumentation.

**Architecture:** Add short‑lived logging at key boundaries (data loader → model input → generator output → combined complex → loss), controlled by `DEBUG_FNO_LOSS=1`. Collect a single artifact bundle and then remove all instrumentation.

**Tech Stack:** PyTorch, Lightning, numpy; existing `ptycho_torch` training workflow.

---

### Task 1: Add a global debug gate

**Files:**
- Modify (temporary): `ptycho_torch/workflows/components.py`
- Modify (temporary): `ptycho_torch/model.py`
- Modify (temporary): `scripts/studies/grid_lines_torch_runner.py`

**Step 1: Add a helper for gated logging**

```python
import os
DEBUG_FNO_LOSS = os.getenv("DEBUG_FNO_LOSS") == "1"
```

**Step 2: Wrap all debug logging blocks with this gate**

---

### Task 2: Loader boundary stats (train + val)

**Files:**
- Modify (temporary): `ptycho_torch/workflows/components.py`

**Step 1: Log first train batch only**
Log:
- `images` min/max/mean/std
- `coords_relative` min/max/mean/std
- `rms_scaling_constant` min/max/mean/std
- `physics_scaling_constant` min/max/mean/std
- `probe` min/max/mean/std (real, imag, abs)

**Step 2: Log val loader presence and length**
- Whether val loader exists
- Its length (if not None)

**Output:** append to a JSONL file under `.artifacts/debug_fno_loss/loader_stats.jsonl`

---

### Task 3: Model input + generator output stats

**Files:**
- Modify (temporary): `ptycho_torch/model.py`

**Step 1: In `forward` / `forward_predict`, log for first batch only**
- Input after scaling (`x`) stats
- Generator output stats (real/imag or amp/phase)
- Combined complex (`x_combined`) abs/angle stats

**Output:** append to `.artifacts/debug_fno_loss/activations.jsonl`

---

### Task 4: Gradient flow stats

**Files:**
- Modify (temporary): `ptycho_torch/model.py` or Lightning `training_step`

**Step 1: Register gradient hooks for first + last generator layers**
Log grad mean/std for first batch per epoch.

**Output:** append to `.artifacts/debug_fno_loss/gradients.jsonl`

---

### Task 5: Patch-level metrics (pre‑stitch)

**Files:**
- Modify (temporary): `scripts/studies/grid_lines_torch_runner.py`

**Step 1: Compute patch MAE/MSE on amplitude**
Compare prediction amplitude vs `Y_I`.
State explicitly whether scaling is applied (if any).

**Output:** `.artifacts/debug_fno_loss/patch_metrics.json`

---

### Task 6: Run a short debug training

**Command:**
```bash
DEBUG_FNO_LOSS=1 python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 --gridsize 1 --output-dir outputs/debug_fno_loss \
  --architectures fno --nepochs 1 --torch-epochs 1 --torch-loss-mode mae
```

---

### Task 7: Remove temporary instrumentation

**Step 1:** Revert all debug logging blocks and helper gate.

---

### Task 8: Verification

**Command:**
```bash
pytest tests/torch/test_grid_lines_torch_runner.py -v
```

**Artifacts:**
- `.artifacts/debug_fno_loss/loader_stats.jsonl`
- `.artifacts/debug_fno_loss/activations.jsonl`
- `.artifacts/debug_fno_loss/gradients.jsonl`
- `.artifacts/debug_fno_loss/patch_metrics.json`

---

## Notes
- Instrumentation is temporary and must be removed after analysis.
- No plan/ledger links required for this request.
