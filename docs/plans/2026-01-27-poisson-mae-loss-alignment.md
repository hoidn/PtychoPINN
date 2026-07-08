# Poisson + MAE Loss Unit Alignment (PyTorch) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align PyTorch Poisson and MAE loss calculations with the amplitude/intensity contracts used in TF and the data pipeline, and add unit tests to prevent regressions.

**Architecture:** Keep the existing physics pipeline and data normalization. Update `PoissonIntensityLayer` to square observed amplitudes before log-likelihood and update `MAELoss` to operate on amplitude (no pred-only squaring). Add minimal unit tests that encode these contracts. No changes to stable TF modules.

**Tech Stack:** PyTorch, pytest

---

### Task 0: Preflight verification (unverified assumptions)

**Files:**
- Inspect: `ptycho_torch/generators/fno.py`, `ptycho_torch/model.py`, `ptycho_torch/dataloader.py`
- Inspect: `ptycho/diffsim.py`, `ptycho/workflows/grid_lines_workflow.py`
- Reference: `docs/workflows/pytorch.md`, `docs/DATA_NORMALIZATION_GUIDE.md`

**Step 1: Confirm neuraloperator fallback status**

Run:
```bash
python - <<'PY'
from ptycho_torch.generators import fno
print("HAS_NEURALOPERATOR", fno.HAS_NEURALOPERATOR)
PY
```
Expected: explicit True/False output to record whether `_FallbackSpectralConv2d` is active.

**Step 2: Confirm diffraction is amplitude in the pipeline**

Run:
```bash
python - <<'PY'
import numpy as np
from pathlib import Path
p = Path('outputs/grid_lines_gs1_n64_e5/datasets/N64/gs1/train.npz')
if p.exists():
    data = np.load(p)
    x = data['diffraction']
    print('diffraction dtype', x.dtype, 'min', float(x.min()), 'max', float(x.max()))
else:
    print('dataset not found; confirm via diffsim + workflow code')
PY
```
Expected: non-negative amplitudes; no squaring/Poisson counts stored in NPZ.

**Step 3: Confirm loss contract locations**

Record the current behavior for `PoissonIntensityLayer.forward` and `MAELoss.forward` in `ptycho_torch/model.py` (no code change).

**Step 4 (Conditional): Gradient diagnostics if loss still stagnates after unit fixes**

Only run this if FNO/Hybrid loss remains flat *after* Task 2. This does not authorize any code changes; it only gathers evidence.

Run (no file creation; avoid committing diagnostics):
```bash
python - <<'PY'
import torch
from ptycho_torch.generators.fno import PtychoBlock

B, H, W, dim = 4, 64, 64, 32
block = PtychoBlock(channels=dim, modes=12)

if hasattr(block.spectral, 'weights'):
    spec_w = block.spectral.weights
    print(f"Spectral abs_mean={spec_w.abs().mean().item():.6e}, std={spec_w.abs().std().item():.6e}")
local_w = block.local_conv.weight
print(f"Local abs_mean={local_w.abs().mean().item():.6e}, std={local_w.abs().std().item():.6e}")

x = torch.randn(B, dim, H, W, requires_grad=True)
loss = block(x).mean()
loss.backward()

if hasattr(block.spectral, 'weights'):
    spec_grad = block.spectral.weights.grad
    print(f"Spectral grad abs_mean={spec_grad.abs().mean().item():.6e}, std={spec_grad.abs().std().item():.6e}")
local_grad = block.local_conv.weight.grad
print(f"Local grad abs_mean={local_grad.abs().mean().item():.6e}, std={local_grad.abs().std().item():.6e}")

if hasattr(block.spectral, 'weights'):
    ratio = local_grad.abs().mean() / (spec_grad.abs().mean() + 1e-9)
    print(f"Local/Spectral grad ratio={ratio.item():.2f}")
PY
```
Expected: non‑zero spectral gradients comparable in order of magnitude to local path. If spectral grads are near‑zero, treat this as evidence only; do not modify initialization without a separate plan.

---

### Task 1: Add failing unit tests for loss unit contracts

**Files:**
- Create: `tests/torch/test_loss_units.py`

**Step 1: Write failing tests**

```python
import torch
import torch.distributions as dist
from ptycho_torch.model import PoissonIntensityLayer, MAELoss


def test_poisson_intensity_layer_squares_observations():
    pred_amp = torch.tensor([[[[2.0]]]])  # (B,C,H,W)
    obs_amp = torch.tensor([[[[3.0]]]])

    layer = PoissonIntensityLayer(pred_amp)
    loss = layer(obs_amp)

    expected = -dist.Independent(
        dist.Poisson(pred_amp ** 2, validate_args=False),
        3,
    ).log_prob(obs_amp ** 2)

    assert torch.allclose(loss, expected)


def test_mae_loss_operates_on_amplitude():
    pred_amp = torch.tensor([[[[2.0]]]])
    obs_amp = torch.tensor([[[[3.5]]]])

    loss_fn = MAELoss()
    loss = loss_fn(pred_amp, obs_amp)

    expected = torch.nn.functional.l1_loss(pred_amp, obs_amp, reduction="none")
    assert torch.allclose(loss, expected)
```

**Step 2: Run tests to verify failure**

Run:
```bash
pytest tests/torch/test_loss_units.py -v
```
Expected: FAIL because Poisson uses raw amplitudes and MAE squares predictions.

---

### Task 2: Implement loss unit fixes

**Files:**
- Modify: `ptycho_torch/model.py`

**Step 1: Fix PoissonIntensityLayer to square observed amplitudes**

```python
# PoissonIntensityLayer.forward
observed_intensity = x ** 2
return -self.poisson_dist.log_prob(observed_intensity)
```

**Step 2: Fix MAELoss to use amplitude**

```python
# MAELoss.forward
loss_mae = self.mae(pred, raw)
```

**Step 3: Run tests to verify pass**

Run:
```bash
pytest tests/torch/test_loss_units.py -v
```
Expected: PASS

**Step 4: Commit**

```bash
git add ptycho_torch/model.py tests/torch/test_loss_units.py
git commit -m "fix(torch): align poisson/mae loss units"
```

---

### Task 3: Update bug record

**Files:**
- Modify: `docs/bugs/POISSON_MAE_UNIT_MISMATCH_PYTORCH.md`

**Step 1: Mark status and reference fix**
- Update Status to Resolved
- Add brief note referencing the commit and tests

**Step 2: Commit**

```bash
git add docs/bugs/POISSON_MAE_UNIT_MISMATCH_PYTORCH.md
git commit -m "docs: close poisson/mae unit mismatch bug"
```

---

### Task 4: Verification & Evidence

**Tests:**
- `pytest tests/torch/test_loss_units.py -v`
- `pytest tests/torch/test_grid_lines_torch_runner.py -v`
- `pytest -m integration -v` (policy for production workflow changes)

**Evidence:**
- Save logs under `.artifacts/poisson_mae_loss_alignment/`
- Link logs from the active plan summary

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-01-27-poisson-mae-loss-alignment.md`.

Two execution options:
1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks
2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
