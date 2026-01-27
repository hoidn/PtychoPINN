# FNO Fallback Spectral Initialization Scaling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate whether `_FallbackSpectralConv2d` initialization suppresses spectral gradients and, if confirmed, adjust the scaling to a variance‑preserving factor with regression tests.

**Architecture:** Keep the existing FNO/Hybrid structure. Add a deterministic unit test for fallback spectral weight scale, then update `_FallbackSpectralConv2d` initialization to use a Xavier‑like `1/sqrt(in_channels)` scale. The change is scoped to the fallback path only (when neuraloperator is unavailable).

**Tech Stack:** PyTorch, pytest

---

### Task 0: Preflight evidence & gating

**Files:**
- Inspect: `ptycho_torch/generators/fno.py`
- Reference: `docs/architecture_torch.md`, `docs/workflows/pytorch.md`, `docs/DATA_NORMALIZATION_GUIDE.md`

**Step 1: Confirm Poisson/MAE unit fix plan is complete**

This plan assumes the loss‑unit mismatch has been addressed (see `docs/plans/2026-01-27-poisson-mae-loss-alignment.md`). If not complete, pause and finish that plan first to avoid confounding.

**Step 2: Confirm fallback path is active in this environment**

Run:
```bash
python - <<'PY'
from ptycho_torch.generators import fno
print("HAS_NEURALOPERATOR", fno.HAS_NEURALOPERATOR)
PY
```
Expected: explicit True/False output recorded in notes.

**Step 3: Gradient diagnostics (evidence only, no code changes)**

Run:
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
Expected: non‑zero spectral gradients. If spectral grads are orders of magnitude smaller (e.g., ratio > 100), proceed to Task 1.

---

### Task 1: Add failing test for fallback spectral weight scale

**Files:**
- Create: `tests/torch/test_fno_fallback_init.py`

**Step 1: Write the failing test**

```python
import math
import torch
from ptycho_torch.generators.fno import _FallbackSpectralConv2d


def test_fallback_spectral_init_scale():
    torch.manual_seed(0)
    in_ch, out_ch, modes = 4, 4, 8
    layer = _FallbackSpectralConv2d(in_ch, out_ch, modes)

    # For complex weights, real/imag parts are N(0, scale).
    real_std = layer.weights.real.std().item()
    expected = 1.0 / math.sqrt(in_ch)

    assert abs(real_std - expected) / expected < 0.2
```

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/torch/test_fno_fallback_init.py -v
```
Expected: FAIL because current scale is `1/(in_channels*out_channels)`.

---

### Task 2: Update `_FallbackSpectralConv2d` initialization

**Files:**
- Modify: `ptycho_torch/generators/fno.py`

**Step 1: Implement new scaling**

```python
# in _FallbackSpectralConv2d.__init__
scale = 1 / (in_channels ** 0.5)
self.weights = nn.Parameter(
    scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
)
```

**Step 2: Run the new test**

Run:
```bash
pytest tests/torch/test_fno_fallback_init.py -v
```
Expected: PASS

**Step 3: Commit**

```bash
git add ptycho_torch/generators/fno.py tests/torch/test_fno_fallback_init.py
git commit -m "fix(torch): adjust fallback spectral init scale"
```

---

### Task 3: Verification & evidence

**Tests:**
- `pytest tests/torch/test_fno_fallback_init.py -v`
- `pytest tests/torch/test_fno_integration.py -v`
- `pytest -m integration -v`

**Evidence:**
- Save logs under `.artifacts/fno_fallback_init_scale/`
- Link logs from the active plan summary

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-01-27-fno-fallback-init-scaling.md`.

Two execution options:
1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks
2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
