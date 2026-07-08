# FNO Fallback Init Scaling & Gradient Diagnostics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a gradient diagnostics script for the FNO fallback spectral path, adjust `_FallbackSpectralConv2d` initialization to a variance-preserving scale, and verify the spectral path participates in training.

**Architecture:** Keep the existing FNO/Hybrid structure. Add a standalone diagnostics script to probe weight/gradient magnitudes for `PtychoBlock`, then update the fallback spectral convolution initialization from `1/(in_channels*out_channels)` to `1/sqrt(in_channels)` with a focused unit test. Re-verify via the diagnostics script and a short FNO training run. Dynamic-range handling in `SpatialLifter` is documented as a follow-up risk only.

**Tech Stack:** PyTorch, pytest

**Note:** User requested no worktree usage; this plan executes directly on `fno2`.

---

### Task 0: Preflight evidence & environment checks

**Files:**
- Inspect: `ptycho_torch/generators/fno.py`
- Reference: `docs/architecture_torch.md`, `docs/workflows/pytorch.md`, `docs/TESTING_GUIDE.md`

**Step 1: Confirm fallback path activation**

Run:
```bash
python - <<'PY'
from ptycho_torch.generators import fno
print("HAS_NEURALOPERATOR", fno.HAS_NEURALOPERATOR)
PY
```
Expected: explicit True/False output recorded in notes.

---

### Task 1: Add gradient diagnostics script

**Files:**
- Create: `scripts/debug_fno_gradients.py`

**Step 1: Write the script**

```python
import torch
from ptycho_torch.generators.fno import PtychoBlock


def debug_gradients():
    B, H, W, dim = 4, 64, 64, 32

    print(f"--- Debugging FNO Gradients (Hidden Dim={dim}) ---")
    block = PtychoBlock(channels=dim, modes=12)

    if hasattr(block.spectral, "weights"):
        spec_w = block.spectral.weights
        print(
            "Spectral Weights: mean={:.6f}, std={:.6f}, abs_mean={:.6f}".format(
                spec_w.mean().item(), spec_w.std().item(), spec_w.abs().mean().item()
            )
        )

    local_w = block.local_conv.weight
    print(
        "Local Conv Weights: mean={:.6f}, std={:.6f}".format(
            local_w.mean().item(), local_w.std().item()
        )
    )

    x = torch.randn(B, dim, H, W, requires_grad=True)
    y = block(x)
    loss = y.mean()
    loss.backward()

    print("\nGradients:")
    if hasattr(block.spectral, "weights"):
        spec_grad = block.spectral.weights.grad
        print(
            "Spectral Grad: mean={:.6f}, std={:.6f}, max={:.6f}".format(
                spec_grad.abs().mean().item(),
                spec_grad.std().item(),
                spec_grad.abs().max().item(),
            )
        )

    local_grad = block.local_conv.weight.grad
    print(
        "Local Conv Grad: mean={:.6f}, std={:.6f}, max={:.6f}".format(
            local_grad.abs().mean().item(),
            local_grad.std().item(),
            local_grad.abs().max().item(),
        )
    )

    if hasattr(block.spectral, "weights"):
        ratio = local_grad.abs().mean() / (spec_grad.abs().mean() + 1e-9)
        print(f"\nRatio (Local/Spectral Grad Magnitude): {ratio.item():.2f}")


if __name__ == "__main__":
    debug_gradients()
```

**Step 2: Run the script (baseline)**

Run:
```bash
python scripts/debug_fno_gradients.py
```
Expected: printout of spectral vs local weight/gradient stats (capture in notes).

**Step 3: Commit**

```bash
git add scripts/debug_fno_gradients.py
git commit -m "chore(fno): add gradient diagnostics script"
```

---

### Task 2: Fix fallback spectral initialization (TDD)

**Files:**
- Create: `tests/torch/test_fno_fallback_init.py`
- Modify: `ptycho_torch/generators/fno.py`

**Step 1: Write the failing test**

```python
import math
import torch
from ptycho_torch.generators.fno import _FallbackSpectralConv2d


def test_fallback_spectral_init_scale():
    torch.manual_seed(0)
    in_ch, out_ch, modes = 4, 4, 8
    layer = _FallbackSpectralConv2d(in_ch, out_ch, modes)

    real_std = layer.weights.real.std().item()
    expected = 1.0 / math.sqrt(in_ch)

    assert abs(real_std - expected) / expected < 0.2
```

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/torch/test_fno_fallback_init.py -v
```
Expected: FAIL (current scale is `1/(in_channels*out_channels)`).

**Step 3: Implement new scaling**

```python
# in _FallbackSpectralConv2d.__init__
scale = 1 / (in_channels ** 0.5)
self.weights = nn.Parameter(
    scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
)
```

**Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/torch/test_fno_fallback_init.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add ptycho_torch/generators/fno.py tests/torch/test_fno_fallback_init.py
git commit -m "fix(torch): adjust fallback spectral init scale"
```

---

### Task 3: Re-verify gradients after init fix

**Files:**
- Re-run: `scripts/debug_fno_gradients.py`

**Step 1: Run diagnostics again**

Run:
```bash
python scripts/debug_fno_gradients.py
```
Expected: spectral gradients non-trivial and closer in magnitude to local conv (record ratio).

---

### Task 4: Short FNO training smoke test

**Files:**
- Run: `scripts/studies/grid_lines_torch_runner.py`

**Step 1: Execute short training run**

Run:
```bash
python scripts/studies/grid_lines_torch_runner.py \
  --N 64 \
  --gridsize 1 \
  --architecture fno \
  --nepochs 5 \
  --torch-epochs 5 \
  --nimgs-train 1 \
  --nimgs-test 1 \
  --torch-infer-batch-size 8 \
  --output-dir outputs/grid_lines_fno_init_fix
```
Expected: loss decreases over the first few epochs (capture log snippet).

---

## Follow-up Risk (Documented Only)

- **Dynamic range/DC peak dominance:** `SpatialLifter` currently applies Conv→GELU→Conv without explicit compression (log/sqrt/InstanceNorm). Ptychographic diffraction patterns have extreme dynamic range (10^3–10^5 between DC peak and speckles). If loss stagnation persists after init fix, evaluate adding a controlled dynamic-range compressor at the lifter input. This is out of scope for this plan.

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-01-27-fno-fallback-init-scaling.md`.

Two execution options:
1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks
2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
