# FNO/Hybrid Convergence Hypothesis Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Determine why FNO/Hybrid training stagnates by validating three concrete hypotheses (spectral gradient collapse, loss unit mismatch, and input dynamic-range collapse) using targeted diagnostics and unit tests before any remediation.

**Architecture:** Add two small debug scripts and unit tests: (1) a gradient‑ratio probe for the FNO fallback spectral conv path, (2) unit tests asserting Poisson/MAE loss units, and (3) use the existing activation monitor on the grid‑lines cached dataset. Do not change model behavior until evidence confirms a root cause.

**Tech Stack:** Python, PyTorch (torch>=2.2), pytest, NumPy, existing FNO generator components.

---

## Hypotheses (Audit Targets)

1) **Spectral gradient collapse (fallback spectral conv)**
- *Theory:* `_FallbackSpectralConv2d` weights are scaled by `1/(in_channels*out_channels)` and produce tiny spectral outputs; gradients for spectral weights may be << local conv gradients, causing the model to behave like a CNN.
- *Evidence required:* spectral weight gradient norms and spectral/local output ratios from a deterministic diagnostic.

2) **Loss unit mismatch (Poisson/MAE)**
- *Theory:* Poisson loss should operate on **intensity** (squared amplitude). If the loss is applied directly to amplitude, gradients can be distorted and physics consistency violated. MAE should be clearly defined as amplitude‑space or intensity‑space, not a mix.
- *Evidence required:* unit tests that lock in expected behavior for Poisson and MAE.

3) **Input dynamic‑range collapse (DC dominance)**
- *Theory:* Diffraction amplitudes have large DC peaks; the lifter is unnormalized, which could overpower spectral mixing and suppress high‑frequency features.
- *Evidence required:* activation statistics and low‑frequency energy ratios on real grid‑lines data.

---

## Task 1: Add gradient‑ratio diagnostic script

**Files:**
- Create: `scripts/debug_fno_gradients.py`
- Test: `tests/torch/test_debug_fno_gradients.py`

**Step 1: Write the failing test**

Create `tests/torch/test_debug_fno_gradients.py`:

```python
import json
import subprocess
from pathlib import Path


def test_debug_fno_gradients_emits_report(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    cmd = [
        "python",
        "scripts/debug_fno_gradients.py",
        "--output",
        str(out_dir),
        "--channels",
        "32",
        "--modes",
        "12",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    report_path = out_dir / "gradient_report.json"
    assert report_path.exists(), "gradient_report.json not written"
    report = json.loads(report_path.read_text())
    assert "spectral_grad_mean" in report
    assert "local_grad_mean" in report
    assert "spectral_local_ratio" in report
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_debug_fno_gradients.py -vv`

Expected: FAIL (script missing).

**Step 3: Write minimal implementation**

Create `scripts/debug_fno_gradients.py`:

```python
import argparse
import json
from pathlib import Path

import torch

from ptycho_torch.generators.fno import PtychoBlock
from ptycho.log_config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FNO gradient diagnostic")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    torch.manual_seed(args.seed)
    block = PtychoBlock(channels=args.channels, modes=args.modes)
    block.train()

    x = torch.randn(2, args.channels, 32, 32, requires_grad=True)
    y = block(x)
    loss = y.abs().mean()
    loss.backward()

    spectral_grad = block.spectral.weights.grad.abs().mean().item()
    local_grad = block.local_conv.weight.grad.abs().mean().item()
    ratio = spectral_grad / local_grad if local_grad else None

    report = {
        "spectral_grad_mean": spectral_grad,
        "local_grad_mean": local_grad,
        "spectral_local_ratio": ratio,
    }
    (out_dir / "gradient_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_debug_fno_gradients.py -vv`

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/debug_fno_gradients.py tests/torch/test_debug_fno_gradients.py
git commit -m "add fno gradient debug script"
```

---

## Task 2: Loss unit tests (Poisson + MAE)

**Files:**
- Create: `tests/torch/test_loss_units.py`

**Step 1: Write the failing tests**

Create `tests/torch/test_loss_units.py`:

```python
import torch

from ptycho_torch.model import PoissonIntensityLayer, MAELoss


def test_poisson_loss_uses_intensity() -> None:
    # A_pred=2 => I_pred=4; A_obs=3 => I_obs=9
    pred_amp = torch.tensor([2.0])
    obs_amp = torch.tensor([3.0])
    layer = PoissonIntensityLayer()
    loss = layer(pred_amp, obs_amp)

    expected = torch.distributions.Poisson(4.0).log_prob(torch.tensor(9.0)).neg()
    assert torch.isclose(loss, expected), (loss.item(), expected.item())


def test_mae_loss_operates_on_amplitude() -> None:
    pred_amp = torch.tensor([2.0])
    obs_amp = torch.tensor([3.0])
    loss_fn = MAELoss()
    loss = loss_fn(pred_amp, obs_amp)
    assert torch.isclose(loss, torch.tensor(1.0)), loss.item()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_loss_units.py -vv`

Expected: FAIL if implementation does not match the asserted unit behavior.

**Step 3: Implement minimal fixes (only if tests fail)**

If Poisson test fails, update `ptycho_torch/model.py` to square amplitudes before Poisson log‑prob. If MAE test fails, align MAE to the desired unit (amplitude) and update the test expectation if spec dictates intensity. Do not change both without evidence.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/torch/test_loss_units.py -vv`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/torch/test_loss_units.py ptycho_torch/model.py
git commit -m "test torch loss units" 
```

---

## Task 3: Run activation monitor on grid‑lines dataset

**Files:**
- Use: `scripts/debug_fno_activations.py`
- Output: `.artifacts/debug_fno_activations/activation_report.json`

**Step 1: Run on cached grid‑lines data**

```bash
python scripts/debug_fno_activations.py \
  --input outputs/grid_lines_gs1_n64_e5/datasets/N64/gs1/train.npz \
  --output .artifacts/debug_fno_activations \
  --architecture fno \
  --batch-size 1 \
  --max-samples 8 \
  --device cpu
```

**Step 2: Evaluate thresholds**

- **Spectral/local ratio**: flag if < 0.1
- **Lifter tail ratio**: p99/p50 > 5 indicates heavy‑tail persistence
- **Low‑freq dominance**: low_freq_ratio ≫ 1 indicates DC dominance

Do not implement changes here; only record evidence.

---

## Task 4: Decision gate (no code changes)

**Outcome matrix:**
- If **spectral gradients** are <10% of local: target spectral weight scaling.
- If **loss unit tests fail**: fix Poisson/MAE units.
- If **activation dynamics show DC dominance**: consider log1p/InstanceNorm only after confirming loss units and gradients are correct.

Document the decision and evidence in `.artifacts/` or a short note in a plan summary.
```
