# FNO Activation Debugging Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a one-off PyTorch script to capture activation/branch statistics for FNO/Hybrid models and emit a JSON report for diagnosing training stagnation.

**Architecture:** Implement a standalone debug script (`scripts/debug_fno_activations.py`) that loads a small NPZ batch, builds the FNO generator via the registry, attaches forward hooks to lifter and PtychoBlock paths, computes robust statistics (percentiles and spectral low-frequency energy), and writes a JSON report under `.artifacts/` with minimal logging.

**Tech Stack:** Python, PyTorch (torch>=2.2), NumPy, existing PyTorch generator registry, pytest.

---

### Task 1: Add a one-off activation debugging script

**Files:**
- Create: `scripts/debug_fno_activations.py`
- Test: `tests/torch/test_debug_fno_activations.py`

**Step 1: Write the failing test**

Create `tests/torch/test_debug_fno_activations.py`:

```python
import json
import subprocess
from pathlib import Path

import numpy as np


def test_debug_fno_activations_emits_report(tmp_path: Path) -> None:
    """Smoke test: script writes a JSON report with expected keys."""
    npz_path = tmp_path / "tiny.npz"
    diffraction = np.random.rand(2, 16, 16).astype(np.float32)
    np.savez(npz_path, diffraction=diffraction)

    out_dir = tmp_path / "out"
    cmd = [
        "python",
        "scripts/debug_fno_activations.py",
        "--input",
        str(npz_path),
        "--output",
        str(out_dir),
        "--architecture",
        "fno",
        "--batch-size",
        "1",
        "--max-samples",
        "1",
        "--device",
        "cpu",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    report_path = out_dir / "activation_report.json"
    assert report_path.exists(), "activation_report.json not written"
    report = json.loads(report_path.read_text())
    assert "layers" in report
    assert "summary" in report
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_debug_fno_activations.py -vv`

Expected: FAIL with `FileNotFoundError` or `No such file or directory` (script missing).

**Step 3: Write minimal implementation**

Create `scripts/debug_fno_activations.py`:

```python
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ptycho.config.config import ModelConfig, TrainingConfig, update_legacy_dict
from ptycho import params
from ptycho_torch.generators.registry import resolve_generator
from ptycho.log_config import setup_logging


class ActivationMonitor:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.stats = defaultdict(list)
        self.hooks = []

    def _record(self, key: str, value: float) -> None:
        self.stats[key].append(float(value))

    def _tensor_stats(self, name: str, tensor: torch.Tensor) -> None:
        t = tensor.detach()
        abs_t = t.abs()
        self._record(f"{name}/mean", abs_t.mean().item())
        self._record(f"{name}/std", abs_t.std().item())
        self._record(f"{name}/p50", abs_t.median().item())
        self._record(f"{name}/p95", torch.quantile(abs_t, 0.95).item())
        self._record(f"{name}/p99", torch.quantile(abs_t, 0.99).item())

    def _low_freq_ratio(self, name: str, tensor: torch.Tensor) -> None:
        # Ratio of low-frequency energy to total energy (per-batch mean)
        t = tensor.detach()
        if t.ndim != 4:
            return
        x_ft = torch.fft.rfft2(t, norm="ortho")
        power = (x_ft.real ** 2 + x_ft.imag ** 2)
        h = power.shape[-2]
        w = power.shape[-1]
        low_h = max(1, h // 8)
        low_w = max(1, w // 8)
        low = power[..., :low_h, :low_w].mean()
        total = power.mean()
        if total > 0:
            self._record(f"{name}/low_freq_ratio", (low / total).item())

    def _hook_fn(self, module, inputs, output, name: str) -> None:
        if not isinstance(output, torch.Tensor):
            return
        self._tensor_stats(name, output)
        self._low_freq_ratio(name, output)
        if isinstance(module, (torch.nn.ReLU, torch.nn.GELU)):
            sparsity = (output <= 0).float().mean().item()
            self._record(f"{name}/sparsity", sparsity)

    def register(self) -> None:
        for module_name, module in self.model.named_modules():
            if any(key in module_name for key in ("lifter", "spectral", "local_conv", "PtychoBlock")):
                hook = module.register_forward_hook(
                    lambda m, i, o, n=module_name: self._hook_fn(m, i, o, n)
                )
                self.hooks.append(hook)

    def close(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FNO activation debug capture")
    parser.add_argument("--input", required=True, help="Path to train.npz")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--architecture", default="fno", choices=["fno", "hybrid"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    with np.load(args.input, allow_pickle=True) as data:
        if "diffraction" not in data:
            raise KeyError("NPZ missing diffraction array")
        X = np.asarray(data["diffraction"], dtype=np.float32)

    if X.ndim == 3:
        X = X[..., np.newaxis]

    n_samples = min(args.max_samples, X.shape[0])
    X = X[:n_samples]

    model_config = ModelConfig(
        N=X.shape[1],
        gridsize=1,
        architecture=args.architecture,
    )
    training_config = TrainingConfig(model=model_config)
    update_legacy_dict(params.cfg, training_config)

    generator = resolve_generator(model_config)
    model = generator.model
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    monitor = ActivationMonitor(model)
    monitor.register()

    with torch.no_grad():
        batch = torch.from_numpy(X).permute(0, 3, 1, 2).to(device)
        _ = model(batch)

    monitor.close()

    report = {
        "summary": {
            "architecture": args.architecture,
            "n_samples": int(n_samples),
            "batch_size": int(args.batch_size),
        },
        "layers": {k: v for k, v in monitor.stats.items()},
    }

    report_path = out_dir / "activation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_debug_fno_activations.py -vv`

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/debug_fno_activations.py tests/torch/test_debug_fno_activations.py
git commit -m "add fno activation debug script"
```

---

### Task 2: Execute one-off run and archive artifacts (manual)

**Files:**
- Create: `.artifacts/debug_fno_activations/activation_report.json`

**Step 1: Run the one-off script**

Run:
```bash
python scripts/debug_fno_activations.py \
  --input /path/to/train.npz \
  --output .artifacts/debug_fno_activations \
  --architecture fno \
  --batch-size 1 \
  --max-samples 8 \
  --device cpu
```

Expected: `.artifacts/debug_fno_activations/activation_report.json` created.

**Step 2: Sanity-check output**

Open the JSON and confirm:
- `layers` contains entries for `lifter`, `spectral`, `local_conv`, and `PtychoBlock` outputs
- `low_freq_ratio` values exist for lifter outputs

**Step 3: Commit (optional)**

Do not commit `.artifacts/` outputs. If a summary is needed, add a short note to the active plan or a local log file.
```
