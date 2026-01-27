# Gradient Norm Logging for Hybrid/FNO Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional gradient-norm logging for PyTorch Hybrid/FNO training and run controlled experiments (clip vs no-clip) to confirm whether gradients exceed the clip threshold near the reported blow‑up.

**Architecture:** Add a small grad‑norm utility, wire it into `PtychoPINN_Lightning.training_step` behind a debug flag, and surface that flag in the grid‑lines torch runner. Use the existing CSV logger to record `grad_norm_preclip` (and `grad_norm_postclip` when clipping is enabled), then run two comparable trainings to capture evidence.

**Tech Stack:** PyTorch + Lightning, `ptycho_torch/` training stack, grid‑lines study runner, pytest.

---

### Task 1: Add a grad‑norm utility with tests

**Files:**
- Modify: `ptycho_torch/train_utils.py`
- Create: `tests/torch/test_grad_norm_utils.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_grad_norm_utils.py
import torch
from ptycho_torch.train_utils import compute_grad_norm


def test_compute_grad_norm_l2_matches_manual():
    lin = torch.nn.Linear(4, 2, bias=False)
    for p in lin.parameters():
        p.grad = torch.ones_like(p)
    expected = (p.grad.numel() ** 0.5)  # L2 norm of all-ones
    got = compute_grad_norm(lin.parameters(), norm_type=2.0)
    assert abs(got - expected) < 1e-6
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grad_norm_utils.py -v`
Expected: FAIL with `ImportError` or `AttributeError` (function missing).

**Step 3: Write minimal implementation**

```python
# ptycho_torch/train_utils.py
import torch
from typing import Iterable


def compute_grad_norm(parameters: Iterable[torch.nn.Parameter], norm_type: float = 2.0) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(norm_type)
        total += param_norm.item() ** norm_type
    return total ** (1.0 / norm_type) if total > 0.0 else 0.0
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_grad_norm_utils.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/torch/test_grad_norm_utils.py ptycho_torch/train_utils.py
git commit -m "test: add grad norm utility"
```

---

### Task 2: Wire grad‑norm logging into Lightning training

**Files:**
- Modify: `ptycho_torch/model.py`
- Modify: `ptycho_torch/config_params.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_grad_norm_logging_flag.py
from ptycho_torch.config_params import TrainingConfig


def test_training_config_has_grad_norm_flags():
    cfg = TrainingConfig()
    assert hasattr(cfg, "log_grad_norm")
    assert hasattr(cfg, "grad_norm_log_freq")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grad_norm_logging_flag.py -v`
Expected: FAIL (fields missing).

**Step 3: Add config flags**

```python
# ptycho_torch/config_params.py (TrainingConfig)
log_grad_norm: bool = False
# log every N steps to reduce noise; 1 = every step
grad_norm_log_freq: int = 1
```

**Step 4: Implement logging in training_step**

```python
# ptycho_torch/model.py (PtychoPINN_Lightning.training_step)
from ptycho_torch.train_utils import compute_grad_norm

# after self.manual_backward(scaled_loss)
if self.training_config.log_grad_norm and (batch_idx % self.training_config.grad_norm_log_freq == 0):
    pre_clip = compute_grad_norm(self.parameters(), norm_type=2.0)
    self.log("grad_norm_preclip", pre_clip, on_step=True, on_epoch=True, prog_bar=False, logger=True)

# existing clip block
if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.gradient_clip_val, norm_type=2.0)
    if self.training_config.log_grad_norm and (batch_idx % self.training_config.grad_norm_log_freq == 0):
        post_clip = compute_grad_norm(self.parameters(), norm_type=2.0)
        self.log("grad_norm_postclip", post_clip, on_step=True, on_epoch=True, prog_bar=False, logger=True)
```

**Step 5: Run tests to verify pass**

Run: `pytest tests/torch/test_grad_norm_logging_flag.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add ptycho_torch/config_params.py ptycho_torch/model.py tests/torch/test_grad_norm_logging_flag.py
git commit -m "feat: add optional grad norm logging flags"
```

---

### Task 3: Expose grad‑norm logging in grid‑lines torch runner

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_grid_lines_torch_runner_grad_norm_flag.py
from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig


def test_runner_config_supports_grad_norm_logging():
    cfg = TorchRunnerConfig(train_npz="/tmp/train.npz", test_npz="/tmp/test.npz", output_dir="/tmp/out", architecture="hybrid")
    assert hasattr(cfg, "log_grad_norm")
    assert hasattr(cfg, "grad_norm_log_freq")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grid_lines_torch_runner_grad_norm_flag.py -v`
Expected: FAIL (fields missing).

**Step 3: Update runner config + CLI flags**

```python
# scripts/studies/grid_lines_torch_runner.py
@dataclass
class TorchRunnerConfig:
    ...
    log_grad_norm: bool = False
    grad_norm_log_freq: int = 1

# argparse
parser.add_argument("--log-grad-norm", action="store_true", help="Log gradient norms during torch training")
parser.add_argument("--grad-norm-log-freq", type=int, default=1, help="Log grad norms every N steps")

# wiring to TrainingConfig
training_config = TrainingConfig(
    ...,
    log_grad_norm=cfg.log_grad_norm,
    grad_norm_log_freq=cfg.grad_norm_log_freq,
)
```

Also update `scripts/studies/grid_lines_compare_wrapper.py` to pass through:

```python
# new args
parser.add_argument("--torch-log-grad-norm", action="store_true")
parser.add_argument("--torch-grad-norm-log-freq", type=int, default=1)

# in run call
TorchRunnerConfig(...,
    log_grad_norm=torch_log_grad_norm,
    grad_norm_log_freq=torch_grad_norm_log_freq,
)
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/torch/test_grid_lines_torch_runner_grad_norm_flag.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py scripts/studies/grid_lines_compare_wrapper.py \
  tests/torch/test_grid_lines_torch_runner_grad_norm_flag.py
git commit -m "feat: expose grad norm logging in grid-lines torch runner"
```

---

### Task 4: Run comparison experiments (clip vs no‑clip) with grad‑norm logging

**Files:**
- Create: `.artifacts/grad_norm_runs/` (logs)
- Create: `tmp/grad_norm_summary.json` (temporary; delete before commit)

**Step 1: Run unclipped hybrid training (expect spike)**

Run:
```bash
python scripts/studies/grid_lines_torch_runner.py \
  --train-npz outputs/grid_lines_gs1_n128_e50_phi_all_rerun1/datasets/N128/gs1/train.npz \
  --test-npz outputs/grid_lines_gs1_n128_e50_phi_all_rerun1/datasets/N128/gs1/test.npz \
  --output-dir outputs/grad_norm_hybrid_unclipped \
  --architecture hybrid \
  --epochs 50 \
  --grad-clip 0 \
  --log-grad-norm \
  --grad-norm-log-freq 1
```
Expected: Training completes; CSV logger includes `grad_norm_preclip` values; likely loss spike near epoch ~30.

**Step 2: Run clipped hybrid training (clip=1)**

Run:
```bash
python scripts/studies/grid_lines_torch_runner.py \
  --train-npz outputs/grid_lines_gs1_n128_e50_phi_all_rerun1/datasets/N128/gs1/train.npz \
  --test-npz outputs/grid_lines_gs1_n128_e50_phi_all_rerun1/datasets/N128/gs1/test.npz \
  --output-dir outputs/grad_norm_hybrid_clip1 \
  --architecture hybrid \
  --epochs 50 \
  --grad-clip 1 \
  --log-grad-norm \
  --grad-norm-log-freq 1
```
Expected: Grad norms should be capped near 1 after clipping; loss plateaus higher than unclipped best.

**Step 3: Extract grad‑norm statistics**

Run:
```bash
python - <<'PY'
import json
from pathlib import Path

def load_csv(log_dir):
    # find the lightning CSV metrics file
    csvs = list(Path(log_dir).rglob("metrics.csv"))
    if not csvs:
        raise SystemExit(f"No metrics.csv under {log_dir}")
    return csvs[0]

for label, outdir in {
    "unclipped": "outputs/grad_norm_hybrid_unclipped",
    "clip1": "outputs/grad_norm_hybrid_clip1",
}.items():
    csv = load_csv(outdir)
    print(label, csv)
PY
```
Then use a small script to compute max/median `grad_norm_preclip` and confirm whether `grad_norm_preclip` exceeds the clip threshold (1.0) after epoch ~30. Save the summary to `.artifacts/grad_norm_runs/summary.json`.

**Step 4: Archive logs**

Copy the run logs and CSV metrics into `.artifacts/grad_norm_runs/` and link them in the plan summary.

**Step 5: Commit (docs only, no artifacts)**

```bash
git add docs/plans/2026-01-27-gradient-norm-logging-hybrid-fno.md
# no commit if code already committed in tasks 1–3
```

---

### Task 5: Update plan summary with findings

**Files:**
- Modify: `docs/plans/2026-01-27-gradient-norm-logging-hybrid-fno.md`

**Step 1: Write summary paragraph**

Include:
- Whether `grad_norm_preclip` exceeds clip threshold after ~epoch 30.
- Whether `grad_norm_postclip` stays near threshold.
- Loss curve comparison (best unclipped vs clipped plateau).
- Link to `.artifacts/grad_norm_runs/summary.json` and CSV files.

**Step 2: Commit**

```bash
git add docs/plans/2026-01-27-gradient-norm-logging-hybrid-fno.md
git commit -m "docs: summarize grad norm logging findings"
```

---

## Test Plan

- Unit tests:
  - `pytest tests/torch/test_grad_norm_utils.py -v`
  - `pytest tests/torch/test_grad_norm_logging_flag.py -v`
  - `pytest tests/torch/test_grid_lines_torch_runner_grad_norm_flag.py -v`

- Evidence/logging:
  - Save pytest output logs under `.artifacts/grad_norm_runs/` per `docs/TESTING_GUIDE.md`.

---

## Notes / Constraints

- Do **not** modify core TF modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Keep grad‑norm logging disabled by default to avoid noisy logs in normal runs.
- Store large artifacts in `.artifacts/` and link them in the plan summary.

---

Plan complete and saved to `docs/plans/2026-01-27-gradient-norm-logging-hybrid-fno.md`. Two execution options:

1. Subagent-Driven (this session) — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. Parallel Session (separate) — Open a new session with executing-plans, batch execution with checkpoints.

Which approach?

---

## Findings (2026-01-27)

- Grad norms are consistently above the clip threshold (1.0). In the unclipped run, `grad_norm_preclip_step` exceeds 1.0 for ~97.8% of steps and spikes to 134,725; in the clipped run, pre‑clip exceeds 1.0 for ~99.1% of steps and post‑clip norms sit at ~1.0 by construction. Evidence: `.artifacts/grad_norm_runs/summary.json`, `.artifacts/grad_norm_runs/metrics_unclipped.csv`, `.artifacts/grad_norm_runs/metrics_clip1.csv`.
- Both grid_lines_torch_runner runs complete training but fail during metrics with `AssertionError` in `ptycho/evaluation.py:514` (stitched vs ground truth shape mismatch). Training logs and grad‑norm metrics are still captured in `training_outputs/lightning_logs/version_121` (unclipped) and `version_122` (clip1). Logs: `.artifacts/grad_norm_runs/train_unclipped.log`, `.artifacts/grad_norm_runs/train_clip1.log`.
