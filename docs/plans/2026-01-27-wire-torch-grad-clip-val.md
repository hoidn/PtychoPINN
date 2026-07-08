# Wire Torch Runner Gradient Clip to TrainingConfig Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ensure `--torch-grad-clip` from grid-lines wrappers actually applies model-level gradient clipping during Lightning manual-optimization training.

**Architecture:** `grid_lines_torch_runner.setup_torch_configs()` builds `TrainingConfig` and `PyTorchExecutionConfig`. Manual optimization in `ptycho_torch.model` uses `TrainingConfig.gradient_clip_val`, so we must propagate the CLI value into `TrainingConfig` (not just `ExecutionConfig`).

**Tech Stack:** Python, PyTorch/Lightning, pytest

---

### Task 1: Add a failing unit test for grad-clip propagation

**Files:**
- Modify: `tests/torch/test_grid_lines_torch_runner_grad_norm_flag.py`

**Step 1: Write the failing test**

```python
from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig, setup_torch_configs


def test_runner_config_propagates_grad_clip_to_training_config():
    cfg = TorchRunnerConfig(
        train_npz="/tmp/train.npz",
        test_npz="/tmp/test.npz",
        output_dir="/tmp/out",
        architecture="hybrid",
        gradient_clip_val=50.0,
    )
    training_config, execution_config = setup_torch_configs(cfg)
    assert training_config.gradient_clip_val == 50.0
    assert execution_config.gradient_clip_val == 50.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grid_lines_torch_runner_grad_norm_flag.py::test_runner_config_propagates_grad_clip_to_training_config -v | tee .artifacts/grad_clip_wire/pytest_red.log`

Expected: FAIL because `training_config.gradient_clip_val` is `None`.

---

### Task 2: Wire `gradient_clip_val` into TrainingConfig

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`

**Step 1: Implement minimal fix**

```python
training_config = TrainingConfig(
    ...,
    torch_loss_mode=cfg.torch_loss_mode,
    gradient_clip_val=cfg.gradient_clip_val,
)
```

(Alternatively, assign immediately after construction: `training_config.gradient_clip_val = cfg.gradient_clip_val`.)

**Step 2: Re-run unit test**

Run: `pytest tests/torch/test_grid_lines_torch_runner_grad_norm_flag.py::test_runner_config_propagates_grad_clip_to_training_config -v | tee .artifacts/grad_clip_wire/pytest_green.log`

Expected: PASS.

**Step 3: Commit**

```bash
git add tests/torch/test_grid_lines_torch_runner_grad_norm_flag.py scripts/studies/grid_lines_torch_runner.py
git commit -m "fix: wire torch grad clip into training config"
```

---

### Task 3: Required integration evidence (policy)

**Files:**
- None (evidence only)

**Step 1: Run integration marker**

Run: `pytest -v -m integration | tee .artifacts/grad_clip_wire/pytest_integration.log`

Expected: PASS.

**Step 2: Record evidence**

Ensure the log is stored under `.artifacts/grad_clip_wire/` and link it in follow-up notes if required.

---

**Notes / Rationale**
- Lightning Trainer `gradient_clip_val` is disabled for manual optimization; model-level clipping in `ptycho_torch/model.py` uses `TrainingConfig.gradient_clip_val`, so this propagation is required for `--torch-grad-clip` to take effect.

