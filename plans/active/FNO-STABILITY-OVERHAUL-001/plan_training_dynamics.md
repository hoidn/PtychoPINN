# Stable Hybrid Training Dynamics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a warmup+cosine learning rate schedule plus hyperparameter plumbing so `stable_hybrid` can avoid the Stage A collapse diagnosed in STABLE-LS-001, then rerun the Stage A arm with the new schedule.

**Architecture:** Extend the canonical config surfaces (TF + Torch dataclasses, config bridge, CLI wrappers) with explicit LR scheduler knobs, implement a deterministic warmup/cosine scheduler inside `PtychoPINN_Lightning.configure_optimizers()`, and prove the change via unit tests and Stage A reruns that share the cached datasets.

**Tech Stack:** Python 3.11, PyTorch/Lightning, pytest, grid_lines CLI harness, rsync.

---

### Task 1: Surface scheduler knobs across configs + CLI

**Files:**
- Modify: `ptycho/config/config.py:TrainingConfig`
- Modify: `ptycho_torch/config_params.py:TrainingConfig`
- Modify: `ptycho_torch/config_bridge.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` status block later (doc sync happens in Task 3)
- Tests: `tests/torch/test_config_bridge.py`, `tests/torch/test_grid_lines_torch_runner.py`, `tests/test_grid_lines_compare_wrapper.py`

**Step 1: Write the failing tests**

```python
# tests/torch/test_config_bridge.py
class TestConfigBridgeParity:
    def test_training_config_lr_scheduler_roundtrip(self):
        overrides = {
            'learning_rate': 5e-4,
            'scheduler': 'WarmupCosine',
            'lr_warmup_epochs': 5,
            'lr_min_ratio': 0.05,
        }
        config = config_bridge.to_training_config(overrides)
        assert config.learning_rate == 5e-4
        assert config.scheduler == 'WarmupCosine'
        assert config.lr_warmup_epochs == 5
        assert config.lr_min_ratio == 0.05
```

```python
# tests/torch/test_grid_lines_torch_runner.py
class TestTorchRunnerConfig:
    def test_setup_configs_threads_scheduler_fields(self):
        cfg = TorchRunnerConfig(
            # existing fields ...,
            learning_rate=5e-4,
            scheduler='WarmupCosine',
            lr_warmup_epochs=5,
            lr_min_ratio=0.05,
        )
        training_cfg, exec_cfg = setup_torch_configs(cfg)
        assert training_cfg.learning_rate == 5e-4
        assert training_cfg.scheduler == 'WarmupCosine'
        assert training_cfg.lr_warmup_epochs == 5
        assert training_cfg.lr_min_ratio == 0.05
```

```python
# tests/test_grid_lines_compare_wrapper.py
@patch("subprocess.run")
def test_wrapper_passes_scheduler_knobs(mock_run, tmp_path):
    args = [
        "--N", "64", "--gridsize", "1",
        "--output-dir", str(tmp_path / "out"),
        "--architectures", "stable_hybrid",
        "--torch-scheduler", "WarmupCosine",
        "--torch-lr-warmup-epochs", "5",
        "--torch-lr-min-ratio", "0.05",
    ]
    parse_args(args)
    run_grid_lines_compare(...)
    torch_call = mock_run.call_args[0][0]
    assert "--scheduler" in torch_call
    assert "WarmupCosine" in torch_call
    assert "--lr-warmup-epochs" in torch_call
    assert "5" in torch_call
    assert "--lr-min-ratio" in torch_call
```

**Step 2: Run tests to verify they fail**
- `pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_training_config_lr_scheduler_roundtrip -v`
- `pytest tests/torch/test_grid_lines_torch_runner.py::TestTorchRunnerConfig::test_setup_configs_threads_scheduler_fields -v`
- `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_passes_scheduler_knobs -v`
- **Expected:** All fail because new dataclass fields/flags don’t exist yet.

**Step 3: Write minimal implementation**
- Extend `TrainingConfig` in both `ptycho/config/config.py` and `ptycho_torch/config_params.py`:
  ```python
  scheduler: Literal['Default','Exponential','WarmupCosine'] = 'Default'
  lr_warmup_epochs: int = 0
  lr_min_ratio: float = 0.1  # eta_min = base_lr * ratio
  ```
- Update `config_bridge.to_training_config()` to map the new keys.
- Update `TorchRunnerConfig` (dataclass) + CLI flags in `grid_lines_torch_runner.py`:
  - Add `scheduler`, `lr_warmup_epochs`, `lr_min_ratio`, `enable_checkpointing` fields.
  - Set `training_config.learning_rate = cfg.learning_rate` inside `setup_torch_configs`.
  - Pass scheduler knobs to the returned training config.
- Extend runner CLI options:
  ```python
  parser.add_argument("--scheduler", choices=['Default','Exponential','WarmupCosine'], default='Default')
  parser.add_argument("--lr-warmup-epochs", type=int, default=0)
  parser.add_argument("--lr-min-ratio", type=float, default=0.1)
  parser.add_argument("--disable-checkpointing", action="store_true")
  ```
  Hand these through `TorchRunnerConfig` and into `PyTorchExecutionConfig`.
- Update compare wrapper CLI to add pass-through flags `--torch-scheduler`, `--torch-lr-warmup-epochs`, `--torch-lr-min-ratio` and append them to the `cmd` list that launches the torch runner.

**Step 4: Run tests to verify they pass**
- Repeat the three pytest commands plus the existing regression selectors that touched these files:
  - `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_passes_grad_clip_algorithm -v`
  - `pytest tests/torch/test_grid_lines_torch_runner.py -k gradient_clip -v`

**Step 5: Commit**
```bash
git add ptycho/config/config.py ptycho_torch/config_params.py \
        ptycho_torch/config_bridge.py scripts/studies/grid_lines_torch_runner.py \
        scripts/studies/grid_lines_compare_wrapper.py tests/torch/test_config_bridge.py \
        tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py
git commit -m "feat: surface LR scheduler knobs in configs and CLI"
```

---

### Task 2: Implement Warmup+Cosine scheduler in Lightning

**Files:**
- Create: `ptycho_torch/schedulers.py`
- Modify: `ptycho_torch/model.py`
- Modify: `ptycho_torch/workflows/components.py` (ensure `training_config.scheduler` flows)
- Tests: `tests/torch/test_lr_scheduler.py` (new), `tests/torch/test_model_training.py` (add smoke test for scheduler selection)

**Step 1: Write the failing tests**

```python
# tests/torch/test_lr_scheduler.py
from ptycho_torch.schedulers import build_warmup_cosine_scheduler

def test_warmup_cosine_scheduler_progression():
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = build_warmup_cosine_scheduler(opt, total_epochs=10, warmup_epochs=2, min_lr_ratio=0.05)
    lrs = []
    for epoch in range(10):
        opt.step(); sched.step()
        lrs.append(opt.param_groups[0]['lr'])
    assert math.isclose(max(lrs), 1e-3, rel_tol=1e-5)
    assert lrs[0] < lrs[2]
    assert lrs[-1] == pytest.approx(5e-5, rel=1e-3)
```

```python
# tests/torch/test_model_training.py (new test class)
def test_configure_optimizers_selects_warmup_scheduler(tmp_path):
    model = PtychoPINN_Lightning(...)
    model.training_config.scheduler = 'WarmupCosine'
    model.training_config.lr_warmup_epochs = 5
    result = model.configure_optimizers()
    scheduler_dict = result['lr_scheduler']
    assert scheduler_dict['scheduler'].__class__.__name__ == 'SequentialLR'
    assert scheduler_dict['interval'] == 'epoch'
```

**Step 2: Run tests to verify they fail**
- `pytest tests/torch/test_lr_scheduler.py::test_warmup_cosine_scheduler_progression -v`
- `pytest tests/torch/test_model_training.py::test_configure_optimizers_selects_warmup_scheduler -v`
- Expect both to fail because helper + scheduler path don’t exist.

**Step 3: Write minimal implementation**
- Create `ptycho_torch/schedulers.py`:
  ```python
  import torch
  from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

  def build_warmup_cosine_scheduler(optimizer, total_epochs, warmup_epochs, min_lr_ratio):
      warmup_epochs = max(0, warmup_epochs)
      eta_min = optimizer.param_groups[0]['lr'] * min_lr_ratio
      if warmup_epochs == 0:
          return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min)
      warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
      cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=eta_min)
      return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
  ```
- Update `ptycho_torch/model.py`:
  - Import helper.
  - In `configure_optimizers`, handle `scheduler_choice == 'WarmupCosine'` by attaching scheduler dict:
    ```python
    if scheduler_choice == 'WarmupCosine':
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            total_epochs=self.training_config.epochs,
            warmup_epochs=self.training_config.lr_warmup_epochs,
            min_lr_ratio=self.training_config.lr_min_ratio,
        )
        result['lr_scheduler'] = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }
    ```
  - Ensure `self.lr = training_config.learning_rate` already set; confirm we log learning_rate metric (already done in `on_train_epoch_start`).
- Update `ptycho_torch/workflows/components.py` (or `config_factory`) if needed so that `training_config.learning_rate` respects CLI override (set inside runner earlier).

**Step 4: Run tests to verify they pass**
- Run the two new pytest commands.
- Re-run `pytest tests/torch/test_fno_generators.py::TestStablePtychoBlock -v` to ensure no regressions in touched modules.

**Step 5: Commit**
```bash
git add ptycho_torch/schedulers.py ptycho_torch/model.py tests/torch/test_lr_scheduler.py \
        tests/torch/test_model_training.py ptycho_torch/workflows/components.py
git commit -m "feat: add warmup+cosine LR scheduler"
```

---

### Task 3: Stage A rerun with warmup schedule + doc sync

**Files / Paths:**
- CLI: `scripts/studies/grid_lines_compare_wrapper.py`
- Outputs: `outputs/grid_lines_stage_a/arm_stable_warmup`
- Artifacts: `plans/active/FNO-STABILITY-OVERHAUL-001/reports/<timestamp>/`
- Docs: `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md`, `docs/strategy/mainstrategy.md`, `docs/fix_plan.md`, `docs/findings.md`, `plans/active/FNO-STABILITY-OVERHAUL-001/summary.md`

**Step 1: Prep datasets + README**
- Copy Stage A control datasets to the new arm:
  ```bash
  rsync -a outputs/grid_lines_stage_a/arm_control/datasets/ \
        outputs/grid_lines_stage_a/arm_stable_warmup/datasets/
  rm -rf outputs/grid_lines_stage_a/arm_stable_warmup/runs
  ```
- Create `plans/active/FNO-STABILITY-OVERHAUL-001/reports/<timestamp>/README.md` capturing shared seed, scheduler knobs, and CLI command.

**Step 2: Run LayerScale + warmup arm**
- Command (tee log):
  ```bash
  AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md \
  python scripts/studies/grid_lines_compare_wrapper.py \
    --N 64 --gridsize 1 \
    --output-dir outputs/grid_lines_stage_a/arm_stable_warmup \
    --architectures stable_hybrid \
    --seed 20260128 \
    --nimgs-train 1 --nimgs-test 1 --nphotons 1e9 \
    --torch-epochs 20 --torch-learning-rate 5e-4 \
    --torch-scheduler WarmupCosine \
    --torch-lr-warmup-epochs 5 \
    --torch-lr-min-ratio 0.05 \
    --torch-grad-clip 0.0 --torch-grad-clip-algorithm norm \
    --torch-loss-mode mae --fno-blocks 4 --torch-infer-batch-size 8 \
    2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/<timestamp>/stage_a_arm_stable_warmup.log
  ```
- Expected: Training completes without NaNs; validation loss remains near best value.

**Step 3: Optional low-LR baseline comparison**
- Repeat the run with `--torch-scheduler Default --torch-learning-rate 2.5e-4` into `arm_stable_lowlr` to confirm improvements come from schedule vs. simple LR drop.
- Archive `stage_a_arm_stable_lowlr.log` + run artifacts alongside warmup run.

**Step 4: Archive metrics + stats**
- Copy `history.json`, `metrics.json`, `model.pt` from each new run into the `<timestamp>` hub.
- Generate stats JSON:
  ```bash
  python scripts/internal/stage_a_dump_stats.py \
    --run-dir outputs/grid_lines_stage_a/arm_stable_warmup/runs/pinn_stable_hybrid \
    --out-json plans/active/FNO-STABILITY-OVERHAUL-001/reports/<timestamp>/stage_a_arm_stable_warmup_stats.json
  ```
- Append rows to `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T010000Z/stage_a_metrics.json` (or create `stage_a_metrics_phase6.json`) capturing warmup + low-LR entries.

**Step 5: Update docs + findings + plan**
- Update `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` Phase 6 status.
- Update `docs/strategy/mainstrategy.md` Stage A table with new rows + note whether warmup solved collapse.
- Add Attempts History entry + FSM state to `docs/fix_plan.md`.
- Refresh `docs/findings.md`: close STABLE-LS-001 if resolved or note results; add new finding if needed.
- Regenerate `plans/active/FNO-STABILITY-OVERHAUL-001/summary.md` Turn Summary.
- Tests: run mapped selectors used throughout this initiative:
  - `pytest tests/torch/test_fno_generators.py::TestStablePtychoBlock -v`
  - `pytest tests/torch/test_grid_lines_torch_runner.py::TestChannelGridsizeAlignment::test_runner_accepts_stable_hybrid -v`
  - `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_handles_stable_hybrid -v`
  Archive logs under the same `<timestamp>`.

**Step 6: Commit**
```bash
git add plans/active/FNO-STABILITY-OVERHAUL-001/reports/<timestamp> \
        plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md \
        docs/strategy/mainstrategy.md docs/fix_plan.md docs/findings.md \
        plans/active/FNO-STABILITY-OVERHAUL-001/summary.md
git commit -m "chore: Stage A warmup scheduler run + docs"
```

---

**Plan complete and saved to `docs/plans/2026-01-29-stable-hybrid-training-dynamics.md`. Two execution options:**

1. **Subagent-Driven (this session)** — fire superpowers:subagent-driven-development and attack tasks sequentially with reviews between each commit.
2. **Parallel Session** — open a fresh session/worktree, invoke superpowers:executing-plans, and run through the plan with checkpoints.

Which approach?
