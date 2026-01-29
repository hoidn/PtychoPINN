# ReduceLROnPlateau for Torch Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add ReduceLROnPlateau support to PyTorch training (including FNO/Hybrid) and run an A/B test to see if it reduces loss spikes.

**Architecture:** Wire a new scheduler option through Torch configs/CLIs to `PtychoPINN_Lightning.configure_optimizers`, returning Lightning’s expected dict with `monitor=self.val_loss_name`. Keep defaults unchanged; only activate when scheduler is set to `ReduceLROnPlateau`.

**Tech Stack:** PyTorch Lightning, ptycho_torch configs, grid-lines runner, pytest.

---

### Task 1: Surface ReduceLROnPlateau in config + CLI (grid-lines path)

**Files:**
- Modify: `ptycho_torch/config_params.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Test: `tests/test_grid_lines_compare_wrapper.py`

**Step 1: Write the failing test**

Add a new test to ensure the compare wrapper accepts and forwards the scheduler:

```python
# tests/test_grid_lines_compare_wrapper.py

def test_wrapper_accepts_plateau_scheduler(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--architectures", "hybrid",
        "--torch-scheduler", "ReduceLROnPlateau",
    ])
    assert args.torch_scheduler == "ReduceLROnPlateau"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_accepts_plateau_scheduler -v`
Expected: FAIL with argparse choice error (scheduler not accepted).

**Step 3: Write minimal implementation**

- Update `ptycho_torch/config_params.py` scheduler literal to include `ReduceLROnPlateau`:

```python
scheduler: Literal[
    'Default', 'Exponential', 'MultiStage', 'Adaptive', 'WarmupCosine', 'ReduceLROnPlateau'
] = 'Default'
```

- Update CLI choices:
  - `scripts/studies/grid_lines_torch_runner.py` `--scheduler` choices list to include `ReduceLROnPlateau`.
  - `scripts/studies/grid_lines_compare_wrapper.py` `--torch-scheduler` choices list to include `ReduceLROnPlateau`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_accepts_plateau_scheduler -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add ptycho_torch/config_params.py scripts/studies/grid_lines_torch_runner.py scripts/studies/grid_lines_compare_wrapper.py tests/test_grid_lines_compare_wrapper.py
git commit -m "feat: allow ReduceLROnPlateau in grid-lines scheduler flags"
```

---

### Task 2: Implement ReduceLROnPlateau in configure_optimizers

**Files:**
- Modify: `ptycho_torch/model.py`
- Test: `tests/torch/test_model_training.py`

**Step 1: Write the failing test**

Add a test that instantiates a minimal `PtychoPINN_Lightning` and checks the scheduler dict:

```python
# tests/torch/test_model_training.py

def test_configure_optimizers_supports_plateau(tmp_path):
    from ptycho_torch.model import PtychoPINN_Lightning
    from ptycho_torch.config_params import ModelConfig, DataConfig, TrainingConfig, InferenceConfig
    from ptycho.config.config import update_legacy_dict
    from ptycho import params

    model_cfg = ModelConfig(N=8, gridsize=1)
    data_cfg = DataConfig()
    train_cfg = TrainingConfig(
        train_data_file=tmp_path / "train.npz",
        test_data_file=tmp_path / "test.npz",
        output_dir=tmp_path,
        scheduler="ReduceLROnPlateau",
    )
    infer_cfg = InferenceConfig(output_dir=tmp_path)
    update_legacy_dict(params.cfg, train_cfg)

    module = PtychoPINN_Lightning(
        model_config=model_cfg,
        data_config=data_cfg,
        training_config=train_cfg,
        inference_config=infer_cfg,
    )
    result = module.configure_optimizers()
    sched = result["lr_scheduler"]
    assert isinstance(sched, dict)
    assert sched["monitor"] == module.val_loss_name
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_model_training.py::test_configure_optimizers_supports_plateau -v`
Expected: FAIL (no ReduceLROnPlateau branch / missing monitor).

**Step 3: Write minimal implementation**

In `ptycho_torch/model.py` (inside `configure_optimizers`):

```python
elif scheduler_choice == 'ReduceLROnPlateau':
    result['lr_scheduler'] = {
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-4,
        ),
        'monitor': self.val_loss_name,
        'interval': 'epoch',
        'frequency': 1,
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_model_training.py::test_configure_optimizers_supports_plateau -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add ptycho_torch/model.py tests/torch/test_model_training.py
git commit -m "feat: add ReduceLROnPlateau scheduler for torch training"
```

---

### Task 3: Ensure general Torch CLI surfaces the scheduler

**Files:**
- Modify: `scripts/training/train.py`
- Test: `tests/scripts/test_training_backend_selector.py`

**Step 1: Write the failing test**

Add to `tests/scripts/test_training_backend_selector.py` a check that `--torch-scheduler ReduceLROnPlateau` is accepted and passed into execution config (mirror existing Exponential test pattern):

```python
# tests/scripts/test_training_backend_selector.py

def test_torch_scheduler_plateau_roundtrip(monkeypatch, tmp_path):
    # follow existing scheduler tests; assert exec_args_passed.scheduler == 'ReduceLROnPlateau'
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/scripts/test_training_backend_selector.py::test_torch_scheduler_plateau_roundtrip -v`
Expected: FAIL if choice missing or not threaded.

**Step 3: Write minimal implementation**

In `scripts/training/train.py`, ensure `--torch-scheduler` choices include `ReduceLROnPlateau` (if not already present). No other behavior change needed.

**Step 4: Run test to verify it passes**

Run: `pytest tests/scripts/test_training_backend_selector.py::test_torch_scheduler_plateau_roundtrip -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/training/train.py tests/scripts/test_training_backend_selector.py
git commit -m "test: add ReduceLROnPlateau CLI scheduler roundtrip"
```

---

### Task 4: A/B test for FNO/Hybrid loss spikes

**Files / Paths:**
- Runs: `outputs/grid_lines_stage_a/arm_control` and `outputs/grid_lines_stage_a/arm_plateau`
- Artifacts: `plans/active/FNO-STABILITY-OVERHAUL-001/reports/<timestamp>/`

**Step 1: Prepare datasets**

```bash
rsync -a outputs/grid_lines_stage_a/arm_control/datasets/ outputs/grid_lines_stage_a/arm_plateau/datasets/
rm -rf outputs/grid_lines_stage_a/arm_plateau/runs
```

**Step 2: Baseline run (constant LR)**

```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 --gridsize 1 \
  --output-dir outputs/grid_lines_stage_a/arm_control \
  --architectures hybrid \
  --seed 20260128 \
  --nimgs-train 1 --nimgs-test 1 --nphotons 1e9 \
  --nepochs 20 --torch-epochs 20 \
  --torch-scheduler Default \
  --torch-loss-mode mae --fno-blocks 4 --torch-infer-batch-size 8 \
  2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/<timestamp>/stage_a_arm_control_default.log
```

**Step 3: Plateau run**

```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 --gridsize 1 \
  --output-dir outputs/grid_lines_stage_a/arm_plateau \
  --architectures hybrid \
  --seed 20260128 \
  --nimgs-train 1 --nimgs-test 1 --nphotons 1e9 \
  --nepochs 20 --torch-epochs 20 \
  --torch-scheduler ReduceLROnPlateau \
  --torch-loss-mode mae --fno-blocks 4 --torch-infer-batch-size 8 \
  2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/<timestamp>/stage_a_arm_control_plateau.log
```

**Step 4: Summarize**
- Compare loss curves and note whether spikes are reduced/delayed.
- Update `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` and `docs/strategy/mainstrategy.md` with the outcome.

**Step 5: Commit**

```bash
git add plans/active/FNO-STABILITY-OVERHAUL-001/reports/<timestamp>/*.log \
  plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md docs/strategy/mainstrategy.md
git commit -m "report: plateau scheduler A/B run"
```

---

## Execution Notes
- Follow @superpowers:test-driven-development for each task.
- Keep commits small and focused after each task.
- Use `pytest <test> -v` for the exact tests listed above.

