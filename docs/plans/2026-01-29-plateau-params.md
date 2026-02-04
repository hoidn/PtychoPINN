# ReduceLROnPlateau Params for Torch Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add configurable ReduceLROnPlateau support (patience/factor/min_lr/threshold) to PyTorch training and the grid‑lines CLI, then validate it against FNO/Hybrid loss spikes.

**Architecture:** Add plateau parameters to Torch training config, expose CLI flags (grid‑lines + compare wrapper), and wire `PtychoPINN_Lightning.configure_optimizers` to return Lightning’s required scheduler dict with `monitor=self.val_loss_name`.

**Tech Stack:** PyTorch Lightning, ptycho_torch configs, grid‑lines runner, pytest.

---

### Task 1: Add plateau params to Torch training config

**Files:**
- Modify: `ptycho_torch/config_params.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write the failing test**

Add a test ensuring plateau fields round‑trip through the grid‑lines config setup:

```python
# tests/torch/test_grid_lines_torch_runner.py

def test_setup_configs_threads_plateau_params(tmp_path):
    from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig, setup_torch_configs

    cfg = TorchRunnerConfig(
        output_dir=tmp_path,
        scheduler='ReduceLROnPlateau',
        plateau_factor=0.25,
        plateau_patience=5,
        plateau_min_lr=1e-5,
        plateau_threshold=1e-3,
    )
    _, training_cfg, _ = setup_torch_configs(cfg)
    assert training_cfg.scheduler == 'ReduceLROnPlateau'
    assert training_cfg.plateau_factor == 0.25
    assert training_cfg.plateau_patience == 5
    assert training_cfg.plateau_min_lr == 1e-5
    assert training_cfg.plateau_threshold == 1e-3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_setup_configs_threads_plateau_params -v`
Expected: FAIL (missing fields / scheduler option).

**Step 3: Write minimal implementation**

In `ptycho_torch/config_params.py`, add:

```python
scheduler: Literal[
    'Default', 'Exponential', 'MultiStage', 'Adaptive', 'WarmupCosine', 'ReduceLROnPlateau'
] = 'Default'
plateau_factor: float = 0.5
plateau_patience: int = 2
plateau_min_lr: float = 1e-4
plateau_threshold: float = 0.0
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_setup_configs_threads_plateau_params -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add ptycho_torch/config_params.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "feat: add plateau params to torch TrainingConfig"
```

---

### Task 2: Expose plateau flags in grid‑lines CLI + wrapper

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Test: `tests/test_grid_lines_compare_wrapper.py`

**Step 1: Write the failing test**

Add a wrapper parse test:

```python
# tests/test_grid_lines_compare_wrapper.py

def test_wrapper_accepts_plateau_params(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--architectures", "hybrid",
        "--torch-scheduler", "ReduceLROnPlateau",
        "--torch-plateau-factor", "0.25",
        "--torch-plateau-patience", "5",
        "--torch-plateau-min-lr", "1e-5",
        "--torch-plateau-threshold", "1e-3",
    ])
    assert args.torch_scheduler == "ReduceLROnPlateau"
    assert args.torch_plateau_patience == 5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_accepts_plateau_params -v`
Expected: FAIL (unknown args/choices).

**Step 3: Write minimal implementation**

- In `scripts/studies/grid_lines_torch_runner.py`:
  - Add `ReduceLROnPlateau` to `--scheduler` choices.
  - Add CLI args: `--plateau-factor`, `--plateau-patience`, `--plateau-min-lr`, `--plateau-threshold`.
  - Add fields to `TorchRunnerConfig` dataclass and thread them into `training_config` in `setup_torch_configs`.
- In `scripts/studies/grid_lines_compare_wrapper.py`:
  - Add flags `--torch-plateau-*`.
  - Pass them through to the torch runner command list.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_accepts_plateau_params -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py scripts/studies/grid_lines_compare_wrapper.py tests/test_grid_lines_compare_wrapper.py
git commit -m "feat: expose plateau scheduler params for grid-lines"
```

---

### Task 3: Wire ReduceLROnPlateau in Lightning

**Files:**
- Modify: `ptycho_torch/model.py`
- Test: `tests/torch/test_model_training.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_model_training.py

def test_configure_optimizers_supports_plateau_scheduler(tmp_path):
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
        plateau_patience=3,
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

Run: `pytest tests/torch/test_model_training.py::test_configure_optimizers_supports_plateau_scheduler -v`
Expected: FAIL (no plateau branch / missing monitor).

**Step 3: Write minimal implementation**

In `ptycho_torch/model.py`:

```python
elif scheduler_choice == 'ReduceLROnPlateau':
    result['lr_scheduler'] = {
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.training_config.plateau_factor,
            patience=self.training_config.plateau_patience,
            min_lr=self.training_config.plateau_min_lr,
            threshold=self.training_config.plateau_threshold,
        ),
        'monitor': self.val_loss_name,
        'interval': 'epoch',
        'frequency': 1,
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_model_training.py::test_configure_optimizers_supports_plateau_scheduler -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add ptycho_torch/model.py tests/torch/test_model_training.py
git commit -m "feat: add configurable ReduceLROnPlateau scheduler"
```

---

### Task 4: (Optional) Wire plateau params into general training CLI

**Files:**
- Modify: `scripts/training/train.py`
- Test: `tests/scripts/test_training_backend_selector.py`

**Step 1: Write the failing test**

Add a CLI round‑trip test mirroring the existing scheduler tests:

```python
# tests/scripts/test_training_backend_selector.py

def test_torch_scheduler_plateau_params_roundtrip(...):
    # Assert execution_config (or overrides) carry plateau params
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/scripts/test_training_backend_selector.py::test_torch_scheduler_plateau_params_roundtrip -v`
Expected: FAIL.

**Step 3: Write minimal implementation**

- Add CLI flags in `scripts/training/train.py`:
  `--torch-plateau-factor`, `--torch-plateau-patience`, `--torch-plateau-min-lr`, `--torch-plateau-threshold`.
- Thread them into the overrides passed to `create_training_payload()` (so they land in `pt_training_config`).

**Step 4: Run test to verify it passes**

Run: `pytest tests/scripts/test_training_backend_selector.py::test_torch_scheduler_plateau_params_roundtrip -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/training/train.py tests/scripts/test_training_backend_selector.py
git commit -m "feat: expose plateau params in training CLI"
```

---

### Task 5: A/B run for FNO/Hybrid loss spikes

**Files / Paths:**
- Runs: `outputs/grid_lines_stage_a/arm_control`, `outputs/grid_lines_stage_a/arm_plateau`
- Artifacts: `plans/active/FNO-STABILITY-OVERHAUL-001/reports/<timestamp>/`

**Step 1: Prepare datasets**

```bash
rsync -a outputs/grid_lines_stage_a/arm_control/datasets/ outputs/grid_lines_stage_a/arm_plateau/datasets/
rm -rf outputs/grid_lines_stage_a/arm_plateau/runs
```

**Step 2: Baseline (constant LR)**

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
  --torch-plateau-patience 3 \
  --torch-plateau-factor 0.5 \
  --torch-plateau-min-lr 1e-4 \
  --torch-loss-mode mae --fno-blocks 4 --torch-infer-batch-size 8 \
  2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/<timestamp>/stage_a_arm_control_plateau.log
```

**Step 4: Summarize**
- Compare loss curves and crash behavior.
- Update `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` + `docs/strategy/mainstrategy.md` with results.

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

