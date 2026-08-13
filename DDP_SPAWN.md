# DDP Spawn Support

Changes to support both `ddp` (torchrun subprocess) and `ddp_spawn` (mp.spawn) training strategies, enabling PtychoPINN to be embedded in long-running host applications that spawn training jobs from a persistent parent process.

## Motivation

Under `ddp`, each GPU runs as a separate process launched via `torchrun`. This requires an independent script invocation per training run. Under `ddp_spawn`, Lightning calls `mp.spawn()` from the parent process — it pickles the model and data module, forks children, trains, and returns control to the parent. This is the correct model for a library embedded in a host app (e.g., ptychodus) where the application must remain alive across multiple train/predict cycles.

## Usage

Set `training_config.strategy = 'ddp_spawn'` via config or `ConfigManager.update()`. Everything else flows through the existing API.

```python
config_manager.update(training_config={'strategy': 'ddp_spawn', 'n_devices': 2})
```

Under spawn, Lightning manages the process group lifecycle automatically. The parent process's model is updated with trained weights after `.fit()` completes.

For consecutive training runs, create a new `Trainer` instance per run — `L.Trainer` cannot be reused across `.fit()` calls under spawn.

---

## Changes

### 1. Pickle Safety (`model.py`, `beta_modules/model.py`, `config_params.py`)

Spawn pickles the entire `LightningModule` to send to child processes. Several components were not pickle-safe.

**LambdaLayer removal** — `ForwardModel.__init__()` stored five `LambdaLayer(func)` instances wrapping helper functions. These were dead code (the `forward()` method calls the helper functions directly), but their function references prevented pickling. Removed the instances and the `LambdaLayer` class from both `model.py` and `beta_modules/model.py`.

**ProbeIllumination.mask** — Previously stored `model_config.probe_mask` as a plain attribute. If the mask was a GPU tensor, pickle would fail. Changed to `register_buffer('mask', ...)` with `.detach().cpu()`, so PyTorch's state dict machinery handles serialization.

**PoissonLoss** — `forward()` assigned `self.poisson = PoissonIntensityLayer(pred, ...)` on every call, attaching a GPU-tensor-backed distribution object as a persistent attribute. Changed to a local variable to prevent both the memory leak and the pickle hazard.

**ModelConfig.__post_init__** — Added post-init hook that normalizes `probe_mask` to CPU if it arrives as a GPU tensor. This ensures config dataclasses are always pickle-safe regardless of how they're constructed.

### 2. Strategy Abstraction (`train_utils.py`)

**`get_training_strategy()`** — Extended to handle `strategy='ddp_spawn'`. Returns `DDPStrategy(start_method='spawn', find_unused_parameters=False)`, using NCCL for CUDA and Lightning's default backend for CPU. The `static_graph=True` optimization used by the regular DDP path is omitted for spawn because the model is re-created from pickle each time.

**`is_spawn_strategy()`** — New helper that checks whether a strategy (string or object) uses spawn. Used by DataModule and other code to branch behavior.

### 3. Run Coordination (shared Lightning service)

**File-based timestamp synchronization removed** — The old pattern had rank 0 write a `.run_name` file while other ranks polled for it with a 30-second timeout. Under spawn, all children thought they were rank 0 (no `RANK` env var before spawn). Replaced with simple pre-spawn `run_name` generation — anything computed before `trainer.fit()` happens in the parent process only.

**Manual `dist.barrier()` / `dist.destroy_process_group()` removed** — These calls were scattered across the training and exception handling paths. Under spawn, Lightning manages the process group lifecycle automatically. Under DDP, the manual calls created fragility (hangs if any rank crashes). Removed all instances; Lightning handles cleanup.

### 4. Single-Pass Fine-Tuning (`train_utils.py`, `trainer_api.py`, `base_api.py`)

The old fine-tuning pattern created a second `L.Trainer` and called `.fit()` again. Under spawn, this meant a second `mp.spawn()` round requiring the first process group to be torn down and a new one created — fragile and error-prone.

**`EncoderFreezeCallback`** — New callback that implements fine-tuning within a single `trainer.fit()` call. At epoch `freeze_at_epoch`, it freezes the encoder and scales all optimizer learning rates by `lr_gamma`. Total training epochs = `epochs + epochs_fine_tune`.

Wired into `ptycho_torch.workflows.components._train_with_lightning` when `epochs_fine_tune > 0`. The shared training service no longer uses the two-trainer `ModelFineTuner_Lightning` pattern; the excluded legacy API still does.

### 5. DataModule Spawn Safety (`train_utils.py`)

**Worker guard** — `PtychoDataModule` and `PrebuiltPtychoDataModule` resolve worker settings at their loader boundary. Under spawn, they force `num_workers=0` and `persistent_workers=False` to prevent nested spawning (worker processes within spawned training processes), which causes hangs on many systems.

**Bug fix** — `PrebuiltPtychoDataModule` referenced `self.config.batch_size` but stored the config as `self.training_config`. Fixed the attribute name.

### 6. State Isolation (`base_api.py`)

**GPU cleanup** — `Trainer.train()` wrapped in `try/finally` with `torch.cuda.empty_cache()` and `gc.collect()`. Prevents GPU memory accumulation across consecutive training runs from a long-running host app.

---

## Verification

Tested on 2x NVIDIA GeForce RTX 4070 with the `pinn_velo_ic_2` dataset (64x64, ~10k samples after coordinate filtering).

| Test | Strategy | Devices | Result |
|------|----------|---------|--------|
| Baseline | `auto` | 1 GPU | Passed |
| Spawn single | `ddp_spawn` | 1 GPU | Passed |
| Spawn multi | `ddp_spawn` | 2 GPUs | Passed |

All tests: model pickled across spawn boundary, both ranks loaded memory maps, checkpoints saved, process groups managed by Lightning without manual intervention.

---

## Files Modified

| File | Summary |
|------|---------|
| `ptycho_torch/model.py` | Remove LambdaLayer class + instances, register probe mask as buffer, fix PoissonLoss leak |
| `ptycho_torch/beta_modules/model.py` | Remove LambdaLayer class + instances |
| `ptycho_torch/config_params.py` | Add `ModelConfig.__post_init__` for probe_mask CPU normalization |
| `ptycho_torch/train_utils.py` | Extend `get_training_strategy()`, add `is_spawn_strategy()`, add `EncoderFreezeCallback`, add spawn worker guards to DataModules |
| `ptycho_torch/api/trainer_api.py` | Wire `EncoderFreezeCallback` when fine-tuning is requested |
| `ptycho_torch/api/base_api.py` | Remove two-trainer fine-tuning, add GPU cleanup in `Trainer.train()` |
| `ptycho_torch/workflows/components.py` | Shared Trainer construction, serving-checkpoint restoration, fine-tune callback, and run-root publication |
