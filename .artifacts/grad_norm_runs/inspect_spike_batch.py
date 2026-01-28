import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch

from scripts.studies.grid_lines_torch_runner import (
    TorchRunnerConfig,
    load_cached_dataset,
    setup_torch_configs,
)
from ptycho_torch.model import PtychoPINN_Lightning
from ptycho_torch.workflows.components import _train_with_lightning

# Identify spike step from previous run metrics
metrics_path = Path('training_outputs/lightning_logs/version_149/metrics.csv')
if not metrics_path.exists():
    raise SystemExit(f"metrics.csv not found: {metrics_path}")

df = pd.read_csv(metrics_path)
step_df = df.dropna(subset=['grad_norm_preclip_step'])
if step_df.empty:
    raise SystemExit("No grad_norm_preclip_step rows found in metrics.csv")

idx = step_df['grad_norm_preclip_step'].idxmax()
row = step_df.loc[idx]
spike_step = int(row['step'])
spike_epoch = int(row['epoch'])

# Patch forward to capture outputs
_orig_forward = PtychoPINN_Lightning.forward

def _patched_forward(self, x, positions, probe, input_scale_factor, output_scale_factor, experiment_ids):
    out = _orig_forward(self, x, positions, probe, input_scale_factor, output_scale_factor, experiment_ids)
    # Store last forward outputs and inputs for logging in training_step
    self._last_forward_out = out
    self._last_forward_in = (x, positions, probe, input_scale_factor, output_scale_factor, experiment_ids)
    return out

PtychoPINN_Lightning.forward = _patched_forward

# Patch training_step to log batch stats at target step
_orig_training_step = PtychoPINN_Lightning.training_step
_logged = {'done': False}


def _tensor_stats(t: torch.Tensor):
    t = t.detach()
    if not (t.is_floating_point() or t.is_complex()):
        t = t.float()
    return {
        'shape': list(t.shape),
        'dtype': str(t.dtype),
        'device': str(t.device),
        'mean': float(t.mean().item()),
        'std': float(t.std().item()),
        'min': float(t.min().item()),
        'max': float(t.max().item()),
    }


def _patched_training_step(self, batch, batch_idx):
    loss = _orig_training_step(self, batch, batch_idx)
    if (not _logged['done']) and int(self.global_step) == spike_step:
        # Inputs
        x = batch[0]['images']
        positions = batch[0]['coords_relative']
        rms_scale = batch[0]['rms_scaling_constant']
        physics_scale = batch[0]['physics_scaling_constant']
        experiment_ids = batch[0]['experiment_id']
        probe = batch[1]

        # Forward outputs captured by patched forward
        pred, amp, phase = getattr(self, '_last_forward_out')

        intensity_norm_factor = float(x.mean().detach().item() + 1e-8)

        stats = {
            'spike_step': spike_step,
            'spike_epoch': spike_epoch,
            'batch_idx': int(batch_idx),
            'global_step': int(self.global_step),
            'intensity_norm_factor': intensity_norm_factor,
            'inputs': {
                'x': _tensor_stats(x),
                'positions': _tensor_stats(positions),
                'rms_scale': _tensor_stats(rms_scale),
                'physics_scale': _tensor_stats(physics_scale),
                'experiment_ids': _tensor_stats(experiment_ids),
                'probe_real': _tensor_stats(torch.view_as_real(probe).float()),
            },
            'outputs': {
                'pred': _tensor_stats(pred),
                'amp': _tensor_stats(amp),
                'phase': _tensor_stats(phase),
            }
        }

        out_path = Path('.artifacts/grad_norm_runs/spike_batch_stats.json')
        out_path.write_text(json.dumps(stats, indent=2))
        _logged['done'] = True

    return loss

PtychoPINN_Lightning.training_step = _patched_training_step

# Run training with same settings/data but disable checkpointing to avoid disk overflow
cfg = TorchRunnerConfig(
    train_npz=Path('outputs/grid_lines_gs1_n64_e20_phi_clip50/datasets/N64/gs1/train.npz'),
    test_npz=Path('outputs/grid_lines_gs1_n64_e20_phi_clip50/datasets/N64/gs1/test.npz'),
    output_dir=Path('outputs/grid_lines_gs1_n64_e20_phi_clip50_spikeinspect'),
    architecture='hybrid',
    seed=42,
    epochs=20,
    batch_size=16,
    learning_rate=1e-3,
    infer_batch_size=16,
    gradient_clip_val=50.0,
    torch_loss_mode='mae',
    log_grad_norm=True,
    grad_norm_log_freq=1,
    N=64,
    gridsize=1,
)

train_data = load_cached_dataset(cfg.train_npz)
test_data = load_cached_dataset(cfg.test_npz)

# Build configs and disable checkpointing
training_config, execution_config = setup_torch_configs(cfg)
execution_config.enable_checkpointing = False

# Build train/test containers (same as grid_lines_torch_runner)
X = np.asarray(train_data['diffraction'])
if X.ndim == 3:
    X = X[..., np.newaxis]
coords = train_data.get('coords_nominal')
if coords is None:
    coords = np.zeros((X.shape[0], 1, 2, X.shape[-1]), dtype=np.float32)
probe = train_data.get('probeGuess')
if probe is None:
    probe = np.ones((cfg.N, cfg.N), dtype=np.complex64)
train_container = {
    'X': X,
    'coords_nominal': coords,
    'probe': probe,
}

test_container = None
if test_data:
    X_te = np.asarray(test_data['diffraction'])
    if X_te.ndim == 3:
        X_te = X_te[..., np.newaxis]
    coords_te = test_data.get('coords_nominal')
    if coords_te is None:
        coords_te = np.zeros((X_te.shape[0], 1, 2, X_te.shape[-1]), dtype=np.float32)
    test_probe = test_data.get('probeGuess', probe)
    test_container = {
        'X': X_te,
        'coords_nominal': coords_te,
        'probe': test_probe,
    }

_train_with_lightning(
    train_container,
    test_container,
    training_config,
    execution_config=execution_config,
)

print('wrote .artifacts/grad_norm_runs/spike_batch_stats.json')
