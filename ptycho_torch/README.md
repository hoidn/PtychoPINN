# PyTorch Model Loading & Inference Guide

This guide explains how to load, instantiate, and run PyTorch models for inference in
PtychoPINN. It is optimized for two common scenarios:

- Recommended: CLI inference (handles CONFIG-001 and wiring for you)
- Manual: state_dict-only inference (when you only have `model.pt`)

## Recommended: CLI Inference Path

Use the CLI for the most reliable workflow. It performs config bridging, calls
`update_legacy_dict(params.cfg, config)`, wires the generator registry, and runs
batched inference with consistent output handling.

Minimal example:

```bash
python -m ptycho_torch.inference \
  --model_path outputs/grid_lines_n128_compare_padex_lr2e4_plateau_e40_seed3/runs/pinn_hybrid_resnet \
  --test_data outputs/grid_lines_n128_compare_padex_lr2e4_plateau_e40_seed3/datasets/N128/gs1/test.npz \
  --output_dir outputs/inference_hybrid_resnet
```

Use this when you have a full training directory (checkpoints, configs, metadata)
and want a safe, supported inference path.

## Manual Reconstruction (state_dict-only)

If you only have a `model.pt` state_dict, you must reconstruct the Lightning module
with the correct architecture before loading weights.

Key steps:

1. Build configs with `create_training_payload(...)` (factory).  
   - **Required**: `n_groups` (no default). For grid-lines NPZs, using the number of
     samples in the test set is a safe choice.
2. Resolve the generator via `resolve_generator(...)`.
3. Build the Lightning module with `generator.build_model(...)`.
4. Load weights with `model.load_state_dict(...)`.
5. Run inference using `forward_predict(...)` or `run_torch_inference(...)`.

Minimal example (grid-lines NPZ + throughput-friendly inference):

```python
from pathlib import Path
import numpy as np
import torch

from ptycho_torch.config_factory import create_training_payload
from ptycho_torch.config_params import InferenceConfig as PTInferenceConfig
from ptycho_torch.generators.registry import resolve_generator
from ptycho.config.config import PyTorchExecutionConfig
from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig, run_torch_inference

output_dir = Path("outputs/grid_lines_n128_compare_padex_lr2e4_plateau_e40_seed3")
train_npz = output_dir / "datasets/N128/gs1/train.npz"
test_npz = output_dir / "datasets/N128/gs1/test.npz"
model_path = output_dir / "runs/pinn_hybrid_resnet/model.pt"

with np.load(test_npz, allow_pickle=True) as data:
    test_data = {
        "diffraction": data["diffraction"],
        "coords_nominal": data["coords_nominal"],
        "probeGuess": data["probeGuess"],
    }

overrides = {
    "n_groups": test_data["diffraction"].shape[0],
    "gridsize": 1,
    "architecture": "hybrid_resnet",
    "model_type": "Unsupervised",
    "nphotons": 1e9,
    "fno_modes": 12,
    "fno_width": 32,
    "fno_blocks": 4,
    "fno_cnn_blocks": 2,
    "fno_input_transform": "none",
    "generator_output_mode": "real_imag",
}

payload = create_training_payload(
    train_data_file=train_npz,
    output_dir=output_dir,
    overrides=overrides,
    execution_config=PyTorchExecutionConfig(),
)

generator = resolve_generator(payload.tf_training_config)
pt_configs = {
    "model_config": payload.pt_model_config,
    "data_config": payload.pt_data_config,
    "training_config": payload.pt_training_config,
    "inference_config": PTInferenceConfig(),
}
model = generator.build_model(pt_configs)
model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
model.eval()

cfg = TorchRunnerConfig(
    train_npz=train_npz,
    test_npz=test_npz,
    output_dir=output_dir,
    architecture="hybrid_resnet",
    N=128,
    gridsize=1,
    infer_batch_size=16,
    fno_modes=12,
    fno_width=32,
    fno_blocks=4,
    fno_cnn_blocks=2,
    fno_input_transform="none",
    generator_output_mode="real_imag",
    torch_loss_mode="mae",
)

predictions = run_torch_inference(model, test_data, cfg)
```

### Required NPZ Keys (grid-lines)

For `run_torch_inference(...)`, the test NPZ should contain:

- `diffraction` (N, H, W, C) or (N, H, W) -> will be normalized to NCHW
- `coords_nominal` (N, C, 1, 2) or compatible shape
- `probeGuess` (H, W) complex64

## Pitfalls & Verification

- **CONFIG-001**: Use the factory or CLI. Direct instantiation can silently mis-sync
  gridsize and channel count.
- **n_groups is required**: The factory rejects missing `n_groups`. Use test sample
  count as a safe default.
- **Output mode matters**: `generator_output_mode="amp_phase"` applies sigmoid/tanh
  inside the generator. Downstream consumers expect physical values.
- **Shape mismatches**: If `load_state_dict(strict=True)` fails, your architecture
  knobs (modes/width/blocks) likely do not match the saved model.

Quick verification checklist:

- Can you `load_state_dict(..., strict=True)` without missing keys?
- Does `forward_predict(...)` run on a single batch?
- Do predictions have the expected shape and dtype?

## Related Docs

- `docs/PYTORCH_CLI_ANALYSIS_README.md` (deep CLI mapping and config translations)
- `docs/workflows/pytorch.md` (end-to-end PyTorch workflow)
