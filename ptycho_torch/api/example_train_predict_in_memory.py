
import sys
import os
import numpy as np

# 1. Fixed output directory - no timestamp, no DDP synchronization issues
output_dir = "/local/Demo_Directory_PtychoPINN"
os.makedirs(output_dir, exist_ok=True)


sys.path.append(os.path.abspath('/local/PtychoPINN'))
from ptycho_torch.api.base_api import ConfigManager, PtychoDataLoader, PtychoModel, Trainer, InferenceEngine
from ptycho_torch.model import PtychoPINN_Lightning

# 2. Config Manager - load from JSON
json_dir = "/local/CDI-PINN/ptychopinn_torch/configs/publication_configs/4_q_multi_gpu_velociprobe.json"
config_manager, _ = ConfigManager._from_json(json_path=json_dir)

# Update config for small demo
training_update = {
    'epochs': 2,
    'epochs_fine_tune': 0,
    'n_devices': 1,
    'batch_size': 8,
    'num_workers': 0,
    'orchestrator': 'Lightning'
}
model_update = {
    'num_datasets': 1
}
config_manager.update(training_config=training_update, model_config=model_update)

# 3. Generate synthetic in-memory data
n_rows, n_cols = 14, 14
N = n_rows * n_cols  # 196 diffraction patterns
H = W = 64  # Pattern dimensions (must match data_config.N=64)

np.random.seed(42)
diff_patterns = np.random.rand(N, H, W).astype(np.float32) * 1000
probe = (np.random.rand(H, W) + 1j * np.random.rand(H, W)).astype(np.complex128)

# Create grid positions (spacing must be within data_config distance filters:
# min_neighbor_distance=0.5, max_neighbor_distance=8)
ypix = np.repeat(np.arange(n_rows) * 3.0, n_cols)
xpix = np.tile(np.arange(n_cols) * 3.0, n_rows)
positions = np.stack([ypix, xpix], axis=1)  # shape (N, 2), [ypix, xpix]

print(f"Synthetic data: diff_patterns={diff_patterns.shape}, probe={probe.shape}, positions={positions.shape}")

# 4. Create dataloader for TRAINING (with output_dir to enable data_module)
training_dataloader = PtychoDataLoader.from_np(
    diff_patterns=diff_patterns,
    probe=probe,
    positions=positions,
    config_manager=config_manager,
    output_dir=output_dir
)

# 5. Create new model
new_ptycho_model = PtychoModel._new_model(
    model=PtychoPINN_Lightning,
    config_manager=config_manager
)

# 6. Train with Lightning
lightning_trainer = Trainer._from_lightning(
    model=new_ptycho_model,
    dataloader=training_dataloader,
    orchestration='lightning',
    config_manager=config_manager
)

lightning_trainer.train(
    orchestration="lightning",
    experiment_name='in_memory_demo'
)

print(f"Output directory is: {output_dir}")
# Checkpoint saved at: {output_dir}/checkpoints/best-checkpoint.ckpt

# ========== INFERENCE ==========

# 8. Create inference dataloader from in-memory data (no output_dir needed)
inference_dataloader = PtychoDataLoader.from_np(
    diff_patterns=diff_patterns,
    probe=probe,
    positions=positions,
    config_manager=config_manager
)

# 9. Load trained model - reuse same config_manager (kept in memory)
trained_ptycho_model = PtychoModel._load(
    config_manager=config_manager,
    strategy='lightning',
    run_path=output_dir,
    model_class=PtychoPINN_Lightning
)

# 10. Run inference
ptycho_inference = InferenceEngine(
    config_manager=config_manager,
    ptycho_model=trained_ptycho_model
)

result = ptycho_inference.predict(inference_dataloader)
print(f"Inference result shape: {result.shape}")
print("In-memory train + predict demo complete!")
