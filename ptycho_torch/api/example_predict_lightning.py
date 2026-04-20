import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath('/local/PtychoPINN'))
from ptycho_torch.api.base_api import ConfigManager, PtychoDataLoader, PtychoModel, Trainer, InferenceEngine, DataloaderFormats
from ptycho_torch.api.base_api import Orchestration
from ptycho_torch.model import PtychoPINN_Lightning
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Inference API — model-first initialization.
#
# The checkpoint is the authority for architecture-critical configs (N,
# C_model, n_filters_scale, attention flags, …).  ConfigManager is
# built *from* the loaded model rather than being constructed
# independently, so there is no risk of silently passing mismatched
# configs to the reconstructor.
#
# Flow:
#   1. Load model  (checkpoint → PtychoPINN_Lightning)
#   2. Extract arch configs  (ConfigManager.from_loaded_model)
#   3. Apply operational overrides  (inference_config, etc.)
#   4. Optional explicit arch-compatibility audit
#   5. Build dataloader + run inference

run_dir = "lightning_outputs/run_20260116_134601"

# ------------------------------------------------------------------
# 1. Load model — checkpoint is the authority
# ------------------------------------------------------------------
print("Loading model...")
trained_ptycho_model = PtychoModel._load(
    strategy='lightning',
    run_path=run_dir,
    model_class=PtychoPINN_Lightning,
)
print("Model loaded.")

# ------------------------------------------------------------------
# 2. Extract frozen arch configs from the loaded model.
#    _arch_frozen=True is set automatically: any subsequent
#    update() call that targets a frozen field (N, C_model, …)
#    will raise ValueError rather than silently corrupting state.
# ------------------------------------------------------------------
config_manager = ConfigManager.from_loaded_model(trained_ptycho_model)

# ------------------------------------------------------------------
# 3. Apply operational overrides — these fields are NOT frozen.
#    Attempting to set an arch field here (e.g. model_config={'N': 128})
#    would raise ValueError.
# ------------------------------------------------------------------
config_manager.update(inference_config={'middle_trim': 32, 'batch_size': 1000})

# ------------------------------------------------------------------
# 4. Optional: explicit field-by-field arch audit.
#    Useful as a sanity check after any manual config manipulation.
# ------------------------------------------------------------------
config_manager.validate_arch_compatibility(trained_ptycho_model)

# ------------------------------------------------------------------
# 5. Dataloader — built after config_manager is fully resolved
# ------------------------------------------------------------------
print("Creating tensordict dataloader...")
ptycho_data_dir = "/local/CDI-PINN/data/pinn_velo_fly001"
tensordict_dataloader = PtychoDataLoader(
    data_dir=ptycho_data_dir,
    config_manager=config_manager,
    data_format='tensordict',
)

# ------------------------------------------------------------------
# 6. Inference engine
# ------------------------------------------------------------------
print("Beginning inference...")
ptycho_inference = InferenceEngine(config_manager=config_manager,
                                   ptycho_model=trained_ptycho_model)

print(f"Beginning inference...")

result_im = ptycho_inference.predict_and_stitch(tensordict_dataloader)

#Extra display stuff just to check outputs.. ignore this if Steve since Ptychodus is processing the results


def plot_amp_and_phase(obj_amp, obj_phase, gt_amp, gt_phase, save_dir = None, filename = None):
    """
    Plot amplitude and phase comparison with ground truth.

    Creates a 2x2 grid showing reconstructed amplitude, reconstructed phase,
    ground truth amplitude, and ground truth phase.

    Args:
        obj_amp: Reconstructed amplitude array
        obj_phase: Reconstructed phase array
        gt_amp: Ground truth amplitude array
        gt_phase: Ground truth phase array
        save_dir: Optional directory to save plot
        filename: Optional filename for saved plot
    """
    fig, axs = plt.subplots(2,2, figsize=(5,5))

    #Object amp
    obj_plot = axs[0,0].imshow(obj_amp, cmap = 'gray')
    plt.colorbar(obj_plot, ax = axs[0,0])
    axs[0,0].set_title('Object Amplitude')
    axs[0,0].axis('off')

    #Object Phase
    phase_plot = axs[0,1].imshow(obj_phase, cmap = 'gray')#, vmin=-1, vmax=1)
    plt.colorbar(phase_plot, ax = axs[0,1])
    axs[0,1].set_title('Object Phase')
    axs[0,1].axis('off')

    #Ground turth amp
    gtamp_plot = axs[1,0].imshow(gt_amp, cmap = 'gray')
    plt.colorbar(gtamp_plot, ax = axs[1,0])
    axs[1,0].set_title('Ground Truth Amplitude')
    axs[1,0].axis('off')

    #ground truth phase
    gtphase_plot = axs[1,1].imshow(gt_phase, cmap = 'gray')
    plt.colorbar(gtphase_plot, ax = axs[1,1])
    axs[1,1].set_title('Ground Truth Phase')
    axs[1,1].axis('off')

    # Save the plot if save_dir is provided
    if save_dir is not None:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"amp_phase_comparison_{timestamp}.svg"

        # Ensure filename has an extension
        if not filename.endswith(('.png', '.jpg', '.pdf', '.svg')):
            filename += '.svg'

        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=900, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

        plt.show()


if len(result_im.shape) == 3:
        result_im = result_im[0].squeeze()

w = 20
result_amp = np.abs(result_im)
result_phase = np.angle(result_im)
gt_amp = np.abs(tensordict_dataloader.dataset.data_dict['objectGuess']).squeeze()
gt_phase = np.angle(tensordict_dataloader.dataset.data_dict['objectGuess']).squeeze()

plot_amp_and_phase(result_amp[w:-w,w:-w], result_phase[w:-w,w:-w],
                    gt_amp[w:-w,w:-w], gt_phase[w:-w,w:-w],
                    save_dir = 'test_output', filename = 'test_inference')
