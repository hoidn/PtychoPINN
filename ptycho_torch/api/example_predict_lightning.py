import sys
import os
sys.path.append(os.path.abspath('/local/PtychoPINN'))
from ptycho_torch.api.base_api import ConfigManager, PtychoDataLoader, PtychoModel, Trainer, InferenceEngine, DataloaderFormats
from ptycho_torch.api.base_api import Orchestration
from ptycho_torch.model import PtychoPINN_Lightning
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
#Following the notebook, we'll be loading a configuration from another mlflow run and doing a short training run to show the api off

#1. Config Manager

print("Loading configs...")

json_base_path = "lightning_outputs/run_20260116_134601"

config_manager = ConfigManager._from_lightning_json(json_base_path)


#2. Dataloader

print("Creating tensordict dataloader...")

ptycho_data_dir = "/local/CDI-PINN/data/pinn_velo_fly001"
tensordict_dataloader = PtychoDataLoader(data_dir = ptycho_data_dir,
                                        config_manager = config_manager,
                                        data_format = 'tensordict')

#3. Model
print("Loading model...")
trained_ptycho_model = PtychoModel._load(config_manager = config_manager,
                                         strategy = 'lightning',
                                         run_path = json_base_path,
                                         model_class = PtychoPINN_Lightning)

print("Model loaded.")

#4. Inference
print(f"Beginning inference...")
ptycho_inference = InferenceEngine(config_manager = config_manager,
                                   ptycho_model = trained_ptycho_model)

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






 



