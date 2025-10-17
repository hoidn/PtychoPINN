import importlib
import ptycho_torch.datagen
import ptycho_torch.reassembly
import ptycho_torch.helper
import ptycho_torch.dataloader
import ptycho_torch.patch_generator
import os
from collections import defaultdict

from ptycho_torch.reassembly import reconstruct_image, reassemble_multi_channel
from ptycho_torch.patch_generator import get_fixed_quadrant_neighbors_c4

from ptycho_torch.eval.frc import frc_preprocess_images, _match_phases_least_squares
from ptycho_torch.eval.eval_metrics import FSC
from scipy.ndimage import fourier_shift
from ptycho_torch.utils import load_all_configs_from_mlflow
from ptycho_torch.config_params import update_existing_config
from ptycho_torch.helper import center_crop
from ptycho_torch.config_params import ModelConfig, TrainingConfig, DataConfig, InferenceConfig
from ptycho_torch.dataloader import TensorDictDataLoader, PtychoDataset, Collate

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
from matplotlib import patheffects

import numpy as np
import mlflow.pytorch
from mlflow import MlflowClient


# Constant vars
relative_mlruns_path = '../../mlruns'
tracking_uri = f"file:{os.path.abspath(relative_mlruns_path)}"

def load_dataset(ptycho_dir, model_id, data_config_replace=None):
    """
    Loads dataset with configuration parameters linked to specific model ID.
    Args:
        ptycho_dir (str): Directory containing the ptychography data.
        model_id (str): Model ID to load the dataset configuration from.
        data_config_replace (dict, optional): Dictionary to replace specific data configuration parameters.
    """
    


    #Load configs
    data_config, model_config, training_config, inference_config, _ = load_all_configs_from_mlflow(model_id,
                                                                                            tracking_uri)


    # Replace data config if avail
    if data_config_replace is not None:
        update_existing_config(data_config, data_config_replace)

    print('Creating dataset...')
    ptycho_dataset = PtychoDataset(ptycho_dir, model_config, data_config,
                                remake_map=True)

    return ptycho_dataset, data_config, model_config, training_config, inference_config

def load_model_and_reconstruct(model_id, ptycho_dataset,
                              data_config, model_config, training_config,
                              inference_config, inference_config_replace = None):
    """
    Loads the model and reconstructs the image from the dataset.
    Args:
        model_id (str): Model ID to load the model from.
        ptycho_dataset (PtychoDataset): Dataset containing the ptychography data.
        data_config (DataConfig): Data configuration parameters.
        model_config (ModelConfig): Model configuration parameters.
        training_config (TrainingConfig): Training configuration parameters.
        inference_config (InferenceConfig): Inference configuration parameters.
    """
    
    # Loading model
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"runs:/{model_id}/model"
    loaded_model = mlflow.pytorch.load_model(model_uri)
    loaded_model.to('cuda')
    loaded_model.training = True

    # Reconstruct image
    print('Reconstructing image...')
    result, recon_dataset = reconstruct_image(loaded_model, ptycho_dataset,
                           training_config, data_config, model_config, inference_config)

    return result, recon_dataset
def get_target_and_prediction_images_and_window(result, recon_dataset, window_size=0):
    """
    Meant to be called on result of reconstruct_image which still resides on GPU.
    Gets target amplitude and phase images, as well as the prediction amplitude and phase images.
    Crops to specified window size if provided.
    Args:
        result (torch.Tensor): Result tensor from the reconstruction. Complex-valued
        recon_dataset (PtychoDataset): Dataset containing the ptychography data.
        window_size (int): Size of the window to crop the images to. If 0, no cropping is done. Will almost surely be greater than 0.
    Returns:
        np.ndarray: Numpy array of the result.
    """
    #Move result to CPU and remove channel dimension if needed
    im_cpu = result.to('cpu')
    if len(im_cpu.shape) == 3:
        im_cpu = im_cpu.squeeze()
    
    # Set to shorter variable for readability
    w = window_size

    # Object recon
    recon = im_cpu.detach().cpu().numpy().copy()[w:-w,w:-w]

    # Ground truth
    gt = recon_dataset.data_dict['objectGuess'][0].copy()[w:-w,w:-w]
    gt = center_crop(gt, recon.shape[0])

    return gt, recon
    

def preprocess_and_calculate_frc(gt, recon):
    """
    Preprocesses the ground truth and reconstructed images, then calculates the Fourier Ring Correlation (FRC).
    Args:
        gt (np.ndarray): Ground truth image.
        recon (np.ndarray): Reconstructed image.
    Returns:
        tuple: FRC values and frequencies.
    """
    # Preprocess images
    aligned_gt, aligned_pred = frc_preprocess_images(gt, recon, image_prop='complex',verbose=True, align = False)

    # Calculate FRC
    FR_curve, x_FR, T_curve, x_T = FSC(aligned_gt, aligned_pred)

    # Calculate sum under curve until 1
    FR_AUC = np.sum(FR_curve[:np.where(x_FR - 1 > 0)[0][0]])/np.where(x_FR - 1 > 0)[0][0]

    return FR_curve, x_FR, FR_AUC
def save_frc_to_dict(FR_curve, x_FR, FR_AUC, frc_dict,
                     experiment_id, model_name, window_size):
    """
    Saves FRC values to dictionary with appropriate keys.
    """
    # In case frc_dict is not defaultdict
    if experiment_id not in frc_dict:
        frc_dict[experiment_id] = {}
    if model_name not in frc_dict[experiment_id]:
        frc_dict[experiment_id][model_name] = {}

    frc_dict[experiment_id][model_name]['x_FR'] = x_FR
    frc_dict[experiment_id][model_name]['FR_curve'] = FR_curve
    frc_dict[experiment_id][model_name]['FR_AUC'] = FR_AUC
    frc_dict[experiment_id][model_name]['window_size'] = window_size

    return frc_dict

def add_to_npz(filename, **new_arrays):
    """
    Adds new arrays to an existing .npz file or creates a new one if it doesn't exist.
    Args:
        filename (str): The name of the .npz file to which arrays will be added.
        **new_arrays: Arbitrary keyword arguments representing the new arrays to add.
    """
    try:
        # Load existing arrays
        existing = dict(np.load(filename))
        existing.update(new_arrays)
        np.savez(filename, **existing)
    except FileNotFoundError:
        # File doesn't exist yet, create it
        print("File not found")
        np.savez(filename, **new_arrays)
# For a given key in frc_dict and experiment name, assign x_FR, FR_curve and FR_AUC to numpy arrays. Concatenate them along the first axis.

def get_frc_arrays(frc_dict, exp_name):
    """
    Gets the FRC arrays for a given experiment name and model name from the frc_dict.
    Args:
        frc_dict (dict): Dictionary containing FRC results.
        exp_name (str): Experiment name.
        model_name (str): Model name.
    Returns:
        tuple: x_FR, FR_curve, FR_AUC as numpy arrays.
    """

    x_FR = []
    FR_curve = []
    FR_AUC = []
    model_list = list(frc_dict[exp_name].keys())  # Get the first model name for the experiment

    for model_name in frc_dict[exp_name].keys():
        x_FR.append(frc_dict[exp_name][model_name]['x_FR'])
        FR_curve.append(frc_dict[exp_name][model_name]['FR_curve'])
        FR_AUC.append(frc_dict[exp_name][model_name]['FR_AUC'])

    return x_FR, FR_curve, FR_AUC, model_list
def calculate_threshold_frc(x_FR):
    """
    Calculates the threshold FRC curve based on the x_FR array.
    Args:
        x_FR (list): List of x_FR arrays for different models.
    Returns:
        x_T, T_curve (tuple): Threshold FRC values and corresponding frequencies.
    """
    # Calculate threshold FRC curve (taken from eval_metrics.py)
    SNRt = 0.2071 #half bit threshold
    eps = np.finfo(float).eps
    r = np.arange(1+x_FR.shape[0]/2)
    n = 2*np.pi*r
    n[0] = 1
    t1 = np.divide(np.ones(np.shape(n)),n+eps)
    t2 = SNRt + 2*np.sqrt(SNRt)*t1 + np.divide(np.ones(np.shape(n)),np.sqrt(n))
    t3 = SNRt + 2*np.sqrt(SNRt)*t1 + 1
    T_curve = np.divide(t2,t3)
    x_T = r/(x_FR.shape[0]/2)

    return x_T, T_curve
def preprocess_and_calculate_frc(gt, recon):
    """
    Preprocesses the ground truth and reconstructed images, then calculates the Fourier Ring Correlation (FRC).
    Args:
        gt (np.ndarray): Ground truth image.
        recon (np.ndarray): Reconstructed image.
    Returns:
        tuple: FRC values and frequencies.
    """
    # Preprocess images
    aligned_gt, aligned_pred = frc_preprocess_images(gt, recon, image_prop='complex',verbose=True, align = False)

    # Calculate FRC
    FR_curve, x_FR, T_curve, x_T = FSC(aligned_gt, aligned_pred)

    # Calculate sum under curve until 1
    FR_AUC = np.sum(FR_curve[:np.where(x_FR - 0.5 > 0)[0][0]])/np.where(x_FR - 0.5 > 0)[0][0]

    return FR_curve, x_FR, FR_AUC

def calculate_all_frcs(file_list, model_dict, window_size_dict,
                       save_path, save_file_name):
    
    """
    Quick and dirty function to calculate all frc values for every experiment in file_list using every single model in model_dict.
    """

    #Load
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.exists(save_path + '/' + save_file_name):
        print("Loading existing frc_results.npz")
        frc_dict = np.load(save_path + '/' + save_file_name, allow_pickle = True)['frc_dict'].item()
    else:
        frc_dict = defaultdict(dict)
    
    for file in file_list:
    # Get experiment name
        exp_name = file.split('/')[-1]

        # Get model ID
        for model_name, model_id in model_dict.items():
            # Check if this iteration was already done
            if exp_name in frc_dict and model_name in frc_dict[exp_name]:
                print(f'Skipping {exp_name} with model {model_name}, already processed.')
                continue

            print(f'Processing {exp_name} with model {model_name}')
            # Check to see if need to replace data config
            data_config_replace = {}

            data_config_replace['normalize'] = 'Batch'
            data_config_replace['probe_normalize'] = 'True'
            data_config_replace['scan_pattern'] = 'Rectangular'
            data_config_replace['n_subsample'] = 1
            data_config_replace["x_bounds"] = [0.05, 0.95]
            data_config_replace["y_bounds"] = [0.05, 0.95]

            if exp_name == 'pinn_velo_gold_tp_1':
                data_config_replace["x_bounds"] = [0.07, 0.93]
                data_config_replace["y_bounds"] = [0.07, 0.93]
            if exp_name == 'pinn_velo_gold_tp_2' or exp_name == 'pinn_velo_ic_2' or exp_name == 'pinn_velo_ncm':
                data_config_replace["min_neighbor_distance"] = 0.5
                data_config_replace["max_neighbor_distance"] = 8

            # Load dataset
            ptycho_dataset, data_config, model_config, training_config, inference_config = load_dataset(file, model_id,
                                                                                                    data_config_replace=data_config_replace)
            
            #Replacing inference config
            inference_config_replace = {'middle_trim': data_config.N//2,
                                'batch_size': 512,
                                'experiment_number': 0,
                                'pad_eval': True}

            update_existing_config(inference_config, inference_config_replace)
            
            print("Beginning reconstruction...")
            # Load model and reconstruct image
            result, recon_dataset = load_model_and_reconstruct(model_id, ptycho_dataset,
                                                            data_config, model_config,
                                                            training_config, inference_config)
            
            print("Recon complete. Beginning FRC calculation...")
            # Get target and prediction images
            gt, recon = get_target_and_prediction_images_and_window(result, recon_dataset,
                                                                    window_size=window_size_dict[exp_name])

            # Preprocess and calculate FRC
            FR_curve, x_FR, FR_AUC = preprocess_and_calculate_frc(gt, recon)
            
            # Create entries if they don't exist
            if exp_name not in frc_dict:
                frc_dict[exp_name] = defaultdict(dict)
                if model_name not in frc_dict[exp_name]:
                    frc_dict[exp_name][model_name] = defaultdict(dict)
            
            # Sanity check if frc_dict[exp_name][model_name] exists
            if model_name in frc_dict[exp_name]:
                print(f"frc_dict[{exp_name}][{model_name}] exists")
            
            frc_dict = save_frc_to_dict(FR_curve, x_FR, FR_AUC,frc_dict,
                                        exp_name, model_name,
                                        window_size=window_size_dict[exp_name])

            # Save intermediate results to .npz file
            print("Saving results to .npz file...")
            npz_filename = os.path.join(save_path, save_file_name)
            np.savez(npz_filename, frc_dict=frc_dict)

## Quick and dirty reconstruction plotting

def generate_gt_and_recon(file, model_name, model_dict,
                          window_size_dict):
    
    model_id = model_dict[model_name]

    exp_name = file.split('/')[-1]

    data_config_replace = {}

    data_config_replace['normalize'] = 'Batch'
    data_config_replace['probe_normalize'] = 'False'
    data_config_replace['scan_pattern'] = 'Rectangular'
    data_config_replace['n_subsample'] = 1
    data_config_replace["x_bounds"] = [0.05, 0.95]
    data_config_replace["y_bounds"] = [0.05, 0.95] 

    # Check to see if need to replace data config
    if exp_name in ['pinn_velo_gold_tp_1', 'pinn_velo_ic_1']:
        data_config_replace["x_bounds"] = [0.07, 0.93]
        data_config_replace["y_bounds"] = [0.07, 0.93]
    
    ptycho_dataset, data_config, model_config, training_config, inference_config = load_dataset(file, model_id,
                                                                                                  data_config_replace=data_config_replace)
        
    #Replacing inference config
    inference_config_replace = {'middle_trim': data_config.N//2,
                        'batch_size': 512,
                        'experiment_number': 0,
                        'pad_eval': True}

    update_existing_config(inference_config, inference_config_replace)
        
        
    print("Beginning reconstruction...")
    # Load model and reconstruct image
    result, recon_dataset = load_model_and_reconstruct(model_id, ptycho_dataset,
                                                    data_config, model_config,
                                                    training_config, inference_config)
    
    print("Recon complete. Beginning FRC calculation...")
    # Get target and prediction images
    gt, recon = get_target_and_prediction_images_and_window(result, recon_dataset,
                                                                window_size= window_size_dict[exp_name])
    
    return gt, recon

# Plotting
# -----------

def plot_object(data, colorbar_size=(0.06, 0.3), 
                      colorbar_position=(0.03, 0.05), figsize=(4,2), 
                      border_thickness=0.02, save_path=None, save_name='probe_default', 
                      format='svg', dpi=600):
    """
    Plot complex-valued object data as horizontally stacked amplitude and phase images with embedded colorbars.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Complex-valued 2D array
    colorbar_size : tuple
        (width, height) of each colorbar as fraction of each sub-image
    colorbar_position : tuple
        (x, y) position of colorbar bottom-left corner as fraction of each sub-image
    figsize : tuple
        Figure size in inches (width, height)
    border_thickness : float
        Thickness of white border between images as fraction of total height
    save_path : str or None
        Directory path to save the image. If None, displays only
    save_name : str
        Base filename for saved image
    format : str
        Export format ('svg', 'pdf', 'png', 'eps')
    dpi : int
        DPI for raster formats
    """
    
# Extract amplitude and phase data
    amplitude_data = np.abs(data)
    phase_data = np.angle(data)
    
    # Create figure with subplots
    fig, (ax_amp, ax_phase) = plt.subplots(1,2, figsize=figsize, facecolor='none')
    fig.patch.set_alpha(0)
    plt.subplots_adjust(wspace=border_thickness)
    
    # Process amplitude image (top)
    amp_vmin, amp_vmax = np.min(amplitude_data), np.max(amplitude_data)
    amp_min_label = f'{amp_vmin:.2f}' if amp_vmin <= 0 else f' {amp_vmin:.2f}'
    amp_max_label = f'{amp_vmax:.2f}' if amp_vmax <= 0 else f' {amp_vmax:.2f}'
    amp_tick_labels = [amp_min_label, amp_max_label]
    
    # Display amplitude image
    im_amp = ax_amp.imshow(amplitude_data, cmap= 'viridis', vmin=amp_vmin, vmax=amp_vmax, 
                          aspect='equal', interpolation='nearest')
    
    # Remove borders and ticks from amplitude axis
    ax_amp.set_xticks([])
    ax_amp.set_yticks([])
    ax_amp.spines['top'].set_visible(False)
    ax_amp.spines['right'].set_visible(False)
    ax_amp.spines['bottom'].set_visible(False)
    ax_amp.spines['left'].set_visible(False)
    
    # Create amplitude colorbar
    amp_pos = ax_amp.get_position()
    cb_left_amp = amp_pos.x0 + colorbar_position[0] * amp_pos.width
    cb_bottom_amp = amp_pos.y0 + colorbar_position[1] * amp_pos.height
    cb_width_amp = colorbar_size[0] * amp_pos.width
    cb_height_amp = colorbar_size[1] * amp_pos.height
    
    ax_cb_amp = fig.add_axes([cb_left_amp, cb_bottom_amp, cb_width_amp, cb_height_amp])
    cb_amp = plt.colorbar(im_amp, cax=ax_cb_amp, orientation='vertical')
    ax_cb_amp.set_aspect('auto')
    
    cb_amp.set_ticks([amp_vmin, amp_vmax])
    cb_amp.set_ticklabels(amp_tick_labels)
    cb_amp.ax.tick_params(color='white', size=1)
    
    cb_amp.ax.tick_params(
        labelsize=8,
        labelcolor='black',
        pad=1,
        left=False, 
        right=True, 
        labelleft=False,
        labelright=True
    )
    
    for label in cb_amp.ax.get_yticklabels():
        label.set_path_effects([
            patheffects.Stroke(linewidth=2, foreground='white'),
            patheffects.Normal()
        ])
    
    cb_amp.outline.set_visible(True)
    cb_amp.outline.set_linewidth(1.5)
    cb_amp.outline.set_edgecolor('white')
    
    # Process phase image (bottom)
    phase_vmin, phase_vmax = np.min(phase_data), np.max(phase_data)
    phase_min_label = f'{phase_vmin:.2f}' if phase_vmin <= 0 else f' {phase_vmin:.2f}'
    phase_max_label = f'{phase_vmax:.2f}' if phase_vmax <= 0 else f' {phase_vmax:.2f}'
    phase_tick_labels = [phase_min_label, phase_max_label]
    
    # Display phase image
    im_phase = ax_phase.imshow(phase_data, cmap= 'gray', vmin=phase_vmin, vmax=phase_vmax, 
                              aspect='equal', interpolation='nearest')
    
    # Remove borders and ticks from phase axis
    ax_phase.set_xticks([])
    ax_phase.set_yticks([])
    ax_phase.spines['top'].set_visible(False)
    ax_phase.spines['right'].set_visible(False)
    ax_phase.spines['bottom'].set_visible(False)
    ax_phase.spines['left'].set_visible(False)
    
    # Create phase colorbar
    phase_pos = ax_phase.get_position()
    cb_left_phase = phase_pos.x0 + colorbar_position[0] * phase_pos.width
    cb_bottom_phase = phase_pos.y0 + colorbar_position[1] * phase_pos.height
    cb_width_phase = colorbar_size[0] * phase_pos.width
    cb_height_phase = colorbar_size[1] * phase_pos.height
    
    ax_cb_phase = fig.add_axes([cb_left_phase, cb_bottom_phase, cb_width_phase, cb_height_phase])
    cb_phase = plt.colorbar(im_phase, cax=ax_cb_phase, orientation='vertical')
    ax_cb_phase.set_aspect('auto')
    
    cb_phase.set_ticks([phase_vmin, phase_vmax])
    cb_phase.set_ticklabels(phase_tick_labels)
    cb_phase.ax.tick_params(color='white', size=1)
    
    cb_phase.ax.tick_params(
        labelsize=8,
        labelcolor='black',
        pad=1,
        left=False, 
        right=True, 
        labelleft=False,
        labelright=True
    )
    
    for label in cb_phase.ax.get_yticklabels():
        label.set_path_effects([
            patheffects.Stroke(linewidth=2, foreground='white'),
            patheffects.Normal()
        ])
    
    cb_phase.outline.set_visible(True)
    cb_phase.outline.set_linewidth(1.5)
    cb_phase.outline.set_edgecolor('white')
    
    if save_path:
        full_save_path = os.path.join(save_path, f"{save_name}.{format}")
        plt.savefig(full_save_path, format=format, dpi=dpi, 
                   bbox_inches='tight', pad_inches=0, 
                   transparent=True, facecolor='none')
        print(f"Image saved as {full_save_path}")
    
    plt.show()
    return fig, ax_amp, ax_phase, cb_amp, cb_phase