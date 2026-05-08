#Most basic modules
import sys
import argparse
import os
import json
import random
import math
from pathlib import Path
from glob import glob

#Typing
from dataclasses import asdict
from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig

#ML libraries
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Subset
import torch.distributed as dist

#Automation modules
#Lightning
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

#MLFlow
import mlflow.pytorch
from mlflow import MlflowClient

#Configs/Params
from ptycho_torch.config_params import update_existing_config

#Custom modules
from ptycho_torch.model import PtychoPINN_Lightning
from ptycho_torch.utils import config_to_json_serializable_dict, load_config_from_json, validate_and_process_config, remove_all_files
from ptycho_torch.train_utils import set_seed, get_training_strategy, find_learning_rate, log_parameters_mlflow, is_effectively_global_rank_zero, print_auto_logged_info
from ptycho_torch.train_utils import ModelFineTuner, PtychoDataModule
from ptycho_torch.datagen.datagen import simulate_multiple_experiments, generate_simulated_data
from ptycho_torch.datagen.datagen import assemble_precomputed_images, simulate_synthetic_objects, simulate_synthetic_probes
from ptycho_torch.datagen.objects import create_density_histogram_reim, compute_reim_statistics
from ptycho_torch.datagen.objects import fit_gmm_from_objects
from ptycho_torch.train import main
from ptycho_torch.beta_modules.train_ccnf import main as train_ccnf

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("PtychoPINN")

def load_all_configs(ptycho_dir, config_path):
    """
    Helper functions that loads all relevant configs specifically for this train_full process
    """
    print('Loading configs...')
    try:
        config_data = load_config_from_json(config_path)
        d_config_replace, m_config_replace, t_config_replace, i_config_replace, dgen_config_replace = validate_and_process_config(config_data)
    except Exception as e:
        print(f"Failed to open/validate config because of: {e}")
    
    data_config = DataConfig()
    if d_config_replace is not None:
        update_existing_config(data_config, d_config_replace)
    
    model_config = ModelConfig()
    if m_config_replace is not None:
        update_existing_config(model_config, m_config_replace)

    print(model_config)

    training_config = TrainingConfig()
    if t_config_replace is not None:
        update_existing_config(training_config, t_config_replace)    

    inference_config = InferenceConfig()

    datagen_config = DatagenConfig()
    if dgen_config_replace is not None:
        # Get list of all used data to save as artifact (for reproducibility) via DatagenConfig
        exp_list = glob(ptycho_dir + '/*.npz')
        dgen_config_replace['probe_paths'] = exp_list
        update_existing_config(datagen_config, dgen_config_replace)

    return [data_config, model_config, training_config, inference_config, datagen_config],\
           dgen_config_replace

def prepare_data(ptycho_dir, mode, config_path,
                 synthetic_path = "synthetic_training",
                 histogram_source_dir = None):
    """
    Processes data from list of files in config.
    Only rank 0 does data preparation, other ranks just load configs.
    """
    
    # Always load configs first (all ranks need this)
    configs, dgen_config_replace = load_all_configs(ptycho_dir, config_path)
    data_config, model_config, training_config, inference_config, datagen_config = configs

    # Check for lightning rank 0
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if mode != 'exp' and mode != 'synth':
        raise ValueError('Expected "exp" or "synth"')
    elif mode == 'exp':
        print("Experimental data specified, proceeding...")
        dgen_config_replace['object_class'] = 'exp'
        training_path = ptycho_dir  # No change needed for experimental
    elif mode == 'synth':
        # Set the synthetic training path
        parent_path = Path(ptycho_dir).parent
        synthetic_path = str(parent_path / synthetic_path)
        training_path = synthetic_path
        
        # Check if we're in distributed training and get rank
        if local_rank == 0:
            # Only rank 0 does the actual data generation
            print("Rank 0: Preparing synthetic data...")
            
            # Legacy object argument, unused for now but allows for easy prototyping
            obj_arg = {}
            probe_arg = {}

            # Assemble probe list (always from ptycho_dir)
            exp_probe_list = assemble_precomputed_images(dir_list = ptycho_dir, mode = 'probe',
                                                         single_dir = True, probe_ramp_removal = False)

            # Resolve histogram source: CLI arg > datagen_config field > ptycho_dir
            hist_dir = histogram_source_dir or datagen_config.histogram_source_dir or ptycho_dir
            exp_obj_list = assemble_precomputed_images(dir_list = hist_dir, mode = 'object',
                                                         single_dir = True, probe_ramp_removal = False)
            if hist_dir != ptycho_dir:
                print(f"Using histogram reference objects from: {hist_dir}")
            datagen_config.histogram_source_dir = hist_dir
            probe_list = [item for item in exp_probe_list for _ in range(datagen_config.objects_per_probe)]
            probe_name_idx = [idx for idx in list(range(len(exp_probe_list))) for _ in range(datagen_config.objects_per_probe)]
            probe_arg['probe_name_idx'] = probe_name_idx

            # Try creating synthetic object
            try:
                print(f"Creating objects for class: {datagen_config.object_class}")

                #Sampling from objects
                material_histogram, _, _ = create_density_histogram_reim(exp_obj_list, origin_threshold = datagen_config.histogram_threshold)
                material_stats = compute_reim_statistics(exp_obj_list)
                obj_arg['gmm_params'] = fit_gmm_from_objects(
                    exp_obj_list,
                    n_clusters=datagen_config.gmm_n_clusters,
                    origin_mask_radius=datagen_config.origin_mask_radius,
                )

                canvas_scale = data_config.N / 64
                image_size = (int(datagen_config.image_size[0] * canvas_scale),
                              int(datagen_config.image_size[1] * canvas_scale))
                obj_arg['mode'] = datagen_config.reim_mode
                obj_arg['perturbation_mode'] = datagen_config.perturbation_mode
                obj_arg['phase_jitter_std'] = datagen_config.phase_jitter_std
                obj_arg['amplitude_scale_std'] = datagen_config.amplitude_scale_std
                obj_arg['center_jitter_std'] = datagen_config.center_jitter_std
                obj_arg['weight_dirichlet_conc'] = datagen_config.weight_dirichlet_conc
                obj_arg['clip_range'] = datagen_config.gmm_clip_range
                synthetic_obj_list = simulate_synthetic_objects(image_size, data_config, len(probe_list),
                                                                datagen_config.object_class, obj_arg,
                                                                histogram = material_histogram,
                                                                stats = material_stats)
            except Exception as e:
                raise ValueError(f"Error: {str(e)}")

            # Remove all existing files from directory
            if not os.path.exists(synthetic_path):
                os.mkdir(synthetic_path)
            else:
                remove_all_files(synthetic_path)

            # Simulate and fill directory
            print("Simulating experiments...")
            probe_arg['beamstop_diameter'] = datagen_config.beamstop_diameter
            simulate_multiple_experiments(synthetic_obj_list, probe_list,
                                        datagen_config.diff_per_object,
                                        image_size, data_config, probe_arg,
                                        synthetic_path)
            
            print("Done with data generation")
            torch.cuda.empty_cache()

    return [data_config, model_config, training_config, inference_config, datagen_config], training_path
        

#Define main function
if __name__ == '__main__':
    #Parsing
    parser = argparse.ArgumentParser(description = "Run full training workflow for PtychoPINN using synthetic or experimental data")
    #Arguments
    parser.add_argument('--ptycho_dir', type = str, help = 'Path to ptycho directory')
    parser.add_argument('--config', type = str, default=None, help = 'Path to JSON configuration file (mandatory)')
    parser.add_argument('--mode', type = str, default=None, help = 'exp (experimental) or synth (synthetic)')
    parser.add_argument('--synth_dir', type = str, default = 'synthetic_training')
    parser.add_argument('--histogram_source_dir', type=str, default=None,
                        help='Directory with .npz files for histogram reference object. Defaults to ptycho_dir.')

    #Parse
    args = parser.parse_args()

    #Assign to vars
    ptycho_dir = args.ptycho_dir
    config_path = args.config
    mode = args.mode
    synth_dir = args.synth_dir
    histogram_source_dir = args.histogram_source_dir

    print(f"Configuration file: {config_path}")
    print(f"Current working directory: {os.getcwd()}")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    try:
        #Prepare data
        configs, training_path = prepare_data(ptycho_dir, mode, config_path,
                                              synthetic_path = synth_dir,
                                              histogram_source_dir = histogram_source_dir)
        main(training_path, config_path, existing_config=None)
        # train_ccnf(training_path, config_path, output_dir='lightning_outputs')

    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1)
