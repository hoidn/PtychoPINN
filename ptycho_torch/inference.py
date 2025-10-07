#Generic
import os
import time
from datetime import datetime
import argparse
import sys

#ML libraries
import mlflow
import matplotlib.pyplot as plt
import numpy as np


#Custom
from ptycho_torch.utils import load_all_configs_from_mlflow
from ptycho_torch.reassembly import reconstruct_image_barycentric
from ptycho_torch.config_params import update_existing_config
from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig
from ptycho_torch.utils import load_config_from_json, validate_and_process_config, remove_all_files
from ptycho_torch.dataloader import PtychoDataset

def load_all_configs(config_path, file_index):
    """
    Helper functions that loads all relevant configs specifically for inference
    File index is updated based on argument from argparse
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
    
    training_config = TrainingConfig()
    if t_config_replace is not None:
        update_existing_config(training_config, t_config_replace)    

    inference_config = InferenceConfig()
    if i_config_replace is not None:
        update_existing_config(inference_config, i_config_replace)

    datagen_config = DatagenConfig()
    if dgen_config_replace is not None:
        update_existing_config(datagen_config, dgen_config_replace)

    return data_config, model_config, training_config, inference_config, datagen_config




#Loads model, training settings
def load_and_predict(run_id,
                     ptycho_files_dir,
                     relative_mlflow_path = 'mlruns',
                     config_override_path = None,
                     file_index = 0,
                     save_dir = "inference/output",
                     plot_name = "Test",
                     verbose = False):
    '''
    Given MLFlow run id, as well as ptycho file directory, will provide predictions 
    Args:
        run_id: Unique MLflow run id generated upon training finishing
        ptycho_files_dir: File where all experimental ptychography files are saved
        relative_mlflow_path: directory where mlruns is bineg saved. Should be modifiable/configurable in train.py
    '''
    
    #MLFlow tracking for model
    tracking_uri = f"file:{os.path.abspath(relative_mlflow_path)}"
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"runs:/{run_id}/model"
    #Loading config
    if not config_override_path:
        data_config, model_config, training_config, inference_config, datagen_config = load_all_configs_from_mlflow(run_id,
                                                                                         tracking_uri)
    else:
        data_config, model_config, training_config, inference_config, datagen_config = load_all_configs(config_override_path)

    # Manually overriding experiment number indexing
    i_config_replace = {}
    i_config_replace['experiment_number'] = file_index
    update_existing_config(inference_config, i_config_replace)

    #Loading model
    model_load_start = time.time()
    loaded_model = mlflow.pytorch.load_model(model_uri)
    loaded_model.to(training_config.device)
    loaded_model.training = True
    model_load_time = time.time() - model_load_start

    #Load data into dataset structure
    data_load_start = time.time()
    ptycho_dataset = PtychoDataset(ptycho_files_dir, model_config, data_config,
                                remake_map=True)
    
    data_load_time = time.time() - data_load_start
    
    #Reconstructing. Automatically puts dataset into dataloader, so don't worry about it
    if verbose:
        print(f"Data config: {data_config}")
        print(f"Model config: {model_config}")
        print(f"Inference config: {inference_config}")
    result, recon_dataset, assembly_stats = reconstruct_image_barycentric(loaded_model, ptycho_dataset,
                           training_config, data_config, model_config, inference_config, gpu_ids = None,
                           use_mixed_precision=True, verbose = False)

    
    #Save results
    result_im = result.to('cpu')
    if len(result_im.shape) == 3:
        result_im = result_im[0].squeeze()
    
    w = inference_config.window
    result_amp = np.abs(result_im)
    result_phase = np.angle(result_im) 
    gt_amp = np.abs(recon_dataset.data_dict['objectGuess']).squeeze()
    gt_phase = np.angle(recon_dataset.data_dict['objectGuess']).squeeze()

    plot_amp_and_phase(result_amp[w:-w,w:-w], result_phase[w:-w,w:-w],
                       gt_amp[w:-w,w:-w], gt_phase[w:-w,w:-w],
                       save_dir = save_dir, filename = plot_name)

    print(f"Model load time: {model_load_time} \n "
          f"Data load time: {data_load_time}\n"
          f"Total inference time: {assembly_stats[0]}\n"
          f"Total assembly time: {assembly_stats[1]}")
    
    return result


def plot_amp_and_phase(obj_amp, obj_phase, gt_amp, gt_phase, save_dir = None, filename = None):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments for inference script")
    parser.add_argument('--run_id', type = str, help = "Unique run id associated with training run")
    parser.add_argument('--infer_dir', type = str, help = "Inference directory")
    parser.add_argument('--file_index', type = int, default = 0, help = "File index if more than one file in infer_dir")
    parser.add_argument('--config', type = str, default = None, help = "Config to override loaded values")

    args = parser.parse_args()

    run_id = args.run_id
    infer_dir = args.infer_dir
    file_index = args.file_index
    config_override = args.config

    try:
        load_and_predict(run_id, infer_dir, 'mlruns',
                         config_override_path=config_override,
                         file_index = file_index)
    except Exception as e:
        print(f"Inference failed because of: {str(e)}")
        sys.exit(1)





