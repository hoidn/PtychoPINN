#Generic
from dataclasses import asdict, is_dataclass
import json
import os
import logging
from typing import Tuple, TypeVar, Type
from dataclasses import dataclass, field
#ML specific
import torch
import mlflow
#Custom functions
from ptycho_torch.config_params import update_existing_config
from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig

#Set mlflow directory

logger = logging.getLogger(__name__)
# Helper function to convert dataclass to JSON serializable dict, filtering Tensors

def config_to_json_serializable_dict(config_obj):
    """
    Function converting a config file to a serializable json dictionary.
    This is specifically for saving a set of config parameters to an easily readable json after training
    Config is saved as an artifact in mlflow's params
    """
    if not is_dataclass(config_obj):
        raise TypeError(f"Expected a dataclass instance, got {type(config_obj)}")

    d = asdict(config_obj)
    serializable_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            print(f"Warning: Skipping field '{k}' of type torch.Tensor during MLflow param logging.")
            continue # Skip tensors
        # Add checks here for other non-serializable types if needed
        serializable_dict[k] = v
    return serializable_dict

T = TypeVar('T')

def _load_single_config_from_mlflow(run_id: str,
                                   ConfigClass: Type[T],
                                   param_name: str,
                                   client: mlflow.tracking.MlflowClient,
                                   required: bool = True,
                                   post_process_fn = None) -> T:
    """Loads, deserializes, and updates a single dataclass config."""
    try:
        #Fetch run based on run id
        run = client.get_run(run_id)
        params = run.data.params
        
        #Grab json file param associated with class
        if param_name not in params:
            if required:
                logger.warning(f"Parameter '{param_name}' not found in MLflow run {run_id}. "
                               f"Returning default {ConfigClass.__name__}.")
            else:
                logger.info(f"Optional parameter '{param_name}' not found in MLflow run {run_id}. "
                           f"Returning default {ConfigClass.__name__}.")
            return ConfigClass() # Return default instance

        #Load json dictionary parameter specifically
        config_json_str = params[param_name]
        loaded_dict = json.loads(config_json_str)

        # Create a default instance
        config_instance = ConfigClass()

        # Apply post-processing if provided
        if post_process_fn:
            loaded_dict = post_process_fn(loaded_dict)

        # Update the default instance with loaded parameters
        update_existing_config(config_instance, loaded_dict)
        logger.info(f"Successfully reloaded and updated {ConfigClass.__name__} from run {run_id}.")
        return config_instance

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON for parameter '{param_name}' in run {run_id}: {e}")
        logger.error(f"JSON String was: {params.get(param_name, 'NOT FOUND')}")
        if required:
            raise ValueError(f"Invalid JSON found for {param_name}") from e
        else:
            logger.warning(f"Using default {ConfigClass.__name__} due to JSON decode error")
            return ConfigClass()
    except Exception as e:
        if required:
            logger.error(f"Failed to load config {ConfigClass.__name__} (param: {param_name}) from run {run_id}: {e}")
            raise # Re-raise other unexpected errors
        else:
            logger.warning(f"Failed to load optional config {ConfigClass.__name__}, using default. Error: {e}")
            return ConfigClass()

def load_all_configs_from_mlflow(run_id: str,
                                 mlflow_tracking_uri: str = None
                                ) -> Tuple[DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig]:
    """..."""
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    client = mlflow.tracking.MlflowClient()

    try:
        # Load required configs (these should always exist)
        data_cfg = _load_single_config_from_mlflow(run_id, DataConfig, "DataConfig_params", client, required=True, post_process_fn = fix_attribute)
        model_cfg = _load_single_config_from_mlflow(run_id, ModelConfig, "ModelConfig_params", client, required=True)
        train_cfg = _load_single_config_from_mlflow(run_id, TrainingConfig, "TrainingConfig_params", client, required=True)
        infer_cfg = _load_single_config_from_mlflow(run_id, InferenceConfig, "InferenceConfig_params", client, required=True, post_process_fn=fix_attribute)

        # Load optional config (backward compatibility)
        datagen_cfg = _load_single_config_from_mlflow(run_id, DatagenConfig, "DatagenConfig_params", client, required=False)

        return data_cfg, model_cfg, train_cfg, infer_cfg, datagen_cfg

    except Exception as e:
        logger.error(f"Failed to load one or more configs from MLflow run {run_id}: {e}")
        raise

def fix_attribute(config_dict):
    """For legacy data format compatibility. Post-processing function to fix normalize attribute."""
    if config_dict.get('normalize', None) and config_dict['normalize'] is True:
        config_dict['normalize'] = "Batch"
        config_dict['scan_pattern'] = "Rectangular"
        logger.info("Fixed normalize attribute: True -> 'Batch'")
    elif config_dict.get('middle_trim', None) and config_dict['middle_trim'] < 32:
        config_dict['middle_trim'] = 32
        logger.info("Fixed middle trim: < 32 to 32")
    
    return config_dict

def load_config_from_json(config_path):
    """
    Load configuration parameters from JSON file.
    Purpose is to load any replacement configurations for train.py
    
    Args:
        config_path (str): Path to the JSON configuration file
        
    Returns:
        dict: Dictionary containing configuration overrides for each config type
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        print(f"Successfully loaded configuration from: {config_path}")
        return config_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file {config_path}: {e}")
    except Exception as e:
        raise Exception(f"Error loading configuration file {config_path}: {e}")
    
def validate_and_process_config(config_data):
    """
    Validate and process the loaded configuration data. 
    Meant to be used after load_config_from_json.
    Takes a single config json dictionary and extracts the different configs, such as
    data, model, training, inference
    
    Args:
        config_data (dict): Raw configuration data from JSON
        
    Returns:
        tuple: (data_config_replace, model_config_replace, training_config_replace, inference_config_replace)
    """
    # Extract individual configuration sections
    data_config_replace = config_data.get('data_config', {})
    model_config_replace = config_data.get('model_config', {})
    training_config_replace = config_data.get('training_config', {})
    inference_config_replace = config_data.get('inference_config', {})
    datagen_config_replace = config_data.get('datagen_config', {})
    
    
    # Handle special cases and validation
    # Set C_forward and C_model based on data_config.C if not explicitly set
    if 'C' in data_config_replace:
        C_value = data_config_replace['C']
        if 'C_forward' not in model_config_replace:
            model_config_replace['C_forward'] = C_value
        if 'C_model' not in model_config_replace:
            model_config_replace['C_model'] = C_value

    # Determine object_big based on C value
    C_value = data_config_replace.get('C', 1)  # Default to 1 if not specified
    model_config_replace['object_big'] = C_value > 1
    
    print("Configuration validation completed.")
    
    return data_config_replace, model_config_replace, training_config_replace, inference_config_replace, datagen_config_replace

def remove_all_files(directory_path):
    """
    Remove all files in the specified directory without removing the directory itself.
    
    Args:
        directory_path (str): Path to the directory whose files should be removed
        
    Returns:
        int: Number of files removed
    """
    count = 0
    
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return count
        
    # Loop through all items in the directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        
        # Check if it's a file and remove it
        if os.path.isfile(item_path):
            os.remove(item_path)
            count += 1
            
    print(f"Removed {count} files from '{directory_path}'")
    return count