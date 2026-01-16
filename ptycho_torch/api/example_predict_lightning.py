import sys
import os
sys.path.append(os.path.abspath('/local/PtychoPINN'))
from ptycho_torch.api.base_api import ConfigManager, PtychoDataLoader, PtychoModel, Trainer, InferenceEngine, DataloaderFormats
from ptycho_torch.api.base_api import Orchestration
from ptycho_torch.model import PtychoPINN_Lightning
#Following the notebook, we'll be loading a configuration from another mlflow run and doing a short training run to show the api off

#1. Config Manager

mlflow_tracking_uri = "/local/CDI-PINN/mlruns"
run_id = "06822d7239504a93ae0f7a6c4577cdc8" #Trained model

config_manager_mlflow = ConfigManager._from_mlflow(run_id,
                                                   mlflow_tracking_uri)


#2. Dataloader
ptycho_data_dir = "/local/CDI-PINN/data/pinn_velo_ncm"
tensordict_dataloader = PtychoDataLoader(data_dir = ptycho_data_dir,
                                        config_manager = config_manager_mlflow,
                                        data_format = 'tensordict')

#3. Model
trained_ptycho_model = PtychoModel._load(config_manager = config_manager_mlflow,
                                         strategy = 'mlflow',
                                         run_id = run_id,
                                         mlflow_tracking_uri = mlflow_tracking_uri)


#4. Inference
print(f"Beginning inference...")
ptycho_inference = InferenceEngine(config_manager = config_manager_mlflow,
                                   ptycho_model = trained_ptycho_model,
                                   ptycho_dataloader = tensordict_dataloader)     
 



