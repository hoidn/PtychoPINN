import sys
import os
sys.path.append(os.path.abspath('/local/PtychoPINN'))
from ptycho_torch.api.base_api import ConfigManager, PtychoDataLoader, PtychoModel, Trainer, InferenceEngine, DataloaderFormats
from ptycho_torch.api.base_api import Orchestration
from ptycho_torch.model import PtychoPINN_Lightning
#Following the notebook, we'll be loading a configuration from another mlflow run and doing a short training run to show the api off

#1. Config Manager

mlflow_tracking_uri = "/local/CDI-PINN/mlruns"
run_id = "f637381fd7fe49158bb0ed2e7a28ca45"

config_manager_mlflow = ConfigManager._from_mlflow(run_id,
                                                   mlflow_tracking_uri)
# Update config manager                                              
training_update = {'epochs': 1,
                   'epochs_fine_tune': 0}
config_manager_mlflow.update(training_config = training_update)
config_manager_mlflow.training_config


#2. Dataloader
ptycho_data_dir = "/local/CDI-PINN/data/pinn_velo_ncm"
lightning_dataloader = PtychoDataLoader(data_dir = ptycho_data_dir,
                                        config_manager = config_manager_mlflow,
                                        data_format = DataloaderFormats('lightning_module'))

#3. Model
new_ptycho_model = PtychoModel._new_model(model = PtychoPINN_Lightning,
                                          config_manager = config_manager_mlflow)

lightning_trainer = Trainer._from_lightning(model = new_ptycho_model,
                                            dataloader = lightning_dataloader,
                                            config_manager = config_manager_mlflow)

run_ids = lightning_trainer.train(orchestration = "mlflow",
                                  experiment_name = 'Test_run')

print(f"Run ids are {run_ids}")

#4. Saving
#A bit awkward since you'll have to load your new model and then save it elsewhere 
trained_ptycho_model = PtychoModel._load(strategy = 'mlflow',
                                        run_id = run_ids.get('training'),
                                        mlflow_tracking_uri = mlflow_tracking_uri)

new_destination = '/local/Demo_Directory_PtychoPINN'
trained_ptycho_model.save(new_destination,
                          strategy = 'mlflow')                                        

