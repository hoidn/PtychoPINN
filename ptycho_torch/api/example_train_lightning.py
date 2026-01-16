
import sys
import os
from datetime import datetime
import time

#1. Synchronize timestamp across processes using filesystem
base_output_dir = "/local/PtychoPINN/lightning_outputs"
timestamp_file = os.path.join(base_output_dir, '.timestamp_lock')

is_rank_zero = os.environ.get('RANK', '0') == '0' or 'RANK' not in os.environ

if is_rank_zero:
    # Rank 0: Create timestamp and write to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(base_output_dir, exist_ok=True)
    with open(timestamp_file, 'w') as f:
        f.write(timestamp)
    print(f"[Rank 0] Created timestamp: {timestamp}")
else:
    # Other ranks: Wait for file and read timestamp
    print(f"[Rank {os.environ.get('RANK', '?')}] Waiting for timestamp file...")
    max_wait = 30
    wait_time = 0
    while not os.path.exists(timestamp_file) and wait_time < max_wait:
        time.sleep(0.1)
        wait_time += 0.1
    
    if not os.path.exists(timestamp_file):
        raise RuntimeError(f"Rank {os.environ.get('RANK', '?')} timeout waiting for timestamp")
    
    with open(timestamp_file, 'r') as f:
        timestamp = f.read().strip()
    print(f"[Rank {os.environ.get('RANK', '?')}] Read timestamp: {timestamp}")


sys.path.append(os.path.abspath('/local/PtychoPINN'))
from ptycho_torch.api.base_api import ConfigManager, PtychoDataLoader, PtychoModel, Trainer, InferenceEngine, DataloaderFormats
from ptycho_torch.api.base_api import Orchestration
from ptycho_torch.model import PtychoPINN_Lightning
from ptycho_torch.train_utils import is_effectively_global_rank_zero

#Following the notebook, we'll be loading a configuration from another mlflow run and doing a short training run to show the api off

#2. Config Manager
json_dir = "/local/CDI-PINN/ptychopinn_torch/configs/publication_configs/4_q_multi_gpu_velociprobe.json"


config_manager, _= ConfigManager._from_json(json_path = json_dir)
# Update config manager                                              
training_update = {'epochs': 2,
                   'epochs_fine_tune': 1,
                   'orchestrator': 'Lightning'}
config_manager.update(training_config = training_update)

#3. Dataloader
ptycho_data_dir = "/local/CDI-PINN/data/pinn_velo_fly001"

lightning_dataloader = PtychoDataLoader(data_dir = ptycho_data_dir,
                                        config_manager = config_manager,
                                        data_format = DataloaderFormats('lightning_only_module'),
                                        output_dir = base_output_dir,
                                        timestamp = timestamp)

#3. Model
new_ptycho_model = PtychoModel._new_model(model = PtychoPINN_Lightning,
                                          config_manager = config_manager)

lightning_trainer = Trainer._from_lightning(model = new_ptycho_model,
                                            dataloader = lightning_dataloader,
                                            orchestration = 'lightning',
                                            config_manager = config_manager)

output_dir = lightning_trainer.train(orchestration = "lightning",
                                  experiment_name = 'test_run')

print(f"Output directory is: {output_dir}")

#4. Saving in new location (do not need to load beforehand)
print("Loading model")
if is_effectively_global_rank_zero():

    new_destination = '/local/Demo_Directory_PtychoPINN'

    #This save function is just a fancy copy function elsewhere.
    new_ptycho_model.save(path = new_destination,
                          source_run_path = output_dir,
                          strategy = "lightning")                                        

