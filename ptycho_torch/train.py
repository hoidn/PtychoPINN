#Most basic modules
import sys
import argparse
import os

#ML libraries
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Subset

#Automation modules
#Lightning
import lightning as L
#MLFlow
import mlflow.pytorch
from mlflow import MlflowClient

#Configs/Params
from ptycho_torch.config_params import ModelConfig, TrainingConfig, DataConfig
from ptycho_torch.config_params import data_config_default, model_config_default, training_config_default

#Dataloader
from ptycho_torch.dset_loader_pt_mmap import TensorDictDataLoader, PtychoDataset

#Custom modules
from ptycho_torch.model import Autoencoder, CombineComplex, ForwardModel, PoissonLoss, MAELoss

#Helper function for mlflow
def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("PtychoPINN")


class PtychoPINN(L.LightningModule):
    '''
    Lightning module equivalent of PtychoPINN module from ptycho_torch.model
    We initialize all hyperparameters within here so that the Lightning trainer checkpoints it all
    '''
    def __init__(self):
        super().__init__()
        self.n_filters_scale = ModelConfig().get('n_filters_scale')
        self.predict = False
        #Autoencoder
        self.autoencoder = Autoencoder(self.n_filters_scale)
        self.combine_complex = CombineComplex()
        #Adding named modules for forward operation
        #Patch operations
        self.forward_model = ForwardModel()
        #Choose loss function
        if ModelConfig().get('loss_function') == 'Poisson':
            self.Loss = PoissonLoss()
        elif ModelConfig().get('loss_function') == 'MAE':
            self.Loss = MAELoss()
    
    def forward(self, x, positions, probe, scale_factor):
        #Autoencoder result
        x_amp, x_phase = self.autoencoder(x)
        #Combine amp and phase
        x_combined = self.combine_complex(x_amp, x_phase)
        #Run through forward model
        x_out = self.forward_model(x_combined, positions, probe, scale_factor)

        return x_out
    
    def training_step(self, batch, batch_id):
        #Grab required data fields from TensorDict
        x, positions, probe, scale = batch[0]['images'], \
                                     batch[0]['coords_relative'], \
                                     batch[1], \
                                     batch[2]
        #Run through forward model
        pred = self(x, positions, probe, scale)
        #Calculate loss
        loss = self.Loss(pred, x)

        #Logging
        self.log("poisson_train_loss", loss, on_epoch = True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 2e-2)

#Training functions

#Custom collation function which pins memory in order to transfer to gpu
#Taken from: https://pytorch.org/tensordict/stable/tutorials/tensorclass_imagenet.html
class Collate(nn.Module):
    def __init__(self, device = None):
        super().__init__()
        self.device = torch.device(device)
    def __call__(self, x):
        '''
        Moves tensor to RAM, and then to GPU.

        Inputs
        -------
        x: TensorDict
        '''
        #Move data from memory map to RAM
        if self.device.type == 'cuda':
            out = x.pin_memory()
        else: #cpu
            out = x
        #Then Ram to GPU
        if self.device:
            out = out.to(self.device)
        return out

def main(ptycho_dir, probe_dir):
    #Define configs
    print('Loading configs...')
    modelconfig = ModelConfig()
    trainingconfig = TrainingConfig()
    dataconfig = DataConfig()

    #Set configs
    modelconfig.set_settings(model_config_default)
    trainingconfig.set_settings(training_config_default)
    dataconfig.set_settings(data_config_default)

    #Creating dataset
    print('Creating dataset...')
    ptycho_dataset = PtychoDataset(ptycho_dir, probe_dir, remake_map=True)

    #Dataloader
    print('Creating dataloader...')
    train_loader = TensorDictDataLoader(ptycho_dataset, batch_size = 64,
                                        collate_fn = Collate(device = TrainingConfig().get('device')))

    #Create model
    print('Creating model...')
    model = PtychoPINN()

    #Create trainer
    trainer = L.Trainer(max_epochs = 3,
                        default_root_dir = os.path.dirname(os.getcwd()),
                        devices = 'auto',
                        accelerator = 'gpu')

    #Mlflow setup
    # mlflow.set_tracking_uri("")
    mlflow.set_experiment("PtychoPINN vanilla")

    mlflow.pytorch.autolog(checkpoint_monitor = "poisson_train_loss")

    #Train the model
    with mlflow.start_run() as run:
        print('Training model...')
        trainer.fit(model, train_loader)

    print_auto_logged_info(mlflow.get_run(run_id = run.info.run_id))

#Define main function
if __name__ == '__main__':
    #Parsing
    parser = argparse.ArgumentParser(description = "Run training for ptycho_torch")
    #Arguments
    parser.add_argument('--ptycho_dir', type = str, help = 'Path to ptycho directory')
    parser.add_argument('--probe_dir', type = str, help = 'Path to probe directory')
    #Parse
    args = parser.parse_args()

    #Assign to vars
    ptycho_dir = args.ptycho_dir
    probe_dir = args.probe_dir

    print(f"Probe: {probe_dir}")
    print(f"Ptycho: {ptycho_dir}")

    print(os.getcwd())

    try:
        main('datasets/dummy_data_small', 'datasets/probes')

    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1)
