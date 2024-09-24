#Most basic modules
import sys
import argparse

#ML libraries
import torch
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


class PtychoPINN(L.LightningModule):
    '''
    Lightning module equivalent of PtychoPINN module from ptycho_torch.model
    We initialize all hyperparameters within here so that the Lightning trainer checkpoints it all
    '''
    def __init__(self):
        super().__init__()
        self.n_filters_scale = ModelConfig().get('n_filters_scale')
        #Submodules
        self.autoencoder = Autoencoder(self.n_filters_scale)
        self.combine_complex = CombineComplex()
        self.forward_model = ForwardModel()

        if ModelConfig().get('loss_function') == 'Poisson':
            self.Loss = PoissonLoss()
        elif ModelConfig().get('loss_function') == 'MAE':
            self.Loss = MAELoss()
    
    def forward(self, x, positions, probe):
        #Autoencoder result
        x_amp, x_phase = self.autoencoder(x)
        #Combine amp and phase
        x_combined = self.combine_complex(x_amp, x_phase)
        #Run through forward model
        x_out = self.forward_model(x_combined, positions, probe)

        return x_out
    
    def training_step(self, batch, batch_id):
        #Grab required data fields from TensorDict
        x, positions, probe = batch[0]['images'], batch[0]['coords_relative'], batch[1]
        #Run through forward model
        pred = self(x, positions, probe)
        #Calculate loss
        loss = self.PoissonLoss(pred, x)

        #Logging
        self.log("poisson_train_loss", loss, on_epoch = True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 2e-2)

def main(ptycho_dir, probe_dir):
    #Define configs
    modelconfig = ModelConfig()
    trainingconfig = TrainingConfig()
    dataconfig = DataConfig()

    #Set configs
    modelconfig.set_settings(model_config_default)
    trainingconfig.set_settings(training_config_default)
    dataconfig.set_settings(data_config_default)

    #Creating dataset
    ptycho_dataset = PtychoDataset(ptycho_dir, probe_dir, remake_map=True)

    #Dataloader
    train_loader = TensorDictDataLoader(ptycho_dataset, batch_size = 64)

    #Create model
    model = PtychoPINN()

    #Create trainer
    trainer = L.Trainer(max_epochs = 100)

    #Mlflow setup
    # mlflow.set_tracking_uri("")
    mlflow.set_experiment("PtychoPINN vanilla")

    mlflow.pytorch.autolog(checkpoint_monitor = "poisson_train_loss")

    #Train the model
    with mlflow.start_run() as run:
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

    try:
        main(ptycho_dir, probe_dir)

    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1)
