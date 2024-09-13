#Most basic modules
import sys

#Going to use lightning to handle most training
import lightning as L
import torch
from torch.nn import functional as F
from torch.utils.data import Subset

#Configs/Params
from ptycho_torch.config_params import Config, Params

import mlflow.pytorch
from mlflow import MlflowClient

from ptycho_torch.model import Autoencoder, CombineComplex, ForwardModel, PoissonLoss
from ptycho_torch.dset_loader_pt_mmap import TensorDictDataLoader

class PtychoPINN(L.LightningModule):
    '''
    Lightning module equivalent of PtychoPINN module from ptycho_torch.model
    We initialize all hyperparameters within here so that the Lightning trainer checkpoints it all
    '''
    def __init__(self):
        super().__init__()


        self.n_filters_scale = Config().get('n_filters_scale')
        #Submodules
        self.autoencoder = Autoencoder(self.n_filters_scale)
        self.combine_complex = CombineComplex()
        self.forward_model = ForwardModel()
        self.PoissonLoss = PoissonLoss()
    
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

        self.log("train_loss", loss, on_epoch = True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 2e-2)

def main(dataset, config_list, params_list):
    #Can grab this from some config/params file later or default

    #Set configs and params
    config = Config()
    config.set_settings(config_list)
    params = Params()
    params.set_settings(params_list)

    #Dataloader
    train_loader = TensorDictDataLoader(dataset, batch_size = 64)

    #Create model
    model = PtychoPINN()

    #Create trainer
    trainer = L.Trainer(max_epochs = 100)

#Define main function
if __name__ == '__main__':
    try:
        print('test')

    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1)
