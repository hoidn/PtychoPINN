#Torch
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist

#Other math
import numpy as np
import math
from typing import Optional

#Helper
from ptycho_torch.config_params import ModelConfig, TrainingConfig, DataConfig, InferenceConfig, update_existing_config
import ptycho_torch.helper as hh
from ptycho_torch.model_attention import CBAM, ECALayer, BasicSpatialAttention
from ptycho_torch.train_utils import MultiStageLRScheduler, AdaptiveLRScheduler
import copy

#Lightning
import lightning as L


#Ensuring 64float b/c of complex numbers
# torch.set_default_dtype(torch.float32)

#Helping modules
#Activation functions
class Tanh_custom_act(nn.Module):
    '''
    Custom tanh activation module used in:
        Decoder_phase
    '''
    def forward(self, x):
        return math.pi * torch.tanh(x)

class Amplitude_activation(nn.Module):
    '''
    Custom amplitude activation module:

    Inputs
    ---------
    activation(string, optional): Defaults to swish, normally grabs from model configs
    beta (float, optional): Beta parameter, controls steepness of swish
    inplace (bool, optional): If set to true, does operation in-place
    model_config (ModelConfig): Configuration object.

    '''
    def __init__(self, model_config: ModelConfig, beta=1.0, inplace=False):
        super(Amplitude_activation, self).__init__()
        self.model_config = model_config
        self.activation_type = self.model_config.amp_activation
        self.beta = beta

    def forward(self, x):
        if self.activation_type == 'sigmoid':
            return torch.sigmoid(x)
        else:
            # if self.inplace and x.is_floating_point():
            #     x.mul_(torch.sigmoid(self.beta * x))
            #     return x
            # else:
            #     return x * torch.sigmoid(self.beta * x)
            return F.silu(x)

#Conv blocks
class ConvBaseBlock(nn.Module):
    '''
    Convolutional base block for Pooling and Upscaling

    If padding = same, padding is half of kernel size
    '''
    def __init__(self, in_channels, out_channels,
                 w1 = 3, w2 = 3,
                 padding = 'same',
                 activation = 'relu',
                 batch_norm = False):
        super(ConvBaseBlock, self).__init__()
        padding_size = w1 // 2 if padding == 'same' else 0
        #NN layers
        self.conv1 = nn.Conv2d(in_channels = in_channels,
                               out_channels = out_channels,
                               kernel_size = (w1, w2),
                               padding = padding_size)
        self.conv2 = nn.Conv2d(in_channels = out_channels,
                               out_channels = out_channels,
                               kernel_size = (w1, w2),
                               padding = padding_size)
        #Activation used in upblock
        self.activation = getattr(F, activation) if activation else None

        #Batchnorm (optional) - Initialize conditionally
        self.batch_norm = batch_norm
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else None

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.activation(x) if self.activation else F.relu(x)
        
        # x = self._pool_or_up(x)  # ← ADD THIS LINE!
        return x
    


    
class ConvPoolBlock(ConvBaseBlock):

    def __init__(self, in_channels, out_channels, model_config: ModelConfig,
                 w1 = 3, w2 = 3, p1 = 2, p2 = 2,
                 padding = 'same', batch_norm = False):
        super(ConvPoolBlock, self).__init__(in_channels, out_channels,
                                            w1=w1, w2=w2, padding=padding,
                                            batch_norm = batch_norm)
        #Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=(p1, p2),
                                 padding = 0)
                   
        #CBAM
        self.use_cbam = model_config.cbam_encoder
        self.use_eca = model_config.eca_encoder

        #Decide on attention mechanism
        if self.use_cbam:
            # CBAM operates on the output channels of the conv layers
            self.attention = CBAM(gate_channels=out_channels)
        elif self.use_eca:
            self.attention = ECALayer(out_channels)
        else:
            self.attention = nn.Identity()
        
    # def _pool_or_up(self, x):
    #     x = self.attention(x)  # Apply attention first
    #     x = self.pool(x)       # Then pool
    #     return x

    def forward(self,x):
        x_new = super().forward(x)
        if self.use_cbam:
            x_new = self.attention(x_new)

        x = self.pool(x_new)
        return x
    
class ConvUpBlock(ConvBaseBlock):

    def __init__(self, in_channels, out_channels,
                 w1 = 3, w2 = 3, p1 = 2, p2 = 2,
                 padding = 'same', batch_norm = False):
        
        super().__init__(in_channels, out_channels,
                                            w1=w1, w2=w2, padding=padding,
                                            batch_norm = batch_norm)
        padding_size = w1 // 2 if padding == 'same' else 0
        #NN layers
        self.up = nn.Upsample(scale_factor = (p1, p2),
                              mode = 'nearest')

    # def _pool_or_up(self, x):
    #     return self.up(x) 
    def forward(self,x):
        x = super().forward(x)
        x = self.up(x)

        return x

#Encoder

class Encoder(nn.Module):
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super(Encoder, self).__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.n_filters_scale = model_config.n_filters_scale

        self.N = self.data_config.N
        starting_coeff = 64 / (self.N / 32)
        self.filters = [model_config.C_model if model_config.object_big else 1]

        #Starting output channels is 64. Last output size will always be n_filters_scale * 128.
        if self.N == 64:
            self.filters = self.filters + [self.n_filters_scale * 32, self.n_filters_scale * 64, self.n_filters_scale * 128]
        elif self.N == 128:
            self.filters = self.filters + [self.n_filters_scale * 16, self.n_filters_scale * 32, self.n_filters_scale * 64, self.n_filters_scale * 128]
        elif self.N == 256:
            self.filters = self.filters + [self.n_filters_scale * 8, self.n_filters_scale * 16, self.n_filters_scale * 32, self.n_filters_scale * 64, self.n_filters_scale * 128]



        if starting_coeff < 3 or starting_coeff > 64:
            raise ValueError(f"Unsupported input size: {self.N}")

        
        self.blocks = nn.ModuleList([
            ConvPoolBlock(in_channels=self.filters[i-1],
                          out_channels=self.filters[i],
                          model_config=model_config, # Pass config here
                          batch_norm=model_config.batch_norm)
            for i in range(1, len(self.filters))
        ])
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x
    
#Decoders

class Decoder_filters(nn.Module):
    '''
    Base decoder class handling dynamic channel sizing in self.filters
    '''
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super(Decoder_filters, self).__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.n_filters_scale = model_config.n_filters_scale
        self.N = self.data_config.N

        #Calculate number of channels for upscaling
        #Start from self.N and divide by 2 until 32 for each layer
        #E.g.
        #N == 64: [self.n_filters_scale * 64, self.n_filters_scale * 32]
        #N == 128: [self.n_filters_scale * 128, self.n_filters_scale * 64, self.n_filters_scale * 32]
        self.filters = [self.n_filters_scale * 128]

        if self.N == 64:
            self.filters = self.filters + [self.n_filters_scale * 64, self.n_filters_scale * 32]
        elif self.N == 128:
            self.filters = self.filters + [self.n_filters_scale * 128, self.n_filters_scale * 64, self.n_filters_scale * 32]
        elif self.N == 256:
            self.filters = self.filters + [self.n_filters_scale * 256, self.n_filters_scale * 128, self.n_filters_scale * 64, self.n_filters_scale * 32]

        if self.N < 64:
            raise ValueError(f"Unsupported input size: {self.N}")

        if self.N < 64:
            raise ValueError(f"Unsupported input size: {self.N}")
        
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward")
    
class Decoder_base(Decoder_filters):
    # Accept batch_norm flag
    def __init__(self, model_config: ModelConfig, data_config: DataConfig, batch_norm=False):
        super(Decoder_base, self).__init__(model_config, data_config)
        #Attention
        self.use_cbam = model_config.cbam_decoder
        self.use_eca = model_config.eca_decoder
        self.use_spatial = model_config.spatial_decoder
        self.spatial_kernel = model_config.decoder_spatial_kernel

        #Layers - Pass batch_norm flag to ConvUpBlock
        self.blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()

        #Add optional attention layers, identity otherwise
        for i in range(1, len(self.filters)):
            in_ch = self.filters[i-1]
            out_ch = self.filters[i]
            self.blocks.append(ConvUpBlock(in_channels=in_ch,
                                           out_channels=out_ch,
                                           batch_norm=batch_norm))
            if self.use_eca:
                # Add ECA Layer - Needs output channels
                print(f"Decoder Block {i}: Adding ECALayer with {out_ch} channels.")
                self.attention_blocks.append(ECALayer(channel=out_ch)) # k_size can be adapted if needed
            elif self.use_spatial:
                # Add Basic Spatial Attention Layer - Needs kernel size
                print(f"Decoder Block {i}: Adding BasicSpatialAttention with kernel {self.spatial_kernel}.")
                self.attention_blocks.append(BasicSpatialAttention(kernel_size=self.spatial_kernel))
            elif self.use_cbam:
                 # Add CBAM Layer - Needs output channels
                 print(f"Decoder Block {i}: Adding CBAM with {out_ch} channels.")
                 self.attention_blocks.append(CBAM(gate_channels=out_ch))
            else:
                # Add Identity if no attention is selected for the decoder
                print(f"Decoder Block {i}: No attention module added.")
                self.attention_blocks.append(nn.Identity())
        
    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
            # x = self.attention_blocks[i](x)

        return x

class Decoder_last(nn.Module):
    '''
    Base class for the final decoder stage. Handles conditional BatchNorm.
    '''
    def __init__(self, model_config: ModelConfig, data_config: DataConfig,
                 in_channels, out_channels,
                 activation = torch.sigmoid, name = '', batch_norm=False): # Added batch_norm flag
        super(Decoder_last, self).__init__()
        #Configs
        self.model_config = model_config
        self.data_config = data_config
        self.n_filters_scale = model_config.n_filters_scale

        #Grab parameters
        self.N = self.data_config.N
        self.gridsize = self.data_config.grid_size
        
        #Dynamically calculate number of outer channels
        c_outer_fraction = getattr(model_config,'decoder_last_c_outer_fraction', 0.25)
        c_outer_fraction = max(0.0, min(0.5, c_outer_fraction))
        self.c_outer = max(1, int(in_channels * c_outer_fraction))

        #Layers
        self.conv1 =  nn.Conv2d(in_channels = in_channels - self.c_outer,
                                out_channels = out_channels,
                                kernel_size = (3, 3),
                                padding = 3//2)

        #conv_up_block and conv2 are separate to conv1
        self.conv_up_block = ConvUpBlock(self.c_outer, self.n_filters_scale * 32,
                                         batch_norm = batch_norm) # Pass flag
        
        self.conv2 =  nn.Conv2d(in_channels = self.n_filters_scale * 32,
                                out_channels = out_channels,
                                kernel_size = (3, 3),
                                padding = 3//2)

        # BatchNorm layers (conditional)
        self.batch_norm = batch_norm
        # BN for conv1 output (applied before activation)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else None
        # BN for conv2 output (applied before silu)
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else None

        #Additional
        self.activation = activation
        self.padding = nn.ConstantPad2d((self.N // 4, self.N // 4,
                                         self.N // 4, self.N //4), 0)
        


    def forward(self,x):
        # Path 1
        x1_in = x[:, :-self.c_outer, :, :]
        # x1_in = self.upsample(x1_in) #(N//2, N//2) -> N,N
        x1 = self.conv1(x1_in)
        if self.batch_norm and self.bn1:
             x1 = self.bn1(x1) # Apply BN before activation
        x1 = self.activation(x1) #05-20-2025 for now
        x1 = self.padding(x1)

        if not self.model_config.probe_big:
            return x1

        # Path 2
        x2 = self.conv_up_block(x[:, -self.c_outer:, :, :])
        x2 = self.conv2(x2)
        if self.batch_norm and self.bn2:
             x2 = self.bn2(x2) # Apply BN before silu

        x2 = F.silu(x2) #05-20-2025 for now

        outputs = x1 + x2

        # outputs = hh.trim_and_pad_output(outputs, self.data_config, self.model_config)


        return outputs



    
class Decoder_last_Amp(Decoder_last):
    '''Final decoder stage for Amplitude. Inherits from Decoder_last, ensuring batch_norm=False.'''
    def __init__(self, model_config: ModelConfig, data_config: DataConfig,
                 in_channels, out_channels,
                 activation = torch.sigmoid, name = ''):
        # Explicitly call parent with batch_norm=False
        super(Decoder_last_Amp, self).__init__(model_config, data_config, in_channels, out_channels,
                                               activation=activation, name=name, batch_norm=False)

class Decoder_last_Phase(Decoder_last):
    '''Final decoder stage for Phase. Inherits from Decoder_last, ensuring batch_norm=True.'''
    def __init__(self, model_config: ModelConfig, data_config: DataConfig,
                 in_channels, out_channels,
                 activation = torch.sigmoid, name = ''):
        # Explicitly call parent with batch_norm=True (or as configured)
        super(Decoder_last_Phase, self).__init__(model_config, data_config, in_channels, out_channels,
                                                 activation=activation, name=name, batch_norm=model_config.batch_norm) # Use config BN

class Decoder_phase(Decoder_base):
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        # Initialize base with batch_norm setting from config
        super(Decoder_phase, self).__init__(model_config, data_config, batch_norm=model_config.batch_norm)
        self.model_config = model_config # Store configs if needed directly
        self.data_config = data_config

        if self.model_config.object_big:
            num_channels = model_config.C_model
        else:
            num_channels = 1
        #Nn layers

        #Custom nn layers with specific identifiable names
        self.add_module('phase_activation', Tanh_custom_act())
        # Use Decoder_last_Phase which includes BatchNorm
        self.add_module('phase', Decoder_last_Phase(model_config, data_config, # Pass configs
                                                    self.n_filters_scale * 32, num_channels,
                                                    activation = self.phase_activation))
        # self.blocks are already correctly initialized with batch_norm in the super().__init__ call

    def forward(self, x):
        # Apply upscale block layers (now with BN from Decoder_base)
        for block in self.blocks:
            x = block(x)

        #Apply final layer
        outputs = self.phase(x)

        return outputs

class Decoder_amp(Decoder_base):
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        # Initialize base with batch_norm=False
        super(Decoder_amp, self).__init__(model_config, data_config, batch_norm=False)
        self.model_config = model_config # Store configs if needed directly
        self.data_config = data_config

        #Set number of decoder last channels
        if model_config.mode == 'Unsupervised':
            # num_channels = int(model_config.decoder_last_amp_channels) #Need for unsupervised
            num_channels = copy.deepcopy(model_config.decoder_last_amp_channels)
        elif model_config.mode == 'Supervised':
            num_channels = copy.deepcopy(model_config.decoder_last_amp_channels) #Need this for supervised learning
        #Assert sizing (can either match channels or is 1)

        assert num_channels in [1, model_config.C_model]      

        #Custom nn layers with specific identifiable names
        self.add_module('amp_activation', Amplitude_activation(model_config)) # Pass config
        # Use Decoder_last_Amp

        self.add_module('amp', Decoder_last_Amp(model_config, data_config, # Pass configs
                                                self.n_filters_scale * 32, num_channels,
                                                activation = self.amp_activation))


    def forward(self, x):
        #Apply upscale block layers
        for block in self.blocks:
            x = block(x)

        #Apply final layer
        outputs = self.amp(x)

        return outputs
    
#Autoencoder

class Autoencoder(nn.Module):
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super(Autoencoder, self).__init__()
        self.model_config = model_config
        self.data_config = data_config
        #Batch norm setting from config
        self.batch_norm = self.model_config.batch_norm
        #CBAM
        self.use_cbam = self.model_config.cbam_bottleneck
        #Encoder
        self.encoder = Encoder(model_config, data_config) # Pass configs
        #CBAM
        if self.use_cbam:
            # Determine bottleneck channels (output of encoder)
            bottleneck_channels = self.encoder.filters[-1]
            self.bottleneck_cbam = CBAM(gate_channels=bottleneck_channels)
        else:
            self.bottleneck_cbam = nn.Identity() # Use Identity if CBAM is off
        #Decoders (Amplitude/Phase)
        self.decoder_amp = Decoder_amp(model_config, data_config) # Pass configs
        self.decoder_phase = Decoder_phase(model_config, data_config) # Pass configs


    def forward(self, x):
        #Encoder
        x = self.encoder(x)
        #CBAM optional
        x = self.bottleneck_cbam(x)
        #Decoders
        x_amp = self.decoder_amp(x)
        x_phase = self.decoder_phase(x)

        return x_amp, x_phase

#Probe modules
class ProbeIllumination(nn.Module):
    '''
    Probe illumination done using hadamard product of object tensor and 2D probe function.
    2D probe function should be supplised by the dataloader

    Handles multiple probe modes in the forward call, adding an additional tensor dimension to
    facilitate probe multiplication

    Output x dimension: (B, C, P, H, W)
    P is the probe dimension, which will be 1 for single probes, but greater than that for multiple probes

    Inputs:
        x - torch.Tensor (B,C,H,W)
        p - torch.Tensor (B,C,P,H,W), P dimension > 1 if multi-modal
    '''
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super(ProbeIllumination, self).__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.N = self.data_config.N
        self.mask = self.model_config.probe_mask # Get mask from config

    def forward(self, x, probe):

        #Add extra dimension to x
        x_reshaped = x.unsqueeze(dim=2) # (B,C,H,W) -> (B,C,1,H,W)

        # print('probe shape', probe.shape)
        # print('xreshaped shape', x_reshaped.shape)

        #Check if probe mask exists
        #If not, probe mask is just a ones matrix
        #If mask exists, save mask is class attribute
        if self.mask is None: # Check if mask is None
            probe_mask = torch.ones((self.N, self.N)).to(x.device)
        else:
            probe_mask = self.mask.to(x.device) # Use the mask from config
        
        #(N, C, P, H, W)
        illuminated = x_reshaped * probe * probe_mask.view(1,1,1,self.N, self.N)

        return illuminated, probe * probe_mask


#Other modules
class CombineComplex(nn.Module):
    '''
    Converts real number amplitude and phase into single complex number for FFT

    Inputs
    ------
    amp: torch.Tensor
        Amplitude of complex number
    phi: torch.Tensor
        Phase of complex number
    
    Outputs
    -------
    out: torch.Tensor
        Complex number
    '''
    def __init__(self):
        super(CombineComplex, self).__init__()

    def forward(self, amp: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:

        out = amp.to(dtype=torch.complex64) * \
                torch.exp(1j * phi.to(dtype=torch.complex64))
        
        return out
    
class LambdaLayer(nn.Module):
    '''
    Generic layer module for helper functions.

    Mostly used for patch reconstruction

    Note from 11/15/2024: Pytorch lightning really doesn't like LambdaLayers. 
    Will treat them as if they were identity operations.
    Replaced all LambdaLayers in the forward model with their respective helper functions
    '''
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func
    
    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class PoissonIntensityLayer(nn.Module):
    '''
    Applies poisson intensity scaling using torch.distributions
    Calculates the negative log likelihood of observing the raw data given the predicted intensities
    '''
    def __init__(self, amplitudes):

        super(PoissonIntensityLayer, self).__init__()
        #Poisson rate parameter (lambda)
        Lambda = amplitudes ** 2
        #Create Poisson distribution
        #Second parameter (batch size) controls how many dimensions are summed over starting from the last
        self.poisson_dist = dist.Independent(dist.Poisson(Lambda), 3)

    def forward(self, x):
        #Apply poisson distribution
        return -self.poisson_dist.log_prob(x)
    
class ForwardModel(nn.Module):
    '''
    Forward model receiving complex object prediction, and applies physics-informed real space overlap
    constraints to the solution space.

    Inputs
    ------
    x: torch.Tensor (N, C, H, W), dtype = complex64
    positions: torch.Tensor (N, C, 1, 2), dtype = float32
        Positions of patches in real space
    probe: torch.Tensor, dtype = complex64
        Probe function
    '''
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super(ForwardModel, self).__init__()
        self.model_config = model_config
        self.data_config = data_config

        #Configuration from passed instances
        self.n_filters_scale = self.model_config.n_filters_scale
        self.N = self.data_config.N
        self.gridsize = self.data_config.grid_size
        self.offset = self.model_config.offset
        self.object_big = self.model_config.object_big

        #Patch operations
        #Lambdalayer here doesn't work for lightning module
        self.reassemble_patches = LambdaLayer(hh.reassemble_patches_position_real)

        self.pad_patches = LambdaLayer(hh.pad_patches)

        self.trim_reconstruction = LambdaLayer(hh.trim_reconstruction)

        self.extract_patches = LambdaLayer(hh.extract_channels_from_region)

        #Probe Illumination - Pass configs
        self.probe_illumination = ProbeIllumination(model_config, data_config)

        #Pad/diffract
        self.pad_and_diffract = LambdaLayer(hh.pad_and_diffract)

        #Scaling
        self.scaler = IntensityScalerModule(model_config)

    def forward(self, x, positions, probe, output_scale_factor):
        #Reassemble patches

        if self.object_big:
            # Pass config objects to helper function
            reassembled_obj, _, _ = hh.reassemble_patches_position_real(x, positions,
                                                                  data_config=self.data_config,
                                                                  model_config=self.model_config)

            #Extract patches - Pass config objects to helper function
            extracted_patch_objs = hh.extract_channels_from_region(reassembled_obj[:,None,:,:], positions,
                                                                   data_config=self.data_config,
                                                                   model_config=self.model_config)

        else:
            extracted_patch_objs = x

        #Apply probe illum
        illuminated_objs, _ = self.probe_illumination(extracted_patch_objs,
                                                    probe)

        #Pad and diffract
        pred_unscaled_diffraction, _ = hh.pad_and_diffract(illuminated_objs,
                                                    pad = False) # Assuming pad=False is intended
        
        #Inverse scaling
        pred_scaled_diffraction = self.scaler.inv_scale(pred_unscaled_diffraction, output_scale_factor)
        
        #Return unscaled product
        return pred_scaled_diffraction

#Loss functions
        
# class PoissonLoss(nn.Module):
#     """
#     Calculate Poisson loss for tensor of size (N,C,H,W)

#     See PoissonIntensityLayer for details
#     """
#     def __init__(self):
#         super(PoissonLoss, self).__init__()
#         self.poisson_layer = PoissonIntensityLayer()
#     def forward(self, pred, raw):
#         # self.poisson = PoissonIntensityLayer(pred)

#         loss_likelihood = self.poisson_layer(raw, pred)

#         return loss_likelihood
    
class PoissonLoss(nn.Module):
    def __init__(self):
        super(PoissonLoss, self).__init__()
    def forward(self, pred, raw):
        self.poisson = PoissonIntensityLayer(pred)
        loss_likelihood = self.poisson(raw)
        return loss_likelihood
    
class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        self.mae = nn.L1Loss(reduction = 'none')

    def forward(self, pred, raw):
        #Note: Prediction has not been squared yet, must be squared here
        loss_mae = self.mae(pred**2, raw)

        return loss_mae
    
class MeanDeviationLoss(nn.Module):
    """
    Calculates L1 loss of absolute mean deviation for each point.
    Does not take the mean (that is done in the neural net loss calc)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x_mean = torch.mean(x, dim=(2,3), keepdim = True)
        #Calculate absolute deviation, sum across (C,H,W) channels to return size B
        abs_dev = torch.sum(torch.abs(x - x_mean),dim=(1,2,3))

        return abs_dev
    
class TotalVariationLoss(nn.Module):
    """
    Calculates variation loss for each point. Punishes non-smooth shapes.
    Returns tensor of size (B)
    """
    def __init__(self):
        super().__init__()

    def forward(self,x):
        vert_diff_loss = torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]),
                                   dim = (1,2,3))
        horz_diff_loss = torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]),
                                   dim = (1,2,3))
        tv_loss =  vert_diff_loss + horz_diff_loss
        
        return tv_loss

    
#Scaling modules

class IntensityScalerModule:
    '''
    Scaler module that works with single experiment data and multi-experiment data.

    If single experiment data, ModelConfig will have an "intensity_scale" parameter that is determined
    during the dataloading process. This is to set up log_scale as a learnable parameter.

    If multi-experiment data, log_scale is no longer learnable since there are multiple different experiments
    and different log_scales to learn.
    '''
    def __init__(self, model_config: ModelConfig):
        super().__init__() # Need super().__init__() for nn.Module if parameters are used
        self.model_config = model_config
        #Define the scaler module (from helper.py) - Not needed if scale/inv_scale are methods
        #Setting log scale values
        if self.model_config.intensity_scale_trainable:
            log_scale_guess = np.log(self.model_config.intensity_scale)
            # Ensure log_scale is registered as a parameter if it's trainable
            self.log_scale = nn.Parameter(torch.tensor(float(log_scale_guess)),
                                          requires_grad=True)
        else:
            # If not trainable, register as a buffer or just keep as None
            # Registering as buffer allows it to be saved with state_dict but not trained
            # self.register_buffer('log_scale', None) # Or simply:
            self.log_scale = None

    #Standalone intensity scaling functions
    def scale(self, x, scale_factor):
        '''
        Goes from experimental input intensity to normalized amplitude scale.
        '''
        if self.log_scale:
            scale_factor = torch.exp(self.log_scale)
        ''

        return x * scale_factor

    def inv_scale(self, x, scale_factor):
        '''
        Undoes the scaling operation, goes from normalized amplitude -> output intensity

        '''
        if self.log_scale is not None: # Check if log_scale is trainable parameter
            scale_factor = torch.exp(self.log_scale)
        
        # Resize scale_factor to match x dimensions if needed.
        # Implemented 06/3/2025. Unsure why this is even here, removed it for now
        # if x.ndim == 4:
        #     scale_factor = scale_factor.squeeze(2)

        return x / scale_factor

#Full module with everything
class PtychoPINN(nn.Module):
    '''
    Full PtychoPINN module with all sub-modules.
    If in training, outputs loss and reconstruction
    If in inference, outputs object functions
    
    Note for forward call, because we're getting data from a memory-mapped tensor

    Inputs
    -------
    x: torch.Tensor (N, C, H, W)
    positions - Tensor input, comes from tensor['coords_relative']
    probe - Tensor input, comes from dataset/dataloader __get__ function (returns x, probe)

    '''
    def __init__(self, model_config: ModelConfig, data_config: DataConfig, training_config: TrainingConfig):
        super(PtychoPINN, self).__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config # Store training config

        self.n_filters_scale = self.model_config.n_filters_scale

        #Scaler - Pass config
        self.scaler = IntensityScalerModule(model_config)
        self.probe_scale = data_config.probe_scale

        #Autoencoder - Pass configs
        self.autoencoder = Autoencoder(model_config, data_config)
        self.combine_complex = CombineComplex()

        #Adding named modules for forward operation
        self.forward_model = ForwardModel(model_config, data_config)

    def forward(self, x, positions, probe, input_scale_factor, output_scale_factor):

        #Scaling down (normalizing to 1)
        x = self.scaler.scale(x, input_scale_factor)
        #Autoencoder result
        x_amp, x_phase = self.autoencoder(x)
        #Combine amp and phase
    
        x_combined = self.combine_complex(x_amp, x_phase)

        #Run through forward model. Unscaled diffraction pattern
        x_out = self.forward_model.forward(x_combined, positions,
                                           probe/self.probe_scale, output_scale_factor)
        
        

        return x_out, x_amp, x_phase
    
    def forward_predict(self, x, positions, probe, input_scale_factor):
        #Scaling
        x = self.scaler.scale(x, input_scale_factor)
        #Autoencoder result
        x_amp, x_phase = self.autoencoder(x)
        #Combine amp and phase
        x_combined = self.combine_complex(x_amp, x_phase)

        return x_combined
    
# Supervised model equivalent

class Ptycho_Supervised(nn.Module):
    '''
    PtychoPINN supervised version. Skips forward model as well as overlap constraint
    
    Note for forward call, because we're getting data from a memory-mapped tensor

    Inputs
    -------
    x: torch.Tensor (N, C, H, W)
    positions - Tensor input, comes from tensor['coords_relative']. Unused, dummy input kept for compatibility
    probe - Tensor input, comes from dataset/dataloader __get__ function (returns x, probe) . Unused, dummy input kept for compatibility

    '''
    def __init__(self, model_config: ModelConfig, data_config: DataConfig, training_config: TrainingConfig):
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config # Store training config

        self.n_filters_scale = self.model_config.n_filters_scale

        #Autoencoder - Pass configs
        self.autoencoder = Autoencoder(model_config, data_config)
        self.combine_complex = CombineComplex()

        #Scaler - Pass config
        self.scaler = IntensityScalerModule(model_config)

    def forward(self, x, positions, probe, input_scale_factor, output_scaling_factor):

        #Scaling
        x = self.scaler.scale(x, input_scale_factor)
        #Autoencoder result
        x_amp, x_phase = self.autoencoder(x)
        #Combine amp and phase
    
        x_combined = self.combine_complex(x_amp, x_phase)

        return x_combined, x_amp, x_phase
    
    def forward_predict(self, x, positions, probe, input_scale_factor):
        """
        Identical to forward specifically for Ptycho_Supervised.
        Kept to keep consistency with the forward_predict method in PtychoPINN.
        """
        #Scaling
        x = self.scaler.scale(x, input_scale_factor)
        #Autoencoder result
        x_amp, x_phase = self.autoencoder(x)
        #Combine amp and phase
        x_combined = self.combine_complex(x_amp, x_phase)

        return x_combined

#PtychoPINN Lightning Module
class PtychoPINN_Lightning(L.LightningModule):
    '''
    Lightning module equivalent of PtychoPINN module from ptycho_torch.model
    Enhanced with multi-stage training:
    Stage 1: RMS normalization only
    Stage 2: Weighted combination of RMS + physics-based loss
    Stage 3: Physics-based normalization only
    Stage 4: Fine-tune decoder only
    
    Requires import: from ptycho_torch.schedulers import MultiStageLRScheduler, AdaptiveLRScheduler
    '''
    def __init__(self, model_config: ModelConfig,
                       data_config: DataConfig,
                       training_config: TrainingConfig,
                       inference_config: InferenceConfig):
        super().__init__()
        self.n_filters_scale = model_config.n_filters_scale
        self.predict = False

        #Configs
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config
        self.inference_config = inference_config

        #Scaling module specifically for multi-scaling
        self.scaler = IntensityScalerModule(model_config)

        #Other params
        self.lr = training_config.learning_rate
        self.accum_steps = training_config.accum_steps
        self.gradient_clip_val = training_config.gradient_clip_val

        #DDP LR. Makes it so we have to manually update the model gradients.
        self.automatic_optimization = False

        #Model
        if model_config.mode == 'Unsupervised':
            self.model = PtychoPINN(model_config, data_config, training_config)
        elif model_config.mode == 'Supervised':
            self.model = Ptycho_Supervised(model_config, data_config, training_config)

        # Multi-stage training parameters with backwards compatibility
        # Default to current behavior: all epochs in stage 1 (RMS only)
        self.stage_1_epochs = getattr(training_config, 'stage_1_epochs', training_config.epochs)
        self.stage_2_epochs = getattr(training_config, 'stage_2_epochs', 0)
        self.stage_3_epochs = getattr(training_config, 'stage_3_epochs', 0)
        # Stage 4 is the existing fine-tuning (epochs_fine_tune)
        
        self.physics_weight_schedule = getattr(training_config, 'physics_weight_schedule', 'cosine')  # 'linear', 'cosine', 'exponential'
        
        # Track current stage and epoch for scheduling
        self.current_stage = 1
        self.stage_start_epoch = 0

        #Choose loss function and logging
        #Poisson Loss only works with Unsupervised
        if model_config.mode == 'Unsupervised' and model_config.loss_function == 'Poisson':
            self.Loss = PoissonLoss()
            self.loss_name = 'poisson_train'
            self.val_loss_name = 'poisson_val'
        #MAE works with both
        elif model_config.mode == 'Unsupervised' and model_config.loss_function == 'MAE':
            self.Loss = MAELoss()
            self.loss_name = 'mae_train'
            self.val_loss_name = 'mae_val'
        elif model_config.mode == 'Supervised' and model_config.loss_function == 'MAE':
            self.Loss = nn.L1Loss(reduction = 'none')
            self.loss_name = 'mae_train'
            self.val_loss_name = 'mae_val'
        

        #Include amplitude loss
        if model_config.amp_loss:
            if model_config.amp_loss == "Mean_Deviation":
                self.AmpLoss = MeanDeviationLoss()
            elif model_config.amp_loss == "Total_Variation":
                self.AmpLoss = TotalVariationLoss()
            self.loss_name += '_Amp'
            self.val_loss_name += '_Amp'

        if model_config.phase_loss:
            if model_config.phase_loss == "Mean_Deviation":
                self.PhaseLoss = MeanDeviationLoss()
            elif model_config.phase_loss == "Total_Variation":
                self.PhaseLoss = TotalVariationLoss()
            self.loss_name += '_Phase'
            self.val_loss_name += '_Phase'

        self.loss_name += '_loss'
        self.val_loss_name += '_loss'
    
    def get_current_stage_and_weight(self):
        """
        Determine current training stage and physics weight based on current epoch
        Handles cases where stage 2 or 3 have 0 epochs (backwards compatibility)
        """
        # Override for fine-tuning mode
        # This only occurs if fine-tuning mode happens after multi-stage, multi-normalization training
        if hasattr(self, '_fine_tuning_mode') and self._fine_tuning_mode and self.stage_2_epochs > 0:
            return 4, 1.0  # Stage 4 (fine-tuning), physics weight = 1.0
        
        current_epoch = self.current_epoch
        
        if current_epoch < self.stage_1_epochs:
            stage = 1
            physics_weight = 0.0
        elif self.stage_2_epochs > 0 and current_epoch < self.stage_1_epochs + self.stage_2_epochs:
            stage = 2
            # Calculate progress through stage 2
            stage_2_progress = (current_epoch - self.stage_1_epochs) / self.stage_2_epochs
            physics_weight = self._get_physics_weight(stage_2_progress)
        elif self.stage_3_epochs > 0 and current_epoch < self.stage_1_epochs + self.stage_2_epochs + self.stage_3_epochs:
            stage = 3
            physics_weight = 1.0
        else:
            # Handle cases where stages are skipped
            if self.stage_3_epochs > 0:
                stage = 3
                physics_weight = 1.0
            elif self.stage_2_epochs > 0:
                # If we're past stage 2 but no stage 3, stay at end of stage 2
                stage = 2
                physics_weight = 1.0
            else:
                # Only stage 1 exists (backwards compatibility)
                stage = 1
                physics_weight = 0.0
            
        return stage, physics_weight
    
    def _get_physics_weight(self, progress):
        """
        Calculate physics weight based on progress through stage 2 (0 to 1)
        """
        if self.physics_weight_schedule == 'linear':
            return progress
        elif self.physics_weight_schedule == 'cosine':
            # Smooth cosine transition
            return 0.5 * (1 - math.cos(math.pi * progress))
        
        elif self.physics_weight_schedule == 'exponential':
            # Exponential ramp-up
            return progress ** 2
        else:
            return progress  # Default to linear
    
    def forward(self, x, positions, probe, input_scale_factor, output_scale_factor):
        x_out = self.model(x, positions, probe, input_scale_factor, output_scale_factor)
        return x_out
    
    def forward_predict(self, x, positions, probe, input_scale_factor):
        #Turns padding off if we need to
        x_combined = self.model.forward_predict(x, positions, probe, input_scale_factor)
        return x_combined
    
    def compute_loss(self, batch):
        """
        Enhanced loss computation supporting multi-stage training
        """
        # Grab required data fields from TensorDict
        x = batch[0]['images']
        positions = batch[0]['coords_relative']
        probe = batch[1]
        rms_scale = batch[0]['rms_scaling_constant']  # RMS scaling
        physics_scale = batch[0]['physics_scaling_constant']
        scale = batch[2]
        # old_scaling = batch[2]
        
        # Get current stage and physics weight
        stage, physics_weight = self.get_current_stage_and_weight()
        rms_weight = 1.0 - physics_weight

        #Custom stuff (TEMPORARY)
        rms_weight = 1.0
        physics_weight = 0.0
        
        # Log current stage info
        self.log('training_stage', float(stage), on_step=True, on_epoch=True)
        self.log('physics_weight', physics_weight, on_step=True, on_epoch=True)
        
        #If supervised, also need to get the amp/phase labels
        if self.model_config.mode == 'Supervised':
            amp_label = batch[0]['label_amp']
            phase_label = batch[0]['label_phase']

        #Calc loss
        total_loss = 0.0
        
        output_scaling_factor = rms_weight * rms_scale + physics_weight * physics_scale
        # Perform forward pass up and scale
        pred, amp, phase = self(x, positions, probe,
                                            input_scale_factor = rms_scale,
                                            output_scale_factor = rms_scale,
                                            )
        
        #Normalization factor for loss output (just to keep it scaled down)
        intensity_norm_factor = torch.mean(x).detach() + 1e-8

        if self.model_config.mode == 'Unsupervised':
            total_loss += self.Loss(pred, x).mean()
            total_loss /= intensity_norm_factor

        elif self.model_config.mode == 'Supervised':
            #Compute loss for phase and amp
            amp_loss = self.Loss(amp, amp_label).sum()
            phase_loss = self.Loss(phase, phase_label).sum()
            #Add to total loss
            total_loss += 0.1*amp_loss + 4 * phase_loss
        
        # Add amplitude and phase regularization losses if specified
        # Use the appropriate amp/phase based on current stage
        if self.model_config.amp_loss:
            if physics_weight > 0.5:  # Use physics-based in later stages
                amp_reg_loss = self.AmpLoss(amp).mean()
            else:  # Use RMS-based in early stages
                amp_reg_loss = self.AmpLoss(amp).mean()
            total_loss += amp_reg_loss * self.model_config.amp_loss_coeff
            
        if self.model_config.phase_loss:
            if physics_weight > 0.5:  # Use physics-based in later stages
                phase_reg_loss = self.PhaseLoss(phase).mean()
            else:  # Use RMS-based in early stages
                phase_reg_loss = self.PhaseLoss(phase).mean()
            total_loss += phase_reg_loss * self.model_config.phase_loss_coeff

        return total_loss
    
    def training_step(self, batch, batch_idx):
        #Manual opt
        opt = self.optimizers()

        #Calc loss
        loss = self.compute_loss(batch)
        scaled_loss = loss / self.accum_steps

        self.manual_backward(scaled_loss)

        #Step every N batches
        if (batch_idx+1) % self.accum_steps == 0:
            #Clip gradients
            if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    parameters = self.parameters(),
                    max_norm = self.gradient_clip_val,
                    norm_type = 2.0 #L2 norm, default
                )

            opt.step()
            opt.zero_grad()

        #Logging
        self.log(self.loss_name, loss, on_epoch = True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step - computes validation loss without gradient updates
        Uses the same multi-stage approach as training
        """
        val_loss = self.compute_loss(batch)
        
        # Log validation loss
        self.log(self.val_loss_name, val_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return val_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr = self.lr)
        
        result = {"optimizer": optimizer}
        
        # Configure scheduler based on training type
        if self.stage_2_epochs > 0 or self.stage_3_epochs > 0:
            # Multi-stage training: use specialized scheduler
            if self.training_config.scheduler == 'MultiStage':
                scheduler = MultiStageLRScheduler(
                    optimizer,
                    stage_1_epochs=self.stage_1_epochs,
                    stage_2_epochs=self.stage_2_epochs,
                    stage_3_epochs=self.stage_3_epochs,
                    stage_3_lr_factor=self.training_config.stage_3_lr_factor
                )
                result['lr_scheduler'] = {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            elif self.training_config.scheduler == 'Adaptive':
                scheduler = AdaptiveLRScheduler(
                    optimizer,
                    lightning_module=self,
                    base_stage_2_lr_factor=self.training_config.stage_2_lr_factor,
                    min_stage_2_lr_factor=self.training_config.stage_3_lr_factor
                )
                result['lr_scheduler'] = {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            elif self.training_config.scheduler == 'Exponential':
                # Fallback to exponential for multi-stage
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
                result['lr_scheduler'] = scheduler
        else:
            # Single-stage training: use traditional schedulers
            if self.training_config.scheduler == 'Exponential':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
                result['lr_scheduler'] = scheduler

        return result
    
    def freeze_encoder(self):
        """
        Freezes all parameters in encoder for fine-tuning step.
        Also sets the model to use only physics-based loss during fine-tuning.
        """
        encoder = self.model.autoencoder.encoder
        for param in encoder.parameters():
            param.requires_grad = False
        
        # Fix: Force physics-only mode for fine-tuning
        self._fine_tuning_mode = True
        
        print("Encoder layers frozen for fine-tuning. Only decoder layers will be updated")
        print("Fine-tuning will use physics-based normalization only.")
        
    def on_train_epoch_start(self):
        """
        Called at the start of each training epoch
        """
        stage, _ = self.get_current_stage_and_weight()
        
        # Log stage transitions
        if hasattr(self, '_last_stage') and self._last_stage != stage:
            print(f"Transitioning from Stage {self._last_stage} to Stage {stage}")
            
        self._last_stage = stage
        
        # Log current learning rate for monitoring
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_epoch=True)


