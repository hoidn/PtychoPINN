"""
PyTorch implementation of the PtychoPINN architecture and physics layers.

This module mirrors the design described in <doc-ref type="guide">docs/architecture.md</doc-ref>
and the TensorFlow reference implementation in `ptycho/model.py`, but rewritten with PyTorch
building blocks and helper functions from `ptycho_torch.helper`. The intent is to provide a
drop-in alternative that can be trained with Lightning while reusing the same data contracts.

Core components
---------------
Autoencoder:
    Encoder/decoder stack that predicts amplitude and phase patches. Filter dimensions adapt
    to `DataConfig().get('N')` and `ModelConfig().get('object.big')` in the same way as the
    TensorFlow U-Net.

ForwardModel:
    Applies probe illumination, diffraction simulation, and intensity scaling to enforce the
    ptychographic physics constraints. Helper functions (`ptycho_torch.helper`) are wrapped so
    that the Lightning training loop can trace them.

CombineComplex:
    Utility that reconstructs complex-valued tensors from the amplitude/phase branches for
    downstream FFT operations.

Tensor contracts
----------------
Inputs follow the conventions established in <doc-ref type="contract">specs/data_contracts.md</doc-ref>.
The training dataloader yields TensorDict tuples `(images, coords_relative, probe, scale_factor)` with
the shapes:
    - images: `(batch, channels, N, N)` real amplitudes.
    - coords_relative: `(batch, channels, 1, 2)` scan offsets in pixel units.
    - probe: `(channels?, N, N)` complex probe patches (broadcast across the batch).
    - scale_factor: `(batch,)` intensity normalization factors.

Outputs:
    - Autoencoder returns amplitude and phase tensors with the same `(batch, channels, N, N)` layout.
    - ForwardModel returns amplitude predictions in experimental units.
    - `PtychoPINN` returns either complex reconstructions (in inference mode) or physics-consistent
      diffraction amplitudes for loss computation (training mode).

Configuration
-------------
Unlike the TensorFlow package, which uses dataclasses (`ptycho.config`), the PyTorch path relies on
singleton settings (`ModelConfig`, `TrainingConfig`, `DataConfig`). Each class must be seeded via
`set_settings(...)` before instantiating modules in this file. See the docstring in
`ptycho_torch/config_params.py` for usage patterns.

Entry points
------------
Exported symbols are consumed by the Lightning wrapper in `ptycho_torch/train.py`:
    - `Autoencoder` — direct use in custom experiments.
    - `ForwardModel` — physics-only inference.
    - `PtychoPINN` — end-to-end network used by the TensorDict dataloader.
    - `PoissonLoss` and `MAELoss` — loss functions matching TensorFlow behaviour.

Differences vs. TensorFlow
--------------------------
The helper API mirrors TensorFlow semantics, but a few behavioural differences exist:
    - Default dtype is `torch.float32`; complex math is handled via explicit amplitude/phase branches.
    - Probe reassembly is optional (`ModelConfig().get('probe.big')`).
    - Poisson loss is implemented with `torch.distributions` to avoid custom kernels.

Use this module as the authoritative reference for PyTorch-side architecture decisions and keep it
in sync with the design notes in the high-level docs when introducing new physics layers or data
formats.
"""

#Torch
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist

#Other math
import numpy as np
import math

#Helper
from ptycho_torch.config_params import ModelConfig, TrainingConfig, DataConfig
import ptycho_torch.helper as hh

#Ensuring 64float b/c of complex numbers
torch.set_default_dtype(torch.float32)

device = TrainingConfig().get('device')

#Helping modules

class Tanh_custom_act(nn.Module):
    '''
    Custom tanh activation module used in:
        Decoder_phase
    '''
    def forward(self, x):
        return math.pi * torch.tanh(x)

#Conv blocks
#Conv blocks
class ConvBaseBlock(nn.Module):
    '''
    Convolutional base block for Pooling and Upscaling

    If padding = same, padding is half of kernel size
    '''
    def __init__(self, in_channels, out_channels,
                 w1 = 3, w2 = 3,
                 padding = 'same',
                 activation = 'relu'):
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
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.activation(self.conv2(x)) if self.activation else F.relu(self.conv2(x))
        x = self._pool_or_up(x)

        return x
    
    def _pool_or_up(self, x):
        raise NotImplementedError("Subclasses must implement pool or up")
    
class ConvPoolBlock(ConvBaseBlock):

    def __init__(self, in_channels, out_channels,
                 w1 = 3, w2 = 3, p1 = 2, p2 = 2,
                 padding = 'same'):
        super(ConvPoolBlock, self).__init__(in_channels, out_channels,
                                            w1=w1, w2=w2, padding=padding)
        #Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=(p1, p2),
                                 padding = 0)
        
    def _pool_or_up(self, x):
        return self.pool(x)
    
class ConvUpBlock(ConvBaseBlock):

    def __init__(self, in_channels, out_channels,
                 w1 = 3, w2 = 3, p1 = 2, p2 = 2,
                 padding = 'same'):
        
        super(ConvUpBlock, self).__init__(in_channels, out_channels,
                                            w1=w1, w2=w2, padding=padding)
        padding_size = w1 // 2 if padding == 'same' else 0
        #NN layers
        self.up = nn.Upsample(scale_factor = (p1, p2),
                              mode = 'nearest')

    def _pool_or_up(self, x):
        return self.up(x)

#Encoder

class Encoder(nn.Module):
    def __init__(self, n_filters_scale):
        super(Encoder, self).__init__()

        self.N = DataConfig().get('N')
        starting_coeff = 64 / (self.N / 32)
        if ModelConfig().get('object.big'):
            starting_filter_n = DataConfig().get('grid_size')[0] * DataConfig().get('grid_size')[1]
        else:
            starting_filter_n = 1
        self.filters = [starting_filter_n]
        #Starting output channels is 64. Last output size will always be n_filters_scale * 128. 
        if self.N == 64:
            self.filters = self.filters + [n_filters_scale * 32, n_filters_scale * 64, n_filters_scale * 128]
        elif self.N == 128:
            self.filters = self.filters + [n_filters_scale * 16, n_filters_scale * 32, n_filters_scale * 64, n_filters_scale * 128]
        elif self.N == 256:
            self.filters = self.filters + [n_filters_scale * 8, n_filters_scale * 16, n_filters_scale * 32, n_filters_scale * 64, n_filters_scale * 128]



        if starting_coeff < 3 or starting_coeff > 64:
            raise ValueError(f"Unsupported input size: {self.N}")
        
        self.blocks = nn.ModuleList([ConvPoolBlock(in_channels = self.filters[i-1],
                                                   out_channels = self.filters[i])
                                    for i in range(1,len(self.filters))])
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x
    
#Decoders

class Decoder_filters(nn.Module):
    '''
    Base decoder class handling dynamic channel sizing in self.filters
    '''
    def __init__(self, n_filters_scale):
        super(Decoder_filters, self).__init__()
        self.N = DataConfig().get('N')

        #Calculate number of channels for upscaling
        #Start from self.N and divide by 2 until 32 for each layer
        #E.g. 
        #N == 64: [n_filters_scale * 64, n_filters_scale * 32]
        #N == 128: [n_filters_scale * 128, n_filters_scale * 64, n_filters_scale * 32]
        self.filters = [n_filters_scale * 128]

        if self.N == 64:
            self.filters = self.filters + [n_filters_scale * 64, n_filters_scale * 32]
        elif self.N == 128:
            self.filters = self.filters + [n_filters_scale * 128, n_filters_scale * 64, n_filters_scale * 32]
        elif self.N == 256:
            self.filters = self.filters + [n_filters_scale * 256, n_filters_scale * 128, n_filters_scale * 64, n_filters_scale * 32]

        if self.N < 64:
            raise ValueError(f"Unsupported input size: {self.N}")
        
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward")
    
class Decoder_base(Decoder_filters):
    def __init__(self, n_filters_scale):
        super(Decoder_base, self).__init__(n_filters_scale)
        #Layers
        self.blocks = nn.ModuleList([ConvUpBlock(in_channels = self.filters[i-1],
                                                   out_channels = self.filters[i]) 
                                    for i in range(1,len(self.filters))])
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x

class Decoder_last(nn.Module):
    '''
    Decoder to get both amplitude and phase, combining to create full object function
    '''
    def __init__(self, in_channels, out_channels, n_filters_scale,
                  activation = torch.sigmoid, name = ''):
        super(Decoder_last, self).__init__()
        #Grab parameters
        self.N = DataConfig().get('N')
        self.gridsize = DataConfig().get('grid_size')

        #Channel splitting
        if ModelConfig().get('object.big'):
            self.c_outer = self.gridsize[0] * self.gridsize[1]
        else:
            self.c_outer = 1

        #Layers
        self.conv1 =  nn.Conv2d(in_channels = in_channels - self.c_outer,
                                out_channels = out_channels,
                                kernel_size = (3, 3),
                                padding = 3//2)
        
        #conv_up_block and conv2 are separate to conv1
        self.conv_up_block = ConvUpBlock(self.c_outer, n_filters_scale * 32)
        self.conv2 =  nn.Conv2d(in_channels = n_filters_scale * 32,
                                out_channels = out_channels,
                                kernel_size = (3, 3),
                                padding = 3//2)

        #Additional
        self.activation = activation
        self.padding = nn.ConstantPad2d((self.N // 4, self.N // 4,
                                         self.N // 4, self.N //4), 0)

        
    def forward(self,x):
        x1 = self.conv1(x[:, :-self.c_outer, :, :])
        x1 = self.activation(x1)
        x1 = self.padding(x1)

        if not ModelConfig().get('probe.big'):
            return x1
        
        x2 = self.conv_up_block(x[:, -self.c_outer:, :, :])
        x2 = self.conv2(x2)
        x2 = F.silu(x2) #Same as swish

        outputs = x1 + x2

        return outputs

class Decoder_phase(Decoder_base):
    def __init__(self, n_filters_scale):
        super(Decoder_phase, self).__init__(n_filters_scale)
        grid_size = DataConfig().get('grid_size')
        if ModelConfig().get('object.big'):
            num_channels = grid_size[0] * grid_size[1]
        else:
            num_channels = 1
        #Nn layers

        #Custom nn layers with specific identifiable names
        self.add_module('phase_activation', Tanh_custom_act())
        self.add_module('phase', Decoder_last(n_filters_scale * 32, num_channels, n_filters_scale,
                                         activation = self.phase_activation))
            
    def forward(self, x):
        #Apply upscale block layers
        for block in self.blocks:
            x = block(x)

        #Apply final layer
        outputs = self.phase(x)

        return outputs

class Decoder_amp(Decoder_base):
    def __init__(self, n_filters_scale):
        super(Decoder_amp, self).__init__(n_filters_scale)

        #Custom nn layers with specific identifiable names
        self.add_module('amp_activation', Tanh_custom_act())
        self.add_module('amp', Decoder_last(n_filters_scale * 32, 1, n_filters_scale,
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
    def __init__(self, n_filters_scale):
        super(Autoencoder, self).__init__()
        #Encoder
        self.encoder = Encoder(n_filters_scale)
        #Decoders (Amplitude/Phase)
        self.decoder_amp = Decoder_amp(n_filters_scale)
        self.decoder_phase = Decoder_phase(n_filters_scale)

    def forward(self, x):
        #Encoder
        x = self.encoder(x)
        #Decoders
        x_amp = self.decoder_amp(x)
        x_phase = self.decoder_phase(x)

        return x_amp, x_phase

#Probe modules
class ProbeIllumination(nn.Module):
    '''
    Probe illumination done using hadamard product of object tensor and 2D probe function.
    2D probe function should be supplised by the dataloader
    '''
    def __init__(self):
        super(ProbeIllumination, self).__init__()
        self.N = DataConfig().get('N')
        self.mask = ModelConfig().get('probe_mask')
    
    def forward(self, x, probe):
        
        #Check if probe mask exists
        #If not, probe mask is just a ones matrix
        #If mask exists, save mask is class attribute
        if not self.mask:
            probe_mask = torch.ones((self.N, self.N)).to(x.device)
        else:
            probe_mask = ModelConfig().get('probe_mask')

        return x * probe * probe_mask, probe * probe_mask


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
    def __init__(self):
        super(ForwardModel, self).__init__()

        #Configuration 
        self.n_filters_scale = ModelConfig().get('n_filters_scale')
        self.N = DataConfig().get('N')
        self.gridsize = DataConfig().get('grid_size')
        self.offset = ModelConfig().get('offset')
        self.object_big = ModelConfig().get('object.big')

        #Patch operations
        #Lambdalayer here doesn't work for lightning module
        self.reassemble_patches = LambdaLayer(hh.reassemble_patches_position_real)

        self.pad_patches = LambdaLayer(hh.pad_patches)

        self.trim_reconstruction = LambdaLayer(hh.trim_reconstruction)

        self.extract_patches = LambdaLayer(hh.extract_channels_from_region)

        #Probe Illumination
        self.probe_illumination = ProbeIllumination()

        #Pad/diffract
        
        self.pad_and_diffract = LambdaLayer(hh.pad_and_diffract)

        #Intensity scaling
        self.scaler = IntensityScalerModule()

        
    def forward(self, x, positions, probe, scale_factor):
        #Reassemble patches
        #Object_big: All patches are together in a solution region
        if self.object_big:
            reassembled_obj = hh.reassemble_patches_position_real(x, positions)
            #Extract patches
            extracted_patch_objs = hh.extract_channels_from_region(reassembled_obj[:,None,:,:], positions)
        else:
        #Single channel, no patch overlap
            #NOTE for albert: Check transformation math
            #Temporarily removed because it seemed superfluous
            # reassembled_obj = self.pad_patches(
            #     torch.flatten(x, start_dim = 0, end_dim = 1),
            #     padded_size = hh.get_padded_size()
            # )
            extracted_patch_objs = x


        #Apply probe illum
        illuminated_objs, _ = self.probe_illumination(extracted_patch_objs,
                                                    probe)
        #Pad and diffract
        pred_diffraction, _ = hh.pad_and_diffract(illuminated_objs,
                                                    pad = False)
        #Inverse scaling
        pred_amp_scaled = self.scaler.inv_scale(pred_diffraction, scale_factor)

        return pred_amp_scaled
        
        # #Performing inference
        # else:
        #     return extracted_patch_objs

#Loss functions
        
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
    
#Scaling modules

# # Example usage
# # Assuming cfg is a dictionary with the key 'intensity_scale' and appropriate value
# cfg = {'intensity_scale': 1.0}
# params = lambda: {'intensity_scale.trainable': True}

# # Initialize the module
# scaler_module = IntensityScalerModule(cfg, params)

# # Apply scaling and inverse scaling
# scaled_tensor = scaler(input_tensor)
# inv_scaled_tensor = inv_scaler(input_tensor)

class IntensityScalerModule:
    '''
    Scaler module that works with single experiment data and multi-experiment data.

    If single experiment data, ModelConfig will have an "intensity_scale" parameter that is determined
    during the dataloading process. This is to set up log_scale as a learnable parameter.

    If multi-experiment data, log_scale is no longer learnable since there are multiple different experiments
    and different log_scales to learn.
    '''
    def __init__(self):
        #Setting log scale values
        if ModelConfig().get('intensity_scale_trainable'):
            log_scale_guess = np.log(ModelConfig().get('intensity_scale'))
            self.log_scale = nn.Parameter(torch.tensor(float(log_scale_guess)),
                                      requires_grad = ModelConfig().get('intensity_scale_trainable'))
        else:
            self.log_scale = None
    
    #Intensity scaler as class
    class IntensityScaler(nn.Module):
        '''
        Scales intensity with log scale factor. Supports inverse and regular scaling
        '''
        def __init__(self, log_scale, inv = False):
            super(IntensityScalerModule.IntensityScaler, self).__init__()
            self.scale_factor = torch.exp(log_scale)
            if inv:
                self.scale_factor = 1 / self.scale_factor

        def forward(self, x):
            return x / self.scale_factor
        
    #Standalone intensity scaling functions
    def scale(self, x, scale_factor):
        if self.log_scale:
            log_scale = torch.exp(self.log_scale)
        else:
            log_scale = scale_factor
        return x * log_scale

    def inv_scale(self, x, scale_factor):
        '''
        Undoes the scaling operation, goes from normalized space -> experimental space
        '''
        if self.log_scale:
            log_scale = torch.exp(self.log_scale)
        else:
            log_scale = scale_factor
        return x / log_scale

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
    def __init__(self):
        super(PtychoPINN, self).__init__()
        self.n_filters_scale = ModelConfig().get('n_filters_scale')
        self.device = TrainingConfig().get('device')
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
        if self.predict:
            return x_combined
        else:
            #Run through forward model
            scale_factor = scale_factor.view(-1, 1, 1, 1)
            x_out = self.forward_model(x_combined, positions, probe, scale_factor)
            #Get loss
            if self.training:
                return self.Loss(x_out, x)

            return x_out
            
