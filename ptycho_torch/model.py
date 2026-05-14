#Standard libs
import logging

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
import copy

logger = logging.getLogger(__name__)

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
        
    def forward_conv(self, x):
        """Apply convolution and attention, but not pooling (for skip connections)."""
        x = super().forward(x)
        x = self.attention(x)
        return x

    def forward_pool(self, x):
        """Apply pooling only."""
        return self.pool(x)

    def forward(self, x):
        x = self.forward_conv(x)
        x = self.forward_pool(x)
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
        skips = []
        for i, block in enumerate(self.blocks):
            x = block.forward_conv(x)
            if i < len(self.blocks) - 1:
                skips.append(x)
            x = block.forward_pool(x)
        return x, skips
    
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
                print(f"Decoder Block {i}: Adding ECALayer with {out_ch} channels.")
                self.attention_blocks.append(ECALayer(channel=out_ch))
            elif self.use_spatial:
                print(f"Decoder Block {i}: Adding BasicSpatialAttention with kernel {self.spatial_kernel}.")
                self.attention_blocks.append(BasicSpatialAttention(kernel_size=self.spatial_kernel))
            elif self.use_cbam:
                 print(f"Decoder Block {i}: Adding CBAM with {out_ch} channels.")
                 self.attention_blocks.append(CBAM(gate_channels=out_ch))
            else:
                print(f"Decoder Block {i}: No attention module added.")
                self.attention_blocks.append(nn.Identity())

        # UNet skip-connection merge blocks
        self.merge_blocks = nn.ModuleList()
        for i in range(len(self.blocks)):
            decoder_ch = self.filters[i+1]
            encoder_ch = self._get_encoder_channels(i, model_config, data_config)
            self.merge_blocks.append(nn.Sequential(
                nn.Conv2d(decoder_ch + encoder_ch, decoder_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(decoder_ch, decoder_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))

    def _get_encoder_channels(self, decoder_level, model_config, data_config):
        N = data_config.N
        n_filters_scale = model_config.n_filters_scale
        encoder_filters = [model_config.C_model if model_config.object_big else 1]

        if N == 64:
            encoder_filters += [n_filters_scale * 32, n_filters_scale * 64, n_filters_scale * 128]
        elif N == 128:
            encoder_filters += [n_filters_scale * 16, n_filters_scale * 32,
                               n_filters_scale * 64, n_filters_scale * 128]
        elif N == 256:
            encoder_filters += [n_filters_scale * 8, n_filters_scale * 16,
                               n_filters_scale * 32, n_filters_scale * 64,
                               n_filters_scale * 128]

        encoder_idx = len(encoder_filters) - 2 - decoder_level
        if 0 <= encoder_idx < len(encoder_filters):
            return encoder_filters[encoder_idx]
        else:
            return self.filters[decoder_level+1]

    def forward(self, x, skips=None):
        for i, (up_block, merge_block) in enumerate(zip(self.blocks, self.merge_blocks)):
            x = up_block(x)

            if skips is not None and i < len(skips):
                skip = skips[-(i+1)]
                if skip.shape[2:] != x.shape[2:]:
                    skip = F.interpolate(skip, size=x.shape[2:],
                                         mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)

            x = merge_block(x)
            x = self.attention_blocks[i](x)

        return x

class Decoder_last(nn.Module):
    '''
    Base class for the final decoder stage. Handles conditional BatchNorm.
    '''
    def __init__(self, model_config: ModelConfig, data_config: DataConfig,
                 in_channels, out_channels,
                 activation = torch.sigmoid, name = '', batch_norm=False,
                 combined = False):
        super(Decoder_last, self).__init__()
        self.n_filters_scale = model_config.n_filters_scale
        self.combined = combined

        #Grab parameters
        self.N = data_config.N
        self.gridsize = data_config.grid_size

        #Dynamically calculate number of outer channels
        c_outer_fraction = getattr(model_config,'decoder_last_c_outer_fraction', 0.25)
        c_outer_fraction = max(0.0, min(0.5, c_outer_fraction))
        self.c_outer = max(1, int(in_channels * c_outer_fraction))

        #Layers
        self.conv1 =  nn.Conv2d(in_channels = in_channels - self.c_outer,
                                out_channels = out_channels,
                                kernel_size = (3, 3),
                                padding = 3//2)

        self.conv_up_block = ConvUpBlock(self.c_outer, self.n_filters_scale * 32,
                                         batch_norm = batch_norm)

        self.conv2 =  nn.Conv2d(in_channels = self.n_filters_scale * 32,
                                out_channels = out_channels,
                                kernel_size = (3, 3),
                                padding = 3//2)

        # BatchNorm layers (conditional)
        self.batch_norm = batch_norm
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else None

        #Additional
        self.activation = activation
        self.padding = nn.ConstantPad2d((self.N // 4, self.N // 4,
                                         self.N // 4, self.N //4), 0)

    def forward(self, x):
        # Path 1: Main features
        x1_in = x[:, :-self.c_outer, :, :]
        x1 = self.conv1(x1_in)
        if self.batch_norm and self.bn1:
            x1 = self.bn1(x1)
        if not self.combined:
            x1 = self.activation(x1)
        x1 = self.padding(x1)

        # Path 2: Detail/Upsample features
        x2_in = x[:, -self.c_outer:, :, :]
        x2 = self.conv_up_block(x2_in)
        x2 = self.conv2(x2)
        if self.batch_norm and self.bn2:
            x2 = self.bn2(x2)

        if not self.combined:
            x2 = F.silu(x2)

        # Center-crop x2 to match x1 spatial dims when they differ
        if x2.shape[2:] != x1.shape[2:]:
            dh = (x2.shape[2] - x1.shape[2]) // 2
            dw = (x2.shape[3] - x1.shape[3]) // 2
            x2 = x2[:, :, dh:dh + x1.shape[2], dw:dw + x1.shape[3]]

        outputs = x1 + x2

        if self.combined:
            outputs = self.activation(outputs)

        return outputs


class Decoder_last_Amp(Decoder_last):
    '''Final decoder stage for Amplitude/Real. batch_norm=False.'''
    def __init__(self, model_config: ModelConfig, data_config: DataConfig,
                 in_channels, out_channels,
                 activation = torch.sigmoid, name = ''):
        polar = getattr(model_config, 'object_representation', 'rectangular') == 'polar'
        if not polar:
            activation = lambda x: torch.tanh(x)
        super(Decoder_last_Amp, self).__init__(model_config, data_config, in_channels, out_channels,
                                               activation=activation, name=name, batch_norm=False,
                                               combined=not polar)

class Decoder_last_Phase(Decoder_last):
    '''Final decoder stage for Phase/Imaginary. batch_norm from config.'''
    def __init__(self, model_config: ModelConfig, data_config: DataConfig,
                 in_channels, out_channels,
                 activation = torch.sigmoid, name = ''):
        polar = getattr(model_config, 'object_representation', 'rectangular') == 'polar'
        if not polar:
            activation = lambda x: torch.tanh(x)
        super(Decoder_last_Phase, self).__init__(model_config, data_config, in_channels, out_channels,
                                                 activation=activation, name=name,
                                                 batch_norm=model_config.batch_norm,
                                                 combined=not polar)

class Decoder_phase(Decoder_base):
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super(Decoder_phase, self).__init__(model_config, data_config, batch_norm=model_config.batch_norm)
        self.model_config = model_config
        self.data_config = data_config

        if self.model_config.object_big:
            num_channels = model_config.C_model
        else:
            num_channels = 1

        self.add_module('phase_activation', Tanh_custom_act())
        self.add_module('phase', Decoder_last_Phase(model_config, data_config,
                                                    self.n_filters_scale * 32, num_channels,
                                                    activation = self.phase_activation))

    def forward(self, x, skips=None):
        x = super().forward(x, skips)
        outputs = self.phase(x)
        return outputs

class Decoder_amp(Decoder_base):
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super(Decoder_amp, self).__init__(model_config, data_config, batch_norm=False)
        self.model_config = model_config
        self.data_config = data_config

        num_channels = copy.deepcopy(model_config.decoder_last_amp_channels)
        assert num_channels in [1, model_config.C_model]

        self.add_module('amp_activation', Amplitude_activation(model_config))
        self.add_module('amp', Decoder_last_Amp(model_config, data_config,
                                                self.n_filters_scale * 32, num_channels,
                                                activation = self.amp_activation))

    def forward(self, x, skips=None):
        x = super().forward(x, skips)
        outputs = self.amp(x)
        return outputs
    
# Shared decoder components

class FeatureRefinementBlock(nn.Module):
    """
    Residual bottleneck expansion block. Temporarily expands channel dimensionality
    to allow the network to disentangle real-vs-imaginary feature subsets before the
    output split in the shared decoder.
    """
    def __init__(self, channels, expansion_factor=2):
        super().__init__()
        expanded = channels * expansion_factor
        self.block = nn.Sequential(
            nn.Conv2d(channels, expanded, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(expanded, expanded, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(expanded, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return F.relu(x + self.block(x), inplace=False)


class Decoder_shared(Decoder_base):
    """
    Shared decoder trunk with per-level refinement blocks.

    Rectangular mode: single head outputs [B, 2*C_out, N, N], split externally.
    Polar mode: two separate heads return (amp, phase) tuple.
    """
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__(model_config, data_config, batch_norm=model_config.batch_norm)

        C_out = model_config.decoder_last_amp_channels if model_config.object_big else 1
        C_phase = model_config.C_model if model_config.object_big else 1
        n_levels = len(self.blocks)
        polar = getattr(model_config, 'object_representation', 'rectangular') == 'polar'
        self._split_heads = polar

        self.refinement_blocks = nn.ModuleList()
        for i in range(n_levels):
            ch = self.filters[i + 1]
            expansion = 2 if i == n_levels - 1 else 1
            self.refinement_blocks.append(
                FeatureRefinementBlock(ch, expansion_factor=expansion)
            )

        base_ch = self.filters[-1]
        self.eca = ECALayer(channel=base_ch)

        if polar:
            self.add_module('amp_activation', Amplitude_activation(model_config))
            self.head_real = Decoder_last_Amp(model_config, data_config,
                                              base_ch, C_out,
                                              activation=self.amp_activation)
            self.add_module('phase_activation', Tanh_custom_act())
            self.head_imag = Decoder_last_Phase(model_config, data_config,
                                                base_ch, C_phase,
                                                activation=self.phase_activation)
        else:
            activation = lambda x: 1.2 * torch.tanh(x)
            self.head = Decoder_last(model_config, data_config,
                                     in_channels=base_ch,
                                     out_channels=C_out * 2,
                                     activation=activation,
                                     batch_norm=model_config.batch_norm,
                                     combined=True)

    def forward(self, x, skips=None):
        for i, (up_block, merge_block) in enumerate(zip(self.blocks, self.merge_blocks)):
            x = up_block(x)

            if skips is not None and i < len(skips):
                skip = skips[-(i+1)]
                if skip.shape[2:] != x.shape[2:]:
                    skip = F.interpolate(skip, size=x.shape[2:],
                                         mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)

            x = merge_block(x)
            x = self.attention_blocks[i](x)
            x = self.refinement_blocks[i](x)

        x = self.eca(x)

        if self._split_heads:
            return self.head_real(x), self.head_imag(x)
        return self.head(x)


#Autoencoder

class Autoencoder(nn.Module):
    def __init__(self, model_config: ModelConfig, data_config: DataConfig,
                 shared_decoder: bool = False):
        super(Autoencoder, self).__init__()
        self.model_config = model_config
        self.data_config = data_config
        self._shared = shared_decoder
        self.batch_norm = self.model_config.batch_norm
        self.use_cbam = self.model_config.cbam_bottleneck

        self.encoder = Encoder(model_config, data_config)

        if self.use_cbam:
            bottleneck_channels = self.encoder.filters[-1]
            self.bottleneck_cbam = CBAM(gate_channels=bottleneck_channels)
        else:
            self.bottleneck_cbam = nn.Identity()

        if shared_decoder:
            self.decoder = Decoder_shared(model_config, data_config)
        else:
            self.decoder_amp = Decoder_amp(model_config, data_config)
            self.decoder_phase = Decoder_phase(model_config, data_config)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck_cbam(x)

        if getattr(self, '_shared', False):
            result = self.decoder(x, skips)
            if isinstance(result, tuple):
                x_real, x_imag = result
            else:
                C = result.shape[1] // 2
                x_real = result[:, :C, :, :]
                x_imag = result[:, C:, :, :]
        else:
            x_real = self.decoder_amp(x, skips)
            x_imag = self.decoder_phase(x, skips)

        return x_real, x_imag


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
    Applies poisson intensity scaling using torch.distributions.
    Calculates the negative log likelihood of observing the raw data given the predicted intensities.

    When pred_is_amplitude=True, squares the input to convert amplitude → intensity
    before constructing the Poisson rate parameter.
    '''
    def __init__(self, pred, pred_is_amplitude=False):
        super(PoissonIntensityLayer, self).__init__()
        Lambda = pred ** 2 if pred_is_amplitude else pred
        self.poisson_dist = dist.Independent(dist.Poisson(Lambda), 3)

    def forward(self, x):
        return -self.poisson_dist.log_prob(x)
    
class ForwardModel(nn.Module):
    '''
    Forward model receiving complex object prediction, and applies physics-informed real space overlap
    constraints to the solution space. Uses RectangularScaledDiffraction for analytically-derived
    intensity scaling with probe-weighted patch reassembly.

    Inputs
    ------
    x: torch.Tensor (N, C, H, W), dtype = complex64
    positions: torch.Tensor (N, C, 1, 2), dtype = float32
    probe: torch.Tensor, dtype = complex64
    '''
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super(ForwardModel, self).__init__()
        self.model_config = model_config
        self.data_config = data_config

        self.n_filters_scale = self.model_config.n_filters_scale
        self.N = self.data_config.N
        self.gridsize = self.data_config.grid_size
        self.offset = self.model_config.offset
        self.object_big = self.model_config.object_big

        self.reassemble_patches = LambdaLayer(hh.reassemble_patches_position_real_probe)
        self.pad_patches = LambdaLayer(hh.pad_patches)
        self.trim_reconstruction = LambdaLayer(hh.trim_reconstruction)
        self.extract_patches = LambdaLayer(hh.extract_channels_from_region)
        self.pad_and_diffract = LambdaLayer(hh.pad_and_diffract)

        self.scaler = IntensityScalerModule(model_config)
        self.rect_scaler = RectangularScaledDiffraction(model_config)

    def forward(self, x, I_measured,
                positions, probe, output_scale_factor,
                experiment_ids=None, fine_tune=False):

        if self.object_big:
            reassembled_obj, _, _ = hh.reassemble_patches_position_real_probe(
                x, positions,
                data_config=self.data_config,
                model_config=self.model_config,
                probe=probe,
                use_probe_weights=True)

            extracted_patch_objs = hh.extract_channels_from_region(
                reassembled_obj[:,None,:,:], positions,
                data_config=self.data_config,
                model_config=self.model_config)
        else:
            extracted_patch_objs = x

        pred_scaled_intensity = self.rect_scaler(
            x=extracted_patch_objs,
            I_raw=I_measured,
            probe=probe,
            scale=output_scale_factor,
            experiment_ids=experiment_ids,
            autograd=True)

        return pred_scaled_intensity


class RectangularScaledDiffraction(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.s1 = nn.Parameter(torch.ones(model_config.num_datasets))
        self.s2 = nn.Parameter(torch.ones(model_config.num_datasets))

    def forward(self,
                x: torch.Tensor,
                I_raw: torch.Tensor,
                probe: torch.Tensor,
                scale: torch.Tensor,
                experiment_ids: torch.Tensor,
                autograd: bool = True):
        x = x.unsqueeze(dim=2)
        x_a, x_b = x.real, x.imag

        if autograd:
            s1 = self.s1[experiment_ids].view(-1,1,1,1,1)
            s2 = self.s2[experiment_ids].view(-1,1,1,1,1)

            scale = scale.unsqueeze(dim=2)

            exit_wave = scale * (s1 * (probe * x_a) + 1j * s2 * (probe * x_b))

            psi_f = torch.fft.fftshift(torch.fft.fft2(exit_wave, norm='ortho'), dim=(-2,-1))

            #Incoherent mode summation: intensity per mode, then sum over modes
            I_pred = torch.sum(torch.abs(psi_f)**2, dim = 2) #(B,C,P,H,W) -> (B,C,H,W)

        else:
            exit_wave_a = probe * x_a * scale
            exit_wave_b = 1j * probe * x_b * scale

            Psi_a = torch.fft.fft2(exit_wave_a, norm='ortho')
            Psi_b = torch.fft.fft2(exit_wave_b, norm='ortho')

            Psi_a = torch.fft.fftshift(Psi_a, dim=(-2,-1)) #(B,C,P,H,W)
            Psi_b = torch.fft.fftshift(Psi_b, dim=(-2,-1)) #(B,C,P,H,W)

            #Incoherent mode summation: compute per-mode basis images, then sum over modes
            X1 = torch.sum(torch.abs(Psi_a)**2, dim=2)                                    # (B,C,H,W)
            X2 = torch.sum(torch.abs(Psi_b)**2, dim=2)                                    # (B,C,H,W)
            X3 = torch.sum(2 * torch.real(Psi_a * torch.conj(Psi_b)), dim=2)              # (B,C,H,W)

            #Solve for scaling factors using mode-summed basis images
            s = self.solve_scaling_factors_from_basis(I_raw, X1, X2, X3)
            s_corr = self.enforce_physics_constraint(s)
            s1, s2 = s_corr[:,:,0,None,None], s_corr[:,:,1,None,None]
            I_pred = s1 * X1 + s2 * X2 + s1 * s2 * X3

        I_pred = torch.clamp(I_pred, min=0.0)
        return I_pred

    def solve_scaling_factors(self, I_measured, Psi_a, Psi_b):
        """
        Solves normal equation for scaling factors via eigendecomposition.
        """
        norm = I_measured.mean() + 1e-9
        y = (I_measured / norm).flatten(2).unsqueeze(-1).to(torch.float64)

        X1 = torch.abs(Psi_a)**2
        X2 = torch.abs(Psi_b)**2
        X3 = 2 * torch.real(Psi_a * torch.conj(Psi_b))

        X = torch.stack([X1.flatten(2), X2.flatten(2), X3.flatten(2)], dim=-1).to(torch.float64)

        col_norms = torch.sqrt(torch.sum(X**2, dim=-2, keepdim=True) + 1e-12)
        X_scaled = X / col_norms

        XT = X_scaled.transpose(-2, -1)
        XTX = XT @ X_scaled
        XTy = XT @ y

        L, V = torch.linalg.eigh(XTX)

        threshold = 1e-7
        L_inv = torch.where(L > threshold, 1.0 / L, torch.zeros_like(L))

        s_scaled = V @ (L_inv.unsqueeze(-1) * (V.transpose(-2, -1) @ XTy))
        s_scaled = s_scaled.squeeze(-1)

        s = (s_scaled / col_norms.squeeze(-2)) * norm

        if self.training and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            if L.min() < threshold:
                print(f"!!! Low Rank Detected: min_eig={L.min().item():.2e}")

        return s.to(torch.float32)

    def solve_scaling_factors_from_basis(self, I_measured, X1, X2, X3):
        """
        Solves normal equation for scaling factors from pre-computed basis images.
        Used for incoherent multi-mode probes where basis images are summed over modes
        before solving.

        :param I_measured: [B,C,H,W] measured intensities
        :param X1: [B,C,H,W] sum_p |Psi_a_p|^2
        :param X2: [B,C,H,W] sum_p |Psi_b_p|^2
        :param X3: [B,C,H,W] sum_p 2*Re(Psi_a_p * conj(Psi_b_p))
        """
        norm = I_measured.mean() + 1e-9
        y = (I_measured / norm).flatten(2).unsqueeze(-1).to(torch.float64)

        X = torch.stack([X1.flatten(2), X2.flatten(2), X3.flatten(2)], dim=-1).to(torch.float64)

        # Column normalization (Jacobi pre-conditioner)
        col_norms = torch.sqrt(torch.sum(X**2, dim=-2, keepdim=True) + 1e-12)
        X_scaled = X / col_norms

        # Normal equation via eigen-decomposition
        XT = X_scaled.transpose(-2, -1)
        XTX = XT @ X_scaled
        XTy = XT @ y

        L, V = torch.linalg.eigh(XTX)
        threshold = 1e-7
        L_inv = torch.where(L > threshold, 1.0 / L, torch.zeros_like(L))

        s_scaled = V @ (L_inv.unsqueeze(-1) * (V.transpose(-2, -1) @ XTy))
        s_scaled = s_scaled.squeeze(-1)

        s = (s_scaled / col_norms.squeeze(-2)) * norm

        if self.training and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            if L.min() < threshold:
                print(f"!!! Low Rank Detected: min_eig={L.min().item():.2e}")

        return s.to(torch.float32)

    def enforce_physics_constraint(self, c):
        """
        Projects 3 coefficients onto a 2D vector space via eigenvalue decomposition
        of the 2x2 coefficient matrix C = [[c1, c3],[c3, c2]].
        """
        c1, c2, c3 = c[:,:,0], c[:,:,1], c[:,:,2]
        lambda_max = 1/2 * (c1 + c2 + torch.sqrt((c1-c2)**2 + 4 * c3**2))

        v_1 = torch.where(torch.abs(c3) > torch.abs(lambda_max - c1), c3, lambda_max - c2)
        v_2 = torch.where(torch.abs(c3) > torch.abs(lambda_max - c1), lambda_max - c1, c3)

        norm = torch.sqrt(v_1**2 + v_2**2 + 1e-9)
        mag = torch.sqrt(torch.clamp(lambda_max, min=0))

        s_1 = v_1 / norm * mag
        s_2 = v_2 / norm * mag

        return torch.stack([s_1, s_2], dim=-1)

class ProbeIllumination(nn.Module):
    '''
    Probe illumination via Hadamard product of complex object and probe.

    Adds probe mode dimension (P) to the object for multi-mode support.

    Inputs:
        x - torch.Tensor (B,C,H,W) complex object patches
        probe - torch.Tensor (B,C,P,H,W) complex probe modes

    Returns:
        illuminated - torch.Tensor (B,C,P,H,W) exit waves
        masked_probe - torch.Tensor probe with mask applied
    '''
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__()
        self.N = data_config.N
        self.mask = model_config.probe_mask

    def forward(self, x, probe):
        x_reshaped = x.unsqueeze(dim=2)  # (B,C,H,W) -> (B,C,1,H,W)

        if self.mask is None:
            probe_mask = torch.ones((self.N, self.N), device=x.device)
        else:
            probe_mask = self.mask.to(x.device)

        illuminated = x_reshaped * probe * probe_mask.view(1, 1, 1, self.N, self.N)
        return illuminated, probe * probe_mask


class PolarForwardModel(nn.Module):
    '''
    Forward model for polar (amplitude/phase) object representation.
    Computes predicted diffraction amplitude via:
        reassemble → extract → probe illumination → FFT → sqrt(intensity) → affine scaling

    Returns predicted amplitude (not intensity).
    '''
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.object_big = model_config.object_big

        self.probe_illumination = ProbeIllumination(model_config, data_config)
        self.scaler = IntensityScalerModule(model_config)

        self.alpha = nn.Parameter(torch.ones(model_config.num_datasets))
        self.beta = nn.Parameter(torch.ones(model_config.num_datasets))

    def forward(self, x, positions, probe, output_scale_factor, experiment_ids=None):
        if self.object_big:
            reassembled_obj, _, _ = hh.reassemble_patches_position_real(
                x, positions,
                data_config=self.data_config,
                model_config=self.model_config)

            extracted_patch_objs = hh.extract_channels_from_region(
                reassembled_obj[:, None, :, :], positions,
                data_config=self.data_config,
                model_config=self.model_config)
        else:
            extracted_patch_objs = x

        illuminated_objs, _ = self.probe_illumination(extracted_patch_objs, probe)

        pred_unscaled, _ = hh.pad_and_diffract(illuminated_objs, pad=False)

        pred_scaled = self.scaler.inv_scale(pred_unscaled, output_scale_factor)

        if self.model_config.intensity_scale_trainable:
            alphas = self.alpha[experiment_ids]
            betas = self.beta[experiment_ids]

            if pred_scaled.ndim == 3:
                alphas = alphas.view(-1, 1, 1)
                betas = betas.view(-1, 1, 1)
            elif pred_scaled.ndim == 4:
                alphas = alphas.view(-1, 1, 1, 1)
                betas = betas.view(-1, 1, 1, 1)

            pred_scaled = alphas * pred_scaled + betas

        return pred_scaled


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
    def __init__(self, pred_is_amplitude=False):
        super(PoissonLoss, self).__init__()
        self.pred_is_amplitude = pred_is_amplitude

    def forward(self, pred, raw):
        self.poisson = PoissonIntensityLayer(pred, pred_is_amplitude=self.pred_is_amplitude)
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

class AmplitudeVarianceLoss(nn.Module):
    """
    Penalizes spatial variance of |z| = sqrt(real^2 + imag^2).
    Pushes toward uniform amplitude (outputs lie on a circle) without
    prescribing the radius.
    """
    def __init__(self):
        super().__init__()

    def forward(self, real, imag):
        modulus = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        mean_mod = torch.mean(modulus, dim=(2, 3), keepdim=True)
        return torch.sum((modulus - mean_mod) ** 2)

class ModulusTargetLoss(nn.Module):
    """
    Penalizes deviation of |z| from a target value with an optional dead zone.
    When dead_zone > 0, no penalty is applied within [target - delta, target + delta].
    """
    def __init__(self, target_modulus=1.0, dead_zone=0.1):
        super().__init__()
        self.target = target_modulus
        self.dead_zone = dead_zone

    def forward(self, real, imag):
        modulus = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        deviation = torch.abs(modulus - self.target)
        if self.dead_zone > 0:
            deviation = torch.clamp(deviation - self.dead_zone, min=0.0)
        return torch.mean(deviation ** 2)

class ChannelEnergyBalanceLoss(nn.Module):
    """
    Penalizes imbalance between real and imaginary channel energies.
    For a unit-circle object with uniformly distributed phase,
    E_real / E_imag ~ 1.
    """
    def __init__(self, target_ratio=1.0):
        super().__init__()
        self.target_ratio = target_ratio

    def forward(self, real, imag):
        E_real = torch.mean(real ** 2, dim=(1, 2, 3))
        E_imag = torch.mean(imag ** 2, dim=(1, 2, 3)) + 1e-8
        ratio = E_real / E_imag
        return torch.mean((ratio - self.target_ratio) ** 2)

class CombineComplexRectangular(nn.Module):
    '''
    Converts rectangular coordinates into torch complex

    Inputs
    ------
    amp: torch.Tensor
        real part of complex number
    phi: torch.Tensor
        imaginary part of complex number

    Outputs
    -------
    out: torch.Tensor
        Complex number
    '''
    def __init__(self):
        super().__init__()

    def forward(self, real: torch.Tensor, imag: torch.Tensor):
        out = real.to(dtype = torch.complex64) + 1j * imag.to(dtype=torch.complex64)
        return out

class ProbeReferenceLoss(nn.Module):
    """
    Probe reference loss for enforcing real-channel dominance.

    For a transparent object (O = 1 + 0j), the measured diffraction pattern
    is |FFT(Probe)|^2. When this reference pattern is fed through the
    autoencoder, the output should be real-dominated (imaginary ~ 0).

    This loss penalizes the L1 norm of the imaginary channel output
    when the input is the probe reference pattern, breaking the
    real/imaginary decoder symmetry during training.

    Loss = coeff * torch.abs(imag_output).sum()
    """
    def __init__(self):
        super().__init__()

    def forward(self, autoencoder: nn.Module,
                probe_single: torch.Tensor,
                C_in: int,
                scaler,
                data_config) -> torch.Tensor:
        """
        Args:
            autoencoder: The autoencoder module (encoder + decoder)
            probe_single: Single complex probe tensor, shape (N, N)
            C_in: Number of input channels (C_model if object_big else 1)
            scaler: IntensityScalerModule for RMS normalization
            data_config: DataConfig for computing RMS scaling factor
        Returns:
            Scalar loss = abs(imag_output).mean()
        """
        # Reference input computation - no learnable parameters involved
        with torch.no_grad():
            # 1. Compute transparent-object diffraction: |FFT(P)|^2
            I_ref = torch.abs(
                torch.fft.fftshift(torch.fft.fft2(probe_single, norm='ortho'))
            ) ** 2  # (N, N) real

            # 2. Expand to autoencoder input shape: (1, C_in, N, N)
            I_ref = I_ref.unsqueeze(0).unsqueeze(0).expand(1, C_in, -1, -1).contiguous()

            # 3. RMS-normalize (same normalization as training inputs)
            rms = torch.sqrt(torch.mean(I_ref ** 2)) + 1e-8
            I_ref_scaled = I_ref / rms

        # 4. Pass through autoencoder (WITH gradients)
        ref_real, ref_imag = autoencoder(I_ref_scaled)

        # 5. Loss: penalize imaginary channel
        return torch.abs(ref_imag).mean()

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
    Supports rectangular (real/imag) and polar (amp/phase) object representations.
    '''
    def __init__(self, model_config: ModelConfig, data_config: DataConfig, training_config: TrainingConfig):
        super(PtychoPINN, self).__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config
        self._polar = getattr(model_config, 'object_representation', 'rectangular') == 'polar'

        self.n_filters_scale = self.model_config.n_filters_scale

        self.scaler = IntensityScalerModule(model_config)
        self.probe_scale = data_config.probe_scale

        shared = getattr(model_config, 'use_shared_decoder', False)
        self.autoencoder = Autoencoder(model_config, data_config, shared_decoder=shared)

        if self._polar:
            self.combine_complex = CombineComplex()
            self.forward_model = PolarForwardModel(model_config, data_config)
        else:
            self.combine_complex = CombineComplexRectangular()
            self.forward_model = ForwardModel(model_config, data_config)

    def forward(self, x, positions, probe, input_scale_factor, output_scale_factor,
                experiment_ids=None, fine_tune=False):
        x = self.scaler.scale(x, input_scale_factor)
        x_real, x_imag = self.autoencoder(x)
        x_combined = self.combine_complex(x_real, x_imag)

        if self._polar:
            x_out = self.forward_model.forward(x_combined, positions,
                                               probe/self.probe_scale, output_scale_factor,
                                               experiment_ids=experiment_ids)
        else:
            x_out = self.forward_model.forward(x_combined, x,
                                               positions, probe/self.probe_scale, output_scale_factor,
                                               experiment_ids=experiment_ids,
                                               fine_tune=fine_tune)

        return x_out, x_real, x_imag

    def forward_predict(self, x, positions, probe, input_scale_factor):
        x = self.scaler.scale(x, input_scale_factor)
        x_real, x_imag = self.autoencoder(x)
        x_combined = self.combine_complex(x_real, x_imag)
        return x_combined

    def get_encoder_bottom_params(self):
        """Returns parameters from bottom 50% of encoder (early conv blocks)."""
        if not hasattr(self, 'autoencoder') or not hasattr(self.autoencoder, 'encoder'):
            return []
        encoder = self.autoencoder.encoder
        if not hasattr(encoder, 'blocks') or len(encoder.blocks) == 0:
            return []
        split_idx = len(encoder.blocks) // 2
        params = []
        for block in encoder.blocks[:split_idx]:
            params.extend(block.parameters())
        return params

    def get_encoder_top_params(self):
        """Returns parameters from top 50% of encoder (later conv blocks)."""
        if not hasattr(self, 'autoencoder') or not hasattr(self.autoencoder, 'encoder'):
            return []
        encoder = self.autoencoder.encoder
        if not hasattr(encoder, 'blocks') or len(encoder.blocks) == 0:
            return []
        split_idx = len(encoder.blocks) // 2
        params = []
        for block in encoder.blocks[split_idx:]:
            params.extend(block.parameters())
        return params

    def get_decoder_params(self):
        """Returns decoder base parameters (excluding final heads)."""
        params = []
        if getattr(self.autoencoder, '_shared', False):
            decoder = self.autoencoder.decoder
            if hasattr(decoder, 'blocks'):
                for block in decoder.blocks:
                    params.extend(block.parameters())
            if hasattr(decoder, 'merge_blocks'):
                for block in decoder.merge_blocks:
                    params.extend(block.parameters())
            if hasattr(decoder, 'attention_blocks'):
                for block in decoder.attention_blocks:
                    params.extend(block.parameters())
            if hasattr(decoder, 'refinement_blocks'):
                for block in decoder.refinement_blocks:
                    params.extend(block.parameters())
            if hasattr(decoder, 'eca'):
                params.extend(decoder.eca.parameters())
        else:
            if hasattr(self.autoencoder, 'decoder_amp'):
                decoder_amp = self.autoencoder.decoder_amp
                if hasattr(decoder_amp, 'blocks'):
                    for block in decoder_amp.blocks:
                        params.extend(block.parameters())
                if hasattr(decoder_amp, 'amp_activation'):
                    params.extend(decoder_amp.amp_activation.parameters())
            if hasattr(self.autoencoder, 'decoder_phase'):
                decoder_phase = self.autoencoder.decoder_phase
                if hasattr(decoder_phase, 'blocks'):
                    for block in decoder_phase.blocks:
                        params.extend(block.parameters())
                if hasattr(decoder_phase, 'phase_activation'):
                    params.extend(decoder_phase.phase_activation.parameters())
        return params

    def get_phase_head_params(self):
        """Returns final phase output layer parameters."""
        if getattr(self.autoencoder, '_shared', False):
            decoder = self.autoencoder.decoder
            if getattr(decoder, '_split_heads', False):
                return list(decoder.head_imag.parameters())
            elif hasattr(decoder, 'head'):
                return list(decoder.head.parameters())
            else:
                return []
        else:
            if not hasattr(self.autoencoder, 'decoder_phase'):
                return []
            decoder_phase = self.autoencoder.decoder_phase
            if hasattr(decoder_phase, 'phase'):
                return list(decoder_phase.phase.parameters())
            return []

    def get_amp_head_params(self):
        """Returns final amplitude output layer parameters."""
        if getattr(self.autoencoder, '_shared', False):
            decoder = self.autoencoder.decoder
            if getattr(decoder, '_split_heads', False):
                return list(decoder.head_real.parameters())
            else:
                return []
        else:
            if not hasattr(self.autoencoder, 'decoder_amp'):
                return []
            decoder_amp = self.autoencoder.decoder_amp
            if hasattr(decoder_amp, 'amp'):
                return list(decoder_amp.amp.parameters())
            return []

    def freeze_encoder(self):
        """Freeze all encoder parameters."""
        if hasattr(self.autoencoder, 'encoder'):
            for param in self.autoencoder.encoder.parameters():
                param.requires_grad = False

    def freeze_encoder_bottom(self):
        """Freeze bottom 50% of encoder (early layers)."""
        for param in self.get_encoder_bottom_params():
            param.requires_grad = False

    def unfreeze_encoder_top(self):
        """Unfreeze top 50% of encoder (later layers)."""
        for param in self.get_encoder_top_params():
            param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def print_trainable_status(self):
        """Debug helper: print which parts of the network are trainable."""
        print("\n" + "="*60)
        print("Network Trainable Status:")
        print("="*60)
        groups = {
            'Encoder (bottom 50%)': self.get_encoder_bottom_params(),
            'Encoder (top 50%)': self.get_encoder_top_params(),
            'Decoder (base)': self.get_decoder_params(),
            'Phase head': self.get_phase_head_params(),
            'Amp head': self.get_amp_head_params(),
        }
        for name, params in groups.items():
            trainable = any(p.requires_grad for p in params) if params else False
            n_params = sum(p.numel() for p in params)
            status = 'Trainable' if trainable else 'Frozen'
            print(f"{name:25s}: {status} ({n_params:,} params)")
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal: {trainable:,} / {total:,} trainable ({100*trainable/total:.1f}%)")
        print("="*60 + "\n")

    def verify_parameter_grouping(self):
        """Verification helper: Check parameter grouping correctness."""
        encoder_bottom = set(id(p) for p in self.get_encoder_bottom_params())
        encoder_top = set(id(p) for p in self.get_encoder_top_params())
        decoder = set(id(p) for p in self.get_decoder_params())
        phase_head = set(id(p) for p in self.get_phase_head_params())
        amp_head = set(id(p) for p in self.get_amp_head_params())
        autoencoder_params = set(id(p) for p in self.autoencoder.parameters())
        all_grouped = encoder_bottom | encoder_top | decoder | phase_head | amp_head
        missing = autoencoder_params - all_grouped
        print(f"Total autoencoder: {len(autoencoder_params)}, Grouped: {len(all_grouped)}, Missing: {len(missing)}")

# Supervised model equivalent

class Ptycho_Supervised(nn.Module):
    '''
    PtychoPINN supervised version. Skips forward model and overlap constraint.
    '''
    def __init__(self, model_config: ModelConfig, data_config: DataConfig, training_config: TrainingConfig):
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config

        self.n_filters_scale = self.model_config.n_filters_scale

        shared = getattr(model_config, 'use_shared_decoder', False)
        self.autoencoder = Autoencoder(model_config, data_config, shared_decoder=shared)
        polar = getattr(model_config, 'object_representation', 'rectangular') == 'polar'
        self.combine_complex = CombineComplex() if polar else CombineComplexRectangular()
        self.scaler = IntensityScalerModule(model_config)

    def forward(self, x, positions, probe, input_scale_factor, output_scaling_factor):
        x = self.scaler.scale(x, input_scale_factor)
        x_amp, x_phase = self.autoencoder(x)
        x_combined = self.combine_complex(x_amp, x_phase)
        return x_combined, x_amp, x_phase

    def forward_predict(self, x, positions, probe, input_scale_factor):
        x = self.scaler.scale(x, input_scale_factor)
        x_amp, x_phase = self.autoencoder(x)
        x_combined = self.combine_complex(x_amp, x_phase)
        return x_combined

#PtychoPINN Lightning Module
class PtychoPINN_Lightning(L.LightningModule):
    '''
    Lightning module for PtychoPINN (UNet variant).
    Uses RMS normalization with cosine LR scheduling.
    '''
    def __init__(self, model_config: ModelConfig,
                       data_config: DataConfig,
                       training_config: TrainingConfig,
                       inference_config: InferenceConfig):
        from torchmetrics import MeanMetric
        super().__init__()

        # Handle checkpoint loading: convert dict kwargs back to dataclass instances
        if isinstance(model_config, dict):
            model_config = ModelConfig(**model_config)
        if isinstance(data_config, dict):
            data_config = DataConfig(**data_config)
        if isinstance(training_config, dict):
            training_config = TrainingConfig(**training_config)
        if isinstance(inference_config, dict):
            inference_config = InferenceConfig(**inference_config)

        from dataclasses import asdict
        self.save_hyperparameters({
            'model_config': asdict(model_config),
            'data_config': asdict(data_config),
            'training_config': asdict(training_config),
            'inference_config': asdict(inference_config),
        })

        self.n_filters_scale = model_config.n_filters_scale
        self.predict = False

        #Configs
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config
        self.inference_config = inference_config

        self.torch_loss_mode = getattr(self.training_config, 'torch_loss_mode', 'poisson')
        if isinstance(self.torch_loss_mode, str):
            self.torch_loss_mode = self.torch_loss_mode.lower()
        if self.torch_loss_mode not in ('poisson', 'mae'):
            raise ValueError(
                f"Invalid torch_loss_mode='{self.torch_loss_mode}'. Expected 'poisson' or 'mae'."
            )

        self.scaler = IntensityScalerModule(model_config)

        #Other params
        self.lr = training_config.learning_rate
        self.accum_steps = training_config.accum_steps
        self.gradient_clip_val = training_config.gradient_clip_val
        self.warmup_epochs = getattr(training_config, 'warmup_epochs', 5)
        self.min_lr_ratio = getattr(training_config, 'min_lr_ratio', 0.01)
        self.validation_step_outputs = []
        self._fine_tuning_mode = False

        self.automatic_optimization = False

        #Model
        if model_config.mode == 'Unsupervised':
            arch = getattr(model_config, 'architecture', 'unet')
            if arch == 'ccnf':
                from ptycho_torch.beta_modules.latent_model import PtychoPINN_CCNF
                self.model = PtychoPINN_CCNF(model_config, data_config, training_config)
            elif arch == 'patterson':
                from ptycho_torch.beta_modules.patterson_model import PtychoPINN_Patterson
                self.model = PtychoPINN_Patterson(model_config, data_config, training_config)
            else:
                self.model = PtychoPINN(model_config, data_config, training_config)
        elif model_config.mode == 'Supervised':
            self.model = Ptycho_Supervised(model_config, data_config, training_config)

        self.total_epochs = training_config.epochs

        if self.model_config.mode == 'Supervised':
            if self.torch_loss_mode != 'mae':
                raise ValueError(
                    "Supervised mode requires torch_loss_mode='mae' so amplitude labels remain consistent."
                )
        else:
            desired_loss = 'Poisson' if self.torch_loss_mode == 'poisson' else 'MAE'
            if self.model_config.loss_function != desired_loss:
                logger.info(
                    "Overriding model_config.loss_function=%s to %s to match torch_loss_mode=%s",
                    self.model_config.loss_function,
                    desired_loss,
                    self.torch_loss_mode,
                )
                self.model_config.loss_function = desired_loss

        self._polar = getattr(model_config, 'object_representation', 'rectangular') == 'polar'

        #Choose loss function and logging
        if model_config.mode == 'Unsupervised' and model_config.loss_function == 'Poisson':
            self.Loss = PoissonLoss(pred_is_amplitude=self._polar)
            self.loss_name = 'poisson_train'
            self.val_loss_name = 'poisson_val'
        elif model_config.mode == 'Unsupervised' and model_config.loss_function == 'MAE':
            self.Loss = MAELoss()
            self.loss_name = 'mae_train'
            self.val_loss_name = 'mae_val'
        elif model_config.mode == 'Supervised' and model_config.loss_function == 'MAE':
            self.Loss = nn.L1Loss(reduction = 'none')
            self.loss_name = 'mae_train'
            self.val_loss_name = 'mae_val'

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

        self.probe_reference_coeff = getattr(model_config, 'probe_reference_coeff', 0.0)
        if self.probe_reference_coeff > 0:
            self.ProbeRefLoss = ProbeReferenceLoss()
            self.loss_name += '_ProbeRef'
            self.val_loss_name += '_ProbeRef'

        if model_config.amplitude_variance_loss:
            self.AmpVarLoss = AmplitudeVarianceLoss()
        if model_config.modulus_target_loss:
            self.ModTargetLoss = ModulusTargetLoss(
                target_modulus=model_config.modulus_target_value,
                dead_zone=model_config.modulus_target_dead_zone
            )
        if model_config.channel_balance_loss:
            self.ChanBalanceLoss = ChannelEnergyBalanceLoss()

        self.loss_name += '_loss'
        self.val_loss_name += '_loss'

    def forward(self, x, positions, probe, input_scale_factor, output_scale_factor, experiment_ids,
                fine_tune=False):
        x_out = self.model(x, positions, probe, input_scale_factor, output_scale_factor, experiment_ids,
                           fine_tune=fine_tune)
        return x_out

    def forward_predict(self, x, positions, probe, input_scale_factor):
        x_combined = self.model.forward_predict(x, positions, probe, input_scale_factor)
        return x_combined

    def _reshape_scale_tensor(self, scale_value, reference_tensor):
        """Convert scalar or 1D scaling factors into broadcastable tensors."""
        device = reference_tensor.device
        dtype = reference_tensor.dtype
        if scale_value is None:
            return torch.ones((reference_tensor.shape[0], 1, 1, 1), device=device, dtype=dtype)
        if not isinstance(scale_value, torch.Tensor):
            scale_tensor = torch.tensor(scale_value, device=device, dtype=dtype)
        else:
            scale_tensor = scale_value.to(device=device, dtype=dtype)
        if scale_tensor.ndim == 0:
            scale_tensor = scale_tensor.view(1, 1, 1, 1)
        elif scale_tensor.ndim == 1:
            scale_tensor = scale_tensor.view(-1, 1, 1, 1)
        return scale_tensor

    def compute_loss(self, batch):
        """Loss computation with RMS normalization."""
        x = batch[0]['images']
        positions = batch[0]['coords_relative']
        probe = batch[1]
        rms_scale = batch[0]['rms_scaling_constant']
        physics_scale = batch[0]['physics_scaling_constant']
        experiment_ids = batch[0]['experiment_id']
        probe_scaling = batch[2]

        if self.model_config.mode == 'Supervised':
            amp_label = batch[0]['label_amp']
            phase_label = batch[0]['label_phase']

        total_loss = 0.0

        if self._polar:
            output_scale = rms_scale
        else:
            output_scale = torch.sqrt(1/(probe_scaling**2 * physics_scale + 1e-9))

        pred, real, imag = self(x, positions, probe,
                                input_scale_factor=rms_scale,
                                output_scale_factor=output_scale,
                                experiment_ids=experiment_ids,
                                fine_tune=self._fine_tuning_mode)

        intensity_norm_factor = torch.mean(x).detach() + 1e-8

        if self.model_config.mode == 'Unsupervised':
            total_loss += self.Loss(pred, x).mean()
            total_loss /= intensity_norm_factor

        elif self.model_config.mode == 'Supervised':
            real_loss = self.Loss(real, amp_label).sum()
            imag_loss = self.Loss(imag, phase_label).sum()
            total_loss += 2 * real_loss + 4 * imag_loss

        if self.model_config.amp_loss:
            amp_reg_loss = self.AmpLoss(real).mean()
            total_loss += amp_reg_loss * self.model_config.amp_loss_coeff

        if self.model_config.phase_loss:
            phase_reg_loss = self.PhaseLoss(imag).mean()
            total_loss += phase_reg_loss * self.model_config.phase_loss_coeff

        if self.probe_reference_coeff > 0 and hasattr(self, 'ProbeRefLoss'):
            C_in = self.model_config.C_model if self.model_config.object_big else 1
            probe_single = probe[0, 0, 0]
            probe_ref_loss = self.ProbeRefLoss(
                self.model.autoencoder, probe_single, C_in,
                self.model.scaler, self.data_config
            )
            total_loss += self.probe_reference_coeff * probe_ref_loss
            self.log('probe_ref_loss', probe_ref_loss.detach(),
                     on_step=False, prog_bar=True, on_epoch=True, sync_dist=True)

        if not self._polar:
            if self.model_config.amplitude_variance_loss:
                amp_var = self.AmpVarLoss(real, imag)
                total_loss += amp_var * self.model_config.amplitude_variance_coeff
                self.log('amp_variance_loss', amp_var.detach(),
                         on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            if self.model_config.modulus_target_loss:
                mod_target = self.ModTargetLoss(real, imag)
                total_loss += mod_target * self.model_config.modulus_target_coeff
                self.log('modulus_target_loss', mod_target.detach(),
                         on_step=False, on_epoch=True, sync_dist=True)

            if self.model_config.channel_balance_loss:
                chan_bal = self.ChanBalanceLoss(real, imag)
                total_loss += chan_bal * self.model_config.channel_balance_coeff
                self.log('channel_balance_loss', chan_bal.detach(),
                         on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        loss = self.compute_loss(batch)
        scaled_loss = loss / self.accum_steps

        self.manual_backward(scaled_loss)

        if (batch_idx+1) % self.accum_steps == 0:
            if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.parameters(),
                    max_norm=self.gradient_clip_val,
                    norm_type=2.0
                )
            opt.step()
            opt.zero_grad()

        self.log(self.loss_name, loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.compute_loss(batch)
        self.log(self.val_loss_name, val_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        result = {"optimizer": optimizer}

        scheduler_choice = getattr(self.training_config, 'scheduler', 'Default')
        if scheduler_choice == 'Cosine':
            scheduler = self.build_warmup_cosine_scheduler(optimizer, total_epochs=self.trainer.max_epochs)
            result['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        elif scheduler_choice == 'Exponential':
            result['lr_scheduler'] = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif scheduler_choice in ('MultiStage', 'Adaptive'):
            logger.warning(
                "Scheduler '%s' is no longer supported. Falling back to constant learning rate.",
                scheduler_choice,
            )

        return result

    def build_warmup_cosine_scheduler(self, optimizer, total_epochs):
        """Build warmup cosine annealing LR scheduler."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_epochs = max(0, self.warmup_epochs)
        base_lr = optimizer.param_groups[0]['lr']
        eta_min = base_lr * self.min_lr_ratio

        if warmup_epochs == 0:
            return CosineAnnealingLR(
                optimizer, T_max=total_epochs, eta_min=eta_min
            )

        linear_warmup = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_epochs
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs - warmup_epochs),
            eta_min=eta_min
        )
        return SequentialLR(
            optimizer, schedulers=[linear_warmup, cosine],
            milestones=[warmup_epochs]
        )

    def freeze_encoder(self):
        """Freezes encoder parameters for fine-tuning."""
        encoder = self.model.autoencoder.encoder
        for param in encoder.parameters():
            param.requires_grad = False
        self._fine_tuning_mode = True
        print("Encoder layers frozen for fine-tuning. Only decoder layers will be updated.")

    def on_train_epoch_start(self):
        if self.global_rank == 0:
            print(f"\n{'='*60}")
            print(f"Starting Epoch {self.current_epoch + 1}/{self.trainer.max_epochs}")
            print(f"{'='*60}")
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_epoch=True, sync_dist=True)

    def on_train_epoch_end(self):
        """Manually step LR scheduler since we use manual optimization."""
        sch = self.lr_schedulers()
        if sch is not None:
            if isinstance(sch, list):
                for scheduler in sch:
                    scheduler.step()
            else:
                sch.step()
            current_lr = self.optimizers().param_groups[0]['lr']
            self.log('learning_rate', current_lr, on_epoch=True, sync_dist=True)
            if self.global_rank == 0:
                print(f"Epoch {self.current_epoch} complete. New LR: {current_lr:.2e}")

    def on_after_backward(self):
        if self.global_step % 50 == 0:
            if getattr(self.model.autoencoder, '_shared', False):
                decoder = self.model.autoencoder.decoder
                if getattr(decoder, '_split_heads', False):
                    grad_norm = decoder.head_real.conv1.weight.grad.norm()
                else:
                    grad_norm = decoder.head.conv1.weight.grad.norm()
            elif hasattr(self.model.autoencoder, 'decoder_amp'):
                grad_norm = self.model.autoencoder.decoder_amp.amp.conv1.weight.grad.norm()
            else:
                return
            self.log("grad_norm", grad_norm, on_step=True, prog_bar=True, logger=True)
