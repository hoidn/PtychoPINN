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

#Lightning
import lightning as L

# Shared base modules imported from model.py to avoid duplication
from ptycho_torch.model import (
    Tanh_custom_act,
    Amplitude_activation,
    ConvBaseBlock,
    ConvPoolBlock,
    ConvUpBlock,
    CombineComplex,
    CombineComplexRectangular,
    LambdaLayer,
    Decoder_filters,
    ProbeIllumination,
    IntensityScalerModule,
    MAELoss,
    MeanDeviationLoss,
    TotalVariationLoss,
    SafePoissonLoss,
    ProbeReferenceLoss,
    Encoder as BaseEncoder,
    Decoder_base as BaseDecoderBase,
)

#Encoder - subclass with skip connections for UNet

class Encoder(BaseEncoder):
    """Encoder with skip connections for UNet architecture.

    Inherits __init__ entirely from BaseEncoder (same blocks, same state_dict keys).
    Only overrides forward() to capture skip connections before pooling.
    """
    def forward(self, x):
        skips = []

        for i, block in enumerate(self.blocks):
            x = block.forward_conv(x)

            # Capture before pooling (except last block)
            if i < len(self.blocks) - 1:
                skips.append(x)

            #Pool/downsample
            x = block.forward_pool(x)

        return x, skips

#Decoders

class Decoder_base(BaseDecoderBase):
    """Decoder base with skip connection merging for UNet architecture.

    Inherits self.blocks and self.attention_blocks from parent __init__.
    Adds self.merge_blocks on top for skip connection handling (UNet-specific).
    """
    def __init__(self, model_config: ModelConfig, data_config: DataConfig, batch_norm=False):
        super(Decoder_base, self).__init__(model_config, data_config, batch_norm=batch_norm)

        self.merge_blocks = nn.ModuleList()
        for i in range(len(self.blocks)):
            decoder_ch = self.filters[i+1] #Output of upsampling
            #Mix with encoder
            encoder_ch = self._get_encoder_channels(i, model_config, data_config)

            self.merge_blocks.append(nn.Sequential(
                #Append actual conv mixing layer for residual
                nn.Conv2d(decoder_ch + encoder_ch, decoder_ch, kernel_size = 3, padding = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(decoder_ch, decoder_ch, kernel_size = 3, padding = 1),
                nn.ReLU(inplace=True)
            ))

    def _get_encoder_channels(self, decoder_level, model_config, data_config):
        # Encoder produces channel number order: [64, 128, 256]
        # Decoder moves backward ([256, 128, 64])
        # Need to do proper accounting for dimension order

        # Recreating encoder structure from above
        N = data_config.N
        n_filters_scale = model_config.n_filters_scale

        # Recreate encoder filter structure
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
            return self.filters[decoder_level+1] #Backup just in case

    def forward(self, x, skips = None):
        # Accounts for skip connections
        for i, (up_block, merge_block) in enumerate(zip(self.blocks, self.merge_blocks)):
            x = up_block(x)
            # Retrieve skip and concatenate
            if skips is not None and i < len(skips):
                skip = skips[-(i+1)]


            # Spatial dimension mismatch if exists
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(skip, size = x.shape[2:],
                                     mode = 'bilinear', align_corners = False)

            #Concatenate
            x = torch.cat([x, skip], dim = 1) # Concatenate along channl dimension

            #Apply merge block for weighted residual addition
            x = merge_block(x)

            # Apply attention
            x = self.attention_blocks[i](x)


        return x

class Decoder_last(nn.Module):
    '''
    Base class for the final decoder stage. Handles conditional BatchNorm.
    '''
    def __init__(self, model_config: ModelConfig, data_config: DataConfig,
                 in_channels, out_channels,
                 activation = torch.sigmoid, name = '', batch_norm=False,
                 combined = False): # Added batch_norm flag
        super(Decoder_last, self).__init__()
        #Configs
        self.model_config = model_config
        self.data_config = data_config
        self.n_filters_scale = model_config.n_filters_scale
        self.combined = combined

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
        # Path 1: Main features
        x1_in = x[:, :-self.c_outer, :, :]
        x1 = self.conv1(x1_in)
        if self.batch_norm and self.bn1:
            x1 = self.bn1(x1)

        x1 = self.padding(x1)

        if not self.combined:
            x1 = self.activation(x1)

        # Path 2: Detail/Upsample features
        x2_in = x[:, -self.c_outer:, :, :]
        x2 = self.conv_up_block(x2_in)
        x2 = self.conv2(x2)
        if self.batch_norm and self.bn2:
            x2 = self.bn2(x2)

        if not self.combined:
            x2 = F.celu(x2, alpha = 1.0)

        # SUM FIRST: Combine the linear information
        # Note: Ensure spatial dimensions match here (padding/upsampling)
        outputs = x1 + x2


        # ACTIVATE LAST: One activation to rule the range
        # outputs = self.activation(combined)
        if self.combined:
            outputs = self.activation(outputs)


        return outputs

class Decoder_last_Amp(Decoder_last):
    '''Final decoder stage for Amplitude/Real. Inherits from Decoder_last, ensuring batch_norm=False.'''
    def __init__(self, model_config: ModelConfig, data_config: DataConfig,
                 in_channels, out_channels,
                 activation = torch.sigmoid, name = ''):
        activation = lambda x: 0.2 + torch.tanh(x)
        # Explicitly call parent with batch_norm=False
        super(Decoder_last_Amp, self).__init__(model_config, data_config, in_channels, out_channels,
                                               activation=activation, name=name, batch_norm=False,
                                               combined = True)



class Decoder_last_Phase(Decoder_last):
    '''Final decoder stage for Phase/Imaginary. Inherits from Decoder_last, ensuring batch_norm=True.'''
    def __init__(self, model_config: ModelConfig, data_config: DataConfig,
                 in_channels, out_channels,
                 activation = torch.sigmoid, name = ''):
        activation = lambda x: 1.2 * torch.tanh(x)
        # Explicitly call parent with batch_norm=True (or as configured)
        super(Decoder_last_Phase, self).__init__(model_config, data_config, in_channels, out_channels,
                                                 activation=activation, name=name, batch_norm=model_config.batch_norm,
                                                 combined = True) # Use config BN

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

    def forward(self, x, skips = None):
        # Apply upscale block layers (now with BN from Decoder_base)
        x = super().forward(x, skips)

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


    def forward(self, x, skips = None):
        #Apply upscale block layers
        x = super().forward(x, skips)

        #Apply final layer
        outputs = self.amp(x)

        return outputs


# Shared decoder components

class FeatureRefinementBlock(nn.Module):
    """
    Residual bottleneck expansion block. Temporarily expands channel dimensionality
    to allow the network to disentangle real-vs-imaginary feature subsets before the
    output split in the shared decoder.

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    expansion_factor : int
        Channel expansion multiplier for the intermediate convolutions.
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
    Shared decoder for real and imaginary components. Inherits all upsampling,
    skip connection, and merge block logic from Decoder_base. Adds per-level
    refinement blocks so modality-specific features develop progressively at
    every spatial scale, not just before the output head.

    Refinement expansion schedule (coarse -> fine):
        Level 0 (coarsest): expansion_factor = 1  -- shared structure dominates
        Level 1:            expansion_factor = 1
        ...
        Level N (finest):   expansion_factor = 2  -- modality divergence sharpest

    Output shape: [B, 2*C_out, N, N]  (first C_out = real, last C_out = imaginary)
    """
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__(model_config, data_config, batch_norm=model_config.batch_norm)

        C_out = model_config.decoder_last_amp_channels if model_config.object_big else 1
        n_levels = len(self.blocks)

        # Per-level refinement blocks with graduated expansion factors.
        # Coarser levels get E=1 (lightweight), finest level gets E=2 (full).
        self.refinement_blocks = nn.ModuleList()
        for i in range(n_levels):
            ch = self.filters[i + 1]  # output channels at this level
            expansion = 2 if i == n_levels - 1 else 1
            self.refinement_blocks.append(
                FeatureRefinementBlock(ch, expansion_factor=expansion)
            )

        # Final ECA before head -- operates on finest decoder features
        base_ch = self.filters[-1]  # n_filters_scale * 32
        self.eca = ECALayer(channel=base_ch)

        # Combined output head -- reuses Decoder_last with doubled output channels
        activation = lambda x: 1.2 * torch.tanh(x)
        self.head = Decoder_last(model_config, data_config,
                                 in_channels=base_ch,
                                 out_channels=C_out * 2,
                                 activation=activation,
                                 batch_norm=model_config.batch_norm,
                                 combined=True)

    def forward(self, x, skips=None):
        # Override Decoder_base.forward to insert refinement after each level

        for i, (up_block, merge_block) in enumerate(zip(self.blocks, self.merge_blocks)):
            x = up_block(x)

            # Skip connection
            if skips is not None and i < len(skips):
                skip = skips[-(i+1)]
                if skip.shape[2:] != x.shape[2:]:
                    skip = F.interpolate(skip, size=x.shape[2:],
                                         mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)

            x = merge_block(x)
            x = self.attention_blocks[i](x)

            # Per-level refinement
            x = self.refinement_blocks[i](x)

        x = self.eca(x)

        return self.head(x)

#Autoencoder

class Autoencoder(nn.Module):
    def __init__(self, model_config: ModelConfig, data_config: DataConfig,
                 shared_decoder: bool = False):
        super(Autoencoder, self).__init__()
        self.model_config = model_config
        self.data_config = data_config
        self._shared = shared_decoder
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
        #Decoders
        if shared_decoder:
            self.decoder = Decoder_shared(model_config, data_config)
        else:
            self.decoder_amp = Decoder_amp(model_config, data_config) # Pass configs
            self.decoder_phase = Decoder_phase(model_config, data_config) # Pass configs


    def forward(self, x):
        #Encoder
        x, skips = self.encoder(x)
        #CBAM optional
        x = self.bottleneck_cbam(x)
        #Decoders
        if getattr(self, '_shared', False):
            combined = self.decoder(x, skips)  # [B, 2*C, N, N]
            C = combined.shape[1] // 2
            x_real = combined[:, :C, :, :]
            x_imag = combined[:, C:, :, :]
        else:
            x_real = self.decoder_amp(x, skips)
            x_imag = self.decoder_phase(x, skips)

        return x_real, x_imag

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
        self.reassemble_patches = LambdaLayer(hh.reassemble_patches_position_real_probe)

        self.pad_patches = LambdaLayer(hh.pad_patches)

        self.trim_reconstruction = LambdaLayer(hh.trim_reconstruction)

        self.extract_patches = LambdaLayer(hh.extract_channels_from_region)

        #Probe Illumination - Pass configs
        self.probe_illumination = ProbeIllumination(model_config, data_config)

        #Pad/diffract
        self.pad_and_diffract = LambdaLayer(hh.pad_and_diffract)

        #Scaling
        self.scaler = IntensityScalerModule(model_config)

        #Rectangular Scaling
        self.rect_scaler = RectangularScaledDiffraction(model_config)


    def forward(self, x, I_measured,
                positions, probe, output_scale_factor,
                experiment_ids = None, fine_tune = False):
        #Note: For this rectangular net patch, the output scale factor will simply be
        #the average intensity of measured patterns PER pixel
        #Reassemble patches

        if self.object_big:
            # Pass config objects to helper function
            reassembled_obj, _, _ = hh.reassemble_patches_position_real_probe(x, positions,
                                                                  data_config=self.data_config,
                                                                  model_config=self.model_config,
                                                                  probe = probe,
                                                                  use_probe_weights = True)

            #Extract patches - Pass config objects to helper function
            extracted_patch_objs = hh.extract_channels_from_region(reassembled_obj[:,None,:,:], positions,
                                                                   data_config=self.data_config,
                                                                   model_config=self.model_config)

        else:
            extracted_patch_objs = x

        #Apply probe illum
        pred_scaled_intensity = self.rect_scaler(x = extracted_patch_objs,
                                                 I_raw = I_measured,
                                                 probe = probe,
                                                 scale = output_scale_factor,
                                                 experiment_ids = experiment_ids,
                                                 autograd = True)

        #Return unscaled product
        return pred_scaled_intensity


class RectangularScaledDiffraction(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()

        #Fitting parameters for scaling
        self.s1 = nn.Parameter(torch.ones(model_config.num_datasets))
        self.s2 = nn.Parameter(torch.ones(model_config.num_datasets))

    def forward(self,
                x: torch.Tensor,
                I_raw: torch.Tensor,
                probe: torch.Tensor,
                scale: torch.Tensor,
                experiment_ids: torch.Tensor,
                autograd: bool = True):
        """
        Docstring for forward

        :param self: Description
        :param x: [B,C,H,W] assumed imaginary tensor, will need to grab real and imaginary components
        :param probe: [H,W]imaginary tensor, will need to possibly grab both components
        :param per_pixel_diff_scale_factor: per pixel scale factor, need to multiply by dimensionality of x to get total

        Actually returns the amplitude of the predicted pattern, not the intensity
        The Poisson module re-squares the amplitude, returning us to intensity units.
        """

        #Labeling a as real, b as imag (following document labeling)
        x = x.unsqueeze(dim=2) # [B, C, H, W] -> [B, C, P, H, W] for extra probe modes
        x_a, x_b = x.real, x.imag
        # print(f"X max:{x_a.abs().max()}")

        #Calculate exit waves (notation from math document)
        #a and b labeling is arbitrary

        if autograd:
            s1 = self.s1[experiment_ids].view(-1,1,1,1,1) # -> (B,1,1,1,1)
            s2 = self.s2[experiment_ids].view(-1,1,1,1,1) # -> (B,1,1,1,1)

            scale = scale.unsqueeze(dim=2) #Add probe mode dimension

            exit_wave = scale * (s1 * (probe * x_a) + 1j * s2 * (probe * x_b))

            psi_f = torch.fft.fftshift(torch.fft.fft2(exit_wave, norm = 'ortho'),dim=(-2,-1))
            psi_f = torch.sum(psi_f, dim = 2) #Collapse empty probe mode dimension

            I_pred = torch.abs(psi_f)**2

        else:
            #Calculat exit waves manually
            exit_wave_a = probe * x_a * scale
            exit_wave_b = 1j * probe * x_b * scale

            #Far field prop, uncentered
            Psi_a = torch.fft.fft2(exit_wave_a, norm = 'ortho')
            Psi_b = torch.fft.fft2(exit_wave_b, norm = 'ortho')
            Psi_p = torch.fft.fft2(probe, norm = 'ortho')

            #Shift
            Psi_a = torch.fft.fftshift(Psi_a, dim = (-2,-1)) #(B,C,P,H,W)
            Psi_b = torch.fft.fftshift(Psi_b, dim = (-2,-1)) #(B,C,P,H,W)
            Psi_p = torch.fft.fftshift(Psi_p, dim = (-2,-1))

            #Sum out probe modes (hif there are any, currently there aren't)
            Psi_a = torch.sum(Psi_a, dim = 2)# (B,C,P,H,W) -> (B,C,H,W)
            # Psi_a = Psi_a + 1e-3 * torch.sum(Psi_p,dim=2) # Add unit exit wave to encourage additional learning
            #Modify Psi_a to include the probe
            Psi_b = torch.sum(Psi_b, dim = 2) # (B,C,P,H,W) -> (B,C,H,W)
            #Otherwise, do an image by image linear solve
            s = self.solve_scaling_factors(I_raw, Psi_a, Psi_b)
            #If fine-tuning, project 3 constants onto 2d vector space since we technically only
            #have 2 scaling terms, the cross term will truly be fitted.
            s_corr = self.enforce_physics_constraint(s)
            s1, s2 = s_corr[:,:,0,None,None], s_corr[:,:,1,None,None]
            # print(f"s1 sample: {s1[0,0,:,:]}")
            # print(f"s2 sample: {s2[0,0,:,:]}")
            I_pred = (s1 * torch.abs(Psi_a)**2 +
                      s2 * torch.abs(Psi_b)**2 +
                      s1 * s2 * 2 * torch.real(Psi_a * torch.conj(Psi_b)))


        # Clamp for no neg. quantities (should not be possible)
        I_pred = torch.clamp(I_pred, min=0.0) #[B,C,H,W]

        # 2. Add epsilon and sqrt for amplitude, to prevent gradient explosion near 0
        return I_pred

    def solve_scaling_factors(self, I_measured, Psi_a, Psi_b):
        """
        Uses torch linalg to solve normal equation for scaling factors,
        which will be used to match predicted unscaled intensities with raw intensities

        We are trying to solve the following equation:
        I_pred = s_1 * |Psi_a|**2 + s_2 * |Psi_b|**2 + 2 * s_1 * s_2 * Re(Psi_a * conj(Psi_b))
        """

        #Compute different linear components
        norm = I_measured.mean() + 1e-9
        y = (I_measured / norm).flatten(2).unsqueeze(-1).to(torch.float64)

        X1 = (torch.abs(Psi_a)**2)
        X2 = (torch.abs(Psi_b)**2)
        X3 = (2 * torch.real(Psi_a * torch.conj(Psi_b)))

        # print(f"X1 sum: {X1[0,0].sum()}")
        # print(f"X2 sum: {X2[0,0].sum()}")
        # print(f"X3 sum: {X3[0,0].sum()}")

        #Flatten H,W dimensions, stack on last dimension
        X = torch.stack([X1.flatten(2), X2.flatten(2), X3.flatten(2)], dim=-1).to(torch.float64)

        # 2. COLUMN NORMALIZATION (The Jacobi Pre-conditioner)
        # Calculate the norm of each column (the 3 basis components)
        # This prevents the determinant from blowing up to 10^18
        col_norms = torch.sqrt(torch.sum(X**2, dim=-2, keepdim=True) + 1e-12) # [B, C, 1, 3]
        X_scaled = X / col_norms

        # 3. Form Normal Equation in float64
        XT = X_scaled.transpose(-2, -1)
        XTX = XT @ X_scaled  # Now the diagonal elements will be roughly 1.0
        det = torch.linalg.det(XTX).mean()
        XTy = XT @ y
        # # 4. Solve with a slightly higher epsilon
        # # Since XTX is now scaled to 1.0, 1e-6 is a very strong, stable epsilon
        # eps = 1e-6 * torch.eye(3, device=XTX.device, dtype=torch.float64)
        # s_scaled = torch.linalg.solve(XTX + eps, XTy).squeeze(-1) # [B, C, 3]

        # # 5. Reverse the scaling to get the actual coefficients
        # s = s_scaled/col_norms.squeeze(-2)

        # --- NEW STABLE SOLVE USING EIGEN-DECOMPOSITION ---
        # Since XTX is only 3x3, we can use eigh (fast and stable)
        # L = eigenvalues, V = eigenvectors
        L, V = torch.linalg.eigh(XTX)

        # Truncated Pseudo-inverse:
        # Any eigenvalue smaller than threshold is treated as 0 to prevent division by 10^-14
        threshold = 1e-7
        L_inv = torch.where(L > threshold, 1.0 / L, torch.zeros_like(L))

        # solve: s = V @ diag(L_inv) @ V.T @ XTy
        s_scaled = V @ (L_inv.unsqueeze(-1) * (V.transpose(-2, -1) @ XTy))
        s_scaled = s_scaled.squeeze(-1)

        # 4. Reverse scaling
        s = (s_scaled / col_norms.squeeze(-2)) * norm

        # Debug Logging for Rank 0 only (to avoid log clutter)
        if self.training and torch.distributed.get_rank() == 0:
            if L.min() < threshold:
                print(f"!!! Low Rank Detected: min_eig={L.min().item():.2e}")

        return s.to(torch.float32)

    def enforce_physics_constraint(self, c):
        """
        Takes 3 "separate" coefficients from the intensity formulation and projects it onto a
        2D vector space. Solution arises from computing eigenvalues for 2D coefficient matrix
        C = [[c1, c3],[c3, c2]] where c3 is the cross term of s_1 * s_2 from the origina lformulation.

        :param c: torch.Tensor, [B,3]. Represents 3 coefficients from original linear fit

        Returns
            torch.Tensor [B,2]
        """
        #Get maximum eigenvalue from quadratic equation solving det(C - lambda * I)
        c1, c2, c3 = c[:,:,0], c[:,:,1], c[:,:,2]
        lambda_max = 1/2 * (c1 + c2 + torch.sqrt((c1-c2)**2 + 4 * c3**2))

        #Get unit vector coordinates for s_1 and s_2
        #Handles numerical instabilities in cases where magnitudes are very diff.
        v_1 = torch.where(torch.abs(c3) > torch.abs(lambda_max - c1), c3, lambda_max - c2)
        v_2 = torch.where(torch.abs(c3) > torch.abs(lambda_max - c1), lambda_max - c1, c3)

        #Other vector quantities
        norm = torch.sqrt(v_1**2 + v_2**2 + 1e-9)
        mag = torch.sqrt(torch.clamp(lambda_max, min=0))

        #Scale unit vectors to correct coordinates
        s_1 = v_1 / norm * mag
        s_2 = v_2 / norm * mag

        return torch.stack([s_1,s_2], dim = -1)


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

class PoissonIntensityLayer(nn.Module):
    '''
    Applies poisson intensity scaling using torch.distributions
    Calculates the negative log likelihood of observing the raw data given the predicted intensities
    '''
    def __init__(self, intensities):

        super(PoissonIntensityLayer, self).__init__()
        #Poisson rate parameter (lambda)
        Lambda = intensities
        #Create Poisson distribution
        #Second parameter (batch size) controls how many dimensions are summed over starting from the last
        self.poisson_dist = dist.Independent(dist.Poisson(Lambda), 3)

    def forward(self, x):
        #Apply poisson distribution
        return -self.poisson_dist.log_prob(x)

class PhaseCenterLoss(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass

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
        shared = getattr(model_config, 'use_shared_decoder', False)
        self.autoencoder = Autoencoder(model_config, data_config, shared_decoder=shared)
        self.combine_complex = CombineComplexRectangular()

        #Adding named modules for forward operation
        self.forward_model = ForwardModel(model_config, data_config)


    def forward(self, x, positions, probe, input_scale_factor, output_scale_factor,
                experiment_ids = None, fine_tune = False):

        #Scaling down (normalizing to 1)
        x = self.scaler.scale(x, input_scale_factor)
        #Autoencoder result
        x_real, x_imag = self.autoencoder(x)
        #Combine amp and phase

        x_combined = self.combine_complex(x_real, x_imag)

        #Run through forward model. Unscaled diffraction pattern
        x_out = self.forward_model.forward(x_combined, x,
                                           positions, probe/self.probe_scale, output_scale_factor,
                                           experiment_ids = experiment_ids,
                                           fine_tune = fine_tune)



        return x_out, x_real, x_imag

    def forward_predict(self, x, positions, probe, input_scale_factor):
        #Scaling
        x = self.scaler.scale(x, input_scale_factor)
        #Autoencoder result
        x_real, x_imag = self.autoencoder(x)
        #Combine amp and phase
        x_combined = self.combine_complex(x_real, x_imag)

        return x_combined

    def get_encoder_bottom_params(self):
        """
        Returns parameters from bottom 50% of encoder (early conv blocks).
        These learn basic diffraction pattern features and should change minimally.

        Structure: self.autoencoder.encoder.blocks (ModuleList of ConvPoolBlock)
        """
        if not hasattr(self, 'autoencoder') or not hasattr(self.autoencoder, 'encoder'):
            print("Warning: autoencoder.encoder not found")
            return []

        encoder = self.autoencoder.encoder

        # encoder.blocks is a ModuleList of ConvPoolBlock instances
        if not hasattr(encoder, 'blocks'):
            print("Warning: encoder.blocks not found")
            return []

        encoder_blocks = encoder.blocks  # This is a ModuleList

        if len(encoder_blocks) == 0:
            print("Warning: encoder.blocks is empty")
            return []

        # Split at midpoint
        split_idx = len(encoder_blocks) // 2
        bottom_blocks = encoder_blocks[:split_idx]

        # Collect all parameters from bottom blocks
        params = []
        for block in bottom_blocks:
            params.extend(block.parameters())

        return params

    def get_encoder_top_params(self):
        """
        Returns parameters from top 50% of encoder (later conv blocks).
        These learn higher-level features and can adapt more to experimental data.

        Structure: self.autoencoder.encoder.blocks (ModuleList of ConvPoolBlock)
        """
        if not hasattr(self, 'autoencoder') or not hasattr(self.autoencoder, 'encoder'):
            print("Warning: autoencoder.encoder not found")
            return []

        encoder = self.autoencoder.encoder

        if not hasattr(encoder, 'blocks'):
            print("Warning: encoder.blocks not found")
            return []

        encoder_blocks = encoder.blocks

        if len(encoder_blocks) == 0:
            print("Warning: encoder.blocks is empty")
            return []

        split_idx = len(encoder_blocks) // 2
        top_blocks = encoder_blocks[split_idx:]

        params = []
        for block in top_blocks:
            params.extend(block.parameters())

        return params

    def get_decoder_params(self):
        """
        Returns decoder base parameters (excluding final heads).

        For dual-branch: upsampling blocks from decoder_amp and decoder_phase.
        For shared decoder: upsampling blocks + refinement + eca (excluding head).
        """
        params = []

        if getattr(self.autoencoder, '_shared', False):
            # Shared decoder: base blocks + refinement + eca (not the head)
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
            # Dual-branch: get decoder_amp base blocks (excluding 'amp' head)
            if hasattr(self, 'autoencoder') and hasattr(self.autoencoder, 'decoder_amp'):
                decoder_amp = self.autoencoder.decoder_amp
                if hasattr(decoder_amp, 'blocks'):
                    for block in decoder_amp.blocks:
                        params.extend(block.parameters())
                if hasattr(decoder_amp, 'amp_activation'):
                    params.extend(decoder_amp.amp_activation.parameters())

            # Get decoder_phase base blocks (excluding 'phase' head)
            if hasattr(self, 'autoencoder') and hasattr(self.autoencoder, 'decoder_phase'):
                decoder_phase = self.autoencoder.decoder_phase
                if hasattr(decoder_phase, 'blocks'):
                    for block in decoder_phase.blocks:
                        params.extend(block.parameters())
                if hasattr(decoder_phase, 'phase_activation'):
                    params.extend(decoder_phase.phase_activation.parameters())

        return params

    def get_phase_head_params(self):
        """
        Returns final output layer parameters for the phase/imaginary branch.
        For shared decoder with split heads, returns head_imag parameters.
        For shared decoder without split heads, returns the combined head parameters.
        """
        if getattr(self.autoencoder, '_shared', False):
            decoder = self.autoencoder.decoder
            if getattr(decoder, '_split_heads', False):
                # Split heads: imag head belongs to phase group
                return list(decoder.head_imag.parameters())
            elif hasattr(decoder, 'head'):
                return list(decoder.head.parameters())
            else:
                print("Warning: shared decoder head not found")
                return []
        else:
            if not hasattr(self, 'autoencoder') or not hasattr(self.autoencoder, 'decoder_phase'):
                print("Warning: autoencoder.decoder_phase not found")
                return []
            decoder_phase = self.autoencoder.decoder_phase
            if hasattr(decoder_phase, 'phase'):
                return list(decoder_phase.phase.parameters())
            else:
                print("Warning: decoder_phase.phase (head) not found")
                return []

    def get_amp_head_params(self):
        """
        Returns final output layer parameters for the amplitude/real branch.
        For shared decoder with split heads, returns head_real parameters.
        For shared decoder without split heads, returns empty (head params are in get_phase_head_params).
        """
        if getattr(self.autoencoder, '_shared', False):
            decoder = self.autoencoder.decoder
            if getattr(decoder, '_split_heads', False):
                # Split heads: real head belongs to amp group
                return list(decoder.head_real.parameters())
            else:
                return []
        else:
            if not hasattr(self, 'autoencoder') or not hasattr(self.autoencoder, 'decoder_amp'):
                print("Warning: autoencoder.decoder_amp not found")
                return []
            decoder_amp = self.autoencoder.decoder_amp
            if hasattr(decoder_amp, 'amp'):
                return list(decoder_amp.amp.parameters())
            else:
                print("Warning: decoder_amp.amp (head) not found")
                return []

    def freeze_encoder(self):
        """Freeze all encoder parameters."""
        if hasattr(self, 'autoencoder') and hasattr(self.autoencoder, 'encoder'):
            for param in self.autoencoder.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen")
        else:
            print("Warning: Could not freeze encoder - autoencoder.encoder not found")

    def freeze_encoder_bottom(self):
        """Freeze bottom 50% of encoder (early layers)."""
        bottom_params = self.get_encoder_bottom_params()
        if bottom_params:
            for param in bottom_params:
                param.requires_grad = False
            print(f"Encoder bottom frozen ({len(bottom_params)} parameter tensors)")
        else:
            print("Warning: No encoder bottom parameters found to freeze")

    def unfreeze_encoder_top(self):
        """Unfreeze top 50% of encoder (later layers)."""
        top_params = self.get_encoder_top_params()
        if top_params:
            for param in top_params:
                param.requires_grad = True
            print(f"Encoder top unfrozen ({len(top_params)} parameter tensors)")
        else:
            print("Warning: No encoder top parameters found to unfreeze")

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        print("All parameters unfrozen")

    def print_trainable_status(self):
        """Debug helper: print which parts of the network are trainable."""
        print("\n" + "="*60)
        print("Network Trainable Status:")
        print("="*60)

        # Get parameters in each group
        encoder_bottom_params = self.get_encoder_bottom_params()
        encoder_top_params = self.get_encoder_top_params()
        decoder_params = self.get_decoder_params()
        phase_head_params = self.get_phase_head_params()
        amp_head_params = self.get_amp_head_params()

        # Check if any are trainable
        encoder_bottom_trainable = any(p.requires_grad for p in encoder_bottom_params) if encoder_bottom_params else False
        encoder_top_trainable = any(p.requires_grad for p in encoder_top_params) if encoder_top_params else False
        decoder_trainable = any(p.requires_grad for p in decoder_params) if decoder_params else False
        phase_head_trainable = any(p.requires_grad for p in phase_head_params) if phase_head_params else False
        amp_head_trainable = any(p.requires_grad for p in amp_head_params) if amp_head_params else False

        # Count parameters
        def count_params(param_list):
            return sum(p.numel() for p in param_list)

        print(f"Encoder (bottom 50%): {'Trainable' if encoder_bottom_trainable else 'Frozen'} "
            f"({count_params(encoder_bottom_params):,} params)")
        print(f"Encoder (top 50%):    {'Trainable' if encoder_top_trainable else 'Frozen'} "
            f"({count_params(encoder_top_params):,} params)")
        print(f"Decoder (base):       {'Trainable' if decoder_trainable else 'Frozen'} "
            f"({count_params(decoder_params):,} params)")
        print(f"Phase head:           {'Trainable' if phase_head_trainable else 'Frozen'} "
            f"({count_params(phase_head_params):,} params)")
        print(f"Amp head:             {'Trainable' if amp_head_trainable else 'Frozen'} "
            f"({count_params(amp_head_params):,} params)")

        total_params = count_params(list(self.parameters()))
        trainable_params = count_params([p for p in self.parameters() if p.requires_grad])
        print(f"\nTotal: {trainable_params:,} / {total_params:,} trainable ({100*trainable_params/total_params:.1f}%)")
        print("="*60 + "\n")

    def verify_parameter_grouping(self):
        """
        Verification helper: Check that parameter grouping is correct.
        Ensures no parameters are missed or duplicated.
        """
        print("\n" + "="*60)
        print("Parameter Grouping Verification:")
        print("="*60)

        # Get all parameter groups
        encoder_bottom = set(id(p) for p in self.get_encoder_bottom_params())
        encoder_top = set(id(p) for p in self.get_encoder_top_params())
        decoder = set(id(p) for p in self.get_decoder_params())
        phase_head = set(id(p) for p in self.get_phase_head_params())
        amp_head = set(id(p) for p in self.get_amp_head_params())

        # Get all model parameters (excluding forward_model parameters like alpha, beta)
        # We only care about autoencoder parameters for grouping
        autoencoder_params = set(id(p) for p in self.autoencoder.parameters())

        # Union of all groups
        all_grouped_params = encoder_bottom | encoder_top | decoder | phase_head | amp_head

        # Check for overlaps
        overlaps = []
        groups = {
            'encoder_bottom': encoder_bottom,
            'encoder_top': encoder_top,
            'decoder': decoder,
            'phase_head': phase_head,
            'amp_head': amp_head
        }

        for name1, group1 in groups.items():
            for name2, group2 in groups.items():
                if name1 < name2:  # Avoid duplicate checks
                    overlap = group1 & group2
                    if overlap:
                        overlaps.append(f"{name1} & {name2}: {len(overlap)} params")

        # Check for missing parameters (only in autoencoder)
        missing = autoencoder_params - all_grouped_params
        extra = all_grouped_params - autoencoder_params

        print(f"Total autoencoder parameters: {len(autoencoder_params)}")
        print(f"Total grouped parameters: {len(all_grouped_params)}")
        print(f"Missing from groups: {len(missing)}")
        print(f"Extra in groups (shouldn't happen): {len(extra)}")

        if overlaps:
            print(f"\nWarning: Parameter overlaps detected:")
            for overlap in overlaps:
                print(f"  {overlap}")
        else:
            print(f"\nNo parameter overlaps")

        if missing:
            print(f"\nWarning: {len(missing)} autoencoder parameters not in any group")
            # Try to identify what's missing
            print("  Attempting to identify missing parameters...")
            for name, param in self.autoencoder.named_parameters():
                if id(param) in missing:
                    print(f"    - {name}")
        else:
            print(f"\nAll autoencoder parameters accounted for")

        if extra:
            print(f"\nWarning: {len(extra)} grouped parameters not in autoencoder")

        print("="*60 + "\n")

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
        shared = getattr(model_config, 'use_shared_decoder', False)
        self.autoencoder = Autoencoder(model_config, data_config, shared_decoder=shared)
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
    Lightning module equivalent of PtychoPINN module from ptycho_torch.model (UNet variant).
    Uses RMS normalization with cosine LR scheduling.
    '''
    def __init__(self, model_config: ModelConfig,
                       data_config: DataConfig,
                       training_config: TrainingConfig,
                       inference_config: InferenceConfig):
        from torchmetrics import MeanMetric
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
        self.validation_step_outputs = []
        self._fine_tuning_mode = False

        #LR scheduler
        if training_config.scheduler == 'Cosine':
            self.warmup_epochs = getattr(training_config, 'warmup_epochs', 5)
            self.min_lr_ratio = getattr(training_config, 'min_lr_ratio', 0.01)

        #DDP LR. Makes it so we have to manually update the model gradients.
        self.automatic_optimization = False

        #Model
        if model_config.mode == 'Unsupervised':
            self.model = PtychoPINN(model_config, data_config, training_config)
        elif model_config.mode == 'Supervised':
            self.model = Ptycho_Supervised(model_config, data_config, training_config)

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

        # Probe reference loss
        if model_config.probe_reference_coeff > 0:
            self.ProbeRefLoss = ProbeReferenceLoss()
            self.loss_name += '_ProbeRef'
            self.val_loss_name += '_ProbeRef'

        self.loss_name += '_loss'
        self.val_loss_name += '_loss'

        #Saving hyperparameters
        self.save_hyperparameters()

    def forward(self, x, positions, probe, input_scale_factor, output_scale_factor, experiment_ids,
                fine_tune = False):
        x_out = self.model(x, positions, probe, input_scale_factor, output_scale_factor, experiment_ids,
                           fine_tune = fine_tune)
        return x_out

    def forward_predict(self, x, positions, probe, input_scale_factor):
        #Turns padding off if we need to
        x_combined = self.model.forward_predict(x, positions, probe, input_scale_factor)
        return x_combined

    def compute_loss(self, batch):
        """
        Loss computation with RMS normalization.
        """
        # Grab required data fields from TensorDict
        x = batch[0]['images']
        positions = batch[0]['coords_relative']
        probe = batch[1]
        rms_scale = batch[0]['rms_scaling_constant']  # RMS scaling
        physics_scale = batch[0]['physics_scaling_constant']
        experiment_ids = batch[0]['experiment_id']
        probe_scaling = batch[2]

        #If supervised, also need to get the amp/phase labels
        if self.model_config.mode == 'Supervised':
            amp_label = batch[0]['label_amp']
            phase_label = batch[0]['label_phase']

        #Calc loss
        total_loss = 0.0

        #Calculate modified output scale for rectangular intensity loss function
        modified_output_scale = torch.sqrt(1/(probe_scaling**2 * physics_scale + 1e-9))

        # Perform forward pass up and scale
        pred, real, imag = self(x, positions, probe,
                                            input_scale_factor = rms_scale,
                                            output_scale_factor = modified_output_scale,
                                            experiment_ids = experiment_ids,
                                            fine_tune = self._fine_tuning_mode
                                            )

        #Normalization factor for loss output (just to keep it scaled down)
        intensity_norm_factor = torch.mean(x).detach() + 1e-8

        if self.model_config.mode == 'Unsupervised':
            total_loss += self.Loss(pred, x).mean()
            total_loss /= intensity_norm_factor

        elif self.model_config.mode == 'Supervised':
            #Compute loss for imag and real
            real_loss = self.Loss(real, amp_label).sum()
            imag_loss = self.Loss(imag, phase_label).sum()
            #Add to total loss
            total_loss += 2 * real_loss + 4 * imag_loss

        # Add real/imag regularization losses if specified
        if self.model_config.amp_loss:
            real_reg_loss = self.AmpLoss(real).mean()
            total_loss += real_reg_loss * self.model_config.real_loss_coeff

        if self.model_config.phase_loss:
            imag_reg_loss = self.PhaseLoss(imag).mean()
            total_loss += imag_reg_loss * self.model_config.imag_loss_coeff

        # Probe reference loss -- penalize imaginary channel for transparent-object input
        if self.model_config.probe_reference_coeff > 0:
            C_in = self.model_config.C_model if self.model_config.object_big else 1
            probe_single = probe[0, 0, 0]  # Single probe from batch, shape (N, N) complex
            probe_ref_loss = self.ProbeRefLoss(
                self.model.autoencoder, probe_single, C_in,
                self.model.scaler, self.data_config
            )
            total_loss += self.model_config.probe_reference_coeff * probe_ref_loss
            self.log('probe_ref_loss', probe_ref_loss.detach(),
                     on_step=False, prog_bar=True, on_epoch=True, sync_dist=True)

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
        self.log(self.val_loss_name,
                 val_loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)

        # self.validation_step_outputs.append(val_loss.detach())

        return val_loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr = self.lr)

        result = {"optimizer": optimizer}

        if self.training_config.scheduler == 'Cosine':
            scheduler = self.build_warmup_cosine_scheduler(optimizer, total_epochs = self.trainer.max_epochs)
            result['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        elif self.training_config.scheduler == 'Exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
            result['lr_scheduler'] = scheduler

        return result

    def build_warmup_cosine_scheduler(self, optimizer, total_epochs):
        """
        Build warmup cosine annealing LR scheduler
        """
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_epochs = max(0, self.warmup_epochs)
        base_lr = optimizer.param_groups[0]['lr']
        eta_min = base_lr * self.min_lr_ratio

        #No linear ramp
        if warmup_epochs == 0:
            return CosineAnnealingLR(
                optimizer,
                T_max = total_epochs,
                eta_min = eta_min
            )

        linear_warmup = LinearLR(
            optimizer,
            start_factor = 0.1,
            end_factor = 1.0,
            total_iters = warmup_epochs
        )

        cosine = CosineAnnealingLR(
            optimizer,
            T_max = max(1, total_epochs - warmup_epochs),
            eta_min = eta_min
        )

        return SequentialLR(
            optimizer,
            schedulers = [linear_warmup, cosine],
            milestones = [warmup_epochs]
        )

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
        if self.global_rank == 0:
            print(f"\n{'='*60}")
            print(f"Starting Epoch {self.current_epoch + 1}/{self.trainer.max_epochs}")
            print(f"{'='*60}")

    def on_train_epoch_end(self):
        """
        Called at the end of each training epoch.
        Manually step LR scheduler since we use manual optimization.
        """
        # Get scheduler(s)
        sch = self.lr_schedulers()

        # Step the scheduler
        if sch is not None:
            if isinstance(sch, list):
                for scheduler in sch:
                    scheduler.step()
            else:
                sch.step()

            # Log the new LR after stepping
            current_lr = self.optimizers().param_groups[0]['lr']
            self.log('learning_rate', current_lr, on_epoch=True, sync_dist=True)

            if self.global_rank == 0:
                print(f"Epoch {self.current_epoch} complete. New LR: {current_lr:.2e}")

    def on_after_backward(self):
    # Log the norm of the gradients for the final layer of the decoder
        if self.global_step % 50 == 0:
            if getattr(self.model.autoencoder, '_shared', False):
                decoder = self.model.autoencoder.decoder
                if getattr(decoder, '_split_heads', False):
                    grad_norm = decoder.head_real.conv1.weight.grad.norm()
                else:
                    grad_norm = decoder.head.conv1.weight.grad.norm()
            else:
                grad_norm = self.model.autoencoder.decoder_amp.amp.conv1.weight.grad.norm()
            # Show in progress bar
            self.log("grad_norm", grad_norm, on_step=True, prog_bar=True, logger=True)
