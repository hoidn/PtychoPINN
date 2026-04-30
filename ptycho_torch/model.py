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
from ptycho_torch.train_utils import compute_grad_norm

logger = logging.getLogger(__name__)


def _build_optimizer(parameters, *, lr, optimizer='adam', momentum=0.9,
                     weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.999):
    """Build optimizer from string name + hyperparams.

    See: plans/active/FNO-STABILITY-OVERHAUL-001/plan_optimizer_diagnostics.md Task 1.
    """
    if optimizer == 'adam':
        return torch.optim.Adam(parameters, lr=lr, betas=(adam_beta1, adam_beta2),
                                weight_decay=weight_decay)
    elif optimizer == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr, betas=(adam_beta1, adam_beta2),
                                 weight_decay=weight_decay)
    elif optimizer == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum,
                               weight_decay=weight_decay, nesterov=(momentum > 0))
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer}'. Choose from: adam, adamw, sgd")


#Lightning
import lightning as L


#Ensuring 64float b/c of complex numbers
# torch.set_default_dtype(torch.float32)


def _real_imag_to_complex_channel_first(real_imag: torch.Tensor) -> torch.Tensor:
    """Convert real/imag tensor from (B, H, W, C, 2) to complex (B, C, H, W).

    This adapter function converts FNO/Hybrid generator outputs (which produce
    real and imaginary parts in the last dimension) to the complex channel-first
    format expected by PtychoPINN's physics pipeline.

    Args:
        real_imag: Tensor with shape (B, H, W, C, 2) where the last dimension
                   contains [real, imag] components.

    Returns:
        Complex tensor with shape (B, C, H, W) in channel-first format.

    Raises:
        ValueError: If input doesn't have 5 dimensions or last dim != 2.

    Example:
        >>> x = torch.zeros(2, 64, 64, 4, 2)  # (batch, H, W, C, real/imag)
        >>> x[..., 0] = 1.0  # Real part
        >>> out = _real_imag_to_complex_channel_first(x)
        >>> out.shape  # (2, 4, 64, 64)
        >>> out.is_complex()  # True
    """
    if real_imag.ndim != 5 or real_imag.shape[-1] != 2:
        raise ValueError(
            f"Expected real/imag tensor with shape (B, H, W, C, 2), got {tuple(real_imag.shape)}"
        )
    complex_last = torch.complex(real_imag[..., 0], real_imag[..., 1])  # (B, H, W, C)
    return complex_last.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)


def _predict_complex_patches(
    autoencoder: nn.Module,
    combine_complex: nn.Module,
    generator_output: str,
    x: torch.Tensor,
):
    """Normalize generator outputs onto the shared complex/amp/phase contract."""
    if generator_output == "amp_phase":
        amp, phase = autoencoder(x)
        x_complex = combine_complex(amp, phase)
    elif generator_output == "amp_phase_logits":
        patches = autoencoder(x)
        if patches.shape[-1] != 2:
            raise ValueError(
                f"amp_phase_logits expects last dim=2, got shape {patches.shape}"
            )
        amp_logits = patches[..., 0].permute(0, 3, 1, 2).contiguous()
        phase_logits = patches[..., 1].permute(0, 3, 1, 2).contiguous()
        amp = torch.sigmoid(amp_logits)
        phase = math.pi * torch.tanh(phase_logits)
        x_complex = combine_complex(amp, phase)
    elif generator_output == "real_imag":
        patches = autoencoder(x)
        x_complex = _real_imag_to_complex_channel_first(patches)
        amp = torch.abs(x_complex)
        phase = torch.angle(x_complex)
    else:
        raise ValueError(f"Unsupported generator_output='{generator_output}'")
    return x_complex, amp, phase


def _generator_output_mode_for_core(generator_output: str) -> str:
    """Map Lightning output contract onto generator-core output mode."""
    return "amp_phase" if generator_output == "amp_phase" else "real_imag"


def _build_generator_module_from_config(
    model_config: ModelConfig,
    data_config: DataConfig,
    *,
    generator_output: str,
) -> Optional[nn.Module]:
    """Rebuild a registered generator core from saved config state."""
    architecture = getattr(model_config, "architecture", "cnn")
    if architecture == "cnn":
        return None

    generator_mode = _generator_output_mode_for_core(generator_output)
    common_kwargs = {
        "in_channels": 1,
        "out_channels": 2,
        "hidden_channels": getattr(model_config, "fno_width", 32),
        "modes": getattr(model_config, "fno_modes", 12),
        "C": getattr(data_config, "C", 4),
        "input_transform": getattr(model_config, "fno_input_transform", "none"),
        "output_mode": generator_mode,
    }

    if architecture == "ffno":
        from ptycho_torch.generators.ffno import FfnoGeneratorModule

        return FfnoGeneratorModule(
            **common_kwargs,
            n_blocks=getattr(model_config, "fno_blocks", 4),
            cnn_blocks=getattr(model_config, "fno_cnn_blocks", 2),
        )

    if architecture == "fno":
        from ptycho_torch.generators.fno import CascadedFNOGenerator

        return CascadedFNOGenerator(
            **common_kwargs,
            fno_blocks=getattr(model_config, "fno_blocks", 4),
            cnn_blocks=getattr(model_config, "fno_cnn_blocks", 2),
        )

    if architecture == "hybrid":
        from ptycho_torch.generators.fno import HybridUNOGenerator

        return HybridUNOGenerator(
            **common_kwargs,
            n_blocks=getattr(model_config, "fno_blocks", 4),
            max_hidden_channels=getattr(model_config, "max_hidden_channels", None),
        )

    if architecture == "stable_hybrid":
        from ptycho_torch.generators.fno import StableHybridUNOGenerator

        return StableHybridUNOGenerator(
            **common_kwargs,
            n_blocks=getattr(model_config, "fno_blocks", 4),
            max_hidden_channels=getattr(model_config, "max_hidden_channels", None),
        )

    if architecture == "fno_vanilla":
        from ptycho_torch.generators.fno_vanilla import FnoVanillaGeneratorModule

        return FnoVanillaGeneratorModule(
            **common_kwargs,
            n_blocks=getattr(model_config, "fno_blocks", 4),
        )

    if architecture == "neuralop_uno":
        from ptycho_torch.generators.neuralop_uno import NeuralopUnoGeneratorModule

        if int(getattr(data_config, "N", 128)) != 128:
            raise ValueError(
                "neuralop_uno checkpoint rebuild only supports the locked Lines128 "
                f"CDI contract (N=128); got N={getattr(data_config, 'N', None)}."
            )
        if tuple(getattr(data_config, "grid_size", (1, 1))) != (1, 1):
            raise ValueError(
                "neuralop_uno checkpoint rebuild only supports the locked "
                f"gridsize=1 CDI contract; got grid_size={getattr(data_config, 'grid_size', None)}."
            )
        return NeuralopUnoGeneratorModule(
            C=getattr(data_config, "C", 1),
            output_mode=generator_mode,
        )

    if architecture == "hybrid_resnet":
        from ptycho_torch.generators.hybrid_resnet import HybridResnetGeneratorModule

        return HybridResnetGeneratorModule(
            **common_kwargs,
            n_blocks=getattr(model_config, "fno_blocks", 4),
            max_hidden_channels=getattr(model_config, "max_hidden_channels", None),
            resnet_width=getattr(model_config, "resnet_width", None),
            resnet_blocks=getattr(model_config, "hybrid_resnet_blocks", 6),
            skip_connections=getattr(model_config, "hybrid_skip_connections", False),
            hybrid_downsample_steps=getattr(model_config, "hybrid_downsample_steps", 2),
            hybrid_downsample_op=getattr(model_config, "hybrid_downsample_op", "stride_conv"),
            hybrid_encoder_conv_hidden_scale=getattr(
                model_config,
                "hybrid_encoder_conv_hidden_scale",
                1.0,
            ),
            hybrid_encoder_spectral_hidden_scale=getattr(
                model_config,
                "hybrid_encoder_spectral_hidden_scale",
                1.0,
            ),
            hybrid_encoder_conv_hidden_channels=getattr(
                model_config,
                "hybrid_encoder_conv_hidden_channels",
                None,
            ),
            hybrid_encoder_spectral_hidden_channels=getattr(
                model_config,
                "hybrid_encoder_spectral_hidden_channels",
                None,
            ),
            hybrid_skip_style=getattr(model_config, "hybrid_skip_style", "add"),
        )

    if architecture == "spectral_resnet_bottleneck_net":
        from ptycho_torch.generators.spectral_resnet_bottleneck import (
            SpectralResnetBottleneckGeneratorModule,
        )

        return SpectralResnetBottleneckGeneratorModule(
            **common_kwargs,
            n_blocks=getattr(model_config, "fno_blocks", 4),
            max_hidden_channels=getattr(model_config, "max_hidden_channels", None),
            resnet_width=getattr(model_config, "resnet_width", None),
            resnet_blocks=getattr(model_config, "hybrid_resnet_blocks", 6),
            hybrid_downsample_steps=getattr(model_config, "hybrid_downsample_steps", 2),
            hybrid_downsample_op=getattr(model_config, "hybrid_downsample_op", "stride_conv"),
            hybrid_encoder_conv_hidden_scale=getattr(
                model_config,
                "hybrid_encoder_conv_hidden_scale",
                1.0,
            ),
            hybrid_encoder_spectral_hidden_scale=getattr(
                model_config,
                "hybrid_encoder_spectral_hidden_scale",
                1.0,
            ),
            hybrid_encoder_conv_hidden_channels=getattr(
                model_config,
                "hybrid_encoder_conv_hidden_channels",
                None,
            ),
            hybrid_encoder_spectral_hidden_channels=getattr(
                model_config,
                "hybrid_encoder_spectral_hidden_channels",
                None,
            ),
            spectral_bottleneck_blocks=getattr(model_config, "spectral_bottleneck_blocks", 6),
            spectral_bottleneck_modes=getattr(model_config, "spectral_bottleneck_modes", 12),
            spectral_bottleneck_share_weights=getattr(
                model_config,
                "spectral_bottleneck_share_weights",
                True,
            ),
            spectral_bottleneck_gate_init=getattr(
                model_config,
                "spectral_bottleneck_gate_init",
                0.1,
            ),
            spectral_bottleneck_gate_mode=getattr(
                model_config,
                "spectral_bottleneck_gate_mode",
                "shared",
            ),
        )

    raise ValueError(
        f"Unsupported generator architecture '{architecture}' for checkpoint rebuild."
    )


def _resolve_generator_from_config(
    model_config: ModelConfig,
    data_config: DataConfig,
    generator: Optional[nn.Module],
    generator_output: str,
) -> tuple[Optional[nn.Module], str]:
    """Resolve generator module/output contract from config plus optional injection."""
    architecture = getattr(model_config, "architecture", "cnn")
    configured_output_mode = getattr(model_config, "generator_output_mode", None)
    resolved_output = generator_output
    if architecture != "cnn" and configured_output_mode:
        resolved_output = configured_output_mode
    if generator is None and architecture != "cnn":
        generator = _build_generator_module_from_config(
            model_config,
            data_config,
            generator_output=resolved_output,
        )
    return generator, resolved_output


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

        if x2.shape[-2:] != x1.shape[-2:]:
            x2 = x2[..., :x1.shape[-2], :x1.shape[-1]]

        outputs = x1 + x2


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
        self.probe_mask = getattr(self.model_config, "probe_mask", False)
        self.probe_mask_tensor = getattr(self.model_config, "probe_mask_tensor", None)
        self.probe_mask_sigma = float(getattr(self.model_config, "probe_mask_sigma", 1.0))
        self.probe_mask_diameter = getattr(self.model_config, "probe_mask_diameter", None)
        self.register_buffer("_cached_probe_mask", torch.empty(0), persistent=False)
        self._cached_mask_key = None

    def _resolve_probe_mask(self, x: torch.Tensor) -> torch.Tensor:
        from ptycho_torch.probe_mask import resolve_probe_mask_torch

        explicit_mask = self.probe_mask_tensor
        if explicit_mask is None and self.probe_mask is not None and not isinstance(self.probe_mask, bool):
            explicit_mask = self.probe_mask

        cacheable = explicit_mask is None
        cache_key = (x.device.type, str(x.dtype))
        if cacheable and self._cached_mask_key == cache_key and self._cached_probe_mask.numel() == self.N * self.N:
            return self._cached_probe_mask

        mask = resolve_probe_mask_torch(
            self.N,
            probe_mask=self.probe_mask,
            probe_mask_tensor=explicit_mask,
            probe_mask_sigma=self.probe_mask_sigma,
            probe_mask_diameter=self.probe_mask_diameter,
            dtype=x.real.dtype if x.is_complex() else x.dtype,
            device=x.device,
        )
        if cacheable:
            self._cached_probe_mask = mask
            self._cached_mask_key = cache_key
        return mask

    def forward(self, x, probe):

        #Add extra dimension to x
        x_reshaped = x.unsqueeze(dim=2) # (B,C,H,W) -> (B,C,1,H,W)

        # print('probe shape', probe.shape)
        # print('xreshaped shape', x_reshaped.shape)

        probe_mask = self._resolve_probe_mask(x)
        
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

    CRITICAL: Both predictions and observations must be converted from amplitudes to
    intensities (squared) before computing Poisson log-likelihood, to match TensorFlow
    behavior (ptycho/model.py:497-511) and satisfy Poisson distribution support constraints.
    '''
    def __init__(self, amplitudes):

        super(PoissonIntensityLayer, self).__init__()
        # Poisson rate parameter (lambda) - square predicted amplitudes to get intensities
        Lambda = amplitudes ** 2
        # Create Poisson distribution with validate_args=False to accept float observations
        # (TensorFlow's tf.nn.log_poisson_loss also accepts floats, not just integers)
        # Second parameter (batch size) controls how many dimensions are summed over starting from the last
        self.poisson_dist = dist.Independent(dist.Poisson(Lambda, validate_args=False), 3)
        

    def forward(self, x):
        '''
        Compute Poisson negative log-likelihood.

        Args:
            x: Observed diffraction amplitudes (NOT intensities)

        Returns:
            Negative log-likelihood

        CRITICAL FIX (ADR-003-BACKEND-API Phase C4.D3):
        The input x contains amplitude values (sqrt of intensity), but Poisson
        distribution expects photon counts (intensities). We must square x before
        computing log_prob to match TensorFlow behavior and satisfy the Poisson
        distribution's IntegerGreaterThan(0) support constraint.

        TensorFlow reference: ptycho/model.py:506-511 (negloglik function)
        - Both y_true and y_pred are squared before Poisson loss computation
        '''
        # Apply poisson distribution to intensities
        observed_intensity = x ** 2
        return -self.poisson_dist.log_prob(observed_intensity)
    
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
        self._reassembly_logged = False

        #Fitting parameters for scaling
        self.alpha = nn.Parameter(torch.ones(model_config.num_datasets))
        self.beta = nn.Parameter(torch.ones(model_config.num_datasets))

    def forward(self, x, positions, probe, output_scale_factor, experiment_ids = None):
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

        #Learnable scaling parameter test (different per dataset)

        if self.model_config.intensity_scale_trainable:
            alphas = self.alpha[experiment_ids]
            betas = self.beta[experiment_ids]

            if len(x.shape) == 3:
                alphas = alphas.view(-1,1,1)
                betas = betas.view(-1,1,1)
            elif len(x.shape) == 4:
                alphas = alphas.view(-1,1,1,1)
                betas = betas.view(-1,1,1,1)

            pred_scaled_diffraction = alphas * pred_scaled_diffraction + betas
        
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
        # MAE operates on amplitude (match TF mean_absolute_error)
        loss_mae = self.mae(pred, raw)

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
    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        training_config: TrainingConfig,
        generator: Optional[nn.Module] = None,
        generator_output: str = "amp_phase"
    ):
        super(PtychoPINN, self).__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config # Store training config
        resolved_generator, resolved_generator_output = _resolve_generator_from_config(
            model_config,
            data_config,
            generator,
            generator_output,
        )

        self.generator = resolved_generator
        self.generator_output = resolved_generator_output

        self.n_filters_scale = self.model_config.n_filters_scale

        #Scaler - Pass config
        self.scaler = IntensityScalerModule(model_config)
        self.probe_scale = data_config.probe_scale

        #Autoencoder or custom generator
        # When generator is None, use default CNN-based Autoencoder
        # When generator is provided (e.g., FNO/Hybrid), use it directly
        self.autoencoder = Autoencoder(model_config, data_config) if resolved_generator is None else resolved_generator
        self.combine_complex = CombineComplex()

        #Adding named modules for forward operation
        self.forward_model = ForwardModel(model_config, data_config)

    def _predict_complex(self, x):
        """Generate complex object patches from input.

        Handles amp_phase output (from CNN autoencoder), real_imag output
        (from FNO/Hybrid generators), and amp_phase_logits output.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Tuple of (x_complex, amp, phase) tensors.
        """
        return _predict_complex_patches(
            self.autoencoder,
            self.combine_complex,
            self.generator_output,
            x,
        )

    def forward(self, x, positions, probe, input_scale_factor, output_scale_factor, experiment_ids = None):

        #Scaling down (normalizing to 1)
        x = self.scaler.scale(x, input_scale_factor)

        #Predict complex object via autoencoder or generator
        x_combined, x_amp, x_phase = self._predict_complex(x)

        #Run through forward model. Unscaled diffraction pattern
        x_out = self.forward_model.forward(x_combined, positions,
                                           probe/self.probe_scale, output_scale_factor, experiment_ids)

        return x_out, x_amp, x_phase

    def forward_predict(self, x, positions, probe, input_scale_factor):
        #Scaling
        x = self.scaler.scale(x, input_scale_factor)

        #Predict complex object via autoencoder or generator
        x_combined, _, _ = self._predict_complex(x)

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
    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        training_config: TrainingConfig,
        generator: Optional[nn.Module] = None,
        generator_output: str = "amp_phase",
    ):
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config # Store training config

        self.n_filters_scale = self.model_config.n_filters_scale

        self.generator, self.generator_output = _resolve_generator_from_config(
            model_config,
            data_config,
            generator,
            generator_output,
        )
        self.autoencoder = (
            Autoencoder(model_config, data_config)
            if self.generator is None
            else self.generator
        )
        self.combine_complex = CombineComplex()

        #Scaler - Pass config
        self.scaler = IntensityScalerModule(model_config)

    def _predict_complex(self, x):
        return _predict_complex_patches(
            self.autoencoder,
            self.combine_complex,
            self.generator_output,
            x,
        )

    def forward(
        self,
        x,
        positions,
        probe,
        input_scale_factor,
        output_scaling_factor,
        experiment_ids=None,
    ):
        # Supervised models ignore experiment ids, but the Lightning wrapper
        # always threads them through the shared forward contract.
        del positions, probe, output_scaling_factor, experiment_ids

        #Scaling
        x = self.scaler.scale(x, input_scale_factor)
        x_combined, x_amp, x_phase = self._predict_complex(x)

        return x_combined, x_amp, x_phase
    
    def forward_predict(self, x, positions, probe, input_scale_factor):
        """
        Identical to forward specifically for Ptycho_Supervised.
        Kept to keep consistency with the forward_predict method in PtychoPINN.
        """
        #Scaling
        x = self.scaler.scale(x, input_scale_factor)
        x_combined, _, _ = self._predict_complex(x)

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
    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        training_config: TrainingConfig,
        inference_config: InferenceConfig,
        generator_module: Optional[nn.Module] = None,
        generator_output: str = "amp_phase"
    ):
        super().__init__()

        # Handle checkpoint loading: convert dict kwargs back to dataclass instances
        # (Lightning passes saved hyperparameters as dicts during load_from_checkpoint)
        if isinstance(model_config, dict):
            model_config = ModelConfig(**model_config)
        if isinstance(data_config, dict):
            data_config = DataConfig(**data_config)
        if isinstance(training_config, dict):
            training_config = TrainingConfig(**training_config)
        if isinstance(inference_config, dict):
            inference_config = InferenceConfig(**inference_config)

        generator_module, generator_output = _resolve_generator_from_config(
            model_config,
            data_config,
            generator_module,
            generator_output,
        )

        # Save hyperparameters for checkpoint serialization (Phase D1c requirement)
        # Convert dataclass instances to dicts to ensure serializability
        # Note: generator_module is not saved (reconstructed at load time)
        from dataclasses import asdict
        self.save_hyperparameters({
            'model_config': asdict(model_config),
            'data_config': asdict(data_config),
            'training_config': asdict(training_config),
            'inference_config': asdict(inference_config),
            'generator_output': generator_output,
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
        self.torch_mae_pred_l2_match_target = bool(
            getattr(self.training_config, "torch_mae_pred_l2_match_target", False)
        )
        self._last_mae_alpha_stats = None

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
            self.model = PtychoPINN(
                model_config,
                data_config,
                training_config,
                generator=generator_module,
                generator_output=generator_output,
            )
        elif model_config.mode == 'Supervised':
            self.model = Ptycho_Supervised(
                model_config,
                data_config,
                training_config,
                generator=generator_module,
                generator_output=generator_output,
            )

        # Enforce single-stage training (legacy stage_* knobs are ignored)
        self.total_epochs = training_config.epochs
        if getattr(training_config, 'stage_2_epochs', 0) or getattr(training_config, 'stage_3_epochs', 0):
            logger.warning(
                "Multi-stage scheduler settings are ignored. "
                "torch_loss_mode enforces single-stage training."
            )
        self.stage_1_epochs = self.total_epochs
        self.stage_2_epochs = 0
        self.stage_3_epochs = 0

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
    
    def forward(self, x, positions, probe, input_scale_factor, output_scale_factor, experiment_ids):
        x_out = self.model(x, positions, probe, input_scale_factor, output_scale_factor, experiment_ids)
        return x_out
    
    def forward_predict(self, x, positions, probe, input_scale_factor):
        #Turns padding off if we need to
        x_combined = self.model.forward_predict(x, positions, probe, input_scale_factor)
        return x_combined
    
    def _reshape_scale_tensor(self, scale_value, reference_tensor):
        """
        Convert scalar or 1D scaling factors into broadcastable tensors on the correct device/dtype.
        """
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

    def _match_prediction_l2_to_target(self, pred_amp: torch.Tensor, target_amp: torch.Tensor, eps: float = 1e-8):
        """Scale each prediction sample so L2 energy matches the corresponding target sample."""
        if pred_amp.ndim != target_amp.ndim:
            raise ValueError(
                f"pred/target ndim mismatch for L2 matching: {pred_amp.ndim} vs {target_amp.ndim}"
            )
        reduce_dims = tuple(range(1, pred_amp.ndim))
        pred_energy = torch.sum(pred_amp * pred_amp, dim=reduce_dims, keepdim=True)
        target_energy = torch.sum(target_amp * target_amp, dim=reduce_dims, keepdim=True)
        alpha = torch.sqrt(target_energy / (pred_energy + eps))
        pred_matched = pred_amp * alpha
        return pred_matched, alpha

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
        experiment_ids = batch[0]['experiment_id']
        scale = batch[2]
        # old_scaling = batch[2]
        physics_weight = 0
    
        
        #If supervised, also need to get the amp/phase labels
        if self.model_config.mode == 'Supervised':
            amp_label = batch[0]['label_amp']
            phase_label = batch[0]['label_phase']

        #Calc loss
        total_loss = 0.0
        self._last_mae_alpha_stats = None

        # Perform forward pass up and scale
        pred, amp, phase = self(x, positions, probe,
                                            input_scale_factor = rms_scale,
                                            output_scale_factor = rms_scale,
                                            experiment_ids = experiment_ids
                                            )
        
        #Normalization factor for loss output (just to keep it scaled down)
        intensity_norm_factor = torch.mean(x).detach() + 1e-8

        if self.model_config.mode == 'Unsupervised':
            pred_physics = pred * physics_scale
            obs_physics = x * physics_scale
            if self.torch_loss_mode == 'mae' and self.torch_mae_pred_l2_match_target:
                pred_physics, alpha = self._match_prediction_l2_to_target(pred_physics, obs_physics)
                alpha_flat = alpha.detach().reshape(alpha.shape[0], -1)
                self._last_mae_alpha_stats = {
                    "mean": alpha_flat.mean(),
                    "std": alpha_flat.std(unbiased=False),
                    "min": alpha_flat.min(),
                    "max": alpha_flat.max(),
                }
            else:
                self._last_mae_alpha_stats = None
            total_loss += self.Loss(pred_physics, obs_physics).mean()
            total_loss /= intensity_norm_factor

        elif self.model_config.mode == 'Supervised':
            #Compute loss for phase and amp
            amp_loss = self.Loss(amp, amp_label).sum()
            phase_loss = self.Loss(phase, phase_label).sum()
            #Add to total loss
            total_loss += 2 * amp_loss + 4 * phase_loss
        
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

        if self.training_config.log_grad_norm and (batch_idx % self.training_config.grad_norm_log_freq == 0):
            pre_clip = compute_grad_norm(self.parameters(), norm_type=2.0)
            self.log("grad_norm_preclip", pre_clip, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        #Step every N batches
        if (batch_idx+1) % self.accum_steps == 0:
            #Clip gradients
            if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
                algo = getattr(self.training_config, 'gradient_clip_algorithm', 'norm')
                if algo == 'norm':
                    torch.nn.utils.clip_grad_norm_(
                        parameters=self.parameters(),
                        max_norm=self.gradient_clip_val,
                        norm_type=2.0,
                    )
                elif algo == 'value':
                    torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)
                elif algo == 'agc':
                    from ptycho_torch.train_utils import adaptive_gradient_clip_
                    adaptive_gradient_clip_(self.parameters(), clip_factor=self.gradient_clip_val)
                if self.training_config.log_grad_norm and (batch_idx % self.training_config.grad_norm_log_freq == 0):
                    post_clip = compute_grad_norm(self.parameters(), norm_type=2.0)
                    self.log("grad_norm_postclip", post_clip, on_step=True, on_epoch=True, prog_bar=False, logger=True)

            opt.step()
            opt.zero_grad()

        #Logging
        self.log(self.loss_name, loss, on_epoch = True, prog_bar=True, logger=True, sync_dist=True)
        if self._last_mae_alpha_stats is not None:
            self.log("mae_pred_l2_alpha_mean_train", self._last_mae_alpha_stats["mean"], on_epoch=True, logger=True, sync_dist=True)
            self.log("mae_pred_l2_alpha_std_train", self._last_mae_alpha_stats["std"], on_epoch=True, logger=True, sync_dist=True)
            self.log("mae_pred_l2_alpha_min_train", self._last_mae_alpha_stats["min"], on_epoch=True, logger=True, sync_dist=True)
            self.log("mae_pred_l2_alpha_max_train", self._last_mae_alpha_stats["max"], on_epoch=True, logger=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step - computes validation loss without gradient updates
        Uses the same multi-stage approach as training
        """
        val_loss = self.compute_loss(batch)
        
        # Log validation loss
        self.log(self.val_loss_name, val_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if self._last_mae_alpha_stats is not None:
            self.log("mae_pred_l2_alpha_mean_val", self._last_mae_alpha_stats["mean"], on_epoch=True, logger=True, sync_dist=True)
            self.log("mae_pred_l2_alpha_std_val", self._last_mae_alpha_stats["std"], on_epoch=True, logger=True, sync_dist=True)
            self.log("mae_pred_l2_alpha_min_val", self._last_mae_alpha_stats["min"], on_epoch=True, logger=True, sync_dist=True)
            self.log("mae_pred_l2_alpha_max_val", self._last_mae_alpha_stats["max"], on_epoch=True, logger=True, sync_dist=True)

        return val_loss

    def on_validation_epoch_end(self):
        """Step ReduceLROnPlateau scheduler manually (automatic_optimization=False
        skips Lightning's built-in scheduler stepping)."""
        sch = self.lr_schedulers()
        if sch is None:
            return
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        if isinstance(sch, ReduceLROnPlateau):
            val_metric = self.trainer.callback_metrics.get(self.val_loss_name)
            if val_metric is not None:
                sch.step(val_metric)
                self.log('learning_rate', sch.optimizer.param_groups[0]['lr'],
                         on_epoch=True, prog_bar=False, logger=True)

    def configure_optimizers(self):
        optimizer = _build_optimizer(
            self.parameters(),
            lr=self.lr,
            optimizer=getattr(self.training_config, 'optimizer', 'adam'),
            momentum=getattr(self.training_config, 'momentum', 0.9),
            weight_decay=getattr(self.training_config, 'weight_decay', 0.0),
            adam_beta1=getattr(self.training_config, 'adam_beta1', 0.9),
            adam_beta2=getattr(self.training_config, 'adam_beta2', 0.999),
        )

        result = {"optimizer": optimizer}
        
        # Configure scheduler based on training type
        scheduler_choice = getattr(self.training_config, 'scheduler', 'Default')
        if scheduler_choice == 'Exponential':
            result['lr_scheduler'] = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif scheduler_choice == 'WarmupCosine':
            from ptycho_torch.schedulers import build_warmup_cosine_scheduler
            scheduler = build_warmup_cosine_scheduler(
                optimizer,
                total_epochs=self.training_config.epochs,
                warmup_epochs=getattr(self.training_config, 'lr_warmup_epochs', 0),
                min_lr_ratio=getattr(self.training_config, 'lr_min_ratio', 0.1),
            )
            result['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        elif scheduler_choice == 'ReduceLROnPlateau':
            result['lr_scheduler'] = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=getattr(self.training_config, 'plateau_factor', 0.5),
                    patience=getattr(self.training_config, 'plateau_patience', 2),
                    min_lr=getattr(self.training_config, 'plateau_min_lr', 5e-5),
                    threshold=getattr(self.training_config, 'plateau_threshold', 0.0),
                ),
                'monitor': self.val_loss_name,
                'reduce_on_plateau': True,
                'interval': 'epoch',
                'frequency': 1,
            }
        elif scheduler_choice in ('MultiStage', 'Adaptive'):
            logger.warning(
                "Scheduler '%s' is no longer supported in single-loss mode. "
                "Falling back to constant learning rate.",
                scheduler_choice,
            )

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
        # Log current learning rate for monitoring
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_epoch=True)
