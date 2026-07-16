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
from typing import Any, Dict, Optional

#Helper
from ptycho.reconstruction_policy import resolve_training_assembly_spec
from ptycho_torch.config_params import ModelConfig, TrainingConfig, DataConfig, InferenceConfig, update_existing_config
from ptycho_torch.scaling_contract import (
    CI_SCALE_CONTRACT,
    normalize_ci_poisson_per_sample,
    validate_scale_contract,
)
import ptycho_torch.helper as hh
from ptycho_torch.model_attention import CBAM, ECALayer, BasicSpatialAttention
from ptycho_torch.train_utils import compute_grad_norm

logger = logging.getLogger(__name__)


def _require_matching_component_shapes(
    branch1: torch.Tensor,
    branch2: torch.Tensor,
    generator_output: str,
) -> None:
    if branch1.shape != branch2.shape:
        raise ValueError(
            f"{generator_output} tuple branches must have matching shapes before "
            f"complex combination, got {tuple(branch1.shape)} and {tuple(branch2.shape)}"
        )


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
        _require_matching_component_shapes(amp, phase, generator_output)
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
        if isinstance(patches, (tuple, list)):
            # CNN real_imag head (Task 2.3 / B1): channel-first (real, imag) tensors
            # each (B, C, H, W). Component broadcasting is forbidden.
            real, imag = patches
            _require_matching_component_shapes(real, imag, generator_output)
            x_complex = torch.complex(real, imag)
        else:
            # FNO/Hybrid tensor path (B, H, W, C, 2) -- byte-identical, untouched.
            x_complex = _real_imag_to_complex_channel_first(patches)
        amp = torch.abs(x_complex)
        phase = torch.angle(x_complex)
    else:
        raise ValueError(f"Unsupported generator_output='{generator_output}'")
    return x_complex, amp, phase


def _generator_output_mode_for_core(generator_output: str) -> str:
    """Map Lightning output contract onto generator-core output mode."""
    return "amp_phase" if generator_output == "amp_phase" else "real_imag"


def _effective_cnn_output_mode(model_config: ModelConfig) -> str:
    """Resolve the CNN Autoencoder's effective output parameterization (Task 2.3 / B1).

    Gates ``ModelConfig.cnn_output_mode`` down to the cases where it actually takes
    effect so a single predicate governs BOTH the output contract
    (``_resolve_generator_from_config``) and the decoder-head activations
    (``Decoder_amp``/``Decoder_phase``), keeping the two in lockstep.

    Returns 'real_imag' only for the default CNN architecture in Unsupervised mode
    with ``cnn_output_mode='real_imag'``. Amendment #4: real_imag is UNSUPERVISED-ONLY
    -- Supervised mode always resolves to 'amp_phase' regardless of the knob, so the
    supervised path (and its output) is unaffected. Non-CNN architectures also resolve
    to 'amp_phase' here (their contract is set by ``generator_output_mode``).
    """
    if getattr(model_config, "architecture", "cnn") != "cnn":
        return "amp_phase"
    if getattr(model_config, "cnn_output_mode", "amp_phase") != "real_imag":
        return "amp_phase"
    if getattr(model_config, "mode", "Unsupervised") != "Unsupervised":
        return "amp_phase"
    return "real_imag"


def _semantic_component_channels(model_config: ModelConfig) -> int:
    return int(model_config.C_model) if model_config.object_big else 1


def _decoder_component_channels(model_config: ModelConfig) -> int:
    semantic_channels = _semantic_component_channels(model_config)
    if not model_config.object_big or not getattr(
        model_config,
        "use_legacy_decoder_channel_override",
        False,
    ):
        return semantic_channels

    legacy_channels = int(model_config.decoder_last_amp_channels)
    if legacy_channels not in {1, semantic_channels}:
        raise ValueError(
            "decoder_last_amp_channels must be 1 or C_model when "
            "use_legacy_decoder_channel_override is enabled"
        )
    return legacy_channels


def _build_generator_module_from_config(
    model_config: ModelConfig,
    data_config: DataConfig,
    *,
    generator_output: str,
    generator_overrides: Optional[Dict[str, Any]] = None,
) -> Optional[nn.Module]:
    """Rebuild a registered generator core from saved config state."""
    architecture = getattr(model_config, "architecture", "cnn")
    if architecture == "cnn":
        return None

    generator_mode = generator_output
    if architecture != "neuralop_uno":
        generator_mode = _generator_output_mode_for_core(generator_output)
    common_kwargs = {
        "in_channels": getattr(model_config, "learned_input_channels", 1),
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

    raise ValueError(
        f"Unsupported generator architecture '{architecture}' for checkpoint rebuild."
    )


def _resolve_generator_from_config(
    model_config: ModelConfig,
    data_config: DataConfig,
    generator: Optional[nn.Module],
    generator_output: str,
    generator_overrides: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[nn.Module], str]:
    """Resolve generator module/output contract from config plus optional injection."""
    architecture = getattr(model_config, "architecture", "cnn")
    configured_output_mode = getattr(model_config, "generator_output_mode", None)
    resolved_output = generator_output
    if architecture != "cnn" and configured_output_mode:
        resolved_output = configured_output_mode
    elif architecture == "cnn" and _effective_cnn_output_mode(model_config) == "real_imag":
        # Task 2.3 / B1: opt-in CNN real_imag contract (Unsupervised only). The CNN
        # Autoencoder emits a (real, imag) tuple that _predict_complex_patches combines
        # via torch.complex; do NOT reuse generator_output_mode (its 'real_imag' default
        # would silently flip the CNN default).
        resolved_output = "real_imag"
    if generator is None and architecture != "cnn":
        generator = _build_generator_module_from_config(
            model_config,
            data_config,
            generator_output=resolved_output,
            generator_overrides=generator_overrides,
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

class ScaledTanh(nn.Module):
    '''Picklable replacement for activations of the form `offset + scale * tanh(x)`.

    Ported verbatim from main (`git show main:ptycho_torch/model.py`; introduced by
    main commit 6a3ba10c which hardwired these bounds in place). Used ONLY by the
    Task 2.3 / B1 CNN `cnn_output_mode='real_imag'` heads (Amendment #11): the amp
    branch becomes real via `tanh + 0.2` (range (-0.8, 1.2)) and the phase branch
    becomes imag via `1.2 * tanh` (range (-1.2, 1.2)). These bounds are the hard
    representability box documented on ModelConfig.cnn_output_mode (Amendment #13).
    The default 'amp_phase' mode does NOT use this activation.
    '''
    def __init__(self, scale: float = 1.0, offset: float = 0.0):
        super().__init__()
        self.scale = scale
        self.offset = offset

    def forward(self, x):
        return self.offset + self.scale * torch.tanh(x)

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
        
        # Normal heads reserve one latent feature per semantic output component,
        # matching the reference decoder's component-wise outer-support path.
        # The historical fractional split remains only for explicitly requested
        # checkpoint-shape compatibility.
        if getattr(model_config, "use_legacy_decoder_channel_override", False):
            c_outer_fraction = getattr(
                model_config, "decoder_last_c_outer_fraction", 0.25
            )
            c_outer_fraction = max(0.0, min(0.5, c_outer_fraction))
            self.c_outer = max(1, int(in_channels * c_outer_fraction))
        else:
            self.c_outer = int(out_channels)

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

        border_width = self.N // 4
        border_mask = torch.ones(
            x2.shape[-2:], dtype=x2.dtype, device=x2.device
        )
        border_mask[
            border_width:-border_width,
            border_width:-border_width,
        ] = 0
        x2 = x2 * border_mask

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

        num_channels = _semantic_component_channels(model_config)
        #Nn layers

        #Custom nn layers with specific identifiable names
        # Task 2.3 / B1 (Amendment #11): the phase head becomes the IMAGINARY head in
        # 'real_imag' mode, gated together with the mode. Default 'amp_phase' keeps
        # fno-stable's pi*tanh phase activation unchanged.
        if _effective_cnn_output_mode(model_config) == 'real_imag':
            phase_activation = ScaledTanh(scale=1.2)
        else:
            phase_activation = Tanh_custom_act()
        self.add_module('phase_activation', phase_activation)
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

        num_channels = _decoder_component_channels(model_config)

        #Custom nn layers with specific identifiable names
        # Task 2.3 / B1 (Amendment #11): the amplitude head becomes the REAL head in
        # 'real_imag' mode, gated together with the mode (ScaledTanh box: tanh+0.2 ->
        # range (-0.8, 1.2)). Default 'amp_phase' keeps fno-stable's Amplitude_activation.
        if _effective_cnn_output_mode(model_config) == 'real_imag':
            amp_activation = ScaledTanh(scale=1.0, offset=0.2)
        else:
            amp_activation = Amplitude_activation(model_config)
        self.add_module('amp_activation', amp_activation) # Pass config
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

# Shared decoder components (Task 2.4 / backlog B2, opt-in via
# ModelConfig.use_shared_decoder). Ported verbatim from main
# (`git show main:ptycho_torch/model.py`, class `FeatureRefinementBlock`).

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
    Shared decoder for real/imaginary (or amplitude/phase) components (Task 2.4 / B2).
    Structure ported from main (`git show main:ptycho_torch/model.py`, class
    `Decoder_shared`) -- per-level `FeatureRefinementBlock`s plus a final `ECALayer` --
    but adapted to fno-stable's `Decoder_base`/`Decoder_last`:

    1. No skip-connection plumbing: fno-stable's `Decoder_base` has no UNet skip/merge
       machinery at all (`Decoder_amp`/`Decoder_phase` don't use it either), so main's
       `skips`/`merge_blocks` loop has no fno-stable equivalent to port and is dropped.
       `forward(x, skips=None)` keeps the signature for interface parity with main;
       `skips` must be None here.
    2. No combined head activation: main's head applies a single hardwired
       `ScaledTanh(1.2)` over the whole summed [B, 2*C_out, N, N] output. That would NOT
       reproduce Task 2.3 (B1)'s per-mode activation gating (the asymmetric real/imag
       ScaledTanh box, or the amp_phase Amplitude_activation/pi*tanh pair), so the head
       here uses `nn.Identity()` and emits RAW channels; `Autoencoder.forward` applies
       the Task 2.3-consistent per-branch activation AFTER splitting (see
       `Autoencoder.__init__`/`Autoencoder.forward` and task-2.4-report.md).

    Output shape: [B, 2*C_out, N, N], raw/pre-activation (first C_out = real/amp
    branch, last C_out = imag/phase branch).
    """
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super(Decoder_shared, self).__init__(model_config, data_config, batch_norm=model_config.batch_norm)
        self.model_config = model_config
        self.data_config = data_config

        C_out = _decoder_component_channels(model_config)
        n_levels = len(self.blocks)

        self.refinement_blocks = nn.ModuleList()
        for i in range(n_levels):
            ch = self.filters[i + 1]
            expansion = 2 if i == n_levels - 1 else 1
            self.refinement_blocks.append(
                FeatureRefinementBlock(ch, expansion_factor=expansion)
            )

        base_ch = self.filters[-1]
        self.eca = ECALayer(channel=base_ch)

        self.add_module('head', Decoder_last(model_config, data_config,
                                             self.n_filters_scale * 32, C_out * 2,
                                             activation=nn.Identity(),
                                             batch_norm=model_config.batch_norm))

    def forward(self, x, skips=None):
        if skips is not None:
            raise NotImplementedError(
                "Decoder_shared (fno-stable port) has no skip-connection plumbing "
                "-- 'skips' must be None."
            )
        #Apply upscale block layers + per-level refinement (no skip merge; see class docstring)
        for i, block in enumerate(self.blocks):
            x = block(x)
            x = self.refinement_blocks[i](x)

        x = self.eca(x)

        #Apply final (raw, pre-activation) head
        outputs = self.head(x)

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

        # Task 2.4 / B2 (opt-in, default False = current architecture untouched).
        self.use_shared_decoder = getattr(model_config, 'use_shared_decoder', False)
        if self.use_shared_decoder:
            #Shared decoder: emits raw 2*C_out channels, split + activated below
            #with the SAME per-mode gating Decoder_amp/Decoder_phase use (Task 2.3 / B1).
            self.decoder_shared = Decoder_shared(model_config, data_config)
            if _effective_cnn_output_mode(model_config) == 'real_imag':
                self.shared_branch1_activation = ScaledTanh(scale=1.0, offset=0.2)  # real
                self.shared_branch2_activation = ScaledTanh(scale=1.2)              # imag
            else:
                self.shared_branch1_activation = Amplitude_activation(model_config)  # amp
                self.shared_branch2_activation = Tanh_custom_act()                   # phase
        else:
            #Decoders (Amplitude/Phase)
            self.decoder_amp = Decoder_amp(model_config, data_config) # Pass configs
            self.decoder_phase = Decoder_phase(model_config, data_config) # Pass configs


    def forward(self, x):
        #Encoder
        x = self.encoder(x)
        #CBAM optional
        x = self.bottleneck_cbam(x)

        if self.use_shared_decoder:
            combined = self.decoder_shared(x)
            C_out = combined.shape[1] // 2
            branch1_raw = combined[:, :C_out, :, :]
            branch2_raw = combined[:, C_out:, :, :]
            branch1 = self.shared_branch1_activation(branch1_raw)
            branch2 = self.shared_branch2_activation(branch2_raw)
            return branch1, branch2

        #Decoders
        x_amp = self.decoder_amp(x)
        x_phase = self.decoder_phase(x)

        return x_amp, x_phase

#Probe modules
class ProbeLayoutError(ValueError):
    """A probe tensor violates the documented (B, C, P, H, W) batch layout.

    Raised by ``ProbeIllumination.forward``'s precondition. Sub-rank-5 probes
    (in particular the legacy flat ``(B, H, W)`` dictionary-flow emission)
    right-align-broadcast into the mode axis, turning ``pad_and_diffract``'s
    coherent mode sum into a silent batch-size-dependent amplitude gain and,
    for per-sample-distinct probes, cross-sample field mixing.

    Contract: docs/specs/spec-ptycho-torch-probe-layout.md; mechanism and
    history (82da77960 / 8b3d7a011): docs/findings.md PROBE-RANK-001; design:
    docs/superpowers/specs/2026-07-12-probe-rank-physics-contract-fix-design.md.
    """


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

    Precondition (PROBE-RANK-001, enforced in forward): the probe batch MUST
    follow the documented layout ``(B|1, C|1, P, N, N)``. Any lower-rank
    tensor raises :class:`ProbeLayoutError` — no implicit right-align
    broadcast into the mode slot is reachable. See
    docs/specs/spec-ptycho-torch-probe-layout.md.
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

    def _require_documented_probe_layout(self, x, probe):
        """Fail fast on any probe violating the (B|1, C|1, P, N, N) contract.

        docs/specs/spec-ptycho-torch-probe-layout.md; docs/findings.md
        PROBE-RANK-001.
        """
        batch, channels = x.shape[0], x.shape[1]
        contract = (
            "documented probe batch layout is (B, C, P, H, W) with "
            f"B in (1, {batch}), C in (1, {channels}), H=W={self.N} "
            "(docs/specs/spec-ptycho-torch-probe-layout.md; "
            "docs/findings.md PROBE-RANK-001; design "
            "docs/superpowers/specs/"
            "2026-07-12-probe-rank-physics-contract-fix-design.md)"
        )
        if probe.ndim != 5:
            raise ProbeLayoutError(
                f"probe has rank {probe.ndim} shape {tuple(probe.shape)}; "
                "sub-rank-5 probes (e.g. the legacy flat (B, H, W) layout) "
                "broadcast into the mode axis and coherently sum across it "
                f"— a silent batch-size physics gain. {contract}"
            )
        if probe.shape[-2:] != (self.N, self.N):
            raise ProbeLayoutError(
                f"probe spatial dims {tuple(probe.shape[-2:])} != "
                f"({self.N}, {self.N}) for shape {tuple(probe.shape)}. {contract}"
            )
        if probe.shape[0] not in (1, batch):
            raise ProbeLayoutError(
                f"probe batch axis {probe.shape[0]} not in (1, {batch}) for "
                f"shape {tuple(probe.shape)}. {contract}"
            )
        if probe.shape[1] not in (1, channels):
            raise ProbeLayoutError(
                f"probe channel axis {probe.shape[1]} not in (1, {channels}) "
                f"for shape {tuple(probe.shape)}. {contract}"
            )

    def forward(self, x, probe):
        # PROBE-RANK-001 precondition: kill the silent batch-into-modes
        # broadcast before any arithmetic.
        self._require_documented_probe_layout(x, probe)

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
    
class RectangularScaledDiffraction(nn.Module):
    """Analytic real/imag intensity forward with folded probe/physics scaling (B5).

    Ported verbatim from ``main``'s ``ptycho_torch.model.RectangularScaledDiffraction``
    (bit-identical to ``ptycho_torch.beta_modules.model``), with the only change
    being the Task 1.2 ``requires_grad`` patch on ``s1``/``s2`` (gated by
    ``model_config.rect_s1s2_trainable``). Used only when
    ``ModelConfig.physics_forward_mode == 'rectangular_scaled'``.

    ``forward`` returns an *intensity* ``I_pred = sum_p |F{scale*(s1*P*Re(x) +
    i*s2*P*Im(x))}|^2`` (incoherent mode summation over probe modes P). The
    ``autograd=False`` branch is the variable-projection solve used for
    fine-tuning/inference; the training path always calls with ``autograd=True``,
    for which ``I_raw`` is a dead argument.
    """
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        # Task 1.2 patch: s1/s2 trainability is config-gated (main hardcodes True).
        self.s1 = nn.Parameter(torch.ones(model_config.num_datasets),
                               requires_grad=model_config.rect_s1s2_trainable)
        self.s2 = nn.Parameter(torch.ones(model_config.num_datasets),
                               requires_grad=model_config.rect_s1s2_trainable)

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

    @torch.no_grad()
    def basis_images(self, x, probe, scale=1.0):
        """Per-mode VarPro basis fields (Psi_a, Psi_b) for the (s1, s2) refit (B3).

        Bit-identical to the ``autograd=False`` branch's field computation
        (``exit_wave_a = probe * Re(x) * scale``,
        ``exit_wave_b = 1j * probe * Im(x) * scale``, ``fft2(norm='ortho')``,
        ``fftshift``), stopped BEFORE incoherent mode summation so the Eq. (5)
        cross-term ``Re(conj(Psi_a) * Psi_b)`` is recoverable downstream.
        Dropping that cross-term is a diagonal-only estimator that does NOT
        satisfy Eq. (8), so callers must accumulate the returned complex fields
        (e.g. via ``ptycho_torch.rect_scaling.accumulate_rect_basis``) rather
        than pre-summed intensities.

        ``probe`` and ``scale`` are the same tensors ``compute_loss`` feeds the
        training forward; their product is the effective probe ``P_eff``. Per
        audit CI-SCALE-001 ``output_scale * probe_training == probe_physical``,
        so passing ``probe = probe_physical`` with ``scale = 1.0`` reproduces
        the training exit-wave physics.

        Args:
            x: complex textures ``(B, C, H, W)`` (``forward_predict`` output).
            probe: effective probe, broadcastable to ``(B, C, P, H, W)``.
            scale: output-scale factor folded into the exit wave.

        Returns:
            ``(Psi_a, Psi_b)``, each complex of shape ``(B, C, P, H, W)``. For a
            single probe mode (``P == 1``, the CI canonical layout) the per-mode
            fields equal the Eq. (5) mode-summed basis, so accumulating them
            directly is exact.
        """
        x = x.unsqueeze(dim=2)
        x_a, x_b = x.real, x.imag
        exit_wave_a = probe * x_a * scale
        exit_wave_b = 1j * probe * x_b * scale
        Psi_a = torch.fft.fftshift(torch.fft.fft2(exit_wave_a, norm='ortho'), dim=(-2, -1))
        Psi_b = torch.fft.fftshift(torch.fft.fft2(exit_wave_b, norm='ortho'), dim=(-2, -1))
        return Psi_a, Psi_b


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
        self.training_assembly_spec = resolve_training_assembly_spec(
            self.object_big,
            self.model_config.training_patch_weighting,
        )

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

        # B5 (Task 2.6): analytic rectangular_scaled forward. Always constructed
        # so state_dicts/parameters are stable across modes; only invoked when
        # physics_forward_mode == 'rectangular_scaled'.
        self.rect_scaler = RectangularScaledDiffraction(model_config)

    def _assemble_training_patches(self, x, positions, probe):
        """Apply the sealed differentiable training assembly specification."""

        spec = self.training_assembly_spec
        if spec.mode == "pass_through_v1":
            return x
        if spec.mode == "central_mask_overlap_v1":
            reassembled_obj, _, _ = hh.reassemble_patches_position_real(
                x,
                positions,
                data_config=self.data_config,
                model_config=self.model_config,
            )
        else:
            reassembled_obj, _, _ = hh.reassemble_patches_position_real_probe(
                x,
                positions,
                data_config=self.data_config,
                model_config=self.model_config,
                probe=probe,
                use_probe_weights=(spec.configured_weighting == "probe"),
            )
        return hh.extract_channels_from_region(
            reassembled_obj[:, None, :, :],
            positions,
            data_config=self.data_config,
            model_config=self.model_config,
        )

    def forward(self, x, I_measured, positions, probe, output_scale_factor, experiment_ids = None):
        # ``I_measured`` is only consumed by RectangularScaledDiffraction's
        # variable-projection (autograd=False) branch, which the training path
        # never takes; it is a dead argument here for both amplitude and the
        # autograd=True rectangular_scaled path, kept for signature parity with
        # main so callers/tests can thread the observed images through.
        # PROBE-RANK-001 is enforced at this shared boundary because the
        # rectangular path returns before ProbeIllumination.forward. Keep the
        # validator in ProbeIllumination.forward as protection for direct use.
        self.probe_illumination._require_documented_probe_layout(x, probe)

        extracted_patch_objs = self._assemble_training_patches(x, positions, probe)

        # B5 (Task 2.6): rectangular_scaled REPLACES the amplitude chain
        # (ProbeIllumination -> pad_and_diffract -> inv_scale -> alpha/beta) with
        # RectangularScaledDiffraction. Physics/probe scaling is folded into
        # ``output_scale_factor`` by compute_loss (main's modified_output_scale),
        # so no loss-time pred*physics_scale multiply is applied downstream.
        if self.model_config.physics_forward_mode == 'rectangular_scaled':
            # amendment #3: resolve the probe (INCLUDING its mask, when
            # model_config.probe_mask is configured) via the SAME resolver the
            # amplitude path uses (ProbeIllumination), so masked/unmasked
            # semantics stay in lockstep. With no mask the resolver returns ones,
            # so ``effective_probe`` numerically equals the bare probe.
            probe_mask = self.probe_illumination._resolve_probe_mask(extracted_patch_objs)
            effective_probe = probe * probe_mask.view(1, 1, 1, self.N, self.N)
            return self.rect_scaler(
                x=extracted_patch_objs,
                I_raw=I_measured,
                probe=effective_probe,
                scale=output_scale_factor,
                experiment_ids=experiment_ids,
                autograd=True,
            )

        #Apply probe illum
        illuminated_objs, _ = self.probe_illumination(extracted_patch_objs,
                                                    probe)

        #Pad and diffract
        pred_unscaled_diffraction, _ = hh.pad_and_diffract(illuminated_objs,
                                                    pad = False) # Assuming pad=False is intended
        
        #Inverse scaling
        pred_scaled_diffraction = self.scaler.inv_scale(pred_unscaled_diffraction, output_scale_factor)

        # PROBE-RANK-001 §3.3: explicit amplitude physics gain, applied ONCE,
        # multiplicatively, to the predicted amplitude — amplitude mode only
        # (the rectangular_scaled branch returned above; inference uses
        # forward_predict and never reaches this site). Replaces the banned
        # flat-probe layout's accidental xB gain with a batch-size-
        # independent, hparams-recorded constant (read live from the shared
        # ModelConfig so checkpoint-loaded modules honor it). Validated by
        # scaling_contract.validate_amplitude_physics_gain; contract in
        # docs/specs/spec-ptycho-torch-probe-layout.md.
        amplitude_physics_gain = float(
            getattr(self.model_config, 'amplitude_physics_gain', 1.0)
        )
        if amplitude_physics_gain != 1.0:
            pred_scaled_diffraction = pred_scaled_diffraction * amplitude_physics_gain

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


class RectangularPoissonLoss(nn.Module):
    """Poisson NLL for the rectangular_scaled forward (intensity domain).

    RectangularScaledDiffraction returns an *intensity* (|psi_f|^2). This
    replicates main's ``PoissonIntensityLayer`` semantics EXACTLY: the prediction
    is the Poisson rate (lambda) directly and the raw observation is compared
    as-is -- NEITHER is re-squared. Contrast the default amplitude-domain
    ``PoissonLoss``/``PoissonIntensityLayer`` above, which square both.
    Only used when physics_forward_mode='rectangular_scaled' (amendment #2).
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, raw):
        poisson_dist = dist.Independent(dist.Poisson(pred), 3)
        return -poisson_dist.log_prob(raw)


class CIIntensityPoissonLoss(nn.Module):
    """Count-domain Poisson NLL for the absolute-scaling CI profile.

    The NLL is accumulated in float64 to avoid cancellation for high float32
    counts, then cast back to the prediction dtype after detector reduction so
    the training loss and its gradients retain the model's floating dtype.
    """

    def forward(self, pred, raw):
        torch._assert_async(
            torch.isfinite(raw).all(),
            "CI Poisson observations must be finite.",
        )
        torch._assert_async(
            (raw >= 0).all(),
            "CI Poisson observations must be non-negative.",
        )
        rate = torch.clamp(pred, min=1e-8)
        torch._assert_async(
            torch.isfinite(rate).all(),
            "CI Poisson rate must be finite.",
        )
        rate64 = rate.to(torch.float64)
        raw64 = raw.to(torch.float64)
        per_pixel_nll = (
            rate64
            - raw64 * torch.log(rate64)
            + torch.lgamma(raw64 + 1.0)
        )
        return per_pixel_nll.sum(dim=(-3, -2, -1)).to(pred.dtype)


class RectangularMAELoss(nn.Module):
    """MAE for the rectangular_scaled forward, replicating main's re-square quirk.

    main's ``MAELoss`` squares ``pred`` before the L1 comparison
    (``mae(pred**2, raw)``) on the assumption ``pred`` is an amplitude; because
    RectangularScaledDiffraction already returns an intensity, this DOUBLE-squares
    the prediction. Reproduced VERBATIM for frozen-fixture parity (amendment #2 --
    a documented quirk, deliberately NOT "fixed"). Only used when
    physics_forward_mode='rectangular_scaled'.
    """
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss(reduction='none')

    def forward(self, pred, raw):
        return self.mae(pred**2, raw)


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

        # B5 (Task 2.6): rectangular_scaled scales the object's real/imag parts
        # independently, which is only physically meaningful when the object is
        # genuinely real/imag-derived (FNO/hybrid generators, or CNN with
        # cnn_output_mode='real_imag'). Fail fast for amp/phase-derived objects.
        if model_config.physics_forward_mode == 'rectangular_scaled' \
                and self.generator_output != 'real_imag':
            raise ValueError(
                "physics_forward_mode='rectangular_scaled' requires real/imag-derived "
                "object patches (FNO/hybrid real_imag generators, or the default CNN "
                "with cnn_output_mode='real_imag'), but the resolved object output is "
                f"'{self.generator_output}'. Set cnn_output_mode='real_imag' or use a "
                "real_imag generator, or keep physics_forward_mode='amplitude'."
            )

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
        # ``x`` (the scaled observed input) is threaded as I_measured for the
        # rectangular_scaled variable-projection branch; it is unused by the
        # amplitude path and the autograd=True rectangular path.
        #
        # RECT-PROBE-SCALE-DOUBLE-DIV-001 (Task P1): the dataloader already
        # folds data_config.probe_scale into the normalized probe
        # (helper.normalize_probe_like_tf), so the rectangular_scaled forward
        # consumes the probe UNDIVIDED -- matching the inference-side VarPro
        # convention (reassembly.compute_varpro_basis). Dividing again here
        # suppressed predicted intensity by probe_scale**2 and drove the
        # Poisson equilibrium object to ~probe_scale x truth (washed
        # reconstructions; deliberate physics divergence from origin/main,
        # which retains the defect). The default 'amplitude' path keeps the
        # historical division byte-for-byte.
        if getattr(self.model_config, 'physics_forward_mode', 'amplitude') \
                == 'rectangular_scaled':
            forward_probe = probe
        else:
            forward_probe = probe / self.probe_scale
        x_out = self.forward_model.forward(x_combined, x, positions,
                                           forward_probe, output_scale_factor, experiment_ids)

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
        generator_output: str = "amp_phase",
        generator_overrides: Optional[Dict[str, Any]] = None,
        parity_scale_mode: str = "off",
        parity_fixed_delta: float = 0.0,
        parity_init_scheme: str = "default",
        model_spec: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        # Handle checkpoint loading: convert dict kwargs back to dataclass instances
        # (Lightning passes saved hyperparameters as dicts during load_from_checkpoint)
        if model_spec is not None:
            from dataclasses import fields

            for section_name, value, config_type in (
                ("model_config", model_config, ModelConfig),
                ("data_config", data_config, DataConfig),
                ("training_config", training_config, TrainingConfig),
                ("inference_config", inference_config, InferenceConfig),
            ):
                if not isinstance(value, dict):
                    continue
                expected = {item.name for item in fields(config_type)}
                received = set(value)
                if received != expected:
                    raise ValueError(
                        f"current checkpoint {section_name} field set is not exact; "
                        f"missing={sorted(expected - received)}, "
                        f"unknown={sorted(received - expected)}"
                    )
        if isinstance(model_config, dict):
            model_config = ModelConfig(**model_config)
        if isinstance(data_config, dict):
            data_config = DataConfig(**data_config)
        if isinstance(training_config, dict):
            training_config = TrainingConfig(**training_config)
        if isinstance(inference_config, dict):
            inference_config = InferenceConfig(**inference_config)

        decoded_model_spec = None
        if model_spec is not None:
            from dataclasses import fields
            from ptycho_torch.model_spec import ModelSpec

            decoded_model_spec = (
                model_spec
                if isinstance(model_spec, ModelSpec)
                else ModelSpec.from_payload(model_spec)
            )
            sealed_model_config = decoded_model_spec.to_model_config()
            mismatches = []
            for item in fields(ModelConfig):
                supplied = getattr(model_config, item.name)
                sealed = getattr(sealed_model_config, item.name)
                if isinstance(supplied, torch.Tensor) or isinstance(sealed, torch.Tensor):
                    equal = (
                        isinstance(supplied, torch.Tensor)
                        and isinstance(sealed, torch.Tensor)
                        and torch.equal(supplied, sealed)
                    )
                else:
                    equal = supplied == sealed
                if not equal:
                    mismatches.append(item.name)
            if mismatches:
                raise ValueError(
                    "checkpoint ModelSpec conflicts with dual-written model_config "
                    f"field(s): {sorted(mismatches)}"
                )
            if (
                parity_scale_mode != decoded_model_spec.parity_scale_mode
                or float(parity_fixed_delta) != decoded_model_spec.parity_fixed_delta
                or parity_init_scheme != decoded_model_spec.parity_init_scheme
            ):
                raise ValueError(
                    "checkpoint ModelSpec parity identity conflicts with dual-written "
                    "Lightning parity hyperparameters"
                )
            model_config = sealed_model_config

        resolved_scale_contract = validate_scale_contract(
            data_config,
            model_config,
            training_config,
        )

        generator_module, generator_output = _resolve_generator_from_config(
            model_config,
            data_config,
            generator_module,
            generator_output,
            generator_overrides=generator_overrides,
        )

        # Save hyperparameters for checkpoint serialization (Phase D1c requirement)
        # Convert dataclass instances to dicts to ensure serializability
        # Note: generator_module is not saved (reconstructed at load time)
        from dataclasses import asdict
        serialized_hparams = {
            'model_config': asdict(model_config),
            'data_config': asdict(data_config),
            'training_config': asdict(training_config),
            'inference_config': asdict(inference_config),
            'generator_output': generator_output,
            'generator_overrides': dict(generator_overrides or {}),
            'parity_scale_mode': parity_scale_mode,
            'parity_fixed_delta': float(parity_fixed_delta),
            'parity_init_scheme': parity_init_scheme,
        }
        if decoded_model_spec is not None:
            serialized_hparams["model_spec"] = decoded_model_spec.to_payload()
        self.save_hyperparameters(serialized_hparams)
        self._model_spec = decoded_model_spec

        # TF-parity global intensity-scale offset (default "off"; see
        # docs/plans/2026-07-08-cnn-n128-tf-parity.md Task 1 and
        # ptycho/model.py:291-298 for the TF tie direction this mirrors).
        # Validated eagerly so a garbage mode fails fast at construction
        # rather than at first forward().
        if parity_scale_mode not in ("off", "tied", "input", "output", "fixed"):
            raise ValueError(
                f"Invalid parity_scale_mode={parity_scale_mode!r}. "
                "Expected one of 'off', 'tied', 'input', 'output', 'fixed'."
            )
        if parity_init_scheme not in ("default", "tf_glorot"):
            raise ValueError(
                f"Invalid parity_init_scheme={parity_init_scheme!r}. "
                "Expected 'default' or 'tf_glorot'."
            )
        self.parity_scale_mode = parity_scale_mode

        self.n_filters_scale = model_config.n_filters_scale
        self.predict = False

        #Configs
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config
        self.inference_config = inference_config
        self._ci_mode = (
            resolved_scale_contract is not None
            and resolved_scale_contract.version == CI_SCALE_CONTRACT
        )
        self.register_buffer(
            "_ci_rms_input_scale",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_ci_mean_measured_intensity",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )

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
        self._last_ci_raw_count_nll = None
        self._last_calibration_means = None

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

        # Trainable/fixed log-scale delta -- ONLY materialized when the parity
        # mechanism is active, so "off" construction and old checkpoints keep
        # a state_dict with no extra key (strict load_state_dict stays clean).
        if self.parity_scale_mode != "off":
            trainable_delta = self.parity_scale_mode in ("tied", "input", "output")
            self.log_scale_delta = nn.Parameter(
                torch.tensor(float(parity_fixed_delta)), requires_grad=trainable_delta,
            )

        # TF-parity weight-init preset: redraw self.model's Conv2d/ConvTranspose2d
        # weights with Keras's glorot_uniform equivalent and zero their biases,
        # matching the TF reference's initializer (evidence base, Task 0).
        if parity_init_scheme == "tf_glorot":
            for module in self.model.modules():
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

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
        # B5 (Task 2.6, amendment #2): the rectangular_scaled forward emits an
        # intensity, so its loss stack must use main's intensity-domain semantics
        # (Poisson rate = pred directly; MAE re-squares pred). The default
        # 'amplitude' mode keeps fno-stable's amplitude-domain PoissonLoss/MAELoss
        # byte-for-byte unchanged.
        _rect_mode = getattr(model_config, 'physics_forward_mode', 'amplitude') == 'rectangular_scaled'
        #Poisson Loss only works with Unsupervised
        if model_config.mode == 'Unsupervised' and model_config.loss_function == 'Poisson':
            if self._ci_mode:
                self.Loss = CIIntensityPoissonLoss()
            else:
                self.Loss = RectangularPoissonLoss() if _rect_mode else PoissonLoss()
            self.loss_name = 'poisson_train'
            self.val_loss_name = 'poisson_val'
        #MAE works with both
        elif model_config.mode == 'Unsupervised' and model_config.loss_function == 'MAE':
            self.Loss = RectangularMAELoss() if _rect_mode else MAELoss()
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
    
    def _parity_scale_factors(self):
        """TF tie direction (ptycho/model.py:291-298, read-only): input is
        DIVIDED by exp(w), output is MULTIPLIED by exp(w). Returns (f_in, f_out);
        ``None`` means "leave this side's scale factor unchanged". Only called
        when parity_scale_mode != 'off' (log_scale_delta exists in that case).
        """
        d = self.log_scale_delta
        if self.parity_scale_mode == "input":
            return torch.exp(-d), None
        if self.parity_scale_mode == "output":
            return None, torch.exp(d)
        return torch.exp(-d), torch.exp(d)  # tied and fixed

    def forward(self, x, positions, probe, input_scale_factor, output_scale_factor, experiment_ids):
        if self.parity_scale_mode != "off":
            f_in, f_out = self._parity_scale_factors()
            if f_in is not None:
                input_scale_factor = input_scale_factor * f_in
            if f_out is not None:
                output_scale_factor = output_scale_factor * f_out
        x_out = self.model(x, positions, probe, input_scale_factor, output_scale_factor, experiment_ids)
        return x_out

    def forward_predict(self, x, positions, probe, input_scale_factor):
        #Turns padding off if we need to
        if self.parity_scale_mode != "off":
            f_in, _f_out = self._parity_scale_factors()
            if f_in is not None:
                input_scale_factor = input_scale_factor * f_in
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

    def register_ci_statistics(self, statistics):
        required = {"rms_input_scale", "mean_measured_intensity"}
        if set(statistics) != required:
            raise ValueError(
                "CI statistics must contain exactly rms_input_scale and "
                "mean_measured_intensity."
            )
        rms_input_scale = torch.as_tensor(
            statistics["rms_input_scale"],
            dtype=torch.float32,
            device=self.device,
        ).detach().reshape(-1).clone()
        mean_measured_intensity = torch.as_tensor(
            statistics["mean_measured_intensity"],
            dtype=torch.float32,
            device=self.device,
        ).detach().reshape(-1).clone()
        if rms_input_scale.numel() == 0 or (
            rms_input_scale.shape != mean_measured_intensity.shape
        ):
            raise ValueError(
                "CI statistic arrays must be nonempty and have matching shapes."
            )
        for name, value in (
            ("rms_input_scale", rms_input_scale),
            ("mean_measured_intensity", mean_measured_intensity),
        ):
            if not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
                raise ValueError(f"{name} must be positive and finite.")
        self._ci_rms_input_scale = rms_input_scale
        self._ci_mean_measured_intensity = mean_measured_intensity

    def get_ci_statistics(self):
        if self._ci_rms_input_scale.numel() == 0:
            return None
        return {
            "rms_input_scale": self._ci_rms_input_scale.detach().clone(),
            "mean_measured_intensity": (
                self._ci_mean_measured_intensity.detach().clone()
            ),
        }

    def on_save_checkpoint(self, checkpoint):
        if self._model_spec is not None:
            from ptycho_torch.artifact_schema import (
                CURRENT_ARTIFACT_SCHEMA_VERSION,
                TORCH_ARTIFACT_BACKEND,
            )

            checkpoint["ptychopinn_artifact"] = {
                "backend": TORCH_ARTIFACT_BACKEND,
                "schema_version": CURRENT_ARTIFACT_SCHEMA_VERSION,
            }
        statistics = self.get_ci_statistics()
        if statistics is not None:
            checkpoint["ci_statistics"] = statistics

    def on_load_checkpoint(self, checkpoint):
        statistics = checkpoint.get("ci_statistics")
        if statistics is not None:
            self.register_ci_statistics(statistics)

    def compute_loss(self, batch):
        """
        Enhanced loss computation supporting multi-stage training
        """
        rectangular_mode = getattr(
            self.model_config,
            'physics_forward_mode',
            'amplitude',
        ) == 'rectangular_scaled'
        ci_mode = rectangular_mode and self._ci_mode

        # Grab required data fields from TensorDict
        fields = batch[0]
        x = fields['images']
        positions = fields['coords_relative']
        experiment_ids = fields['experiment_id']
        if ci_mode:
            observed_images = fields['measured_intensity']
            rms_scale = fields['rms_input_scale']
            mean_measured_intensity = fields['mean_measured_intensity']
            named_probe_fields = (
                'probe_training',
                'probe_physical',
                'probe_normalization',
            )
            named_probe_presence = [name in fields for name in named_probe_fields]
            if any(named_probe_presence):
                # Partial named payloads are invalid; direct indexing identifies
                # the missing field. Complete CI loaders always take this path.
                probe = fields['probe_training']
                probe_physical = fields['probe_physical']
                probe_normalization = fields['probe_normalization']
            else:
                # Deprecated outer tuple aliases remain accepted only as an
                # all-or-nothing compatibility payload for pre-contract callers.
                probe = batch[1]
                probe_physical = probe
                probe_normalization = batch[2].unsqueeze(-1)
            if probe.shape != probe_physical.shape:
                raise ValueError(
                    "probe_training and probe_physical must have matching shapes."
                )
            if probe.device != probe_physical.device or probe.dtype != probe_physical.dtype:
                raise ValueError(
                    "probe_training and probe_physical must have matching dtype/device."
                )
            expected_normalization_shape = (probe.shape[0], 1, 1, 1, 1)
            if probe_normalization.shape != expected_normalization_shape:
                raise ValueError(
                    "probe_normalization must have shape "
                    f"{expected_normalization_shape}; got {tuple(probe_normalization.shape)}."
                )
            scale = probe_normalization.reshape(probe.shape[0], 1, 1, 1)
            physics_scale = None
        else:
            observed_images = fields.get('observed_images', x)
            probe = batch[1]
            rms_scale = fields['rms_scaling_constant']
            physics_scale = fields['physics_scaling_constant']
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
        self._last_ci_raw_count_nll = None

        # B5 (Task 2.6): rectangular_scaled reproduces main's scale routing --
        # fold probe_scaling (batch[2]=scale) and physics_scale into the forward's
        # output_scale_factor as modified_output_scale = sqrt(1/(scale^2 *
        # physics_scale + 1e-9)), instead of the amplitude path's loss-time
        # pred*physics_scale multiply. The default 'amplitude' path is unchanged.
        if rectangular_mode:
            if ci_mode:
                output_scale = scale.reciprocal()
            else:
                output_scale = torch.sqrt(
                    1.0 / (scale ** 2 * physics_scale + 1e-9)
                )
        else:
            output_scale = rms_scale

        # Perform forward pass up and scale
        pred, amp, phase = self(x, positions, probe,
                                            input_scale_factor = rms_scale,
                                            output_scale_factor = output_scale,
                                            experiment_ids = experiment_ids
                                            )

        if self.model_config.mode == 'Unsupervised' and rectangular_mode:
            if ci_mode:
                raw_count_nll = self.Loss(pred, observed_images)
                self._last_ci_raw_count_nll = raw_count_nll.mean().detach()
                total_loss += normalize_ci_poisson_per_sample(
                    raw_count_nll,
                    mean_measured_intensity,
                ).mean()
            else:
                # Explicit legacy retains the historical normalized-amplitude
                # Poisson and double-square MAE reductions.
                intensity_norm_factor = torch.mean(observed_images).detach() + 1e-8
                total_loss += self.Loss(pred, observed_images).mean()
                total_loss /= intensity_norm_factor

        elif self.model_config.mode == 'Unsupervised':
            intensity_norm_factor = torch.mean(observed_images).detach() + 1e-8
            pred_physics = pred * physics_scale
            obs_physics = observed_images * physics_scale
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

    def _loss_target_intensity(self, batch, pred):
        """Return the intensity tensor compute_loss compares ``pred`` against.

        Mirrors the target expression of the active rectangular_scaled loss
        branch (compute_loss above): the CI branch feeds
        ``fields['measured_intensity']`` (raw counts) straight into
        CIIntensityPoissonLoss(pred, observed_images); the explicit-legacy
        branch feeds ``fields.get('observed_images', fields['images'])`` into
        RectangularPoissonLoss(pred, observed_images). In both branches the
        comparison happens in the intensity domain with no further rescaling
        of either side, so the target is exactly these batch fields.
        """
        fields = batch[0]
        rectangular_mode = getattr(
            self.model_config,
            'physics_forward_mode',
            'amplitude',
        ) == 'rectangular_scaled'
        if rectangular_mode and self._ci_mode:
            target = fields['measured_intensity']
        else:
            target = fields.get('observed_images', fields['images'])
        return target.detach().to(device=pred.device, dtype=pred.dtype)

    @torch.no_grad()
    def calibrate_rect_s1s2(self, batch):
        """Initialize rectangular-forward s1/s2 so predicted intensity matches
        the measured intensity scale on one batch, avoiding a large initial
        mismatch between predicted and measured count intensities.

        With s1 = s2 = s the exit wave is linear in s, so the predicted
        intensity scales as s**2; s = sqrt(mean(target)/mean(pred @ s=1))
        equalizes the means the Poisson NLL actually compares. Runs no-grad,
        touches only s1/s2 data (plus the diagnostic _last_calibration_means);
        module training state is left untouched.

        Returns the calibrated scalar, or None when not applicable.
        """
        if getattr(self.model_config, "physics_forward_mode", None) != "rectangular_scaled":
            return None
        scaler = next((m for m in self.modules()
                       if isinstance(m, RectangularScaledDiffraction)), None)
        if scaler is None:
            return None
        captured = {}

        def _cap(_mod, _inp, out):
            captured["pred"] = out.detach()

        handle = scaler.register_forward_hook(_cap)
        try:
            self.compute_loss(batch)
        finally:
            handle.remove()
        pred = captured.get("pred")
        if pred is None:
            return None
        target = self._loss_target_intensity(batch, pred)  # same tensor domain
        s = torch.sqrt(target.mean() / pred.mean().clamp_min(1e-30)).item()
        scaler.s1.data.fill_(s)
        scaler.s2.data.fill_(s)
        # re-run to record post-calibration means for verification/logging
        handle = scaler.register_forward_hook(_cap)
        try:
            self.compute_loss(batch)
        finally:
            handle.remove()
        self._last_calibration_means = (
            captured["pred"].mean(), target.mean())
        return s


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
        if self._last_ci_raw_count_nll is not None:
            self.log(
                "raw_count_nll_train",
                self._last_ci_raw_count_nll,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
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
        if self._last_ci_raw_count_nll is not None:
            self.log(
                "raw_count_nll_val",
                self._last_ci_raw_count_nll,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
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

    def on_train_epoch_end(self):
        """Log the TF-parity log-scale delta once per epoch (no-op when the
        mechanism is off) so its trajectory is recoverable from Lightning CSV
        logs -- see docs/plans/2026-07-08-cnn-n128-tf-parity.md Task 1."""
        if self.parity_scale_mode != "off":
            self.log("parity_log_scale_delta", self.log_scale_delta.detach(), on_epoch=True, logger=True)
