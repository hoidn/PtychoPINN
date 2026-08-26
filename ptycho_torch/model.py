#Standard libs
import math
from typing import Any, Dict, Optional

#Torch
import torch
from torch import nn

#Helper
from ptycho_torch.config_params import ModelConfig, DataConfig
from ptycho_torch.object_compatibility import resolve_model_object_compatibility

#nn.Module block zoo (W4 split) — re-exported for the spec-pinned public paths.
from ptycho_torch.model_blocks import (
    Autoencoder,
    CombineComplex,
    CIIntensityPoissonLoss,
    Decoder_last,
    Encoder,
    ForwardModel,
    MAELoss,
    PoissonIntensityLayer,
    PoissonLoss,
    ProbeIllumination,
    ProbeLayoutError,
    PtychoPINN,
    PtychoPINN_Lightning,
    Ptycho_Supervised,
    RectangularMAELoss,
    RectangularPoissonLoss,
    RectangularScaledDiffraction,
)


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

    See: docs/findings.md (see git history for the originating plan) Task 1.
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


def _semantic_component_channels(
    model_config: ModelConfig,
    data_config: DataConfig,
) -> int:
    compatibility = resolve_model_object_compatibility(model_config)
    if compatibility.layout == "grouped_patch_components_v1":
        return data_config.gridsize * data_config.gridsize
    return 1

def _decoder_component_channels(
    model_config: ModelConfig,
    data_config: DataConfig,
) -> int:
    compatibility = resolve_model_object_compatibility(model_config)
    semantic_channels = _semantic_component_channels(model_config, data_config)
    if compatibility.layout == "single_patch_components_v1" or not getattr(
        model_config,
        "use_legacy_decoder_channel_override",
        False,
    ):
        return semantic_channels

    legacy_channels = int(model_config.decoder_last_amp_channels)
    if legacy_channels not in {1, semantic_channels}:
        raise ValueError(
            "decoder_last_amp_channels must be 1 or the semantic component "
            "count when "
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
        "C": data_config.gridsize * data_config.gridsize,

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
        if int(getattr(data_config, "gridsize", 1)) != 1:
            raise ValueError(
                "neuralop_uno checkpoint rebuild only supports the locked "
                f"gridsize=1 CDI contract; got gridsize={getattr(data_config, 'gridsize', None)}."
            )
        return NeuralopUnoGeneratorModule(
            C=data_config.gridsize * data_config.gridsize,
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


