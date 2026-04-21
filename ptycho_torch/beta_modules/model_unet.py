"""
Deprecation shim — all classes have been merged into ptycho_torch.model.

Import directly from ptycho_torch.model instead of this module.
"""
import warnings as _warnings

_warnings.warn(
    "ptycho_torch.beta_modules.model_unet is deprecated. "
    "Import from ptycho_torch.model instead.",
    DeprecationWarning,
    stacklevel=2,
)

from ptycho_torch.model import (  # noqa: F401
    Encoder,
    Decoder_base,
    Decoder_last,
    Decoder_last_Amp,
    Decoder_last_Phase,
    Decoder_phase,
    Decoder_amp,
    FeatureRefinementBlock,
    Decoder_shared,
    Autoencoder,
    ForwardModel,
    RectangularScaledDiffraction,
    PoissonIntensityLayer,
    PoissonLoss,
    PtychoPINN,
    Ptycho_Supervised,
    PtychoPINN_Lightning,
)
