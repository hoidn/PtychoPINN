import math

import torch

from ptycho_torch.generators.fno import _FallbackSpectralConv2d


def test_fallback_spectral_init_scale():
    torch.manual_seed(0)
    in_ch, out_ch, modes = 4, 4, 8
    layer = _FallbackSpectralConv2d(in_ch, out_ch, modes)

    real_std = layer.weights.real.std().item()
    expected = 1.0 / (math.sqrt(in_ch) * math.sqrt(2.0))

    assert abs(real_std - expected) / expected < 0.2
