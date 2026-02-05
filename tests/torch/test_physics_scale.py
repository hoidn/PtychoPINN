import torch
from ptycho_torch import helper as hh


def test_derive_intensity_scale_from_amplitudes():
    # Two samples, 1 channel, 2x2
    x = torch.tensor([
        [[[1.0, 1.0], [1.0, 1.0]]],  # sum(x**2)=4
        [[[2.0, 0.0], [0.0, 0.0]]],  # sum(x**2)=4
    ])
    # mean(sum(x**2)) = 4
    nphotons = 100.0
    scale = hh.derive_intensity_scale_from_amplitudes(x, nphotons)
    assert torch.is_tensor(scale)
    assert torch.isclose(scale, torch.tensor(5.0), atol=1e-6)  # sqrt(100/4)=5
