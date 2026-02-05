import torch
from ptycho_torch import helper as hh


def test_derive_intensity_scale_from_amplitudes():
    # Two samples, 2 channels, 2x2 (channel-last)
    x = torch.tensor([
        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],  # per-channel sum=4, avg=4
        [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],  # per-channel sum=4/0, avg=2
    ])
    # mean(avg_channel_sum) = (4 + 2) / 2 = 3
    nphotons = 12.0
    scale = hh.derive_intensity_scale_from_amplitudes(x, nphotons)
    assert torch.is_tensor(scale)
    assert torch.isclose(scale, torch.tensor(2.0), atol=1e-6)  # sqrt(12/3)=2
