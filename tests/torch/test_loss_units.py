import torch
import torch.distributions as dist

from ptycho_torch.model import PoissonIntensityLayer, MAELoss


def test_poisson_intensity_layer_squares_observations():
    pred_amp = torch.tensor([[[[2.0]]]])  # (B, C, H, W)
    obs_amp = torch.tensor([[[[3.0]]]])

    layer = PoissonIntensityLayer(pred_amp)
    loss = layer(obs_amp)

    expected = -dist.Independent(
        dist.Poisson(pred_amp ** 2, validate_args=False),
        3,
    ).log_prob(obs_amp ** 2)

    assert torch.allclose(loss, expected)


def test_mae_loss_operates_on_amplitude():
    pred_amp = torch.tensor([[[[2.0]]]])
    obs_amp = torch.tensor([[[[3.5]]]])

    loss_fn = MAELoss()
    loss = loss_fn(pred_amp, obs_amp)

    expected = torch.nn.functional.l1_loss(pred_amp, obs_amp, reduction="none")
    assert torch.allclose(loss, expected)
