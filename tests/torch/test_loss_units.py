import torch
import torch.distributions as dist

from ptycho_torch.model import PoissonIntensityLayer, MAELoss


def test_poisson_intensity_layer_uses_intensities_directly():
    pred_intensity = torch.tensor([[[[4.0]]]])  # (B, C, H, W)
    obs_intensity = torch.tensor([[[[9.0]]]])

    layer = PoissonIntensityLayer(pred_intensity)
    loss = layer(obs_intensity)

    expected = -dist.Independent(
        dist.Poisson(pred_intensity, validate_args=False),
        3,
    ).log_prob(obs_intensity)

    assert torch.allclose(loss, expected)


def test_mae_loss_squares_predictions():
    pred_amp = torch.tensor([[[[2.0]]]])
    obs_intensity = torch.tensor([[[[3.5]]]])

    loss_fn = MAELoss()
    loss = loss_fn(pred_amp, obs_intensity)

    expected = torch.nn.functional.l1_loss(pred_amp ** 2, obs_intensity, reduction="none")
    assert torch.allclose(loss, expected)


def test_ci_gate_tripwire_delete_me():
    assert False, "deliberate red-path verification of the CI gate"
