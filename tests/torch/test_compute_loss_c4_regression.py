"""C=4 object_big regression coverage for symmetric and historical CNN heads."""

import numpy as np
import pytest
import torch

from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig, InferenceConfig
from ptycho_torch.model import PtychoPINN_Lightning

SEED = 20260701


def _build_model_and_batch(*, use_legacy_decoder_channel_override=False):
    """Construct the Lightning module and a seeded synthetic C=4 batch.

    object_big=True and C=C_model=C_forward=4 are fno-stable's defaults; all other
    knobs are left at their default values (Unsupervised mode, Poisson loss).
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data_cfg = DataConfig(N=64, C=4, grid_size=(2, 2))
    model_cfg = ModelConfig(
        object_big=True,
        C_model=4,
        C_forward=4,
        use_legacy_decoder_channel_override=use_legacy_decoder_channel_override,
    )
    train_cfg = TrainingConfig()
    infer_cfg = InferenceConfig()

    model = PtychoPINN_Lightning(model_cfg, data_cfg, train_cfg, infer_cfg)
    model.eval()

    B, C, N = 2, data_cfg.C, data_cfg.N
    gen = torch.Generator().manual_seed(SEED)

    images = torch.rand((B, C, N, N), generator=gen, dtype=torch.float32)

    # Small 2x2-grid-like offsets (pixels) plus per-sample jitter, consistent with
    # DataConfig.max_neighbor_distance=3.0.
    base_offsets = torch.tensor([[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]])
    jitter = (torch.rand((B, C, 2), generator=gen) - 0.5) * 0.5
    positions = (base_offsets.unsqueeze(0) + jitter).unsqueeze(2)  # (B, C, 1, 2)

    rms_scale = 0.5 + torch.rand((B, 1, 1, 1), generator=gen)
    physics_scale = 0.5 + torch.rand((B, 1, 1, 1), generator=gen)
    experiment_id = torch.zeros(B, dtype=torch.long)

    probe_base = (
        torch.randn((1, 1, N, N), generator=gen)
        + 1j * torch.randn((1, 1, N, N), generator=gen)
    )
    probe = probe_base.unsqueeze(2).expand(B, C, 1, N, N).contiguous().to(torch.complex64)

    scaling_const = 0.5 + torch.rand((B, 1, 1, 1), generator=gen)

    batch = (
        {
            "images": images,
            "coords_relative": positions,
            "rms_scaling_constant": rms_scale,
            "physics_scaling_constant": physics_scale,
            "experiment_id": experiment_id,
        },
        probe,
        scaling_const,
    )

    return model, batch, images, positions, probe, rms_scale, experiment_id


@pytest.mark.torch
def test_compute_loss_c4_symmetric_prediction_is_finite():
    model, batch, images, positions, probe, rms_scale, experiment_id = _build_model_and_batch()
    autoencoder = model.model.autoencoder

    assert autoencoder.decoder_amp.amp.conv1.out_channels == 4
    assert autoencoder.decoder_phase.phase.conv1.out_channels == 4

    with torch.no_grad():
        pred, _amp, _phase = model(
            images, positions, probe,
            input_scale_factor=rms_scale,
            output_scale_factor=rms_scale,
            experiment_ids=experiment_id,
        )
        loss = model.compute_loss(batch)

    assert torch.isfinite(loss), f"compute_loss produced a non-finite loss: {loss}"
    assert torch.isfinite(pred).all(), "predicted diffraction tensor contains non-finite values"


@pytest.mark.torch
def test_c4_legacy_asymmetric_override_fails_closed_at_prediction():
    model, _batch, images, positions, probe, rms_scale, experiment_id = (
        _build_model_and_batch(use_legacy_decoder_channel_override=True)
    )
    autoencoder = model.model.autoencoder

    assert autoencoder.decoder_amp.amp.conv1.out_channels == 1
    assert autoencoder.decoder_phase.phase.conv1.out_channels == 4

    with pytest.raises(
        ValueError,
        match="amp_phase tuple branches must have matching shapes before complex combination",
    ):
        model(
            images,
            positions,
            probe,
            input_scale_factor=rms_scale,
            output_scale_factor=rms_scale,
            experiment_ids=experiment_id,
        )
