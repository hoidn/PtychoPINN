"""Regression pin for PtychoPINN_Lightning.compute_loss at C=4 with object_big=True.

Freezes the loss (and predicted-diffraction tensor) produced by a seeded synthetic
batch under fno-stable's current default knobs. Every later change to the forward
model / loss path can replay this test to prove "defaults unchanged". On first run
(fixture NPZ absent) the test computes and saves the fixture, asserting only
finiteness. On subsequent runs it asserts the freshly computed values match the
frozen fixture to a tight tolerance.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig, InferenceConfig
from ptycho_torch.model import PtychoPINN_Lightning

SEED = 20260701
FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "varpro_parity" / "compute_loss_c4_regression.npz"


def _build_model_and_batch():
    """Construct the Lightning module and a seeded synthetic C=4 batch.

    object_big=True and C=C_model=C_forward=4 are fno-stable's defaults; all other
    knobs are left at their default values (Unsupervised mode, Poisson loss).
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data_cfg = DataConfig(N=64, C=4, grid_size=(2, 2))
    model_cfg = ModelConfig(object_big=True, C_model=4, C_forward=4)
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
def test_compute_loss_c4_regression_pin():
    model, batch, images, positions, probe, rms_scale, experiment_id = _build_model_and_batch()

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

    if not FIXTURE_PATH.exists():
        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            FIXTURE_PATH,
            loss=loss.item(),
            expected=pred.detach().numpy(),
            seed=SEED,
        )
        return

    fixture = np.load(FIXTURE_PATH)
    assert loss.item() == pytest.approx(float(fixture["loss"]), abs=1e-5, rel=1e-5), (
        "compute_loss output drifted from the frozen C=4 regression fixture"
    )
    np.testing.assert_allclose(
        pred.detach().numpy(), fixture["expected"], atol=1e-5, rtol=1e-5,
        err_msg="predicted diffraction tensor drifted from the frozen C=4 regression fixture",
    )
