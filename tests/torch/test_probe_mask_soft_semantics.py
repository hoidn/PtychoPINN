import numpy as np
import pytest
import torch

from ptycho_torch.config_params import DataConfig, ModelConfig
from ptycho_torch.helper import normalize_probe_like_tf
from ptycho_torch.model import ProbeIllumination


def _hard_disk_mask(n: int, diameter: float) -> np.ndarray:
    centered = np.arange(n, dtype=np.float32) - (n // 2) + 0.5
    xx, yy = np.meshgrid(centered, centered)
    radius = float(diameter) / 2.0
    return (np.sqrt(xx * xx + yy * yy) < radius).astype(np.float32)


def test_model_config_probe_mask_defaults_enabled():
    cfg = ModelConfig()
    assert cfg.probe_mask is True
    assert cfg.probe_mask_sigma == pytest.approx(1.0)


def test_normalize_probe_like_tf_supports_soft_mask_controls():
    rng = np.random.default_rng(7)
    n = 16
    probe_guess = (rng.random((n, n)) + 1j * rng.random((n, n))).astype(np.complex64)

    normalized_soft, _ = normalize_probe_like_tf(
        probe_guess,
        probe_scale=4.0,
        probe_mask=True,
        probe_mask_sigma=1.0,
        probe_mask_diameter=n / 2.0,
    )
    normalized_hard, _ = normalize_probe_like_tf(
        probe_guess,
        probe_scale=4.0,
        probe_mask=True,
        probe_mask_tensor=_hard_disk_mask(n, diameter=n / 2.0),
        probe_mask_sigma=0.0,
    )

    assert normalized_soft.shape == (n, n)
    assert normalized_hard.shape == (n, n)
    assert not np.allclose(normalized_soft, normalized_hard, atol=1e-6)


def test_probe_illumination_uses_soft_disk_when_probe_mask_enabled():
    n = 16
    data_cfg = DataConfig(N=n, C=1, grid_size=(1, 1))
    model_cfg = ModelConfig(
        probe_mask=True,
        probe_mask_sigma=1.0,
        probe_mask_diameter=n / 2.0,
    )

    layer = ProbeIllumination(model_cfg, data_cfg)

    x = torch.ones((1, 1, n, n), dtype=torch.complex64)
    probe = torch.ones((1, 1, 1, n, n), dtype=torch.complex64)
    illuminated, _ = layer(x, probe)
    mask_like = illuminated[0, 0, 0].real.detach().cpu().numpy()

    assert mask_like[n // 2, n // 2] > 0.9
    assert mask_like[0, 0] < 0.1
    assert 0.0 < mask_like[n // 2, n // 2 + (n // 4)] < 1.0
