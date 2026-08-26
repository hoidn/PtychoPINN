"""Parity: both mirrors route grouping through the single shared call site.

``ptycho.grouping.group_from_config`` is the one place that decides seed and
count semantics.  This fixture pins that the TensorFlow mirror
(``create_ptycho_data_container``) and the PyTorch mirror
(``create_torch_data_container``) produce identical sample indices and group
shapes for the same seeded input.
"""

import numpy as np
import pytest

from ptycho import params
from ptycho.config.config import ModelConfig, TrainingConfig, update_legacy_dict
from ptycho.raw_data import RawData


def _make_raw_data(n_points: int = 25, n_pixels: int = 32) -> RawData:
    """Small deterministic synthetic scan (no ground-truth Y, no objectGuess)."""
    rng = np.random.default_rng(0)
    coords = np.linspace(0.0, float(n_points - 1), n_points)
    return RawData(
        xcoords=coords,
        ycoords=coords,
        xcoords_start=coords,
        ycoords_start=coords,
        diff3d=rng.random((n_points, n_pixels, n_pixels)).astype(np.float32),
        probeGuess=np.ones((n_pixels, n_pixels), dtype=np.complex64),
        scan_index=np.arange(n_points, dtype=int),
    )


@pytest.fixture
def legacy_params_snapshot():
    """Snapshot and restore ``params.cfg`` around each test."""
    original = params.cfg.copy()
    yield params.cfg
    params.cfg.clear()
    params.cfg.update(original)


def test_both_mirrors_share_sample_indices_and_group_shapes(legacy_params_snapshot):
    from ptycho.workflows.components import create_ptycho_data_container
    from ptycho_torch.workflows.components import create_torch_data_container

    raw = _make_raw_data()
    config = TrainingConfig(
        model=ModelConfig(N=32, gridsize=2, model_type="pinn"),
        training_groups=8,
        neighbor_count=4,
        subsample_seed=12345,
    )
    update_legacy_dict(params.cfg, config)

    tf_container = create_ptycho_data_container(raw, config)
    torch_container = create_torch_data_container(raw, config)

    tf_indices = np.asarray(tf_container.nn_indices)
    torch_indices = np.asarray(torch_container.nn_indices.cpu().numpy())

    # Group shapes: (n_groups, C) plan and (n_groups, N, N, C) diffraction.
    assert tf_indices.shape == tuple(torch_container.nn_indices.shape)
    assert tuple(tf_container._X_np.shape) == tuple(torch_container.X.shape)

    # Sample indices: identical seeded selection across mirrors.
    assert np.array_equal(tf_indices, torch_indices)
