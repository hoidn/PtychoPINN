"""CI count-intensity adaptation for the PtychoDataContainerTorch path.

The synthetic pipeline hands the
Lightning backend a ``PtychoDataContainerTorch``, not the plain dict the
the public synthetic workflow builds. The legacy normalized-amplitude adapter
serves the dict path, so the container path needs its own bridge to publish the
named CI batch contract.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ptycho_torch.scaling_contract import derive_ci_experiment_statistics
from ptycho_torch.workflows import components as torch_components


N = 8
CHANNELS = 1


def _count_container(n_samples: int = 6, scale: float = 3.0):
    """Build a container whose X is already count intensity, as the CI NPZ is."""

    from ptycho_torch.data_container_bridge import PtychoDataContainerTorch

    rng = np.random.default_rng(0)
    counts = (
        rng.random((n_samples, N, N, CHANNELS)).astype(np.float32) * 1000.0
    )
    grouped = {
        "X_full": counts,
        "Y": np.ones((n_samples, N, N, CHANNELS), dtype=np.complex64),
        "coords_relative": np.zeros((n_samples, 1, 2, CHANNELS), dtype=np.float32),
        "coords_offsets": np.zeros((n_samples, 1, 2, 1), dtype=np.float64),
        "nn_indices": np.arange(n_samples * CHANNELS, dtype=np.int32).reshape(
            n_samples, CHANNELS
        ),
    }
    probe = (np.ones((N, N), dtype=np.complex64) * scale)
    container = PtychoDataContainerTorch(grouped, probe)
    # Mirror synthetic_pipeline.py: X_full is RMS-normalized by
    # RawData.generate_grouped_data; the raw grouped counts are retained
    # separately.
    container.raw_grouped_diffraction = counts
    norm = np.float32(
        np.sqrt((N / 2) ** 2 / np.mean(np.square(counts.astype(np.float64)).sum(axis=(1, 2))))
    )
    container.X = torch.from_numpy(counts * norm).to(torch.float32)
    return container


def test_container_bridge_publishes_the_named_ci_fields():
    container = _count_container()

    statistics = torch_components.attach_container_ci_fields(
        container,
        N=N,
        probe_scale=4.0,
    )

    for name in (
        "measured_intensity",
        "rms_input_scale",
        "mean_measured_intensity",
        "probe_training",
        "probe_physical",
        "probe_normalization",
    ):
        assert getattr(container, name) is not None, name
    assert statistics.rms_input_scale is not None
    assert statistics.mean_measured_intensity is not None


def test_container_bridge_uses_raw_counts_not_the_normalized_images():
    """container.X is RMS-normalized by the grouping path and must not be used.

    Using it would apply the input scaling twice and put the Poisson NLL on a
    non-physical scale (the defect that produced count-contract-run-02).
    """

    container = _count_container()
    raw_counts = torch.from_numpy(container.raw_grouped_diffraction).to(torch.float32)
    normalized_x = container.X.clone()
    original_probe = container.probe.clone()
    assert not torch.allclose(raw_counts, normalized_x)

    torch_components.attach_container_ci_fields(container, N=N, probe_scale=4.0)

    torch.testing.assert_close(container.measured_intensity, raw_counts)
    # images and measured_intensity must be the same physical array, as in the
    # mmap PtychoDataset path.
    torch.testing.assert_close(container.X, raw_counts)
    torch.testing.assert_close(
        container.probe_physical.reshape(N, N),
        original_probe.reshape(N, N),
    )


def test_container_bridge_fails_closed_without_the_raw_measurement():
    container = _count_container()
    container.raw_grouped_diffraction = None

    with pytest.raises(ValueError, match="raw_grouped_diffraction"):
        torch_components.attach_container_ci_fields(
            container, N=N, probe_scale=4.0
        )


def test_container_bridge_statistics_match_the_canonical_derivation():
    container = _count_container()

    torch_components.attach_container_ci_fields(container, N=N, probe_scale=4.0)

    expected = derive_ci_experiment_statistics(
        torch.from_numpy(_count_container().raw_grouped_diffraction)
        .to(torch.float32)
        .permute(0, 3, 1, 2),
        N,
    )
    torch.testing.assert_close(
        container.rms_input_scale, expected.rms_input_scale
    )
    torch.testing.assert_close(
        container.mean_measured_intensity, expected.mean_measured_intensity
    )


def test_validation_container_reuses_frozen_training_statistics():
    train = _count_container(n_samples=6)
    validation = _count_container(n_samples=3)

    training_statistics = torch_components.attach_container_ci_fields(
        train, N=N, probe_scale=4.0
    )
    torch_components.attach_container_ci_fields(
        validation,
        N=N,
        probe_scale=4.0,
        statistics=training_statistics,
    )

    torch.testing.assert_close(
        validation.rms_input_scale, train.rms_input_scale
    )
    torch.testing.assert_close(
        validation.mean_measured_intensity, train.mean_measured_intensity
    )


def test_container_bridge_exposes_finalized_statistics_to_the_backend():
    """_get_finalized_ci_statistics accepts only a dict or get_ci_statistics()."""

    container = _count_container()

    torch_components.attach_container_ci_fields(container, N=N, probe_scale=4.0)

    statistics = torch_components._get_finalized_ci_statistics(container)
    assert set(statistics) == {"rms_input_scale", "mean_measured_intensity"}


def test_container_bridge_removes_legacy_amplitude_scale_constants():
    """CI uses named physical quantities; legacy generic scales are not sources.

    They must be REMOVED, not nulled: ``train_cdi_model_torch`` gates on
    ``hasattr(train_container, 'physics_scaling_constant')`` and then calls
    ``torch.as_tensor`` on it, so a None-valued attribute raises
    ``RuntimeError: Could not infer dtype of NoneType``.
    """

    container = _count_container()
    container.physics_scaling_constant = torch.ones(1, 1, 1)
    container.rms_scaling_constant = torch.ones(1, 1, 1)

    torch_components.attach_container_ci_fields(container, N=N, probe_scale=4.0)

    assert not hasattr(container, "physics_scaling_constant")
    assert not hasattr(container, "rms_scaling_constant")


def test_adapted_container_survives_the_post_training_scale_readback():
    """Reproduces the count-contract-run-01 crash after epoch 5."""

    container = _count_container()
    container.physics_scaling_constant = torch.ones(1, 1, 1)

    torch_components.attach_container_ci_fields(container, N=N, probe_scale=4.0)

    # Verbatim shape of the guarded read in train_cdi_model_torch.
    if hasattr(container, "physics_scaling_constant"):
        torch.as_tensor(container.physics_scaling_constant)


def test_adapted_container_satisfies_the_ci_dataset_required_fields():
    container = _count_container()
    torch_components.attach_container_ci_fields(container, N=N, probe_scale=4.0)

    for name in (
        "measured_intensity",
        "rms_input_scale",
        "mean_measured_intensity",
        "probe_training",
        "probe_physical",
        "probe_normalization",
    ):
        value = getattr(container, name)
        assert isinstance(value, torch.Tensor), name
        assert torch.isfinite(value.real if value.is_complex() else value).all()


def test_container_bridge_rejects_a_degenerate_measurement():
    container = _count_container()
    container.raw_grouped_diffraction = np.zeros_like(
        container.raw_grouped_diffraction
    )

    with pytest.raises(ValueError, match="degenerate|positive|nonzero"):
        torch_components.attach_container_ci_fields(
            container, N=N, probe_scale=4.0
        )
