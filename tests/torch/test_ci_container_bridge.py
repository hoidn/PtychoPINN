"""Raw count-intensity adaptation for the in-memory Torch container path."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ptycho_torch.scaling_contract import derive_ci_experiment_statistics
from ptycho_torch.workflows import components


N = 8


def _grouped_counts(n_samples: int = 6) -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(0)
    counts = (
        rng.random((n_samples, N, N, 1)).astype(np.float32) * 1000.0
    )
    normalization = np.float32(
        np.sqrt(
            (N / 2) ** 2
            / np.mean(
                np.square(counts.astype(np.float64)).sum(axis=(1, 2))
            )
        )
    )
    grouped = {
        "diffraction": counts,
        "X_full": np.ascontiguousarray(counts * normalization),
        "Y": None,
        "coords_relative": np.zeros(
            (n_samples, 1, 2, 1), dtype=np.float32
        ),
        "coords_offsets": np.zeros(
            (n_samples, 1, 2, 1), dtype=np.float64
        ),
        "nn_indices": np.arange(n_samples, dtype=np.int32).reshape(-1, 1),
    }
    return grouped, counts


def _count_container(n_samples: int = 6):
    from ptycho_torch.data_container_bridge import PtychoDataContainerTorch

    grouped, counts = _grouped_counts(n_samples)
    container = PtychoDataContainerTorch(
        grouped,
        np.ones((N, N), dtype=np.complex64) * 3.0,
    )
    container.raw_grouped_diffraction = np.array(counts, copy=True)
    return container, counts


def test_container_ci_bridge_uses_raw_counts_not_normalized_x():
    container, counts = _count_container()
    normalized_x = container.X.clone()
    raw = torch.from_numpy(counts)
    assert not torch.allclose(normalized_x, raw)

    statistics = components.attach_container_ci_fields(
        container,
        N=N,
        probe_scale=4.0,
    )

    torch.testing.assert_close(container.X, raw)
    torch.testing.assert_close(container.measured_intensity, raw)
    torch.testing.assert_close(container.observed_images, raw)
    assert statistics.rms_input_scale is container.rms_input_scale
    assert (
        statistics.mean_measured_intensity
        is container.mean_measured_intensity
    )
    for name in (
        "probe_training",
        "probe_physical",
        "probe_normalization",
        "rms_input_scale",
        "mean_measured_intensity",
    ):
        assert isinstance(getattr(container, name), torch.Tensor), name


def test_container_ci_bridge_fails_closed_without_raw_counts():
    container, _ = _count_container()
    del container.raw_grouped_diffraction

    with pytest.raises(ValueError, match="raw_grouped_diffraction"):
        components.attach_container_ci_fields(container, N=N)


def test_container_ci_bridge_statistics_use_raw_counts():
    container, counts = _count_container()

    components.attach_container_ci_fields(container, N=N)

    expected = derive_ci_experiment_statistics(
        torch.from_numpy(counts).permute(0, 3, 1, 2),
        N,
    )
    torch.testing.assert_close(container.rms_input_scale, expected.rms_input_scale)
    torch.testing.assert_close(
        container.mean_measured_intensity,
        expected.mean_measured_intensity,
    )


def test_validation_container_reuses_training_ci_statistics():
    train, _ = _count_container(n_samples=6)
    validation, _ = _count_container(n_samples=3)

    training_statistics = components.attach_container_ci_fields(train, N=N)
    components.attach_container_ci_fields(
        validation,
        N=N,
        statistics=training_statistics,
    )

    assert validation.rms_input_scale is train.rms_input_scale
    assert (
        validation.mean_measured_intensity
        is train.mean_measured_intensity
    )


def _payload(*, ci: bool):
    from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig

    return SimpleNamespace(
        pt_data_config=DataConfig(
            N=N,
            gridsize=1,
            scale_contract_version=("ci_intensity_v2" if ci else "legacy_v1"),
            measurement_domain=(
                "count_intensity" if ci else "normalized_amplitude"
            ),
        ),
        pt_model_config=ModelConfig(
            mode="Unsupervised",
            architecture="cnn",
            physics_forward_mode=("rectangular_scaled" if ci else "amplitude"),
            cnn_output_mode=("real_imag" if ci else "amp_phase"),
            amplitude_physics_gain=1.0,
        ),
        pt_training_config=TrainingConfig(
            batch_size=2,
            torch_loss_mode=("poisson" if ci else "mae"),
        ),
        execution_config=SimpleNamespace(
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
        ),
    )


def _public_config():
    return SimpleNamespace(
        sequential_sampling=True,
        sampling=SimpleNamespace(subsample_seed=11),
    )


def test_lightning_loader_boundary_adapts_ci_train_and_validation_containers():
    train, train_counts = _count_container(n_samples=6)
    validation, validation_counts = _count_container(n_samples=3)

    train_loader, validation_loader = components._build_lightning_dataloaders(
        train,
        validation,
        _public_config(),
        payload=_payload(ci=True),
    )

    torch.testing.assert_close(train.X, torch.from_numpy(train_counts))
    torch.testing.assert_close(validation.X, torch.from_numpy(validation_counts))
    assert validation.rms_input_scale is train.rms_input_scale
    assert train_loader.dataset.ci_active is True
    assert validation_loader.dataset.ci_active is True


def test_lightning_loader_boundary_leaves_amplitude_x_unchanged():
    train, _ = _count_container(n_samples=6)
    before = train.X.clone()

    train_loader, _ = components._build_lightning_dataloaders(
        train,
        None,
        _public_config(),
        payload=_payload(ci=False),
    )

    torch.testing.assert_close(train.X, before)
    assert not hasattr(train, "measured_intensity")
    assert train_loader.dataset.ci_active is False
