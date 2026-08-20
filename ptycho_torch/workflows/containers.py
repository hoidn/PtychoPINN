"""Torch data-container adapters and the RAM container factory.

Publishes the CI batch contract onto ``PtychoDataContainerTorch`` and builds the
retained RAM container from ``RawData``. No serialization and no Lightning glue
live here.
"""
import logging
from typing import Optional, Union

from ptycho.config.config import TrainingConfig
from ptycho.grouping import group_from_config
from ptycho.metadata import MetadataManager
from ptycho.raw_data import RawData
from ptycho_torch.data_container_bridge import PtychoDataContainerTorch
from ptycho_torch.dataloader import PtychoDataset
from ptycho_torch.scaling_contract import (
    CIExperimentStatistics,
    derive_ci_experiment_statistics,
)

logger = logging.getLogger("ptycho_torch.workflows.components")

def _canonicalize_ci_probe_modes(probe, N: int):
    """Validate CI probe layouts and move a trailing singleton to mode-first."""
    expected_spatial_shape = (N, N)
    if probe.ndim == 2:
        if tuple(probe.shape) != expected_spatial_shape:
            raise ValueError(
                "CI probe must have shape (N,N), (N,N,1), or (P,N,N); "
                f"got {tuple(probe.shape)} for N={N}."
            )
        return probe

    if probe.ndim == 3:
        if tuple(probe.shape) == (N, N, 1):
            return probe.permute(2, 0, 1).contiguous()
        if probe.shape[0] > 0 and tuple(probe.shape[-2:]) == expected_spatial_shape:
            return probe

    raise ValueError(
        "CI probe must have shape (N,N), (N,N,1), or (P,N,N); "
        f"got {tuple(probe.shape)} for N={N}."
    )


def _get_finalized_ci_statistics(container):
    """Read finalized CI statistics from native datasets or dict containers."""
    statistics_getter = getattr(container, "get_ci_statistics", None)
    if callable(statistics_getter):
        statistics = statistics_getter()
        if statistics is None:
            raise RuntimeError(
                "Native CI training dataset has no finalized training statistics."
            )
    elif isinstance(container, dict):
        statistics = {
            field_name: container.get(field_name)
            for field_name in ("rms_input_scale", "mean_measured_intensity")
        }
    else:
        statistics = {}

    for field_name in ("rms_input_scale", "mean_measured_intensity"):
        if statistics.get(field_name) is None:
            raise RuntimeError(
                "Standalone CI training requires finalized training statistics; "
                f"missing {field_name!r} on the training container."
            )
    return statistics


def _resolve_nphotons(data, config):
    metadata = getattr(data, "metadata", None)
    if metadata is not None:
        return MetadataManager.get_nphotons(metadata), "metadata"
    return getattr(getattr(config, "data", config), "nphotons", 1e9), "config"


def _attach_physics_scale(container, config, nphotons_source: Optional[str] = None):
    from ptycho_torch import helper as hh

    nphotons, source = _resolve_nphotons(container, config)
    if nphotons_source is not None:
        source = nphotons_source

    scale = hh.derive_intensity_scale_from_amplitudes(container.X, nphotons)
    container.physics_scaling_constant = scale.view(1, 1, 1)
    container.nphotons_source = source
    container.nphotons_resolved = nphotons
    return scale, source


def _get_container_tensor_required(container, name: str):
    import numpy as np
    import torch

    value = getattr(container, name, None)
    if value is None:
        raise ValueError(f"CI container adaptation requires {name!r}.")
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(np.asarray(value))
    return value


def attach_container_ci_fields(
    container,
    *,
    N: int,
    probe_scale: float = 4.0,
    statistics: Optional[CIExperimentStatistics] = None,
    probe_mask: bool = False,
    probe_mask_sigma: float = 1.0,
    probe_mask_diameter: Optional[float] = None,
) -> CIExperimentStatistics:
    """Publish physical count fields on an in-memory Torch data container.

    ``RawData.generate_grouped_data`` places its normalized network input in
    ``container.X``. The physical count measurement retained by the shared
    training service is therefore the only valid source for CI images and the
    Poisson target. The stored probe is already the CI-scaled physical probe.
    """

    import torch

    from ptycho_torch import helper as hh

    if getattr(container, "raw_grouped_diffraction", None) is None:
        raise ValueError(
            "CI count-intensity training requires 'raw_grouped_diffraction'; "
            "container.X is normalized and cannot be the Poisson target"
        )
    measured_intensity = _get_container_tensor_required(
        container,
        "raw_grouped_diffraction",
    )
    if not torch.is_floating_point(measured_intensity):
        measured_intensity = measured_intensity.to(torch.float32)
    if measured_intensity.ndim != 4:
        raise ValueError(
            "CI raw_grouped_diffraction must have shape (B, H, W, C); got "
            f"{tuple(measured_intensity.shape)}"
        )
    if not bool(torch.isfinite(measured_intensity).all()):
        raise ValueError("CI raw_grouped_diffraction must contain finite values")
    if bool((measured_intensity < 0).any()):
        raise ValueError("CI raw_grouped_diffraction must contain nonnegative counts")

    probe = _get_container_tensor_required(container, "probe")
    probe_physical = _canonicalize_ci_probe_modes(
        probe.to(device=measured_intensity.device),
        N,
    )
    statistics = statistics or derive_ci_experiment_statistics(
        measured_intensity.permute(0, 3, 1, 2),
        N,
    )

    probe_training_np, probe_normalization = hh.normalize_probe_like_tf(
        probe_physical.detach().cpu().numpy(),
        probe_scale=probe_scale,
        probe_mask=probe_mask,
        probe_mask_sigma=probe_mask_sigma,
        probe_mask_diameter=probe_mask_diameter,
    )
    probe_training = torch.as_tensor(
        probe_training_np,
        device=probe_physical.device,
    ).to(probe_physical.dtype)
    probe_normalization_tensor = measured_intensity.new_tensor(
        probe_normalization
    )

    container.X = measured_intensity
    container.measured_intensity = measured_intensity
    container.observed_images = measured_intensity
    container.probe = probe_physical
    container.probe_physical = probe_physical
    container.probe_training = probe_training
    container.probe_normalization = probe_normalization_tensor
    container.scaling_constant = probe_normalization_tensor.view(1, 1, 1)
    container.rms_input_scale = statistics.rms_input_scale
    container.mean_measured_intensity = statistics.mean_measured_intensity

    for legacy_name in ("rms_scaling_constant", "physics_scaling_constant"):
        if hasattr(container, legacy_name):
            try:
                delattr(container, legacy_name)
            except AttributeError:
                setattr(container, legacy_name, None)

    container.get_ci_statistics = lambda: {
        "rms_input_scale": statistics.rms_input_scale,
        "mean_measured_intensity": statistics.mean_measured_intensity,
    }
    return statistics


def _adapt_container_for_ci(
    container,
    *,
    data_config,
    model_config,
    statistics: Optional[CIExperimentStatistics] = None,
) -> Optional[CIExperimentStatistics]:
    """Adapt only the in-memory container path to the named CI batch fields."""

    if container is None or isinstance(container, dict):
        return None
    if isinstance(container, PtychoDataset):
        return None
    if getattr(container, "measured_intensity", None) is not None:
        return None
    return attach_container_ci_fields(
        container,
        N=int(data_config.N),
        probe_scale=float(getattr(data_config, "probe_scale", 4.0)),
        statistics=statistics,
        probe_mask=bool(getattr(model_config, "probe_mask", False)),
        probe_mask_sigma=float(getattr(model_config, "probe_mask_sigma", 1.0)),
        probe_mask_diameter=getattr(model_config, "probe_mask_diameter", None),
    )


def _ensure_container(
    data: Union[RawData, 'PtychoDataContainerTorch'],
    config: TrainingConfig
) -> 'PtychoDataContainerTorch':
    """
    Normalize input data to the retained Torch RAM container.

    This helper mirrors the pattern in ptycho.workflows.components.create_ptycho_data_container,
    providing a single normalization pathway for all data types.

    Args:
        data: Input data (RawData or PtychoDataContainerTorch)
        config: TrainingConfig for grouped data generation parameters

    Returns:
        PtychoDataContainerTorch: Normalized container ready for Lightning training

    Raises:
        TypeError: If data is not one of the supported types
        ImportError: If Phase C adapters not available (should not occur in Phase D2.B)

    Implementation Notes:
        - RawData → generate grouped data → PtychoDataContainerTorch
        - PtychoDataContainerTorch → return as-is (already normalized)
    """
    # Case 1: Already a container - return as-is.
    if hasattr(data, 'X') and hasattr(data, 'Y'):  # Duck-type check for PtychoDataContainerTorch
        logger.debug("Input is already PtychoDataContainerTorch, returning as-is")
        if not hasattr(data, 'physics_scaling_constant'):
            _attach_physics_scale(data, config, nphotons_source=None)
        return data

    # Case 2: RawData owns canonical grouping and materializes one RAM carrier.
    if isinstance(data, RawData):
        logger.debug("Generating grouped Torch RAM data from RawData")
        sample_indices = getattr(data, 'sample_indices', None)
        metadata = getattr(data, 'metadata', None)
        grouped_data = group_from_config(
            data,
            config,
            dataset_path=(
                str(config.data.train_data_file)
                if config.data.train_data_file
                else None
            ),
        )
        actual_sample_indices = grouped_data.get('sample_indices')
        if sample_indices is not None and actual_sample_indices is not None:
            import numpy as np
            if not np.array_equal(np.asarray(sample_indices), np.asarray(actual_sample_indices)):
                raise RuntimeError(
                    "Subsample index mismatch between TensorFlow and PyTorch data pipelines. "
                    "Verify that load_data() and the PyTorch backend share the same subsample_seed."
                )
        grouped_data.pop('sample_indices', None)
        import numpy as np
        for key in ('X_full', 'diffraction'):
            if key in grouped_data and grouped_data[key].dtype != np.float32:
                grouped_data[key] = grouped_data[key].astype(np.float32, copy=False)
        probe = data.probeGuess
        container = PtychoDataContainerTorch(grouped_data, probe)
        if metadata is not None:
            container.metadata = metadata
        _attach_physics_scale(container, config, nphotons_source=None)
        return container

    # Case 3: Unknown type
    raise TypeError(
        f"data must be RawData or PtychoDataContainerTorch, got {type(data)}"
    )
