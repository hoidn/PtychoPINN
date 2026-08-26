"""Torch data-container adapters and the RAM container factory.

Publishes the CI batch contract onto ``PtychoDataContainerTorch`` and builds the
retained RAM container from ``RawData``.  No serialization and no Lightning glue
live here.
"""
import logging
from typing import Any, Dict, Optional, Union
from ptycho_torch.config_params import PROBE_SCALE_DEFAULT

from ptycho.metadata import MetadataManager
from ptycho.raw_data import RawData
from ptycho.grouping import group_from_config
from ptycho.config.config import TrainingConfig
from ptycho_torch.scaling_contract import (
    CIExperimentStatistics,
    derive_ci_experiment_statistics,
)
from ptycho_torch.data_container_bridge import PtychoDataContainerTorch
from ptycho_torch.dataloader import PtychoDataset

# Preserves pre-split log provenance: records stay on the components facade logger.
logger = logging.getLogger('ptycho_torch.workflows.components')

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
    return getattr(config, "nphotons", 1e9), "config"


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


def attach_container_ci_fields(
    container,
    *,
    N: int,
    probe_scale: float = PROBE_SCALE_DEFAULT,
    statistics: Optional[CIExperimentStatistics] = None,
    probe_mask: bool = False,
    probe_mask_sigma: float = 1.0,
    probe_mask_diameter: Optional[float] = None,
) -> CIExperimentStatistics:
    """Publish the named CI batch contract on a ``PtychoDataContainerTorch``.

    This performs **no** amplitude-to-count conversion: a CI flat acquisition
    already stores count intensity in
    ``diff3d`` and the calibrated physical probe in ``probeGuess``
    (``docs/specs/spec-ptycho-core.md``), so ``container.probe`` is already
    ``probe_physical``.  The training probe is the ``1/q``-compensated
    normalization of the physical probe.

    ``container.X`` is NOT usable as ``measured_intensity``:
    ``RawData.generate_grouped_data`` RMS-normalizes the measurement into
    ``X_full`` (``ptycho/raw_data.py::normalize_data``, whose contract assumes
    amplitude), so using it would apply the input scaling twice and put the
    Poisson NLL on a non-physical scale.  The grouped raw counts retained as
    ``container.raw_grouped_diffraction`` are used instead, and ``X`` is
    replaced by them so ``images`` and ``measured_intensity`` are both physical
    counts -- the same invariant the mmap ``PtychoDataset`` path maintains.

    Pass ``statistics`` from the finalized TRAINING container so validation and
    inference reuse one immutable experiment scalar pair.
    """

    import torch

    from ptycho_torch import helper as hh

    if getattr(container, "raw_grouped_diffraction", None) is None:
        raise ValueError(
            "CI count-intensity training requires the grouped raw measurement "
            "on the container ('raw_grouped_diffraction'); container.X is "
            "RMS-normalized by RawData.generate_grouped_data and would "
            "double-normalize the Poisson objective."
        )
    measured_intensity = _get_container_tensor_required(
        container, "raw_grouped_diffraction"
    )
    if not torch.is_floating_point(measured_intensity):
        measured_intensity = measured_intensity.to(torch.float32)
    if measured_intensity.ndim != 4:
        raise ValueError(
            "CI container X must have shape (B, H, W, C); got "
            f"{tuple(measured_intensity.shape)}."
        )
    if not bool(torch.isfinite(measured_intensity).all()):
        raise ValueError("CI container X must contain only finite values.")
    if bool((measured_intensity < 0).any()):
        raise ValueError("CI container X must contain nonnegative counts.")

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
    probe_normalization_tensor = measured_intensity.new_tensor(probe_normalization)

    container.measured_intensity = measured_intensity
    container.observed_images = measured_intensity
    # Both the network input and the loss target are physical counts, matching
    # the mmap path where images and measured_intensity are the same array.
    container.X = measured_intensity
    container.probe = probe_physical
    container.probe_physical = probe_physical
    container.probe_training = probe_training
    container.probe_normalization = probe_normalization_tensor
    container.scaling_constant = probe_normalization_tensor.view(1, 1, 1)
    container.rms_input_scale = statistics.rms_input_scale
    container.mean_measured_intensity = statistics.mean_measured_intensity

    # CI uses named physical quantities; legacy generic scales are not sources.
    # Delete rather than null: consumers such as train_cdi_model_torch gate on
    # ``hasattr``, which a None-valued attribute would still satisfy.  This
    # mirrors the dict adapter, which pops the same two keys.
    for legacy in ("rms_scaling_constant", "physics_scaling_constant"):
        if hasattr(container, legacy):
            try:
                delattr(container, legacy)
            except AttributeError:
                # Class-level attribute; shadow it with None as a last resort.
                setattr(container, legacy, None)

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
    """Adapt a non-dict container in place; return its finalized statistics.

    Dict containers and native ``PtychoDataset`` instances already own the CI
    contract, so they are left untouched.
    """

    if container is None or isinstance(container, dict):
        return None
    if isinstance(container, PtychoDataset):
        return None
    if getattr(container, "measured_intensity", None) is not None:
        # Already carries the CI contract (adapted by an upstream caller).
        return None
    return attach_container_ci_fields(
        container,
        N=int(data_config.N),
        probe_scale=float(getattr(data_config, "probe_scale", PROBE_SCALE_DEFAULT)),
        statistics=statistics,
        probe_mask=bool(getattr(model_config, "probe_mask", False)),
        probe_mask_sigma=float(getattr(model_config, "probe_mask_sigma", 1.0)),
        probe_mask_diameter=getattr(model_config, "probe_mask_diameter", None),
    )


def _get_container_tensor_required(container, name: str):
    import numpy as np
    import torch

    value = getattr(container, name, None)
    if value is None:
        raise ValueError(f"CI container adaptation requires {name!r}.")
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(np.asarray(value))
    return value


def create_torch_data_container(
    data: Union[RawData, 'PtychoDataContainerTorch'],
    config: TrainingConfig
) -> 'PtychoDataContainerTorch':
    """
    Build the retained Torch RAM container from raw or container data.

    Public container factory for embedders (Ptychodus, notebooks): consumes an
    in-memory ``RawData`` (or an already-normalized ``PtychoDataContainerTorch``)
    and returns a ``PtychoDataContainerTorch`` without any flat-NPZ round-trip.
    Grouping delegates to ``ptycho.grouping.group_from_config``, the single
    backend-neutral group policy authority (Phase 1).

    Args:
        data: ``RawData`` (grouped in memory) or ``PtychoDataContainerTorch``
            (returned as-is after attaching a physics scale when missing).
        config: ``TrainingConfig`` supplying grouping and scale parameters.

    Returns:
        PtychoDataContainerTorch: Normalized container ready for Lightning
        training or inference.

    Raises:
        TypeError: If data is neither ``RawData`` nor a
            ``PtychoDataContainerTorch``-shaped value.
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
        metadata = getattr(data, 'metadata', None)
        grouped_data = group_from_config(
            data,
            config,
            dataset_path=str(config.train_data_file) if config.train_data_file else None,
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


