"""
PyTorch workflow orchestration layer — parity with ptycho/workflows/components.py.

This module provides PyTorch equivalents of the TensorFlow workflow orchestration
functions, enabling transparent backend selection from Ptychodus per the reconstructor
contract defined in specs/ptychodus_api_spec.md §4.

Architecture Role:
This module mirrors ptycho.workflows.components, sitting at the same orchestration
layer and providing identical entry point signatures. The key differences are:
1. Uses PyTorch backend (ptycho_torch.model, Lightning trainer)
2. Leverages config_bridge for TensorFlow dataclass compatibility
3. Uses canonical RawData grouping plus the retained Torch RAM container
4. Implements PyTorch-specific persistence via ModelManager or TorchModelManager

Critical Requirements (CONFIG-001 + spec §4.5):
- Modern Torch entry points pass resolved configuration explicitly
- Module MUST be torch-optional (importable when PyTorch unavailable)
- Signatures MUST match TensorFlow equivalents for transparent backend selection

Torch-Optional Design:
- Guarded imports using TORCH_AVAILABLE flag (from ptycho_torch.config_params)
- All torch-specific types aliased to fallback types when torch unavailable
- Entry points raise NotImplementedError when backend unavailable (Phases D2.B/C)

Core Workflow Functions (Scaffold):
Entry Points:
    - run_cdi_example_torch(): Complete training → reconstruction → visualization
    - train_cdi_model_torch(): Orchestrate data prep, probe setup, and Lightning training
    - load_inference_bundle_torch(): Load trained model for inference (archive compat)

Integration Points (Phase D2.B/C TODO):
- Config Bridge: ptycho_torch.config_bridge for dataclass translation
- Data Pipeline: canonical RawData grouping + PtychoDataContainerTorch
- Training: Lightning trainer + MLflow autologging
- Persistence: TorchModelManager or extended ModelManager (Phase D3)

Example Usage (Post Phase D2.B/C):
    >>> from ptycho_torch.workflows.components import run_cdi_example_torch
    >>> from ptycho.config.config import TrainingConfig, ModelConfig
    >>>
    >>> # Configure via TensorFlow dataclasses (config bridge handles translation)
    >>> config = TrainingConfig(model=ModelConfig(N=64, gridsize=2), ...)
    >>>
    >>> # Execute PyTorch pipeline (identical signature to TensorFlow version)
    >>> amplitude, phase, results = run_cdi_example_torch(
    ...     train_data, test_data, config, do_stitching=True
    ... )

Artifacts:
- Phase D2.A scaffold: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T091450Z/
- Design decisions: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T085431Z/phase_d_decision.md
"""

# Standard library imports (no torch dependency)
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import io
import logging
import math
from numbers import Integral
from pathlib import Path
import zipfile
from typing import Union, Optional, Tuple, Dict, Any

# Core imports (always available)
from ptycho import params
from ptycho.config.config import TrainingConfig, InferenceConfig, PyTorchExecutionConfig
from ptycho.config.legacy_state import (
    transactional_legacy_params,
)
from ptycho.metadata import MetadataManager
from ptycho.raw_data import RawData
from ptycho_torch.scaling_contract import (
    AmplitudePhysicsGainRecord,
    CI_SCALE_CONTRACT,
    LEGACY_SCALE_CONTRACT,
    NORMALIZED_AMPLITUDE,
    CIExperimentStatistics,
    adapt_normalized_amplitude_to_ci,
    amplitude_physics_gain_record_from_json,
    amplitude_physics_gain_record_to_json,
    derive_ci_experiment_statistics,
    ci_scaling_active,
    resolve_scale_contract,
    validate_amplitude_physics_gain,
    validate_scale_contract,
)
from ptycho_torch.object_compatibility import resolve_model_object_compatibility
from ptycho_torch.rect_s1s2_initialization import (
    RECT_S1S2_DOSE_CLOSURE_PATTERNS,
    RectS1S2InitializationRecord,
)
from ptycho_torch.rect_s1s2_sampling import (
    SelectedDoseClosureRow,
    _base_row_for_logical,
    build_dose_closure_sample_plan,
)

# PyTorch imports (now mandatory per Phase F3.1/F3.2)
try:
    from ptycho_torch.config_factory import TrainingPayload, InferencePayload
    from ptycho_torch.data_container_bridge import PtychoDataContainerTorch
    from ptycho_torch.model_manager import (
        _read_torch_bundle_manifest_and_params,
        _reconstruct_torch_bundle_explicit,
        save_torch_bundle,
    )
    from ptycho_torch.dataloader import (
        PtychoDataset,
        _PtychoContainerDataset,
        build_ptycho_loader,
    )
    from ptycho_torch.train_utils import (
        PrebuiltPtychoDataModule,
        is_spawn_strategy,
    )
    from ptycho_torch.runtime_provenance import (
        build_effective_runtime as _build_effective_runtime,
        write_effective_runtime_json,
    )
except ImportError as e:
    raise RuntimeError(
        "PyTorch backend modules not available. "
        "Ensure the project's Torch dependencies are installed. "
        "Original error: " + str(e)
    ) from e

# Set up logging
logger = logging.getLogger(__name__)

_BUNDLE_SCALING_METADATA = "torch_scaling_metadata.pt"
_BUNDLE_AMPLITUDE_PHYSICS_GAIN_RECORD = "amplitude_physics_gain_record.json"


def _validate_training_execution_input(
    execution_config: Optional[Any],
    resolved_payload: Optional[TrainingPayload],
) -> None:
    """Validate unresolved workflow input before any legacy-state mutation."""
    if resolved_payload is not None:
        if execution_config is not None:
            raise TypeError(
                "execution_config must be omitted when resolved_payload owns "
                "the resolved PyTorchExecutionConfig"
            )
        return

    from ptycho_torch.execution_request import normalize_execution_input

    normalize_execution_input(execution_config, mode="training")


def _persist_bundle_scaling_metadata(
    archive_path: Path,
    model,
    *,
    amplitude_physics_gain_record: Optional[
        AmplitudePhysicsGainRecord
    ] = None,
) -> None:
    """Append the torch config and frozen CI statistics needed for strict reload."""
    statistics = model.get_ci_statistics()
    profile = resolve_scale_contract(
        model.data_config.scale_contract_version,
        model.data_config.measurement_domain,
    )
    ci_bundle = ci_scaling_active(model.model_config) and profile.version == CI_SCALE_CONTRACT
    if ci_bundle and statistics is None:
        raise ValueError(
            "Cannot persist a CI bundle without frozen training ci_statistics."
        )
    serialized_statistics = None
    if statistics is not None:
        serialized_statistics = {
            name: value.detach().cpu().reshape(-1).tolist()
            for name, value in statistics.items()
        }
    import torch

    from ptycho_torch.artifact_schema import (
        CURRENT_ARTIFACT_SCHEMA_VERSION,
        TORCH_ARTIFACT_BACKEND,
        encode_artifact_identity,
        validate_torch_bundle_manifest,
    )
    from ptycho_torch.config_bridge import to_model_config
    from ptycho_torch.model_spec import derive_model_spec

    model_spec = getattr(model, "_model_spec", None)
    if model_spec is None:
        model_spec = derive_model_spec(
            to_model_config(model.data_config, model.model_config),
            model.model_config,
            model.data_config,
            parity_scale_mode=getattr(model, "parity_scale_mode", "off"),
            parity_fixed_delta=float(model.hparams.get("parity_fixed_delta", 0.0)),
            parity_init_scheme=model.hparams.get("parity_init_scheme", "default"),
        )
    payload = encode_artifact_identity(
        model_spec,
        model.data_config,
        model.training_config,
        model.inference_config,
        ci_statistics=serialized_statistics,
    )
    sidecar_json = (
        amplitude_physics_gain_record_to_json(amplitude_physics_gain_record)
        if amplitude_physics_gain_record is not None
        else None
    )
    buffer = io.BytesIO()
    torch.save(payload, buffer)

    import dill
    import os
    import tempfile

    with zipfile.ZipFile(archive_path, "r") as archive:
        members = {
            info.filename: archive.read(info.filename)
            for info in archive.infolist()
            if info.filename
            not in {
                "manifest.dill",
                _BUNDLE_SCALING_METADATA,
                _BUNDLE_AMPLITUDE_PHYSICS_GAIN_RECORD,
            }
        }
        manifest = dill.loads(archive.read("manifest.dill"))
    validate_torch_bundle_manifest(manifest)
    manifest.update(
        backend=TORCH_ARTIFACT_BACKEND,
        artifact_schema_version=CURRENT_ARTIFACT_SCHEMA_VERSION,
    )
    handle, temporary_name = tempfile.mkstemp(
        prefix=archive_path.name,
        suffix=".tmp",
        dir=archive_path.parent,
    )
    os.close(handle)
    try:
        with zipfile.ZipFile(temporary_name, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.dill", dill.dumps(manifest))
            for name, content in members.items():
                archive.writestr(name, content)
            archive.writestr(_BUNDLE_SCALING_METADATA, buffer.getvalue())
            if sidecar_json is not None:
                archive.writestr(
                    _BUNDLE_AMPLITUDE_PHYSICS_GAIN_RECORD,
                    sidecar_json.encode("utf-8"),
                )
        os.replace(temporary_name, archive_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _read_bundle_scaling_metadata(archive_path: Path):
    import torch

    if not archive_path.is_file():
        return None
    with zipfile.ZipFile(archive_path, "r") as archive:
        if _BUNDLE_SCALING_METADATA not in archive.namelist():
            return None
        return torch.load(
            io.BytesIO(archive.read(_BUNDLE_SCALING_METADATA)),
            map_location="cpu",
            weights_only=False,
        )


def _read_bundle_amplitude_physics_gain_record(
    archive_path: Path,
) -> Optional[AmplitudePhysicsGainRecord]:
    if not archive_path.is_file():
        return None
    with zipfile.ZipFile(archive_path, "r") as archive:
        if _BUNDLE_AMPLITUDE_PHYSICS_GAIN_RECORD not in archive.namelist():
            return None
        encoded = archive.read(_BUNDLE_AMPLITUDE_PHYSICS_GAIN_RECORD)
    return amplitude_physics_gain_record_from_json(encoded)


def _decode_bundle_metadata(metadata):
    from ptycho_torch.artifact_schema import (
        ARTIFACT_SCHEMA_V1_VERSION,
        CURRENT_ARTIFACT_SCHEMA_VERSION,
        decode_artifact_identity,
        upgrade_unversioned_sections,
    )

    schema = metadata.get("schema_version") if isinstance(metadata, dict) else None
    if schema in {
        ARTIFACT_SCHEMA_V1_VERSION,
        CURRENT_ARTIFACT_SCHEMA_VERSION,
    }:
        return decode_artifact_identity(metadata)
    if schema == "ci-entrypoints-v1":
        return upgrade_unversioned_sections(
            data_config=metadata["data_config"],
            model_config=metadata["model_config"],
            training_config=metadata["training_config"],
            inference_config=metadata["inference_config"],
            ci_statistics=metadata.get("ci_statistics"),
        )
    raise ValueError(
        f"unsupported wts.h5.zip Torch metadata schema {schema!r}"
    )


def _strictly_reconstruct_bundle_model(archive_path: Path, identity, model_name: str):
    import torch
    from ptycho_torch.application_factory import build_ptychopinn_application

    model = build_ptychopinn_application(
        identity.model_spec,
        identity.data_config,
        identity.training_config,
        identity.inference_config,
    )
    with zipfile.ZipFile(archive_path, "r") as archive:
        try:
            state_dict = torch.load(
                io.BytesIO(archive.read(f"{model_name}/model.pth")),
                map_location="cpu",
                weights_only=False,
            )
        except KeyError as exc:
            raise RuntimeError(
                f"Bundle weights are missing for model '{model_name}'. Regenerate "
                "the bundle from a successful training result."
            ) from exc
    if not isinstance(state_dict, dict) or "_sentinel" in state_dict:
        raise RuntimeError(
            f"Bundle weights for model '{model_name}' are not a trained state_dict. "
            "Regenerate the bundle from a successful training result."
        )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Bundle architecture-era incompatibility for model '{model_name}': "
            "strict physics/model weight "
            "loading failed. Do not use strict=False; regenerate this bundle with "
            f"the current architecture. Original error: {exc}"
        ) from exc

    profile = resolve_scale_contract(
        identity.data_config.scale_contract_version,
        identity.data_config.measurement_domain,
    )
    ci_bundle = (
        ci_scaling_active(model.model_config)
        and profile.version == CI_SCALE_CONTRACT
    )
    statistics = identity.ci_statistics
    if ci_bundle and statistics is None:
        raise ValueError(
            "CI bundle is missing frozen training ci_statistics; regenerate the bundle."
        )
    if statistics is not None:
        model.register_ci_statistics(statistics)
    return model


def _reconstruct_inference_bundle_explicit(
    archive_path: Path,
    zip_path: Path,
    *,
    manifest: dict,
    params_dict: dict,
    identity: Optional[Any],
    explicit_profile: Optional[Tuple[str, str]],
    model_name: str,
) -> Tuple[Dict[str, Any], dict, Optional[Any]]:
    """Reconstruct a decoded bundle without consulting or mutating params.cfg."""
    decoded_params = dict(params_dict)
    available_models = manifest["models"]

    if identity is None:
        required_profile = (
            LEGACY_SCALE_CONTRACT,
            NORMALIZED_AMPLITUDE,
        )
        if explicit_profile != required_profile:
            raise ValueError(
                "This metadata-free bundle is provenance-known legacy. Supply both "
                "scale_contract_version='legacy_v1' and "
                "measurement_domain='normalized_amplitude'."
            )
        models_dict, _ = _reconstruct_torch_bundle_explicit(
            str(archive_path),
            manifest=manifest,
            params_dict=params_dict,
            model_name=model_name,
        )
        for loaded_model in models_dict.values():
            loaded_model.data_config.scale_contract_version = (
                LEGACY_SCALE_CONTRACT
            )
            loaded_model.data_config.measurement_domain = NORMALIZED_AMPLITUDE
        decoded_params["scale_contract_version"] = LEGACY_SCALE_CONTRACT
        decoded_params["measurement_domain"] = NORMALIZED_AMPLITUDE
        return models_dict, decoded_params, None

    persisted_profile = resolve_scale_contract(
        identity.data_config.scale_contract_version,
        identity.data_config.measurement_domain,
    )
    if explicit_profile is not None and explicit_profile != (
        persisted_profile.version,
        persisted_profile.measurement_domain,
    ):
        raise ValueError(
            "Explicit bundle profile overrides contradict persisted metadata."
        )
    models_dict = {
        archived_model_name: _strictly_reconstruct_bundle_model(
            zip_path,
            identity,
            archived_model_name,
        )
        for archived_model_name in available_models
    }
    decoded_params["scale_contract_version"] = persisted_profile.version
    decoded_params["measurement_domain"] = (
        persisted_profile.measurement_domain
    )
    decoded_params["ci_statistics"] = identity.ci_statistics
    return models_dict, decoded_params, identity


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


def derive_dict_physics_scale(
    container: Dict[str, Any], nphotons: float, mode: str
) -> Optional[Any]:
    """Attach an absolute photon-count physics scale to a plain dict container.

    Sibling to ``_attach_physics_scale`` for the grid-lines dict-container path
    (``run_torch_training``), which builds a plain dict and therefore never
    reaches ``_ensure_container`` -> ``_attach_physics_scale``. ``auto``
    reproduces the native convention (``S =
    derive_intensity_scale_from_amplitudes(amplitudes, nphotons)``) applied to
    ``container['observed_images']`` — the loss-side raw diffraction, which
    stays unconditioned in every ``--input-conditioning-mode`` — rather than
    ``X``, which may carry appended non-physical conditioning channels that
    would corrupt S. The loss lifts ``pred`` and ``observed_images``
    (model.py compute_loss), so S must calibrate exactly that array.
    ``off`` leaves ``physics_scaling_constant`` absent so the existing
    ``_get_tensor``/``_select_scale`` wiring defaults it to 1.0 (today's
    behavior, unchanged).

    Args:
        container: Plain dict container with an 'observed_images' key
            (normalized raw diffraction amplitudes).
        nphotons: Photon count to derive the scale against.
        mode: 'auto' or 'off'.

    Returns:
        The derived scale tensor (float32, shape (1, 1, 1) before assignment)
        in 'auto' mode, else None.
    """
    if mode == "auto":
        from ptycho_torch import helper as hh

        scale = hh.derive_intensity_scale_from_amplitudes(
            container["observed_images"], nphotons
        )
        container["physics_scaling_constant"] = scale.view(1, 1, 1).float()
        return scale
    if mode == "off":
        return None
    raise ValueError(f"Unknown count_scale_mode {mode!r}; expected 'auto' or 'off'")


@dataclass(frozen=True)
class NormalizedAmplitudeCIDictAdapter:
    """Adapt one grid-lines amplitude dict into the named CI batch contract."""

    count_amplitude_scale: Any
    N: int
    statistics: Optional[CIExperimentStatistics] = None
    probe_scale: float = 4.0
    probe_mask: bool = False
    probe_mask_sigma: float = 1.0
    probe_mask_diameter: Optional[float] = None

    def adapt(self, container: Dict[str, Any]) -> CIExperimentStatistics:
        import torch

        from ptycho_torch import helper as hh

        if "observed_images" not in container:
            raise ValueError("CI dict adapter requires 'observed_images' amplitude data.")
        if container.get("probe") is None:
            raise ValueError("CI dict adapter requires a calibrated 'probe'.")

        amplitude = torch.as_tensor(container["observed_images"])
        if not torch.is_floating_point(amplitude):
            amplitude = amplitude.to(torch.float32)
        probe = _canonicalize_ci_probe_modes(
            torch.as_tensor(container["probe"], device=amplitude.device),
            self.N,
        )
        container["probe"] = probe

        measured_intensity, probe_physical = adapt_normalized_amplitude_to_ci(
            amplitude,
            probe,
            self.count_amplitude_scale,
        )
        if measured_intensity.ndim != 4:
            raise ValueError(
                "CI grid-lines amplitude must have shape (B, H, W, C)."
            )

        measured_channel_first = measured_intensity.permute(0, 3, 1, 2)
        statistics = self.statistics or derive_ci_experiment_statistics(
            measured_channel_first,
            self.N,
        )

        probe_training_np, probe_normalization = hh.normalize_probe_like_tf(
            probe_physical.detach().cpu().numpy(),
            probe_scale=self.probe_scale,
            probe_mask=self.probe_mask,
            probe_mask_sigma=self.probe_mask_sigma,
            probe_mask_diameter=self.probe_mask_diameter,
        )
        probe_training = torch.as_tensor(
            probe_training_np,
            device=probe_physical.device,
        ).to(probe_physical.dtype)
        probe_normalization_tensor = measured_intensity.new_tensor(
            probe_normalization
        )

        original_x = container.get("X")
        if original_x is not None and tuple(torch.as_tensor(original_x).shape) == tuple(
            measured_intensity.shape
        ):
            container["X"] = measured_intensity
        container["measured_intensity"] = measured_intensity
        container["observed_images"] = measured_intensity
        container["probe_physical"] = probe_physical
        container["probe_training"] = probe_training
        container["probe_normalization"] = probe_normalization_tensor
        container["scaling_constant"] = probe_normalization_tensor.view(1, 1, 1)
        container["rms_input_scale"] = statistics.rms_input_scale
        container["mean_measured_intensity"] = statistics.mean_measured_intensity
        container["count_amplitude_scale"] = torch.as_tensor(
            self.count_amplitude_scale,
            dtype=measured_intensity.dtype,
            device=measured_intensity.device,
        )

        # CI uses named physical quantities; legacy generic scales are not sources.
        container.pop("rms_scaling_constant", None)
        container.pop("physics_scaling_constant", None)
        return statistics


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


def run_cdi_example_torch(
    train_data: Union[RawData, 'PtychoDataContainerTorch'],
    test_data: Optional[Union[RawData, 'PtychoDataContainerTorch']],
    config: TrainingConfig,
    flip_x: bool = False,
    flip_y: bool = False,
    transpose: bool = False,
    M: int = 20,
    do_stitching: bool = False,
    execution_config: Optional[Any] = None,
    overrides: Optional[dict] = None,
    *,
    resolved_payload: Optional[TrainingPayload] = None,
    amplitude_physics_gain_record: Optional[
        AmplitudePhysicsGainRecord
    ] = None,
    torch_training_seed: Optional[int] = None,
) -> Tuple[Optional[Any], Optional[Any], Dict[str, Any]]:
    """
    Run the main CDI example execution flow using PyTorch backend.

    This function provides API parity with ptycho.workflows.components.run_cdi_example,
    enabling transparent backend selection from Ptychodus per specs/ptychodus_api_spec.md §4.5.

    Resolved configuration is passed to descendants explicitly. The workflow
    does not project the complete config into ``params.cfg``; any surviving
    legacy leaf owns its own narrow compatibility scope.

    Args:
        train_data: Training data (RawData or PtychoDataContainerTorch)
        test_data: Optional test data (same type constraints as train_data)
        config: TrainingConfig instance (TensorFlow dataclass, translated via config_bridge)
        flip_x: Whether to flip the x coordinates during reconstruction
        flip_y: Whether to flip the y coordinates during reconstruction
        transpose: Whether to transpose the image by swapping dimensions
        M: Parameter for reassemble_position function (default: 20)
        do_stitching: Whether to perform image stitching after training
        execution_config: Optional unresolved ExecutionRequest for runtime
                         control. The factory resolves it exactly once.
        overrides: Optional torch-only factory overrides forwarded unchanged to
            the Lightning training boundary.
        amplitude_physics_gain_record: Optional provenance record for the
            already-resolved scalar persisted in the strict bundle sidecar.
        torch_training_seed: Optional dedicated seed for Torch parameter and
            dataloader initialization.

    Returns:
        Tuple containing:
        - reconstructed amplitude (or None if stitching disabled)
        - reconstructed phase (or None if stitching disabled)
        - results dictionary (training history, containers, metrics)

    Implementation:
        1. Train the model via train_cdi_model_torch (Lightning trainer
           orchestration), forwarding execution_config when provided.
        2. Initialize reconstruction outputs (recon_amp, recon_phase) to None.
        3. If do_stitching and test_data is provided, stitch reconstructed patches
           via _reassemble_cdi_image_torch and merge the results into train_results.
        4. If config.output_dir is set and train_results contains models, persist
           them via save_torch_bundle (wts.h5.zip, TensorFlow-convention archive path).
        5. Return (recon_amp, recon_phase, train_results), matching the TensorFlow
           baseline signature (specs/ptychodus_api_spec.md §4.5).

    Example:
        >>> from ptycho_torch.workflows.components import run_cdi_example_torch
        >>> from ptycho.config.config import TrainingConfig, ModelConfig
        >>> from ptycho.raw_data import RawData
        >>>
        >>> # Load data
        >>> train_data = RawData.from_file("train.npz")
        >>> config = TrainingConfig(model=ModelConfig(N=64), ...)
        >>>
        >>> # Execute PyTorch pipeline
        >>> amp, phase, results = run_cdi_example_torch(
        ...     train_data, None, config, do_stitching=False
        ... )

    Contract: docs/architecture_torch.md §Component Contracts.
    """
    _validate_training_execution_input(execution_config, resolved_payload)

    # Step 1: Train the model (Phase D2.B — delegates to Lightning trainer stub)
    logger.info("Invoking PyTorch training orchestration via train_cdi_model_torch")
    # Note: train_cdi_model_torch will need to be updated to accept execution_config
    # For now, we pass it as a keyword argument for forward compatibility
    training_kwargs = {}
    if execution_config is not None:
        training_kwargs["execution_config"] = execution_config
    if overrides is not None:
        training_kwargs["overrides"] = overrides
    if resolved_payload is not None:
        training_kwargs["resolved_payload"] = resolved_payload
    if torch_training_seed is not None:
        training_kwargs["torch_training_seed"] = torch_training_seed
    train_results = train_cdi_model_torch(
        train_data,
        test_data,
        config,
        **training_kwargs,
    )
    if amplitude_physics_gain_record is not None:
        train_results["amplitude_physics_gain_record"] = (
            amplitude_physics_gain_record
        )
        train_results["amplitude_physics_gain_metadata"] = (
            amplitude_physics_gain_record.to_metadata()
        )

    # Step 2: Initialize return values for reconstruction outputs
    recon_amp, recon_phase = None, None

    # Step 3: Optional stitching path (when explicitly requested + test data provided)
    # Mirrors TensorFlow baseline ptycho/workflows/components.py:714-721
    if do_stitching and test_data is not None:
        logger.info("Performing image stitching (do_stitching=True, test_data provided)...")
        # Phase D2.C: Invoke reassembly helper to stitch reconstructed patches
        recon_amp, recon_phase, reassemble_results = _reassemble_cdi_image_torch(
            test_data, config, flip_x, flip_y, transpose, M, train_results=train_results
        )
        # Merge reassembly outputs into training results (update pattern from TF baseline)
        train_results.update(reassemble_results)
        logger.info("Image stitching complete")
    else:
        logger.info("Skipping image stitching (do_stitching=False or no test data available)")

    # Step 4: Optional persistence (Phase D4.C1 — save models when output_dir specified)
    # Mirrors TensorFlow baseline ptycho/workflows/components.py:709-723
    if config.output_dir and 'models' in train_results and train_results['models']:
        logger.info(f"Saving trained models to {config.output_dir} via save_torch_bundle")
        # Build archive path following TensorFlow convention (wts.h5.zip)
        archive_path = Path(config.output_dir) / "wts.h5"
        intensity_scale = train_results.get('intensity_scale')
        save_torch_bundle(
            models_dict=train_results['models'],
            base_path=str(archive_path),
            config=config,
            intensity_scale=intensity_scale
        )
        bundle_path = archive_path.with_suffix(".h5.zip")
        persisted_model = train_results['models']['diffraction_to_obj']
        if bundle_path.is_file() and all(
            hasattr(persisted_model, name)
            for name in (
                "data_config",
                "model_config",
                "training_config",
                "inference_config",
                "get_ci_statistics",
            )
        ):
            _persist_bundle_scaling_metadata(
                bundle_path,
                persisted_model,
                amplitude_physics_gain_record=(
                    amplitude_physics_gain_record
                ),
            )
        elif amplitude_physics_gain_record is not None:
            raise RuntimeError(
                "Cannot persist amplitude_physics_gain_record because the "
                "training bundle or resolved model metadata is unavailable."
            )
        train_results["bundle_path"] = bundle_path
        logger.info(f"Models saved successfully to {archive_path}.zip")
    else:
        logger.debug("Skipping model persistence (no output_dir or no models in train_results)")

    # Step 5: Return tuple matching TensorFlow baseline signature
    # (amplitude, phase, results) per specs/ptychodus_api_spec.md §4.5
    return recon_amp, recon_phase, train_results


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
        grouped_data = data.generate_grouped_data(
            N=config.model.N,
            K=config.sampling.neighbor_count,
            nsamples=config.sampling.n_groups,
            dataset_path=str(config.data.train_data_file) if config.data.train_data_file else None,
            sequential_sampling=config.sampling.sequential_sampling,
            gridsize=config.model.gridsize,
            seed=config.sampling.subsample_seed,
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


def _resolve_torch_training_seed(
    config: Optional[TrainingConfig],
    torch_training_seed: Optional[int],
) -> int:
    """Resolve the dedicated Torch stream or the direct-call fallback."""

    if torch_training_seed is None:
        sampling = getattr(config, "sampling", None)
        configured_seed = getattr(sampling, "subsample_seed", None)
        torch_training_seed = 42 if configured_seed is None else configured_seed
    if (
        isinstance(torch_training_seed, bool)
        or not isinstance(torch_training_seed, int)
    ):
        raise TypeError("torch_training_seed must be a nonnegative integer")
    if torch_training_seed < 0:
        raise ValueError("torch_training_seed must be a nonnegative integer")
    return torch_training_seed


def _build_lightning_dataloaders(
    train_container: Union['PtychoDataContainerTorch', Dict, 'PtychoDataset'],
    test_container: Optional[
        Union['PtychoDataContainerTorch', Dict, 'PtychoDataset']
    ],
    config: Optional[TrainingConfig],
    payload: Optional[TrainingPayload] = None,
    *,
    torch_training_seed: Optional[int] = None,
):
    """Build the one native RAM/mmap loader path for Lightning training."""

    import lightning.pytorch as L
    import torch
    from dataclasses import replace

    if payload is not None and config is None:
        config = getattr(payload, "tf_training_config", None)

    from ptycho_torch.config_params import (
        DataConfig as PTDataConfig,
        ModelConfig as PTModelConfig,
        TrainingConfig as PTTrainingConfig,
    )

    data_config = getattr(payload, "pt_data_config", None) if payload else None
    if data_config is None:
        data_source = getattr(config, "data", config)
        data_config = PTDataConfig(
            **{
                name: getattr(data_source, name)
                for name in ("scale_contract_version", "measurement_domain")
                if data_source is not None and hasattr(data_source, name)
            }
        )

    model_config = getattr(payload, "pt_model_config", None) if payload else None
    if model_config is None:
        from ptycho.config.config import resolve_model_object_policy

        source = resolve_model_object_policy(
            getattr(config, "model", None),
            backend="torch",
            warn_deprecated=False,
        )
        mode = getattr(source, "mode", None) or {
            "pinn": "Unsupervised",
            "supervised": "Supervised",
        }.get(getattr(source, "model_type", None), "Unsupervised")
        model_config = PTModelConfig(
            mode=mode,
            object_big=source.object_big,
            object_layout=source.object_layout,
            training_canvas=source.training_canvas,
            physics_forward_mode=getattr(
                source, "physics_forward_mode", "amplitude"
            ),
        )

    training_config = (
        getattr(payload, "pt_training_config", None) if payload else None
    )
    if training_config is None:
        public_loss = getattr(config, "loss", None)
        training_config = PTTrainingConfig(
            batch_size=getattr(config, "batch_size", PTTrainingConfig().batch_size),
            torch_loss_mode=getattr(public_loss, "torch_loss_mode", "poisson"),
        )

    scale_contract = validate_scale_contract(
        data_config, model_config, training_config
    )
    ci_active = (
        scale_contract is not None
        and scale_contract.version == CI_SCALE_CONTRACT
    )
    if ci_active and not isinstance(train_container, PtychoDataset):
        statistics = _adapt_container_for_ci(
            train_container,
            data_config=data_config,
            model_config=model_config,
        )
        if statistics is not None:
            _adapt_container_for_ci(
                test_container,
                data_config=data_config,
                model_config=model_config,
                statistics=statistics,
            )

    seed = _resolve_torch_training_seed(config, torch_training_seed)
    L.seed_everything(seed)
    shuffle = not bool(
        getattr(getattr(config, "sampling", config), "sequential_sampling", False)
    )

    execution_config = getattr(payload, "execution_config", None)
    strategy = (
        getattr(execution_config, "strategy", None)
        if execution_config is not None
        else getattr(training_config, "strategy", None)
    )
    distributed = strategy == "ddp" or is_spawn_strategy(strategy)

    if isinstance(train_container, PtychoDataset) and distributed:
        runtime_training = training_config
        if execution_config is not None:
            runtime_training = replace(
                training_config,
                strategy=execution_config.strategy,
                n_devices=execution_config.devices,
                num_workers=execution_config.num_workers,
                device=(
                    "cuda"
                    if execution_config.accelerator in {"cuda", "gpu"}
                    else execution_config.accelerator
                ),
            )
        return PrebuiltPtychoDataModule(
            train_container.data_dir_path,
            model_config,
            data_config,
            runtime_training,
            validation_map_path=(
                test_container.data_dir_path
                if isinstance(test_container, PtychoDataset)
                else None
            ),
            execution_config=execution_config,
            shuffle_training=shuffle,
            torch_training_seed=seed,
        )

    if isinstance(train_container, PtychoDataset):
        train_dataset = train_container
        validation_dataset = (
            test_container
            if isinstance(test_container, PtychoDataset)
            else None
        )
        if ci_active:
            statistics = train_dataset.get_ci_statistics()
            if statistics is None:
                statistics = train_dataset.set_ci_statistics_from_indices(
                    torch.arange(len(train_dataset))
                )
            if validation_dataset is not None:
                validation_dataset.data_dict["ci_statistics"] = {
                    name: value.detach().clone()
                    for name, value in statistics.items()
                }
    else:
        train_dataset = _PtychoContainerDataset(
            train_container,
            model_config=model_config,
            ci_active=ci_active,
        )
        validation_dataset = (
            _PtychoContainerDataset(
                test_container,
                model_config=model_config,
                ci_active=ci_active,
            )
            if test_container is not None
            else None
        )

    if execution_config is None:
        worker_settings = {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
        }
    else:
        worker_settings = {
            "num_workers": execution_config.num_workers,
            "pin_memory": execution_config.pin_memory,
            "persistent_workers": execution_config.persistent_workers,
            "prefetch_factor": execution_config.prefetch_factor,
        }

    train_loader = build_ptycho_loader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=shuffle,
        seed=seed,
        **worker_settings,
    )
    validation_loader = (
        build_ptycho_loader(
            validation_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            seed=seed,
            **worker_settings,
        )
        if validation_dataset is not None
        else None
    )
    return train_loader, validation_loader


def _build_inference_dataloader(
    container: 'PtychoDataContainerTorch',
    config: TrainingConfig,
    execution_config: Optional['PyTorchExecutionConfig'] = None
) -> 'DataLoader':
    """
    Build deterministic PyTorch DataLoader for inference/stitching.

    This helper creates a DataLoader optimized for inference: no shuffling,
    sequential iteration, and batch sizing configured for memory efficiency.

    Args:
        container: Inference data container (PtychoDataContainerTorch or dict)
        config: TrainingConfig with batch_size setting
        execution_config: Optional PyTorchExecutionConfig with runtime knobs (Phase C3.B1)

    Returns:
        DataLoader: Sequential loader for inference predictions

    Notes:
        - Always uses shuffle=False for deterministic stitching order
        - drop_last=False ensures all samples are processed
        - Batch size can be overridden via execution_config.inference_batch_size (Phase C3.B2)
        - num_workers and pin_memory controlled by execution_config
        - Compatible with _build_lightning_dataloaders duck-typing pattern
    """
    # torch-optional import guarded here
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        raise RuntimeError(
            "PyTorch backend requires torch. "
            "Install with: pip install -e .[torch]\n"
            "See docs/workflows/pytorch.md for installation guidance."
        ) from e

    # Extract tensors using same helper pattern as training loader
    def _get_tensor(container, key, default=None):
        """Helper to extract tensor from container or dict."""
        if hasattr(container, key):
            val = getattr(container, key)
        elif isinstance(container, dict):
            val = container.get(key, default)
        else:
            val = default

        # Convert numpy arrays to torch tensors if needed
        if val is not None and not isinstance(val, torch.Tensor):
            import numpy as np
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
        return val

    # Build inference dataset
    infer_X = _get_tensor(container, 'X')
    infer_coords = _get_tensor(container, 'coords_nominal')

    # Fallback for missing tensors
    if infer_X is None:
        infer_X = torch.randn(5, 64, 64)
    if infer_coords is None:
        batch_size = infer_X.size(0) if isinstance(infer_X, torch.Tensor) else 5
        infer_coords = torch.randn(batch_size, 2)

    # DTYPE ENFORCEMENT (Phase D1d): Cast to float32 to prevent Lightning Conv2d dtype mismatch
    # Requirement: specs/data_contracts.md §1 mandates diffraction arrays be float32
    # Root cause: torch.from_numpy preserves dtype; legacy/checkpoint data may be float64
    # Symptom: RuntimeError "Input type (double) and bias type (float)" in Lightning forward
    # Solution: Explicit cast before TensorDataset construction
    infer_X = infer_X.to(torch.float32, copy=False)
    infer_coords = infer_coords.to(torch.float32, copy=False)

    # SHAPE CONVERSION: Container X is channel-last (B, H, W, C), model expects channel-first (B, C, H, W)
    # Permute to match model input format
    if infer_X.ndim == 4:
        infer_X = infer_X.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)

    infer_dataset = TensorDataset(infer_X, infer_coords)

    # Import execution config defaults if not provided (Phase C3.B1)
    if execution_config is None:
        from ptycho.config.config import PyTorchExecutionConfig
        execution_config = PyTorchExecutionConfig()
        logger.info(f"PyTorchExecutionConfig auto-instantiated for inference dataloader (accelerator resolved to '{execution_config.accelerator}')")

    # Determine batch size: execution_config.inference_batch_size overrides config.batch_size (Phase C3.B2)
    batch_size = execution_config.inference_batch_size or getattr(config, 'batch_size', 1)

    # Create deterministic loader with execution config knobs
    return DataLoader(
        infer_dataset,
        batch_size=batch_size,  # Controlled by execution_config.inference_batch_size
        shuffle=False,  # Deterministic order for stitching
        drop_last=False,  # Process all samples
        num_workers=execution_config.num_workers,  # Controlled by execution_config
        pin_memory=execution_config.pin_memory  # GPU-only flag, CPU-safe default False
    )


def _move_batch_to_device(batch, device):
    """Move tensors in a nested Lightning batch structure to ``device``."""
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {
            key: _move_batch_to_device(value, device)
            for key, value in batch.items()
        }
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_batch_to_device(value, device) for value in batch)
    return batch


@dataclass(frozen=True, slots=True)
class _RectS1S2IndexedRows:
    value: Any
    access_rows: tuple[SelectedDoseClosureRow, ...]


@dataclass(frozen=True, slots=True)
class _RectS1S2SelectedBatch:
    value: Any
    access_rows: tuple[SelectedDoseClosureRow, ...]


_RECT_S1S2_IDENTITY_FIELD = "__rect_s1s2_logical_row_identity__"


def _rect_s1s2_attach_identities(value, access_rows, *, batched_indexing):
    import torch

    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            "rect_s1s2 selected indexing must return a sequence whose first "
            "item is the batch field mapping"
        )
    fields = value[0]
    if _RECT_S1S2_IDENTITY_FIELD in fields:
        raise ValueError(
            "rect_s1s2 reserved identity field collides with dataset fields"
        )
    if isinstance(fields, dict):
        identified_fields = dict(fields)
    elif hasattr(fields, "batch_size") and callable(
        getattr(fields, "clone", None)
    ):
        identified_fields = fields.clone(recurse=False)
    else:
        raise ValueError(
            "rect_s1s2 selected indexing requires mutable mapping fields"
        )
    logical_rows = torch.tensor(
        [row.logical_row for row in access_rows],
        dtype=torch.int64,
    )
    identity = logical_rows if batched_indexing else logical_rows[0]
    identified_fields[_RECT_S1S2_IDENTITY_FIELD] = identity
    if isinstance(value, tuple):
        return (identified_fields, *value[1:])
    return [identified_fields, *value[1:]]


def _rect_s1s2_verify_collated_identities(batch, access_rows):
    import torch

    try:
        fields = batch[0]
        identity = fields[_RECT_S1S2_IDENTITY_FIELD]
        collated_logical_rows = tuple(
            int(value)
            for value in torch.as_tensor(identity).reshape(-1).tolist()
        )
    except Exception as error:
        raise ValueError(
            "rect_s1s2 maintained collation must preserve row identity"
        ) from error
    expected_logical_rows = tuple(row.logical_row for row in access_rows)
    if (
        len(collated_logical_rows) != len(expected_logical_rows)
        or set(collated_logical_rows) != set(expected_logical_rows)
    ):
        raise ValueError(
            "rect_s1s2 maintained collation has missing or extra identity "
            "coverage"
        )
    if collated_logical_rows != expected_logical_rows:
        raise ValueError(
            "rect_s1s2 maintained collation has reordered identity coverage"
        )
    try:
        del fields[_RECT_S1S2_IDENTITY_FIELD]
    except Exception as error:
        raise ValueError(
            "rect_s1s2 maintained collation must expose removable row identity"
        ) from error


class _RectS1S2SelectedDataset:
    """Index only the immutable logical rows selected for dose closure."""

    def __init__(self, dataset, access_rows, *, batched_indexing):
        self.dataset = dataset
        self.access_rows = tuple(access_rows)
        self.batched_indexing = bool(batched_indexing)
        self._ptycho_vectorized_batch = self.batched_indexing

    def __len__(self):
        return len(self.access_rows)

    def __getitem__(self, index):
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError(
                "rect_s1s2 selected dataset requires an integer index"
            )
        row = self.access_rows[int(index)]
        try:
            value = self.dataset[row.logical_row]
        except Exception as error:
            raise ValueError(
                "rect_s1s2 selected dataset does not support logical-row "
                "indexing"
            ) from error
        value = _rect_s1s2_attach_identities(
            value,
            (row,),
            batched_indexing=False,
        )
        return _RectS1S2IndexedRows(value=value, access_rows=(row,))

    def __getitems__(self, indices):
        if not self.batched_indexing:
            return [self[index] for index in indices]
        rows = tuple(self.access_rows[int(index)] for index in indices)
        try:
            value = self.dataset.__getitems__(
                [row.logical_row for row in rows]
            )
        except Exception as error:
            raise ValueError(
                "rect_s1s2 selected dataset does not support maintained "
                "vectorized indexing"
            ) from error
        value = _rect_s1s2_attach_identities(
            value,
            rows,
            batched_indexing=True,
        )
        return _RectS1S2IndexedRows(value=value, access_rows=rows)


class _RectS1S2MaintainedCollation:
    def __init__(self, collate_fn, *, batched_indexing):
        self.collate_fn = collate_fn
        self.batched_indexing = bool(batched_indexing)

    def __call__(self, indexed):
        if self.batched_indexing:
            if not isinstance(indexed, _RectS1S2IndexedRows):
                raise ValueError(
                    "rect_s1s2 selected TensorDict indexing returned an "
                    "unsupported value"
                )
            values = indexed.value
            access_rows = indexed.access_rows
        else:
            if not isinstance(indexed, list) or not all(
                isinstance(value, _RectS1S2IndexedRows) for value in indexed
            ):
                raise ValueError(
                    "rect_s1s2 selected dataset indexing returned an "
                    "unsupported value"
                )
            values = [value.value for value in indexed]
            access_rows = tuple(
                row for value in indexed for row in value.access_rows
            )
        try:
            batch = self.collate_fn(values)
        except Exception as error:
            raise ValueError(
                "rect_s1s2 selected rows could not use the maintained "
                "training-loader collation"
            ) from error
        _rect_s1s2_verify_collated_identities(batch, access_rows)
        return _RectS1S2SelectedBatch(value=batch, access_rows=access_rows)


def _rect_s1s2_indexable_dataset(training_loader):
    dataset = getattr(training_loader, "dataset", None)
    if not callable(getattr(dataset, "__len__", None)) or not callable(
        getattr(dataset, "__getitem__", None)
    ):
        raise TypeError(
            "rect_s1s2 dose closure requires an indexable training-loader "
            "dataset"
        )
    try:
        len(dataset)
    except Exception as error:
        raise ValueError(
            "rect_s1s2 dose-closure dataset must have a valid length"
        ) from error
    return dataset


def _rebuild_rect_s1s2_loader(
    training_loader,
    *,
    access_rows,
    batch_size,
):
    """Rebuild one loader over selected logical rows without ambient state."""

    import torch

    dataset = _rect_s1s2_indexable_dataset(training_loader)
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise TypeError("rect_s1s2 selected batch size must be a positive integer")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("rect_s1s2 selected batch size must be a positive integer")
    collate_fn = getattr(training_loader, "collate_fn", None)
    if not callable(collate_fn):
        raise ValueError(
            "rect_s1s2 dose closure requires maintained callable collation"
        )
    rows = tuple(access_rows)
    if not all(isinstance(row, SelectedDoseClosureRow) for row in rows):
        raise TypeError(
            "rect_s1s2 selected access rows must be immutable selection values"
        )
    if not isinstance(training_loader, torch.utils.data.DataLoader):
        raise TypeError(
            "rect_s1s2 dose closure supports PyTorch DataLoader instances"
        )

    capability_owner = dataset
    while isinstance(capability_owner, torch.utils.data.Subset):
        capability_owner = capability_owner.dataset
    batched_indexing = bool(
        getattr(capability_owner, "_ptycho_vectorized_batch", False)
        and callable(getattr(dataset, "__getitems__", None))
    )
    selected_dataset = _RectS1S2SelectedDataset(
        dataset,
        rows,
        batched_indexing=batched_indexing,
    )
    local_generator = torch.Generator()
    local_generator.manual_seed(0)
    return torch.utils.data.DataLoader(
        selected_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=_RectS1S2MaintainedCollation(
            collate_fn,
            batched_indexing=batched_indexing,
        ),
        generator=local_generator,
    )


def _rect_s1s2_batch_axes(batch, *, inspected_channels=None):
    try:
        fields = batch[0]
    except Exception as error:
        raise ValueError(
            "rect_s1s2 dose closure requires maintained batch collation"
        ) from error
    if "measured_intensity" not in fields:
        raise ValueError(
            "rect_s1s2 dose closure requires CI count-intensity batches "
            "with measured_intensity; legacy normalized-amplitude loaders "
            "are unsupported"
        )
    try:
        images = fields["images"]
        target = fields["measured_intensity"]
    except Exception as error:
        raise ValueError(
            "rect_s1s2 dose closure images and measured_intensity must share "
            "canonical (B, C, H, W) leading axes"
        ) from error
    if (
        images.ndim != 4
        or target.ndim != 4
        or tuple(target.shape[:2]) != tuple(images.shape[:2])
    ):
        raise ValueError(
            "rect_s1s2 dose closure images and measured_intensity must share "
            "canonical (B, C, H, W) leading axes"
        )
    batch_size = int(target.shape[0])
    channels = int(target.shape[1])
    if channels <= 0:
        raise ValueError(
            "rect_s1s2 dose closure requires a positive inspected channel count"
        )
    if inspected_channels is not None and channels != inspected_channels:
        raise ValueError(
            "rect_s1s2 selected row channel count "
            f"{channels} must match inspected channel count {inspected_channels}"
        )
    return fields, batch_size, channels


def _inspect_rect_s1s2_channels(training_loader):
    dataset = _rect_s1s2_indexable_dataset(training_loader)
    if len(dataset) == 0:
        raise ValueError("rect_s1s2 dose closure requires a non-empty dataset")
    access_row = SelectedDoseClosureRow(
        logical_row=0,
        base_row=_base_row_for_logical(dataset, 0),
        channels=(),
    )
    loader = _rebuild_rect_s1s2_loader(
        training_loader,
        access_rows=(access_row,),
        batch_size=1,
    )
    iterator = iter(loader)
    try:
        selected_batch = next(iterator)
    except StopIteration as error:
        raise ValueError(
            "rect_s1s2 row-zero inspection produced no batch"
        ) from error
    if not isinstance(selected_batch, _RectS1S2SelectedBatch):
        raise ValueError("rect_s1s2 row-zero inspection lost identity coverage")
    if selected_batch.access_rows != (access_row,):
        raise ValueError("rect_s1s2 row-zero inspection reordered identity coverage")
    _, batch_size, channels = _rect_s1s2_batch_axes(selected_batch.value)
    if batch_size != 1:
        raise ValueError(
            "rect_s1s2 row-zero inspection must collate exactly one logical row"
        )
    try:
        next(iterator)
    except StopIteration:
        return channels
    raise ValueError("rect_s1s2 row-zero inspection produced extra batches")


def _initialize_rect_s1s2_unmanaged(
    model,
    *,
    mode,
    training_loader=None,
):
    """Initialize the shared rectangular gauge from the fixed uniform sample."""

    import torch

    if mode not in {"ones", "dose_closure"}:
        raise ValueError(f"unsupported rect_s1s2 initialization mode {mode!r}")
    forward_model = getattr(getattr(model, "model", None), "forward_model", None)
    scaler = getattr(forward_model, "rect_scaler", None)
    if scaler is None:
        if mode == "ones":
            return RectS1S2InitializationRecord.ones().to_jsonable()
        raise ValueError(
            "rect_s1s2 dose closure requires a model with a rectangular "
            "physics scaler"
        )
    scaler.s1.data.fill_(1.0)
    scaler.s2.data.fill_(1.0)
    if mode == "ones":
        return RectS1S2InitializationRecord.ones().to_jsonable()
    if training_loader is None:
        raise ValueError("rect_s1s2 dose closure requires a CI training loader")
    dataset = _rect_s1s2_indexable_dataset(training_loader)
    channels = _inspect_rect_s1s2_channels(training_loader)
    available_patterns = len(dataset) * channels
    if available_patterns < RECT_S1S2_DOSE_CLOSURE_PATTERNS:
        raise ValueError(
            "rect_s1s2 dose closure has insufficient detector-pattern slots: "
            f"sampled {available_patterns}, required "
            f"{RECT_S1S2_DOSE_CLOSURE_PATTERNS}. Provide enough training "
            "patterns or use '--rect-s1s2-init ones'."
        )
    plan = build_dose_closure_sample_plan(dataset, channels=channels)
    selected_loader = _rebuild_rect_s1s2_loader(
        training_loader,
        access_rows=plan.access_rows,
        batch_size=getattr(training_loader, "batch_size", None),
    )
    selected_iterator = iter(selected_loader)
    expected_chunks = tuple(
        plan.access_rows[offset : offset + selected_loader.batch_size]
        for offset in range(0, len(plan.access_rows), selected_loader.batch_size)
    )
    observed_pattern_sums = []
    predicted_pattern_sums = []
    contributed_flat_slots = []
    for expected_rows in expected_chunks:
        try:
            selected_batch = next(selected_iterator)
        except StopIteration as error:
            raise ValueError(
                "rect_s1s2 selected loader has missing identity coverage"
            ) from error
        if not isinstance(selected_batch, _RectS1S2SelectedBatch):
            raise ValueError(
                "rect_s1s2 selected loader returned unsupported identity coverage"
            )
        if selected_batch.access_rows != expected_rows:
            raise ValueError(
                "rect_s1s2 selected loader has reordered identity coverage"
            )
        fields, batch_size, selected_channels = _rect_s1s2_batch_axes(
            selected_batch.value,
            inspected_channels=channels,
        )
        if batch_size != len(expected_rows):
            raise ValueError(
                "rect_s1s2 selected batch cardinality must match its exact "
                "identity chunk"
            )
        if selected_channels != channels:
            raise ValueError(
                "rect_s1s2 selected batch channel count changed unexpectedly"
            )
        batch = _move_batch_to_device(selected_batch.value, scaler.s1.device)
        fields = batch[0]
        positions = fields["coords_relative"]
        experiment_ids = fields["experiment_id"]
        target = fields["measured_intensity"]
        probe = fields["probe_training"]
        probe_normalization = fields["probe_normalization"]
        output_scale = probe_normalization.reshape(
            batch_size, 1, 1, 1
        ).reciprocal()
        unit_object = torch.ones_like(fields["images"], dtype=torch.complex64)
        with torch.no_grad():
            predicted = forward_model(
                unit_object,
                target,
                positions,
                probe,
                output_scale,
                experiment_ids,
            )
        if predicted.ndim != 4 or tuple(predicted.shape) != tuple(target.shape):
            raise ValueError(
                "rect_s1s2 dose closure predicted intensity must match "
                "measured_intensity shape (B, C, H, W)"
            )
        mask = torch.zeros(
            (batch_size, channels),
            dtype=torch.bool,
            device=target.device,
        )
        for row_index, access_row in enumerate(expected_rows):
            for channel in access_row.channels:
                flat_slot = access_row.logical_row * channels + channel
                contributed_flat_slots.append(flat_slot)
                mask[row_index, channel] = True
        selected_target = target.to(torch.float64)[mask]
        selected_predicted = predicted.to(torch.float64)[mask]
        expected_selected = sum(len(row.channels) for row in expected_rows)
        if int(mask.sum().item()) != expected_selected:
            raise ValueError(
                "rect_s1s2 selected channel masks have duplicate or missing "
                "flat-slot coverage"
            )
        if bool((selected_target < 0).any().item()):
            raise ValueError(
                "rect_s1s2 dose closure observed counts must be nonnegative"
            )
        observed_pattern_sums.append(
            selected_target.reshape(expected_selected, -1).sum(dim=1)
        )
        predicted_pattern_sums.append(
            selected_predicted.reshape(expected_selected, -1).sum(dim=1)
        )
    try:
        next(selected_iterator)
    except StopIteration:
        pass
    else:
        raise ValueError(
            "rect_s1s2 selected loader has extra identity coverage"
        )
    if (
        len(contributed_flat_slots) != RECT_S1S2_DOSE_CLOSURE_PATTERNS
        or len(set(contributed_flat_slots))
        != RECT_S1S2_DOSE_CLOSURE_PATTERNS
        or set(contributed_flat_slots) != set(plan.flat_slots)
    ):
        raise ValueError(
            "rect_s1s2 selected channel masks have missing, extra, or "
            "duplicate flat-slot coverage"
        )
    observed_sum = float(torch.cat(observed_pattern_sums).sum().item())
    predicted_sum = float(torch.cat(predicted_pattern_sums).sum().item())
    if not math.isfinite(observed_sum) or observed_sum <= 0.0:
        raise ValueError(
            "rect_s1s2 dose closure observed count sum must be positive and "
            f"finite; got {observed_sum!r}"
        )
    if not math.isfinite(predicted_sum) or predicted_sum <= 0.0:
        raise ValueError(
            "rect_s1s2 dose closure predicted intensity sum must be positive "
            f"and finite; got {predicted_sum!r}"
        )

    closure = observed_sum / predicted_sum
    if not math.isfinite(closure) or closure <= 0.0:
        raise ValueError(
            "rect_s1s2 dose closure c* must be positive and finite; "
            f"got {closure!r}"
        )
    gauge = math.sqrt(closure)
    if not math.isfinite(gauge) or gauge <= 0.0:
        raise ValueError(
            "rect_s1s2 dose closure gauge must be positive and finite; "
            f"got {gauge!r}"
        )
    scaler.s1.data.fill_(gauge)
    scaler.s2.data.fill_(gauge)
    return RectS1S2InitializationRecord.dose_closure(gauge).to_jsonable()


def _initialize_rect_s1s2(
    model,
    *,
    mode,
    training_loader=None,
):
    """Run initialization inference while preserving every module state."""

    if mode != "dose_closure":
        return _initialize_rect_s1s2_unmanaged(
            model,
            mode=mode,
            training_loader=training_loader,
        )
    training_states = tuple(
        (module, bool(module.training)) for module in model.modules()
    )
    model.eval()
    try:
        return _initialize_rect_s1s2_unmanaged(
            model,
            mode=mode,
            training_loader=training_loader,
        )
    finally:
        for module, training in training_states:
            module.training = training


def _write_training_summary_atomic(path, record):
    """Crash-safe JSON publication for the rank-zero training summary."""

    import json
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validated = RectS1S2InitializationRecord.from_mapping(record)
    encoded = (
        json.dumps(
            validated.to_jsonable(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_training_summary_and_barrier(trainer, path, record):
    """Publish on global zero, then release every rank from the live strategy."""

    if bool(getattr(trainer, "is_global_zero", False)):
        _write_training_summary_atomic(path, record)
    strategy = getattr(trainer, "strategy", None)
    barrier = getattr(strategy, "barrier", None)
    if not callable(barrier):
        raise RuntimeError(
            "Lightning strategy must expose barrier() while publishing the "
            "training summary"
        )
    barrier("rect_s1s2_training_summary")


def _rect_s1s2_training_loader(data_product, train_loader, mode):
    """Resolve the training source only when dose closure consumes it."""

    if mode == "ones":
        return None
    if isinstance(data_product, PrebuiltPtychoDataModule):
        data_product.setup("fit")
        return data_product.train_dataloader()
    return train_loader


def _effective_dataloader_settings(
    data_product,
    train_loader,
    execution_config,
):
    """Return the loader settings used by this Trainer invocation."""

    if isinstance(data_product, PrebuiltPtychoDataModule):
        return data_product._loader_settings()
    num_workers = int(getattr(train_loader, "num_workers", 0))
    return {
        "num_workers": num_workers,
        "pin_memory": bool(getattr(train_loader, "pin_memory", False)),
        "persistent_workers": (
            bool(getattr(train_loader, "persistent_workers", False))
            if num_workers > 0
            else False
        ),
        "prefetch_factor": (
            getattr(train_loader, "prefetch_factor", None)
            if num_workers > 0
            else None
        ),
    }


def _train_with_lightning(
    train_container: Union['PtychoDataContainerTorch','PtychoDataset'],
    test_container: Optional['PtychoDataContainerTorch'],
    config: TrainingConfig,
    execution_config: Optional[Any] = None,
    overrides: Optional[dict] = None,
    *,
    resolved_payload: Optional[TrainingPayload] = None,
    torch_training_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Orchestrate Lightning trainer execution for PyTorch model training.

    This function implements the Lightning training workflow per Phase D2.B blueprint:
    1. Derives PyTorch config objects from TensorFlow TrainingConfig
    2. Instantiates PtychoPINN_Lightning module with all four config dependencies
    3. Builds train/val dataloaders via _build_lightning_dataloaders helper
    4. Configures Lightning Trainer with checkpoint/logging settings (ADR-003 Phase C3)
    5. Executes training via trainer.fit()
    6. Returns structured results dict with history, containers, and module handle

    Args:
        train_container: Normalized training data container
        test_container: Optional normalized test data container
        config: TrainingConfig with training hyperparameters
        execution_config: Optional unresolved ExecutionRequest. Ignored only
            when absent because ``resolved_payload`` already owns the resolved
            runtime carrier.
        overrides: Optional dict of torch-only ``resolve_training_payload`` overrides
            (highest precedence, applied last). This is the forwarding channel for
            ModelConfig knobs that exist only on the torch-side
            ptycho_torch.config_params.ModelConfig (e.g. training_patch_weighting,
            physics_forward_mode, cnn_output_mode, rect_s1s2_trainable) and therefore
            cannot be threaded through the read-only TF-side TrainingConfig/ModelConfig
            (ptycho/config/config.py). See Task 2.7 (B7) follow-up.

    Returns:
        Dict[str, Any]: Training results including:
            - history: Dict with train_loss and optional val_loss trajectories
            - train_container: Original training container
            - test_container: Original test container
            - models: Dict with 'diffraction_to_obj' (Lightning module) and 'autoencoder' (sentinel)
                      for dual-model bundle persistence per spec §4.6

    Raises:
        RuntimeError: If torch or lightning packages are not installed (POLICY-001)

    References:
        - Blueprint: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T020940Z/phase_d2_completion/phase_b2_implementation.md
        - Spec: specs/ptychodus_api_spec.md:187 (reconstructor lifecycle contract)
        - Findings: POLICY-001 (PyTorch mandatory), CONFIG-001 (params.cfg already populated by caller)
        - ADR-003 Phase C3: execution_config controls Trainer kwargs (accelerator, deterministic, gradient_clip_val)
    """
    _validate_training_execution_input(execution_config, resolved_payload)

    # B2.2: torch-optional imports with POLICY-001 compliant error messaging
    try:
        import lightning.pytorch as L
        from ptycho_torch.train_utils import PrebuiltPtychoDataModule
    except ImportError as e:
        raise RuntimeError(
            "PyTorch backend requires torch>=2.2 and lightning. "
            "Install with: pip install -e .[torch]\n"
            "See docs/workflows/pytorch.md for installation guidance."
        ) from e

    logger.info("_train_with_lightning orchestrating Lightning training")
    logger.info(f"Training config: nepochs={config.nepochs}, n_groups={config.sampling.n_groups}")

    # B2.1: Use the pure config resolver to derive PyTorch configs with correct
    # channel propagation. The compatibility factory remains for declared
    # CONFIG-001 callers.
    # CRITICAL (Phase C4.D B2): Factory ensures C = gridsize**2 is propagated to
    # pt_model_config.C_model and pt_model_config.C_forward, preventing channel mismatch
    # when gridsize > 1 (see docs/findings.md#BUG-TF-001).
    from ptycho_torch.config_factory import (
        build_training_factory_overrides,
        resolve_training_payload,
    )
    factory_overrides = build_training_factory_overrides(config)
    # Caller-supplied torch-only overrides take highest precedence. This is how
    # ModelConfig knobs that live exclusively on the torch-side config_params.ModelConfig
    # (training_patch_weighting, physics_forward_mode, cnn_output_mode,
    # rect_s1s2_trainable) reach resolve_training_payload despite ptycho/config/config.py
    # (TF-side TrainingConfig/ModelConfig/PyTorchExecutionConfig) being read-only.
    if overrides:
        factory_overrides.update(overrides)

    # Supported CLI callers pass their already-resolved payload. Direct workflow
    # callers provide an unresolved request, which the factory resolves once.
    # Optimizer settings come only from the canonical training baseline/overrides.
    payload = resolved_payload
    if payload is None:
        payload = resolve_training_payload(
            train_data_file=Path(config.data.train_data_file),
            output_dir=Path(getattr(config, 'output_dir', './outputs')),
            execution_config=execution_config,
            overrides=factory_overrides,
            training_baseline=config,
        )

    # Extract PyTorch configs from payload (gridsize → C propagation already applied)
    pt_data_config = payload.pt_data_config
    pt_model_config = payload.pt_model_config
    pt_training_config = payload.pt_training_config
    execution_config = payload.execution_config

    # Seed before module construction and reuse the same stream at the loader
    # boundary so initialization and sampling are reproducible together.
    effective_torch_training_seed = _resolve_torch_training_seed(
        config,
        torch_training_seed,
    )
    L.seed_everything(effective_torch_training_seed)

    resolved_scale_contract = validate_scale_contract(
        pt_data_config,
        pt_model_config,
        pt_training_config,
    )

    # Build the module from the sealed structural identity plus the separately
    # owned scientific/data, training, and inference sections. Runtime execution
    # remains below at the Trainer boundary and cannot alter graph topology.
    from ptycho_torch.application_factory import build_ptychopinn_application

    model = build_ptychopinn_application(
        payload.model_spec,
        pt_data_config,
        pt_training_config,
        payload.pt_inference_config,
    )

    # Save hyperparameters so checkpoint can reconstruct module without external state
    model.save_hyperparameters()

    # B2.3: Build dataloaders via helper
    data_product = _build_lightning_dataloaders(
        train_container,
        test_container,
        config,
        payload=payload,
        torch_training_seed=effective_torch_training_seed,
    )
    
    # Data product is a Lightning datamodule for DDP-style launchers and a
    # regular train/validation loader tuple otherwise.
    if isinstance(data_product, PrebuiltPtychoDataModule):
        train_loader, val_loader = None, None  # Set to None when using datamodule
    else:
        train_loader, val_loader = data_product

    if (
        resolved_scale_contract is not None
        and resolved_scale_contract.version == CI_SCALE_CONTRACT
        and not isinstance(data_product, PrebuiltPtychoDataModule)
    ):
        model.register_ci_statistics(
            _get_finalized_ci_statistics(train_container)
        )

    # DATA-SUP-001: Supervised mode requires labeled data
    # Check if supervised mode is requested but training data lacks required labels
    if pt_model_config.mode == 'Supervised':
        # Inspect first batch to verify label keys exist
        try:
            first_batch = next(iter(train_loader))
            batch_dict = first_batch[0]  # Extract tensor dict from batch tuple
            if 'label_amp' not in batch_dict or 'label_phase' not in batch_dict:
                raise RuntimeError(
                    f"Supervised mode (model_type='supervised') requires labeled datasets with "
                    f"'label_amp' and 'label_phase' keys, but training data lacks these fields. "
                    f"Either: (1) Use a labeled NPZ dataset (see ptycho_torch/notebooks/create_supervised_datasets.ipynb), "
                    f"or (2) Switch to PINN mode (--model_type pinn) for self-supervised physics-based training."
                )
        except StopIteration:
            raise RuntimeError(
                f"Training dataloader is empty. Check dataset path and n_groups configuration."
            )

    # B2.5: Configure Trainer with settings from config
    # C3.A3: Thread execution config values to Trainer kwargs
    output_dir = Path(getattr(config, 'output_dir', './outputs'))
    debug_mode = getattr(config, 'debug', False)
    training_summary_path = output_dir / "training_summary.json"

    # Custom callback to track loss history across epochs
    class _LossHistoryCallback(L.Callback):
        """Callback to collect train/val loss per epoch for history dict.

        The model logs metrics with dynamic names like 'poisson_train_Amp_loss'
        based on model configuration. This callback searches for any metric
        containing 'train' and 'loss' (or 'val' and 'loss') to capture the loss.
        """

        def __init__(self):
            self.train_loss = []
            self.val_loss = []

        def _find_loss_metric(self, metrics, prefix):
            """Find loss metric by prefix ('train' or 'val')."""
            for key in metrics:
                if prefix in key and 'loss' in key:
                    return float(metrics[key])
            return None

        def on_train_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            loss_val = self._find_loss_metric(metrics, 'train')
            if loss_val is not None:
                self.train_loss.append(loss_val)

        def on_validation_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            loss_val = self._find_loss_metric(metrics, 'val')
            if loss_val is not None:
                self.val_loss.append(loss_val)

    loss_history_cb = _LossHistoryCallback()

    class _TrainingSummaryCallback(L.Callback):
        """Publish initialization identity while the distributed group is live."""

        def __init__(self, path):
            super().__init__()
            self.path = Path(path)
            self.record = None

        def set_record(self, record):
            self.record = RectS1S2InitializationRecord.from_mapping(record)

        def on_fit_start(self, trainer, pl_module):
            if self.record is None:
                raise RuntimeError(
                    "rect_s1s2 initialization record must be set before fit"
                )
            _publish_training_summary_and_barrier(
                trainer,
                self.path,
                self.record,
            )

    training_summary_cb = _TrainingSummaryCallback(training_summary_path)

    # EB1.D: Configure checkpoint/early-stop callbacks (ADR-003 Phase EB1)
    callbacks: list = [loss_history_cb, training_summary_cb]
    if execution_config.enable_checkpointing:
        from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

        # Determine if we have validation data to use val metrics
        # Ptycho Datamodule automatically creates a validation dataset on instantiation (see train_utils.py)
        # so this means validation set exists if data product is a datamodule.
        has_validation = test_container is not None or isinstance(data_product, PrebuiltPtychoDataModule)

        # EB2.B: Derive monitor metric from model.val_loss_name (ADR-003 Phase EB2)
        # The model's val_loss_name is dynamically constructed based on model_type and loss configuration
        # (e.g., 'poisson_val_Amp_loss' for PINN with amplitude loss, 'mae_val_Phase_loss' for supervised)
        # This ensures checkpoint/early-stop callbacks watch the correct logged metric
        if has_validation and hasattr(model, 'val_loss_name'):
            # Use the model's dynamic validation loss name
            monitor_metric = model.val_loss_name
        else:
            # Fall back to execution config default or train loss
            monitor_metric = execution_config.checkpoint_monitor_metric
            if 'val_' in monitor_metric and not has_validation:
                # Fall back to train_loss if val metric requested but no validation data
                monitor_metric = monitor_metric.replace('val_', 'train_')

        # Build checkpoint filename template using dynamic metric name
        # Format: epoch={epoch:02d}-<metric_short_name>={<full_metric_name>:.4f}
        if has_validation:
            # Extract short name for filename (remove '_loss' suffix if present)
            metric_short_name = monitor_metric.replace('_loss', '')
            filename_template = f'epoch={{epoch:02d}}-{metric_short_name}={{{monitor_metric}:.4f}}'
        else:
            filename_template = 'epoch={epoch:02d}'

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename=filename_template,
            monitor=monitor_metric,
            mode=execution_config.checkpoint_mode,
            save_top_k=execution_config.checkpoint_save_top_k,
            save_last=True,  # Always keep last checkpoint for recovery
            verbose=False,
        )
        callbacks.append(checkpoint_callback)

        # EarlyStopping callback (ADR-003 Phase EB1.D)
        # Only add early stopping if validation data is available (otherwise no metric to monitor)
        if has_validation:
            early_stop_callback = EarlyStopping(
                monitor=monitor_metric,
                mode=execution_config.checkpoint_mode,
                patience=execution_config.early_stop_patience,
                verbose=False,
            )
            callbacks.append(early_stop_callback)

    # Recon logging callback (MLflow only, opt-in via recon_log_every_n_epochs)
    if (execution_config.logger_backend == 'mlflow'
            and execution_config.recon_log_every_n_epochs is not None):
        from ptycho_torch.workflows.recon_logging import PtychoReconLoggingCallback
        recon_cb = PtychoReconLoggingCallback(
            every_n_epochs=execution_config.recon_log_every_n_epochs,
            num_patches=execution_config.recon_log_num_patches,
            fixed_indices=execution_config.recon_log_fixed_indices,
            log_stitch=execution_config.recon_log_stitch,
            max_stitch_samples=execution_config.recon_log_max_stitch_samples,
        )
        callbacks.append(recon_cb)
        logger.info("Enabled recon logging callback (every %d epochs, %d patches, stitch=%s)",
                     execution_config.recon_log_every_n_epochs,
                     execution_config.recon_log_num_patches,
                     execution_config.recon_log_stitch)

    # Instantiate logger based on execution config (Phase EB3.B - ADR-003)
    lightning_logger = False  # Default: no logger
    if execution_config.logger_backend is not None:
        try:
            if execution_config.logger_backend == 'csv':
                from lightning.pytorch.loggers import CSVLogger
                lightning_logger = CSVLogger(
                    save_dir=str(output_dir),
                    name='lightning_logs',
                )
                logger.info(f"Enabled CSVLogger: metrics saved to {output_dir}/lightning_logs/")
            elif execution_config.logger_backend == 'tensorboard':
                from lightning.pytorch.loggers import TensorBoardLogger
                lightning_logger = TensorBoardLogger(
                    save_dir=str(output_dir),
                    name='lightning_logs',
                )
                logger.info(f"Enabled TensorBoardLogger: run `tensorboard --logdir={output_dir}/lightning_logs/`")
            elif execution_config.logger_backend == 'mlflow':
                from lightning.pytorch.loggers import MLFlowLogger
                lightning_logger = MLFlowLogger(
                    experiment_name=getattr(config, 'experiment_name', 'PtychoPINN'),
                    tracking_uri=str(output_dir / 'mlruns'),
                )
                logger.info(f"Enabled MLFlowLogger: tracking URI={output_dir}/mlruns")
            else:
                logger.warning(
                    f"Unknown logger_backend '{execution_config.logger_backend}'. "
                    f"Falling back to logger=False. Supported: 'csv', 'tensorboard', 'mlflow'."
                )
        except ImportError as e:
            logger.warning(
                f"Failed to import Lightning logger '{execution_config.logger_backend}': {e}. "
                f"Metrics logging disabled. Install the required package to enable logging."
            )
            lightning_logger = False
    else:
        logger.info("Logger disabled (logger_backend=None). Loss metrics will not be saved to disk.")

    automatic_optimization = getattr(model, "automatic_optimization", True)
    effective_accum_steps = pt_training_config.accum_steps
    effective_clip_val = pt_training_config.gradient_clip_val
    effective_clip_algorithm = pt_training_config.gradient_clip_algorithm

    if not automatic_optimization and effective_clip_val:
        logger.info(
            "Manual optimization enabled; disabling Lightning Trainer gradient_clip_val "
            "and relying on model-level gradient clipping."
        )
    if automatic_optimization and effective_clip_algorithm == "agc":
        raise ValueError(
            "gradient_clip_algorithm='agc' requires manual optimization; "
            "Lightning automatic optimization accepts only 'norm' or 'value'"
        )

    trainer_kwargs = dict(
        max_epochs=pt_training_config.epochs,
        # Execution config overrides (ADR-003 Phase C3)
        accelerator=execution_config.accelerator,  # CPU-safe default, GPU via override
        strategy=execution_config.strategy,
        deterministic=execution_config.deterministic,  # Triggers torch.use_deterministic_algorithms
        gradient_clip_val=(
            effective_clip_val if automatic_optimization else None
        ),
        accumulate_grad_batches=(
            effective_accum_steps if automatic_optimization else 1
        ),
        # Checkpoint/logging knobs
        enable_progress_bar=execution_config.enable_progress_bar or debug_mode,
        enable_checkpointing=execution_config.enable_checkpointing,
        callbacks=callbacks,  # EB1.D: Pass configured callbacks to Trainer
        # Standard settings
        devices=execution_config.devices,
        precision=execution_config.precision,
        log_every_n_steps=1,
        default_root_dir=str(output_dir),
        logger=lightning_logger,  # Phase EB3.B: Use configured logger (False if disabled)
    )
    if automatic_optimization:
        trainer_kwargs["gradient_clip_algorithm"] = effective_clip_algorithm
    trainer = L.Trainer(**trainer_kwargs)
    effective_runtime = _build_effective_runtime(
        effective_torch_training_seed,
        trainer_kwargs,
        execution_config,
        _effective_dataloader_settings(
            data_product,
            train_loader,
            execution_config,
        ),
        trainer=trainer,
    )

    rect_s1s2_mode = getattr(pt_model_config, "rect_s1s2_init", "ones")
    rect_s1s2_initialization = _initialize_rect_s1s2(
        model,
        mode=rect_s1s2_mode,
        training_loader=_rect_s1s2_training_loader(
            data_product,
            train_loader,
            rect_s1s2_mode,
        ),
    )
    logger.info(
        "rect_s1s2 initialization: %s",
        rect_s1s2_initialization,
    )
    training_summary_cb.set_record(rect_s1s2_initialization)

    # B2.6: Execute training cycle
    logger.info(
        "Starting Lightning training: %s epochs",
        pt_training_config.epochs,
    )
    if isinstance(data_product, PrebuiltPtychoDataModule):
        try:
            trainer.fit(model, datamodule = data_product)
        except Exception as e:
            logger.error(f"Lightning training failed: {e}")
            raise RuntimeError(f"Lightning training failed. See logs for details.") from e
    else:
        try:
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        except Exception as e:
            logger.error(f"Lightning training failed: {e}")
            raise RuntimeError(f"Lightning training failed. See logs for details.") from e

    if bool(getattr(trainer, "is_global_zero", False)):
        write_effective_runtime_json(
            output_dir / "effective_runtime.json",
            effective_runtime,
        )

    # Extract loss history from the custom callback
    # The _LossHistoryCallback collects losses per epoch during training
    history = {
        "train_loss": loss_history_cb.train_loss,
        "val_loss": loss_history_cb.val_loss if test_container is not None or isinstance(data_product, PrebuiltPtychoDataModule) else None
    }

    logger.info("Lightning training complete")

    # B2.7: Build results payload with dual-model dict for bundle persistence (Phase C4.D3)
    # save_torch_bundle requires 'autoencoder' and 'diffraction_to_obj' keys per spec §4.6
    # PyTorch uses one trained unified module for both logical bundle roles.
    return {
        "history": history,
        "train_container": train_container,
        "test_container": test_container,
        "rect_s1s2_initialization": rect_s1s2_initialization,
        "training_summary_path": training_summary_path,
        "execution_config": execution_config,
        "effective_runtime": effective_runtime,
        "models": {
            "diffraction_to_obj": model,
            "autoencoder": model,
        }
    }

def _reassemble_cdi_image_torch_mmap(
   test_data: PtychoDataset,
   config: InferenceConfig,
   payload: InferencePayload,
   execution_config: PyTorchExecutionConfig,
   train_results: Optional[Dict[str, Any]] = None,
   verbose = True
):
    """
    Reassemble CDI image using optimized CDI image functions from ptycho_torch library
    """
    
    #Import 
    try:
        from ptycho_torch.reassembly import reconstruct_image_barycentric
        from ptycho_torch.config_params import TrainingConfig
    except Exception as e:
        print(f"Could not import due to exception: {e}")
    
    #Getting proper configs
    data_config = payload.pt_data_config
    inference_config = payload.pt_inference_config
    #Dummy argument to get reconstruct function tow ork
    model_config = None

    #Loading model
    loaded_model = train_results['models']['diffraction_to_obj']
    loaded_model.eval()
    loaded_model.to(execution_config.accelerator)
    loaded_model.training = True

    #Workaround since the only use of training_config is for device
    #Will pass in PyTorch TrainingConfig so we don't need to modify reconstruct_image_baryecentric
    training_config = TrainingConfig(device = execution_config.accelerator)

    #Reconstructing. Automatically puts dataset into dataloader, so don't worry about it
    if verbose:
        print(f"Data config: {data_config}")
        print(f"Inference config: {inference_config}")

    #Call optimized reconstruction method

    result, recon_dataset, _ = reconstruct_image_barycentric(loaded_model, test_data,
                           training_config, data_config, model_config, inference_config, gpu_ids = None,
                           use_mixed_precision=True, verbose = False)
    
    
    return result.to('cpu')
    



def _reassemble_cdi_image_torch(
    test_data: Union[RawData, 'PtychoDataContainerTorch'],
    config: TrainingConfig,
    flip_x: bool,
    flip_y: bool,
    transpose: bool,
    M: int,
    train_results: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Reassemble CDI image using trained PyTorch model.

    This function provides API parity with ptycho.workflows.components.reassemble_cdi_image,
    orchestrating model inference and patch reassembly to produce final reconstruction.

    Args:
        test_data: Test data for reconstruction (RawData or PtychoDataContainerTorch)
        config: TrainingConfig for inference parameters
        flip_x: Whether to flip the x coordinates during reconstruction
        flip_y: Whether to flip the y coordinates during reconstruction
        transpose: Whether to transpose the image by swapping dimensions
        M: Parameter for reassemble_position function
        train_results: Optional training results dict containing 'models' with trained Lightning module

    Returns:
        Tuple containing:
        - recon_amp: Reconstructed amplitude array (np.ndarray)
        - recon_phase: Reconstructed phase array (np.ndarray)
        - results: Dictionary with intermediate outputs (obj_tensor_full, global_offsets, etc.)

    Raises:
        RuntimeError: If PyTorch not available or train_results not provided
        ValueError: If models dict missing from train_results

    Example:
        >>> train_results = run_cdi_example_torch(train_data, test_data, config, do_stitching=False)
        >>> recon_amp, recon_phase, results = _reassemble_cdi_image_torch(
        ...     test_data, config, flip_x=False, flip_y=False, transpose=False, M=20,
        ...     train_results=train_results
        ... )
    """
    # torch-optional import guarded here
    try:
        import torch
        import numpy as np
    except ImportError as e:
        raise RuntimeError(
            "PyTorch backend requires torch. "
            "Install with: pip install -e .[torch]\n"
            "See docs/workflows/pytorch.md for installation guidance."
        ) from e

    # Validate train_results contains models
    if train_results is None:
        # For backward compatibility with tests expecting NotImplementedError,
        # raise NotImplementedError to maintain RED test expectations
        raise NotImplementedError(
            "PyTorch stitching path not yet fully implemented without train_results. "
            "Must pass train_results from run_cdi_example_torch(..., do_stitching=False) output. "
            "See plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md C3 for implementation status."
        )
    if 'models' not in train_results or not train_results['models']:
        raise ValueError("train_results['models'] dict required for inference")

    # Step 1: Normalize test_data → PtychoDataContainerTorch
    test_container = _ensure_container(test_data, config)

    # Step 2: Extract trained Lightning module and set to eval mode
    # Extract Lightning module from dual-model dict (Phase C4.D3 structure)
    lightning_module = train_results['models']['diffraction_to_obj']
    lightning_module.eval()

    # Step 3: Build inference dataloader
    infer_loader = _build_inference_dataloader(test_container, config)

    # Step 4: Extract probe and scale factors for inference
    # Probe tensor is required for forward_predict; extract from container
    probe_tensor = getattr(test_container, 'probeGuess', None)
    if probe_tensor is None:
        # Fallback: create dummy probe if not available
        logger.warning("probeGuess not found in test_container; using dummy probe")
        probe_tensor = torch.ones(1, config.model.N, config.model.N, dtype=torch.complex64)
    if not isinstance(probe_tensor, torch.Tensor):
        probe_tensor = torch.tensor(probe_tensor)

    # Step 5: Run inference to collect predictions and offsets
    obj_patches = []
    global_offsets = test_container.global_offsets.clone()  # (n_samples, 1, 2, 1)

    with torch.no_grad():
        for batch in infer_loader:
            # batch is (X, coords) from TensorDataset
            X_batch, coords_batch = batch
            # DTYPE ENFORCEMENT (Phase D1d): Ensure float32 before Lightning forward
            X_batch = X_batch.to(torch.float32)
            coords_batch = coords_batch.to(torch.float32)

            # Create batch-sized scale factors with proper shape for broadcasting
            # IntensityScalerModule expects (B, 1, 1, 1) shaped scale factor
            batch_size = X_batch.shape[0]
            input_scale = torch.ones(batch_size, 1, 1, 1, device=X_batch.device, dtype=X_batch.dtype)

            # Call forward_predict which returns complex object patches
            # Signature: forward_predict(x, positions, probe, input_scale_factor)
            pred = lightning_module.forward_predict(
                X_batch,
                coords_batch,
                probe_tensor.to(X_batch.device),
                input_scale
            )
            obj_patches.append(pred.cpu())

    # Concatenate all predictions
    obj_tensor_full = torch.cat(obj_patches, dim=0)  # (n_samples, ...)

    # Ensure 4D tensor for reassembly (n_samples, 1, H, W) or (n_samples, C, H, W)
    if obj_tensor_full.ndim == 3:
        obj_tensor_full = obj_tensor_full.unsqueeze(1)  # Add channel dim

    # Step 5: Apply coordinate transformations
    if transpose:
        # Transpose spatial dimensions
        obj_tensor_full = obj_tensor_full.transpose(-2, -1)

    if flip_x:
        global_offsets[:, 0, 0, :] = -global_offsets[:, 0, 0, :]
    if flip_y:
        global_offsets[:, 0, 1, :] = -global_offsets[:, 0, 1, :]

    # Step 6: Prepare tensor for TensorFlow reassembly helper
    # TensorFlow helper expects (n_samples, H, W, 1) single-channel complex tensor
    # PyTorch models output (n_samples, C, H, W) with C=gridsize**2 channels
    # See debug_shape_triage.md (2025-10-19T092448Z) for root cause analysis

    # Convert channel-first to channel-last if needed
    if obj_tensor_full.ndim == 4 and obj_tensor_full.shape[1] < obj_tensor_full.shape[2] and obj_tensor_full.shape[1] < obj_tensor_full.shape[3]:
        # Channel dim is dim=1 (channel-first); move to end
        obj_tensor_full = obj_tensor_full.permute(0, 2, 3, 1)  # (n, C, H, W) → (n, H, W, C)

    # Reduce multi-channel output to single channel for TensorFlow reassembly
    # For gridsize > 1, model outputs multiple channels (gridsize**2); take mean across channels
    if obj_tensor_full.shape[-1] > 1:
        obj_tensor_full = torch.mean(obj_tensor_full, dim=-1, keepdim=True)  # (n, H, W, C) → (n, H, W, 1)

    # Step 7: Reassemble patches (using TensorFlow helper for MVP parity)
    # For Phase D2.C, delegate to TensorFlow reassembly to maintain exact parity
    # Future enhancement: use native PyTorch reassembly from ptycho_torch.reassembly
    from ptycho import tf_helper as hh
    obj_tensor_np = obj_tensor_full.cpu().numpy()
    global_offsets_np = global_offsets.cpu().numpy()
    if (global_offsets_np.ndim == 4
            and global_offsets_np.shape[2] == 2
            and global_offsets_np.shape[3] == 1):
        global_offsets_np = np.swapaxes(global_offsets_np, 2, 3)

    obj_image = _reassemble_position_with_legacy_geometry(
        obj_tensor_np,
        global_offsets_np,
        M=M,
        config=config,
    )

    # Squeeze trailing channel dimension if present (reassembly may return (H, W, 1))
    if obj_image.ndim == 3 and obj_image.shape[-1] == 1:
        obj_image = np.squeeze(obj_image, axis=-1)

    # Step 8: Extract amplitude and phase
    recon_amp = np.absolute(obj_image)
    recon_phase = np.angle(obj_image)

    # Step 8: Build results dict
    results = {
        "obj_tensor_full": obj_tensor_np,
        "global_offsets": global_offsets_np,
        "recon_amp": recon_amp,
        "recon_phase": recon_phase,
        "containers": {
            "test": test_container
        }
    }

    return recon_amp, recon_phase, results


def _reassemble_position_with_legacy_geometry(
    obj_tensor,
    global_offsets,
    *,
    M: int,
    config: TrainingConfig,
):
    """Contain the protected TensorFlow helper's remaining global geometry read.

    Remove this adapter once ``tf_helper.reassemble_position`` accepts its
    detector size explicitly.
    """
    from ptycho import params as legacy_params
    from ptycho import tf_helper
    from ptycho.config.config import update_legacy_dict
    from ptycho.config.legacy_state import (
        configured_params_scope,
        legacy_params_scope,
    )

    with legacy_params_scope():
        with configured_params_scope():
            update_legacy_dict(legacy_params.cfg, config)
            return tf_helper.reassemble_position(
                obj_tensor,
                global_offsets,
                M=M,
            )


def train_cdi_model_torch(
    train_data: Union[RawData, 'PtychoDataContainerTorch', 'PtychoDataset'],
    test_data: Optional[Union[RawData, 'PtychoDataContainerTorch']],
    config: TrainingConfig,
    execution_config: Optional[Any] = None,
    overrides: Optional[dict] = None,
    *,
    resolved_payload: Optional[TrainingPayload] = None,
    torch_training_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train the CDI model using PyTorch Lightning backend.

    This function provides API parity with ptycho.workflows.components.train_cdi_model,
    orchestrating data preparation, probe initialization, and Lightning trainer execution.

    Args:
        train_data: Training data (RawData, PtychoDataContainerTorch, or PtychoDataset)
        test_data: Optional test data for validation
        config: TrainingConfig instance (TensorFlow dataclass)
        execution_config: Optional unresolved ExecutionRequest for runtime control
        overrides: Optional torch-only factory overrides.

    Returns:
        Dict[str, Any]: Results dictionary containing:
        - 'history': Training history (losses, metrics)
        - 'train_container': PtychoDataContainerTorch for training data
        - 'test_container': Optional PtychoDataContainerTorch for test data
        - Additional outputs from Lightning trainer

    Raises:
        ImportError: If Phase C adapters not available
        TypeError: If input data types are invalid

    Phase D2.B Status:
        - Entry signature: ✅ COMPLETE (matches TensorFlow)
        - _ensure_container helper: ✅ COMPLETE (normalizes inputs via Phase C adapters)
        - Lightning orchestration: 🔶 STUB (returns minimal dict, full impl pending)
        - Torch-optional: ✅ COMPLETE (importable without torch)

    Example:
        >>> config = TrainingConfig(model=ModelConfig(N=64), nepochs=10, ...)
        >>> results = train_cdi_model_torch(train_data, test_data, config)
        >>> print(results['history']['train_loss'][-1])
    """
    _validate_training_execution_input(execution_config, resolved_payload)

    # Step 1: Normalize train_data to PtychoDataContainerTorch
    logger.info("Normalizing training data via _ensure_container")
    train_container = _ensure_container(train_data, config)

    # Step 2: Normalize test_data if provided
    test_container = None
    if test_data is not None:
        logger.info("Normalizing test data via _ensure_container")
        test_container = _ensure_container(test_data, config)

    # Step 3: Initialize probe (TODO: implement probe handling for PyTorch)
    # TensorFlow baseline: probe.set_probe_guess(None, train_container.probe)
    # For Phase D2.B stub, skip probe initialization
    logger.debug("Probe initialization deferred to full Lightning implementation")

    # Step 4: Delegate to Lightning trainer
    logger.info("Delegating to Lightning trainer via _train_with_lightning")
    lightning_kwargs = {}
    if execution_config is not None:
        lightning_kwargs["execution_config"] = execution_config
    if overrides is not None:
        lightning_kwargs["overrides"] = overrides
    if resolved_payload is not None:
        lightning_kwargs["resolved_payload"] = resolved_payload
    if torch_training_seed is not None:
        lightning_kwargs["torch_training_seed"] = torch_training_seed
    results = _train_with_lightning(
        train_container,
        test_container,
        config,
        **lightning_kwargs,
    )
    if hasattr(train_container, 'physics_scaling_constant'):
        import torch
        scale_tensor = torch.as_tensor(train_container.physics_scaling_constant)
        results['intensity_scale'] = float(scale_tensor.reshape(-1)[0].item())

    return results


@contextmanager
def _pinned_bundle_snapshot(zip_path: Path):
    """Yield an immutable private snapshot of one archive generation."""
    from collections import Counter
    import shutil
    import tempfile

    if not zip_path.is_file():
        raise FileNotFoundError(f"Model archive not found: {zip_path}")

    with tempfile.TemporaryDirectory(
        prefix="ptycho-torch-bundle-snapshot-"
    ) as temporary_directory:
        snapshot_zip_path = Path(temporary_directory) / zip_path.name
        with zip_path.open("rb") as source, snapshot_zip_path.open("wb") as target:
            shutil.copyfileobj(source, target)

        with zipfile.ZipFile(snapshot_zip_path, "r") as archive:
            counts = Counter(info.filename for info in archive.infolist())
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(
                "Torch bundle contains duplicate archive member(s): "
                + ", ".join(duplicates)
            )

        yield snapshot_zip_path.with_suffix(""), snapshot_zip_path


def _decode_pinned_inference_bundle(
    archive_path: Path,
    zip_path: Path,
    *,
    model_name: str,
    explicit_profile: Optional[Tuple[str, str]],
) -> Tuple[Any, dict, Optional[AmplitudePhysicsGainRecord]]:
    """Decode and reconstruct all members from one private snapshot."""
    from ptycho_torch.artifact_schema import (
        ARTIFACT_SCHEMA_V1_VERSION,
        CURRENT_ARTIFACT_SCHEMA_VERSION,
        validate_torch_bundle_manifest,
    )

    manifest, params_dict = _read_torch_bundle_manifest_and_params(
        str(archive_path)
    )
    manifest_era = validate_torch_bundle_manifest(manifest)
    metadata = _read_bundle_scaling_metadata(zip_path)
    amplitude_physics_gain_record = (
        _read_bundle_amplitude_physics_gain_record(zip_path)
    )
    if metadata is None:
        known_legacy = (
            params_dict.get("_version") == "2.0-pytorch"
            and "scale_contract_version" not in params_dict
            and "measurement_domain" not in params_dict
        )
        if not known_legacy:
            raise ValueError(
                "wts.h5.zip has no versioned Torch metadata and is not the "
                "declared metadata-free legacy_v1 era"
            )
        identity = None
    else:
        metadata_schema = metadata.get("schema_version")
        if (
            manifest_era in {
                ARTIFACT_SCHEMA_V1_VERSION,
                CURRENT_ARTIFACT_SCHEMA_VERSION,
            }
            and metadata_schema != manifest_era
        ):
            raise ValueError(
                "wts.h5.zip root manifest and metadata schemas disagree: "
                f"manifest={manifest_era!r}, declares {metadata_schema!r}"
            )
        if (
            manifest_era == "metadata-free-legacy"
            and metadata_schema != "ci-entrypoints-v1"
        ):
            raise ValueError(
                "wts.h5.zip legacy root supports only transitional "
                f"ci-entrypoints-v1 metadata, found {metadata_schema!r}"
            )
        identity = _decode_bundle_metadata(metadata)

    if amplitude_physics_gain_record is not None:
        if identity is None:
            raise ValueError(
                "amplitude physics gain sidecar requires persisted ModelSpec "
                "metadata for its scalar join"
            )
        model_gain = validate_amplitude_physics_gain(
            identity.model_spec.to_model_config()
        )
        if amplitude_physics_gain_record.value != model_gain:
            raise ValueError(
                "amplitude physics gain record disagrees with persisted "
                "ModelSpec amplitude_physics_gain"
            )

    models_dict, params_dict, _ = _reconstruct_inference_bundle_explicit(
        archive_path,
        zip_path,
        manifest=manifest,
        params_dict=params_dict,
        identity=identity,
        explicit_profile=explicit_profile,
        model_name=model_name,
    )
    return models_dict, params_dict, amplitude_physics_gain_record


@transactional_legacy_params
def load_inference_bundle_torch(
    bundle_dir: Union[str, Path],
    model_name: str = "diffraction_to_obj",
    *,
    scale_contract_version: Optional[str] = None,
    measurement_domain: Optional[str] = None,
) -> Tuple[Any, dict]:
    """Strictly load a trained PyTorch bundle from a pinned snapshot."""
    archive_path = Path(bundle_dir) / "wts.h5"
    zip_path = archive_path.with_suffix(".h5.zip")
    logger.info("Loading PyTorch inference bundle from %s.zip", archive_path)

    from ptycho_torch.config_factory import resolve_profile_overrides

    explicit_profile = resolve_profile_overrides(
        {
            "scale_contract_version": scale_contract_version,
            "measurement_domain": measurement_domain,
        }
    )
    with _pinned_bundle_snapshot(zip_path) as (
        pinned_archive_path,
        pinned_zip_path,
    ):
        models_dict, params_dict, amplitude_physics_gain_record = (
            _decode_pinned_inference_bundle(
                pinned_archive_path,
                pinned_zip_path,
                model_name=model_name,
                explicit_profile=explicit_profile,
            )
        )

    params.cfg.update(params_dict)
    returned_params = dict(params_dict)
    if amplitude_physics_gain_record is not None:
        returned_params["amplitude_physics_gain_record"] = (
            amplitude_physics_gain_record
        )

    logger.info(
        "Inference bundle loaded successfully. Models: %s, Params keys: %s...",
        list(models_dict),
        list(params_dict)[:5],
    )
    return models_dict, returned_params
