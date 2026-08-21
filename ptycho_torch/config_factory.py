"""
Configuration factory functions for PyTorch backend.

This module provides centralized factory functions that translate canonical TensorFlow
configurations plus PyTorch execution overrides into the objects consumed by the PyTorch
backend, eliminating duplicated config construction logic scattered across CLI and workflow
entry points.

Design documentation: plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/factory_design.md

Architecture:
    CLI Args/Workflow Params
      ↓
    [Pure Resolved-Payload Entry Point]
      ↓
    [Validate + Infer + Apply Overrides]
      ↓
    [Translate to TensorFlow Canonical Configs via config_bridge]
      ↓
    [Return Payload]
      ↓
    [Optional create_* compatibility entry point]
      ↓
    [Populate params.cfg (CONFIG-001 checkpoint)]

Core Functions:
    resolve_training_payload(): Pure training payload resolution
    resolve_inference_payload(): Pure inference payload resolution
    create_training_payload(): Compatibility resolution plus CONFIG-001
    create_inference_payload(): Compatibility resolution plus CONFIG-001
    infer_probe_size(): Extracts probe size from NPZ metadata
    populate_legacy_params(): Wrapper around update_legacy_dict with validation

Design Principles:
    - Single Responsibility: Each factory handles one workflow (training vs inference)
    - Bridge Delegation: All TensorFlow dataclass translation delegated to config_bridge.py
    - CONFIG-001 Compliance: Factories ensure update_legacy_dict() called before data loading
    - Override Transparency: Explicit override dict parameter for execution-specific knobs
    - Test-Driven: RED tests written before implementation (Phase B2.b)

Ownership:
    - ``overrides`` contains canonical scientific, model, and training inputs.
    - ``ExecutionRequest`` contains unresolved runtime inputs only.
    - ``PyTorchExecutionConfig`` is the resolved runtime output carrier.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Union
import warnings

# Import canonical TensorFlow configs (single source of truth)
from ptycho.config.config import (
    TrainingConfig as TFTrainingConfig,
    InferenceConfig as TFInferenceConfig,
)

# Import PyTorch singleton configs
from ptycho_torch.config_params import (
    DataConfig as PTDataConfig,
    DatagenConfig as PTDatagenConfig,
    ModelConfig as PTModelConfig,
    TrainingConfig as PTTrainingConfig,
    InferenceConfig as PTInferenceConfig,
)

# Import PyTorchExecutionConfig (Option A canonical location per ADR-003 Phase C1)
# Per supervisor decision at 2025-10-19T234458Z (factory_design.md §2.2)
from ptycho.config.config import PyTorchExecutionConfig
from ptycho.config.legacy_state import configured_legacy_params

from ptycho import params
from ptycho_torch.scaling_contract import (
    CI_SCALE_CONTRACT,
    COUNT_INTENSITY,
    resolve_scale_contract,
)
from ptycho_torch.model_spec import ModelSpec, derive_model_spec
from ptycho_torch.execution_request import (
    ExecutionCapabilities,
    ExecutionRequest,
    ResolutionNotice,
    normalize_execution_input,
    resolve_runtime_execution_request,
    validate_execution_input_phase,
    validate_execution_input_structure,
)
from ptycho_torch.config_resolution import (
    InferenceObservations,
    TRAINING_OWNER_FIELDS as TRAINING_OWNER_FIELDS,
    TrainingObservations,
    inference_factory_baseline,
    normalize_inference_patch,
    normalize_training_patch,
    observe_probe_size,
    resolve_inference_bundle,
    resolve_training_bundle,
    training_factory_baseline,
)

# Conformance D3 (Theme 3, docs/superpowers/plans/
# 2026-07-14-ci-paper-conformance-audit.md): the paper's "PtychoPINN-CI" as a
# single named preset. The five CONTRACT fields are fail-closed — an explicit
# user override contradicting them raises instead of silently mixing profiles.
CI_PROFILE_CONTRACT_FIELDS: Dict[str, Any] = {
    "scale_contract_version": CI_SCALE_CONTRACT,
    "measurement_domain": COUNT_INTENSITY,
    "physics_forward_mode": "rectangular_scaled",
    "torch_loss_mode": "poisson",
    "loss_function": "Poisson",
}

# Full coherent bundle. Non-contract entries are profile defaults a user may
# override: rect_s1s2_init='dose_closure', rect_s1s2_trainable=True (trainable s1/s2
# own the training scale), amplitude_physics_gain=1.0 (the
# rectangular contract rejects anything else fail-closed), and
# cnn_output_mode='real_imag' (cnn architecture only; other generators already
# default generator_output_mode='real_imag').
CI_PROFILE_BUNDLE: Dict[str, Any] = {
    **CI_PROFILE_CONTRACT_FIELDS,
    "amplitude_physics_gain": 1.0,
    "rect_s1s2_trainable": True,
    "rect_s1s2_init": "dose_closure",
    "cnn_output_mode": "real_imag",
}


def _emit_resolution_notices(
    notices: tuple[ResolutionNotice, ...],
) -> None:
    for notice in notices:
        warnings.warn(
            notice.message,
            notice.category,
            stacklevel=3,
        )


def resolve_ci_profile(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the canonical coherent CI override bundle merged with ``overrides``.

    Pure function. Precedence: user overrides win for NON-CONTRACT fields; an
    explicit override contradicting one of the five contract fields
    (``CI_PROFILE_CONTRACT_FIELDS``) raises ``ValueError`` naming both values.

    Raises:
        ValueError: an override contradicts a CI contract field (fail-closed).
    """
    overrides = dict(overrides or {})
    for field_name, required in CI_PROFILE_CONTRACT_FIELDS.items():
        if field_name in overrides and overrides[field_name] != required:
            raise ValueError(
                f"the CI profile requires {field_name}={required!r}; you "
                f"passed {overrides[field_name]!r}. The named CI profile is "
                "fail-closed: remove the contradicting override, or drop "
                "profile='ci' to assemble a custom configuration explicitly."
            )
    resolved = dict(CI_PROFILE_BUNDLE)
    resolved.update(overrides)
    return resolved


def simulation_from_datagen_config(
    datagen_config: PTDatagenConfig,
    *,
    base=None,
):
    """Convert the legacy Torch generation payload at the factory boundary."""
    if not isinstance(datagen_config, PTDatagenConfig):
        raise TypeError("datagen_config must be a DatagenConfig")
    return datagen_config.to_simulation_config(base=base)


def datagen_config_from_simulation(simulation) -> PTDatagenConfig:
    """Project a canonical recipe into the unchanged Torch checkpoint payload."""
    return PTDatagenConfig.from_simulation_config(simulation)


def resolve_profile_overrides(overrides: Optional[Dict[str, Any]]) -> Optional[tuple[str, str]]:
    """Validate an explicit scale-contract override as an inseparable pair."""
    overrides = overrides or {}
    version = overrides.get("scale_contract_version")
    measurement_domain = overrides.get("measurement_domain")
    supplied = (version is not None, measurement_domain is not None)
    if supplied == (False, False):
        return None
    if supplied[0] != supplied[1]:
        raise ValueError(
            "scale_contract_version and measurement_domain must be supplied together. "
            "Explicit legacy compatibility requires scale_contract_version='legacy_v1' "
            "and measurement_domain='normalized_amplitude'."
        )
    try:
        resolved = resolve_scale_contract(version, measurement_domain)
    except ValueError as exc:
        raise ValueError(
            "scale_contract_version and measurement_domain must select a supported "
            f"profile together: {exc}"
        ) from exc
    return resolved.version, resolved.measurement_domain


@dataclass
class TrainingPayload:
    """
    Complete configuration bundle for training workflows.

    Returned by create_training_payload(). Contains all config objects needed
    to execute PyTorch training: canonical TensorFlow config (for params.cfg bridge),
    PyTorch singleton configs (for Lightning module), execution config (runtime knobs),
    and audit trail of applied overrides.
    """
    tf_training_config: TFTrainingConfig  # Canonical TensorFlow format
    pt_data_config: PTDataConfig  # PyTorch singleton
    pt_model_config: PTModelConfig  # PyTorch singleton
    pt_training_config: PTTrainingConfig  # PyTorch singleton
    pt_inference_config: PTInferenceConfig  # PyTorch singleton (patch-stats, inference defaults)
    model_spec: ModelSpec  # Versioned internal Torch structural identity
    execution_config: PyTorchExecutionConfig  # Execution knobs (Phase C2)
    overrides_applied: Dict[str, Any] = field(default_factory=dict)  # Audit trail


@dataclass
class InferencePayload:
    """
    Complete configuration bundle for inference workflows.

    Returned by create_inference_payload(). Contains all config objects needed
    to execute PyTorch inference: canonical TensorFlow config (for params.cfg bridge),
    PyTorch singleton configs (for Lightning module), execution config (runtime knobs),
    and audit trail of applied overrides.
    """
    tf_inference_config: TFInferenceConfig  # Canonical TensorFlow format
    pt_data_config: PTDataConfig  # PyTorch singleton
    pt_inference_config: PTInferenceConfig  # PyTorch singleton
    execution_config: PyTorchExecutionConfig  # Execution knobs (Phase C2)
    overrides_applied: Dict[str, Any] = field(default_factory=dict)  # Audit trail


def build_training_factory_overrides(
    config: TFTrainingConfig,
) -> Dict[str, Any]:
    """Project a public config into the shared historical factory baseline."""

    if not isinstance(config, TFTrainingConfig):
        raise TypeError("config must be a public TrainingConfig")
    from ptycho.config.config import resolve_model_object_policy

    model = resolve_model_object_policy(
        config.model,
        backend="torch",
        warn_deprecated=False,
    )
    mode = {
        "pinn": "Unsupervised",
        "supervised": "Supervised",
    }[model.model_type]
    data = config.data
    sampling = config.sampling
    loss = config.loss
    gradient_clip = config.gradient_clip
    optimizer = config.optimizer
    scheduler = config.scheduler
    overrides: Dict[str, Any] = {
        "n_groups": sampling.n_groups,
        "gridsize": model.gridsize,
        "architecture": model.architecture,
        "model_type": mode,
        "amp_activation": model.amp_activation,
        "n_filters_scale": model.n_filters_scale,
        "object_layout": model.object_layout,
        "training_canvas": model.training_canvas,
        "training_patch_weighting": model.training_patch_weighting,
        "probe_big": model.probe_big,
        "probe_mask": model.probe_mask,
        "probe_mask_sigma": model.probe_mask_sigma,
        "probe_mask_diameter": model.probe_mask_diameter,
        "pad_object": model.pad_object,
        "probe_scale": model.probe_scale,
        "gaussian_smoothing_sigma": model.gaussian_smoothing_sigma,
        "nphotons": data.nphotons,
        "neighbor_count": sampling.neighbor_count,
        "max_epochs": config.nepochs,
        "batch_size": config.batch_size,
        "subsample_seed": sampling.subsample_seed,
        "enable_oversampling": sampling.enable_oversampling,
        "neighbor_pool_size": sampling.neighbor_pool_size,
        "sequential_sampling": sampling.sequential_sampling,
        "test_data_file": data.test_data_file,
        "torch_loss_mode": loss.torch_loss_mode,
        "torch_mae_pred_l2_match_target": (
            loss.torch_mae_pred_l2_match_target
        ),
        "intensity_scale_trainable": config.intensity_scale_trainable,
        "gradient_clip_val": gradient_clip.val,
        "gradient_clip_algorithm": gradient_clip.algorithm,
        "optimizer": optimizer.algorithm,
        "momentum": optimizer.sgd.momentum,
        "weight_decay": optimizer.weight_decay,
        "adam_beta1": optimizer.adam.beta1,
        "adam_beta2": optimizer.adam.beta2,
        "scheduler": scheduler.kind,
        "lr_warmup_epochs": scheduler.lr_warmup_epochs,
        "lr_min_ratio": scheduler.lr_min_ratio,
        "plateau_factor": scheduler.plateau_factor,
        "plateau_patience": scheduler.plateau_patience,
        "plateau_min_lr": scheduler.plateau_min_lr,
        "plateau_threshold": scheduler.plateau_threshold,
        "log_grad_norm": getattr(config, "log_grad_norm", False),
        "grad_norm_log_freq": getattr(config, "grad_norm_log_freq", 1),
    }
    if model.model_type == "supervised":
        overrides["torch_loss_mode"] = "mae"
    if sampling.n_subsample is not None:
        overrides["n_raw_frames_selected"] = sampling.n_subsample
    for name in (
        "fno_modes",
        "fno_width",
        "fno_blocks",
        "fno_cnn_blocks",
        "fno_input_transform",
        "learned_input_channels",
        "max_hidden_channels",
        "resnet_width",
        "generator_output_mode",
    ):
        value = getattr(model, name, None)
        if value is not None:
            overrides[name] = value
    return overrides


def _load_nphotons_from_metadata(data_file: Path) -> Optional[float]:
    """Return nphotons from embedded NPZ metadata if present."""
    import json
    import numpy as np
    from ptycho.metadata import MetadataManager

    try:
        with np.load(data_file, allow_pickle=True) as data:
            if MetadataManager.METADATA_KEY not in data.files:
                return None
            raw = data[MetadataManager.METADATA_KEY]
            # Metadata stored as 0-d object array or string
            if hasattr(raw, "item"):
                raw = raw.item()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if raw is None:
                return None
            metadata = json.loads(raw)
    except Exception:
        return None

    return MetadataManager.get_nphotons(metadata, default=None)


def _resolve_training_payload(
    train_data_file: Path,
    output_dir: Path,
    overrides: Optional[Dict[str, Any]] = None,
    execution_config: Optional[ExecutionRequest] = None,
    profile: Optional[str] = None,
    *,
    training_baseline: TFTrainingConfig | PTTrainingConfig | None = None,
    execution_capabilities: ExecutionCapabilities | None = None,
) -> tuple[
    TrainingPayload,
    tuple[ResolutionNotice, ...],
    tuple[ResolutionNotice, ...],
]:
    """
    Resolve a complete training configuration payload without legacy mutation.

    Centralizes all config construction logic for PyTorch training workflows.
    Eliminates duplicated wiring in CLI and workflow entry points by providing
    a single factory function that:
    1. Validates required arguments (train_data_file, output_dir, n_groups)
    2. Infers probe size from NPZ metadata (or uses override)
    3. Constructs PyTorch singleton configs (DataConfig, ModelConfig, TrainingConfig, InferenceConfig)
    4. Applies CLI overrides with precedence rules
    5. Translates to TensorFlow canonical configs via config_bridge
    6. Constructs the canonical compatibility config without projecting it
    7. Constructs PyTorchExecutionConfig for runtime knobs
    8. Returns TrainingPayload with all config objects + audit trail

    Args:
        train_data_file: Path to training NPZ dataset (must exist per DATA-001)
        output_dir: Path to output directory for checkpoints/logs (created if missing)
        overrides: Dict of field overrides (highest precedence). Required keys:
            - n_groups: Number of grouped samples (no default, raises error if missing)
            Optional keys: batch_size, gridsize, max_epochs, nphotons, etc.
        execution_config: Unresolved runtime request. ``None`` uses request
            defaults. A resolved ``PyTorchExecutionConfig`` is not an input.
        profile: Optional named configuration profile. ``None`` (default)
            preserves prior behavior bit-for-bit. ``'ci'`` resolves the
            canonical PtychoPINN-CI bundle via resolve_ci_profile() and merges
            it under the user overrides (contract-field contradictions raise).

    Returns:
        TrainingPayload containing:
            - tf_training_config: TrainingConfig (canonical TensorFlow format)
            - pt_data_config: DataConfig (PyTorch singleton)
            - pt_model_config: ModelConfig (PyTorch singleton)
            - pt_training_config: TrainingConfig (PyTorch singleton)
            - pt_inference_config: InferenceConfig (PyTorch singleton)
            - pt_inference_config: InferenceConfig (PyTorch singleton)
            - execution_config: PyTorchExecutionConfig (runtime knobs)
            - overrides_applied: Dict[str, Any] (audit trail)

    Raises:
        FileNotFoundError: train_data_file does not exist
        ValueError: n_groups missing in overrides (required field)
        ValueError: Invalid field values (N <= 0, batch_size <= 0, etc.)

    Example:
        >>> from pathlib import Path
        >>> payload = create_training_payload(
        ...     train_data_file=Path('datasets/train.npz'),
        ...     output_dir=Path('outputs/exp001'),
        ...     overrides={
        ...         'n_groups': 512,
        ...         'batch_size': 16,
        ...         'gridsize': 2,
        ...         'max_epochs': 10,
        ...     },
        ...     execution_config=ExecutionRequest(
        ...         values={'accelerator': 'cpu'},
        ...         explicit_fields=frozenset({'accelerator'}),
        ...     ),
        ... )
        >>> assert isinstance(payload.tf_training_config, TrainingConfig)
        >>> assert payload.tf_training_config.sampling.n_groups == 512

    See also:
        - Design: plans/active/ADR-003-BACKEND-API/reports/.../factory_design.md §3.1
        - Override precedence: .../override_matrix.md §6
        - Integration: .../factory_design.md §3 (CLI/workflow call sites)
    """
    if execution_config is not None and not isinstance(
        execution_config,
        ExecutionRequest,
    ):
        raise TypeError(
            "execution_config must be an ExecutionRequest or None; "
            "PyTorchExecutionConfig is a resolved output carrier"
        )

    raw_patch = dict(overrides or {})
    if profile is not None:
        if profile != "ci":
            raise ValueError(
                f"Unknown configuration profile {profile!r}; supported "
                "profiles: 'ci'."
            )
        raw_patch = resolve_ci_profile(raw_patch)

    resolved_profile = resolve_profile_overrides(raw_patch)
    if resolved_profile is not None:
        (
            raw_patch["scale_contract_version"],
            raw_patch["measurement_domain"],
        ) = resolved_profile

    normalized_execution = normalize_execution_input(
        execution_config,
        mode="training",
    )
    if normalized_execution is not None:
        validate_execution_input_structure(normalized_execution)
        validate_execution_input_phase(
            normalized_execution,
            mode="training",
        )
    normalized = normalize_training_patch(raw_patch)
    if not train_data_file.exists():
        raise FileNotFoundError(f"Training data file not found: {train_data_file}")
    probe_observation = observe_probe_size(train_data_file)
    resolved = resolve_training_bundle(
        baseline=training_factory_baseline(
            training_baseline=training_baseline,
        ),
        normalized=normalized,
        observations=TrainingObservations(
            train_data_file=train_data_file,
            output_dir=output_dir,
            inferred_probe_size=probe_observation.value,
            photon_metadata=_load_nphotons_from_metadata(train_data_file),
            notices=probe_observation.notices,
        ),
    )

    from ptycho_torch.config_bridge import to_model_config, to_training_config

    deferred_notices: list[ResolutionNotice] = list(resolved.notices)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tf_model_config = to_model_config(resolved.data, resolved.model)
        tf_training_config = to_training_config(
            tf_model_config,
            resolved.data,
            resolved.model,
            resolved.training,
            overrides=dict(resolved.bridge),
        )
    deferred_notices.extend(
        ResolutionNotice(item.category, str(item.message))
        for item in caught
    )

    runtime = resolve_runtime_execution_request(
        normalized_execution,
        mode="training",
        execution_capabilities=execution_capabilities,
    )
    overrides_applied = dict(resolved.audit)
    if resolved.aliases:
        overrides_applied["input_aliases"] = {
            name: tuple(sources)
            for name, sources in resolved.aliases.items()
        }
    if profile is not None:
        overrides_applied["profile"] = profile
    overrides_applied["execution_runtime"] = runtime.audit_dict()
    for name in (
        "accelerator",
        "deterministic",
        "num_workers",
        "enable_progress_bar",
        "logger_backend",
    ):
        overrides_applied[name] = getattr(runtime.config, name)

    model_spec = derive_model_spec(
        tf_model_config,
        resolved.model,
        resolved.data,
    )
    payload = TrainingPayload(
        tf_training_config=tf_training_config,
        pt_data_config=resolved.data,
        pt_model_config=resolved.model,
        pt_training_config=resolved.training,
        pt_inference_config=resolved.inference,
        model_spec=model_spec,
        execution_config=runtime.config,
        overrides_applied=overrides_applied,
    )
    return payload, tuple(deferred_notices), runtime.notices


def _resolve_inference_payload(
    model_path: Path,
    test_data_file: Path,
    output_dir: Path,
    overrides: Optional[Dict[str, Any]] = None,
    execution_config: Optional[ExecutionRequest] = None,
    *,
    execution_capabilities: ExecutionCapabilities | None = None,
) -> tuple[
    InferencePayload,
    tuple[ResolutionNotice, ...],
    tuple[ResolutionNotice, ...],
]:
    """
    Resolve a complete inference configuration payload without legacy mutation.

    Centralizes all config construction logic for PyTorch inference workflows.
    Eliminates duplicated wiring in CLI and workflow entry points by providing
    a single factory function that:
    1. Validates required arguments (model_path, test_data_file, output_dir, n_groups)
    2. Loads checkpoint config from model_path (or infers from NPZ)
    3. Constructs PyTorch singleton configs (DataConfig, InferenceConfig)
    4. Applies CLI overrides with precedence rules
    5. Translates to TensorFlow canonical configs via config_bridge
    6. Constructs the canonical compatibility config without projecting it
    7. Constructs PyTorchExecutionConfig for runtime knobs
    8. Returns InferencePayload with all config objects + audit trail

    Args:
        model_path: Path to trained model directory (must contain wts.h5.zip)
        test_data_file: Path to test NPZ dataset (must exist per DATA-001)
        output_dir: Path to output directory for reconstructions (created if missing)
        overrides: Dict of field overrides (highest precedence). Required keys:
            - n_groups: Number of grouped samples (no default, raises error if missing)
            Optional keys: gridsize, batch_size, middle_trim, pad_eval, etc.
        execution_config: Unresolved runtime request. ``None`` uses request
            defaults. A resolved ``PyTorchExecutionConfig`` is not an input.

    Returns:
        InferencePayload containing:
            - tf_inference_config: InferenceConfig (canonical TensorFlow format)
            - pt_data_config: DataConfig (PyTorch singleton)
            - pt_inference_config: InferenceConfig (PyTorch singleton)
            - execution_config: PyTorchExecutionConfig (runtime knobs)
            - overrides_applied: Dict[str, Any] (audit trail)

    Raises:
        FileNotFoundError: model_path or test_data_file does not exist
        ValueError: model_path missing wts.h5.zip
        ValueError: n_groups missing in overrides (required field)

    Example:
        >>> payload = create_inference_payload(
        ...     model_path=Path('outputs/exp001'),
        ...     test_data_file=Path('datasets/test.npz'),
        ...     output_dir=Path('outputs/exp001/inference'),
        ...     overrides={
        ...         'n_groups': 128,
        ...         'gridsize': 2,
        ...     },
        ...     execution_config=ExecutionRequest(
        ...         values={'inference_batch_size': 64},
        ...         explicit_fields=frozenset({'inference_batch_size'}),
        ...     ),
        ... )

    See also:
        - Design: .../factory_design.md §3.3
        - Checkpoint loading: specs/ptychodus_api_spec.md §4.6
    """
    if execution_config is not None and not isinstance(
        execution_config,
        ExecutionRequest,
    ):
        raise TypeError(
            "execution_config must be an ExecutionRequest or None; "
            "PyTorchExecutionConfig is a resolved output carrier"
        )

    raw_patch = dict(overrides or {})
    resolved_profile = resolve_profile_overrides(raw_patch)
    if resolved_profile is not None:
        (
            raw_patch["scale_contract_version"],
            raw_patch["measurement_domain"],
        ) = resolved_profile

    normalized_execution = normalize_execution_input(
        execution_config,
        mode="inference",
    )
    if normalized_execution is not None:
        validate_execution_input_structure(normalized_execution)
        validate_execution_input_phase(
            normalized_execution,
            mode="inference",
        )
    normalized = normalize_inference_patch(raw_patch)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    checkpoint_file = model_path / "wts.h5.zip"
    if not checkpoint_file.exists():
        raise ValueError(
            f"Model archive not found: {checkpoint_file}. "
            "Expected wts.h5.zip in model_path directory."
        )

    if not test_data_file.exists():
        raise FileNotFoundError(f"Test data file not found: {test_data_file}")

    probe_observation = observe_probe_size(test_data_file)
    resolved = resolve_inference_bundle(
        baseline=inference_factory_baseline(),
        normalized=normalized,
        observations=InferenceObservations(
            model_path=model_path,
            test_data_file=test_data_file,
            output_dir=output_dir,
            inferred_probe_size=probe_observation.value,
            notices=probe_observation.notices,
        ),
    )

    from ptycho_torch.config_bridge import to_inference_config, to_model_config

    tf_model_config = to_model_config(resolved.data, resolved.model)
    tf_inference_config = to_inference_config(
        tf_model_config,
        resolved.data,
        resolved.inference,
        overrides=dict(resolved.bridge),
    )

    deferred_notices: list[ResolutionNotice] = list(resolved.notices)
    runtime = resolve_runtime_execution_request(
        normalized_execution,
        mode="inference",
        execution_capabilities=execution_capabilities,
    )
    overrides_applied = dict(resolved.audit)
    if resolved.aliases:
        overrides_applied["input_aliases"] = {
            name: tuple(sources)
            for name, sources in resolved.aliases.items()
        }
    overrides_applied["execution_runtime"] = runtime.audit_dict()
    for name in ("accelerator", "num_workers", "inference_batch_size"):
        overrides_applied[name] = getattr(runtime.config, name)

    payload = InferencePayload(
        tf_inference_config=tf_inference_config,
        pt_data_config=resolved.data,
        pt_inference_config=resolved.inference,
        execution_config=runtime.config,
        overrides_applied=overrides_applied,
    )
    return payload, tuple(deferred_notices), runtime.notices


def resolve_training_payload(
    train_data_file: Path,
    output_dir: Path,
    overrides: Optional[Dict[str, Any]] = None,
    execution_config: Optional[ExecutionRequest] = None,
    profile: Optional[str] = None,
    *,
    training_baseline: TFTrainingConfig | PTTrainingConfig | None = None,
    execution_capabilities: ExecutionCapabilities | None = None,
) -> TrainingPayload:
    """Resolve Torch training owners without reading or writing ``params.cfg``."""
    payload, deferred_notices, runtime_notices = _resolve_training_payload(
        train_data_file=train_data_file,
        output_dir=output_dir,
        overrides=overrides,
        execution_config=execution_config,
        profile=profile,
        training_baseline=training_baseline,
        execution_capabilities=execution_capabilities,
    )
    _emit_resolution_notices(deferred_notices + runtime_notices)
    return payload


def resolve_inference_payload(
    model_path: Path,
    test_data_file: Path,
    output_dir: Path,
    overrides: Optional[Dict[str, Any]] = None,
    execution_config: Optional[ExecutionRequest] = None,
    *,
    execution_capabilities: ExecutionCapabilities | None = None,
) -> InferencePayload:
    """Resolve Torch inference owners without reading or writing ``params.cfg``."""
    payload, deferred_notices, runtime_notices = _resolve_inference_payload(
        model_path=model_path,
        test_data_file=test_data_file,
        output_dir=output_dir,
        overrides=overrides,
        execution_config=execution_config,
        execution_capabilities=execution_capabilities,
    )
    _emit_resolution_notices(deferred_notices + runtime_notices)
    return payload


def _project_legacy_config(
    tf_config: Union[TFTrainingConfig, TFInferenceConfig],
    deferred_notices: tuple[ResolutionNotice, ...],
    runtime_notices: tuple[ResolutionNotice, ...],
) -> None:
    """Commit one compatibility projection while preserving notice ordering."""
    params.unseal()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        populate_legacy_params(tf_config)
    params.seal()
    projection_notices = tuple(
        ResolutionNotice(item.category, str(item.message))
        for item in caught
    )
    _emit_resolution_notices(
        deferred_notices + projection_notices + runtime_notices
    )


@configured_legacy_params
def create_training_payload(
    train_data_file: Path,
    output_dir: Path,
    overrides: Optional[Dict[str, Any]] = None,
    execution_config: Optional[ExecutionRequest] = None,
    profile: Optional[str] = None,
    *,
    training_baseline: TFTrainingConfig | PTTrainingConfig | None = None,
    execution_capabilities: ExecutionCapabilities | None = None,
) -> TrainingPayload:
    """Resolve training owners and perform the CONFIG-001 legacy projection."""
    payload, deferred_notices, runtime_notices = _resolve_training_payload(
        train_data_file=train_data_file,
        output_dir=output_dir,
        overrides=overrides,
        execution_config=execution_config,
        profile=profile,
        training_baseline=training_baseline,
        execution_capabilities=execution_capabilities,
    )
    _project_legacy_config(
        payload.tf_training_config,
        deferred_notices,
        runtime_notices,
    )
    return payload


@configured_legacy_params
def create_training_payload_from_resolved_configs(
    data_config: PTDataConfig,
    model_config: PTModelConfig,
    training_config: PTTrainingConfig,
    inference_config: PTInferenceConfig,
    execution_config: PyTorchExecutionConfig,
    *,
    train_data_file: Path,
    output_dir: Path,
    n_groups: int | None,
    test_data_file: Path | None = None,
    parity_scale_mode: str = "off",
    parity_fixed_delta: float = 0.0,
    parity_init_scheme: str = "default",
) -> TrainingPayload:
    """Adapt already-resolved Torch records without resolving defaults again."""

    expected = (
        (data_config, PTDataConfig, "data_config"),
        (model_config, PTModelConfig, "model_config"),
        (training_config, PTTrainingConfig, "training_config"),
        (inference_config, PTInferenceConfig, "inference_config"),
        (execution_config, PyTorchExecutionConfig, "execution_config"),
    )
    for value, value_type, name in expected:
        if not isinstance(value, value_type):
            raise TypeError(f"{name} must be a {value_type.__name__}")
    if n_groups is not None and (
        isinstance(n_groups, bool)
        or not isinstance(n_groups, int)
        or n_groups <= 0
    ):
        raise ValueError("n_groups must be a positive integer")

    from ptycho_torch.config_bridge import to_model_config, to_training_config

    canonical_model = to_model_config(data_config, model_config)
    canonical_training = to_training_config(
        canonical_model,
        data_config,
        model_config,
        training_config,
        overrides={
            "train_data_file": Path(train_data_file),
            "test_data_file": (
                Path(test_data_file) if test_data_file is not None else None
            ),
            "output_dir": Path(output_dir),
            "n_groups": n_groups,
            "nphotons": data_config.nphotons,
        },
        require_group_count=False,
    )
    if n_groups is None:
        canonical_training = canonical_training.model_copy(
            update={
                "sampling": canonical_training.sampling.model_copy(
                    update={"n_groups": None}
                )
            }
        )
    payload = TrainingPayload(
        tf_training_config=canonical_training,
        pt_data_config=data_config,
        pt_model_config=model_config,
        pt_training_config=training_config,
        pt_inference_config=inference_config,
        model_spec=derive_model_spec(
            canonical_model,
            model_config,
            data_config,
            parity_scale_mode=parity_scale_mode,
            parity_fixed_delta=parity_fixed_delta,
            parity_init_scheme=parity_init_scheme,
        ),
        execution_config=execution_config,
        overrides_applied={"source": "resolved_torch_configs"},
    )
    _project_legacy_config(payload.tf_training_config, (), ())
    return payload


@configured_legacy_params
def create_inference_payload(
    model_path: Path,
    test_data_file: Path,
    output_dir: Path,
    overrides: Optional[Dict[str, Any]] = None,
    execution_config: Optional[ExecutionRequest] = None,
    *,
    execution_capabilities: ExecutionCapabilities | None = None,
) -> InferencePayload:
    """Resolve inference owners and perform the CONFIG-001 legacy projection."""
    payload, deferred_notices, runtime_notices = _resolve_inference_payload(
        model_path=model_path,
        test_data_file=test_data_file,
        output_dir=output_dir,
        overrides=overrides,
        execution_config=execution_config,
        execution_capabilities=execution_capabilities,
    )
    _project_legacy_config(
        payload.tf_inference_config,
        deferred_notices,
        runtime_notices,
    )
    return payload


def infer_probe_size(data_file: Path) -> int:
    """
    Extract probe size (N) from NPZ metadata.

    Factored out from ptycho_torch/train.py:96-140 for reusability across
    training and inference factories. Loads probeGuess array from NPZ dataset
    and extracts first dimension (assumes square probe).

    Args:
        data_file: Path to NPZ dataset file

    Returns:
        int: Probe size (N value), typically 64, 128, or 256

    Raises:
        FileNotFoundError: data_file does not exist
        KeyError: probeGuess key missing from NPZ
        ValueError: probeGuess shape invalid (non-square or wrong dimensions)

    Fallback Behavior:
        On any error (missing file, invalid NPZ, non-square probe), logs warning
        and returns fallback N=64. Design decision documented in
        .../open_questions.md §5 (hard error vs warning + fallback).

    Example:
        >>> from pathlib import Path
        >>> N = infer_probe_size(Path('datasets/train.npz'))
        >>> assert N in [64, 128, 256]  # Common probe sizes

    See also:
        - Original implementation: ptycho_torch/train.py:96-140
        - Override precedence: .../override_matrix.md row "N"
        - NPZ data contract: specs/data_contracts.md §1
    """
    observation = observe_probe_size(data_file)
    _emit_resolution_notices(observation.notices)
    return observation.value


@configured_legacy_params
def populate_legacy_params(
    tf_config: Union[TFTrainingConfig, TFInferenceConfig],
    force: bool = False,
) -> None:
    """
    Wrapper around update_legacy_dict with validation and logging.

    Ensures CONFIG-001 compliance checkpoint is explicit in factory workflows.
    Provides audit trail of params.cfg population for debugging and governance.

    This function is the critical compatibility bridge that enables legacy modules
    (over 20 files dependent on params.cfg) to consume modern structured configs.
    It MUST be called before any data loading or model construction operations.

    Args:
        tf_config: TrainingConfig or InferenceConfig (canonical TensorFlow format)
        force: If True, overwrites existing params.cfg values without warning.
            If False (default), logs warning if params.cfg already populated.

    Side Effects:
        - Updates ptycho.params.cfg dictionary via update_legacy_dict()
        - Logs params.cfg snapshot for audit trail (if logging enabled)

    Raises:
        ValueError: tf_config validation failed (missing required fields)
        TypeError: tf_config is not TrainingConfig or InferenceConfig instance

    Example:
        >>> from ptycho.config.config import TrainingConfig, ModelConfig
        >>> config = TrainingConfig(
        ...     model=ModelConfig(N=64, gridsize=2),
        ...     train_data_file=Path('data.npz'),
        ...     n_groups=512,
        ... )
        >>> populate_legacy_params(config)
        # params.cfg now contains: {'N': 64, 'gridsize': 2, 'n_groups': 512, ...}

    See also:
        - Bridge function: ptycho/config/config.py update_legacy_dict()
        - CONFIG-001: docs/findings.md CONFIG-001 (initialization order requirement)
        - Key mappings: ptycho/config/config.py KEY_MAPPINGS
    """
    from ptycho.config.config import update_legacy_dict
    import ptycho.params as params
    import warnings

    # Type validation
    if not isinstance(tf_config, (TFTrainingConfig, TFInferenceConfig)):
        raise TypeError(
            f"tf_config must be TrainingConfig or InferenceConfig instance, got {type(tf_config)}"
        )

    # Warn if params.cfg already populated (unless force=True)
    if not force and params.cfg:
        warnings.warn(
            "params.cfg already populated. Set force=True to overwrite existing values.",
            UserWarning
        )

    # Call the canonical bridge function (CONFIG-001 compliance)
    update_legacy_dict(params.cfg, tf_config)
