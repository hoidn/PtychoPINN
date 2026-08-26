"""
Configuration factory functions for PyTorch backend.

This module provides centralized factory functions that translate canonical TensorFlow
configurations plus PyTorch execution overrides into the objects consumed by the PyTorch
backend, eliminating duplicated config construction logic scattered across CLI and workflow
entry points.

Design documentation: docs/findings.md (see git history for the originating plan)

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

Core Functions:
    resolve_training_payload(): Pure training payload resolution
    resolve_inference_payload(): Pure inference payload resolution
    create_training_payload(): Compatibility resolution (no CONFIG-001 projection)
    create_inference_payload(): Compatibility resolution (no CONFIG-001 projection)
    infer_probe_size(): Extracts probe size from NPZ metadata

Design Principles:
    - Single Responsibility: Each factory handles one workflow (training vs inference)
    - Bridge Delegation: All TensorFlow dataclass translation delegated to config_bridge.py
    - No CONFIG-001 Projection: factories no longer populate legacy params.cfg
    - Override Transparency: Explicit override dict parameter for execution-specific knobs
    - Test-Driven: RED tests written before implementation

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

# Import PyTorchExecutionConfig (canonical location per docs/specs/spec-ptycho-config-bridge.md)
# Per supervisor decision at 2025-10-19T234458Z (factory_design.md §2.2)
from ptycho.config.config import PyTorchExecutionConfig

from ptycho_torch.scaling_contract import (
    CI_SCALE_CONTRACT,
    COUNT_INTENSITY,
    resolve_scale_contract,
)
from ptycho_torch.model_spec import (
    MODEL_TYPE_TO_MODE,
    ModelSpec,
    derive_model_spec,
)
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
    TRAINING_INPUT_RULES,
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

# docs/specs/spec-ptycho-conformance.md (D3) (Theme 3, docs/superpowers/plans/
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
    execution_config: PyTorchExecutionConfig  # Execution knobs
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
    execution_config: PyTorchExecutionConfig  # Execution knobs
    overrides_applied: Dict[str, Any] = field(default_factory=dict)  # Audit trail


# Import-time totality tripwire (W1): this surface has no target dataclass —
# its target is the resolver's declared canonical+alias vocabulary. The literal
# possible-key set below is enumerated adjacent to the function so drift is
# visible; every key the function can emit must be resolvable.
_TRAINING_RESOLVER_VOCABULARY = frozenset(
    name
    for rule in TRAINING_INPUT_RULES
    for name in (rule.canonical, *rule.aliases)
)

_TRAINING_FACTORY_OVERRIDE_KEYS = frozenset({
    "training_groups", "gridsize", "architecture", "mode", "amp_activation",
    "n_filters_scale", "object_layout", "training_canvas",
    "training_patch_weighting", "probe_big", "probe_mask",
    "probe_mask_sigma", "probe_mask_diameter", "pad_object", "probe_scale",
    "gaussian_smoothing_sigma", "nphotons", "neighbor_count", "max_epochs",
    "batch_size", "subsample_seed", "enable_oversampling",
    "neighbor_pool_size", "sequential_sampling", "test_data_file",
    "torch_loss_mode", "torch_mae_pred_l2_match_target",
    "intensity_scale_trainable", "gradient_clip_val",
    "gradient_clip_algorithm", "optimizer", "momentum", "weight_decay",
    "adam_beta1", "adam_beta2", "scheduler", "lr_warmup_epochs",
    "lr_min_ratio", "plateau_factor", "plateau_patience", "plateau_min_lr",
    "plateau_threshold", "log_grad_norm", "grad_norm_log_freq",
    "n_raw_frames_selected", "fno_modes", "fno_width", "fno_blocks",
    "fno_cnn_blocks", "fno_input_transform", "learned_input_channels",
    "max_hidden_channels", "resnet_width", "generator_output_mode",
})

assert _TRAINING_FACTORY_OVERRIDE_KEYS <= _TRAINING_RESOLVER_VOCABULARY, (
    "build_training_factory_overrides can emit keys outside the resolver "
    "vocabulary: "
    f"{sorted(_TRAINING_FACTORY_OVERRIDE_KEYS - _TRAINING_RESOLVER_VOCABULARY)}"
)


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
    mode = MODEL_TYPE_TO_MODE[model.model_type]
    overrides: Dict[str, Any] = {
        "training_groups": config.training_groups,
        "gridsize": model.gridsize,
        "architecture": model.architecture,
        "mode": mode,
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
        "nphotons": config.nphotons,
        "neighbor_count": config.neighbor_count,
        "max_epochs": config.nepochs,
        "batch_size": config.batch_size,
        "subsample_seed": config.subsample_seed,
        "enable_oversampling": config.enable_oversampling,
        "neighbor_pool_size": config.neighbor_pool_size,
        "sequential_sampling": config.sequential_sampling,
        "test_data_file": config.test_data_file,
        "torch_loss_mode": config.torch_loss_mode,
        "torch_mae_pred_l2_match_target": (
            config.torch_mae_pred_l2_match_target
        ),
        "intensity_scale_trainable": config.intensity_scale_trainable,
        "gradient_clip_val": config.gradient_clip_val,
        "gradient_clip_algorithm": config.gradient_clip_algorithm,
        "optimizer": config.optimizer,
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
        "adam_beta1": config.adam_beta1,
        "adam_beta2": config.adam_beta2,
        "scheduler": config.scheduler,
        "lr_warmup_epochs": config.lr_warmup_epochs,
        "lr_min_ratio": config.lr_min_ratio,
        "plateau_factor": config.plateau_factor,
        "plateau_patience": config.plateau_patience,
        "plateau_min_lr": config.plateau_min_lr,
        "plateau_threshold": config.plateau_threshold,
        "log_grad_norm": getattr(config, "log_grad_norm", False),
        "grad_norm_log_freq": getattr(config, "grad_norm_log_freq", 1),
    }
    if model.model_type == "supervised":
        overrides["torch_loss_mode"] = "mae"
    if config.train_raw_selection is not None:
        # Canonical TF config keeps the train_raw_selection spelling; the torch
        # resolver's honest key is n_raw_frames_selected (v3 field split).
        overrides["n_raw_frames_selected"] = config.train_raw_selection
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
    assert set(overrides) <= _TRAINING_FACTORY_OVERRIDE_KEYS, (
        "build_training_factory_overrides emitted a key outside "
        "_TRAINING_FACTORY_OVERRIDE_KEYS"
    )
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
    1. Validates required arguments (train_data_file, output_dir, training_groups)
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
            - training_groups: Number of grouped samples (no default, raises error if missing)
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
        ValueError: training_groups missing in overrides (required field)
        ValueError: Invalid field values (N <= 0, batch_size <= 0, etc.)

    Example:
        >>> from pathlib import Path
        >>> payload = create_training_payload(
        ...     train_data_file=Path('datasets/train.npz'),
        ...     output_dir=Path('outputs/exp001'),
        ...     overrides={
        ...         'training_groups': 512,
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
        >>> assert payload.tf_training_config.training_groups == 512

    See also:
        - Design: docs/findings.md (see git history for the originating plan) §3.1
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
    1. Validates required arguments (model_path, test_data_file, output_dir, inference_groups)
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
            - inference_groups: Number of grouped samples (no default, raises error if missing)
            Optional keys: gridsize, batch_size, middle_trim, pad_eval, etc.
            The legacy spelling ``training_groups`` is permanently accepted as
            a fenced alias for ``inference_groups`` (see docs/specs/spec-ptycho-config-bridge.md §3).
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
        ValueError: inference_groups missing in overrides (required field)

    Example:
        >>> payload = create_inference_payload(
        ...     model_path=Path('outputs/exp001'),
        ...     test_data_file=Path('datasets/test.npz'),
        ...     output_dir=Path('outputs/exp001/inference'),
        ...     overrides={
        ...         'inference_groups': 128,
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
    """Resolve training owners without the CONFIG-001 legacy projection."""
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


def create_training_payload_from_resolved_configs(
    data_config: PTDataConfig,
    model_config: PTModelConfig,
    training_config: PTTrainingConfig,
    inference_config: PTInferenceConfig,
    execution_config: PyTorchExecutionConfig,
    *,
    train_data_file: Path,
    output_dir: Path,
    training_groups: int | None,
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
    if training_groups is not None and (
        isinstance(training_groups, bool)
        or not isinstance(training_groups, int)
        or training_groups <= 0
    ):
        raise ValueError("training_groups must be a positive integer")

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
            "training_groups": training_groups,
            "nphotons": data_config.nphotons,
        },
        require_group_count=False,
    )
    if training_groups is None:
        canonical_training.training_groups = None
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
    return payload


def create_inference_payload(
    model_path: Path,
    test_data_file: Path,
    output_dir: Path,
    overrides: Optional[Dict[str, Any]] = None,
    execution_config: Optional[ExecutionRequest] = None,
    *,
    execution_capabilities: ExecutionCapabilities | None = None,
) -> InferencePayload:
    """Resolve inference owners without the CONFIG-001 legacy projection."""
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


