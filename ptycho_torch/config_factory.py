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
    [Factory Entry Point]
      ↓
    [Validate + Infer + Apply Overrides]
      ↓
    [Translate to TensorFlow Canonical Configs via config_bridge]
      ↓
    [Populate params.cfg (CONFIG-001 checkpoint)]
      ↓
    [Return Payload (TF config + PyTorch configs + execution config)]

Core Functions:
    create_training_payload(): Constructs complete training configuration bundle
    create_inference_payload(): Constructs complete inference configuration bundle
    infer_probe_size(): Extracts probe size from NPZ metadata
    populate_legacy_params(): Wrapper around update_legacy_dict with validation

Design Principles:
    - Single Responsibility: Each factory handles one workflow (training vs inference)
    - Bridge Delegation: All TensorFlow dataclass translation delegated to config_bridge.py
    - CONFIG-001 Compliance: Factories ensure update_legacy_dict() called before data loading
    - Override Transparency: Explicit override dict parameter for execution-specific knobs
    - Test-Driven: RED tests written before implementation (Phase B2.b)

Override Precedence (highest to lowest):
    1. Explicit overrides dict (user-provided via factory call)
    2. Execution config fields (PyTorchExecutionConfig instance)
    3. CLI argument defaults (from argparse)
    4. PyTorch config defaults (DataConfig, ModelConfig, TrainingConfig)
    5. TensorFlow config defaults (TrainingConfig, ModelConfig, InferenceConfig)
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Import canonical TensorFlow configs (single source of truth)
from ptycho.config.config import (
    ModelConfig as TFModelConfig,
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
    validate_amplitude_physics_gain,
    validate_contract_coherence,
)
from ptycho_torch.model_spec import ModelSpec, derive_model_spec

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
# override: rect_s1s2_init='data' per docs/model_baselines.md (one-batch s1/s2
# calibration; required for bounded-head CNN CI arms), rect_s1s2_trainable=True
# (trainable s1/s2 own the training scale), amplitude_physics_gain=1.0 (the
# rectangular contract rejects anything else fail-closed), and
# cnn_output_mode='real_imag' (cnn architecture only; other generators already
# default generator_output_mode='real_imag').
CI_PROFILE_BUNDLE: Dict[str, Any] = {
    **CI_PROFILE_CONTRACT_FIELDS,
    "amplitude_physics_gain": 1.0,
    "rect_s1s2_trainable": True,
    "rect_s1s2_init": "data",
    "cnn_output_mode": "real_imag",
}


# These graph-topology fields historically leaked into the normatively
# runtime-only PyTorchExecutionConfig. They remain accepted there only as
# deprecated input aliases at this factory boundary. All downstream model
# construction reads the resolved PTModelConfig.
DEPRECATED_EXECUTION_MODEL_ALIASES = (
    "spectral_bottleneck_blocks",
    "spectral_bottleneck_modes",
    "spectral_bottleneck_share_weights",
    "spectral_bottleneck_gate_init",
    "spectral_bottleneck_gate_mode",
)

_TRAINING_OVERRIDE_ALIASES = frozenset(
    {"gridsize", "max_epochs", "neighbor_count", "model_type"}
)
_TRAINING_CONFIG_TYPES = (
    PTDataConfig,
    PTModelConfig,
    PTTrainingConfig,
    PTInferenceConfig,
)
_TRAINING_OVERRIDE_FIELDS = frozenset(
    field_info.name
    for config_type in _TRAINING_CONFIG_TYPES
    for field_info in fields(config_type)
) | _TRAINING_OVERRIDE_ALIASES


def _config_kwargs(config_type, values: Dict[str, Any]) -> Dict[str, Any]:
    """Select the declared constructor fields for one configuration owner."""
    owned = {field_info.name for field_info in fields(config_type)}
    return {name: value for name, value in values.items() if name in owned}


def _reject_unknown_training_overrides(overrides: Dict[str, Any]) -> None:
    unknown = sorted(set(overrides) - _TRAINING_OVERRIDE_FIELDS)
    if unknown:
        raise ValueError(
            "unknown training override field(s): " + ", ".join(unknown)
        )


def _merge_deprecated_execution_model_aliases(
    overrides: Dict[str, Any],
    execution_config: Optional[PyTorchExecutionConfig],
) -> tuple[str, ...]:
    """Resolve explicitly supplied execution aliases into structural inputs."""
    if execution_config is None:
        return ()

    explicit_aliases = getattr(
        execution_config, "_explicit_structural_aliases", frozenset()
    )
    used = []
    for name in DEPRECATED_EXECUTION_MODEL_ALIASES:
        if name not in explicit_aliases:
            continue
        alias_value = getattr(execution_config, name)
        if name in overrides and overrides[name] != alias_value:
            raise ValueError(
                f"structural field {name!r} conflicts between ModelConfig input "
                f"({overrides[name]!r}) and deprecated PyTorchExecutionConfig "
                f"alias ({alias_value!r})"
            )
        overrides[name] = alias_value
        used.append(name)

    if used:
        import warnings

        warnings.warn(
            "PyTorchExecutionConfig topology fields are deprecated aliases; "
            "pass them through the structural ModelConfig input instead: "
            + ", ".join(used),
            DeprecationWarning,
            stacklevel=3,
        )
    return tuple(used)


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


def _reject_half_configured_ci(
    overrides: Dict[str, Any],
    physics_forward_mode: str,
) -> None:
    """Fail-closed guard against half-configured CI intent (Theme 3.3/3.4).

    Fires only on explicit, CI-flavored, NON-DEFAULT evidence of intent in the
    overrides dict: ``rect_s1s2_init='data'`` (a CI-only calibration knob,
    silently ignored by the amplitude forward) without
    ``physics_forward_mode='rectangular_scaled'``.

    The contract pair (``scale_contract_version``/``measurement_domain``) and
    ``rect_s1s2_trainable`` deliberately do NOT trigger this guard: their
    CI-flavored values equal the dataclass defaults and workflow glue (e.g.
    the grid-lines runner's ``lightning_overrides``) forwards them
    unconditionally for legacy amplitude arms, so their presence in overrides
    proves nothing about user intent. Bare-default construction remains valid
    legacy behavior per the 2026-07-09 CI ablation design.
    """
    if physics_forward_mode == "rectangular_scaled":
        return
    if overrides.get("rect_s1s2_init") == "data":
        raise ValueError(
            "rect_s1s2_init='data' is a CI-contract knob (one-batch s1/s2 "
            "calibration for the rectangular_scaled forward) but "
            f"physics_forward_mode resolved to {physics_forward_mode!r}, "
            "where it is silently ignored. Half-configured CI is fail-closed: "
            "use create_training_payload(..., profile='ci') (or --profile ci "
            "on the training CLI) for the coherent PtychoPINN-CI bundle, or "
            "drop rect_s1s2_init."
        )


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


@configured_legacy_params
def create_training_payload(
    train_data_file: Path,
    output_dir: Path,
    overrides: Optional[Dict[str, Any]] = None,
    execution_config: Optional[PyTorchExecutionConfig] = None,
    profile: Optional[str] = None,
) -> TrainingPayload:
    """
    Create complete training configuration payload.

    Centralizes all config construction logic for PyTorch training workflows.
    Eliminates duplicated wiring in CLI and workflow entry points by providing
    a single factory function that:
    1. Validates required arguments (train_data_file, output_dir, n_groups)
    2. Infers probe size from NPZ metadata (or uses override)
    3. Constructs PyTorch singleton configs (DataConfig, ModelConfig, TrainingConfig, InferenceConfig)
    4. Applies CLI overrides with precedence rules
    5. Translates to TensorFlow canonical configs via config_bridge
    6. Populates params.cfg (CONFIG-001 compliance checkpoint)
    7. Constructs PyTorchExecutionConfig for runtime knobs
    8. Returns TrainingPayload with all config objects + audit trail

    Args:
        train_data_file: Path to training NPZ dataset (must exist per DATA-001)
        output_dir: Path to output directory for checkpoints/logs (created if missing)
        overrides: Dict of field overrides (highest precedence). Required keys:
            - n_groups: Number of grouped samples (no default, raises error if missing)
            Optional keys: batch_size, gridsize, max_epochs, nphotons, etc.
        execution_config: PyTorchExecutionConfig instance for runtime knobs (accelerator,
            deterministic, num_workers, etc.). If None, uses defaults.
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
        ...     execution_config=PyTorchExecutionConfig(
        ...         accelerator='cpu',
        ...         enable_progress_bar=True,
        ...     ),
        ... )
        >>> assert isinstance(payload.tf_training_config, TrainingConfig)
        >>> assert payload.tf_training_config.n_groups == 512

    See also:
        - Design: plans/active/ADR-003-BACKEND-API/reports/.../factory_design.md §3.1
        - Override precedence: .../override_matrix.md §6
        - Integration: .../factory_design.md §3 (CLI/workflow call sites)
    """
    # Step 0: Resolve named profile (Conformance D3). Must precede everything
    # else so profile contradictions fail before any state is touched.
    if profile is not None:
        if profile != "ci":
            raise ValueError(
                f"Unknown configuration profile {profile!r}; supported "
                "profiles: 'ci'."
            )
        overrides = resolve_ci_profile(overrides)

    params.unseal()
    from ptycho_torch.config_bridge import to_model_config, to_training_config

    # Defensive copy of overrides
    overrides = dict(overrides or {})
    resolved_profile = resolve_profile_overrides(overrides)
    if resolved_profile is not None:
        overrides["scale_contract_version"], overrides["measurement_domain"] = resolved_profile
    overrides_applied = dict(overrides)  # Audit trail
    if profile is not None:
        overrides_applied['profile'] = profile

    # Bridge naming compatibility: accept legacy/CLI-friendly keys
    if 'max_epochs' in overrides and 'epochs' not in overrides:
        overrides['epochs'] = overrides['max_epochs']
        overrides_applied['epochs'] = overrides['max_epochs']
    if 'neighbor_count' in overrides and 'K' not in overrides:
        overrides['K'] = overrides['neighbor_count']
        overrides_applied['K'] = overrides['neighbor_count']
    if 'model_type' in overrides and 'mode' not in overrides:
        overrides['mode'] = overrides['model_type']
        overrides_applied['mode'] = overrides['model_type']

    alias_fields = _merge_deprecated_execution_model_aliases(
        overrides, execution_config
    )
    for name in alias_fields:
        overrides_applied[name] = overrides[name]
    _reject_unknown_training_overrides(overrides)

    # Step 1: Validate required arguments
    if not train_data_file.exists():
        raise FileNotFoundError(f"Training data file not found: {train_data_file}")

    if 'n_groups' not in overrides:
        raise ValueError(
            "n_groups is required in overrides (no default). "
            "Provide as: overrides={'n_groups': 512, ...}"
        )

    # Step 2: Infer probe size from NPZ (or use override)
    if 'N' in overrides:
        N = overrides['N']
    else:
        N = infer_probe_size(train_data_file)
        overrides['N'] = N
        overrides_applied['N'] = N  # Record inferred value

    # Step 2b: Resolve nphotons (override > metadata > TF default)
    if 'nphotons' not in overrides:
        nphotons_from_metadata = _load_nphotons_from_metadata(train_data_file)
        if nphotons_from_metadata is not None:
            overrides['nphotons'] = nphotons_from_metadata
            overrides_applied['nphotons'] = nphotons_from_metadata
            overrides_applied['nphotons_source'] = 'metadata'
        else:
            tf_default_nphotons = TFTrainingConfig(model=TFModelConfig()).nphotons
            overrides['nphotons'] = tf_default_nphotons
            overrides_applied['nphotons'] = tf_default_nphotons
            overrides_applied['nphotons_source'] = 'tf_default'

    # Step 3: Build PyTorch singleton configs with defaults + overrides
    # DataConfig: Extract data-related fields from overrides
    grid_size = overrides.get('grid_size', (overrides.get('gridsize', 1), overrides.get('gridsize', 1)))
    # Compute C from grid_size (number of channels = gridsize_x * gridsize_y)
    C = grid_size[0] * grid_size[1]
    # Ensure downstream config updates see derived grid_size and channels
    overrides['grid_size'] = grid_size
    overrides['C'] = C

    pt_data_config = PTDataConfig(**_config_kwargs(PTDataConfig, overrides))

    # ModelConfig: Extract model architecture fields from overrides
    # CRITICAL: Synchronize C_forward and C_model with pt_data_config.C to ensure
    # PyTorch helpers (reassemble_patches_position_real) receive tensor shapes
    # consistent with the grouping strategy. Fixes ADR-003 C4.D3 coords_relative mismatch.
    overrides['C_forward'] = C
    overrides['C_model'] = C

    pt_model_config = PTModelConfig(**_config_kwargs(PTModelConfig, overrides))

    # PROBE-RANK-001 §3.3: the explicit amplitude physics gain is a
    # provenance-carrying training constant — validate fail-fast at payload
    # creation and always record the effective value in the audit trail
    # (it also reaches Lightning hparams via ModelConfig serialization).
    validate_amplitude_physics_gain(pt_model_config)
    overrides_applied['amplitude_physics_gain'] = pt_model_config.amplitude_physics_gain

    # TrainingConfig: Extract training-specific fields from overrides
    overrides['nll'] = overrides.get('nll_weight', 1.0) > 0
    overrides['train_data_file'] = str(train_data_file)
    overrides['test_data_file'] = str(overrides['test_data_file']) if 'test_data_file' in overrides else None
    overrides['output_dir'] = str(output_dir)

    pt_training_config = PTTrainingConfig(
        **_config_kwargs(PTTrainingConfig, overrides)
    )

    # Resolve the selected objective into structural loss identity before the
    # ModelSpec is sealed. Lightning must not silently mutate this field after
    # construction has begun.
    from dataclasses import replace

    resolved_loss = (
        "Poisson" if pt_training_config.torch_loss_mode == "poisson" else "MAE"
    )
    if pt_model_config.loss_function != resolved_loss:
        pt_model_config = replace(pt_model_config, loss_function=resolved_loss)
    from ptycho_torch.object_compatibility import (
        resolve_torch_model_object_policy,
    )
    pt_model_config = resolve_torch_model_object_policy(pt_model_config)

    # Conformance D3: fail-closed contract coherence (Theme 3, 2026-07-14).
    # The half-configured-CI guard runs here (not on bare dataclasses) because
    # only the factory sees the explicit overrides dict; the unconditional
    # coherence validation is a no-op pass for coherent legacy AND coherent CI.
    _reject_half_configured_ci(overrides, pt_model_config.physics_forward_mode)
    validate_contract_coherence(pt_data_config, pt_model_config, pt_training_config)

    # InferenceConfig: track patch-stats flags for instrumentation
    pt_inference_config = PTInferenceConfig(
        patch_weighting=overrides.get('patch_weighting', 'probe'),
        varpro_scaling=overrides.get('varpro_scaling', True),
        log_patch_stats=overrides.get('log_patch_stats', False),
        patch_stats_limit=overrides.get('patch_stats_limit'),
    )


    # Step 4: Translate to TensorFlow canonical configs via config_bridge
    tf_model_config = to_model_config(pt_data_config, pt_model_config)

    # Build overrides dict for config_bridge (includes required fields)
    bridge_overrides = {
        'train_data_file': train_data_file,
        'output_dir': output_dir,
        'n_groups': overrides['n_groups'],  # Required field (validated above)
    }

    # Handle nphotons: config_bridge requires explicit override if using PyTorch default
    # to avoid silent divergence. Always include nphotons in bridge_overrides.
    if 'nphotons' in overrides:
        bridge_overrides['nphotons'] = overrides['nphotons']
    else:
        # Use PyTorch default and mark as explicit override to pass config_bridge validation
        bridge_overrides['nphotons'] = pt_data_config.nphotons
    # Add optional fields from overrides
    if 'test_data_file' in overrides:
        bridge_overrides['test_data_file'] = overrides['test_data_file']
    if 'n_subsample' in overrides:
        bridge_overrides['n_subsample'] = overrides['n_subsample']
    if 'subsample_seed' in overrides:
        bridge_overrides['subsample_seed'] = overrides['subsample_seed']

    tf_training_config = to_training_config(
        tf_model_config,
        pt_data_config,
        pt_model_config,
        pt_training_config,
        overrides=bridge_overrides
    )

    # Step 5: Populate params.cfg (CONFIG-001 compliance checkpoint)
    populate_legacy_params(tf_training_config)
    params.seal()

    # Step 6: Construct execution config (Phase C2.B1+C2.B2)
    # If execution_config not provided, instantiate default PyTorchExecutionConfig
    if execution_config is None:
        execution_config = PyTorchExecutionConfig()

    # Merge execution knobs into overrides_applied audit trail (Phase C2.B2)
    # Record applied execution knobs for transparency
    overrides_applied['accelerator'] = execution_config.accelerator
    overrides_applied['deterministic'] = execution_config.deterministic
    overrides_applied['num_workers'] = execution_config.num_workers
    overrides_applied['enable_progress_bar'] = execution_config.enable_progress_bar
    overrides_applied['learning_rate'] = execution_config.learning_rate
    # Optimization knobs (Phase EB2.B1 - ADR-003)
    overrides_applied['scheduler'] = execution_config.scheduler
    overrides_applied['accum_steps'] = execution_config.accum_steps
    # Logger backend (Phase EB3.B - ADR-003)
    overrides_applied['logger_backend'] = execution_config.logger_backend

    # Step 7: Return TrainingPayload with all config objects + audit trail
    model_spec = derive_model_spec(
        tf_model_config,
        pt_model_config,
        pt_data_config,
    )

    return TrainingPayload(
        tf_training_config=tf_training_config,
        pt_data_config=pt_data_config,
        pt_model_config=pt_model_config,
        pt_training_config=pt_training_config,
        pt_inference_config=pt_inference_config,
        model_spec=model_spec,
        execution_config=execution_config,  # Now always PyTorchExecutionConfig instance
        overrides_applied=overrides_applied,
    )


@configured_legacy_params
def create_inference_payload(
    model_path: Path,
    test_data_file: Path,
    output_dir: Path,
    overrides: Optional[Dict[str, Any]] = None,
    execution_config: Optional[PyTorchExecutionConfig] = None,
) -> InferencePayload:
    """
    Create complete inference configuration payload.

    Centralizes all config construction logic for PyTorch inference workflows.
    Eliminates duplicated wiring in CLI and workflow entry points by providing
    a single factory function that:
    1. Validates required arguments (model_path, test_data_file, output_dir, n_groups)
    2. Loads checkpoint config from model_path (or infers from NPZ)
    3. Constructs PyTorch singleton configs (DataConfig, InferenceConfig)
    4. Applies CLI overrides with precedence rules
    5. Translates to TensorFlow canonical configs via config_bridge
    6. Populates params.cfg (CONFIG-001 compliance checkpoint)
    7. Constructs PyTorchExecutionConfig for runtime knobs
    8. Returns InferencePayload with all config objects + audit trail

    Args:
        model_path: Path to trained model directory (must contain wts.h5.zip)
        test_data_file: Path to test NPZ dataset (must exist per DATA-001)
        output_dir: Path to output directory for reconstructions (created if missing)
        overrides: Dict of field overrides (highest precedence). Required keys:
            - n_groups: Number of grouped samples (no default, raises error if missing)
            Optional keys: gridsize, batch_size, middle_trim, pad_eval, etc.
        execution_config: PyTorchExecutionConfig instance for runtime knobs (accelerator,
            inference_batch_size, etc.). If None, uses defaults.

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
        ...     execution_config=PyTorchExecutionConfig(
        ...         inference_batch_size=64,
        ...     ),
        ... )

    See also:
        - Design: .../factory_design.md §3.3
        - Checkpoint loading: specs/ptychodus_api_spec.md §4.6
    """
    params.unseal()
    from ptycho_torch.config_bridge import to_model_config, to_inference_config

    # Defensive copy of overrides
    overrides = dict(overrides or {})
    resolved_profile = resolve_profile_overrides(overrides)
    overrides_applied = dict(overrides)  # Audit trail

    # Step 1: Validate required arguments
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    # Check for checkpoint file
    checkpoint_file = model_path / "wts.h5.zip"
    if not checkpoint_file.exists():
        raise ValueError(
            f"Model archive not found: {checkpoint_file}. "
            "Expected wts.h5.zip in model_path directory."
        )

    if not test_data_file.exists():
        raise FileNotFoundError(f"Test data file not found: {test_data_file}")

    if 'n_groups' not in overrides:
        raise ValueError(
            "n_groups is required in overrides (no default). "
            "Provide as: overrides={'n_groups': 128, ...}"
        )

    # Step 2: Infer probe size from NPZ (or use override)
    if 'N' in overrides:
        N = overrides['N']
    else:
        N = infer_probe_size(test_data_file)
        overrides_applied['N'] = N  # Record inferred value

    # Step 3: Build PyTorch singleton configs with defaults + overrides
    # DataConfig: Extract data-related fields from overrides
    grid_size = overrides.get('grid_size', (overrides.get('gridsize', 1), overrides.get('gridsize', 1)))
    # Compute C from grid_size (number of channels = gridsize_x * gridsize_y)
    C = grid_size[0] * grid_size[1]

    pt_data_config = PTDataConfig(
        N=N,
        grid_size=grid_size,
        C=C,  # Set C based on grid_size
        K=overrides.get('neighbor_count', 4),  # Canonical default=4 per specs/ptychodus_api_spec.md §4.6
        probe_scale=overrides.get('probe_scale', 4.0),  # Align with TF defaults
        subsample_seed=overrides.get('subsample_seed'),  # Optional field
        scale_contract_version=(
            resolved_profile[0] if resolved_profile is not None else CI_SCALE_CONTRACT
        ),
        measurement_domain=(
            resolved_profile[1] if resolved_profile is not None else COUNT_INTENSITY
        ),
    )

    # ModelConfig: Extract model architecture fields from overrides (for config_bridge)
    # Note: In inference, model config typically loaded from checkpoint, but we need
    # a ModelConfig instance for config_bridge translation
    # CRITICAL: Synchronize C_forward and C_model with pt_data_config.C (ADR-003 C4.D3)
    pt_model_config = PTModelConfig(
        mode=overrides.get('model_type', 'Unsupervised'),  # Map TF → PT naming
        amp_activation=overrides.get('amp_activation', 'silu'),
        n_filters_scale=overrides.get('n_filters_scale', 2),  # PyTorch default
        object_big=overrides.get('object_big'),
        object_layout=overrides.get('object_layout'),
        training_canvas=overrides.get('training_canvas'),
        training_patch_weighting=overrides.get('training_patch_weighting'),
        # Full decoder support is the normal object-big contract. Callers may
        # still request False explicitly for historical zero-border artifacts.
        probe_big=overrides.get('probe_big', True),
        probe_mask=overrides.get('probe_mask', False),
        probe_mask_tensor=overrides.get('probe_mask_tensor'),
        probe_mask_sigma=overrides.get('probe_mask_sigma', 1.0),
        probe_mask_diameter=overrides.get('probe_mask_diameter'),
        C_forward=C,  # Match data config channel count
        C_model=C,    # Match data config channel count
        pad_object=overrides.get('pad_object', True),  # Spec default
        gaussian_smoothing_sigma=overrides.get('gaussian_smoothing_sigma', 0.0),  # Spec default
    )
    from ptycho_torch.object_compatibility import (
        resolve_torch_model_object_policy,
    )
    pt_model_config = resolve_torch_model_object_policy(pt_model_config)

    # InferenceConfig: Extract inference-specific fields from overrides
    pt_inference_config = PTInferenceConfig(
        batch_size=overrides.get('batch_size', 16),  # PyTorch default
        patch_weighting=overrides.get('patch_weighting', 'probe'),
        varpro_scaling=overrides.get('varpro_scaling', True),
        log_patch_stats=overrides.get('log_patch_stats', False),
        patch_stats_limit=overrides.get('patch_stats_limit'),
    )

    # Step 4: Translate to TensorFlow canonical configs via config_bridge
    tf_model_config = to_model_config(pt_data_config, pt_model_config)

    # Build overrides dict for config_bridge (includes required fields)
    bridge_overrides = {
        'model_path': model_path,
        'test_data_file': test_data_file,
        'output_dir': output_dir,
        'n_groups': overrides['n_groups'],  # Required field (validated above)
    }
    # Add optional fields from overrides
    if 'n_subsample' in overrides:
        bridge_overrides['n_subsample'] = overrides['n_subsample']
    if 'subsample_seed' in overrides:
        bridge_overrides['subsample_seed'] = overrides['subsample_seed']

    tf_inference_config = to_inference_config(
        tf_model_config,
        pt_data_config,
        pt_inference_config,
        overrides=bridge_overrides
    )

    # Step 5: Populate params.cfg (CONFIG-001 compliance checkpoint)
    populate_legacy_params(tf_inference_config)
    params.seal()

    # Step 6: Construct execution config (Phase C2.B1+C2.B2)
    # If execution_config not provided, instantiate default PyTorchExecutionConfig
    if execution_config is None:
        execution_config = PyTorchExecutionConfig()

    # Merge execution knobs into overrides_applied audit trail (Phase C2.B2)
    # Record applied execution knobs for transparency
    overrides_applied['accelerator'] = execution_config.accelerator
    overrides_applied['num_workers'] = execution_config.num_workers
    overrides_applied['inference_batch_size'] = execution_config.inference_batch_size

    # Step 7: Return InferencePayload with all config objects + audit trail
    return InferencePayload(
        tf_inference_config=tf_inference_config,
        pt_data_config=pt_data_config,
        pt_inference_config=pt_inference_config,
        execution_config=execution_config,  # Now always PyTorchExecutionConfig instance
        overrides_applied=overrides_applied,
    )


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
    import numpy as np
    import warnings

    fallback_N = 64

    try:
        # Load NPZ with allow_pickle=False for security
        with np.load(data_file, allow_pickle=False) as npz_data:
            if 'probeGuess' not in npz_data:
                warnings.warn(
                    f"probeGuess key missing from {data_file}. Using fallback N={fallback_N}.",
                    UserWarning
                )
                return fallback_N

            probe = npz_data['probeGuess']

            # Extract first dimension (assumes square probe)
            if probe.ndim < 2:
                warnings.warn(
                    f"probeGuess has invalid shape {probe.shape} (expected 2D square array). "
                    f"Using fallback N={fallback_N}.",
                    UserWarning
                )
                return fallback_N

            N = probe.shape[0]

            # Validate square probe
            if probe.shape[0] != probe.shape[1]:
                warnings.warn(
                    f"probeGuess is non-square {probe.shape}. Using first dimension N={N}.",
                    UserWarning
                )

            return N

    except FileNotFoundError:
        warnings.warn(
            f"Data file {data_file} not found. Using fallback N={fallback_N}.",
            UserWarning
        )
        return fallback_N
    except Exception as e:
        warnings.warn(
            f"Error reading probeGuess from {data_file}: {e}. Using fallback N={fallback_N}.",
            UserWarning
        )
        return fallback_N


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
    (over 20 files dependent on params.cfg) to consume modern dataclass configs.
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
