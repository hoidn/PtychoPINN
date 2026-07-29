"""
Modern dataclass-based configuration system for PtychoPINN.

This module defines the type-safe, structured configuration architecture that replaces
the legacy params.cfg dictionary pattern. It serves as the single source of truth for
all configuration while maintaining backward compatibility with 20+ legacy modules
through a one-way data flow translation system.

Architecture & Data Flow:
    Modern dataclass → update_legacy_dict() → Legacy params.cfg dictionary
    
    The data flow is strictly one-way: configuration originates in structured dataclasses
    and flows to the legacy dictionary via update_legacy_dict(). This function serves
    as the critical compatibility bridge, using KEY_MAPPINGS to translate between
    modern field names (object_big) and legacy parameter names (object.big).

Configuration Classes:
    ModelConfig: Core architecture (N, gridsize, model_type, activations, etc.)
    TrainingConfig: Training workflow (epochs, loss weights, data paths, sampling)
    InferenceConfig: Inference workflow (model paths, output settings, debug flags)

Core Functions:
    update_legacy_dict(cfg, dataclass_obj): THE compatibility bridge function
        - Translates dataclass fields to legacy parameter names via KEY_MAPPINGS
        - Updates params.cfg dictionary for consumption by legacy modules
        - Handles Path object conversion and nested model configurations
    
    validate_*_config(config): Validates configuration constraints and dependencies
    load_yaml_config(path): Loads YAML files for script-based configuration
    dataclass_to_legacy_dict(obj): Internal translation with KEY_MAPPINGS application

Critical Dependencies:
    KEY_MAPPINGS dictionary: Defines field name translations (e.g., object_big → object.big)
    - Required for legacy module compatibility
    - Handles nested configurations and Path object serialization
    - Must be maintained when adding new configuration fields

Workflow Integration:
    ```python
    # 1. Modern configuration creation
    config = TrainingConfig(
        model=ModelConfig(N=128, model_type='pinn'),
        train_data_file='data.npz', nepochs=100)
    
    # 2. Enable legacy module compatibility (CRITICAL STEP)
    import ptycho.params as params  
    update_legacy_dict(params.cfg, config)  # One-way data flow
    
    # 3. YAML-based configuration for scripts
    yaml_data = load_yaml_config(Path('config.yaml'))
    config = TrainingConfig(**yaml_data)
    update_legacy_dict(params.cfg, config)  # Always required for legacy compatibility
    ```

Migration Pattern:
    - New code: Uses dataclasses directly (TrainingConfig, ModelConfig, etc.)
    - Legacy modules: Continue using params.get('key') unchanged
    - Compatibility: Maintained via update_legacy_dict() calling dataclass_to_legacy_dict()
    - Translation: KEY_MAPPINGS handles all field name conversions automatically

State Dependencies: 
    Consumers rely on params.cfg being updated via update_legacy_dict() before
    legacy module initialization. Over 23 modules depend on this translation.
"""

from collections.abc import Mapping
from dataclasses import dataclass, asdict, field, fields, replace
from pathlib import Path
from typing import Annotated, Dict, Any, List, Optional, Literal, Union
import json
import hashlib
import inspect
import math
import tomllib
from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    StrictBool,
    StrictFloat,
    StrictInt,
    TypeAdapter,
    ValidationError,
    with_config,
)
import yaml
import warnings

# Export list for public API (ADR-003 Phase C3.A1)
# Restores exports removed during Phase C2; ensures PyTorchExecutionConfig is discoverable
__all__ = [
    # Dataclass configurations
    'ModelConfig',
    'TrainingConfig',
    'InferenceConfig',
    'PyTorchExecutionConfig',
    'ProbeSimulationConfig',
    'SyntheticObjectConfig',
    'ScanSimulationConfig',
    'DetectorSimulationConfig',
    'SimulationConfig',
    'resolve_model_object_policy',
    # Core compatibility bridge
    'update_legacy_dict',
    # Validation functions
    'validate_model_config',
    'validate_training_config',
    'validate_inference_config',
    'validate_simulation_config',
    'validate_simulation_compatibility',
    'simulation_config_from_mapping',
    'simulation_config_to_dict',
    'simulation_config_sha256',
    'load_simulation_config',
    # YAML loading
    'load_yaml_config',
    # Internal translation (exposed for advanced use)
    'dataclass_to_legacy_dict',
]


def _require_exact_int(value: Any) -> Any:
    if type(value) is not int:
        raise ValueError("must be an exact built-in integer")
    return value


def _require_exact_optional_int(value: Any) -> Any:
    if value is not None and type(value) is not int:
        raise ValueError("must be an exact built-in integer or None")
    return value


def _require_exact_bool(value: Any) -> Any:
    if type(value) is not bool:
        raise ValueError("must be an exact built-in boolean")
    return value


def _require_exact_finite_number(value: Any) -> Any:
    if type(value) not in {int, float}:
        raise ValueError("must be an exact built-in integer or float")
    return value


def _require_exact_str(value: Any) -> Any:
    if type(value) is not str:
        raise ValueError("must be an exact built-in string")
    return value


def _require_pair_container(value: Any) -> Any:
    if type(value) not in {list, tuple}:
        raise ValueError("must be a list or tuple")
    return value


def _require_positive_int(value: int) -> int:
    if value <= 0:
        raise ValueError("must be positive")
    return value


def _require_non_negative_int(value: int) -> int:
    if value < 0:
        raise ValueError("must be non-negative")
    return value


def _require_finite_positive_number(value: int | float) -> int | float:
    if value <= 0 or (type(value) is float and not math.isfinite(value)):
        raise ValueError("must be finite and positive")
    return value


_StrictPositiveInt = Annotated[
    StrictInt,
    BeforeValidator(_require_exact_int),
    AfterValidator(_require_positive_int),
]
_StrictNonNegativeInt = Annotated[
    StrictInt,
    BeforeValidator(_require_exact_int),
    AfterValidator(_require_non_negative_int),
]
_StrictOptionalInt = Annotated[
    StrictInt | None,
    BeforeValidator(_require_exact_optional_int),
]
_StrictBool = Annotated[
    StrictBool,
    BeforeValidator(_require_exact_bool),
]
_StrictFinitePositiveNumber = Annotated[
    StrictInt | StrictFloat,
    BeforeValidator(_require_exact_finite_number),
    AfterValidator(_require_finite_positive_number),
]
_StrictPositivePair = Annotated[
    tuple[_StrictPositiveInt, _StrictPositiveInt],
    BeforeValidator(_require_pair_container),
]
_ProbeSource = Annotated[
    Literal["custom", "ideal"],
    BeforeValidator(_require_exact_str),
]
_SyntheticObjectKind = Annotated[
    Literal["lines", "dead_leaves", "natural_patch"],
    BeforeValidator(_require_exact_str),
]
_ScanKind = Annotated[
    Literal["grid", "nongrid"],
    BeforeValidator(_require_exact_str),
]

_SIMULATION_ADAPTER_CONFIG = ConfigDict(
    extra="forbid",
    revalidate_instances="always",
    validate_default=True,
)


@with_config(_SIMULATION_ADAPTER_CONFIG)
@dataclass(frozen=True)
class ProbeSimulationConfig:
    """Probe source and transforms that are baked into generated data.

    ``source_path=None`` is retained for direct APIs that supply an in-memory
    custom probe. File-based generation entry points require a path before
    invoking the simulator.
    """

    source: _ProbeSource = "custom"
    source_path: Path | None = None
    transform_pipeline: Annotated[
        str,
        BeforeValidator(_require_exact_str),
    ] = "pad_preserve:64"
    mask_diameter: _StrictFinitePositiveNumber | None = None


@with_config(_SIMULATION_ADAPTER_CONFIG)
@dataclass(frozen=True)
class SyntheticObjectConfig:
    """Synthetic object family and generation counts."""

    kind: _SyntheticObjectKind = "lines"
    image_size: _StrictPositivePair = (392, 392)
    objects_per_probe: _StrictPositiveInt = 4
    diffractions_per_object: _StrictPositiveInt = 7000
    set_phi: _StrictBool = False


@with_config(_SIMULATION_ADAPTER_CONFIG)
@dataclass(frozen=True)
class ScanSimulationConfig:
    """Scan layout and train/test geometry for generated data."""

    kind: _ScanKind = "grid"
    grid_size: _StrictPositivePair = (1, 1)
    offset: _StrictNonNegativeInt = 4
    outer_offset_train: _StrictNonNegativeInt = 8
    outer_offset_test: _StrictNonNegativeInt = 20
    train_groups: _StrictPositiveInt = 2
    test_groups: _StrictPositiveInt = 2
    buffer: _StrictNonNegativeInt = 0


@with_config(_SIMULATION_ADAPTER_CONFIG)
@dataclass(frozen=True)
class DetectorSimulationConfig:
    """Detector/noise properties baked into generated diffraction data."""

    photons_per_pattern: _StrictFinitePositiveNumber = 1e9
    beamstop_diameter: _StrictFinitePositiveNumber | None = None


@with_config(_SIMULATION_ADAPTER_CONFIG)
@dataclass(frozen=True)
class SimulationConfig:
    """Canonical top-level recipe for generated ptychography data."""

    N: _StrictPositiveInt = 64
    probe: ProbeSimulationConfig = field(default_factory=ProbeSimulationConfig)
    object: SyntheticObjectConfig = field(default_factory=SyntheticObjectConfig)
    scan: ScanSimulationConfig = field(default_factory=ScanSimulationConfig)
    detector: DetectorSimulationConfig = field(default_factory=DetectorSimulationConfig)
    seed: _StrictOptionalInt = None


_SIMULATION_CONFIG_ADAPTER = TypeAdapter(SimulationConfig)


def _raise_simulation_validation_error(error: ValidationError) -> None:
    messages = []
    for detail in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    ):
        location = ".".join(str(part) for part in detail["loc"])
        path = f"simulation.{location}" if location else "simulation"
        messages.append(f"{path}: {detail['msg']}")
    raise ValueError("; ".join(messages)) from error


def _materialize_simulation_mappings(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _materialize_simulation_mappings(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_materialize_simulation_mappings(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize_simulation_mappings(item) for item in value)
    return value


def simulation_config_from_mapping(values: Mapping[str, Any]) -> SimulationConfig:
    """Build and validate ``SimulationConfig`` from YAML/TOML-shaped values.

    Unknown keys are errors at every level. This prevents training fields from
    being silently accepted under the simulation namespace.
    """

    if not isinstance(values, Mapping):
        raise ValueError("simulation must be a mapping")
    materialized = _materialize_simulation_mappings(values)
    try:
        validated = _SIMULATION_CONFIG_ADAPTER.validate_python(materialized)
    except ValidationError as error:
        _raise_simulation_validation_error(error)
    validate_simulation_config(validated)
    return validated


def simulation_config_to_dict(config: SimulationConfig) -> Dict[str, Any]:
    """Return the stable JSON-compatible canonical simulation recipe."""

    validate_simulation_config(config)
    return {
        "N": int(config.N),
        "seed": int(config.seed) if config.seed is not None else None,
        "probe": {
            "source": config.probe.source,
            "source_path": (
                str(config.probe.source_path)
                if config.probe.source_path is not None
                else None
            ),
            "transform_pipeline": config.probe.transform_pipeline,
            "mask_diameter": config.probe.mask_diameter,
        },
        "object": {
            "kind": config.object.kind,
            "image_size": list(config.object.image_size),
            "objects_per_probe": int(config.object.objects_per_probe),
            "diffractions_per_object": int(config.object.diffractions_per_object),
            "set_phi": bool(config.object.set_phi),
        },
        "scan": {
            "kind": config.scan.kind,
            "grid_size": list(config.scan.grid_size),
            "offset": int(config.scan.offset),
            "outer_offset_train": int(config.scan.outer_offset_train),
            "outer_offset_test": int(config.scan.outer_offset_test),
            "train_groups": int(config.scan.train_groups),
            "test_groups": int(config.scan.test_groups),
            "buffer": int(config.scan.buffer),
        },
        "detector": {
            "photons_per_pattern": float(config.detector.photons_per_pattern),
            "beamstop_diameter": config.detector.beamstop_diameter,
        },
    }


def simulation_config_sha256(config: SimulationConfig) -> str:
    """Return the canonical SHA-256 identity of one resolved simulation recipe."""

    encoded = json.dumps(
        simulation_config_to_dict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_simulation_config(path: str | Path) -> SimulationConfig:
    """Load a closed nested simulation recipe from TOML, YAML, or JSON."""
    source = Path(path)
    suffix = source.suffix.lower()
    try:
        if suffix == ".toml":
            raw = tomllib.loads(source.read_text(encoding="utf-8"))
        elif suffix in {".yaml", ".yml"}:
            raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        elif suffix == ".json":
            raw = json.loads(source.read_text(encoding="utf-8"))
        else:
            raise ValueError(
                "simulation config path must end in .toml, .yaml, .yml, or .json"
            )
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load simulation config {source}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("simulation config document must contain a mapping")
    values = raw.get("simulation", raw)
    if not isinstance(values, Mapping):
        raise ValueError("simulation config document's simulation value must be a mapping")
    unexpected_top_level = set(raw) - {"simulation"} if "simulation" in raw else set()
    if unexpected_top_level:
        raise ValueError(
            "simulation config document has unexpected top-level keys "
            f"{sorted(unexpected_top_level)}"
        )
    return simulation_config_from_mapping(values)


def _pipeline_final_size(pipeline: str) -> int:
    if not pipeline.strip():
        raise ValueError("simulation.probe.transform_pipeline must be non-empty")
    current_size: int | None = None
    boundary_index: int | None = None
    segments = pipeline.split("|")
    for index, raw_segment in enumerate(segments):
        if boundary_index is not None:
            raise ValueError(
                "pad_extrapolate_boundary_matched must be the final operation in "
                "simulation.probe.transform_pipeline"
            )
        segment = raw_segment.strip()
        if not segment:
            raise ValueError(
                "simulation.probe.transform_pipeline contains an empty operation"
            )
        op, separator, raw_parameters = segment.partition(":")
        op = op.strip()
        if op == "smooth":
            if boundary_index is not None:
                raise ValueError(
                    "simulation.probe.transform_pipeline cannot smooth after "
                    "pad_extrapolate_boundary_matched"
                )
            if not separator:
                raise ValueError("smooth requires a sigma")
            sigma = float(raw_parameters)
            if not math.isfinite(sigma) or sigma < 0:
                raise ValueError("smooth sigma must be finite and non-negative")
            continue
        if op not in {
            "pad",
            "pad_preserve",
            "interp",
            "pad_extrapolate",
            "pad_extrapolate_boundary_matched",
        }:
            raise ValueError(
                f"Unknown simulation.probe.transform_pipeline operation {op!r}"
            )
        if not separator or not raw_parameters.strip():
            raise ValueError(f"{op} requires a target size")
        target_text = raw_parameters.split(",", 1)[0].strip()
        current_size = int(target_text)
        if current_size <= 0:
            raise ValueError(f"{op} target size must be positive")
        if op == "pad_extrapolate_boundary_matched":
            boundary_index = index
    if current_size is None:
        raise ValueError(
            "simulation.probe.transform_pipeline must resolve an explicit final size"
        )
    return current_size


def _validate_square_pair(value: tuple[int, int], path: str) -> None:
    if value[0] != value[1]:
        raise ValueError(f"{path} must be square, got {value}")


def validate_simulation_config(config: SimulationConfig) -> None:
    """Validate one complete generated-data recipe."""

    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    try:
        _SIMULATION_CONFIG_ADAPTER.validate_python(config, strict=True)
    except ValidationError as error:
        _raise_simulation_validation_error(error)
    if config.probe.source == "ideal" and config.probe.source_path is not None:
        raise ValueError(
            "simulation.probe.source_path must be omitted when "
            "simulation.probe.source='ideal'"
        )
    final_size = _pipeline_final_size(config.probe.transform_pipeline)
    if final_size != config.N:
        raise ValueError(
            "simulation.probe.transform_pipeline final size "
            f"{final_size} does not match simulation.N {config.N}"
        )

    _validate_square_pair(config.object.image_size, "simulation.object.image_size")
    _validate_square_pair(config.scan.grid_size, "simulation.scan.grid_size")


def validate_simulation_compatibility(
    simulation: SimulationConfig,
    model: "ModelConfig",
) -> None:
    """Reject duplicated model/simulation shape fields that disagree."""

    validate_simulation_config(simulation)
    if simulation.N != model.N:
        raise ValueError(
            f"simulation.N={simulation.N} conflicts with model.N={model.N}"
        )
    expected_grid = (model.gridsize, model.gridsize)
    if simulation.scan.grid_size != expected_grid:
        raise ValueError(
            "simulation.scan.grid_size="
            f"{simulation.scan.grid_size} conflicts with model.gridsize={model.gridsize}"
        )

@dataclass(frozen=True)
class ModelConfig:
    """Core model architecture parameters."""
    N: Literal[64, 128, 256] = 64
    gridsize: int = 1
    n_filters_scale: int = 2
    model_type: Literal['pinn', 'supervised'] = 'pinn'
    architecture: Literal[
        'cnn', 'ffno', 'fno', 'hybrid', 'stable_hybrid', 'fno_vanilla', 'neuralop_uno', 'hybrid_resnet', 'hybrid_resnet_ffno_ptychoblock_encoder', 'hybrid_resnet_ptychoblock_ffno_encoder', 'spectral_resnet_bottleneck_net', 'spectral_resnet_bottleneck_linear_decoder', 'hybrid_resnet_ffno_bottleneck', 'hybrid_resnet_convnext_bottleneck'
    ] = 'cnn'
    fno_modes: int = 12
    fno_width: int = 32
    fno_blocks: int = 4
    fno_cnn_blocks: int = 2
    learned_input_channels: int = 1
    max_hidden_channels: Optional[int] = None
    resnet_width: Optional[int] = None
    fno_input_transform: Literal['none', 'sqrt', 'log1p', 'instancenorm'] = 'none'
    generator_output_mode: Literal['real_imag', 'amp_phase_logits', 'amp_phase'] = 'real_imag'
    amp_activation: Literal['sigmoid', 'swish', 'softplus', 'relu'] = 'sigmoid'
    object_big: Optional[bool] = None
    object_layout: Optional[Literal['single_patch', 'grouped_patches']] = None
    training_canvas: Optional[Literal['independent', 'relative_overlap']] = None
    training_patch_weighting: Optional[
        Literal['central_mask', 'uniform', 'probe']
    ] = None
    probe_big: bool = True  # Changed default
    probe_mask: bool = False  # Changed default
    probe_mask_sigma: float = 1.0
    probe_mask_diameter: Optional[float] = None
    pad_object: bool = True
    probe_scale: float = 4.
    gaussian_smoothing_sigma: float = 0.0


def resolve_model_object_policy(
    config: ModelConfig,
    *,
    backend: Optional[Literal['tensorflow', 'torch']] = None,
    warn_deprecated: bool = True,
) -> ModelConfig:
    """Return a fully materialized immutable public object policy."""
    if not isinstance(config, ModelConfig):
        raise TypeError("config must be ModelConfig")
    from ptycho.object_compatibility import resolve_public_object_policy

    policy = resolve_public_object_policy(
        object_big=config.object_big,
        object_layout=config.object_layout,
        training_canvas=config.training_canvas,
        training_patch_weighting=config.training_patch_weighting,
        pad_object=config.pad_object,
        probe_big=config.probe_big,
        backend=backend,
        warn_deprecated=warn_deprecated,
    )
    return replace(
        config,
        object_big=policy.object_big,
        object_layout=policy.object_layout,
        training_canvas=policy.training_canvas,
        training_patch_weighting=policy.training_patch_weighting,
    )

@dataclass
class TrainingConfig:
    """Training specific configuration."""
    model: ModelConfig
    train_data_file: Optional[Path] = None  # Made optional for simulation scripts
    test_data_file: Optional[Path] = None  # Added
    batch_size: int = 16
    nepochs: int = 50
    mae_weight: float = 0.0
    nll_weight: float = 1.0
    realspace_mae_weight: float = 0.0
    realspace_weight: float = 0.0
    nphotons: float = 1e9
    n_groups: Optional[int] = None  # Number of groups to generate (always means groups, regardless of gridsize)
    n_images: Optional[int] = None  # DEPRECATED: Use n_groups instead (kept for backward compatibility)
    n_subsample: Optional[int] = None  # Number of images to subsample before grouping (independent control)
    subsample_seed: Optional[int] = None  # Random seed for reproducible subsampling
    neighbor_count: int = 4  # K value: number of nearest neighbors for grouping (use higher values like 7 for K choose C oversampling)
    enable_oversampling: bool = False  # Explicit opt-in for K choose C oversampling (requires gridsize>1 and neighbor_pool_size>=C)
    neighbor_pool_size: Optional[int] = None  # Pool size for K choose C oversampling (if None, defaults to neighbor_count)
    positions_provided: bool = True
    probe_trainable: bool = False
    intensity_scale_trainable: bool = True  # Changed default
    output_dir: Path = Path("training_outputs")
    sequential_sampling: bool = False  # Use sequential sampling instead of random
    backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'  # Backend selection: defaults to TensorFlow for backward compatibility
    torch_loss_mode: Literal['poisson', 'mae'] = 'poisson'  # Backend-specific loss mode selector
    torch_mae_pred_l2_match_target: bool = False  # Optional Torch MAE prediction scaling mode
    gradient_clip_val: Optional[float] = None  # Gradient clipping threshold (None = disabled)
    gradient_clip_algorithm: Literal['norm', 'value', 'agc'] = 'norm'  # Gradient clipping algorithm: norm, value, or agc
    optimizer: Literal['adam', 'adamw', 'sgd'] = 'adam'  # Optimizer algorithm
    momentum: float = 0.9  # SGD momentum (ignored for Adam/AdamW)
    weight_decay: float = 0.0  # Weight decay (L2 penalty)
    adam_beta1: float = 0.9  # Adam/AdamW beta1
    adam_beta2: float = 0.999  # Adam/AdamW beta2
    scheduler: Literal['Default', 'Exponential', 'WarmupCosine', 'ReduceLROnPlateau'] = 'Default'  # LR scheduler type
    lr_warmup_epochs: int = 0  # Number of warmup epochs for WarmupCosine scheduler
    lr_min_ratio: float = 0.1  # Minimum LR ratio for WarmupCosine scheduler (eta_min = base_lr * ratio)
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    plateau_min_lr: float = 5e-5
    plateau_threshold: float = 0.0

    def __post_init__(self):
        """Handle backward compatibility for n_images → n_groups migration."""
        # Handle the deprecated n_images parameter
        if self.n_images is not None and self.n_groups is None:
            warnings.warn(
                "Parameter 'n_images' is deprecated and will be removed in a future version. "
                "Use 'n_groups' instead, which always means the number of groups regardless of gridsize.",
                DeprecationWarning,
                stacklevel=2
            )
            # Use object.__setattr__ to modify dataclass (not frozen anymore)
            object.__setattr__(self, 'n_groups', self.n_images)
        
        # Set default if neither was provided
        if self.n_groups is None:
            object.__setattr__(self, 'n_groups', 512)

@dataclass
class InferenceConfig:
    """Inference specific configuration."""
    model: ModelConfig
    model_path: Path
    test_data_file: Path
    n_groups: Optional[int] = None  # Number of groups to use (None = use all)
    n_images: Optional[int] = None  # DEPRECATED: Use n_groups instead (kept for backward compatibility)
    n_subsample: Optional[int] = None  # Number of images to subsample for inference (independent control)
    subsample_seed: Optional[int] = None  # Random seed for reproducible subsampling
    neighbor_count: int = 4  # K value: number of nearest neighbors for grouping (use higher values like 7 for K choose C oversampling)
    enable_oversampling: bool = False  # Explicit opt-in for K choose C oversampling (requires gridsize>1 and neighbor_pool_size>=C)
    neighbor_pool_size: Optional[int] = None  # Pool size for K choose C oversampling (if None, defaults to neighbor_count)
    debug: bool = False
    output_dir: Path = Path("inference_outputs")
    backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'  # Backend selection: defaults to TensorFlow for backward compatibility
    
    def __post_init__(self):
        """Handle backward compatibility for n_images → n_groups migration."""
        # Handle the deprecated n_images parameter
        if self.n_images is not None and self.n_groups is None:
            warnings.warn(
                "Parameter 'n_images' is deprecated and will be removed in a future version. "
                "Use 'n_groups' instead, which always means the number of groups regardless of gridsize.",
                DeprecationWarning,
                stacklevel=2
            )
            # Use object.__setattr__ to modify dataclass
            object.__setattr__(self, 'n_groups', self.n_images)

_STRUCTURAL_EXECUTION_ALIAS_NAMES = frozenset({
    'hybrid_skip_connections',
    'hybrid_downsample_steps',
    'hybrid_downsample_op',
    'hybrid_encoder_conv_hidden_scale',
    'hybrid_encoder_spectral_hidden_scale',
    'hybrid_encoder_conv_hidden_channels',
    'hybrid_encoder_spectral_hidden_channels',
    'hybrid_resnet_blocks',
    'hybrid_skip_style',
    'hybrid_resnet_bottleneck_layerscale_mode',
    'hybrid_resnet_bottleneck_layerscale_value',
    'hybrid_encoder_fusion_mode',
    'hybrid_encoder_layerscale_init',
    'hybrid_encoder_branch_gate_init',
    'hybrid_encoder_branch_select',
    'ffno_encoder_blocks',
    'ffno_encoder_modes',
    'ffno_encoder_share_weights',
    'ffno_encoder_gate_init',
    'ffno_encoder_norm',
    'ffno_encoder_mlp_ratio',
    'spectral_bottleneck_blocks',
    'spectral_bottleneck_modes',
    'spectral_bottleneck_share_weights',
    'spectral_bottleneck_gate_init',
    'spectral_bottleneck_gate_mode',
})


def _validate_execution_pre_environment_values(
    values: Mapping[str, Any],
) -> None:
    """Validate execution fields whose errors precede accelerator resolution."""

    devices = values["devices"]
    if (
        isinstance(devices, bool)
        or not (
            (isinstance(devices, int) and devices > 0)
            or devices == "auto"
        )
    ):
        raise ValueError(
            f"devices must be a positive integer or 'auto', got {devices!r}"
        )

    precision = values["precision"]
    valid_precisions = {"32-true", "16-mixed", "bf16-mixed"}
    if not isinstance(precision, str) or precision not in valid_precisions:
        raise ValueError(
            f"Invalid precision: {precision!r}. "
            f"Expected one of {sorted(valid_precisions)}."
        )

    accelerator = values["accelerator"]
    if accelerator == "tpu":
        raise ValueError(
            "Torch-XLA TPU execution is unsupported by this PyTorch runtime. "
            "Use accelerator='cpu', 'gpu'/'cuda', or 'mps'."
        )

    valid_accelerators = {"auto", "cpu", "gpu", "cuda", "mps"}
    if accelerator not in valid_accelerators:
        raise ValueError(
            f"Invalid accelerator: '{accelerator}'. "
            f"Expected one of {sorted(valid_accelerators)}."
        )


def _validate_execution_post_environment_values(
    values: Mapping[str, Any],
) -> None:
    """Validate execution fields whose errors follow accelerator resolution."""

    if values["num_workers"] < 0:
        raise ValueError(
            f"num_workers must be non-negative, got {values['num_workers']}"
        )

    if values["persistent_workers"] and values["num_workers"] <= 0:
        raise ValueError(
            "persistent_workers=True requires num_workers > 0"
        )

    if values["logger_backend"] not in {
        "csv",
        "tensorboard",
        "mlflow",
        None,
    }:
        raise ValueError(
            "logger_backend must be 'csv', 'tensorboard', 'mlflow', or None"
        )

    if values["learning_rate"] <= 0:
        raise ValueError(
            f"learning_rate must be positive, got {values['learning_rate']}"
        )

    if (
        values["inference_batch_size"] is not None
        and values["inference_batch_size"] <= 0
    ):
        raise ValueError(
            "inference_batch_size must be positive, "
            f"got {values['inference_batch_size']}"
        )

    if values["accum_steps"] <= 0:
        raise ValueError(
            f"accum_steps must be positive, got {values['accum_steps']}"
        )

    if values["checkpoint_save_top_k"] < 0:
        raise ValueError(
            "checkpoint_save_top_k must be non-negative, "
            f"got {values['checkpoint_save_top_k']}"
        )

    if values["early_stop_patience"] <= 0:
        raise ValueError(
            "early_stop_patience must be positive, "
            f"got {values['early_stop_patience']}"
        )

    valid_checkpoint_modes = {"min", "max"}
    if values["checkpoint_mode"] not in valid_checkpoint_modes:
        raise ValueError(
            f"Invalid checkpoint_mode: '{values['checkpoint_mode']}'. "
            f"Expected one of {sorted(valid_checkpoint_modes)}."
        )

    if values["hybrid_downsample_steps"] not in {1, 2}:
        raise ValueError(
            "hybrid_downsample_steps must be in [1, 2] "
            f"(got {values['hybrid_downsample_steps']})."
        )

    valid_downsample_ops = {"stride_conv", "avgpool_conv", "blurpool_conv"}
    if values["hybrid_downsample_op"] not in valid_downsample_ops:
        raise ValueError(
            f"Invalid hybrid_downsample_op '{values['hybrid_downsample_op']}'. "
            f"Expected one of {sorted(valid_downsample_ops)}."
        )

    if (
        not math.isfinite(float(values["hybrid_encoder_conv_hidden_scale"]))
        or float(values["hybrid_encoder_conv_hidden_scale"]) <= 0.0
    ):
        raise ValueError(
            "hybrid_encoder_conv_hidden_scale must be finite and > 0, "
            f"got {values['hybrid_encoder_conv_hidden_scale']}."
        )

    if (
        not math.isfinite(
            float(values["hybrid_encoder_spectral_hidden_scale"])
        )
        or float(values["hybrid_encoder_spectral_hidden_scale"]) <= 0.0
    ):
        raise ValueError(
            "hybrid_encoder_spectral_hidden_scale must be finite and > 0, "
            f"got {values['hybrid_encoder_spectral_hidden_scale']}."
        )

    if (
        values["hybrid_encoder_conv_hidden_channels"] is not None
        and values["hybrid_encoder_conv_hidden_channels"] <= 0
    ):
        raise ValueError(
            "hybrid_encoder_conv_hidden_channels must be positive when set, "
            f"got {values['hybrid_encoder_conv_hidden_channels']}."
        )

    if (
        values["hybrid_encoder_spectral_hidden_channels"] is not None
        and values["hybrid_encoder_spectral_hidden_channels"] <= 0
    ):
        raise ValueError(
            "hybrid_encoder_spectral_hidden_channels must be positive when set, "
            f"got {values['hybrid_encoder_spectral_hidden_channels']}."
        )

    if values["hybrid_resnet_blocks"] <= 0:
        raise ValueError(
            "hybrid_resnet_blocks must be positive, "
            f"got {values['hybrid_resnet_blocks']}."
        )

    if values["spectral_bottleneck_blocks"] <= 0:
        raise ValueError(
            "spectral_bottleneck_blocks must be positive, "
            f"got {values['spectral_bottleneck_blocks']}."
        )

    if values["spectral_bottleneck_modes"] <= 0:
        raise ValueError(
            "spectral_bottleneck_modes must be positive, "
            f"got {values['spectral_bottleneck_modes']}."
        )

    if not math.isfinite(float(values["spectral_bottleneck_gate_init"])):
        raise ValueError(
            "spectral_bottleneck_gate_init must be finite, "
            f"got {values['spectral_bottleneck_gate_init']}."
        )

    valid_gate_modes = {"shared", "per_block"}
    if values["spectral_bottleneck_gate_mode"] not in valid_gate_modes:
        raise ValueError(
            "Invalid spectral_bottleneck_gate_mode "
            f"'{values['spectral_bottleneck_gate_mode']}'. "
            f"Expected one of {sorted(valid_gate_modes)}."
        )

    valid_skip_styles = {"add", "concat", "gated_add"}
    if values["hybrid_skip_style"] not in valid_skip_styles:
        raise ValueError(
            f"Invalid hybrid_skip_style '{values['hybrid_skip_style']}'. "
            f"Expected one of {sorted(valid_skip_styles)}."
        )

    valid_layerscale_modes = {"learned", "fixed"}
    layerscale_mode = values["hybrid_resnet_bottleneck_layerscale_mode"]
    layerscale_value = values["hybrid_resnet_bottleneck_layerscale_value"]
    if layerscale_mode not in valid_layerscale_modes:
        raise ValueError(
            "Invalid hybrid_resnet_bottleneck_layerscale_mode "
            f"'{layerscale_mode}'. "
            f"Expected one of {sorted(valid_layerscale_modes)}."
        )
    if layerscale_mode == "learned":
        if layerscale_value is not None:
            raise ValueError(
                "hybrid_resnet_bottleneck_layerscale_value must be omitted when "
                "hybrid_resnet_bottleneck_layerscale_mode='learned'."
            )
    else:
        if layerscale_value is None:
            raise ValueError(
                "hybrid_resnet_bottleneck_layerscale_value must be provided when "
                "hybrid_resnet_bottleneck_layerscale_mode='fixed'."
            )
        if (
            not math.isfinite(float(layerscale_value))
            or float(layerscale_value) <= 0.0
        ):
            raise ValueError(
                "hybrid_resnet_bottleneck_layerscale_value must be finite and > 0 "
                f"(got {layerscale_value})."
            )

    valid_encoder_fusion_modes = {
        "baseline",
        "layerscale",
        "branch_gated",
        "branch_gated_layerscale",
    }
    if values["hybrid_encoder_fusion_mode"] not in valid_encoder_fusion_modes:
        raise ValueError(
            "Invalid hybrid_encoder_fusion_mode "
            f"'{values['hybrid_encoder_fusion_mode']}'. "
            f"Expected one of {sorted(valid_encoder_fusion_modes)}."
        )

    if (
        not math.isfinite(float(values["hybrid_encoder_layerscale_init"]))
        or float(values["hybrid_encoder_layerscale_init"]) <= 0.0
    ):
        raise ValueError(
            "hybrid_encoder_layerscale_init must be finite and > 0, "
            f"got {values['hybrid_encoder_layerscale_init']}."
        )

    if (
        not math.isfinite(float(values["hybrid_encoder_branch_gate_init"]))
        or float(values["hybrid_encoder_branch_gate_init"]) <= 0.0
    ):
        raise ValueError(
            "hybrid_encoder_branch_gate_init must be finite and > 0, "
            f"got {values['hybrid_encoder_branch_gate_init']}."
        )

    if values["ffno_encoder_blocks"] <= 0:
        raise ValueError(
            "ffno_encoder_blocks must be positive, "
            f"got {values['ffno_encoder_blocks']}."
        )

    if values["ffno_encoder_modes"] <= 0:
        raise ValueError(
            "ffno_encoder_modes must be positive, "
            f"got {values['ffno_encoder_modes']}."
        )

    if (
        not math.isfinite(float(values["ffno_encoder_gate_init"]))
        or float(values["ffno_encoder_gate_init"]) <= 0.0
    ):
        raise ValueError(
            "ffno_encoder_gate_init must be finite and > 0, "
            f"got {values['ffno_encoder_gate_init']}."
        )

    if (
        not math.isfinite(float(values["ffno_encoder_mlp_ratio"]))
        or float(values["ffno_encoder_mlp_ratio"]) <= 0.0
    ):
        raise ValueError(
            "ffno_encoder_mlp_ratio must be finite and > 0, "
            f"got {values['ffno_encoder_mlp_ratio']}."
        )


@dataclass
class PyTorchExecutionConfig:
    """
    PyTorch-specific execution configuration for runtime behavior control.

    This configuration controls PyTorch Lightning execution knobs, dataloader settings,
    and optimization parameters that do NOT exist in TensorFlow canonical configs.
    These fields are backend-specific and should not be bridged to params.cfg via
    update_legacy_dict (CONFIG-001 applies only to canonical configs).

    Design Context:
        - Introduced in ADR-003 Phase C1 to centralize execution-only parameters
        - Fields sourced from override_matrix.md §5 (PyTorch Execution Configuration)
        - Priority level 2 in override precedence (between explicit overrides and CLI defaults)
        - Referenced by: ptycho_torch/config_factory.py (factory payload construction)
        - Consumed by: ptycho_torch/workflows/components.py (Lightning Trainer + DataLoader)

    Usage:
        >>> from ptycho.config.config import PyTorchExecutionConfig
        >>> exec_cfg = PyTorchExecutionConfig(
        ...     accelerator='cpu',
        ...     deterministic=True,
        ...     num_workers=4,
        ...     enable_progress_bar=False,
        ... )
        >>> # Pass to factory:
        >>> payload = create_training_payload(..., execution_config=exec_cfg)

    Policy Compliance:
        - POLICY-001: PyTorch >=2.2 required for all ptycho_torch/ code
        - CONFIG-001: This config is execution-only; does NOT populate params.cfg

    Field Categories:
        1. Lightning Trainer knobs: accelerator, strategy, deterministic, gradient_clip_val
        2. DataLoader knobs: num_workers, pin_memory, persistent_workers, prefetch_factor
        3. Optimization knobs: learning_rate, scheduler, accum_steps
        4. Checkpoint/logging knobs: enable_progress_bar, enable_checkpointing, checkpoint_save_top_k, checkpoint_monitor_metric, checkpoint_mode, early_stop_patience
        5. Inference knobs: inference_batch_size, middle_trim, pad_eval
    """
    # Lightning Trainer knobs
    accelerator: str = 'auto'  # Options: 'cpu', 'gpu', 'cuda', 'mps', 'auto' (TPU/Torch-XLA unsupported)
    devices: Union[int, Literal["auto"]] = 1
    strategy: str = 'auto'  # Options: 'auto', 'ddp', 'fsdp', 'deepspeed'
    deterministic: Union[bool, Literal["warn"]] = True  # Enforce reproducibility (seed_everything + deterministic mode); "warn" allows non-deterministic ops with a warning
    precision: Literal["32-true", "16-mixed", "bf16-mixed"] = "32-true"
    gradient_clip_val: Optional[float] = None  # Gradient clipping threshold (None = disabled)
    gradient_clip_algorithm: Literal['norm', 'value', 'agc'] = 'norm'  # Gradient clipping algorithm
    accum_steps: int = 1  # Gradient accumulation steps (simulate larger batch size)

    # DataLoader knobs
    num_workers: int = 0  # Number of dataloader worker processes (0 = main process only; CPU-safe)
    pin_memory: bool = False  # Pin memory for faster CPU→GPU transfer (GPU-only; False for CPU safety)
    persistent_workers: bool = False  # Keep workers alive between epochs (requires num_workers > 0)
    prefetch_factor: Optional[int] = None  # Batches to prefetch per worker (None = default 2)

    # Optimization knobs
    learning_rate: float = 1e-3  # Optimizer learning rate (hardcoded in legacy code)
    scheduler: str = 'Default'  # LR scheduler type: 'Default', 'Exponential', 'MultiStage'

    # Checkpoint/logging knobs
    enable_progress_bar: bool = False  # Show training progress bar (derived from config.debug in legacy code)
    enable_checkpointing: bool = True  # Enable Lightning automatic checkpointing
    checkpoint_save_top_k: int = 1  # How many best checkpoints to keep
    checkpoint_monitor_metric: str = 'val_loss'  # Metric for best checkpoint selection
    checkpoint_mode: str = 'min'  # Mode for checkpoint monitoring: 'min' (lower is better) or 'max' (higher is better)
    early_stop_patience: int = 100  # Early stopping patience epochs (hardcoded in legacy code)

    # Logging knobs (Phase EB3.B - ADR-003)
    logger_backend: Optional[str] = 'csv'  # Experiment tracking backend: 'csv' (default), 'tensorboard', 'mlflow', or None

    # Reconstruction logging knobs (MLflow only)
    recon_log_every_n_epochs: Optional[int] = None  # Log intermediate reconstructions every N epochs (None = disabled)
    recon_log_num_patches: int = 4  # Number of fixed patch indices to log
    recon_log_fixed_indices: Optional[List[int]] = None  # Explicit patch indices (None = auto-select)
    recon_log_stitch: bool = False  # Log stitched full-resolution reconstructions (opt-in)
    recon_log_max_stitch_samples: Optional[int] = None  # Cap stitched samples (None = no limit)

    # Deprecated Torch topology input aliases. The training factory records
    # explicit use, maps it one-way into Torch ModelConfig, and rejects conflicts.
    hybrid_skip_connections: bool = False
    hybrid_downsample_steps: int = 2
    hybrid_downsample_op: Literal['stride_conv', 'avgpool_conv', 'blurpool_conv'] = 'stride_conv'
    hybrid_encoder_conv_hidden_scale: float = 1.0
    hybrid_encoder_spectral_hidden_scale: float = 1.0
    # Legacy absolute-width aliases retained for backwards compatibility.
    hybrid_encoder_conv_hidden_channels: Optional[int] = None
    hybrid_encoder_spectral_hidden_channels: Optional[int] = None
    hybrid_resnet_blocks: int = 6
    hybrid_skip_style: Literal['add', 'concat', 'gated_add'] = 'add'
    hybrid_resnet_bottleneck_layerscale_mode: Literal['learned', 'fixed'] = 'learned'
    hybrid_resnet_bottleneck_layerscale_value: Optional[float] = None
    # Encoder-fusion controls (per-block scalars; Torch-only study plumbing).
    hybrid_encoder_fusion_mode: Literal[
        'baseline',
        'layerscale',
        'branch_gated',
        'branch_gated_layerscale',
    ] = 'baseline'
    hybrid_encoder_layerscale_init: float = 0.1
    hybrid_encoder_branch_gate_init: float = 0.1
    # Orthogonal deterministic encoder-branch ablation control. Values:
    # 'both' (default), 'conv_only' (drop spectral branch), 'spectral_only' (drop conv branch).
    hybrid_encoder_branch_select: Literal[
        'both',
        'conv_only',
        'spectral_only',
    ] = 'both'
    ffno_encoder_blocks: int = 24
    ffno_encoder_modes: int = 12
    ffno_encoder_share_weights: bool = True
    ffno_encoder_gate_init: float = 0.1
    ffno_encoder_norm: str = 'instance'
    ffno_encoder_mlp_ratio: float = 2.0
    spectral_bottleneck_blocks: int = 6
    spectral_bottleneck_modes: int = 12
    spectral_bottleneck_share_weights: bool = True
    spectral_bottleneck_gate_init: float = 0.1
    spectral_bottleneck_gate_mode: Literal['shared', 'per_block'] = 'shared'

    # Inference-specific knobs
    inference_batch_size: Optional[int] = None  # Override batch_size for inference (None = use training batch_size)
    middle_trim: int = 0  # Inference trimming parameter (not yet implemented)
    pad_eval: bool = False  # Padding for evaluation (not yet implemented)

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        positional_names = {
            field_info.name for field_info in fields(cls)[:len(args)]
        }
        instance._explicit_structural_aliases = frozenset(
            (positional_names | set(kwargs)) & _STRUCTURAL_EXECUTION_ALIAS_NAMES
        )
        return instance

    def __post_init__(self):
        """
        Validate PyTorchExecutionConfig fields and resolve 'auto' accelerator (ADR-003 Phase D.B).

        Auto-Resolution Logic (POLICY-001 compliance):
            When accelerator='auto':
            - Resolves to 'cuda' if torch.cuda.is_available() == True
            - Falls back to 'cpu' with POLICY-001 warning if no CUDA device found
            - Ensures GPU-first behavior per docs/workflows/pytorch.md §12

        Raises:
            ValueError: If validation fails with descriptive message

        Validation Rules (from training_refactor.md §Component 2 + EB1.A):
            1. accelerator must be in whitelist {'auto', 'cpu', 'gpu', 'cuda', 'mps'}
            2. num_workers must be non-negative
            3. learning_rate must be positive
            4. inference_batch_size (if provided) must be positive
            5. accum_steps must be positive
            6. checkpoint_save_top_k must be non-negative
            7. early_stop_patience must be positive
            8. checkpoint_mode must be in whitelist {'min', 'max'}

        Notes:
            - Warnings for deterministic+num_workers handled in CLI helper (build_execution_config_from_args)
            - Field defaults are safe; validation catches programmatic misuse
            - Auto-resolution modifies the accelerator field in-place via object.__setattr__
        """
        _validate_execution_pre_environment_values(self.__dict__)

        # Auto-resolution: 'auto' → 'cuda' if available, else 'cpu' with POLICY-001 warning
        if self.accelerator == 'auto':
            try:
                import torch
                if torch.cuda.is_available():
                    object.__setattr__(self, 'accelerator', 'cuda')
                else:
                    object.__setattr__(self, 'accelerator', 'cpu')
                    warnings.warn(
                        "POLICY-001: PyTorch backend defaults to GPU execution. "
                        "No CUDA device detected; falling back to CPU. "
                        "For production workloads, ensure CUDA is available or explicitly set accelerator='cpu'.",
                        UserWarning,
                        stacklevel=3
                    )
            except ImportError:
                # Should not happen given POLICY-001 (torch is mandatory), but handle gracefully
                object.__setattr__(self, 'accelerator', 'cpu')
                warnings.warn(
                    "POLICY-001: PyTorch not available. Falling back to CPU accelerator. "
                    "Install PyTorch (torch>=2.2) for GPU acceleration.",
                    UserWarning,
                    stacklevel=3
                )

        _validate_execution_post_environment_values(self.__dict__)


_execution_init_signature = inspect.signature(PyTorchExecutionConfig.__init__)
PyTorchExecutionConfig.__signature__ = _execution_init_signature.replace(
    parameters=tuple(_execution_init_signature.parameters.values())[1:]
)


def validate_model_config(config: ModelConfig) -> None:
    """Validate model configuration."""
    resolve_model_object_policy(config)
    valid_arches = {
        'cnn',
        'ffno',
        'fno',
        'hybrid',
        'stable_hybrid',
        'fno_vanilla',
        'neuralop_uno',
        'hybrid_resnet',
        'hybrid_resnet_ffno_ptychoblock_encoder',
        'hybrid_resnet_ptychoblock_ffno_encoder',
        'spectral_resnet_bottleneck_net',
        'spectral_resnet_bottleneck_linear_decoder',
        'hybrid_resnet_ffno_bottleneck',
        'hybrid_resnet_convnext_bottleneck',
    }
    if config.architecture not in valid_arches:
        raise ValueError(
            f"Invalid architecture '{config.architecture}'. "
            f"Expected one of {sorted(valid_arches)}."
        )
    if config.architecture in {
        "hybrid_resnet",
        "hybrid_resnet_ffno_ptychoblock_encoder",
        "hybrid_resnet_ptychoblock_ffno_encoder",
        "spectral_resnet_bottleneck_net",
        "spectral_resnet_bottleneck_linear_decoder",
        "hybrid_resnet_ffno_bottleneck",
        "hybrid_resnet_convnext_bottleneck",
    }:
        if config.fno_blocks < 3:
            raise ValueError(
                f"{config.architecture} requires fno_blocks >= 3 to downsample to N/4 "
                f"(got fno_blocks={config.fno_blocks})."
            )
        if config.resnet_width is not None:
            if config.resnet_width <= 0:
                raise ValueError(
                    f"resnet_width must be positive when set, got {config.resnet_width}."
                )
            if config.resnet_width % 4 != 0:
                raise ValueError(
                    "resnet_width must be divisible by 4 so the CycleGAN upsamplers "
                    f"produce integer channel sizes (got {config.resnet_width})."
                )
    if config.gridsize <= 0:
        raise ValueError(f"gridsize must be positive, got {config.gridsize}")
    if config.n_filters_scale <= 0:
        raise ValueError(f"n_filters_scale must be positive, got {config.n_filters_scale}")
    if config.probe_scale <= 0:
        raise ValueError(f"probe_scale must be positive, got {config.probe_scale}")
    if config.gaussian_smoothing_sigma < 0:
        raise ValueError(f"gaussian_smoothing_sigma must be non-negative, got {config.gaussian_smoothing_sigma}")

def validate_training_config(config: TrainingConfig) -> None:
    """Validate training configuration."""
    validate_model_config(config.model)
    if config.batch_size <= 0 or (config.batch_size & (config.batch_size - 1)):
        raise ValueError(f"batch_size must be positive power of 2, got {config.batch_size}")
    if config.nepochs <= 0:
        raise ValueError(f"nepochs must be positive, got {config.nepochs}")
    if not (0 <= config.mae_weight <= 1):
        raise ValueError(f"mae_weight must be in [0,1], got {config.mae_weight}")
    if not (0 <= config.nll_weight <= 1):
        raise ValueError(f"nll_weight must be in [0,1], got {config.nll_weight}")
    if config.nphotons <= 0:
        raise ValueError(f"nphotons must be positive, got {config.nphotons}")

def validate_inference_config(config: InferenceConfig) -> None:
    """Validate inference configuration."""
    validate_model_config(config.model)
    # Check if model_path is a directory containing wts.h5.zip
    if config.model_path.is_dir():
        expected_model_file = config.model_path / "wts.h5.zip"
        if not expected_model_file.exists():
            raise ValueError(f"Model archive not found: {expected_model_file}")
    else:
        # Check if the path itself exists (could be a zip file)
        if not config.model_path.exists():
            # Try with .zip extension  
            zip_path = config.model_path.with_suffix('.zip')
            if not zip_path.exists():
                # Special case: check if this looks like a wts.h5 path and try wts.h5.zip
                if config.model_path.name == "wts.h5":
                    alt_path = config.model_path.with_suffix('.h5.zip')
                    if not alt_path.exists():
                        raise ValueError(f"model_path does not exist: {config.model_path} (also checked {zip_path} and {alt_path})")
                else:
                    raise ValueError(f"model_path does not exist: {config.model_path} (also checked {zip_path})")

def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        path: Path to YAML config file
        
    Returns:
        Dictionary containing configuration values
        
    Raises:
        OSError: If file cannot be read
        yaml.YAMLError: If YAML is invalid
    """
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise type(e)(f"Failed to load config from {path}: {str(e)}")

def dataclass_to_legacy_dict(obj: Any) -> Dict[str, Any]:
    """Convert dataclass to legacy dictionary format with key mappings.
    
    Args:
        obj: Dataclass instance to convert
        
    Returns:
        Dictionary with legacy parameter names and values
    """
    if isinstance(obj, SimulationConfig):
        validate_simulation_config(obj)
        return {
            "N": obj.N,
            "probe_source": (
                "ideal_disk" if obj.probe.source == "ideal" else obj.probe.source
            ),
            "probe_npz": (
                str(obj.probe.source_path)
                if obj.probe.source_path is not None
                else None
            ),
            "probe_transform_pipeline": obj.probe.transform_pipeline,
            "probe_mask_diameter": obj.probe.mask_diameter,
            "data_source": obj.object.kind,
            "object_class": obj.object.kind,
            "size": obj.object.image_size[0],
            "objects_per_probe": obj.object.objects_per_probe,
            "diff_per_object": obj.object.diffractions_per_object,
            "set_phi": obj.object.set_phi,
            "scan_kind": obj.scan.kind,
            "gridsize": obj.scan.grid_size[0],
            "offset": obj.scan.offset,
            "outer_offset_train": obj.scan.outer_offset_train,
            "outer_offset_test": obj.scan.outer_offset_test,
            "nimgs_train": obj.scan.train_groups,
            "nimgs_test": obj.scan.test_groups,
            "max_position_jitter": obj.scan.buffer,
            "nphotons": obj.detector.photons_per_pattern,
            "beamstop_diameter": obj.detector.beamstop_diameter,
            "npseed": obj.seed,
        }

    if isinstance(obj, ModelConfig):
        obj = resolve_model_object_policy(obj)
    elif hasattr(obj, "model") and isinstance(obj.model, ModelConfig):
        obj = replace(obj, model=resolve_model_object_policy(obj.model))

    # Key mappings from dataclass field names to legacy param names
    KEY_MAPPINGS = {
        'object_big': 'object.big',
        'probe_big': 'probe.big', 
        'probe_mask': 'probe.mask',
        'probe_trainable': 'probe.trainable',
        'intensity_scale_trainable': 'intensity_scale.trainable',
        'positions_provided': 'positions.provided',
        'output_dir': 'output_prefix',
        'train_data_file': 'train_data_file_path',
        'test_data_file': 'test_data_file_path'
    }

    # Convert dataclass to dict
    d = asdict(obj)

    # Handle nested ModelConfig
    if 'model' in d:
        model_dict = d.pop('model')
        d.update(model_dict)

    # Apply key mappings and convert Path objects to strings
    for old_key, new_key in KEY_MAPPINGS.items():
        if old_key in d:
            value = d.pop(old_key)
            # Convert Path objects to strings
            if isinstance(value, Path):
                d[new_key] = str(value)
            else:
                d[new_key] = value

    # Convert Path to string (legacy fallback)
    if 'output_dir' in d:
        d['output_prefix'] = str(d.pop('output_dir'))

    return d

def update_legacy_dict(cfg: Dict[str, Any], dataclass_obj: Any) -> None:
    """Update legacy dictionary with dataclass values.

    ⚠️ CRITICAL: Call this BEFORE any data loading operations!

    Common failure scenario:
    - Symptom: Shape (*, 64, 64, 1) instead of (*, 64, 64, 4) with gridsize=2
    - Cause: This function wasn't called before generate_grouped_data()
    - Fix: Call immediately after config setup, before load_data()

    Updates values from the dataclass, but skips None values to preserve
    existing parameter values when new configuration doesn't specify them.

    Args:
        cfg: Legacy dictionary to update
        dataclass_obj: Dataclass instance containing new values
    """
    new_values = dataclass_to_legacy_dict(dataclass_obj)

    # Update values from dataclass, but skip None values to preserve existing params
    # Convert any remaining Path objects to strings for legacy compatibility
    for key, value in new_values.items():
        if value is not None:
            # Convert Path to string if not already done by KEY_MAPPINGS
            if isinstance(value, Path):
                cfg[key] = str(value)
            else:
                cfg[key] = value
