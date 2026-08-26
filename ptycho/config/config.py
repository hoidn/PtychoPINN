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
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
from typing import Annotated, Dict, Any, List, Optional, Literal, Union
import json
import hashlib
import math
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib
from pydantic import (
    BeforeValidator,
    ConfigDict,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    with_config,
)
import yaml
import warnings

from .strict_types import (
    _require_exact_int,
    _require_exact_str,
    _StrictBool,
    _StrictClosedUnitNumber,
    _StrictFiniteNonNegativeNumber,
    _StrictFinitePositiveNumber,
    _StrictHalfOpenUnitNumber,
    _StrictNonNegativeInt,
    _StrictOpenUnitNumber,
    _StrictOptionalInt,
    _StrictPositiveInt,
)

# Export list for public API (config-factory contract; docs/specs/spec-ptycho-config-bridge.md)
# Ensures PyTorchExecutionConfig is discoverable from the canonical config module
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


def _require_pair_container(value: Any) -> Any:
    if type(value) not in {list, tuple}:
        raise ValueError("must be a list or tuple")
    return value


def _materialize_public_path(value: Any, info: ValidationInfo) -> Path:
    if (info.context or {}).get("strict_instance", False):
        if not isinstance(value, Path):
            raise ValueError("must be a pathlib.Path")
        return value
    try:
        return Path(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"must be path-like, got {value!r}") from error


_PublicPath = Annotated[Path, BeforeValidator(_materialize_public_path)]
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
_ScanPositionLayout = Annotated[
    Literal["uniform_random", "raster", "fixed_pitch_raster"],
    BeforeValidator(_require_exact_str),
]
_PatchAmplitudeNormalization = Annotated[
    Literal["none", "mean_patch_max"],
    BeforeValidator(_require_exact_str),
]
#: Elided from recipe digests when unchanged so that adding the field does not
#: move the identity of any dataset generated before it existed.
DEFAULT_SCAN_POSITION_LAYOUT = "uniform_random"

_DATACLASS_ADAPTER_CONFIG = ConfigDict(
    extra="forbid",
    revalidate_instances="always",
    validate_default=True,
)


# --- Gauge constants (single stated home, TF backend) -----------------------
# Normative owner: docs/specs/spec-ptycho-core.md (probe/object gauge) and
# docs/specs/spec-ptycho-torch-probe-layout.md (torch mirror). These pins fix
# the probe normalization gauge; changing either value is a scientific gauge
# change requiring its own reviewed decision, never a refactor.
PROBE_SCALE_DEFAULT = 4.0
PROBE_NORMALIZE_DEFAULT = True


@with_config(_DATACLASS_ADAPTER_CONFIG)
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
    ideal_scale: _StrictFinitePositiveNumber = 0.7
    simulation_normalization_scale: _StrictFinitePositiveNumber | None = None


@with_config(_DATACLASS_ADAPTER_CONFIG)
@dataclass(frozen=True)
class SyntheticObjectConfig:
    """Synthetic object family and generation counts."""

    kind: _SyntheticObjectKind = "lines"
    image_size: _StrictPositivePair = (392, 392)
    objects_per_probe: _StrictPositiveInt = 4
    diffractions_per_object: _StrictPositiveInt = 7000
    set_phi: _StrictBool = False
    # Appended last so existing positional construction keeps working.
    patch_amplitude_normalization: _PatchAmplitudeNormalization = "none"
    # Used only by registered source-backed object recipes.
    source_path: Path | None = None


@with_config(_DATACLASS_ADAPTER_CONFIG)
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
    # Appended last so existing positional construction keeps working.
    position_layout: _ScanPositionLayout = DEFAULT_SCAN_POSITION_LAYOUT


@with_config(_DATACLASS_ADAPTER_CONFIG)
@dataclass(frozen=True)
class DetectorSimulationConfig:
    """Detector/noise properties baked into generated diffraction data."""

    photons_per_pattern: _StrictFinitePositiveNumber = 1e9
    beamstop_diameter: _StrictFinitePositiveNumber | None = None


@with_config(_DATACLASS_ADAPTER_CONFIG)
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
    probe = {
        "source": config.probe.source,
        "source_path": (
            str(config.probe.source_path)
            if config.probe.source_path is not None
            else None
        ),
        "transform_pipeline": config.probe.transform_pipeline,
        "mask_diameter": config.probe.mask_diameter,
        "ideal_scale": config.probe.ideal_scale,
    }
    if config.probe.simulation_normalization_scale is not None:
        probe["simulation_normalization_scale"] = (
            config.probe.simulation_normalization_scale
        )
    object_config = {
        "kind": config.object.kind,
        "image_size": list(config.object.image_size),
        "objects_per_probe": int(config.object.objects_per_probe),
        "diffractions_per_object": int(config.object.diffractions_per_object),
        "set_phi": bool(config.object.set_phi),
        "patch_amplitude_normalization": (
            config.object.patch_amplitude_normalization
        ),
    }
    if config.object.source_path is not None:
        object_config["source_path"] = str(config.object.source_path)
    return {
        "N": int(config.N),
        "seed": int(config.seed) if config.seed is not None else None,
        "probe": probe,
        "object": object_config,
        "scan": {
            "kind": config.scan.kind,
            "position_layout": config.scan.position_layout,
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


def simulation_config_digest_input(config: SimulationConfig) -> Dict[str, Any]:
    """Return the canonical recipe reduced to its identity-bearing fields.

    ``scan.position_layout`` is dropped when it holds its default so that every
    recipe authored before the field existed keeps its historical digest; a
    non-default layout changes positions and therefore must change identity.
    """

    payload = simulation_config_to_dict(config)
    scan = payload["scan"]
    if scan.get("position_layout") == DEFAULT_SCAN_POSITION_LAYOUT:
        scan.pop("position_layout")
    object_recipe = payload["object"]
    if object_recipe.get("patch_amplitude_normalization") == "none":
        object_recipe.pop("patch_amplitude_normalization")
    if object_recipe.get("source_path") is None:
        object_recipe.pop("source_path", None)
    return payload


def simulation_config_sha256(config: SimulationConfig) -> str:
    """Return the canonical SHA-256 identity of one resolved simulation recipe."""

    encoded = json.dumps(
        simulation_config_digest_input(config),
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
    if config.probe.source == "custom" and config.probe.ideal_scale != 0.7:
        raise ValueError(
            "simulation.probe.ideal_scale must remain 0.7 when "
            "simulation.probe.source='custom' because ideal_scale only affects "
            "ideal probe construction"
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

@with_config(_DATACLASS_ADAPTER_CONFIG)
@dataclass(frozen=True)
class ModelConfig:
    """Core model architecture parameters."""
    N: Annotated[
        Literal[64, 128, 256],
        BeforeValidator(_require_exact_int),
    ] = 64
    gridsize: _StrictPositiveInt = 1
    n_filters_scale: _StrictPositiveInt = 2
    model_type: Annotated[Literal['pinn', 'supervised'], BeforeValidator(_require_exact_str)] = 'pinn'
    architecture: Annotated[
        Literal[
            'cnn', 'ffno', 'fno', 'hybrid', 'stable_hybrid', 'fno_vanilla', 'neuralop_uno'
        ],
        BeforeValidator(_require_exact_str),
    ] = 'cnn'
    fno_modes: _StrictPositiveInt = 12
    fno_width: _StrictPositiveInt = 32
    fno_blocks: _StrictPositiveInt = 4
    fno_cnn_blocks: _StrictNonNegativeInt = 2
    learned_input_channels: _StrictPositiveInt = 1
    max_hidden_channels: _StrictPositiveInt | None = None
    resnet_width: _StrictPositiveInt | None = None
    fno_input_transform: Annotated[Literal['none', 'sqrt', 'log1p', 'instancenorm'], BeforeValidator(_require_exact_str)] = 'none'
    generator_output_mode: Annotated[Literal['real_imag', 'amp_phase_logits', 'amp_phase'], BeforeValidator(_require_exact_str)] = 'real_imag'
    amp_activation: Annotated[Literal['sigmoid', 'swish', 'softplus', 'relu'], BeforeValidator(_require_exact_str)] = 'sigmoid'
    object_big: _StrictBool | None = None
    object_layout: Annotated[Literal['single_patch', 'grouped_patches'], BeforeValidator(_require_exact_str)] | None = None
    training_canvas: Annotated[Literal['independent', 'relative_overlap'], BeforeValidator(_require_exact_str)] | None = None
    training_patch_weighting: Annotated[Literal['central_mask', 'uniform', 'probe'], BeforeValidator(_require_exact_str)] | None = None
    probe_big: _StrictBool = True  # Changed default
    probe_mask: _StrictBool = False  # Changed default
    probe_mask_sigma: _StrictFiniteNonNegativeNumber = 1.0
    probe_mask_diameter: _StrictFinitePositiveNumber | None = None
    pad_object: _StrictBool = True
    probe_scale: _StrictFinitePositiveNumber = PROBE_SCALE_DEFAULT
    gaussian_smoothing_sigma: _StrictFiniteNonNegativeNumber = 0.0


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

@with_config(_DATACLASS_ADAPTER_CONFIG)
@dataclass
class TrainingConfig:
    """Training specific configuration."""
    model: ModelConfig
    train_data_file: _PublicPath | None = None  # Made optional for simulation scripts
    test_data_file: _PublicPath | None = None  # Added
    batch_size: _StrictPositiveInt = 16
    nepochs: _StrictNonNegativeInt = 50
    mae_weight: _StrictClosedUnitNumber = 0.0
    nll_weight: _StrictClosedUnitNumber = 1.0
    realspace_mae_weight: _StrictClosedUnitNumber = 0.0
    realspace_weight: _StrictClosedUnitNumber = 0.0
    nphotons: _StrictFiniteNonNegativeNumber = 1e9
    training_groups: _StrictNonNegativeInt | None = None  # Number of groups to generate (always means groups, regardless of gridsize)
    n_images: _StrictNonNegativeInt | None = None  # DEPRECATED: Use training_groups instead (kept for backward compatibility)
    train_raw_selection: _StrictNonNegativeInt | None = None  # Number of images to subsample before grouping (independent control)
    subsample_seed: _StrictNonNegativeInt | None = None  # Random seed for reproducible subsampling
    neighbor_count: _StrictPositiveInt = 4  # K value: number of nearest neighbors for grouping (use higher values like 7 for K choose C oversampling)
    enable_oversampling: _StrictBool = False  # Explicit opt-in for K choose C oversampling (requires gridsize>1 and neighbor_pool_size>=C)
    neighbor_pool_size: _StrictPositiveInt | None = None  # Pool size for K choose C oversampling (if None, defaults to neighbor_count)
    positions_provided: _StrictBool = True
    probe_trainable: _StrictBool = False
    intensity_scale_trainable: _StrictBool = True  # Changed default
    output_dir: _PublicPath = Path("training_outputs")
    sequential_sampling: _StrictBool = False  # Use sequential sampling instead of random
    backend: Annotated[Literal['tensorflow', 'pytorch'], BeforeValidator(_require_exact_str)] = 'tensorflow'  # Backend selection: defaults to TensorFlow for backward compatibility
    torch_loss_mode: Annotated[Literal['poisson', 'mae'], BeforeValidator(_require_exact_str)] = 'poisson'  # Backend-specific loss mode selector
    torch_mae_pred_l2_match_target: _StrictBool = False  # Optional Torch MAE prediction scaling mode
    gradient_clip_val: _StrictFiniteNonNegativeNumber | None = None  # Gradient clipping threshold (None = disabled)
    gradient_clip_algorithm: Annotated[Literal['norm', 'value', 'agc'], BeforeValidator(_require_exact_str)] = 'norm'  # Gradient clipping algorithm: norm, value, or agc
    optimizer: Annotated[Literal['adam', 'adamw', 'sgd'], BeforeValidator(_require_exact_str)] = 'adam'  # Optimizer algorithm
    momentum: _StrictClosedUnitNumber = 0.9  # SGD momentum (ignored for Adam/AdamW)
    weight_decay: _StrictFiniteNonNegativeNumber = 0.0  # Weight decay (L2 penalty)
    adam_beta1: _StrictHalfOpenUnitNumber = 0.9  # Adam/AdamW beta1
    adam_beta2: _StrictHalfOpenUnitNumber = 0.999  # Adam/AdamW beta2
    scheduler: Annotated[Literal['Default', 'Exponential', 'WarmupCosine', 'ReduceLROnPlateau'], BeforeValidator(_require_exact_str)] = 'Default'  # LR scheduler type
    lr_warmup_epochs: _StrictNonNegativeInt = 0  # Number of warmup epochs for WarmupCosine scheduler
    lr_min_ratio: _StrictClosedUnitNumber = 0.1  # Minimum LR ratio for WarmupCosine scheduler (eta_min = base_lr * ratio)
    plateau_factor: _StrictOpenUnitNumber = 0.5
    plateau_patience: _StrictNonNegativeInt = 2
    plateau_min_lr: _StrictFiniteNonNegativeNumber = 5e-5
    plateau_threshold: _StrictFiniteNonNegativeNumber = 0.0

    def __post_init__(self):
        """Handle backward compatibility for n_images → training_groups migration."""
        # Handle the deprecated n_images parameter
        if self.n_images is not None and self.training_groups is None:
            warnings.warn(
                "Parameter 'n_images' is deprecated and will be removed in a future version. "
                "Use 'training_groups' instead, which always means the number of groups regardless of gridsize.",
                DeprecationWarning,
                stacklevel=2
            )
            # Use object.__setattr__ to modify dataclass (not frozen anymore)
            object.__setattr__(self, 'training_groups', self.n_images)

        # Set default if neither was provided
        if self.training_groups is None:
            object.__setattr__(self, 'training_groups', 512)

@with_config(_DATACLASS_ADAPTER_CONFIG)
@dataclass
class InferenceConfig:
    """Inference specific configuration."""
    model: ModelConfig
    model_path: _PublicPath
    test_data_file: _PublicPath
    inference_groups: _StrictNonNegativeInt | None = None  # Number of groups to use (None = use all)
    n_images: _StrictNonNegativeInt | None = None  # DEPRECATED: Use inference_groups instead (kept for backward compatibility)
    inference_raw_selection: _StrictNonNegativeInt | None = None  # Number of images to subsample for inference (independent control)
    subsample_seed: _StrictNonNegativeInt | None = None  # Random seed for reproducible subsampling
    neighbor_count: _StrictPositiveInt = 4  # K value: number of nearest neighbors for grouping (use higher values like 7 for K choose C oversampling)
    enable_oversampling: _StrictBool = False  # Explicit opt-in for K choose C oversampling (requires gridsize>1 and neighbor_pool_size>=C)
    neighbor_pool_size: _StrictPositiveInt | None = None  # Pool size for K choose C oversampling (if None, defaults to neighbor_count)
    debug: _StrictBool = False
    output_dir: _PublicPath = Path("inference_outputs")
    backend: Annotated[Literal['tensorflow', 'pytorch'], BeforeValidator(_require_exact_str)] = 'tensorflow'  # Backend selection: defaults to TensorFlow for backward compatibility
    
    def __post_init__(self):
        """Handle backward compatibility for n_images → inference_groups migration."""
        # Handle the deprecated n_images parameter
        if self.n_images is not None and self.inference_groups is None:
            warnings.warn(
                "Parameter 'n_images' is deprecated and will be removed in a future version. "
                "Use 'inference_groups' instead, which always means the number of groups regardless of gridsize.",
                DeprecationWarning,
                stacklevel=2
            )
            # Use object.__setattr__ to modify dataclass
            object.__setattr__(self, 'inference_groups', self.n_images)

def _validate_execution_pre_environment_values(
    values: Mapping[str, Any],
) -> None:
    """Validate unresolved request fields before capability observation."""

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
    """Validate environment-independent runtime fields."""

    if (
        isinstance(values["num_workers"], bool)
        or not isinstance(values["num_workers"], int)
        or values["num_workers"] < 0
    ):
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

    if (
            values["inference_batch_size"] is not None
            and (
                isinstance(values["inference_batch_size"], bool)
                or not isinstance(values["inference_batch_size"], int)
                or values["inference_batch_size"] <= 0
            )
    ):
        raise ValueError(
            "inference_batch_size must be positive, "
            f"got {values['inference_batch_size']}"
        )

    if (
        isinstance(values["checkpoint_save_top_k"], bool)
        or not isinstance(values["checkpoint_save_top_k"], int)
        or values["checkpoint_save_top_k"] < 0
    ):
        raise ValueError(
            "checkpoint_save_top_k must be non-negative, "
            f"got {values['checkpoint_save_top_k']}"
        )

    if (
        isinstance(values["early_stop_patience"], bool)
        or not isinstance(values["early_stop_patience"], int)
        or values["early_stop_patience"] <= 0
    ):
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


def _validate_execution_resolved_environment_values(
    values: Mapping[str, Any],
) -> None:
    """Reject request-only environment values on the resolved carrier."""

    accelerator = values["accelerator"]
    if accelerator == "tpu":
        raise ValueError(
            "Torch-XLA TPU execution is unsupported by this PyTorch runtime. "
            "Use accelerator='cpu', 'gpu'/'cuda', or 'mps'."
        )
    valid_accelerators = {"cpu", "gpu", "cuda", "mps"}
    if accelerator not in valid_accelerators:
        raise ValueError(
            f"Invalid resolved accelerator: {accelerator!r}. "
            f"Expected one of {sorted(valid_accelerators)}."
        )

    devices = values["devices"]
    if (
        isinstance(devices, bool)
        or not isinstance(devices, int)
        or devices <= 0
    ):
        raise ValueError(
            f"resolved devices must be a positive integer, got {devices!r}"
        )

    precision = values["precision"]
    valid_precisions = {"32-true", "16-mixed", "bf16-mixed"}
    if not isinstance(precision, str) or precision not in valid_precisions:
        raise ValueError(
            f"Invalid precision: {precision!r}. "
            f"Expected one of {sorted(valid_precisions)}."
        )


@dataclass
class PyTorchExecutionConfig:
    """Resolved PyTorch Trainer and DataLoader runtime carrier.

    User requests, unresolved ``"auto"`` values, and explicit-input provenance
    belong to :class:`ptycho_torch.execution_request.ExecutionRequest`.
    Optimizer semantics belong to Torch ``TrainingConfig`` and model topology
    belongs to Torch ``ModelConfig``.  Constructing this record is pure: it
    validates already-resolved runtime values and never observes hardware.
    """

    # Lightning Trainer runtime
    accelerator: Literal["cpu", "gpu", "cuda", "mps"] = "cpu"
    devices: int = 1
    strategy: str = "auto"
    deterministic: Union[bool, Literal["warn"]] = True
    precision: Literal["32-true", "16-mixed", "bf16-mixed"] = "32-true"

    # DataLoader runtime
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None

    # Checkpoint, logging, and progress runtime
    enable_progress_bar: bool = False
    enable_checkpointing: bool = True
    checkpoint_save_top_k: int = 1
    checkpoint_monitor_metric: str = "val_loss"
    checkpoint_mode: Literal["min", "max"] = "min"
    early_stop_patience: int = 100
    logger_backend: Optional[
        Literal["csv", "tensorboard", "mlflow"]
    ] = "csv"

    # Reconstruction logging runtime
    recon_log_every_n_epochs: Optional[int] = None
    recon_log_num_patches: int = 4
    recon_log_fixed_indices: Optional[List[int]] = None
    recon_log_stitch: bool = False
    recon_log_max_stitch_samples: Optional[int] = None

    # Inference runtime
    inference_batch_size: Optional[int] = None
    middle_trim: int = 0
    pad_eval: bool = False

    def __post_init__(self) -> None:
        """Validate resolved values without capability observation."""

        _validate_execution_resolved_environment_values(self.__dict__)
        _validate_execution_post_environment_values(self.__dict__)


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
    }
    if config.architecture not in valid_arches:
        raise ValueError(
            f"Invalid architecture '{config.architecture}'. "
            f"Expected one of {sorted(valid_arches)}."
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

# Key mappings from dataclass field names to legacy param names. Module-level
# (referenced as a module symbol by the module docstring and
# docs/specs/spec-ptycho-config-bridge.md); consumers import it instead of
# re-deriving the translation table.
KEY_MAPPINGS: Dict[str, str] = {
    'object_big': 'object.big',
    'probe_big': 'probe.big',
    'probe_mask': 'probe.mask',
    'probe_trainable': 'probe.trainable',
    'intensity_scale_trainable': 'intensity_scale.trainable',
    'positions_provided': 'positions.provided',
    'output_dir': 'output_prefix',
    'train_data_file': 'train_data_file_path',
    'test_data_file': 'test_data_file_path',
}

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
