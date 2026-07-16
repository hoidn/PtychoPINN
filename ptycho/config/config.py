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
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal, Union
import json
import hashlib
import inspect
import math
import tomllib
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


@dataclass(frozen=True)
class ProbeSimulationConfig:
    """Probe source and transforms that are baked into generated data.

    ``source_path=None`` is retained for direct APIs that supply an in-memory
    custom probe. File-based generation entry points require a path before
    invoking the simulator.
    """

    source: Literal["custom", "ideal"] = "custom"
    source_path: Path | None = None
    transform_pipeline: str = "pad_preserve:64"
    mask_diameter: float | None = None


@dataclass(frozen=True)
class SyntheticObjectConfig:
    """Synthetic object family and generation counts."""

    kind: Literal["lines", "dead_leaves", "natural_patch"] = "lines"
    image_size: tuple[int, int] = (392, 392)
    objects_per_probe: int = 4
    diffractions_per_object: int = 7000
    set_phi: bool = False


@dataclass(frozen=True)
class ScanSimulationConfig:
    """Scan layout and train/test geometry for generated data."""

    kind: Literal["grid", "nongrid"] = "grid"
    grid_size: tuple[int, int] = (1, 1)
    offset: int = 4
    outer_offset_train: int = 8
    outer_offset_test: int = 20
    train_groups: int = 2
    test_groups: int = 2
    buffer: int = 0


@dataclass(frozen=True)
class DetectorSimulationConfig:
    """Detector/noise properties baked into generated diffraction data."""

    photons_per_pattern: float = 1e9
    beamstop_diameter: float | None = None


@dataclass(frozen=True)
class SimulationConfig:
    """Canonical top-level recipe for generated ptychography data."""

    N: int = 64
    probe: ProbeSimulationConfig = field(default_factory=ProbeSimulationConfig)
    object: SyntheticObjectConfig = field(default_factory=SyntheticObjectConfig)
    scan: ScanSimulationConfig = field(default_factory=ScanSimulationConfig)
    detector: DetectorSimulationConfig = field(default_factory=DetectorSimulationConfig)
    seed: int | None = None


def _reject_unknown_mapping_keys(
    values: Mapping[str, Any],
    config_type: type,
    path: str,
) -> None:
    allowed = {item.name for item in fields(config_type)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            ", ".join(f"{path}.{name}" for name in unknown)
            + " is not a recognized simulation configuration field"
        )


def _pair_from_mapping(value: Any, path: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{path} must be a two-element dimension pair")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{path} dimensions must be integers")
    return (value[0], value[1])


def _section_from_mapping(values: Any, config_type: type, path: str):
    if isinstance(values, config_type):
        return values
    if not isinstance(values, Mapping):
        raise ValueError(f"{path} must be a mapping")
    _reject_unknown_mapping_keys(values, config_type, path)
    kwargs = dict(values)
    if config_type is ProbeSimulationConfig and kwargs.get("source_path") is not None:
        try:
            kwargs["source_path"] = Path(kwargs["source_path"])
        except TypeError as exc:
            raise ValueError(f"{path}.source_path must be a filesystem path") from exc
    if config_type is SyntheticObjectConfig and "image_size" in kwargs:
        kwargs["image_size"] = _pair_from_mapping(
            kwargs["image_size"], f"{path}.image_size"
        )
    if config_type is ScanSimulationConfig and "grid_size" in kwargs:
        kwargs["grid_size"] = _pair_from_mapping(
            kwargs["grid_size"], f"{path}.grid_size"
        )
    return config_type(**kwargs)


def simulation_config_from_mapping(values: Mapping[str, Any]) -> SimulationConfig:
    """Build and validate ``SimulationConfig`` from YAML/TOML-shaped values.

    Unknown keys are errors at every level. This prevents training fields from
    being silently accepted under the simulation namespace.
    """

    if not isinstance(values, Mapping):
        raise ValueError("simulation must be a mapping")
    _reject_unknown_mapping_keys(values, SimulationConfig, "simulation")
    kwargs = dict(values)
    for name, config_type in (
        ("probe", ProbeSimulationConfig),
        ("object", SyntheticObjectConfig),
        ("scan", ScanSimulationConfig),
        ("detector", DetectorSimulationConfig),
    ):
        if name in kwargs:
            kwargs[name] = _section_from_mapping(
                kwargs[name], config_type, f"simulation.{name}"
            )
    config = SimulationConfig(**kwargs)
    validate_simulation_config(config)
    return config


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
    if not isinstance(pipeline, str) or not pipeline.strip():
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


def _validate_square_positive_pair(value: Any, path: str) -> None:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{path} must be a two-element tuple")
    if any(not isinstance(item, int) or item <= 0 for item in value):
        raise ValueError(f"{path} dimensions must be positive integers")
    if value[0] != value[1]:
        raise ValueError(f"{path} must be square, got {value}")


def validate_simulation_config(config: SimulationConfig) -> None:
    """Validate one complete generated-data recipe."""

    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    if isinstance(config.N, bool) or not isinstance(config.N, int) or config.N <= 0:
        raise ValueError(f"simulation.N must be a positive integer, got {config.N}")
    if config.probe.source not in {"custom", "ideal"}:
        raise ValueError(
            f"simulation.probe.source must be 'custom' or 'ideal', got {config.probe.source!r}"
        )
    if config.probe.source == "ideal" and config.probe.source_path is not None:
        raise ValueError(
            "simulation.probe.source_path must be omitted when "
            "simulation.probe.source='ideal'"
        )
    if config.probe.source_path is not None and not isinstance(
        config.probe.source_path, Path
    ):
        raise ValueError("simulation.probe.source_path must be a Path when set")
    if config.probe.mask_diameter is not None:
        if isinstance(config.probe.mask_diameter, bool) or not isinstance(
            config.probe.mask_diameter, (int, float)
        ):
            raise ValueError(
                "simulation.probe.mask_diameter must be a finite positive number"
            )
        if (
            not math.isfinite(config.probe.mask_diameter)
            or config.probe.mask_diameter <= 0
        ):
            raise ValueError("simulation.probe.mask_diameter must be finite and positive")
    final_size = _pipeline_final_size(config.probe.transform_pipeline)
    if final_size != config.N:
        raise ValueError(
            "simulation.probe.transform_pipeline final size "
            f"{final_size} does not match simulation.N {config.N}"
        )

    if config.object.kind not in {"lines", "dead_leaves", "natural_patch"}:
        raise ValueError(f"Unsupported simulation.object.kind {config.object.kind!r}")
    _validate_square_positive_pair(
        config.object.image_size, "simulation.object.image_size"
    )
    if (
        isinstance(config.object.objects_per_probe, bool)
        or not isinstance(config.object.objects_per_probe, int)
        or config.object.objects_per_probe <= 0
    ):
        raise ValueError("simulation.object.objects_per_probe must be positive")
    if (
        isinstance(config.object.diffractions_per_object, bool)
        or not isinstance(config.object.diffractions_per_object, int)
        or config.object.diffractions_per_object <= 0
    ):
        raise ValueError("simulation.object.diffractions_per_object must be positive")

    if config.scan.kind not in {"grid", "nongrid"}:
        raise ValueError(f"Unsupported simulation.scan.kind {config.scan.kind!r}")
    _validate_square_positive_pair(config.scan.grid_size, "simulation.scan.grid_size")
    for name in ("offset", "outer_offset_train", "outer_offset_test", "buffer"):
        value = getattr(config.scan, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"simulation.scan.{name} must be a non-negative integer")
    for name in ("train_groups", "test_groups"):
        value = getattr(config.scan, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"simulation.scan.{name} must be a positive integer")

    if isinstance(config.detector.photons_per_pattern, bool) or not isinstance(
        config.detector.photons_per_pattern, (int, float)
    ):
        raise ValueError(
            "simulation.detector.photons_per_pattern must be a finite positive number"
        )
    photons = float(config.detector.photons_per_pattern)
    if not math.isfinite(photons) or photons <= 0:
        raise ValueError(
            "simulation.detector.photons_per_pattern must be finite and positive"
        )
    if config.detector.beamstop_diameter is not None:
        if isinstance(config.detector.beamstop_diameter, bool) or not isinstance(
            config.detector.beamstop_diameter, (int, float)
        ):
            raise ValueError(
                "simulation.detector.beamstop_diameter must be a finite positive number"
            )
        if (
            not math.isfinite(config.detector.beamstop_diameter)
            or config.detector.beamstop_diameter <= 0
        ):
            raise ValueError(
                "simulation.detector.beamstop_diameter must be finite and positive"
            )
    if config.seed is not None and (
        isinstance(config.seed, bool) or not isinstance(config.seed, int)
    ):
        raise ValueError("simulation.seed must be an integer or None")


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
        'cnn', 'ffno', 'fno', 'fno_vanilla', 'neuralop_uno'
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
    object_big: bool = True
    probe_big: bool = True  # Changed default
    probe_mask: bool = False  # Changed default
    probe_mask_sigma: float = 1.0
    probe_mask_diameter: Optional[float] = None
    pad_object: bool = True
    probe_scale: float = 4.
    gaussian_smoothing_sigma: float = 0.0

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
    'spectral_bottleneck_blocks',
    'spectral_bottleneck_modes',
    'spectral_bottleneck_share_weights',
    'spectral_bottleneck_gate_init',
    'spectral_bottleneck_gate_mode',
})


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
        if (
            isinstance(self.devices, bool)
            or not (
                (isinstance(self.devices, int) and self.devices > 0)
                or self.devices == "auto"
            )
        ):
            raise ValueError(
                f"devices must be a positive integer or 'auto', got {self.devices!r}"
            )

        valid_precisions = {"32-true", "16-mixed", "bf16-mixed"}
        if not isinstance(self.precision, str) or self.precision not in valid_precisions:
            raise ValueError(
                f"Invalid precision: {self.precision!r}. "
                f"Expected one of {sorted(valid_precisions)}."
            )

        if self.accelerator == 'tpu':
            raise ValueError(
                "Torch-XLA TPU execution is unsupported by this PyTorch runtime. "
                "Use accelerator='cpu', 'gpu'/'cuda', or 'mps'."
            )

        # Accelerator whitelist supported by this runtime.
        valid_accelerators = {'auto', 'cpu', 'gpu', 'cuda', 'mps'}
        if self.accelerator not in valid_accelerators:
            raise ValueError(
                f"Invalid accelerator: '{self.accelerator}'. "
                f"Expected one of {sorted(valid_accelerators)}."
            )

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

        # Non-negative workers
        if self.num_workers < 0:
            raise ValueError(
                f"num_workers must be non-negative, got {self.num_workers}"
            )

        # Positive learning rate
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )

        # Positive inference batch size (if provided)
        if self.inference_batch_size is not None and self.inference_batch_size <= 0:
            raise ValueError(
                f"inference_batch_size must be positive, got {self.inference_batch_size}"
            )

        # Positive accumulation steps
        if self.accum_steps <= 0:
            raise ValueError(
                f"accum_steps must be positive, got {self.accum_steps}"
            )

        # Non-negative checkpoint save count
        if self.checkpoint_save_top_k < 0:
            raise ValueError(
                f"checkpoint_save_top_k must be non-negative, got {self.checkpoint_save_top_k}"
            )

        # Positive early stopping patience
        if self.early_stop_patience <= 0:
            raise ValueError(
                f"early_stop_patience must be positive, got {self.early_stop_patience}"
            )

        # Checkpoint mode whitelist
        valid_checkpoint_modes = {'min', 'max'}
        if self.checkpoint_mode not in valid_checkpoint_modes:
            raise ValueError(
                f"Invalid checkpoint_mode: '{self.checkpoint_mode}'. "
                f"Expected one of {sorted(valid_checkpoint_modes)}."
            )

        if self.spectral_bottleneck_blocks <= 0:
            raise ValueError(
                f"spectral_bottleneck_blocks must be positive, got {self.spectral_bottleneck_blocks}."
            )
        if self.spectral_bottleneck_modes <= 0:
            raise ValueError(
                f"spectral_bottleneck_modes must be positive, got {self.spectral_bottleneck_modes}."
            )
        if not math.isfinite(float(self.spectral_bottleneck_gate_init)):
            raise ValueError(
                "spectral_bottleneck_gate_init must be finite, "
                f"got {self.spectral_bottleneck_gate_init}."
            )
        valid_gate_modes = {'shared', 'per_block'}
        if self.spectral_bottleneck_gate_mode not in valid_gate_modes:
            raise ValueError(
                f"Invalid spectral_bottleneck_gate_mode '{self.spectral_bottleneck_gate_mode}'. "
                f"Expected one of {sorted(valid_gate_modes)}."
            )

        if self.ffno_encoder_blocks <= 0:
            raise ValueError(
                f"ffno_encoder_blocks must be positive, got {self.ffno_encoder_blocks}."
            )
        if self.ffno_encoder_modes <= 0:
            raise ValueError(
                f"ffno_encoder_modes must be positive, got {self.ffno_encoder_modes}."
            )
        if (
            not math.isfinite(float(self.ffno_encoder_gate_init))
            or float(self.ffno_encoder_gate_init) <= 0.0
        ):
            raise ValueError(
                "ffno_encoder_gate_init must be finite and > 0, "
                f"got {self.ffno_encoder_gate_init}."
            )
        if (
            not math.isfinite(float(self.ffno_encoder_mlp_ratio))
            or float(self.ffno_encoder_mlp_ratio) <= 0.0
        ):
            raise ValueError(
                "ffno_encoder_mlp_ratio must be finite and > 0, "
                f"got {self.ffno_encoder_mlp_ratio}."
            )


_execution_init_signature = inspect.signature(PyTorchExecutionConfig.__init__)
PyTorchExecutionConfig.__signature__ = _execution_init_signature.replace(
    parameters=tuple(_execution_init_signature.parameters.values())[1:]
)


def validate_model_config(config: ModelConfig) -> None:
    """Validate model configuration."""
    valid_arches = {
        'cnn',
        'ffno',
        'fno',
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
