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

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Literal
import yaml
import warnings

__all__ = [
    'ModelConfig',
    'PyTorchExecutionConfig',
    'TrainingConfig',
    'InferenceConfig',
    'validate_model_config',
    'validate_training_config',
    'validate_inference_config',
    'load_yaml_config',
    'dataclass_to_legacy_dict',
    'update_legacy_dict',
]

@dataclass(frozen=True)
class ModelConfig:
    """Core model architecture parameters."""
    N: Literal[64, 128, 256] = 64
    gridsize: int = 1
    n_filters_scale: int = 2
    model_type: Literal['pinn', 'supervised'] = 'pinn'
    amp_activation: Literal['sigmoid', 'swish', 'softplus', 'relu'] = 'sigmoid'
    object_big: bool = True
    probe_big: bool = True  # Changed default
    probe_mask: bool = False  # Changed default
    pad_object: bool = True
    probe_scale: float = 4.
    gaussian_smoothing_sigma: float = 0.0

@dataclass
class PyTorchExecutionConfig:
    """
    PyTorch-specific execution parameters (ADR-003 Phase C).

    These knobs control PyTorch backend runtime behavior: hardware selection,
    optimization settings, logging, and checkpointing. They do NOT affect model
    topology or data pipeline (use ModelConfig, TrainingConfig for those).

    Execution config is orthogonal to the legacy params.cfg system (CONFIG-001).
    It does not populate params.cfg and is consumed directly by PyTorch Lightning
    Trainer and DataLoader instantiation in ptycho_torch.workflows.components.

    Runtime Requirements:
        - PyTorch >= 2.2 (POLICY-001: mandatory dependency for PyTorch backend)
        - See docs/workflows/pytorch.md §§5-13 for usage examples
        - See specs/ptychodus_api_spec.md §4.8 for backend selection contract

    Field Groups:
        Hardware & Distributed: accelerator, strategy, n_devices, deterministic
        Data Loading: num_workers, pin_memory, persistent_workers, prefetch_factor
        Optimization: learning_rate, scheduler, gradient_clip_val, accum_steps
        Checkpointing: enable_checkpointing, checkpoint_save_top_k, checkpoint_monitor_metric, early_stop_patience
        Logging: enable_progress_bar, logger_backend, disable_mlflow
        Inference: inference_batch_size, middle_trim, pad_eval

    References:
        - docs/findings.md POLICY-001 (PyTorch mandatory), CONFIG-001 (params.cfg sync)
        - plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/override_matrix.md §5
        - plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/factory_design.md §2.2
    """

    # Hardware & Distributed Training
    accelerator: str = 'auto'  # cpu/gpu/tpu/mps/auto
    strategy: str = 'auto'  # auto/ddp/fsdp/deepspeed
    n_devices: int = 1
    deterministic: bool = True  # Reproducibility flag

    # Data Loading
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None

    # Optimization
    learning_rate: float = 1e-3
    scheduler: str = 'Default'  # Default/Exponential/MultiStage/Adaptive
    gradient_clip_val: Optional[float] = None
    accum_steps: int = 1

    # Checkpointing & Early Stopping
    enable_checkpointing: bool = True
    checkpoint_save_top_k: int = 1
    checkpoint_monitor_metric: str = 'val_loss'
    early_stop_patience: int = 100

    # Logging & Experiment Tracking
    enable_progress_bar: bool = False  # Controlled by config.debug
    logger_backend: Optional[str] = None  # tensorboard/wandb/mlflow
    disable_mlflow: bool = False

    # Inference-Specific
    inference_batch_size: Optional[int] = None
    middle_trim: int = 0
    pad_eval: bool = False

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
    positions_provided: bool = True
    probe_trainable: bool = False
    intensity_scale_trainable: bool = True  # Changed default
    output_dir: Path = Path("training_outputs")
    sequential_sampling: bool = False  # Use sequential sampling instead of random
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

def validate_model_config(config: ModelConfig) -> None:
    """Validate model configuration."""
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
