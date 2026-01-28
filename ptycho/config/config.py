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

# Export list for public API (ADR-003 Phase C3.A1)
# Restores exports removed during Phase C2; ensures PyTorchExecutionConfig is discoverable
__all__ = [
    # Dataclass configurations
    'ModelConfig',
    'TrainingConfig',
    'InferenceConfig',
    'PyTorchExecutionConfig',
    # Core compatibility bridge
    'update_legacy_dict',
    # Validation functions
    'validate_model_config',
    'validate_training_config',
    'validate_inference_config',
    # YAML loading
    'load_yaml_config',
    # Internal translation (exposed for advanced use)
    'dataclass_to_legacy_dict',
]

@dataclass(frozen=True)
class ModelConfig:
    """Core model architecture parameters."""
    N: Literal[64, 128, 256] = 64
    gridsize: int = 1
    n_filters_scale: int = 2
    model_type: Literal['pinn', 'supervised'] = 'pinn'
    architecture: Literal['cnn', 'fno', 'hybrid'] = 'cnn'
    fno_modes: int = 12
    fno_width: int = 32
    fno_blocks: int = 4
    fno_cnn_blocks: int = 2
    fno_input_transform: Literal['none', 'sqrt', 'log1p', 'instancenorm'] = 'none'
    generator_output_mode: Literal['real_imag', 'amp_phase_logits', 'amp_phase'] = 'real_imag'
    amp_activation: Literal['sigmoid', 'swish', 'softplus', 'relu'] = 'sigmoid'
    object_big: bool = True
    probe_big: bool = True  # Changed default
    probe_mask: bool = False  # Changed default
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
    gradient_clip_val: Optional[float] = None  # Gradient clipping threshold (None = disabled)
    gradient_clip_algorithm: Literal['norm', 'value', 'agc'] = 'norm'  # Gradient clipping algorithm: norm, value, or agc

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
    accelerator: str = 'auto'  # Options: 'cpu', 'gpu', 'tpu', 'mps', 'auto' (default 'auto' → 'cuda' if available, else 'cpu')
    strategy: str = 'auto'  # Options: 'auto', 'ddp', 'fsdp', 'deepspeed'
    deterministic: bool = True  # Enforce reproducibility (seed_everything + deterministic mode)
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

    # Inference-specific knobs
    inference_batch_size: Optional[int] = None  # Override batch_size for inference (None = use training batch_size)
    middle_trim: int = 0  # Inference trimming parameter (not yet implemented)
    pad_eval: bool = False  # Padding for evaluation (not yet implemented)

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
            1. accelerator must be in whitelist {'auto', 'cpu', 'gpu', 'cuda', 'tpu', 'mps'}
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
        # Accelerator whitelist (Lightning supported values)
        valid_accelerators = {'auto', 'cpu', 'gpu', 'cuda', 'tpu', 'mps'}
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


def validate_model_config(config: ModelConfig) -> None:
    """Validate model configuration."""
    valid_arches = {'cnn', 'fno', 'hybrid'}
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
