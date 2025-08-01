from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Literal
import yaml

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

@dataclass(frozen=True)
class TrainingConfig:
    """Training specific configuration."""
    model: ModelConfig
    train_data_file: Path  # Added
    test_data_file: Optional[Path] = None  # Added
    batch_size: int = 16
    nepochs: int = 50
    mae_weight: float = 0.0
    nll_weight: float = 1.0
    realspace_mae_weight: float = 0.0
    realspace_weight: float = 0.0
    nphotons: float = 1e9
    positions_provided: bool = True  
    probe_trainable: bool = False
    intensity_scale_trainable: bool = True  # Changed default
    output_dir: Path = Path("training_outputs")

@dataclass(frozen=True)
class InferenceConfig:
    """Inference specific configuration."""
    model: ModelConfig
    model_path: Path
    test_data_file: Path
    debug: bool = False
    output_dir: Path = Path("inference_outputs")

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
    if not config.model_path.exists():
        raise ValueError(f"model_path does not exist: {config.model_path}")

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
        'output_dir': 'output_prefix'
    }

    # Convert dataclass to dict
    d = asdict(obj)

    # Handle nested ModelConfig
    if 'model' in d:
        model_dict = d.pop('model')
        d.update(model_dict)

    # Apply key mappings
    for old_key, new_key in KEY_MAPPINGS.items():
        if old_key in d:
            d[new_key] = d.pop(old_key)

    # Convert Path to string
    if 'output_dir' in d:
        d['output_prefix'] = str(d.pop('output_dir'))

    return d

def update_legacy_dict(cfg: Dict[str, Any], dataclass_obj: Any) -> None:
    """Update legacy dictionary with dataclass values.
    
    Updates all values from the dataclass, adding new keys if needed.
    
    Args:
        cfg: Legacy dictionary to update
        dataclass_obj: Dataclass instance containing new values
    """
    new_values = dataclass_to_legacy_dict(dataclass_obj)
    
    # Update all values from dataclass
    cfg.update(new_values)
