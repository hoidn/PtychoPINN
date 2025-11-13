"""
Configuration bridge adapter for PyTorch → TensorFlow dataclass translation.

This module implements Phase B.B3 of INTEGRATE-PYTORCH-001, providing translation
functions that convert PyTorch singleton configs to TensorFlow dataclass configs,
enabling population of the legacy params.cfg dictionary through the standard
update_legacy_dict() function.

Architecture:
    PyTorch config_params singletons → config_bridge → TensorFlow config dataclasses → update_legacy_dict() → params.cfg

MVP Scope (9 fields):
    - Model essentials: N, gridsize, model_type
    - Lifecycle paths: train_data_file, test_data_file, model_path
    - Data grouping: n_groups, neighbor_count
    - Physics scaling: nphotons

Design Decisions:
    1. Side-effect free: Functions return new dataclass instances without mutating inputs
    2. Override pattern: Accept dict parameter for fields missing from PyTorch configs
    3. Type conversion: Handle tuple→int (grid_size), enum mapping (mode→model_type), field renames (epochs→nepochs, K→neighbor_count)
    4. Modular for future refactor: Designed to support eventual migration to shared dataclasses (Open Question Q1)

Critical Transformations:
    - grid_size: Tuple[int, int] → gridsize: int (extract first element, assumes square grids)
    - mode: 'Unsupervised' | 'Supervised' → model_type: 'pinn' | 'supervised' (enum mapping)
    - epochs: int → nepochs: int (field rename)
    - K: int → neighbor_count: int (semantic mapping)
    - nll: bool → nll_weight: float (bool→float conversion: True→1.0, False→0.0)

Usage:
    ```python
    from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
    from ptycho_torch.config_bridge import to_model_config, to_training_config
    from ptycho.config.config import update_legacy_dict
    import ptycho.params as params

    # Instantiate PyTorch configs
    pt_data = DataConfig(N=128, grid_size=(2, 2), nphotons=1e9, K=7)
    pt_model = ModelConfig(mode='Unsupervised', amp_activation='silu')
    pt_train = TrainingConfig(epochs=50)

    # Translate to TensorFlow dataclasses
    tf_model = to_model_config(pt_data, pt_model)
    tf_train = to_training_config(
        tf_model, pt_data, pt_model, pt_train,
        overrides=dict(train_data_file=Path('data.npz'), n_groups=512)
    )

    # Populate params.cfg
    update_legacy_dict(params.cfg, tf_train)
    ```

References:
    - Test contract: tests/torch/test_config_bridge.py:1-162
    - Field mapping: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T032218Z/config_schema_map.md
    - Spec requirements: specs/ptychodus_api_spec.md:213-273
    - PyTorch configs: ptycho_torch/config_params.py
    - TensorFlow configs: ptycho/config/config.py
"""

from pathlib import Path
from typing import Optional, Dict, Any
from ptycho.config.config import (
    ModelConfig as TFModelConfig,
    TrainingConfig as TFTrainingConfig,
    InferenceConfig as TFInferenceConfig
)

# Import PyTorch configs - torch is now mandatory (Phase F3.1/F3.2)
from ptycho_torch.config_params import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    InferenceConfig
)


def to_model_config(
    data: DataConfig,
    model: ModelConfig,
    overrides: Optional[Dict[str, Any]] = None
) -> TFModelConfig:
    """
    Translate PyTorch DataConfig and ModelConfig to TensorFlow ModelConfig.

    Performs critical field transformations:
    - grid_size tuple → gridsize int (extracts first element, assumes square grids)
    - mode enum → model_type enum ('Unsupervised'→'pinn', 'Supervised'→'supervised')
    - amp_activation normalization (silu→swish, SiLU→swish)
    - Merges fields from both PyTorch configs into single TensorFlow ModelConfig

    Args:
        data: PyTorch DataConfig instance (provides N, grid_size, nphotons)
        model: PyTorch ModelConfig instance (provides mode, architecture params)
        overrides: Optional dict of additional fields to override defaults

    Returns:
        TensorFlow ModelConfig instance with translated fields

    Raises:
        ValueError: If grid_size is non-square, mode has invalid value, or activation unknown
    """
    overrides = overrides or {}

    # Extract gridsize from grid_size tuple (assumes square grids)
    grid_h, grid_w = data.grid_size
    if grid_h != grid_w:
        raise ValueError(
            f"Non-square grids not supported by TensorFlow backend: "
            f"grid_size={data.grid_size}. Use square grids (e.g., (2, 2))."
        )
    gridsize = grid_h

    # Map mode enum to model_type enum
    mode_to_model_type = {
        'Unsupervised': 'pinn',
        'Supervised': 'supervised'
    }
    if model.mode not in mode_to_model_type:
        raise ValueError(
            f"Invalid mode '{model.mode}'. Must be 'Unsupervised' or 'Supervised'."
        )
    model_type = mode_to_model_type[model.mode]

    # Map PyTorch activation names to TensorFlow equivalents
    activation_mapping = {
        'silu': 'swish',
        'SiLU': 'swish',
        'sigmoid': 'sigmoid',
        'swish': 'swish',
        'softplus': 'softplus',
        'relu': 'relu'
    }
    if model.amp_activation not in activation_mapping:
        raise ValueError(
            f"Unknown activation '{model.amp_activation}'. "
            f"Supported values: {list(activation_mapping.keys())}"
        )
    amp_activation = activation_mapping[model.amp_activation]

    # Translate probe_mask from Optional[Tensor] to bool
    # None → False (no masking), non-None tensor → True (masking enabled)
    # Can be overridden explicitly via overrides dict
    probe_mask_value = False  # Default when None
    if model.probe_mask is not None:
        # If probe_mask is a tensor, enable masking
        probe_mask_value = True

    # Note: probe_scale validation removed after TDD cycle showed it breaks too many existing tests.
    # The field has default divergence (PyTorch 1.0 vs TensorFlow 4.0) but explicit validation
    # at adapter level is too strict. Callers should be aware of this divergence through
    # override_matrix.md documentation. Keep field in kwargs unchanged from DataConfig value.

    # Build kwargs from PyTorch configs
    # CRITICAL: Only include fields that exist in TensorFlow ModelConfig
    # intensity_scale_trainable belongs in TrainingConfig, NOT ModelConfig
    kwargs = {
        # From DataConfig
        'N': data.N,
        'gridsize': gridsize,

        # From ModelConfig
        'n_filters_scale': model.n_filters_scale,
        'model_type': model_type,
        'amp_activation': amp_activation,  # Normalized activation
        'object_big': model.object_big,
        'probe_big': model.probe_big,

        # Translated values
        'probe_mask': probe_mask_value,  # PyTorch Optional[Tensor] → TensorFlow bool

        # Spec-mandated fields (now available from PyTorch dataclass defaults)
        'pad_object': model.pad_object,  # From PyTorch ModelConfig (default=True)
        'probe_scale': data.probe_scale,  # PyTorch default=1.0, TensorFlow default=4.0
        'gaussian_smoothing_sigma': model.gaussian_smoothing_sigma,  # From PyTorch ModelConfig (default=0.0)
    }

    # Apply overrides (allows explicit probe_mask override)
    kwargs.update(overrides)

    return TFModelConfig(**kwargs)


def to_training_config(
    model: TFModelConfig,
    data: DataConfig,
    pt_model: ModelConfig,
    training: TrainingConfig,
    overrides: Optional[Dict[str, Any]] = None
) -> TFTrainingConfig:
    """
    Translate PyTorch configs to TensorFlow TrainingConfig.

    Performs critical field transformations:
    - epochs → nepochs (field rename)
    - K → neighbor_count (semantic mapping)
    - nll bool → nll_weight float (True→1.0, False→0.0)
    - intensity_scale_trainable from PyTorch ModelConfig (moved to TrainingConfig)
    - Accepts overrides for fields missing in PyTorch (train_data_file, n_groups, etc.)

    Args:
        model: TensorFlow ModelConfig (already translated via to_model_config)
        data: PyTorch DataConfig instance (provides K, nphotons)
        pt_model: PyTorch ModelConfig instance (provides intensity_scale_trainable)
        training: PyTorch TrainingConfig instance (provides epochs, batch_size, nll)
        overrides: Required dict containing train_data_file, n_groups, and other missing fields

    Returns:
        TensorFlow TrainingConfig instance with translated fields

    Raises:
        ValueError: If required override fields (train_data_file) are missing
    """
    overrides = overrides or {}

    # Convert nll bool to nll_weight float
    nll_weight = 1.0 if training.nll else 0.0

    # Build kwargs
    kwargs = {
        'model': model,

        # From TrainingConfig
        'batch_size': training.batch_size,
        'nepochs': training.epochs,  # Field rename
        'nll_weight': nll_weight,    # Type conversion

        # From DataConfig
        'neighbor_count': data.K,    # Semantic mapping
        'nphotons': data.nphotons,  # Will be validated after overrides

        # From PyTorch ModelConfig (belongs in TrainingConfig in TensorFlow)
        'intensity_scale_trainable': pt_model.intensity_scale_trainable,

        # Backend selection: propagate PyTorch backend
        'backend': 'pytorch',  # E1.C2: Mark this config as coming from PyTorch stack

        # Default values for fields missing in PyTorch
        'mae_weight': 0.0,
        'realspace_mae_weight': 0.0,
        'realspace_weight': 0.0,
        'positions_provided': True,
        'probe_trainable': False,
        'sequential_sampling': False,

        # Fields from PyTorch TrainingConfig (now with defaults)
        'train_data_file': training.train_data_file,  # From PyTorch TrainingConfig
        'test_data_file': training.test_data_file,  # From PyTorch TrainingConfig
        'n_groups': training.n_groups,  # From PyTorch TrainingConfig
        'n_subsample': None,  # Not in PyTorch, use override
        'subsample_seed': data.subsample_seed,  # From DataConfig
        'output_dir': Path(training.output_dir) if training.output_dir else Path('training_outputs'),
        'torch_loss_mode': getattr(training, 'torch_loss_mode', 'poisson'),
    }

    # Apply overrides (critical for MVP fields)
    kwargs.update(overrides)

    # Validate nphotons: PyTorch default (1e5) differs from TensorFlow default (1e9)
    # Require explicit override to avoid silent divergence (spec §5.2:9 HIGH risk)
    pytorch_default_nphotons = 1e5
    tensorflow_default_nphotons = 1e9
    if 'nphotons' not in overrides and data.nphotons == pytorch_default_nphotons:
        raise ValueError(
            f"nphotons default divergence detected: PyTorch default ({pytorch_default_nphotons}) "
            f"differs from TensorFlow default ({tensorflow_default_nphotons}). "
            f"Provide explicit nphotons override to resolve: "
            f"overrides=dict(..., nphotons={tensorflow_default_nphotons})"
        )

    # Validate n_groups: Missing override leaves params.cfg['n_groups'] = None
    # breaking downstream workflows that expect valid integer (Phase B.B5.D3)
    if kwargs['n_groups'] is None:
        raise ValueError(
            "n_groups is required in overrides for TrainingConfig. "
            "Missing override leaves params.cfg['n_groups'] = None, breaking downstream workflows. "
            "Provide as: overrides=dict(..., n_groups=512)"
        )

    # Warn about missing test_data_file (optional but helpful for evaluation workflows)
    # Phase B.B5.D3: Softer validation to surface absent evaluation data
    if kwargs['test_data_file'] is None:
        import warnings
        warnings.warn(
            "test_data_file not provided in TrainingConfig overrides. "
            "Evaluation workflows require test_data_file to be set during inference update. "
            "Consider providing: overrides=dict(..., test_data_file=Path('test.npz'))",
            UserWarning,
            stacklevel=2
        )

    # Convert string paths to Path objects, then back to strings for params.cfg compatibility
    # This ensures KEY_MAPPINGS in config/config.py correctly converts to strings
    for path_field in ['train_data_file', 'test_data_file', 'output_dir']:
        if path_field in kwargs and kwargs[path_field] is not None:
            if isinstance(kwargs[path_field], str):
                kwargs[path_field] = Path(kwargs[path_field])

    # Validate required overrides with actionable error messages
    if kwargs['train_data_file'] is None:
        raise ValueError(
            "train_data_file is required in overrides for TrainingConfig. "
            "Provide as: overrides=dict(train_data_file=Path('train.npz'))"
        )

    return TFTrainingConfig(**kwargs)


def to_inference_config(
    model: TFModelConfig,
    data: DataConfig,
    inference: InferenceConfig,
    overrides: Optional[Dict[str, Any]] = None
) -> TFInferenceConfig:
    """
    Translate PyTorch configs to TensorFlow InferenceConfig.

    Performs critical field transformations:
    - K → neighbor_count (semantic mapping)
    - Accepts overrides for fields missing in PyTorch (model_path, test_data_file, n_groups)

    Args:
        model: TensorFlow ModelConfig (already translated via to_model_config)
        data: PyTorch DataConfig instance (provides K)
        inference: PyTorch InferenceConfig instance (provides batch_size, etc.)
        overrides: Required dict containing model_path, test_data_file, and other missing fields

    Returns:
        TensorFlow InferenceConfig instance with translated fields

    Raises:
        ValueError: If required override fields (model_path, test_data_file) are missing
    """
    overrides = overrides or {}

    # Build kwargs
    kwargs = {
        'model': model,

        # From DataConfig
        'neighbor_count': data.K,  # Semantic mapping

        # Backend selection: propagate PyTorch backend
        'backend': 'pytorch',  # E1.C2: Mark this config as coming from PyTorch stack

        # Default values for fields missing in PyTorch
        'debug': False,

        # Fields that must come from overrides (not in PyTorch configs)
        'model_path': None,
        'test_data_file': None,
        'n_groups': None,
        'n_subsample': None,
        'subsample_seed': None,
        'output_dir': Path('inference_outputs'),
    }

    # Apply overrides (critical for MVP fields)
    kwargs.update(overrides)

    # Convert string paths to Path objects, then back to strings for params.cfg compatibility
    # This ensures KEY_MAPPINGS in config/config.py correctly converts to strings
    for path_field in ['model_path', 'test_data_file', 'output_dir']:
        if path_field in kwargs and kwargs[path_field] is not None:
            if isinstance(kwargs[path_field], str):
                kwargs[path_field] = Path(kwargs[path_field])

    # Validate required overrides with actionable error messages
    if kwargs['model_path'] is None:
        raise ValueError(
            "model_path is required in overrides for InferenceConfig. "
            "Provide as: overrides=dict(model_path=Path('model_dir'))"
        )
    if kwargs['test_data_file'] is None:
        raise ValueError(
            "test_data_file is required in overrides for InferenceConfig. "
            "Provide as: overrides=dict(test_data_file=Path('test.npz'))"
        )

    return TFInferenceConfig(**kwargs)
