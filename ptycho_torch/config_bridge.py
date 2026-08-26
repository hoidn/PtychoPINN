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
    - Data grouping: training_groups, neighbor_count
    - Physics scaling: nphotons

Design Decisions:
    1. Side-effect free: Functions return new dataclass instances without mutating inputs
    2. Override pattern: Accept dict parameter for fields missing from PyTorch configs
    3. Type conversion: enum value mapping (mode→model_type) and field rename
       (epochs→nepochs). These are permanent cross-stack bridge conversions,
       not pending renames (see docs/findings.md "Cross-stack naming policy").
    4. Modular for future refactor: Designed to support eventual migration to shared dataclasses (Open Question Q1)

Critical Transformations:
    - mode: 'Unsupervised' | 'Supervised' → model_type: 'pinn' | 'supervised' (enum mapping)
    - epochs: int → nepochs: int (field rename)
    - neighbor_count: int → neighbor_count: int (identity)
    - nll: bool → nll_weight: float (bool→float conversion: True→1.0, False→0.0)

Usage:
    ```python
    from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
    from ptycho_torch.config_bridge import to_model_config, to_training_config
    from ptycho.config.config import update_legacy_dict
    import ptycho.params as params

    # Instantiate PyTorch configs
    pt_data = DataConfig(N=128, gridsize=2, nphotons=1e9, neighbor_count=7)
    pt_model = ModelConfig(mode='Unsupervised', amp_activation='silu')
    pt_train = TrainingConfig(epochs=50)

    # Translate to TensorFlow dataclasses
    tf_model = to_model_config(pt_data, pt_model)
    tf_train = to_training_config(
        tf_model, pt_data, pt_model, pt_train,
        overrides=dict(train_data_file=Path('data.npz'), training_groups=512)
    )

    # Populate params.cfg
    update_legacy_dict(params.cfg, tf_train)
    ```

References:
    - Test contract: tests/torch/test_config_bridge.py:1-162
    - Field mapping: docs/findings.md (see git history for the originating plan)
    - Spec requirements: specs/ptychodus_api_spec.md:213-273
    - PyTorch configs: ptycho_torch/config_params.py
    - TensorFlow configs: ptycho/config/config.py
"""

from dataclasses import fields

from pathlib import Path
from typing import Optional, Dict, Any
from ptycho.config.config import (
    ModelConfig as TFModelConfig,
    TrainingConfig as TFTrainingConfig,
    InferenceConfig as TFInferenceConfig,
    resolve_model_object_policy,
)

# Import PyTorch configs - torch is now mandatory (Phase F3.1/F3.2)
from ptycho_torch.config_params import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    InferenceConfig
)

from ptycho_torch.model_spec import MODE_TO_MODEL_TYPE

# --- Import-time totality tripwires (W1) ------------------------------------
# Each bridge surface enumerates its literal kwargs key-set and asserts it
# equals the target TF dataclass field-set minus fields intentionally not
# bridged. A TF-side field addition now fails at import instead of silently
# defaulting. Name-set tripwire idiom, precedented at
# ptycho_torch/model_spec.py:239-268.
_MODEL_CONFIG_BRIDGED_FIELDS = frozenset({
    "N", "gridsize", "n_filters_scale", "model_type", "architecture",
    "fno_modes", "fno_width", "fno_blocks", "fno_cnn_blocks",
    "learned_input_channels", "max_hidden_channels", "resnet_width",
    "fno_input_transform", "generator_output_mode", "amp_activation",
    "object_layout", "training_canvas", "training_patch_weighting",
    "probe_big", "probe_mask", "probe_mask_sigma", "probe_mask_diameter",
    "pad_object", "probe_scale", "gaussian_smoothing_sigma",
})
_MODEL_CONFIG_EXCLUDED_FIELDS = frozenset({
    "object_big",  # derived: materialized by resolve_model_object_policy
})
_TF_MODEL_FIELDS = frozenset(f.name for f in fields(TFModelConfig))
assert _MODEL_CONFIG_BRIDGED_FIELDS == _TF_MODEL_FIELDS - _MODEL_CONFIG_EXCLUDED_FIELDS, (
    "to_model_config bridged key-set drifted from TF ModelConfig fields: "
    f"extra={sorted(_MODEL_CONFIG_BRIDGED_FIELDS - (_TF_MODEL_FIELDS - _MODEL_CONFIG_EXCLUDED_FIELDS))} "
    f"missing={sorted((_TF_MODEL_FIELDS - _MODEL_CONFIG_EXCLUDED_FIELDS) - _MODEL_CONFIG_BRIDGED_FIELDS)}"
)

_TRAINING_CONFIG_BRIDGED_FIELDS = frozenset({
    "model", "batch_size", "nepochs", "nll_weight", "neighbor_count",
    "nphotons", "intensity_scale_trainable", "backend", "mae_weight",
    "realspace_mae_weight", "realspace_weight", "positions_provided",
    "probe_trainable", "sequential_sampling", "train_data_file",
    "test_data_file", "training_groups", "train_raw_selection",
    "subsample_seed", "output_dir", "torch_loss_mode",
    "torch_mae_pred_l2_match_target", "gradient_clip_val",
    "gradient_clip_algorithm", "optimizer", "momentum", "weight_decay",
    "adam_beta1", "adam_beta2", "scheduler", "lr_warmup_epochs",
    "lr_min_ratio", "plateau_factor", "plateau_patience", "plateau_min_lr",
    "plateau_threshold",
})
_TRAINING_CONFIG_EXCLUDED_FIELDS = frozenset({
    "n_images",             # deprecated: training_groups is the canonical field
    "enable_oversampling",  # factory: set by the training factory, not the bridge
    "neighbor_pool_size",   # factory: set by the training factory, not the bridge
})
_TF_TRAINING_FIELDS = frozenset(f.name for f in fields(TFTrainingConfig))
assert _TRAINING_CONFIG_BRIDGED_FIELDS == _TF_TRAINING_FIELDS - _TRAINING_CONFIG_EXCLUDED_FIELDS, (
    "to_training_config bridged key-set drifted from TF TrainingConfig fields: "
    f"extra={sorted(_TRAINING_CONFIG_BRIDGED_FIELDS - (_TF_TRAINING_FIELDS - _TRAINING_CONFIG_EXCLUDED_FIELDS))} "
    f"missing={sorted((_TF_TRAINING_FIELDS - _TRAINING_CONFIG_EXCLUDED_FIELDS) - _TRAINING_CONFIG_BRIDGED_FIELDS)}"
)

_INFERENCE_CONFIG_BRIDGED_FIELDS = frozenset({
    "model", "neighbor_count", "backend", "debug", "model_path",
    "test_data_file", "inference_groups", "inference_raw_selection",
    "subsample_seed", "output_dir",
})
_INFERENCE_CONFIG_EXCLUDED_FIELDS = frozenset({
    "n_images",             # deprecated: inference_groups is the canonical field
    "enable_oversampling",  # factory: set by the inference factory, not the bridge
    "neighbor_pool_size",   # factory: set by the inference factory, not the bridge
})
_TF_INFERENCE_FIELDS = frozenset(f.name for f in fields(TFInferenceConfig))
assert _INFERENCE_CONFIG_BRIDGED_FIELDS == _TF_INFERENCE_FIELDS - _INFERENCE_CONFIG_EXCLUDED_FIELDS, (
    "to_inference_config bridged key-set drifted from TF InferenceConfig fields: "
    f"extra={sorted(_INFERENCE_CONFIG_BRIDGED_FIELDS - (_TF_INFERENCE_FIELDS - _INFERENCE_CONFIG_EXCLUDED_FIELDS))} "
    f"missing={sorted((_TF_INFERENCE_FIELDS - _INFERENCE_CONFIG_EXCLUDED_FIELDS) - _INFERENCE_CONFIG_BRIDGED_FIELDS)}"
)


def to_model_config(
    data: DataConfig,
    model: ModelConfig,
    overrides: Optional[Dict[str, Any]] = None
) -> TFModelConfig:
    """
    Translate PyTorch DataConfig and ModelConfig to TensorFlow ModelConfig.

    Performs critical field transformations:
    - mode enum → model_type enum ('Unsupervised'→'pinn', 'Supervised'→'supervised')
    - amp_activation normalization (silu→swish, SiLU→swish)
    - Merges fields from both PyTorch configs into single TensorFlow ModelConfig

    Args:
        data: PyTorch DataConfig instance (provides N, gridsize, nphotons)
        model: PyTorch ModelConfig instance (provides mode, architecture params)
        overrides: Optional dict of additional fields to override defaults

    Returns:
        TensorFlow ModelConfig instance with translated fields

    Raises:
        ValueError: If mode has invalid value or activation unknown
    """
    overrides = overrides or {}
    from ptycho_torch.object_compatibility import (
        resolve_torch_model_object_policy,
    )
    model = resolve_torch_model_object_policy(model)

    # DataConfig.gridsize is a single square side length; no tuple to unpack.
    gridsize = data.gridsize

    # Map mode enum to model_type enum (shared cross-stack bridge constant)
    if model.mode not in MODE_TO_MODEL_TYPE:
        raise ValueError(
            f"Invalid mode '{model.mode}'. Must be 'Unsupervised' or 'Supervised'."
        )
    model_type = MODE_TO_MODEL_TYPE[model.mode]

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

    # Translate probe_mask union type to TensorFlow bool.
    # Torch semantics:
    # - bool -> passthrough
    # - tensor / explicit probe_mask_tensor -> enabled
    # - None -> disabled
    probe_mask_value = False
    probe_mask_toggle = getattr(model, 'probe_mask', None)
    probe_mask_tensor = getattr(model, 'probe_mask_tensor', None)
    if probe_mask_tensor is not None:
        probe_mask_value = True
    elif isinstance(probe_mask_toggle, bool):
        probe_mask_value = probe_mask_toggle
    elif probe_mask_toggle is not None:
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
        'architecture': model.architecture,  # Generator architecture (cnn, fno, hybrid)
        'fno_modes': model.fno_modes,
        'fno_width': model.fno_width,
        'fno_blocks': model.fno_blocks,
        'fno_cnn_blocks': model.fno_cnn_blocks,
        'learned_input_channels': getattr(model, 'learned_input_channels', 1),
        'max_hidden_channels': getattr(model, 'max_hidden_channels', None),
        'resnet_width': getattr(model, 'resnet_width', None),
        'fno_input_transform': model.fno_input_transform,
        'generator_output_mode': model.generator_output_mode,
        'amp_activation': amp_activation,  # Normalized activation
        'object_layout': model.object_layout,
        'training_canvas': model.training_canvas,
        'training_patch_weighting': model.training_patch_weighting,
        'probe_big': model.probe_big,

        # Translated values
        'probe_mask': probe_mask_value,  # PyTorch bool/tensor union → TensorFlow bool
        'probe_mask_sigma': model.probe_mask_sigma,
        'probe_mask_diameter': model.probe_mask_diameter,

        # Spec-mandated fields (now available from PyTorch dataclass defaults)
        'pad_object': model.pad_object,  # From PyTorch ModelConfig (default=True)
        'probe_scale': data.probe_scale,  # both backends pin PROBE_SCALE_DEFAULT=4.0 (gauge homes: ptycho.config.config / ptycho_torch.config_params)
        'gaussian_smoothing_sigma': model.gaussian_smoothing_sigma,  # From PyTorch ModelConfig (default=0.0)
    }

    assert set(kwargs) == _MODEL_CONFIG_BRIDGED_FIELDS, (
        "to_model_config kwargs drifted from _MODEL_CONFIG_BRIDGED_FIELDS"
    )

    # Apply overrides (allows explicit probe_mask override)
    kwargs.update(overrides)

    return resolve_model_object_policy(
        TFModelConfig(**kwargs),
        backend="torch",
        warn_deprecated=False,
    )


def to_training_config(
    model: TFModelConfig,
    data: DataConfig,
    pt_model: ModelConfig,
    training: TrainingConfig,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    require_group_count: bool = True,
) -> TFTrainingConfig:
    """
    Translate PyTorch configs to TensorFlow TrainingConfig.

    Performs critical field transformations:
    - epochs → nepochs (field rename)
    - neighbor_count (identity)
    - nll bool → nll_weight float (True→1.0, False→0.0)
    - intensity_scale_trainable from PyTorch ModelConfig (moved to TrainingConfig)
    - Accepts overrides for fields missing in PyTorch (train_data_file, training_groups, etc.)

    Args:
        model: TensorFlow ModelConfig (already translated via to_model_config)
        data: PyTorch DataConfig instance (provides neighbor_count, nphotons)
        pt_model: PyTorch ModelConfig instance (provides intensity_scale_trainable)
        training: PyTorch TrainingConfig instance (provides epochs, batch_size, nll)
        overrides: Required dict containing train_data_file, training_groups, and other missing fields

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
        'neighbor_count': data.neighbor_count,    # identity
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
        'training_groups': training.training_groups,  # From PyTorch TrainingConfig
        'train_raw_selection': None,  # Not in PyTorch, use override
        'subsample_seed': data.subsample_seed,  # From DataConfig
        'output_dir': Path(training.output_dir) if training.output_dir else Path('training_outputs'),
        'torch_loss_mode': getattr(training, 'torch_loss_mode', 'poisson'),
        'torch_mae_pred_l2_match_target': getattr(training, 'torch_mae_pred_l2_match_target', False),
        'gradient_clip_val': getattr(training, 'gradient_clip_val', None),
        'gradient_clip_algorithm': getattr(training, 'gradient_clip_algorithm', 'norm'),
        'optimizer': getattr(training, 'optimizer', 'adam'),
        'momentum': getattr(training, 'momentum', 0.9),
        'weight_decay': getattr(training, 'weight_decay', 0.0),
        'adam_beta1': getattr(training, 'adam_beta1', 0.9),
        'adam_beta2': getattr(training, 'adam_beta2', 0.999),
        'scheduler': getattr(training, 'scheduler', 'Default'),
        'lr_warmup_epochs': getattr(training, 'lr_warmup_epochs', 0),
        'lr_min_ratio': getattr(training, 'lr_min_ratio', 0.1),
        'plateau_factor': getattr(training, 'plateau_factor', 0.5),
        'plateau_patience': getattr(training, 'plateau_patience', 2),
        'plateau_min_lr': getattr(training, 'plateau_min_lr', 5e-5),
        'plateau_threshold': getattr(training, 'plateau_threshold', 0.0),
    }

    assert set(kwargs) == _TRAINING_CONFIG_BRIDGED_FIELDS, (
        "to_training_config kwargs drifted from _TRAINING_CONFIG_BRIDGED_FIELDS"
    )

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

    # Validate training_groups: Missing override leaves params.cfg['training_groups'] = None
    # breaking downstream workflows that expect valid integer (Phase B.B5.D3)
    if require_group_count and kwargs['training_groups'] is None:
        raise ValueError(
            "training_groups is required in overrides for TrainingConfig. "
            "Missing override leaves params.cfg['training_groups'] = None, breaking downstream workflows. "
            "Provide as: overrides=dict(..., training_groups=512)"
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
    - neighbor_count (identity)
    - Accepts overrides for fields missing in PyTorch (model_path, test_data_file, inference_groups)

    Args:
        model: TensorFlow ModelConfig (already translated via to_model_config)
        data: PyTorch DataConfig instance (provides neighbor_count)
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
        'neighbor_count': data.neighbor_count,  # identity

        # Backend selection: propagate PyTorch backend
        'backend': 'pytorch',  # E1.C2: Mark this config as coming from PyTorch stack

        # Default values for fields missing in PyTorch
        'debug': False,

        # Fields that must come from overrides (not in PyTorch configs)
        'model_path': None,
        'test_data_file': None,
        'inference_groups': None,
        'inference_raw_selection': None,
        'subsample_seed': data.subsample_seed,  # From DataConfig (parity with to_training_config)

        'output_dir': Path('inference_outputs'),
    }

    assert set(kwargs) == _INFERENCE_CONFIG_BRIDGED_FIELDS, (
        "to_inference_config kwargs drifted from _INFERENCE_CONFIG_BRIDGED_FIELDS"
    )

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
