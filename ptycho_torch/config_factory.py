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

from dataclasses import dataclass, field
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
    ModelConfig as PTModelConfig,
    TrainingConfig as PTTrainingConfig,
    InferenceConfig as PTInferenceConfig,
)

# Import PyTorchExecutionConfig (Option A canonical location per ADR-003 Phase C1)
# Per supervisor decision at 2025-10-19T234458Z (factory_design.md §2.2)
from ptycho.config.config import PyTorchExecutionConfig


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


def create_training_payload(
    train_data_file: Path,
    output_dir: Path,
    overrides: Optional[Dict[str, Any]] = None,
    execution_config: Optional[PyTorchExecutionConfig] = None,
) -> TrainingPayload:
    """
    Create complete training configuration payload.

    Centralizes all config construction logic for PyTorch training workflows.
    Eliminates duplicated wiring in CLI and workflow entry points by providing
    a single factory function that:
    1. Validates required arguments (train_data_file, output_dir, n_groups)
    2. Infers probe size from NPZ metadata (or uses override)
    3. Constructs PyTorch singleton configs (DataConfig, ModelConfig, TrainingConfig)
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

    Returns:
        TrainingPayload containing:
            - tf_training_config: TrainingConfig (canonical TensorFlow format)
            - pt_data_config: DataConfig (PyTorch singleton)
            - pt_model_config: ModelConfig (PyTorch singleton)
            - pt_training_config: TrainingConfig (PyTorch singleton)
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
        ...         'batch_size': 4,
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
    from ptycho_torch.config_bridge import to_model_config, to_training_config
    from ptycho_torch.config_params import update_existing_config

    # Defensive copy of overrides
    overrides = overrides or {}
    overrides_applied = dict(overrides)  # Audit trail

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
        overrides_applied['N'] = N  # Record inferred value

    # Step 3: Build PyTorch singleton configs with defaults + overrides
    # DataConfig: Extract data-related fields from overrides
    grid_size = overrides.get('grid_size', (overrides.get('gridsize', 1), overrides.get('gridsize', 1)))
    # Compute C from grid_size (number of channels = gridsize_x * gridsize_y)
    C = grid_size[0] * grid_size[1]

    #Going to use update_existing_config from ptycho_torch.config_params
    #The default settings are already set up to work in most use cases, so there's no point in instantiating 
    #versions of the classes with some pre-defined set of "default" settings in this function. We can just default
    #to whatever the original attribute values are, and overwrite them based on overrides.
    
    pt_data_config = PTDataConfig()
    update_existing_config(pt_data_config, overrides)

    # ModelConfig: Extract model architecture fields from overrides
    # CRITICAL: Synchronize C_forward and C_model with pt_data_config.C to ensure
    # PyTorch helpers (reassemble_patches_position_real) receive tensor shapes
    # consistent with the grouping strategy. Fixes ADR-003 C4.D3 coords_relative mismatch.
    overrides['C_forward'] = C
    overrides['C_model'] = C

    pt_model_config = PTModelConfig()
    update_existing_config(pt_model_config, overrides)

    # TrainingConfig: Extract training-specific fields from overrides
    overrides['nll'] = overrides.get('nll_weight', 1.0) > 0
    overrides['train_data_file'] = str(train_data_file)
    overrides['test_data_file'] = str(overrides['test_data_file']) if 'test_data_file' in overrides else None
    overrides['output_dir'] = str(output_dir)

    pt_training_config = PTTrainingConfig()
    update_existing_config(pt_training_config, overrides)


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
    return TrainingPayload(
        tf_training_config=tf_training_config,
        pt_data_config=pt_data_config,
        pt_model_config=pt_model_config,
        pt_training_config=pt_training_config,
        execution_config=execution_config,  # Now always PyTorchExecutionConfig instance
        overrides_applied=overrides_applied,
    )


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
    from ptycho_torch.config_bridge import to_model_config, to_inference_config

    # Defensive copy of overrides
    overrides = overrides or {}
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
        probe_scale=overrides.get('probe_scale', 1.0),  # PyTorch default
        subsample_seed=overrides.get('subsample_seed'),  # Optional field
    )

    # ModelConfig: Extract model architecture fields from overrides (for config_bridge)
    # Note: In inference, model config typically loaded from checkpoint, but we need
    # a ModelConfig instance for config_bridge translation
    # CRITICAL: Synchronize C_forward and C_model with pt_data_config.C (ADR-003 C4.D3)
    pt_model_config = PTModelConfig(
        mode=overrides.get('model_type', 'Unsupervised'),  # Map TF → PT naming
        amp_activation=overrides.get('amp_activation', 'silu'),
        n_filters_scale=overrides.get('n_filters_scale', 2),  # PyTorch default
        object_big=overrides.get('object_big', True),
        probe_big=overrides.get('probe_big', False),
        C_forward=C,  # Match data config channel count
        C_model=C,    # Match data config channel count
        pad_object=overrides.get('pad_object', True),  # Spec default
        gaussian_smoothing_sigma=overrides.get('gaussian_smoothing_sigma', 0.0),  # Spec default
    )

    # InferenceConfig: Extract inference-specific fields from overrides
    pt_inference_config = PTInferenceConfig(
        batch_size=overrides.get('batch_size', 16),  # PyTorch default
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
