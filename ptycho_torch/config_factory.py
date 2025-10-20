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

# Import for Option A: PyTorchExecutionConfig will be added to ptycho.config.config
# Per supervisor decision at 2025-10-19T234458Z (factory_design.md §2.2)
# For now, using type annotation only to avoid import error during RED phase
try:
    from ptycho.config.config import PyTorchExecutionConfig
except ImportError:
    # Placeholder for RED phase; Option A implementation pending in ptycho/config/config.py
    PyTorchExecutionConfig = None  # type: ignore


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
    execution_config: Any  # PyTorchExecutionConfig (type: ignore during RED phase)
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
    execution_config: Any  # PyTorchExecutionConfig (type: ignore during RED phase)
    overrides_applied: Dict[str, Any] = field(default_factory=dict)  # Audit trail


def create_training_payload(
    train_data_file: Path,
    output_dir: Path,
    overrides: Optional[Dict[str, Any]] = None,
    execution_config: Optional[Any] = None,  # PyTorchExecutionConfig
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
    raise NotImplementedError(
        "create_training_payload() is a Phase B2 RED scaffold. "
        "Implementation pending in Phase B3.a per "
        "plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/plan.md"
    )


def create_inference_payload(
    model_path: Path,
    test_data_file: Path,
    output_dir: Path,
    overrides: Optional[Dict[str, Any]] = None,
    execution_config: Optional[Any] = None,  # PyTorchExecutionConfig
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
    raise NotImplementedError(
        "create_inference_payload() is a Phase B2 RED scaffold. "
        "Implementation pending in Phase B3.a per "
        "plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/plan.md"
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
    raise NotImplementedError(
        "infer_probe_size() is a Phase B2 RED scaffold. "
        "Implementation pending in Phase B3.a per "
        "plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/plan.md"
    )


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
    raise NotImplementedError(
        "populate_legacy_params() is a Phase B2 RED scaffold. "
        "Implementation pending in Phase B3.a per "
        "plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/plan.md"
    )
