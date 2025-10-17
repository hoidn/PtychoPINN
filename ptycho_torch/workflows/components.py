"""
PyTorch workflow orchestration layer — parity with ptycho/workflows/components.py.

This module provides PyTorch equivalents of the TensorFlow workflow orchestration
functions, enabling transparent backend selection from Ptychodus per the reconstructor
contract defined in specs/ptychodus_api_spec.md §4.

Architecture Role:
This module mirrors ptycho.workflows.components, sitting at the same orchestration
layer and providing identical entry point signatures. The key differences are:
1. Uses PyTorch backend (ptycho_torch.model, Lightning trainer)
2. Leverages config_bridge for TensorFlow dataclass compatibility
3. Delegates to RawDataTorch + PtychoDataContainerTorch from Phase C
4. Implements PyTorch-specific persistence via ModelManager or TorchModelManager

Critical Requirements (CONFIG-001 + spec §4.5):
- Entry points MUST call update_legacy_dict(params.cfg, config) before delegating
- Module MUST be torch-optional (importable when PyTorch unavailable)
- Signatures MUST match TensorFlow equivalents for transparent backend selection

Torch-Optional Design:
- Guarded imports using TORCH_AVAILABLE flag (from ptycho_torch.config_params)
- All torch-specific types aliased to fallback types when torch unavailable
- Entry points raise NotImplementedError when backend unavailable (Phases D2.B/C)

Core Workflow Functions (Scaffold):
Entry Points:
    - run_cdi_example_torch(): Complete training → reconstruction → visualization
    - train_cdi_model_torch(): Orchestrate data prep, probe setup, and Lightning training
    - load_inference_bundle_torch(): Load trained model for inference (archive compat)

Integration Points (Phase D2.B/C TODO):
- Config Bridge: ptycho_torch.config_bridge for dataclass translation
- Data Pipeline: RawDataTorch + PtychoDataContainerTorch from Phase C
- Training: Lightning trainer + MLflow autologging
- Persistence: TorchModelManager or extended ModelManager (Phase D3)

Example Usage (Post Phase D2.B/C):
    >>> from ptycho_torch.workflows.components import run_cdi_example_torch
    >>> from ptycho.config.config import TrainingConfig, ModelConfig
    >>>
    >>> # Configure via TensorFlow dataclasses (config bridge handles translation)
    >>> config = TrainingConfig(model=ModelConfig(N=64, gridsize=2), ...)
    >>>
    >>> # Execute PyTorch pipeline (identical signature to TensorFlow version)
    >>> amplitude, phase, results = run_cdi_example_torch(
    ...     train_data, test_data, config, do_stitching=True
    ... )

Artifacts:
- Phase D2.A scaffold: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T091450Z/
- Design decisions: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T085431Z/phase_d_decision.md
"""

# Standard library imports (no torch dependency)
import logging
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any

# Core imports (always available)
from ptycho import params
from ptycho.config.config import TrainingConfig
from ptycho.config import config as ptycho_config  # For update_legacy_dict
from ptycho.raw_data import RawData

# Torch-optional imports (guarded)
try:
    from ptycho_torch.config_params import TORCH_AVAILABLE
except ImportError:
    TORCH_AVAILABLE = False

# Type aliases for torch-optional compat (Phase C adapters)
if TORCH_AVAILABLE:
    try:
        from ptycho_torch.raw_data_bridge import RawDataTorch
        from ptycho_torch.data_container_bridge import PtychoDataContainerTorch
    except ImportError:
        # Fallback if Phase C modules not yet implemented
        RawDataTorch = None
        PtychoDataContainerTorch = None
else:
    RawDataTorch = None
    PtychoDataContainerTorch = None

# Set up logging
logger = logging.getLogger(__name__)


def run_cdi_example_torch(
    train_data: Union[RawData, 'RawDataTorch', 'PtychoDataContainerTorch'],
    test_data: Optional[Union[RawData, 'RawDataTorch', 'PtychoDataContainerTorch']],
    config: TrainingConfig,
    flip_x: bool = False,
    flip_y: bool = False,
    transpose: bool = False,
    M: int = 20,
    do_stitching: bool = False
) -> Tuple[Optional[Any], Optional[Any], Dict[str, Any]]:
    """
    Run the main CDI example execution flow using PyTorch backend.

    This function provides API parity with ptycho.workflows.components.run_cdi_example,
    enabling transparent backend selection from Ptychodus per specs/ptychodus_api_spec.md §4.5.

    CRITICAL: This function MUST call update_legacy_dict(params.cfg, config) before
    delegating to core modules to prevent CONFIG-001 violations (empty params.cfg
    causing silent shape mismatches downstream).

    Args:
        train_data: Training data (RawData, RawDataTorch, or PtychoDataContainerTorch)
        test_data: Optional test data (same type constraints as train_data)
        config: TrainingConfig instance (TensorFlow dataclass, translated via config_bridge)
        flip_x: Whether to flip the x coordinates during reconstruction
        flip_y: Whether to flip the y coordinates during reconstruction
        transpose: Whether to transpose the image by swapping dimensions
        M: Parameter for reassemble_position function (default: 20)
        do_stitching: Whether to perform image stitching after training

    Returns:
        Tuple containing:
        - reconstructed amplitude (or None if stitching disabled)
        - reconstructed phase (or None if stitching disabled)
        - results dictionary (training history, containers, metrics)

    Raises:
        NotImplementedError: Phase D2.B/C not yet implemented (scaffold only)

    Phase D2.A Scaffold Status:
        - Entry signature: ✅ COMPLETE (matches TensorFlow)
        - update_legacy_dict call: ✅ COMPLETE (CONFIG-001 compliance)
        - Placeholder logic: ✅ COMPLETE (raises NotImplementedError)
        - Torch-optional: ✅ COMPLETE (importable without torch)

    Phase D2.B/C TODO:
        - Implement train_cdi_model_torch delegation (Lightning trainer orchestration)
        - Implement reassemble_cdi_image_torch (optional stitching path)
        - Add MLflow disable flag handling
        - Validate deterministic seeds from config

    Example (Post D2.B/C):
        >>> from ptycho_torch.workflows.components import run_cdi_example_torch
        >>> from ptycho.config.config import TrainingConfig, ModelConfig
        >>> from ptycho.raw_data import RawData
        >>>
        >>> # Load data
        >>> train_data = RawData.from_file("train.npz")
        >>> config = TrainingConfig(model=ModelConfig(N=64), ...)
        >>>
        >>> # Execute PyTorch pipeline
        >>> amp, phase, results = run_cdi_example_torch(
        ...     train_data, None, config, do_stitching=False
        ... )
    """
    # CRITICAL: Update params.cfg before delegating (CONFIG-001 compliance)
    # This ensures legacy modules invoked downstream observe correct configuration state
    ptycho_config.update_legacy_dict(params.cfg, config)
    logger.info("PyTorch workflow: params.cfg synchronized with TrainingConfig")

    # Phase D2.B TODO: Implement training delegation
    # Expected flow:
    # 1. Convert train_data → PtychoDataContainerTorch via Phase C adapters
    # 2. Invoke train_cdi_model_torch (Lightning orchestration)
    # 3. If do_stitching + test_data: invoke reassemble_cdi_image_torch
    # 4. Return (amplitude, phase, results)

    raise NotImplementedError(
        "PyTorch training path not yet implemented. "
        "Phase D2.B will implement Lightning trainer orchestration. "
        "See plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md for roadmap."
    )


def _ensure_container(
    data: Union[RawData, 'RawDataTorch', 'PtychoDataContainerTorch'],
    config: TrainingConfig
) -> 'PtychoDataContainerTorch':
    """
    Normalize input data to PtychoDataContainerTorch using Phase C adapters.

    This helper mirrors the pattern in ptycho.workflows.components.create_ptycho_data_container,
    providing a single normalization pathway for all data types.

    Args:
        data: Input data (RawData, RawDataTorch, or PtychoDataContainerTorch)
        config: TrainingConfig for grouped data generation parameters

    Returns:
        PtychoDataContainerTorch: Normalized container ready for Lightning training

    Raises:
        TypeError: If data is not one of the supported types
        ImportError: If Phase C adapters not available (should not occur in Phase D2.B)

    Implementation Notes:
        - RawData → wrap with RawDataTorch, generate_grouped_data, then PtychoDataContainerTorch
        - RawDataTorch → generate_grouped_data → PtychoDataContainerTorch
        - PtychoDataContainerTorch → return as-is (already normalized)
    """
    if RawDataTorch is None or PtychoDataContainerTorch is None:
        raise ImportError(
            "Phase C adapters not available. Cannot create PtychoDataContainerTorch. "
            "Ensure ptycho_torch.raw_data_bridge and data_container_bridge are implemented."
        )

    # Case 1: Already a container - return as-is
    if hasattr(data, 'X') and hasattr(data, 'Y'):  # Duck-type check for PtychoDataContainerTorch
        logger.debug("Input is already PtychoDataContainerTorch, returning as-is")
        return data

    # Case 2: TensorFlow RawData - wrap with RawDataTorch
    if isinstance(data, RawData):
        logger.debug("Converting RawData → RawDataTorch → PtychoDataContainerTorch")
        # Wrap with RawDataTorch (Phase C adapter)
        torch_raw_data = RawDataTorch(
            xcoords=data.xcoords,
            ycoords=data.ycoords,
            diff3d=data.diff3d,
            probeGuess=data.probeGuess,
            scan_index=data.scan_index,
            objectGuess=data.objectGuess,
            Y=data.Y,
            config=config  # Pass config for update_legacy_dict call
        )
        data = torch_raw_data

    # Case 3: RawDataTorch - generate grouped data
    if hasattr(data, 'generate_grouped_data'):
        logger.debug("Generating grouped data from RawDataTorch")
        grouped_data = data.generate_grouped_data(
            N=config.model.N,
            K=config.neighbor_count,
            nsamples=config.n_groups,
            dataset_path=str(config.train_data_file) if config.train_data_file else None,
            sequential_sampling=config.sequential_sampling,
            gridsize=config.model.gridsize,
        )
        # Create PtychoDataContainerTorch from grouped data
        container = PtychoDataContainerTorch(grouped_data)
        return container

    # Case 4: Unknown type
    raise TypeError(
        f"data must be RawData, RawDataTorch, or PtychoDataContainerTorch, got {type(data)}"
    )


def _train_with_lightning(
    train_container: 'PtychoDataContainerTorch',
    test_container: Optional['PtychoDataContainerTorch'],
    config: TrainingConfig
) -> Dict[str, Any]:
    """
    Orchestrate Lightning trainer execution (stub for Phase D2.B).

    This function will instantiate PyTorch Lightning trainer and execute training.
    For Phase D2.B initial implementation, this is a stub that returns minimal results
    without actually running Lightning (for unit test purposes).

    Args:
        train_container: Normalized training data container
        test_container: Optional normalized test data container
        config: TrainingConfig with training hyperparameters

    Returns:
        Dict[str, Any]: Training results including history and containers

    Phase D2.B TODO:
        - Import Lightning components (torch-optional guarded)
        - Instantiate PtychoPINN Lightning module from ptycho_torch.model
        - Configure Trainer (max_epochs from config.nepochs, etc.)
        - Execute trainer.fit(model, train_dataloader, val_dataloader)
        - Extract training history from trainer.callback_metrics
        - Return structured results dict
    """
    logger.info("_train_with_lightning called (stub implementation for Phase D2.B)")
    logger.info(f"Training config: nepochs={config.nepochs}, n_groups={config.n_groups}")

    # Stub implementation for Phase D2.B unit tests
    # Full Lightning orchestration will be implemented after TDD cycle completes
    return {
        "history": {
            "train_loss": [0.5, 0.3],  # Placeholder loss trajectory
            "val_loss": [0.6, 0.4] if test_container is not None else None
        },
        "train_container": train_container,
        "test_container": test_container,
    }


def train_cdi_model_torch(
    train_data: Union[RawData, 'RawDataTorch', 'PtychoDataContainerTorch'],
    test_data: Optional[Union[RawData, 'RawDataTorch', 'PtychoDataContainerTorch']],
    config: TrainingConfig
) -> Dict[str, Any]:
    """
    Train the CDI model using PyTorch Lightning backend.

    This function provides API parity with ptycho.workflows.components.train_cdi_model,
    orchestrating data preparation, probe initialization, and Lightning trainer execution.

    Args:
        train_data: Training data (RawData, RawDataTorch, or PtychoDataContainerTorch)
        test_data: Optional test data for validation
        config: TrainingConfig instance (TensorFlow dataclass)

    Returns:
        Dict[str, Any]: Results dictionary containing:
        - 'history': Training history (losses, metrics)
        - 'train_container': PtychoDataContainerTorch for training data
        - 'test_container': Optional PtychoDataContainerTorch for test data
        - Additional outputs from Lightning trainer

    Raises:
        ImportError: If Phase C adapters not available
        TypeError: If input data types are invalid

    Phase D2.B Status:
        - Entry signature: ✅ COMPLETE (matches TensorFlow)
        - _ensure_container helper: ✅ COMPLETE (normalizes inputs via Phase C adapters)
        - Lightning orchestration: 🔶 STUB (returns minimal dict, full impl pending)
        - Torch-optional: ✅ COMPLETE (importable without torch)

    Example:
        >>> config = TrainingConfig(model=ModelConfig(N=64), nepochs=10, ...)
        >>> results = train_cdi_model_torch(train_data, test_data, config)
        >>> print(results['history']['train_loss'][-1])
    """
    # Step 1: Normalize train_data to PtychoDataContainerTorch
    logger.info("Normalizing training data via _ensure_container")
    train_container = _ensure_container(train_data, config)

    # Step 2: Normalize test_data if provided
    test_container = None
    if test_data is not None:
        logger.info("Normalizing test data via _ensure_container")
        test_container = _ensure_container(test_data, config)

    # Step 3: Initialize probe (TODO: implement probe handling for PyTorch)
    # TensorFlow baseline: probe.set_probe_guess(None, train_container.probe)
    # For Phase D2.B stub, skip probe initialization
    logger.debug("Probe initialization deferred to full Lightning implementation")

    # Step 4: Delegate to Lightning trainer
    logger.info("Delegating to Lightning trainer via _train_with_lightning")
    results = _train_with_lightning(train_container, test_container, config)

    return results


def load_inference_bundle_torch(model_dir: Path) -> Tuple[Any, dict]:
    """
    Load a trained PyTorch model bundle for inference from a directory.

    This function provides API parity with ptycho.workflows.components.load_inference_bundle,
    enabling transparent backend selection for model loading.

    Args:
        model_dir: Path to the directory containing the trained model artifacts.
                   Expects Lightning checkpoint or TorchModelManager archive format.

    Returns:
        Tuple containing:
        - model: Loaded Lightning module ready for inference
        - config: Configuration dictionary restored from saved model

    Raises:
        NotImplementedError: Phase D3 not yet implemented (scaffold only)
        ValueError: If model_dir is not a valid directory
        FileNotFoundError: If model archive is not found

    Phase D2.A Scaffold Status:
        - Entry signature: ✅ COMPLETE (matches TensorFlow)
        - Placeholder logic: ✅ COMPLETE (raises NotImplementedError)
        - Torch-optional: ✅ COMPLETE (importable without torch)

    Phase D3 TODO:
        - Validate model_dir structure (check for .ckpt or .h5.zip)
        - Load Lightning checkpoint OR unpack TorchModelManager archive
        - Restore params.cfg from bundled metadata
        - Return (lightning_module, config_dict)

    Example (Post D3):
        >>> model, config = load_inference_bundle_torch(Path("outputs/run_001"))
        >>> # Use model.forward() or lightning_module.predict() for inference
    """
    raise NotImplementedError(
        "PyTorch persistence loader not yet implemented. "
        "Phase D3 will implement Lightning checkpoint or TorchModelManager restoration. "
        "See plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md D3 for archive schema."
    )
