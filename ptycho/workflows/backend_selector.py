"""
Backend selection dispatcher for transparent TensorFlow ↔ PyTorch routing.

This module provides the core backend selection mechanism that enables Ptychodus
to transparently switch between TensorFlow and PyTorch implementations based on
runtime configuration (TrainingConfig.backend or InferenceConfig.backend).

Architecture Pattern:
    The dispatcher follows a "CONFIG-001 compliance → backend routing → delegation"
    pattern, ensuring that params.cfg is synchronized before workflow execution
    regardless of which backend is selected.

    Flow:
    1. Client calls run_cdi_example_with_backend(train_data, test_data, config, ...)
    2. Dispatcher calls update_legacy_dict(params.cfg, config) — CONFIG-001 gate
    3. Dispatcher inspects config.backend field ('tensorflow' or 'pytorch')
    4. Dispatcher imports and delegates to backend-specific workflow module
    5. Results are returned with backend metadata injected into results dict

Critical Requirements (specs/ptychodus_api_spec.md §4.1):
    - MUST call update_legacy_dict before backend dispatch (CONFIG-001)
    - MUST handle PyTorch unavailability with actionable error message
    - MUST preserve API parity: identical signatures between backends
    - MUST inject 'backend' field into results dict for traceability

Torch-Optional Design:
    - TensorFlow path has zero PyTorch dependencies (always available)
    - PyTorch path uses guarded import with try/except ImportError
    - RuntimeError raised with installation guidance when PyTorch unavailable
    - Error messages include "pip install torch" or "pip install .[torch]" guidance

Core Functions:
    - run_cdi_example_with_backend(): Training + optional reconstruction pipeline
    - train_cdi_model_with_backend(): Training-only orchestration
    - load_inference_bundle_with_backend(): Model loading from wts.h5.zip archive

Usage Example:
    >>> from ptycho.workflows.backend_selector import run_cdi_example_with_backend
    >>> from ptycho.config.config import TrainingConfig, ModelConfig
    >>> from ptycho.raw_data import RawData
    >>>
    >>> # Configure for PyTorch backend
    >>> config = TrainingConfig(
    ...     model=ModelConfig(N=64, gridsize=2),
    ...     train_data_file='train.npz',
    ...     backend='pytorch'  # or 'tensorflow' (default)
    ... )
    >>>
    >>> # Load data
    >>> train_data = RawData.from_file('train.npz')
    >>>
    >>> # Execute with automatic backend routing
    >>> amp, phase, results = run_cdi_example_with_backend(
    ...     train_data, None, config, do_stitching=False
    ... )
    >>> print(results['backend'])  # 'pytorch'

References:
    - Phase E plan: plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md
    - Backend design: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T180500Z/phase_e_backend_design.md
    - Spec §4.1: specs/ptychodus_api_spec.md (Ptychodus reconstructor contract)
"""

import logging
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any

# Core imports (always available)
from ptycho import params
from ptycho.config.config import TrainingConfig, InferenceConfig, update_legacy_dict
from ptycho.raw_data import RawData
from ptycho.loader import PtychoDataContainer

# Set up logging
logger = logging.getLogger(__name__)


def run_cdi_example_with_backend(
    train_data: Union[RawData, PtychoDataContainer],
    test_data: Optional[Union[RawData, PtychoDataContainer]],
    config: TrainingConfig,
    flip_x: bool = False,
    flip_y: bool = False,
    transpose: bool = False,
    M: int = 20,
    do_stitching: bool = False,
    torch_execution_config: Optional[Any] = None
) -> Tuple[Optional[Any], Optional[Any], Dict[str, Any]]:
    """
    Run the complete CDI workflow with automatic backend selection.

    This function provides a unified entry point for Ptychodus that automatically
    routes to TensorFlow or PyTorch implementations based on config.backend field.
    It ensures CONFIG-001 compliance by calling update_legacy_dict before dispatch.

    Args:
        train_data: Training data (RawData or PtychoDataContainer for TensorFlow;
                   may also be RawDataTorch or PtychoDataContainerTorch for PyTorch)
        test_data: Optional test data (same type constraints as train_data)
        config: TrainingConfig instance with backend field ('tensorflow' or 'pytorch')
        flip_x: Whether to flip the x coordinates during reconstruction
        flip_y: Whether to flip the y coordinates during reconstruction
        transpose: Whether to transpose the image by swapping dimensions
        M: Parameter for reassemble_position function (default: 20)
        do_stitching: Whether to perform image stitching after training
        torch_execution_config: Optional PyTorchExecutionConfig for PyTorch backend only
                               (ignored for TensorFlow). See CONFIG-002, CONFIG-LOGGER-001.

    Returns:
        Tuple containing:
        - reconstructed amplitude (or None if stitching disabled)
        - reconstructed phase (or None if stitching disabled)
        - results dictionary with 'backend' field injected for traceability

    Raises:
        RuntimeError: When backend='pytorch' but PyTorch unavailable
        ValueError: When config.backend is not 'tensorflow' or 'pytorch'

    Example:
        >>> config = TrainingConfig(model=ModelConfig(N=64), backend='pytorch')
        >>> amp, phase, results = run_cdi_example_with_backend(train_data, None, config)
        >>> assert results['backend'] == 'pytorch'
    """
    # CRITICAL: Update params.cfg BEFORE backend dispatch (CONFIG-001 compliance)
    update_legacy_dict(params.cfg, config)
    logger.info(f"Backend dispatcher: params.cfg synchronized with TrainingConfig (backend={config.backend})")

    # Validate backend field
    if config.backend not in ('tensorflow', 'pytorch'):
        raise ValueError(
            f"Invalid backend: {config.backend!r}. "
            f"TrainingConfig.backend must be 'tensorflow' or 'pytorch'."
        )

    # Route to backend-specific workflow implementation
    if config.backend == 'tensorflow':
        logger.info("Backend dispatcher: routing to TensorFlow workflow (ptycho.workflows.components)")
        from ptycho.workflows import components as tf_components

        # Delegate to TensorFlow run_cdi_example
        recon_amp, recon_phase, results = tf_components.run_cdi_example(
            train_data, test_data, config, flip_x, flip_y, transpose, M, do_stitching
        )

    elif config.backend == 'pytorch':
        logger.info("Backend dispatcher: routing to PyTorch workflow (ptycho_torch.workflows.components)")

        # Guarded PyTorch import
        try:
            from ptycho_torch.workflows import components as torch_components
        except ImportError as e:
            raise RuntimeError(
                "PyTorch backend selected (config.backend='pytorch') but ptycho_torch module unavailable. "
                "This typically means PyTorch is not installed or the PyTorch integration is incomplete. "
                "\n\nTo install PyTorch support, run:\n"
                "  pip install torch  # or\n"
                "  pip install .[torch]  # if using editable install\n\n"
                f"Original error: {e}"
            ) from e

        # Auto-instantiate execution_config if None (GPU-first defaults per POLICY-001)
        if torch_execution_config is None:
            from ptycho.config.config import PyTorchExecutionConfig
            torch_execution_config = PyTorchExecutionConfig()  # Triggers auto-resolution to cuda/cpu
            logger.info(
                f"Backend dispatcher: auto-instantiated PyTorchExecutionConfig with "
                f"accelerator='{torch_execution_config.accelerator}' (POLICY-001 GPU-first defaults)"
            )

        # Delegate to PyTorch run_cdi_example_torch
        recon_amp, recon_phase, results = torch_components.run_cdi_example_torch(
            train_data, test_data, config, flip_x, flip_y, transpose, M, do_stitching,
            execution_config=torch_execution_config
        )

    # Inject backend metadata into results for traceability
    results['backend'] = config.backend
    logger.info(f"Backend dispatcher: workflow complete (backend={config.backend})")

    return recon_amp, recon_phase, results


def train_cdi_model_with_backend(
    train_data: Union[RawData, PtychoDataContainer],
    test_data: Optional[Union[RawData, PtychoDataContainer]],
    config: TrainingConfig
) -> Dict[str, Any]:
    """
    Train the CDI model with automatic backend selection.

    This function provides a training-only entry point that routes to TensorFlow or
    PyTorch implementations based on config.backend field.

    Args:
        train_data: Training data (RawData or PtychoDataContainer)
        test_data: Optional test data for validation
        config: TrainingConfig instance with backend field

    Returns:
        Dict[str, Any]: Results dictionary containing:
        - 'history': Training history (losses, metrics)
        - 'train_container': Normalized training data container
        - 'test_container': Optional normalized test data container
        - 'backend': Backend identifier ('tensorflow' or 'pytorch')
        - Additional backend-specific outputs

    Raises:
        RuntimeError: When backend='pytorch' but PyTorch unavailable
        ValueError: When config.backend is invalid
    """
    # CRITICAL: Update params.cfg BEFORE backend dispatch (CONFIG-001 compliance)
    update_legacy_dict(params.cfg, config)
    logger.info(f"Backend dispatcher: params.cfg synchronized for training (backend={config.backend})")

    # Validate backend field
    if config.backend not in ('tensorflow', 'pytorch'):
        raise ValueError(
            f"Invalid backend: {config.backend!r}. "
            f"TrainingConfig.backend must be 'tensorflow' or 'pytorch'."
        )

    # Route to backend-specific training implementation
    if config.backend == 'tensorflow':
        logger.info("Backend dispatcher: routing to TensorFlow training (ptycho.workflows.components)")
        from ptycho.workflows import components as tf_components

        results = tf_components.train_cdi_model(train_data, test_data, config)

    elif config.backend == 'pytorch':
        logger.info("Backend dispatcher: routing to PyTorch training (ptycho_torch.workflows.components)")

        # Guarded PyTorch import
        try:
            from ptycho_torch.workflows import components as torch_components
        except ImportError as e:
            raise RuntimeError(
                "PyTorch backend selected (config.backend='pytorch') but ptycho_torch module unavailable. "
                "This typically means PyTorch is not installed or the PyTorch integration is incomplete. "
                "\n\nTo install PyTorch support, run:\n"
                "  pip install torch  # or\n"
                "  pip install .[torch]  # if using editable install\n\n"
                f"Original error: {e}"
            ) from e

        results = torch_components.train_cdi_model_torch(train_data, test_data, config)

    # Inject backend metadata
    results['backend'] = config.backend
    logger.info(f"Backend dispatcher: training complete (backend={config.backend})")

    return results


def load_inference_bundle_with_backend(
    bundle_dir: Union[str, Path],
    config: InferenceConfig,
    model_name: str = 'diffraction_to_obj'
) -> Tuple[Any, dict]:
    """
    Load a trained model bundle for inference with automatic backend selection.

    This function provides a loading entry point that routes to TensorFlow or PyTorch
    model loaders based on config.backend field. It ensures CONFIG-001 compliance by
    restoring params.cfg from the saved bundle.

    Args:
        bundle_dir: Path to directory containing wts.h5.zip archive
        config: InferenceConfig instance with backend field
        model_name: Name of model to load from dual-model bundle (default: 'diffraction_to_obj')

    Returns:
        Tuple containing:
        - model: Loaded model ready for inference (TensorFlow or PyTorch)
        - params_dict: Configuration dictionary restored from saved bundle

    Raises:
        RuntimeError: When backend='pytorch' but PyTorch unavailable
        ValueError: When config.backend is invalid
        FileNotFoundError: When bundle archive not found

    Notes:
        - TensorFlow: delegates to ptycho.workflows.components.load_inference_bundle
        - PyTorch: delegates to ptycho_torch.workflows.components.load_inference_bundle_torch
        - Both backends restore params.cfg before model reconstruction (CONFIG-001)
    """
    # Validate backend field
    if config.backend not in ('tensorflow', 'pytorch'):
        raise ValueError(
            f"Invalid backend: {config.backend!r}. "
            f"InferenceConfig.backend must be 'tensorflow' or 'pytorch'."
        )

    # Normalize bundle_dir to Path
    bundle_path = Path(bundle_dir)

    # Route to backend-specific loader implementation
    if config.backend == 'tensorflow':
        logger.info("Backend dispatcher: loading TensorFlow model (ptycho.workflows.components)")
        from ptycho.workflows import components as tf_components

        # TensorFlow load_inference_bundle returns (model, params_dict)
        model, params_dict = tf_components.load_inference_bundle(bundle_path)

    elif config.backend == 'pytorch':
        logger.info("Backend dispatcher: loading PyTorch model (ptycho_torch.workflows.components)")

        # Guarded PyTorch import
        try:
            from ptycho_torch.workflows import components as torch_components
        except ImportError as e:
            raise RuntimeError(
                "PyTorch backend selected (config.backend='pytorch') but ptycho_torch module unavailable. "
                "This typically means PyTorch is not installed or the PyTorch integration is incomplete. "
                "\n\nTo install PyTorch support, run:\n"
                "  pip install torch  # or\n"
                "  pip install .[torch]  # if using editable install\n\n"
                f"Original error: {e}"
            ) from e

        # PyTorch load_inference_bundle_torch returns (models_dict, params_dict)
        # Extract single model from dict for API parity with TensorFlow
        models_dict, params_dict = torch_components.load_inference_bundle_torch(
            bundle_path, model_name=model_name
        )
        model = models_dict.get(model_name)

        if model is None:
            raise KeyError(
                f"Model '{model_name}' not found in loaded PyTorch bundle. "
                f"Available models: {list(models_dict.keys())}"
            )

    # CRITICAL: params.cfg was already restored inside backend-specific loader (CONFIG-001)
    # No additional update_legacy_dict call needed here

    logger.info(f"Backend dispatcher: inference bundle loaded (backend={config.backend})")

    return model, params_dict
