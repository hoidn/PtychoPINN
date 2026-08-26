"""Orchestration entry points and reconstruction wrappers.

Holds the public doors (``run_cdi_example_torch``, ``train_cdi_model_torch``)
plus the stitching/reassembly helpers.  Collaborators that tests patch through
the ``components`` facade are resolved late-bound via ``_components``.
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from ptycho.config.config import InferenceConfig, PyTorchExecutionConfig, TrainingConfig
from ptycho.raw_data import RawData
from ptycho_torch.scaling_contract import AmplitudePhysicsGainRecord
from ptycho_torch.config_factory import InferencePayload, TrainingPayload
from ptycho_torch.data_container_bridge import PtychoDataContainerTorch
from ptycho_torch.dataloader import PtychoDataset
from . import containers, lightning_service

from .lightning_service import _validate_training_execution_input
from .dataloaders import _build_inference_dataloader

# Preserves pre-split log provenance: records stay on the components facade logger.
logger = logging.getLogger('ptycho_torch.workflows.components')

def run_cdi_example_torch(
    train_data: Union[RawData, 'PtychoDataContainerTorch'],
    test_data: Optional[Union[RawData, 'PtychoDataContainerTorch']],
    config: TrainingConfig,
    flip_x: bool = False,
    flip_y: bool = False,
    transpose: bool = False,
    M: int = 20,
    do_stitching: bool = False,
    execution_config: Optional[Any] = None,
    overrides: Optional[dict] = None,
    *,
    resolved_payload: Optional[TrainingPayload] = None,
    amplitude_physics_gain_record: Optional[
        AmplitudePhysicsGainRecord
    ] = None,
    torch_training_seed: Optional[int] = None,
    batch_order_recipe: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[Any], Dict[str, Any]]:
    """
    Run the main CDI example execution flow using PyTorch backend.

    This function provides API parity with ptycho.workflows.components.run_cdi_example,
    enabling transparent backend selection from Ptychodus per specs/ptychodus_api_spec.md §4.5.

    Resolved configuration is passed to descendants explicitly. The workflow
    does not project the complete config into ``params.cfg``; any surviving
    legacy leaf owns its own narrow compatibility scope.

    Args:
        train_data: Training data (RawData or PtychoDataContainerTorch)
        test_data: Optional test data (same type constraints as train_data)
        config: TrainingConfig instance (TensorFlow dataclass, translated via config_bridge)
        flip_x: Whether to flip the x coordinates during reconstruction
        flip_y: Whether to flip the y coordinates during reconstruction
        transpose: Whether to transpose the image by swapping dimensions
        M: Parameter for reassemble_position function (default: 20)
        do_stitching: Whether to perform image stitching after training
        execution_config: Optional unresolved ExecutionRequest for runtime
                         control. The factory resolves it exactly once.
        overrides: Optional torch-only factory overrides forwarded unchanged to
            the Lightning training boundary.
        amplitude_physics_gain_record: Optional complete provenance record for
            the already-resolved scalar in ModelConfig/ModelSpec. The record is
            persisted only in its strict bundle sidecar.

    Returns:
        Tuple containing:
        - reconstructed amplitude (or None if stitching disabled)
        - reconstructed phase (or None if stitching disabled)
        - results dictionary (training history, containers, metrics)

    Implementation:
        1. Train the model via train_cdi_model_torch (Lightning trainer
           orchestration), forwarding execution_config when provided.
        2. Initialize reconstruction outputs (recon_amp, recon_phase) to None.
        3. If do_stitching and test_data is provided, stitch reconstructed patches
           via _reassemble_cdi_image_torch and merge the results into train_results.
        4. The shared training service persists the selected serving state to
           wts.h5.zip under the run root.
        5. Return (recon_amp, recon_phase, train_results), matching the TensorFlow
           baseline signature (specs/ptychodus_api_spec.md §4.5).

    Example:
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

    Contract: docs/architecture_torch.md §Component Contracts.
    """
    _validate_training_execution_input(execution_config, resolved_payload)

    # Step 1: Train the model through the shared Lightning service.
    logger.info("Invoking PyTorch training orchestration via train_cdi_model_torch")
    training_kwargs = {
        "persist_bundle": True,
        "amplitude_physics_gain_record": amplitude_physics_gain_record,
    }
    if execution_config is not None:
        training_kwargs["execution_config"] = execution_config
    if overrides is not None:
        training_kwargs["overrides"] = overrides
    if resolved_payload is not None:
        training_kwargs["resolved_payload"] = resolved_payload
    if torch_training_seed is not None:
        training_kwargs["torch_training_seed"] = torch_training_seed
    if batch_order_recipe is not None:
        training_kwargs["batch_order_recipe"] = batch_order_recipe
    train_results = train_cdi_model_torch(
        train_data,
        test_data,
        config,
        **training_kwargs,
    )
    if amplitude_physics_gain_record is not None:
        train_results["amplitude_physics_gain_record"] = (
            amplitude_physics_gain_record
        )
        train_results["amplitude_physics_gain_metadata"] = (
            amplitude_physics_gain_record.to_metadata()
        )

    # Step 2: Initialize return values for reconstruction outputs
    recon_amp, recon_phase = None, None

    # Step 3: Optional stitching path (when explicitly requested + test data provided)
    # Mirrors TensorFlow baseline ptycho/workflows/components.py:714-721
    if do_stitching and test_data is not None:
        logger.info("Performing image stitching (do_stitching=True, test_data provided)...")
        # Invoke reassembly helper to stitch reconstructed patches
        recon_amp, recon_phase, reassemble_results = _reassemble_cdi_image_torch(
            test_data, config, flip_x, flip_y, transpose, M, train_results=train_results
        )
        # Merge reassembly outputs into training results (update pattern from TF baseline)
        train_results.update(reassemble_results)
        logger.info("Image stitching complete")
    else:
        logger.info("Skipping image stitching (do_stitching=False or no test data available)")

    # Step 4: Return tuple matching TensorFlow baseline signature
    # (amplitude, phase, results) per specs/ptychodus_api_spec.md §4.5
    return recon_amp, recon_phase, train_results


def _reassemble_cdi_image_torch_mmap(
   test_data: PtychoDataset,
   config: InferenceConfig,
   payload: InferencePayload,
   execution_config: PyTorchExecutionConfig,
   train_results: Optional[Dict[str, Any]] = None,
   verbose = True
):
    """
    Reassemble CDI image using optimized CDI image functions from ptycho_torch library
    """
    
    #Import 
    try:
        from ptycho_torch.reassembly import reconstruct_image_barycentric
        from ptycho_torch.config_params import TrainingConfig
    except Exception as e:
        print(f"Could not import due to exception: {e}")
    
    #Getting proper configs
    data_config = payload.pt_data_config
    inference_config = payload.pt_inference_config
    #Dummy argument to get reconstruct function tow ork
    model_config = None

    #Loading model
    loaded_model = train_results['models']['diffraction_to_obj']
    loaded_model.eval()
    loaded_model.to(execution_config.accelerator)
    loaded_model.training = True

    #Workaround since the only use of training_config is for device
    #Will pass in PyTorch TrainingConfig so we don't need to modify reconstruct_image_baryecentric
    training_config = TrainingConfig(device = execution_config.accelerator)

    #Reconstructing. Automatically puts dataset into dataloader, so don't worry about it
    if verbose:
        print(f"Data config: {data_config}")
        print(f"Inference config: {inference_config}")

    #Call optimized reconstruction method

    result, recon_dataset, _ = reconstruct_image_barycentric(loaded_model, test_data,
                           training_config, data_config, model_config, inference_config, gpu_ids = None,
                           use_mixed_precision=True, verbose = False)
    
    
    return result.to('cpu')
    



def _reassemble_cdi_image_torch(
    test_data: Union[RawData, 'PtychoDataContainerTorch'],
    config: TrainingConfig,
    flip_x: bool,
    flip_y: bool,
    transpose: bool,
    M: int,
    train_results: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Reassemble CDI image using trained PyTorch model.

    This function provides API parity with ptycho.workflows.components.reassemble_cdi_image,
    orchestrating model inference and patch reassembly to produce final reconstruction.

    Args:
        test_data: Test data for reconstruction (RawData or PtychoDataContainerTorch)
        config: TrainingConfig for inference parameters
        flip_x: Whether to flip the x coordinates during reconstruction
        flip_y: Whether to flip the y coordinates during reconstruction
        transpose: Whether to transpose the image by swapping dimensions
        M: Parameter for reassemble_position function
        train_results: Optional training results dict containing 'models' with trained Lightning module

    Returns:
        Tuple containing:
        - recon_amp: Reconstructed amplitude array (np.ndarray)
        - recon_phase: Reconstructed phase array (np.ndarray)
        - results: Dictionary with intermediate outputs (obj_tensor_full, global_offsets, etc.)

    Raises:
        RuntimeError: If PyTorch not available or train_results not provided
        ValueError: If models dict missing from train_results

    Example:
        >>> train_results = run_cdi_example_torch(train_data, test_data, config, do_stitching=False)
        >>> recon_amp, recon_phase, results = _reassemble_cdi_image_torch(
        ...     test_data, config, flip_x=False, flip_y=False, transpose=False, M=20,
        ...     train_results=train_results
        ... )
    """
    # torch-optional import guarded here
    try:
        import torch
        import numpy as np
    except ImportError as e:
        raise RuntimeError(
            "PyTorch backend requires torch. "
            "Install with: pip install -e .[torch]\n"
            "See docs/findings.md#policy-001 for PyTorch requirement policy."
        ) from e

    # Validate train_results contains models
    if train_results is None:
        # For backward compatibility with tests expecting NotImplementedError,
        # raise NotImplementedError to maintain RED test expectations
        raise NotImplementedError(
            "PyTorch stitching path not yet fully implemented without train_results. "
            "Must pass train_results from run_cdi_example_torch(..., do_stitching=False) output. "
            "See docs/findings.md (see git history for the originating plan) C3 for implementation status."
        )
    if 'models' not in train_results or not train_results['models']:
        raise ValueError("train_results['models'] dict required for inference")

    # Step 1: Normalize test_data → PtychoDataContainerTorch
    test_container = containers.create_torch_data_container(test_data, config)

    # Step 2: Extract trained Lightning module and set to eval mode
    # Extract Lightning module from dual-model dict
    lightning_module = train_results['models']['diffraction_to_obj']
    lightning_module.eval()

    # Step 3: Reuse the run's resolved carrier; do not materialize a second
    # runtime default at the inference boundary.
    resolved_execution = train_results.get("execution_config")
    if resolved_execution is None:
        raise ValueError(
            "train_results must contain the run's resolved execution_config"
        )
    infer_loader = _build_inference_dataloader(
        test_container,
        config,
        execution_config=resolved_execution,
    )

    # Step 4: Extract probe and scale factors for inference
    # Probe tensor is required for forward_predict; extract from container
    probe_tensor = getattr(test_container, 'probeGuess', None)
    if probe_tensor is None:
        # Fallback: create dummy probe if not available
        logger.warning("probeGuess not found in test_container; using dummy probe")
        probe_tensor = torch.ones(1, config.model.N, config.model.N, dtype=torch.complex64)
    if not isinstance(probe_tensor, torch.Tensor):
        probe_tensor = torch.tensor(probe_tensor)

    # Step 5: Run inference to collect predictions and offsets
    obj_patches = []
    global_offsets = test_container.global_offsets.clone()  # (n_samples, 1, 2, 1)

    with torch.no_grad():
        for batch in infer_loader:
            # batch is (X, coords) from TensorDataset
            X_batch, coords_batch = batch
            # DTYPE ENFORCEMENT: Ensure float32 before Lightning forward
            X_batch = X_batch.to(torch.float32)
            coords_batch = coords_batch.to(torch.float32)

            # Create batch-sized scale factors with proper shape for broadcasting
            # IntensityScalerModule expects (B, 1, 1, 1) shaped scale factor
            batch_size = X_batch.shape[0]
            input_scale = torch.ones(batch_size, 1, 1, 1, device=X_batch.device, dtype=X_batch.dtype)

            # Call forward_predict which returns complex object patches
            # Signature: forward_predict(x, positions, probe, input_scale_factor)
            pred = lightning_module.forward_predict(
                X_batch,
                coords_batch,
                probe_tensor.to(X_batch.device),
                input_scale
            )
            obj_patches.append(pred.cpu())

    # Concatenate all predictions
    obj_tensor_full = torch.cat(obj_patches, dim=0)  # (n_samples, ...)

    # Ensure 4D tensor for reassembly (n_samples, 1, H, W) or (n_samples, C, H, W)
    if obj_tensor_full.ndim == 3:
        obj_tensor_full = obj_tensor_full.unsqueeze(1)  # Add channel dim

    # Step 5: Apply coordinate transformations
    if transpose:
        # Transpose spatial dimensions
        obj_tensor_full = obj_tensor_full.transpose(-2, -1)

    if flip_x:
        global_offsets[:, 0, 0, :] = -global_offsets[:, 0, 0, :]
    if flip_y:
        global_offsets[:, 0, 1, :] = -global_offsets[:, 0, 1, :]

    # Step 6: Prepare tensor for TensorFlow reassembly helper
    # TensorFlow helper expects (n_samples, H, W, 1) single-channel complex tensor
    # PyTorch models output (n_samples, C, H, W) with C=gridsize**2 channels
    # See debug_shape_triage.md (2025-10-19T092448Z) for root cause analysis

    # Convert channel-first to channel-last if needed
    if obj_tensor_full.ndim == 4 and obj_tensor_full.shape[1] < obj_tensor_full.shape[2] and obj_tensor_full.shape[1] < obj_tensor_full.shape[3]:
        # Channel dim is dim=1 (channel-first); move to end
        obj_tensor_full = obj_tensor_full.permute(0, 2, 3, 1)  # (n, C, H, W) → (n, H, W, C)

    # Reduce multi-channel output to single channel for TensorFlow reassembly
    # For gridsize > 1, model outputs multiple channels (gridsize**2); take mean across channels
    if obj_tensor_full.shape[-1] > 1:
        obj_tensor_full = torch.mean(obj_tensor_full, dim=-1, keepdim=True)  # (n, H, W, C) → (n, H, W, 1)

    # Step 7: Reassemble patches (using TensorFlow helper for MVP parity)
    # For, delegate to TensorFlow reassembly to maintain exact parity
    # Future enhancement: use native PyTorch reassembly from ptycho_torch.reassembly
    from ptycho import tf_helper as hh
    obj_tensor_np = obj_tensor_full.cpu().numpy()
    global_offsets_np = global_offsets.cpu().numpy()
    if (global_offsets_np.ndim == 4
            and global_offsets_np.shape[2] == 2
            and global_offsets_np.shape[3] == 1):
        global_offsets_np = np.swapaxes(global_offsets_np, 2, 3)

    obj_image = hh.reassemble_position(
        obj_tensor_np,
        global_offsets_np,
        M=M,
    )

    # Squeeze trailing channel dimension if present (reassembly may return (H, W, 1))
    if obj_image.ndim == 3 and obj_image.shape[-1] == 1:
        obj_image = np.squeeze(obj_image, axis=-1)

    # Step 8: Extract amplitude and phase
    recon_amp = np.absolute(obj_image)
    recon_phase = np.angle(obj_image)

    # Step 8: Build results dict
    results = {
        "obj_tensor_full": obj_tensor_np,
        "global_offsets": global_offsets_np,
        "recon_amp": recon_amp,
        "recon_phase": recon_phase,
        "containers": {
            "test": test_container
        }
    }

    return recon_amp, recon_phase, results


def train_cdi_model_torch(
    train_data: Union[RawData, 'PtychoDataContainerTorch', 'PtychoDataset'],
    test_data: Optional[Union[RawData, 'PtychoDataContainerTorch']],
    config: TrainingConfig,
    execution_config: Optional[Any] = None,
    overrides: Optional[dict] = None,
    *,
    resolved_payload: Optional[TrainingPayload] = None,
    torch_training_seed: Optional[int] = None,
    batch_order_recipe: Optional[str] = None,
    persist_bundle: bool = False,
    amplitude_physics_gain_record: Optional[
        AmplitudePhysicsGainRecord
    ] = None,
    rescaled_source_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train the CDI model using PyTorch Lightning backend.

    This function provides API parity with ptycho.workflows.components.train_cdi_model,
    orchestrating data preparation, probe initialization, and Lightning trainer execution.

    Args:
        train_data: Training data (RawData, PtychoDataContainerTorch, or PtychoDataset)
        test_data: Optional test data for validation
        config: TrainingConfig instance (TensorFlow dataclass)
        execution_config: Optional unresolved ExecutionRequest for runtime control
        overrides: Optional torch-only factory overrides.

    Returns:
        Dict[str, Any]: Results dictionary containing:
        - 'history': Training history (losses, metrics)
        - 'train_container': PtychoDataContainerTorch for training data
        - 'test_container': Optional PtychoDataContainerTorch for test data
        - Additional outputs from Lightning trainer

    Raises:
        ImportError: If Phase C adapters not available
        TypeError: If input data types are invalid

    Example:
        >>> config = TrainingConfig(model=ModelConfig(N=64), nepochs=10, ...)
        >>> results = train_cdi_model_torch(train_data, test_data, config)
        >>> print(results['history']['train_loss'][-1])
    """
    _validate_training_execution_input(execution_config, resolved_payload)

    # Step 1: Normalize train_data to PtychoDataContainerTorch
    logger.info("Normalizing training data via create_torch_data_container")
    train_container = containers.create_torch_data_container(train_data, config)

    # Step 2: Normalize test_data if provided
    test_container = None
    if test_data is not None:
        logger.info("Normalizing test data via create_torch_data_container")
        test_container = containers.create_torch_data_container(test_data, config)

    # Probe ownership remains with the normalized data/model boundary.

    # Resolve the payload once: an already-resolved payload wins; otherwise
    # derive the Torch owners from the canonical training baseline plus the
    # torch-only overrides (the resolution the trainer previously did inline).
    payload = resolved_payload
    if payload is None:
        from ptycho_torch.config_factory import (
            build_training_factory_overrides,
            resolve_training_payload,
        )

        factory_overrides = build_training_factory_overrides(config)
        if overrides:
            factory_overrides.update(overrides)
        payload = resolve_training_payload(
            train_data_file=Path(config.train_data_file),
            output_dir=Path(getattr(config, 'output_dir', './outputs')),
            execution_config=execution_config,
            overrides=factory_overrides,
            training_baseline=config,
        )

    # Step 4: Delegate to Lightning trainer
    logger.info("Delegating to Lightning trainer via _train_with_lightning")
    lightning_kwargs = {}
    if torch_training_seed is not None:
        lightning_kwargs["torch_training_seed"] = torch_training_seed
    if batch_order_recipe is not None:
        lightning_kwargs["batch_order_recipe"] = batch_order_recipe
    if persist_bundle:
        lightning_kwargs["persist_bundle"] = True
    if amplitude_physics_gain_record is not None:
        lightning_kwargs["amplitude_physics_gain_record"] = (
            amplitude_physics_gain_record
        )
    if rescaled_source_sha256 is not None:
        lightning_kwargs["rescaled_source_sha256"] = rescaled_source_sha256
    intensity_scale = None
    if hasattr(train_container, 'physics_scaling_constant'):
        import torch

        scale_tensor = torch.as_tensor(train_container.physics_scaling_constant)
        intensity_scale = float(scale_tensor.reshape(-1)[0].item())
    if intensity_scale is not None:
        lightning_kwargs["intensity_scale"] = intensity_scale
    results = lightning_service._train_with_lightning(
        payload,
        train_container,
        test_container,
        **lightning_kwargs,
    )
    if intensity_scale is not None:
        results['intensity_scale'] = intensity_scale

    return results

