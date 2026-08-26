"""Lightning and inference dataloader builders for the Torch training service.

Builds the one native RAM/mmap loader path and the deterministic inference
loader.  Configuration->primitive unpacking happens here exactly once per phase.
"""
from typing import Dict, Optional, Union

from ptycho.config.config import PyTorchExecutionConfig, TrainingConfig
from ptycho_torch.config_factory import TrainingPayload
from ptycho_torch.data_container_bridge import PtychoDataContainerTorch
from ptycho_torch.dataloader import PtychoDataset, _PtychoContainerDataset
from ptycho_torch.train_utils import PrebuiltPtychoDataModule, is_spawn_strategy
from ptycho_torch.model_spec import MODEL_TYPE_TO_MODE
from ptycho_torch.scaling_contract import CI_SCALE_CONTRACT
from ptycho_torch.batch_order import (
    DEFAULT_BATCH_ORDER_RECIPE,
    JULY2026_BATCH_ORDER_RECIPE,
    July2026BatchOrderSampler,
    validate_batch_order_loader_schedule,
)
from ptycho_torch import dataloader, scaling_contract

from .containers import _adapt_container_for_ci

def _resolve_torch_training_seed(
    config: Optional[TrainingConfig],
    torch_training_seed: Optional[int],
) -> int:
    """Resolve the dedicated Torch stream or the historical direct-call fallback."""

    if torch_training_seed is None:
        configured_seed = getattr(config, "subsample_seed", None)
        torch_training_seed = 42 if configured_seed is None else configured_seed
    if (
        isinstance(torch_training_seed, bool)
        or not isinstance(torch_training_seed, int)
    ):
        raise TypeError("torch_training_seed must be a nonnegative integer")
    if torch_training_seed < 0:
        raise ValueError("torch_training_seed must be a nonnegative integer")
    return torch_training_seed


def _build_lightning_dataloaders(
    train_container: Union['PtychoDataContainerTorch', Dict, 'PtychoDataset'],
    test_container: Optional[
        Union['PtychoDataContainerTorch', Dict, 'PtychoDataset']
    ],
    config: Optional[TrainingConfig],
    payload: Optional[TrainingPayload] = None,
    *,
    torch_training_seed: Optional[int] = None,
    batch_order_recipe: str = DEFAULT_BATCH_ORDER_RECIPE,
):
    """Build the one native RAM/mmap loader path for Lightning training."""

    import lightning.pytorch as L
    import torch
    from dataclasses import replace

    batch_order_recipe = validate_batch_order_loader_schedule(
        batch_order_recipe,
        has_validation_loader=test_container is not None,
    )
    if payload is not None and config is None:
        config = getattr(payload, "tf_training_config", None)

    from ptycho_torch.config_params import (
        DataConfig as PTDataConfig,
        ModelConfig as PTModelConfig,
        TrainingConfig as PTTrainingConfig,
    )

    data_config = getattr(payload, "pt_data_config", None) if payload else None
    if data_config is None:
        data_source = getattr(config, "data_config", config)
        data_config = PTDataConfig(
            **{
                name: getattr(data_source, name)
                for name in ("scale_contract_version", "measurement_domain")
                if data_source is not None and hasattr(data_source, name)
            }
        )

    model_config = getattr(payload, "pt_model_config", None) if payload else None
    if model_config is None:
        source = getattr(config, "model", None)
        mode = getattr(source, "mode", None) or MODEL_TYPE_TO_MODE.get(
            getattr(source, "model_type", None), "Unsupervised"
        )
        model_config = PTModelConfig(
            mode=mode,
            object_big=bool(getattr(source, "object_big", True)),
            physics_forward_mode=getattr(
                source, "physics_forward_mode", "amplitude"
            ),
        )

    training_config = (
        getattr(payload, "pt_training_config", None) if payload else None
    )
    if training_config is None:
        training_config = PTTrainingConfig(
            batch_size=getattr(config, "batch_size", PTTrainingConfig().batch_size),
            torch_loss_mode=getattr(config, "torch_loss_mode", "poisson"),
        )

    scale_contract = scaling_contract.validate_scale_contract(
        data_config, model_config, training_config
    )
    ci_active = (
        scale_contract is not None
        and scale_contract.version == CI_SCALE_CONTRACT
    )
    if ci_active and not isinstance(train_container, PtychoDataset):
        statistics = _adapt_container_for_ci(
            train_container,
            data_config=data_config,
            model_config=model_config,
        )
        if statistics is not None:
            _adapt_container_for_ci(
                test_container,
                data_config=data_config,
                model_config=model_config,
                statistics=statistics,
            )

    seed = _resolve_torch_training_seed(config, torch_training_seed)
    # LOAD-BEARING RE-SEED: _construct_application builds the model AFTER this
    # function returns (Phase 2 seam order). Model-init RNG equivalence with the
    # pre-split coordinator depends on this being the LAST global-RNG operation
    # here — everything downstream uses private generators. Do not delete as
    # "redundant" with the coordinator's earlier seed_everything.
    L.seed_everything(seed)
    shuffle = not bool(getattr(config, "sequential_sampling", False))
    if not shuffle and batch_order_recipe == JULY2026_BATCH_ORDER_RECIPE:
        raise ValueError(
            "sequential_sampling=True conflicts with the historical "
            f"batch_order_recipe={JULY2026_BATCH_ORDER_RECIPE!r}"
        )

    execution_config = getattr(payload, "execution_config", None)
    strategy = (
        getattr(execution_config, "strategy", None)
        if execution_config is not None
        else getattr(training_config, "strategy", None)
    )
    distributed = strategy == "ddp" or is_spawn_strategy(strategy)

    if isinstance(train_container, PtychoDataset) and distributed:
        if batch_order_recipe == JULY2026_BATCH_ORDER_RECIPE:
            raise ValueError(
                f"batch_order_recipe={JULY2026_BATCH_ORDER_RECIPE!r} is a "
                "single-device historical reproduction contract; Lightning "
                "DDP replaces custom samplers and is not supported"
            )
        runtime_training = training_config
        if execution_config is not None:
            runtime_training = replace(
                training_config,
                strategy=execution_config.strategy,
                n_devices=execution_config.devices,
                num_workers=execution_config.num_workers,
                device=(
                    "cuda"
                    if execution_config.accelerator in {"cuda", "gpu"}
                    else execution_config.accelerator
                ),
            )
        return PrebuiltPtychoDataModule(
            train_container.data_dir_path,
            model_config,
            data_config,
            runtime_training,
            validation_map_path=(
                test_container.data_dir_path
                if isinstance(test_container, PtychoDataset)
                else None
            ),
            execution_config=execution_config,
            shuffle_training=shuffle,
            torch_training_seed=seed,
        )

    if isinstance(train_container, PtychoDataset):
        train_dataset = train_container
        validation_dataset = (
            test_container
            if isinstance(test_container, PtychoDataset)
            else None
        )
        if ci_active:
            statistics = train_dataset.get_ci_statistics()
            if statistics is None:
                statistics = train_dataset.set_ci_statistics_from_indices(
                    torch.arange(len(train_dataset))
                )
            if validation_dataset is not None:
                validation_dataset.data_dict["ci_statistics"] = {
                    name: value.detach().clone()
                    for name, value in statistics.items()
                }
    else:
        train_dataset = _PtychoContainerDataset(
            train_container,
            model_config=model_config,
            data_config=data_config,
            ci_active=ci_active,
        )
        validation_dataset = (
            _PtychoContainerDataset(
                test_container,
                model_config=model_config,
                data_config=data_config,
                ci_active=ci_active,
            )
            if test_container is not None
            else None
        )

    sampler = None
    if shuffle and batch_order_recipe == JULY2026_BATCH_ORDER_RECIPE:
        sampler = July2026BatchOrderSampler(train_dataset, seed=seed)

    if execution_config is None:
        worker_settings = {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
        }
    else:
        worker_settings = {
            "num_workers": execution_config.num_workers,
            "pin_memory": execution_config.pin_memory,
            "persistent_workers": execution_config.persistent_workers,
            "prefetch_factor": execution_config.prefetch_factor,
        }

    train_loader = dataloader.build_ptycho_loader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        seed=seed,
        **worker_settings,
    )
    validation_loader = (
        dataloader.build_ptycho_loader(
            validation_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            seed=seed,
            **worker_settings,
        )
        if validation_dataset is not None
        else None
    )
    return train_loader, validation_loader


def _build_inference_dataloader(
    container: 'PtychoDataContainerTorch',
    config: TrainingConfig,
    execution_config: Optional['PyTorchExecutionConfig'] = None
) -> 'DataLoader':
    """
    Build deterministic PyTorch DataLoader for inference/stitching.

    This helper creates a DataLoader optimized for inference: no shuffling,
    sequential iteration, and batch sizing configured for memory efficiency.

    Args:
        container: Inference data container (PtychoDataContainerTorch or dict)
        config: TrainingConfig with batch_size setting
        execution_config: Optional PyTorchExecutionConfig with runtime knobs

    Returns:
        DataLoader: Sequential loader for inference predictions

    Notes:
        - Always uses shuffle=False for deterministic stitching order
        - drop_last=False ensures all samples are processed
        - Batch size can be overridden via execution_config.inference_batch_size
        - num_workers and pin_memory controlled by execution_config
        - Compatible with _build_lightning_dataloaders duck-typing pattern
    """
    # torch-optional import guarded here
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        raise RuntimeError(
            "PyTorch backend requires torch. "
            "Install with: pip install -e .[torch]\n"
            "See docs/findings.md#policy-001 for PyTorch requirement policy."
        ) from e

    # Extract tensors using same helper pattern as training loader
    def _get_tensor(container, key, default=None):
        """Helper to extract tensor from container or dict."""
        if hasattr(container, key):
            val = getattr(container, key)
        elif isinstance(container, dict):
            val = container.get(key, default)
        else:
            val = default

        # Convert numpy arrays to torch tensors if needed
        if val is not None and not isinstance(val, torch.Tensor):
            import numpy as np
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
        return val

    # Build inference dataset
    infer_X = _get_tensor(container, 'X')
    infer_coords = _get_tensor(container, 'coords_nominal')

    # Fallback for missing tensors
    if infer_X is None:
        infer_X = torch.randn(5, 64, 64)
    if infer_coords is None:
        batch_size = infer_X.size(0) if isinstance(infer_X, torch.Tensor) else 5
        infer_coords = torch.randn(batch_size, 2)

    # DTYPE ENFORCEMENT: Cast to float32 to prevent Lightning Conv2d dtype mismatch
    # Requirement: specs/data_contracts.md §1 mandates diffraction arrays be float32
    # Root cause: torch.from_numpy preserves dtype; legacy/checkpoint data may be float64
    # Symptom: RuntimeError "Input type (double) and bias type (float)" in Lightning forward
    # Solution: Explicit cast before TensorDataset construction
    infer_X = infer_X.to(torch.float32, copy=False)
    infer_coords = infer_coords.to(torch.float32, copy=False)

    # SHAPE CONVERSION: Container X is channel-last (B, H, W, C), model expects channel-first (B, C, H, W)
    # Permute to match model input format
    if infer_X.ndim == 4:
        infer_X = infer_X.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)

    infer_dataset = TensorDataset(infer_X, infer_coords)

    if execution_config is None:
        raise ValueError(
            "inference dataloader requires the run's resolved "
            "PyTorchExecutionConfig"
        )

    # Determine batch size: execution_config.inference_batch_size overrides config.batch_size
    batch_size = execution_config.inference_batch_size or getattr(config, 'batch_size', 1)

    # Create deterministic loader with execution config knobs
    return DataLoader(
        infer_dataset,
        batch_size=batch_size,  # Controlled by execution_config.inference_batch_size
        shuffle=False,  # Deterministic order for stitching
        drop_last=False,  # Process all samples
        num_workers=execution_config.num_workers,  # Controlled by execution_config
        pin_memory=execution_config.pin_memory  # GPU-only flag, CPU-safe default False
    )


