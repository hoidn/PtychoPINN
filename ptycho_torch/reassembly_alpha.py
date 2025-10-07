import math
import time
import gc
from typing import Tuple, Optional, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ptycho_torch.dataloader_old import PtychoDataset, TensorDictDataLoader
from ptycho_torch.dataloader import Collate
from ptycho_torch.config_params import ModelConfig, TrainingConfig, DataConfig, InferenceConfig

torch.set_default_dtype(torch.float32)


def profile_memory():
    """Print current GPU memory usage."""
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")


class VectorizedBarycentricAccumulator:
    """
    Fully vectorized barycentric accumulation for ptychography reconstruction.
    Processes all patches simultaneously using scatter operations.
    """
    
    def __init__(self, canvas_shape: Tuple[int, int], device: torch.device):
        self.canvas_shape = canvas_shape
        self.device = device
    
    def accumulate_batch(self, 
                        canvas: torch.Tensor,
                        canvas_counts: torch.Tensor,
                        patches: torch.Tensor,
                        positions_px: torch.Tensor,
                        patch_size: int) -> None:
        """
        Vectorized accumulation of all patches at once.
        
        Args:
            canvas: (H, W) canvas tensor
            canvas_counts: (H, W) counts tensor
            patches: (N, patch_size, patch_size) patches to accumulate
            positions_px: (N, 2) positions in pixels
            patch_size: Size of each patch
        """
        N = patches.shape[0]
        half_size = patch_size / 2
        
        # Compute corners for all patches
        xmin = positions_px[:, 0] - half_size
        ymin = positions_px[:, 1] - half_size
        
        xmin_wh = xmin.floor().long()
        ymin_wh = ymin.floor().long()
        xmin_fr = xmin - xmin_wh.float()
        ymin_fr = ymin - ymin_wh.float()
        
        # Bounds checking
        valid_mask = (
            (xmin_wh >= 0) & (ymin_wh >= 0) &
            (xmin_wh + patch_size + 1 < self.canvas_shape[1]) &
            (ymin_wh + patch_size + 1 < self.canvas_shape[0])
        )
        
        if not valid_mask.all():
            valid_idx = torch.where(valid_mask)[0]
            if len(valid_idx) == 0:
                return
            patches = patches[valid_idx]
            xmin_wh, ymin_wh = xmin_wh[valid_idx], ymin_wh[valid_idx]
            xmin_fr, ymin_fr = xmin_fr[valid_idx], ymin_fr[valid_idx]
            N = len(valid_idx)
        
        # Bilinear interpolation weights
        xmin_fr_c = 1.0 - xmin_fr
        ymin_fr_c = 1.0 - ymin_fr
        
        w00 = (ymin_fr_c * xmin_fr_c).unsqueeze(-1).unsqueeze(-1)
        w01 = (ymin_fr_c * xmin_fr).unsqueeze(-1).unsqueeze(-1)
        w10 = (ymin_fr * xmin_fr_c).unsqueeze(-1).unsqueeze(-1)
        w11 = (ymin_fr * xmin_fr).unsqueeze(-1).unsqueeze(-1)
        
        # Create index tensors for vectorized operations
        patch_y, patch_x = torch.meshgrid(
            torch.arange(patch_size, device=self.device),
            torch.arange(patch_size, device=self.device),
            indexing='ij'
        )
        
        patch_y_exp = patch_y.unsqueeze(0).expand(N, -1, -1)
        patch_x_exp = patch_x.unsqueeze(0).expand(N, -1, -1)
        
        # Canvas coordinates for each patch pixel
        canvas_y_base = ymin_wh.unsqueeze(-1).unsqueeze(-1) + patch_y_exp
        canvas_x_base = xmin_wh.unsqueeze(-1).unsqueeze(-1) + patch_x_exp
        
        # Flatten for advanced indexing
        patches_flat = patches.reshape(N, -1)
        canvas_y_flat = canvas_y_base.reshape(N, -1)
        canvas_x_flat = canvas_x_base.reshape(N, -1)
        
        # Weighted patches
        w00_flat = w00.expand(-1, patch_size, patch_size).reshape(N, -1)
        w01_flat = w01.expand(-1, patch_size, patch_size).reshape(N, -1)
        w10_flat = w10.expand(-1, patch_size, patch_size).reshape(N, -1)
        w11_flat = w11.expand(-1, patch_size, patch_size).reshape(N, -1)
        
        weighted_patches_00 = (patches_flat * w00_flat).reshape(-1)
        weighted_patches_01 = (patches_flat * w01_flat).reshape(-1)
        weighted_patches_10 = (patches_flat * w10_flat).reshape(-1)
        weighted_patches_11 = (patches_flat * w11_flat).reshape(-1)
        
        # Canvas indices (flattened)
        idx_00 = (canvas_y_flat * self.canvas_shape[1] + canvas_x_flat).reshape(-1)
        idx_01 = (canvas_y_flat * self.canvas_shape[1] + canvas_x_flat + 1).reshape(-1)
        idx_10 = ((canvas_y_flat + 1) * self.canvas_shape[1] + canvas_x_flat).reshape(-1)
        idx_11 = ((canvas_y_flat + 1) * self.canvas_shape[1] + canvas_x_flat + 1).reshape(-1)
        
        # Accumulate on flattened canvas
        canvas_flat = canvas.reshape(-1)
        counts_flat = canvas_counts.reshape(-1)
        
        canvas_flat.scatter_add_(0, idx_00, weighted_patches_00)
        canvas_flat.scatter_add_(0, idx_01, weighted_patches_01)
        canvas_flat.scatter_add_(0, idx_10, weighted_patches_10)
        canvas_flat.scatter_add_(0, idx_11, weighted_patches_11)
        
        # Update counts
        counts_flat.scatter_add_(0, idx_00, w00_flat.reshape(-1))
        counts_flat.scatter_add_(0, idx_01, w01_flat.reshape(-1))
        counts_flat.scatter_add_(0, idx_10, w10_flat.reshape(-1))
        counts_flat.scatter_add_(0, idx_11, w11_flat.reshape(-1))


class PtychoDataParallelWrapper(nn.Module):
    """Wrapper to make ptychography models compatible with nn.DataParallel."""
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def forward(self, inputs):
        x, positions, probe, in_scale = inputs
        return self.model.forward_predict(x, positions, probe, in_scale)


def setup_multi_gpu_model(model: nn.Module, 
                         gpu_ids: Optional[List[int]] = None) -> Tuple[nn.Module, List[int], torch.device]:
    """
    Setup model for multi-GPU using DataParallel.
    
    Returns:
        (model, gpu_ids_used, primary_device)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    total_gpus = torch.cuda.device_count()
    
    if gpu_ids is None:
        gpu_ids = list(range(total_gpus))
    
    if len(gpu_ids) == 1:
        primary_device = torch.device(f'cuda:{gpu_ids[0]}')
        model = model.to(primary_device)
        return model, gpu_ids, primary_device
    
    primary_device = torch.device(f'cuda:{gpu_ids[0]}')
    model = model.to(primary_device)
    
    wrapped_model = PtychoDataParallelWrapper(model)
    parallel_model = nn.DataParallel(wrapped_model, device_ids=gpu_ids)
    
    return parallel_model, gpu_ids, primary_device


def reconstruct_image(model: nn.Module,
                     ptycho_dset: PtychoDataset,
                     training_config: TrainingConfig,
                     data_config: DataConfig,
                     model_config: ModelConfig,
                     inference_config: InferenceConfig,
                     gpu_ids: Optional[List[int]] = None,
                     use_mixed_precision: bool = True,
                     verbose: bool = True) -> Tuple[torch.Tensor, Any]:
    """
    Multi-GPU ptychography reconstruction using barycentric coordinate assembly.
    Automatically uses DataParallel if multiple GPUs are specified.
    
    Args:
        model: Neural network model
        ptycho_dset: Ptychography dataset
        training_config: Training configuration
        data_config: Data configuration  
        model_config: Model configuration
        inference_config: Inference configuration
        gpu_ids: List of GPU IDs to use (None for single GPU on training_config.device)
        verbose: Whether to print progress
    
    Returns:
        Tuple of (reconstructed_canvas, dataset_subset)
    """
    
    gpu_ids = list(range(torch.cuda.device_count()))

    # Setup model (single or multi-GPU)
    if gpu_ids is None or len(gpu_ids) <= 1:
        # Single GPU mode
        if gpu_ids and len(gpu_ids) == 1:
            device = torch.device(f'cuda:{gpu_ids[0]}')
        else:
            device = training_config.device
        model = model.to(device)
        primary_device = device
        if verbose:
            print(f"Using single GPU: {device}")
    else:
        # Multi-GPU mode
        model, gpu_ids, primary_device = setup_multi_gpu_model(model, gpu_ids)
        if verbose:
            print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
    
    # Get dataset subset
    n_files = ptycho_dset.n_files
    experiment_number = inference_config.experiment_number
    
    if n_files > 1:
        ptycho_subset = ptycho_dset.get_experiment_dataset(experiment_number)
    else:
        ptycho_subset = ptycho_dset
    
    # Pre-compute constants
    global_coords = ptycho_subset.mmap_ptycho['coords_global'].squeeze()
    
    if 'com' in ptycho_subset.data_dict:
        center_of_mass = torch.mean(global_coords, 
                                  dim=tuple(range(global_coords.dim()-1)))
    else:
        center_of_mass = ptycho_subset.data_dict['com']
    
    center_of_mass = center_of_mass.to(primary_device)
    
    # Determine canvas size
    adjusted_coords = global_coords - center_of_mass.cpu()
    max_offset = torch.ceil(torch.max(torch.abs(adjusted_coords))).int().item()
    canvas_size = (inference_config.middle_trim + 2 * max_offset, 
                   inference_config.middle_trim + 2 * max_offset)
    
    if verbose:
        print(f"Canvas size: {canvas_size}, Max offset: {max_offset}")
    
    # Initialize canvas
    canvas = torch.zeros(canvas_size, device=primary_device, dtype=torch.complex64)
    canvas_counts = torch.zeros(canvas_size, device=primary_device, dtype=torch.float32)
    
    # Create dataloader
    infer_loader = TensorDictDataLoader(
        ptycho_subset, 
        batch_size=inference_config.batch_size,
        num_workers=training_config.num_workers,
        collate_fn=Collate(device=primary_device),
        pin_memory = True,
        persistent_workers=True
    )
    
    # Create accumulator
    accumulator = VectorizedBarycentricAccumulator(canvas_size, primary_device)
    
    model.eval()
    total_inference_time = 0
    total_assembly_time = 0
    
    with torch.no_grad():
        for i, batch in enumerate(infer_loader):
            batch_start_time = time.time()
            
            # Prepare data
            batch_data = batch[0]
            x = batch_data['images'].to(primary_device, non_blocking=True)
            positions = batch_data['coords_relative'].to(primary_device, non_blocking=True)
            probe = batch[1].to(primary_device, non_blocking=True)
            in_scale = batch_data['rms_scaling_constant'].to(primary_device, non_blocking=True)
            batch_coords_global = batch_data['coords_global'].to(primary_device, non_blocking=True)
            
            # Model inference with optional mixed precision
            inference_start = time.time()
            if use_mixed_precision:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    if isinstance(model, nn.DataParallel):
                        inputs = (x, positions, probe, in_scale)
                        batch_output = model(inputs)
                    else:
                        batch_output = model.forward_predict(x, positions, probe, in_scale)
            else:
                if isinstance(model, nn.DataParallel):
                    inputs = (x, positions, probe, in_scale)
                    batch_output = model(inputs)
                else:
                    batch_output = model.forward_predict(x, positions, probe, in_scale)
            
            torch.cuda.synchronize()
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            # Canvas assembly
            assembly_start = time.time()
            
            # Extract center regions
            N = data_config.N
            middle = inference_config.middle_trim
            center_start = N // 2 - middle // 2
            center_end = N // 2 + middle // 2
            
            # Handle model output shape: [B, C, H, W, 2] -> complex
            if batch_output.dim() == 5 and batch_output.shape[-1] == 2:
                batch_output = torch.complex(batch_output[..., 0], batch_output[..., 1])
            
            if data_config.C == 1:
                # Single channel
                im_center = batch_output[:, :, center_start:center_end, center_start:center_end].squeeze(1)
                batch_coords_2d = batch_coords_global.squeeze()
            else:
                # Multi-channel: flatten batch and channel dimensions
                B, C = batch_output.shape[:2]
                im_center = batch_output[:, :, center_start:center_end, center_start:center_end]
                im_center = im_center.view(B * C, middle, middle)
                batch_coords_2d = batch_coords_global.squeeze(2).view(B * C, 2)
            
            # Compute canvas positions
            relative_positions = batch_coords_2d - center_of_mass.unsqueeze(0)
            canvas_center = torch.tensor([canvas_size[1] // 2, canvas_size[0] // 2], 
                                       device=primary_device, dtype=torch.float32)
            canvas_positions = relative_positions + canvas_center.unsqueeze(0)
            
            # Accumulate patches
            accumulator.accumulate_batch(canvas, canvas_counts, im_center, 
                                       canvas_positions, middle)
            
            assembly_time = time.time() - assembly_start
            total_assembly_time += assembly_time
            
            # Memory cleanup
            del x, positions, probe, in_scale, batch_coords_global
            del batch_output, im_center, canvas_positions
            
            if i % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            if verbose:
                batch_time = time.time() - batch_start_time
                efficiency = inference_time / assembly_time if assembly_time > 0 else float('inf')
                print(f'Batch {i+1}/{len(infer_loader)}: {batch_time:.3f}s '
                      f'(inference: {inference_time:.3f}s, assembly: {assembly_time:.3f}s, '
                      f'efficiency: {efficiency:.1f}x)')
    
    # Performance summary
    if verbose:
        avg_inference_time = total_inference_time / len(infer_loader)
        avg_assembly_time = total_assembly_time / len(infer_loader)
        efficiency = avg_inference_time / avg_assembly_time if avg_assembly_time > 0 else float('inf')
        
        print(f"\nPerformance Summary:")
        print(f"  Average inference time per batch: {avg_inference_time:.3f}s")
        print(f"  Average assembly time per batch: {avg_assembly_time:.3f}s")
        print(f"  Parallel efficiency ratio: {efficiency:.1f}x")
        print(f"  Total reconstruction time: {total_inference_time + total_assembly_time:.2f}s")
    
    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    return canvas / canvas_counts, ptycho_subset, [total_inference_time, total_assembly_time]