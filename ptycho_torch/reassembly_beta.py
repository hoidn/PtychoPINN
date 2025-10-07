import math

#Type helpers
from typing import Tuple, Optional, Union, Callable, Any, List
from ptycho_torch.dataloader_old import PtychoDataset, TensorDictDataLoader
import torch.nn as nn

#Pytorch-related
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import ptycho_torch.helper as hh
from ptycho_torch.dataloader import Collate

#Other useful libraries
import time
import gc

#Configurations
from ptycho_torch.config_params import ModelConfig, TrainingConfig, DataConfig, InferenceConfig

from ptycho_torch.helper import reassemble_patches_position_real



#Default casting
torch.set_default_dtype(torch.float32)

def profile_memory():
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")
    
    # Get detailed breakdown
    print(torch.cuda.memory_summary())

class BatchedObjectPatchInterpolator:
    """
    Vectorized barycentric interpolation for ptychography reassembly.
    Handles batches of patches and solution regions efficiently.
    """
    
    def __init__(self, 
                 canvas_shape: Tuple[int, int],
                 positions_px: torch.Tensor,  # (B, 2) or (B, C, 2)
                 patch_size: int,
                 device: torch.device,
                 dtype: torch.dtype = torch.complex64):
        """
        Args:
            canvas_shape: (H, W) of final reconstruction canvas
            positions_px: Positions in pixels, shape (B, 2) for single channel or (B, C, 2) for multi-channel
            patch_size: Size of each patch to interpolate
            device: Target device
            dtype: Data type for tensors
        """
        self.canvas_shape = canvas_shape
        self.patch_size = patch_size
        self.device = device
        self.dtype = dtype
        
        # Handle multi-channel coordinates
        if positions_px.dim() == 3:  # (B, C, 2)
            self.is_multi_channel = True
            self.B, self.C = positions_px.shape[:2]
            positions_px = positions_px.view(-1, 2)  # Flatten to (B*C, 2)
        else:  # (B, 2)
            self.is_multi_channel = False
            self.B, self.C = positions_px.shape[0], 1
        
        self.total_patches = positions_px.shape[0]
        
        # Pre-compute interpolation weights and indices
        self._precompute_interpolation_data(positions_px, patch_size)
    
    def _precompute_interpolation_data(self, positions_px: torch.Tensor, patch_size: int):
        """Pre-compute all barycentric weights and canvas indices."""
        half_size = patch_size / 2
        
        # Top-left corners of patch support regions
        xmin = positions_px[:, 0] - half_size  # (total_patches,)
        ymin = positions_px[:, 1] - half_size
        
        # Integer and fractional components
        xmin_wh = xmin.floor().long()
        ymin_wh = ymin.floor().long()
        xmin_fr = xmin - xmin_wh.float()
        ymin_fr = ymin - ymin_wh.float()
        
        # Bottom-right corners (add 1 for bilinear interpolation)
        xmax_wh = xmin_wh + patch_size + 1
        ymax_wh = ymin_wh + patch_size + 1
        
        # Store bounds
        self.xmin_wh = xmin_wh
        self.ymin_wh = ymin_wh
        self.xmax_wh = xmax_wh
        self.ymax_wh = ymax_wh
        
        # Pre-compute barycentric weights for all patches
        xmin_fr_c = 1.0 - xmin_fr
        ymin_fr_c = 1.0 - ymin_fr
        
        # Weights for bilinear interpolation (total_patches,)
        self.weight00 = (ymin_fr_c * xmin_fr_c).to(self.device)
        self.weight01 = (ymin_fr_c * xmin_fr).to(self.device)
        self.weight10 = (ymin_fr * xmin_fr_c).to(self.device)
        self.weight11 = (ymin_fr * xmin_fr).to(self.device)
        
        # Create patch coordinate grids (for vectorized operations)
        y_patch, x_patch = torch.meshgrid(
            torch.arange(patch_size, device=self.device),
            torch.arange(patch_size, device=self.device),
            indexing='ij'
        )
        self.y_patch = y_patch.flatten()  # (patch_size²,)
        self.x_patch = x_patch.flatten()
    
    def accumulate_patches(self, 
                          canvas: torch.Tensor, 
                          canvas_counts: torch.Tensor,
                          patches: torch.Tensor) -> None:
        """
        Accumulate patches onto canvas using barycentric interpolation.
        
        Args:
            canvas: Canvas tensor to accumulate onto (H, W)
            canvas_counts: Count tensor for normalization (H, W)
            patches: Patch tensors to accumulate (total_patches, patch_size, patch_size)
        """
        patch_size = self.patch_size
        total_patches = patches.shape[0]
        
        # Flatten patches for vectorized operations
        patches_flat = patches.view(total_patches, -1)  # (total_patches, patch_size²)
        
        # Vectorized bounds checking
        valid_mask = (
            (self.xmin_wh >= 0) & (self.ymin_wh >= 0) &
            (self.xmax_wh < self.canvas_shape[1]) & (self.ymax_wh < self.canvas_shape[0])
        )
        
        if not valid_mask.all():
            print(f"Warning: {(~valid_mask).sum().item()} patches fall outside canvas bounds")
        
        # Process valid patches
        valid_indices = torch.where(valid_mask)[0]
        
        for i in valid_indices:
            y_start, y_end = self.ymin_wh[i], self.ymax_wh[i]
            x_start, x_end = self.xmin_wh[i], self.xmax_wh[i]
            
            # Extract current patch
            patch = patches[i]  # (patch_size, patch_size)
            
            # Get interpolation weights for this patch
            w00, w01, w10, w11 = self.weight00[i], self.weight01[i], self.weight10[i], self.weight11[i]
            
            # Bilinear interpolation accumulation
            canvas[y_start:y_end-1, x_start:x_end-1] += w00 * patch
            canvas[y_start:y_end-1, x_start+1:x_end] += w01 * patch
            canvas[y_start+1:y_end, x_start:x_end-1] += w10 * patch
            canvas[y_start+1:y_end, x_start+1:x_end] += w11 * patch
            
            # Update counts for normalization
            canvas_counts[y_start:y_end-1, x_start:x_end-1] += w00
            canvas_counts[y_start:y_end-1, x_start+1:x_end] += w01
            canvas_counts[y_start+1:y_end, x_start:x_end-1] += w10
            canvas_counts[y_start+1:y_end, x_start+1:x_end] += w11


def barycentric_reassemble_single_channel(im_tensor: torch.Tensor,
                                        com: torch.Tensor,
                                        global_coords: torch.Tensor,
                                        data_config: DataConfig,
                                        middle: int = 10,
                                        canvas_size: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-efficient reassembly using barycentric coordinates instead of padding + translation.
    
    Args:
        im_tensor: (B, C, H, W) - model predictions
        com: (2,) - center of mass
        global_coords: (B, 2) or (B, C, 2) - global coordinates
        data_config: DataConfig object
        middle: Center region size to extract
        canvas_size: Optional canvas size, auto-computed if None
    """
    device = im_tensor.device
    N = data_config.N
    
    # Extract center regions
    center_start = N // 2 - middle // 2
    center_end = N // 2 + middle // 2
    im_center = im_tensor[:, :, center_start:center_end, center_start:center_end].squeeze(1)
    
    # Compute relative positions in pixels
    global_coords_2d = global_coords.float().squeeze()
    relative_positions = global_coords_2d - com.unsqueeze(0)
    
    # Determine canvas size if not provided
    if canvas_size is None:
        max_offset = torch.ceil(torch.max(torch.abs(relative_positions))).int().item()
        canvas_size = (middle + 2 * max_offset, middle + 2 * max_offset)
    
    # Adjust positions to canvas coordinates (center canvas)
    canvas_center = torch.tensor([canvas_size[1] // 2, canvas_size[0] // 2], 
                                device=device, dtype=torch.float32)
    canvas_positions = relative_positions + canvas_center.unsqueeze(0)
    
    # Initialize canvas and counts
    canvas = torch.zeros(canvas_size, device=device, dtype=im_tensor.dtype)
    canvas_counts = torch.zeros(canvas_size, device=device, dtype=torch.float32)
    
    # Create interpolator
    interpolator = BatchedObjectPatchInterpolator(
        canvas_shape=canvas_size,
        positions_px=canvas_positions,
        patch_size=middle,
        device=device,
        dtype=im_tensor.dtype
    )
    
    # Accumulate patches
    interpolator.accumulate_patches(canvas, canvas_counts, im_center)
    
    return canvas, canvas_counts


def barycentric_reassemble_multi_channel(im_tensor: torch.Tensor,
                                       com: torch.Tensor,
                                       relative_coords: torch.Tensor,
                                       coord_centers: torch.Tensor,
                                       data_config: DataConfig,
                                       model_config: ModelConfig,
                                       middle: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-channel barycentric reassembly. First assembles solution regions, 
    then uses barycentric interpolation for final canvas assembly.
    """
    device = im_tensor.device
    
    # Step 1: Reassemble patches within solution regions (keep your existing logic)

    soln_patches, ones_mask, padded_size = vectorized_barycentric_reassemble_patches(
        im_tensor, relative_coords, data_config, model_config, agg=True
    )
    
    # Extract center regions
    center_start = padded_size // 2 - middle // 2
    center_end = padded_size // 2 + middle // 2
    soln_patches_center = soln_patches[:, center_start:center_end, center_start:center_end]
    ones_mask_center = ones_mask[:, center_start:center_end, center_start:center_end].float()
    
    # Step 2: Use barycentric interpolation for final assembly
    B = soln_patches_center.shape[0]
    canvas_positions = coord_centers.squeeze(1).squeeze(1) - com.unsqueeze(0)
    
    # Determine canvas size
    max_offset = torch.ceil(torch.max(torch.abs(canvas_positions))).int().item()
    canvas_size = (middle + 2 * max_offset, middle + 2 * max_offset)
    
    # Adjust positions to canvas coordinates
    canvas_center = torch.tensor([canvas_size[1] // 2, canvas_size[0] // 2], 
                                device=device, dtype=torch.float32)
    canvas_positions_adjusted = canvas_positions + canvas_center.unsqueeze(0)
    
    # Initialize canvas
    canvas = torch.zeros(canvas_size, device=device, dtype=soln_patches_center.dtype)
    canvas_counts = torch.zeros(canvas_size, device=device, dtype=torch.float32)
    
    # Create interpolator for solution patches
    interpolator = BatchedObjectPatchInterpolator(
        canvas_shape=canvas_size,
        positions_px=canvas_positions_adjusted,
        patch_size=middle,
        device=device,
        dtype=soln_patches_center.dtype
    )
    
    # Accumulate solution patches onto canvas
    interpolator.accumulate_patches(canvas, canvas_counts, soln_patches_center)
    
    return canvas, canvas_counts


# High-performance vectorized version
class VectorizedBarycentricAccumulator:
    """
    Fully vectorized barycentric accumulation for maximum performance.
    Processes all patches simultaneously using advanced indexing.
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
            # Filter to valid patches only
            valid_idx = torch.where(valid_mask)[0]
            patches = patches[valid_idx]
            xmin_wh, ymin_wh = xmin_wh[valid_idx], ymin_wh[valid_idx]
            xmin_fr, ymin_fr = xmin_fr[valid_idx], ymin_fr[valid_idx]
            N = valid_idx.shape[0]
        
        # Compute weights
        xmin_fr_c = 1.0 - xmin_fr
        ymin_fr_c = 1.0 - ymin_fr
        
        w00 = (ymin_fr_c * xmin_fr_c).unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1)
        w01 = (ymin_fr_c * xmin_fr).unsqueeze(-1).unsqueeze(-1)
        w10 = (ymin_fr * xmin_fr_c).unsqueeze(-1).unsqueeze(-1)
        w11 = (ymin_fr * xmin_fr).unsqueeze(-1).unsqueeze(-1)
        
        # Create index tensors for all patches simultaneously
        # This is the key vectorization step
        patch_y, patch_x = torch.meshgrid(
            torch.arange(patch_size, device=self.device),
            torch.arange(patch_size, device=self.device),
            indexing='ij'
        )
        
        # Expand to all patches: (N, patch_size, patch_size)
        patch_y_exp = patch_y.unsqueeze(0).expand(N, -1, -1)
        patch_x_exp = patch_x.unsqueeze(0).expand(N, -1, -1)
        
        # Canvas coordinates for each patch pixel
        canvas_y_base = ymin_wh.unsqueeze(-1).unsqueeze(-1) + patch_y_exp
        canvas_x_base = xmin_wh.unsqueeze(-1).unsqueeze(-1) + patch_x_exp
        
        # Flatten for advanced indexing - use reshape to handle non-contiguous tensors
        patches_flat = patches.reshape(N, -1)  # (N, patch_size²)
        canvas_y_flat = canvas_y_base.reshape(N, -1)  # (N, patch_size²)
        canvas_x_flat = canvas_x_base.reshape(N, -1)
        
        # Weights (broadcast across patch pixels) - use reshape for safety
        w00_flat = w00.expand(-1, patch_size, patch_size).reshape(N, -1)
        w01_flat = w01.expand(-1, patch_size, patch_size).reshape(N, -1)
        w10_flat = w10.expand(-1, patch_size, patch_size).reshape(N, -1)
        w11_flat = w11.expand(-1, patch_size, patch_size).reshape(N, -1)
        
        # Vectorized accumulation using scatter_add - use reshape for all tensor flattening
        weighted_patches_00 = (patches_flat * w00_flat).reshape(-1)
        weighted_patches_01 = (patches_flat * w01_flat).reshape(-1)
        weighted_patches_10 = (patches_flat * w10_flat).reshape(-1)
        weighted_patches_11 = (patches_flat * w11_flat).reshape(-1)
        
        # Canvas indices (flattened) - use reshape for safety
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
        
        # Update counts (same pattern with weights only) - use reshape for consistency
        counts_flat.scatter_add_(0, idx_00, w00_flat.reshape(-1))
        counts_flat.scatter_add_(0, idx_01, w01_flat.reshape(-1))
        counts_flat.scatter_add_(0, idx_10, w10_flat.reshape(-1))
        counts_flat.scatter_add_(0, idx_11, w11_flat.reshape(-1))


def memory_efficient_reconstruct_image(model: nn.Module,
                                     ptycho_dset: PtychoDataset,
                                     training_config: TrainingConfig,
                                     data_config: DataConfig,
                                     model_config: ModelConfig,
                                     inference_config: InferenceConfig,
                                     use_vectorized: bool = True) -> Tuple[torch.Tensor, Any]:
    """
    Memory-efficient reconstruction using barycentric coordinate accumulation.
    Eliminates the memory bottleneck from dynamic padding.
    """
    n_files = ptycho_dset.n_files
    experiment_number = inference_config.experiment_number
    
    # Get dataset subset
    if n_files > 1:
        ptycho_subset = ptycho_dset.get_experiment_dataset(experiment_number)
    else:
        ptycho_subset = ptycho_dset
    
    device = training_config.device
    
    # Pre-compute constants
    global_coords = ptycho_subset.mmap_ptycho['coords_global'].squeeze()
    
    if 'com' in ptycho_subset.data_dict:
        center_of_mass = torch.mean(global_coords, 
                                  dim=tuple(range(global_coords.dim()-1)))
    else:
        center_of_mass = ptycho_subset.data_dict['com']
    
    center_of_mass = center_of_mass.to(device)
    
    # Determine canvas size based on coordinate range
    adjusted_coords = global_coords - center_of_mass.cpu()
    max_offset = torch.ceil(torch.max(torch.abs(adjusted_coords))).int().item()
    canvas_size = (inference_config.middle_trim + 2 * max_offset, 
                   inference_config.middle_trim + 2 * max_offset)
    
    print(f"Canvas size: {canvas_size}, Max offset: {max_offset}")
    print(f"Memory savings vs padding: {((canvas_size[0] / (data_config.N + 2 * max_offset))**2):.2f}x")
    
    # Initialize canvas
    canvas = torch.zeros(canvas_size, device=device, dtype=torch.complex64)
    canvas_counts = torch.zeros(canvas_size, device=device, dtype=torch.float32)
    
    # Create dataloader
    infer_loader = TensorDictDataLoader(
        ptycho_subset, 
        batch_size=inference_config.batch_size,
        collate_fn=Collate(device=device)
    )
    
    # Choose accumulation method
    if use_vectorized:
        accumulator = VectorizedBarycentricAccumulator(canvas_size, device)
    
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(infer_loader):
            start_time = time.time()
            
            # Efficient device transfer
            batch_data = batch[0]
            x = batch_data['images'].to(device, non_blocking=True)
            positions = batch_data['coords_relative'].to(device, non_blocking=True)
            probe = batch[1].to(device, non_blocking=True)
            in_scale = batch_data['rms_scaling_constant'].to(device, non_blocking=True)
            batch_coords_global = batch_data['coords_global'].to(device, non_blocking=True)
            
            # Model inference
            batch_output = model.forward_predict(x, positions, probe, in_scale)
            
            # Extract center regions
            N = data_config.N
            middle = inference_config.middle_trim
            center_start = N // 2 - middle // 2
            center_end = N // 2 + middle // 2
            
            if data_config.C == 1:
                im_center = batch_output[:, :, center_start:center_end, center_start:center_end].squeeze(1)
                batch_coords_2d = batch_coords_global.squeeze()
            else:
                # Handle multi-channel case
                B, C = batch_output.shape[:2]
                im_center = batch_output[:, :, center_start:center_end, center_start:center_end]
                im_center = im_center.view(B * C, middle, middle)
                batch_coords_2d = batch_coords_global.view(B * C, 2)
            
            # Compute canvas positions for this batch
            relative_positions = batch_coords_2d - center_of_mass.unsqueeze(0)
            canvas_center = torch.tensor([canvas_size[1] // 2, canvas_size[0] // 2], 
                                       device=device, dtype=torch.float32)
            canvas_positions = relative_positions + canvas_center.unsqueeze(0)
            
            # Accumulate using barycentric interpolation
            if use_vectorized:
                accumulator.accumulate_batch(canvas, canvas_counts, im_center, 
                                           canvas_positions, middle)
            else:
                # Use the simpler BatchedObjectPatchInterpolator
                interpolator = BatchedObjectPatchInterpolator(
                    canvas_shape=canvas_size,
                    positions_px=canvas_positions,
                    patch_size=middle,
                    device=device,
                    dtype=im_center.dtype
                )
                interpolator.accumulate_patches(canvas, canvas_counts, im_center)
            
            # Memory cleanup
            del x, positions, probe, in_scale, batch_coords_global
            del batch_output, im_center, canvas_positions
            
            if i % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            print(f'Batch {i+1}/{len(infer_loader)} completed in {time.time() - start_time:.3f} seconds')
    
    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    return canvas/canvas_counts, ptycho_subset


# Comparison utility
def estimate_barycentric_memory_usage(batch_size: int, 
                                     data_config: DataConfig, 
                                     middle: int,
                                     max_offset: int,
                                     safety_factor: float = 1.5) -> float:
    """
    Estimate GPU memory usage for barycentric reassembly approach.
    
    Args:
        batch_size: Number of samples per batch
        data_config: DataConfig containing N and C
        middle: Size of extracted center region
        max_offset: Maximum coordinate offset
        safety_factor: Multiplicative safety factor
    
    Returns:
        Estimated memory usage in GB
    """
    N = data_config.N
    C = data_config.C
    
    # Memory components (in bytes):
    # 1. Input tensors: batch_size * C * N * N * 8 (complex64)
    input_memory = batch_size * C * N * N * 8
    
    # 2. Model activations (conservative estimate)
    model_memory = input_memory * 3
    
    # 3. Center extracted patches: batch_size * C * middle * middle * 8
    center_patches_memory = batch_size * C * middle * middle * 8
    
    # 4. Canvas memory: (middle + 2*max_offset)² * 8 (complex64)
    canvas_size = middle + 2 * max_offset
    canvas_memory = canvas_size * canvas_size * 8
    
    # 5. Canvas counts: same size as canvas but float32
    canvas_counts_memory = canvas_size * canvas_size * 4
    
    # 6. Barycentric weights and indices: batch_size * C * middle² * 4 * 4 (4 weights per pixel, float32)
    barycentric_memory = batch_size * C * middle * middle * 4 * 4
    
    # 7. Coordinate tensors: batch_size * C * 2 * 4 (float32)
    coord_memory = batch_size * C * 2 * 4
    
    total_memory = (input_memory + model_memory + center_patches_memory + 
                   canvas_memory + canvas_counts_memory + barycentric_memory + coord_memory)
    
    return total_memory * safety_factor / (1024**3)  # Convert to GB


def get_optimal_barycentric_batch_size(original_batch_size: int,
                                      data_config: DataConfig,
                                      middle: int,
                                      max_offset: int,
                                      max_memory_gb: Optional[float] = None,
                                      min_batch_size: int = 1) -> int:
    """
    Determine optimal batch size for barycentric reassembly based on GPU memory.
    
    Args:
        original_batch_size: Desired batch size
        data_config: DataConfig object
        middle: Size of center region to extract
        max_offset: Maximum coordinate offset
        max_memory_gb: Maximum memory to use (None for auto-detection)
        min_batch_size: Minimum allowable batch size
    
    Returns:
        Optimal batch size that fits in memory
    """
    if not torch.cuda.is_available():
        return original_batch_size
    
    # Auto-detect available memory if not specified
    if max_memory_gb is None:
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # Use 80% of total GPU memory (barycentric is more memory efficient)
        max_memory_gb = total_memory * 0.8
        print(f"Auto-detected GPU memory limit: {max_memory_gb:.2f}GB")
    
    # Test if original batch size fits
    estimated_usage = estimate_barycentric_memory_usage(
        original_batch_size, data_config, middle, max_offset
    )
    
    print(f"Estimated memory for batch_size {original_batch_size}: {estimated_usage:.2f}GB")
    
    if estimated_usage <= max_memory_gb:
        return original_batch_size
    
    # Binary search for optimal batch size
    left, right = min_batch_size, original_batch_size
    optimal_batch_size = min_batch_size
    
    while left <= right:
        mid = (left + right) // 2
        estimated_usage = estimate_barycentric_memory_usage(
            mid, data_config, middle, max_offset
        )
        
        if estimated_usage <= max_memory_gb:
            optimal_batch_size = mid
            left = mid + 1
        else:
            right = mid - 1
    
    print(f"Optimal batch size found: {optimal_batch_size}")
    print(f"Final estimated memory usage: {estimate_barycentric_memory_usage(optimal_batch_size, data_config, middle, max_offset):.2f}GB")
    
    return optimal_batch_size


def adaptive_memory_reconstruct_image(model: nn.Module,
                                    ptycho_dset: PtychoDataset,
                                    training_config: TrainingConfig,
                                    data_config: DataConfig,
                                    model_config: ModelConfig,
                                    inference_config: InferenceConfig,
                                    max_memory_gb: Optional[float] = None,
                                    force_barycentric: bool = True,
                                    verbose: bool = True) -> Tuple[torch.Tensor, Any]:
    """
    Adaptive reconstruction that automatically chooses between padding and barycentric methods
    based on memory efficiency, with dynamic batch size optimization.
    
    Args:
        force_barycentric: If True, always use barycentric method regardless of memory comparison
    """
    n_files = ptycho_dset.n_files
    experiment_number = inference_config.experiment_number
    
    # Get dataset subset
    if n_files > 1:
        ptycho_subset = ptycho_dset.get_experiment_dataset(experiment_number)
    else:
        ptycho_subset = ptycho_dset
    
    device = training_config.device
    
    # Pre-compute constants
    global_coords = ptycho_subset.mmap_ptycho['coords_global'].squeeze()
    
    if 'com' in ptycho_subset.data_dict:
        center_of_mass = torch.mean(global_coords, 
                                  dim=tuple(range(global_coords.dim()-1)))
    else:
        center_of_mass = ptycho_subset.data_dict['com']
    
    center_of_mass = center_of_mass.to(device)
    
    # Compute max offset
    adjusted_coords = global_coords - center_of_mass.cpu()
    max_offset = torch.ceil(torch.max(torch.abs(adjusted_coords))).int().item()
    middle = inference_config.middle_trim
    
    if verbose:
        print(f"Max offset: {max_offset}, Middle size: {middle}")
    
    # Choose method based on memory efficiency
    if force_barycentric:
        method = "barycentric"
    
    # Optimize batch size for chosen method
    if method == "barycentric":
        optimal_batch_size = get_optimal_barycentric_batch_size(
            inference_config.batch_size, data_config, middle, max_offset, max_memory_gb
        )
        
        # Create new config with optimal batch size
        if optimal_batch_size != inference_config.batch_size:
            optimized_config = InferenceConfig(
                batch_size=optimal_batch_size,
                experiment_number=inference_config.experiment_number,
                middle_trim=inference_config.middle_trim
            )
        else:
            optimized_config = inference_config
        
        # Use barycentric reconstruction
        return memory_efficient_reconstruct_image(
            model, ptycho_dset, training_config, data_config, 
            model_config, optimized_config, use_vectorized=True
        )


#--- Multi-GPU slop---

class PtychoDataParallelWrapper(nn.Module):
    """
    Wrapper to make ptychography models compatible with nn.DataParallel.
    Handles the multiple input signature of forward_predict().
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def forward(self, inputs):
        """
        Forward pass that unpacks multiple inputs for DataParallel compatibility.
        
        Args:
            inputs: Tuple of (x, positions, probe, in_scale)
        
        Returns:
            Model output tensor
        """
        x, positions, probe, in_scale = inputs

        return self.model.forward_predict(x, positions, probe, in_scale)


def setup_dataparallel_model(model: nn.Module, 
                            gpu_ids: Optional[List[int]] = None,
                            verbose: bool = True) -> Tuple[nn.Module, List[int], torch.device]:
    """
    Setup model with DataParallel for multi-GPU inference.
    
    Args:
        model: The ptychography model to parallelize
        gpu_ids: List of GPU IDs to use (None for auto-detection)
        verbose: Whether to print setup information
    
    Returns:
        Tuple of (parallel_model, gpu_ids_used, primary_device)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available - multi-GPU requires CUDA")
    
    total_gpus = torch.cuda.device_count()
    
    if gpu_ids is None:
        # Use all available GPUs
        gpu_ids = list(range(total_gpus))
    else:
        # Validate provided GPU IDs
        invalid_ids = [gid for gid in gpu_ids if gid >= total_gpus]
        if invalid_ids:
            raise ValueError(f"Invalid GPU IDs {invalid_ids}. Available GPUs: 0-{total_gpus-1}")
    
    if len(gpu_ids) == 1:
        if verbose:
            print(f"Only 1 GPU specified ({gpu_ids[0]}). Using single-GPU mode.")
        primary_device = torch.device(f'cuda:{gpu_ids[0]}')
        model = model.to(primary_device)
        return model, gpu_ids, primary_device
    
    primary_device = torch.device(f'cuda:{gpu_ids[0]}')
    
    # Print GPU information
    if verbose:
        print(f"Setting up DataParallel on {len(gpu_ids)} GPUs:")
        for gpu_id in gpu_ids:
            props = torch.cuda.get_device_properties(gpu_id)
            memory_gb = props.total_memory / (1024**3)
            print(f"  GPU {gpu_id}: {props.name} ({memory_gb:.1f}GB)")
    
    # Move model to primary GPU first
    model = model.to(primary_device)
    
    # Wrap model for DataParallel compatibility
    wrapped_model = PtychoDataParallelWrapper(model)
    
    # Create DataParallel model
    parallel_model = nn.DataParallel(wrapped_model, device_ids=gpu_ids)
    
    return parallel_model, gpu_ids, primary_device


def get_multi_gpu_optimal_batch_size(original_batch_size: int,
                                    data_config: DataConfig,
                                    middle: int,
                                    max_offset: int,
                                    gpu_ids: List[int],
                                    memory_utilization: float = 0.75,
                                    min_batch_size: int = 1) -> int:
    """
    Optimize batch size for DataParallel setup considering total memory pool.
    
    Args:
        original_batch_size: Desired batch size
        data_config: DataConfig object
        middle: Size of center region
        max_offset: Maximum coordinate offset
        gpu_ids: List of GPU IDs to use
        memory_utilization: Fraction of GPU memory to use per device
        min_batch_size: Minimum allowable batch size
    
    Returns:
        Optimal batch size for multi-GPU setup
    """
    # Find the GPU with minimum memory (bottleneck)
    min_memory_gb = float('inf')
    total_memory_gb = 0
    
    for gpu_id in gpu_ids:
        props = torch.cuda.get_device_properties(gpu_id)
        gpu_memory_gb = props.total_memory / (1024**3)
        total_memory_gb += gpu_memory_gb
        min_memory_gb = min(min_memory_gb, gpu_memory_gb)
    
    # With DataParallel, each GPU processes batch_size/num_gpus samples
    # But we also need to account for model replication overhead
    effective_memory_per_gpu = min_memory_gb * memory_utilization
    
    print(f"Multi-GPU memory analysis:")
    print(f"  Total memory: {total_memory_gb:.1f}GB across {len(gpu_ids)} GPUs")
    print(f"  Minimum GPU memory: {min_memory_gb:.1f}GB (bottleneck)")
    print(f"  Effective memory per GPU: {effective_memory_per_gpu:.1f}GB")
    
    # Start with a conservative estimate: original_batch_size * num_gpus
    # DataParallel can potentially handle larger total batches
    max_total_batch = original_batch_size * len(gpu_ids)
    
    # Binary search for optimal batch size
    left, right = min_batch_size, max_total_batch
    optimal_batch_size = min_batch_size
    
    while left <= right:
        mid = (left + right) // 2
        
        # Estimate memory per GPU for this total batch size
        per_gpu_batch = math.ceil(mid / len(gpu_ids))
        estimated_memory_per_gpu = estimate_barycentric_memory_usage(
            per_gpu_batch, data_config, middle, max_offset, safety_factor=1.3
        )
        
        if estimated_memory_per_gpu <= effective_memory_per_gpu:
            optimal_batch_size = mid
            left = mid + 1
        else:
            right = mid - 1
    
    print(f"Optimal total batch size: {optimal_batch_size}")
    print(f"Per-GPU batch size: ~{math.ceil(optimal_batch_size / len(gpu_ids))}")
    
    return optimal_batch_size


def multi_gpu_memory_efficient_reconstruct_image(model: nn.Module,
                                                ptycho_dset: PtychoDataset,
                                                training_config: TrainingConfig,
                                                data_config: DataConfig,
                                                model_config: ModelConfig,
                                                inference_config: InferenceConfig,
                                                gpu_ids: Optional[List[int]] = None,
                                                max_memory_gb: Optional[float] = None,
                                                use_vectorized: bool = True,
                                                verbose: bool = True) -> Tuple[torch.Tensor, Any]:
    """
    Multi-GPU memory-efficient reconstruction using DataParallel and barycentric accumulation.
    
    This function uses nn.DataParallel to distribute model inference across multiple GPUs
    while maintaining efficient canvas assembly on the primary GPU.
    
    Args:
        model: Neural network model
        ptycho_dset: Ptychography dataset
        training_config: Training configuration
        data_config: Data configuration  
        model_config: Model configuration
        inference_config: Inference configuration
        gpu_ids: List of GPU IDs to use (None for auto-detection)
        max_memory_gb: Maximum memory per GPU (None for auto-detection)
        use_vectorized: Whether to use vectorized accumulation
        verbose: Whether to print detailed progress
    
    Returns:
        Tuple of (reconstructed_canvas, dataset_subset)
    """
    # Setup DataParallel model
    parallel_model, used_gpu_ids, primary_device = setup_dataparallel_model(
        model, gpu_ids, verbose
    )
    
    # If only 1 GPU, fall back to single-GPU implementation
    if len(used_gpu_ids) == 1:
        if verbose:
            print("Falling back to single-GPU implementation")
        return memory_efficient_reconstruct_image(
            model, ptycho_dset, training_config, data_config,
            model_config, inference_config, use_vectorized
        )
    
    n_files = ptycho_dset.n_files
    experiment_number = inference_config.experiment_number
    
    # Get dataset subset
    if n_files > 1:
        ptycho_subset = ptycho_dset.get_experiment_dataset(experiment_number)
    else:
        ptycho_subset = ptycho_dset
    
    # Pre-compute constants on primary GPU
    global_coords = ptycho_subset.mmap_ptycho['coords_global'].squeeze()
    
    if 'com' in ptycho_subset.data_dict:
        center_of_mass = torch.mean(global_coords, 
                                  dim=tuple(range(global_coords.dim()-1)))
    else:
        center_of_mass = ptycho_subset.data_dict['com']
    
    center_of_mass = center_of_mass.to(primary_device)
    
    # Determine canvas size based on coordinate range
    adjusted_coords = global_coords - center_of_mass.cpu()
    max_offset = torch.ceil(torch.max(torch.abs(adjusted_coords))).int().item()
    canvas_size = (inference_config.middle_trim + 2 * max_offset, 
                   inference_config.middle_trim + 2 * max_offset)
    
    if verbose:
        print(f"Multi-GPU setup complete:")
        print(f"  Canvas size: {canvas_size}")
        print(f"  Max offset: {max_offset}")
        print(f"  Using {len(used_gpu_ids)} GPUs: {used_gpu_ids}")
    
    # Optimize batch size for multi-GPU
    optimal_batch_size = get_multi_gpu_optimal_batch_size(
        inference_config.batch_size, data_config, inference_config.middle_trim, 
        max_offset, used_gpu_ids, memory_utilization=0.75
    )
    
    # Initialize canvas on primary GPU
    canvas = torch.zeros(canvas_size, device=primary_device, dtype=torch.complex64)
    canvas_counts = torch.zeros(canvas_size, device=primary_device, dtype=torch.float32)
    
    # Create dataloader with optimized batch size
    infer_loader = TensorDictDataLoader(
        ptycho_subset, 
        batch_size=optimal_batch_size,
        collate_fn=Collate(device=primary_device)
    )
    
    # Choose accumulation method
    if use_vectorized:
        accumulator = VectorizedBarycentricAccumulator(canvas_size, primary_device)
    
    # Memory efficiency comparison
    single_gpu_memory = estimate_barycentric_memory_usage(
        inference_config.batch_size, data_config, inference_config.middle_trim, max_offset
    )
    multi_gpu_memory_per_gpu = estimate_barycentric_memory_usage(
        math.ceil(optimal_batch_size / len(used_gpu_ids)), data_config, 
        inference_config.middle_trim, max_offset
    )
    
    if verbose:
        print(f"Memory comparison:")
        print(f"  Single GPU batch size {inference_config.batch_size}: {single_gpu_memory:.2f}GB")
        print(f"  Multi GPU per-GPU memory: {multi_gpu_memory_per_gpu:.2f}GB")
        print(f"  Total batch size: {optimal_batch_size}")
        print(f"  Theoretical speedup: {len(used_gpu_ids)}x")
    
    total_batches = len(infer_loader)
    total_inference_time = 0
    total_assembly_time = 0
    
    parallel_model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(infer_loader):
            batch_start_time = time.time()
            
            # Prepare data on primary GPU
            print("Before loading tensors")
            profile_memory()
            
            batch_data = batch[0]
            x = batch_data['images'].to(primary_device, non_blocking=True)
            positions = batch_data['coords_relative'].to(primary_device, non_blocking=True)
            probe = batch[1].to(primary_device, non_blocking=True)
            in_scale = batch_data['rms_scaling_constant'].to(primary_device, non_blocking=True)
            batch_coords_global = batch_data['coords_global'].to(primary_device, non_blocking=True)

            print("After loading batch tensors")
            profile_memory()
            
            # Multi-GPU model inference using DataParallel
            inference_start = time.time()
            
            # Package inputs for DataParallel
            inputs = (x, positions, probe, in_scale)
            batch_output = parallel_model(inputs)

            print("After model inference")
            profile_memory()
            
            torch.cuda.synchronize()  # Ensure all GPU operations complete
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            # Canvas assembly (on primary GPU) - unchanged from your original
            assembly_start = time.time()
            
            # Extract center regions
            N = data_config.N
            middle = inference_config.middle_trim
            center_start = N // 2 - middle // 2
            center_end = N // 2 + middle // 2
            
            # Debug shapes to understand the issue
            if verbose and i == 0:
                print(f"Debug - Batch shapes:")
                print(f"  batch_output: {batch_output.shape}")
                print(f"  batch_coords_global: {batch_coords_global.shape}")
                print(f"  data_config.C: {data_config.C}")

            if batch_output.dim() == 5:
                batch_output = batch_output.squeeze(-1) 
                print(f"  batch_output: {batch_output.shape}")
                       
            if data_config.C == 1:
                # Single channel case
                im_center = batch_output[:, :, center_start:center_end, center_start:center_end].squeeze(1)
                batch_coords_2d = batch_coords_global.squeeze()
            else:
                # Multi-channel case - handle 5D output [B, C, H, W, 2]
                B, C = batch_output.shape[:2]
                
                # Extract center regions keeping the complex dimension
                im_center = batch_output[:, :, center_start:center_end, center_start:center_end, :]
                # Reshape to [B*C, middle, middle, 2]
                im_center = im_center.view(B * C, middle, middle, 2)
                
                # Convert [B*C, H, W, 2] to complex [B*C, H, W]
                if im_center.shape[-1] == 2:
                    im_center = torch.complex(im_center[..., 0], im_center[..., 1])
                
                # Handle coordinates: [B, C, 1, 2] -> [B*C, 2]
                batch_coords_2d = batch_coords_global.squeeze(2).view(B * C, 2)
                
                if verbose and i == 0:
                    print(f"  After processing:")
                    print(f"    im_center: {im_center.shape}")
                    print(f"    batch_coords_2d: {batch_coords_2d.shape}")

            # Compute canvas positions for this batch
            relative_positions = batch_coords_2d - center_of_mass.unsqueeze(0)
            canvas_center = torch.tensor([canvas_size[1] // 2, canvas_size[0] // 2], 
                                       device=primary_device, dtype=torch.float32)
            canvas_positions = relative_positions + canvas_center.unsqueeze(0)

            print("Before accumulation")
            profile_memory()
            
            # Accumulate using barycentric interpolation (your existing optimized code)
            if use_vectorized:
                accumulator.accumulate_batch(canvas, canvas_counts, im_center, 
                                           canvas_positions, middle)
            else:
                # Use the simpler BatchedObjectPatchInterpolator
                interpolator = BatchedObjectPatchInterpolator(
                    canvas_shape=canvas_size,
                    positions_px=canvas_positions,
                    patch_size=middle,
                    device=primary_device,
                    dtype=im_center.dtype
                )
                interpolator.accumulate_patches(canvas, canvas_counts, im_center)

            print("After accumulation")
            profile_memory()
            
            assembly_time = time.time() - assembly_start
            total_assembly_time += assembly_time
            
            # Memory cleanup
            del x, positions, probe, in_scale, batch_coords_global
            del batch_output, im_center, canvas_positions
            
            # Clean up all GPUs used by DataParallel
            for gpu_id in used_gpu_ids:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
            gc.collect()
            
            if verbose and (i % 5 == 0 or i == total_batches - 1):
                batch_time = time.time() - batch_start_time
                efficiency = inference_time / assembly_time if assembly_time > 0 else float('inf')
                print(f'Batch {i+1}/{total_batches}: {batch_time:.3f}s '
                      f'(inference: {inference_time:.3f}s, assembly: {assembly_time:.3f}s, '
                      f'parallel_efficiency: {efficiency:.1f}x)')
    
    # Performance summary
    if verbose:
        avg_inference_time = total_inference_time / total_batches
        avg_assembly_time = total_assembly_time / total_batches
        parallel_efficiency = avg_inference_time / avg_assembly_time if avg_assembly_time > 0 else float('inf')
        
        print(f"\nPerformance Summary:")
        print(f"  Average inference time per batch: {avg_inference_time:.3f}s")
        print(f"  Average assembly time per batch: {avg_assembly_time:.3f}s")
        print(f"  Parallel efficiency ratio: {parallel_efficiency:.1f}x")
        print(f"  Total reconstruction time: {total_inference_time + total_assembly_time:.2f}s")
        print(f"  Effective GPU utilization: {len(used_gpu_ids)} GPUs")
    
    # Final cleanup
    for gpu_id in used_gpu_ids:
        with torch.cuda.device(gpu_id):
            torch.cuda.empty_cache()
    gc.collect()
    
    return canvas / canvas_counts, ptycho_subset


def estimate_dataparallel_memory_usage(batch_size: int,
                                      data_config: DataConfig,
                                      middle: int,
                                      max_offset: int,
                                      num_gpus: int,
                                      safety_factor: float = 1.3) -> Tuple[float, float]:
    """
    Estimate memory usage for DataParallel setup.
    
    Returns:
        per_gpu_memory_gb: Memory usage per GPU (including model replication)
        primary_gpu_memory_gb: Additional memory on primary GPU (canvas)
    """
    # Each GPU processes approximately batch_size/num_gpus samples
    per_gpu_batch_size = math.ceil(batch_size / num_gpus)
    
    # Base memory per GPU (model inference)
    per_gpu_memory = estimate_barycentric_memory_usage(
        per_gpu_batch_size, data_config, middle, max_offset, safety_factor=1.0
    )
    
    # DataParallel overhead: model replication, communication buffers
    # Estimate ~20% overhead for DataParallel mechanics
    dataparallel_overhead = per_gpu_memory * 0.2
    
    # Primary GPU additional memory (canvas + gather operations)
    canvas_size = middle + 2 * max_offset
    canvas_memory_gb = 2 * canvas_size * canvas_size * 8 / (1024**3)  # canvas + counts
    
    # Temporary memory for gathering results from other GPUs
    gather_overhead_gb = batch_size * data_config.C * middle * middle * 8 / (1024**3)
    
    total_per_gpu = (per_gpu_memory + dataparallel_overhead) * safety_factor
    total_primary_gpu = total_per_gpu + canvas_memory_gb + gather_overhead_gb
    
    return total_per_gpu, total_primary_gpu


def adaptive_multi_gpu_reconstruct_image(model: nn.Module,
                                        ptycho_dset: PtychoDataset,
                                        training_config: TrainingConfig,
                                        data_config: DataConfig,
                                        model_config: ModelConfig,
                                        inference_config: InferenceConfig,
                                        gpu_ids: Optional[List[int]] = None,
                                        max_memory_gb: Optional[float] = None,
                                        force_multi_gpu: bool = False,
                                        verbose: bool = True) -> Tuple[torch.Tensor, Any]:
    """
    Adaptive reconstruction that automatically chooses between single and multi-GPU
    based on efficiency and availability.
    
    Args:
        force_multi_gpu: Force multi-GPU even if single GPU might be more efficient
        
    Returns:
        Tuple of (reconstructed_canvas, dataset_subset)
    """
    try:
        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        total_gpus = torch.cuda.device_count()
        if total_gpus < 2 and not force_multi_gpu:
            if verbose:
                print("Only 1 GPU available, using single-GPU implementation")
            return memory_efficient_reconstruct_image(
                model, ptycho_dset, training_config, data_config,
                model_config, inference_config, use_vectorized=True
            )
        
        # Setup GPU configuration
        if gpu_ids is None:
            gpu_ids = list(range(min(total_gpus, 4)))  # Limit to 4 GPUs for stability
        
        if len(gpu_ids) < 2 and not force_multi_gpu:
            if verbose:
                print("Less than 2 GPUs specified, using single-GPU implementation")
            return memory_efficient_reconstruct_image(
                model, ptycho_dset, training_config, data_config,
                model_config, inference_config, use_vectorized=True
            )
        
        # Estimate if multi-GPU will be beneficial
        if verbose:
            print(f"Attempting DataParallel reconstruction with {len(gpu_ids)} GPUs")
        
        # Try multi-GPU reconstruction
        return multi_gpu_memory_efficient_reconstruct_image(
            model, ptycho_dset, training_config, data_config,
            model_config, inference_config, gpu_ids, max_memory_gb,
            True, verbose
        )
        
    except Exception as e:
        if verbose:
            print(f"Multi-GPU reconstruction failed: {e}")
            print("Falling back to single-GPU implementation")
        
        # Fallback to single GPU on primary device
        original_device = training_config.device
        try:
            # Try to use the original device
            single_gpu_config = TrainingConfig(
                device=original_device,
                **{k: v for k, v in training_config.__dict__.items() if k != 'device'}
            )
        except:
            # If that fails, use cuda:0
            single_gpu_config = TrainingConfig(
                device=torch.device('cuda:0'),
                **{k: v for k, v in training_config.__dict__.items() if k != 'device'}
            )
        
        # Ensure model is on correct device for fallback
        model = model.to(single_gpu_config.device)
        
        return memory_efficient_reconstruct_image(
            model, ptycho_dset, single_gpu_config, data_config,
            model_config, inference_config, use_vectorized=True
        )


# Utility function for testing multi-GPU setup
def test_multi_gpu_setup(model: nn.Module, 
                        sample_batch_size: int = 8,
                        gpu_ids: Optional[List[int]] = None) -> bool:
    """
    Test multi-GPU setup with a small dummy batch to verify functionality.
    
    Returns:
        True if multi-GPU setup works, False otherwise
    """
    try:
        parallel_model, used_gpu_ids, primary_device = setup_dataparallel_model(
            model, gpu_ids, verbose=True
        )
        
        if len(used_gpu_ids) < 2:
            print("Not enough GPUs for testing")
            return False
        
        print("Testing multi-GPU setup with dummy data...")
        
        # Create dummy inputs matching your model's expected shapes
        # You'll need to adjust these dimensions based on your actual model
        dummy_x = torch.randn(sample_batch_size, 1, 64, 64, device=primary_device)
        dummy_positions = torch.randn(sample_batch_size, 1, 1, 2, device=primary_device)
        dummy_probe = torch.randn(64, 64, device=primary_device, dtype=torch.complex64)
        dummy_scale = torch.ones(sample_batch_size, device=primary_device)
        
        # Test forward pass
        with torch.no_grad():
            inputs = (dummy_x, dummy_positions, dummy_probe, dummy_scale)
            output = parallel_model(inputs)
            
        print(f"✓ Multi-GPU test successful! Output shape: {output.shape}")
        print(f"✓ Using {len(used_gpu_ids)} GPUs: {used_gpu_ids}")
        
        # Cleanup
        del dummy_x, dummy_positions, dummy_probe, dummy_scale, output
        for gpu_id in used_gpu_ids:
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"✗ Multi-GPU test failed: {e}")
        return False
    
class BarycentricTranslator:
    """
    Memory-efficient barycentric translation alternative to grid_sample.
    
    Instead of creating massive grids, this directly accumulates patches
    onto a canvas using barycentric interpolation - similar to your 
    existing barycentric reassembly but optimized for translation.
    """
    
    def __init__(self, canvas_size: Tuple[int, int], device: torch.device):
        self.canvas_size = canvas_size
        self.device = device
        self.H, self.W = canvas_size
    
    def translate_and_accumulate(self, 
                                patches: torch.Tensor,     # (B*C, H, W) 
                                offsets: torch.Tensor,     # (B*C, 1, 2)
                                canvas: torch.Tensor,      # (B, H_canvas, W_canvas)
                                counts: torch.Tensor,      # (B, H_canvas, W_canvas)
                                batch_idx: int,            # Which batch this belongs to
                                agg: bool = True) -> None:
        """
        Translate patches and accumulate directly onto canvas using barycentric interpolation.
        This replaces the memory-intensive Translation + summation approach.
        
        Args:
            patches: Input patches to translate (B*C, H, W)
            offsets: Translation offsets (B*C, 1, 2) in pixel units
            canvas: Output canvas to accumulate onto
            counts: Count tensor for normalization
            batch_idx: Which batch in the canvas this corresponds to
            agg: Whether to aggregate overlaps
        """
        N_patches, patch_H, patch_W = patches.shape
        
        # Flatten offsets: (B*C, 1, 2) -> (B*C, 2)
        offsets_2d = offsets.squeeze(1)  # (B*C, 2)
        
        if agg:
            # Create center mask for patches (equivalent to your prototype mask)
            center_slice_h = slice(patch_H // 4, patch_H // 4 + patch_H // 2)
            center_slice_w = slice(patch_W // 4, patch_W // 4 + patch_W // 2)
            
            # Extract only the center regions that contribute to aggregation
            patches_center = patches[:, center_slice_h, center_slice_w]
            center_h, center_w = patches_center.shape[1:]
            
            # Adjust positions for center extraction
            center_offset_h = patch_H // 4
            center_offset_w = patch_W // 4
            
            # Calculate canvas positions for center regions
            canvas_positions = offsets_2d + torch.tensor([center_offset_w, center_offset_h], 
                                                        device=self.device, dtype=torch.float32)
            
            # Use barycentric accumulation for memory efficiency
            self._accumulate_patches_barycentric(
                patches_center, canvas_positions, canvas[batch_idx], counts[batch_idx]
            )
        else:
            # No aggregation - accumulate full patches
            canvas_positions = offsets_2d
            self._accumulate_patches_barycentric(
                patches, canvas_positions, canvas[batch_idx], counts[batch_idx]
            )
    
    def _accumulate_patches_barycentric(self, 
                                       patches: torch.Tensor,      # (N, H, W)
                                       positions: torch.Tensor,    # (N, 2)
                                       canvas: torch.Tensor,       # (H_canvas, W_canvas)
                                       counts: torch.Tensor) -> None:
        """
        Core barycentric accumulation - similar to your VectorizedBarycentricAccumulator
        but optimized for translation patterns.
        """
        N, patch_H, patch_W = patches.shape
        
        # Calculate patch corners in canvas coordinates
        # positions are the TOP-LEFT corners after translation
        x_start = positions[:, 0]  # (N,)
        y_start = positions[:, 1]  # (N,)
        
        # Integer and fractional parts for bilinear interpolation
        x_start_int = x_start.floor().long()
        y_start_int = y_start.floor().long()
        x_frac = x_start - x_start_int.float()
        y_frac = y_start - y_start_int.float()
        
        # Bounds check - only process patches that fit in canvas
        valid_mask = (
            (x_start_int >= 0) & (y_start_int >= 0) &
            (x_start_int + patch_W < self.W) & (y_start_int + patch_H < self.H)
        )
        
        if not valid_mask.all():
            # Filter to valid patches
            valid_idx = torch.where(valid_mask)[0]
            if len(valid_idx) == 0:
                return  # No valid patches
                
            patches = patches[valid_idx]
            x_start_int = x_start_int[valid_idx]
            y_start_int = y_start_int[valid_idx]
            x_frac = x_frac[valid_idx]
            y_frac = y_frac[valid_idx]
            N = len(valid_idx)
        
        # Bilinear interpolation weights
        w00 = (1 - x_frac) * (1 - y_frac)  # (N,)
        w01 = x_frac * (1 - y_frac)
        w10 = (1 - x_frac) * y_frac
        w11 = x_frac * y_frac
        
        # For each patch, accumulate onto canvas using bilinear weights
        for i in range(N):
            y_s, y_e = y_start_int[i], y_start_int[i] + patch_H
            x_s, x_e = x_start_int[i], x_start_int[i] + patch_W
            
            patch = patches[i]  # (patch_H, patch_W)
            
            # Accumulate with bilinear interpolation
            canvas[y_s:y_e, x_s:x_e] += w00[i] * patch
            canvas[y_s:y_e, x_s+1:x_e+1] += w01[i] * patch
            canvas[y_s+1:y_e+1, x_s:x_e] += w10[i] * patch
            canvas[y_s+1:y_e+1, x_s+1:x_e+1] += w11[i] * patch
            
            # Update counts for normalization
            counts[y_s:y_e, x_s:x_e] += w00[i]
            counts[y_s:y_e, x_s+1:x_e+1] += w01[i]
            counts[y_s+1:y_e+1, x_s:x_e] += w10[i]
            counts[y_s+1:y_e+1, x_s+1:x_e+1] += w11[i]


def barycentric_reassemble_patches_position_real(inputs: torch.Tensor, 
                                                offsets_xy: torch.Tensor,
                                                data_config, 
                                                model_config,
                                                agg: bool = True,
                                                padded_size: Optional[int] = None,
                                                **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Memory-efficient replacement for reassemble_patches_position_real using barycentric translation.
    
    This function eliminates the memory-intensive Translation function by directly accumulating
    patches onto a canvas using barycentric interpolation.
    
    Args:
        inputs: diffraction images from model -> (n_images, n_patches, N, N)
        offsets_xy: offset patches in x, y -> (n_images, n_patches, 1, 2)
        agg: aggregation boolean
        padded_size: Amount of padding
    
    Returns:
        imgs_merged: Assembled images (B, M, M)
        boolean_mask: Valid region mask (B, M, M) 
        M: Padded size used
    """
    assert inputs.dtype == torch.complex64, 'Input must be complex'
    
    B, C, N, _ = inputs.shape
    
    if padded_size is None:
        # You'll need to implement get_padded_size or pass it in
        # For now, using a simple estimate
        max_offset = torch.max(torch.abs(offsets_xy)).item()
        padded_size = N + 2 * int(math.ceil(max_offset))
    
    M = padded_size
    device = inputs.device
    
    # Initialize output tensors
    canvas = torch.zeros(B, M, M, device=device, dtype=torch.complex64)
    counts = torch.zeros(B, M, M, device=device, dtype=torch.float32)
    
    # Create barycentric translator
    translator = BarycentricTranslator((M, M), device)
    
    # Process each batch separately to avoid huge intermediate tensors
    for batch_idx in range(B):
        # Get patches and offsets for this batch: (C, N, N) and (C, 1, 2)
        batch_patches = inputs[batch_idx]  # (C, N, N)
        batch_offsets = offsets_xy[batch_idx]  # (C, 1, 2)
        
        # Pad patches to target size
        pad_dim = (M - N) // 2
        batch_patches_padded = F.pad(batch_patches, (pad_dim, pad_dim, pad_dim, pad_dim), "constant", 0.)
        
        # Adjust offsets for padding (so positions are relative to padded coordinate system)
        batch_offsets_adjusted = batch_offsets + pad_dim
        
        # Translate and accumulate directly onto canvas
        translator.translate_and_accumulate(
            batch_patches_padded,          # (C, M, M)
            batch_offsets_adjusted,        # (C, 1, 2)
            canvas,                        # (B, M, M)
            counts,                        # (B, M, M)
            batch_idx,                     # int
            agg=agg
        )
    
    if agg:
        # Normalize using counts
        boolean_mask = counts > 1e-6
        norm_factor = torch.clamp(counts, min=1.0)
        imgs_merged = (canvas * boolean_mask) / norm_factor
        
        return imgs_merged, boolean_mask, M
    else:
        # No aggregation case
        return canvas, torch.ones_like(counts, dtype=torch.bool), M


def vectorized_barycentric_reassemble_patches(inputs: torch.Tensor, 
                                            offsets_xy: torch.Tensor,
                                            data_config,
                                            model_config, 
                                            agg: bool = True,
                                            padded_size: Optional[int] = None,
                                            **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Fully vectorized version that processes all batches at once but uses scatter operations
    instead of explicit loops for maximum efficiency.
    """
    assert inputs.dtype == torch.complex64, 'Input must be complex'
    
    B, C, N, _ = inputs.shape
    
    if padded_size is None:
        max_offset = torch.max(torch.abs(offsets_xy)).item()
        padded_size = N + 2 * int(math.ceil(max_offset))
    
    M = padded_size
    device = inputs.device
    
    # Pad all patches at once: (B, C, N, N) -> (B, C, M, M)
    pad_dim = (M - N) // 2
    inputs_padded = F.pad(inputs, (pad_dim, pad_dim, pad_dim, pad_dim), "constant", 0.)
    
    # Adjust offsets for padding
    offsets_adjusted = offsets_xy + pad_dim
    
    # Flatten batch and channel dimensions: (B*C, M, M) and (B*C, 1, 2)
    patches_flat = inputs_padded.reshape(B * C, M, M)
    offsets_flat = offsets_adjusted.reshape(B * C, 1, 2).squeeze(1)  # (B*C, 2)
    
    if agg:
        # Extract center regions for aggregation
        center_slice = slice(M // 4, M // 4 + M // 2)
        patches_center = patches_flat[:, center_slice, center_slice]
        center_size = patches_center.shape[1]
        
        # Adjust positions for center extraction
        center_offset = M // 4
        canvas_positions = offsets_flat + center_offset
        
        # Initialize canvas for center regions only
        canvas = torch.zeros(B, M, M, device=device, dtype=torch.complex64)
        counts = torch.zeros(B, M, M, device=device, dtype=torch.float32)
        
        # Vectorized accumulation using your existing VectorizedBarycentricAccumulator approach
        accumulator = VectorizedBarycentricAccumulator((M, M), device)
        
        # Process each batch's center regions
        for batch_idx in range(B):
            start_idx = batch_idx * C
            end_idx = (batch_idx + 1) * C
            
            batch_patches = patches_center[start_idx:end_idx]  # (C, center_size, center_size)
            batch_positions = canvas_positions[start_idx:end_idx]  # (C, 2)
            
            accumulator.accumulate_batch(
                canvas[batch_idx], counts[batch_idx], 
                batch_patches, batch_positions, center_size
            )
        
        # Normalize
        boolean_mask = counts > 1e-6
        norm_factor = torch.clamp(counts, min=1.0)
        imgs_merged = (canvas * boolean_mask) / norm_factor
        
        return imgs_merged, boolean_mask, M
    
    else:
        # No aggregation - simpler case
        canvas = torch.zeros(B, M, M, device=device, dtype=torch.complex64)
        
        # Direct accumulation without overlap handling
        for batch_idx in range(B):
            start_idx = batch_idx * C
            end_idx = (batch_idx + 1) * C
            
            batch_patches = patches_flat[start_idx:end_idx]
            batch_positions = offsets_flat[start_idx:end_idx]
            
            # Simple placement (assuming no overlaps in no-agg case)
            for c in range(C):
                y_pos = int(batch_positions[c, 1].item())
                x_pos = int(batch_positions[c, 0].item())
                canvas[batch_idx, y_pos:y_pos+M, x_pos:x_pos+M] = batch_patches[c]
        
        return canvas, torch.ones(B, M, M, device=device, dtype=torch.bool), M
    
def use_barycentric_translation_in_model():
    """
    Example of how to integrate this into your existing pipeline.
    
    Replace calls to reassemble_patches_position_real with:
    """
    # Old way (memory intensive):
    # soln_patches, ones_mask, padded_size = reassemble_patches_position_real(
    #     im_tensor, relative_coords, data_config, model_config, agg=True
    # )
    
    # New way (memory efficient):
    if VectorizedBarycentricAccumulator is not None:
        soln_patches, ones_mask, padded_size = vectorized_barycentric_reassemble_patches(
            im_tensor, relative_coords, data_config, model_config, agg=True
        )
    else:
        soln_patches, ones_mask, padded_size = barycentric_reassemble_patches_position_real(
            im_tensor, relative_coords, data_config, model_config, agg=True
        )
    
    return soln_patches, ones_mask, padded_size