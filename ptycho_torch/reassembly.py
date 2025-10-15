#Type helpers
from typing import Tuple, Optional, Union, Callable, Any, List
from ptycho_torch.dataloader import PtychoDataset, TensorDictDataLoader
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

#Default casting
torch.set_default_dtype(torch.float32)
 
#Currently adapted for multi-channel, but finding some problems in terms of reassembly
#Adapted from Oliver's tf_helper/shift_and_Sum
def reassemble_single_channel(im_tensor: torch.Tensor,
                              com: float,
                              max_offset: float,
                              global_coords: torch.Tensor,
                              data_config: DataConfig,
                              middle: int = 10) -> torch.Tensor:
    '''
    Takes a batch stack of object tensors, and then lays them out on a "canvas" and averages
    them based on their true positions.

    This is essentially the core behind reconstructing a full object from the independent "smaller" object
    patch predictions (through the neural net)

    Inputs
    -------
    im_tensor: torch.Tensor (B,C,H,W), output from neural net
    global_coords: torch.Tensor(B,C,1,2), global coordinates from ptycho scan
    com: torch.Tensor(2), center of mass coordinates
    data_config: DataConfig, contains all relevant data parameters used for NN
    middle: int, We want to exclude outer sections due to Nyquist sampling (outer border has bad signal)
            This parameter controls how much of the center section we grab

    Returns
    -------
    reassembled_image: sum of reassembled images on canvas
    reassembled_ones: sum of ones with same shape as reassmbled_images for norm
    '''
    
    N = data_config.N
    M = middle #Easier to read code
    

    #Squeezing global_coords
    #Need to modify shape to work with Translation properly, need singleton second dim
    global_coords = global_coords.float().squeeze()

    #Select the central part of the object tensor
    im_tensor = im_tensor[:,:,
                        N // 2 - M // 2:N // 2 + M // 2,
                        N // 2 - M // 2:N // 2 + M // 2].squeeze()
    #Get dynamic padding value
    dynamic_pad = torch.ceil(max_offset).int()

    #Subtracting COM to get relative coordinates (to center)
    if data_config.C == 1:
        #Recenter
        adjusted_offsets = global_coords - com[None,:]
        #Reshape offset
        adjusted_offsets = adjusted_offsets[:,None,:]
    elif data_config.C > 1: #Unused at the moment
        B, C, H, W = im_tensor.shape
        #Recenter
        adjusted_offsets = global_coords - com[None, None,:]
        #Reshape offset
        adjusted_offsets = adjusted_offsets.reshape(B*C,2).unsqueeze(1)
        #Reshape image tensor itself
        im_tensor = im_tensor.reshape(B*C, H, W)

    print(f'Padding Size: {dynamic_pad}')

    #Pad all tensors to the same size. Introduce a ones tensor for normalization
    # padded_images = hh.pad_patches(im_tensor, dynamic_pad)
        # padded_ones = hh.pad_patches(torch.ones_like(im_tensor),dynamic_pad)
    padded_images = F.pad(im_tensor, (dynamic_pad, dynamic_pad, dynamic_pad, dynamic_pad), "constant", 0)
    padded_ones = F.pad(torch.ones_like(im_tensor),  (dynamic_pad, dynamic_pad, dynamic_pad, dynamic_pad), "constant", 0)

    #Translate
    translated_images_4d = hh.Translation(padded_images, adjusted_offsets, jitter_amt=0.0)
    ones_4d = hh.Translation(padded_ones, adjusted_offsets, jitter_amt=0.0)

    if data_config.C == 1:
        #Squeeze channel dimension
        translated_images = translated_images_4d.squeeze(1)
        translated_ones = ones_4d.squeeze(1)
    else:
        _, _, padded_H, padded_W = translated_images_4d.shape
        translated_images = translated_images_4d.reshape(B,C,padded_H,padded_W)
        translated_ones = ones_4d.reshape(B,C,padded_H,padded_W)

    #Get output
    reassembled_image = torch.sum(translated_images, dim=0)
    reassembled_ones = torch.sum(translated_ones, dim = 0)

    return reassembled_image, reassembled_ones

def reassemble_multi_channel(im_tensor: torch.Tensor,
                              com: float,
                              max_offset: float,
                              relative_coords: torch.Tensor,
                              coord_centers: torch.Tensor,
                              data_config: DataConfig,
                              model_config: ModelConfig,
                              middle: int = 10) -> torch.Tensor:
    '''
    Takes a batch stack of object tensors, and then lays them out on a "canvas" and averages
    them based on their true positions.

    Specialized for mult-channel situation. We will assemble the solution region first, and then from there reassemble the canvas

    Inputs
    -------
    im_tensor: torch.Tensor (B,C,H,W), output from neural net
    relative_coords: torch.Tensor(B,C,1,2), relative coordinates for each sol'n region
    coord_centers: torch.Tensor(B,1,1,2), com coordinates for each sol'n region
    com: torch.Tensor(2), center of mass coordinates for global ptycho scan
    data_config: DataConfig, contains all relevant data parameters used for NN
    middle: int, We want to exclude outer sections due to Nyquist sampling (outer border has bad signal)
            This parameter controls how much of the center section we grab

    Returns
    -------
    reassembled_image: sum of reassembled images on canvas
    reassembled_ones: sum of ones with same shape as reassmbled_images for norm
    '''
    M = middle #Easier to read code

    #Perform the initial solution patch assembly and get the ones vector as well
    soln_patches, ones_mask, padded_size = hh.reassemble_patches_position_real(im_tensor,
                                                                  relative_coords,
                                                                  data_config, model_config,
                                                                  agg=True)

    #Select the central part of the object tensor
    soln_patches = soln_patches[:,
                        padded_size // 2 - M // 2:padded_size // 2 + M // 2,
                        padded_size // 2 - M // 2:padded_size // 2 + M // 2]
    
    ones_mask = ones_mask.float() #Convert from bool to float
    ones_mask = ones_mask[:,
                          padded_size // 2 - M // 2:padded_size // 2 + M // 2,
                          padded_size // 2 - M // 2:padded_size // 2 + M // 2]
    
    #Get dynamic padding value
    dynamic_pad = torch.ceil(max_offset).int()

    #Subtracting COM to get relative coordinates (to center)
    B, _, _ = soln_patches.shape
    #Recenter
    adjusted_offsets = coord_centers - com[None, None,:]
    #Reshape offset
    adjusted_offsets = adjusted_offsets.reshape(B,2).unsqueeze(1)

    #Pad all tensors to the same size. Introduce a ones tensor for normalization
    # padded_images = hh.pad_patches(im_tensor, dynamic_pad)
        # padded_ones = hh.pad_patches(torch.ones_like(im_tensor),dynamic_pad)
    padded_images = F.pad(soln_patches, (dynamic_pad, dynamic_pad, dynamic_pad, dynamic_pad), "constant", 0)
    padded_ones = F.pad(ones_mask,  (dynamic_pad, dynamic_pad, dynamic_pad, dynamic_pad), "constant", 0)

    #Translate (this was originally positive, setting to negative for now)
    translated_images = hh.Translation(padded_images, adjusted_offsets, jitter_amt=0.0)
    translated_ones= hh.Translation(padded_ones, adjusted_offsets, jitter_amt=0.0)

    #Get output
    reassembled_image = torch.sum(translated_images, dim=0)
    reassembled_ones = torch.sum(translated_ones, dim = 0)

    return reassembled_image, reassembled_ones


def reconstruct_image(model: nn.Module,
                      ptycho_dset: PtychoDataset,
                      training_config = TrainingConfig,
                      data_config = DataConfig,
                      model_config = ModelConfig,
                      inference_config = InferenceConfig):
    '''
    Reconstructs an image given a model as well as ptycho dataset. Assumes one image per dataset at the moment.

    Inputs
    ---------
    model: This must be a PtychoPINN Lightning module from train.py, assumed to be on GPU
    ptycho_dset: This must be the custom PtychoDataset 
    '''
    n_files = ptycho_dset.n_files
    experiment_number = inference_config.experiment_number

    #Get dataset subset
    if n_files > 1:
        ptycho_subset = ptycho_dset.get_experiment_dataset(experiment_number)
    else:
        ptycho_subset = ptycho_dset
    device = training_config.device

    infer_loader = TensorDictDataLoader(ptycho_subset, batch_size = inference_config.batch_size,
                                    collate_fn = Collate(device = training_config.device))

    

    #Get center of mass and max difference
    global_coords = ptycho_subset.mmap_ptycho['coords_global'].squeeze()

    #Dynamic center of mass that's channel agnostic
    if 'com' in ptycho_subset.data_dict:
        center_of_mass = torch.mean(global_coords,
                                    dim = tuple(range(global_coords.dim()-1)))
    else:
        center_of_mass = ptycho_subset.data_dict['com']
  
    adjusted_offsets_float = global_coords - center_of_mass
    max_abs_offset = torch.ceil(torch.max(torch.abs(adjusted_offsets_float))).int()

    #Moving com and offset to device
    center_of_mass = center_of_mass.to(device)
    max_abs_offset = max_abs_offset.to(device)

    #Reassembly function
    if data_config.C == 1:
        reassemble_fn = reassemble_single_channel  # Your original function
    else:
        reassemble_fn = reassemble_multi_channel   # Your original function
    
    #Initialize accumulation tensors
    reassembled_image = None

    with torch.no_grad():
        for i, batch in enumerate(infer_loader):
            start = time.time()

            # Unpack and transfer to device efficiently
            batch_data = batch[0]
            x = batch_data['images'].to(device, non_blocking=True)
            positions = batch_data['coords_relative'].to(device, non_blocking=True)
            probe = batch[1].to(device, non_blocking=True)
            in_scale = batch_data['rms_scaling_constant'].to(device, non_blocking=True)
            batch_coords_global = batch_data['coords_global'].to(device, non_blocking=True)
            
            # Model inference
            batch_output = model.forward_predict(x, positions, probe, in_scale)
            
            # Reassembly
            if data_config.C == 1:
                reassembled_batch_image, reassembled_batch_ones = reassemble_fn(
                    batch_output,           # im_tensor
                    center_of_mass,         # com
                    max_abs_offset,         # max_offset
                    batch_coords_global,    # global_coords
                    data_config,            # data_config
                    inference_config.middle_trim  # middle
                )
            else:
                batch_relative_center = batch_data['coords_center'].to(device, non_blocking=True)
                reassembled_batch_image, reassembled_batch_ones = reassemble_fn(
                    batch_output,           # im_tensor
                    center_of_mass,         # com
                    max_abs_offset,         # max_offset
                    positions,              # relative_coords
                    batch_relative_center,  # coord_centers
                    data_config,            # data_config
                    model_config,           # model_config
                    inference_config.middle_trim  # middle
                )
                # Clean up multi-channel specific tensor
                del batch_relative_center

            if reassembled_image is not None:
                reassembled_image += reassembled_batch_image
                reassembled_ones += reassembled_batch_ones
            else:
                reassembled_image = reassembled_batch_image
                reassembled_ones = reassembled_batch_ones

            print(f'Batch {i} completed in {time.time()-start} seconds')

    # Free all batch tensors
    del x, positions, probe, in_scale, batch_coords_global
    del batch_output, reassembled_batch_image, reassembled_batch_ones, batch
    torch.cuda.empty_cache()
    gc.collect()


    return reassembled_image/(reassembled_ones), ptycho_subset

def profile_memory():
    """Print current GPU memory usage."""
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")

# --- Barycentric Interpolation Bellow ---


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


def reconstruct_image_barycentric(model: nn.Module,
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

             

