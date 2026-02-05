#Type helpers
from typing import Tuple, Optional, Union, Callable, Any

#Additional helper functions ported from original tensorflow lib
import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import gc

#Configurations
from ptycho_torch.config_params import ModelConfig, DataConfig # Removed TrainingConfig as it's not used here

#Complex functions
#---------------------
def is_complex(x: torch.Tensor) -> bool:
    '''
    Check if tensor is complex dtype.

    Inputs
    ------
    x: torch.Tensor
        Tensor to check

    Outputs
    -------
    out: bool
    '''

    return x.is_complex()

#Patch reassembly, for overlapping patch physics
#---------------------
def reassemble_patches_position_real(inputs: torch.Tensor, offsets_xy: torch.Tensor,
                                      data_config: DataConfig, model_config: ModelConfig, # Added configs
                                      agg: bool = True,
                                      padded_size: Optional[int] = None,
                                      **kwargs: Any) -> torch.Tensor:
    '''
    Given image patches (shaped such that the channel dimension indexes
    patches within a single solution region), reassemble into an image
    for the entire solution region. Overlaps between patches are
    averaged.

    This function is passed as an argument (it is wrapped). This is because it is applied as a Lambda function
    to every single image patch in the dataloader stack.

    Inputs
    ------
    inputs: diffraction images from model -> (n_images, n_patches, N, N)
    offsets_xy: offset patches in x, y -> (n_images, n_patches, 1, 2)
    agg: aggregation boolean
    padded_size: Amount of padding

    Output
    ------
    out: torch.Tensor
        Single tensor of shape (n_images, padded_size, padded_size) representing the assembled region.
    '''

    assert inputs.dtype == torch.complex64, 'Input must be complex'

    B, _, N, _ = inputs.shape

    #Setting the channels for forward model to a specific C_forward. Model may use this differently

    C = model_config.C_forward

    if padded_size is None:
        padded_size = get_padded_size(data_config, model_config)
    M = padded_size # Use M for clarity

    # --- 1. Prepare Flat Inputs and Offsets ---
    # Flatten batch and channel dimensions for efficient translation
    # Input images: (B, C, N, N) -> (B*C, N, N)
    imgs_flat = inputs.flatten(start_dim=0, end_dim=1)
    # Offsets: (B, C, 1, 2) -> (B*C, 1, 2)
    offsets_flat = offsets_xy.flatten(start_dim=0, end_dim=1)

    # --- 2. Pad Images ---
    # Pad flat images to the target size M
    # (B*C, N, N) -> (B*C, M, M)
    # Support odd total padding by splitting remainder across sides
    total_pad = M - N
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad
    top_pad = total_pad // 2
    bottom_pad = total_pad - top_pad
    imgs_flat_bigN = F.pad(imgs_flat, (left_pad, right_pad, top_pad, bottom_pad), "constant", 0.)


    # --- 3. Translate Images ---
    # (B*C, M, M) -> (B*C, 1, M, M) -> (B*C, M, M) complex
    imgs_flat_bigN_translated = Translation(imgs_flat_bigN, offsets_flat, 0.).squeeze(1)

    # --- 4. Handle Aggregation vs. No Aggregation ---
    if agg:
        # --- 4a. Prepare and Translate *Prototype* Normalization Mask ---
        # Create ONE small (N, N) mask for the central region
        with torch.no_grad(): # Mask creation doesn't need gradients
            prototype_mask_N = torch.zeros(N, N, device=inputs.device, dtype=torch.float32)
            center_slice = slice(N // 4, N // 4 + N // 2)
            prototype_mask_N[center_slice, center_slice] = 1.0

            # Pad the SINGLE prototype mask to (M, M)
            prototype_mask_M = F.pad(prototype_mask_N, (left_pad, right_pad, top_pad, bottom_pad), "constant", 0.)

            # Expand the single (M, M) mask to (B*C, M, M) without copying memory (views)
            # This is the input to the Translation function for the counts
            prototype_mask_M_expanded = prototype_mask_M.expand(B * C, M, M)

        # Translate the expanded *prototype* mask using the offsets.
        # Input: (B*C, M, M) float -> Output: (B*C, 1, M, M) -> (B*C, M, M) float
        norm_flat_bigN_translated = Translation(prototype_mask_M_expanded, offsets_flat, 0.).squeeze(1)

        # --- 4b. Calculate Summed Images and Normalization Factor ---

        # (B*C, M, M) -> (B, C, M, M) -> sum(dim=1) -> (B, M, M) complex
        imgs_summed = imgs_flat_bigN_translated.reshape(B, C, M, M).sum(dim=1)
        #gc.collect()

        # Reshape translated masks and sum along the channel dimension to get counts
        # (B*C, M, M) -> (B, C, M, M) -> sum(dim=1) -> (B, M, M) float
        non_zeros_float = norm_flat_bigN_translated.reshape(B, C, M, M).sum(dim=1)
        #gc.collect()
        # --- 4c. Normalize ---
        # Create a boolean mask where at least one patch contributed centrally
        boolean_mask = non_zeros_float.real > 1e-6 # Use tolerance for float comparison

        # Prepare normalization factor, avoiding division by zero
        # Clamp ensures values are at least 1.0 where division occurs
        norm_factor = torch.clamp(non_zeros_float.real, min=1.0)

        # Apply normalization: zero out regions outside mask, then divide by count
        imgs_merged = (imgs_summed * boolean_mask) / norm_factor

        return imgs_merged, boolean_mask, M # Shape: (B, M, M)

    else:
        # No aggregation: simply reshape the translated patches back
        # (B*C, M, M) -> (B, C, M, M)
        print('no aggregation in patch reassembly')
        # Ensure the helper function receives the correct number of channels
        return _flat_to_channel(imgs_flat_bigN_translated, channels=C) 


def norm_mask(inputs):
    '''
    Creates normalization mask for patch reassembly. Simply returns a mask of ones which has reduced dimensionality
    '''
    B, C, N, _ = inputs.shape

    #Mask with half-size
    ones = torch.ones(B, C, N//2, N//2).to(inputs.device)
    #Pad rest
    ones = nn.ZeroPad2d(N//4)(ones)

    return ones


def extract_channels_from_region(inputs: torch.Tensor,
                                 offsets_xy: torch.Tensor,
                                 data_config: DataConfig, model_config: ModelConfig, # Added configs
                                 jitter_amt: float = 0.0) -> torch.Tensor:

    '''
    Extracts averaged objects from the summed M x M solution region.

    Essentially we're translating the solution region in reverse; we reverse the offsets we used to translate
    the original image, so that the image patch of interest (within the solution region) is
    centered at the origin.

    We can then use this modified translated solution region and do a simple crop extraction that's N x N about the origin.

    Inputs
    ------
    inputs: torch.Tensor (batch_size, 1, M, M), M = N + some padding size
    offsets: torch.Tensor (batch_size, C, 1, 2)
    jitter_amt: float
        Assuming zero jitter

    Output
    ------
    output: torch.Tensor (batch_size, C, N, N)
        - Shifted images, cropped symmetrically
    
    '''
    offsets_flat = torch.flatten(offsets_xy, start_dim = 0, end_dim = 1)
    #Check offset and input dimensions
    #List of assertions
    if inputs.shape[0] is not None:
        assert int(inputs.shape[0]) == int(offsets_xy.shape[0])
    assert int(inputs.shape[1]) == 1
    assert int(offsets_xy.shape[3]) == 2

    
    #We need to repeat the solution patch C # of times, so we can perform unique translations
    #for all C image patches.
    #Steps: Repeat -> Flatten
    n_channels = offsets_xy.shape[1]
    stacked_inputs = inputs.expand(-1, n_channels, -1, -1)
    flat_stacked_inputs = torch.flatten(stacked_inputs, start_dim = 0, end_dim = 1)

    #Obtain translation of patches
    #Note, offsets must be inverted in sign compared to Translation in: reassemble_patches_position_real
    translated_patches = Translation(flat_stacked_inputs, -offsets_flat, 0.0) # Assuming Translation doesn't need config
    cropped_patches = trim_reconstruction(translated_patches, data_config, model_config) # Pass configs
    cropped_patches = torch.reshape(cropped_patches,
                                    (-1, n_channels, data_config.N, data_config.N)) # Use config N

    return cropped_patches

    
def trim_reconstruction(inputs: torch.Tensor, data_config: DataConfig, model_config: ModelConfig, # Added configs
                        N: Optional[int] = None) -> torch.Tensor:
    '''
    Trim from shape (-1, 1, M, M) to (-1, 1, N, N) where M >= N
    Second dimension is 1 because of Translation function always returning 4D tensor
    M is the expanded solution region size
    N is the exact size of the measured diffraction input image (e.g. 64 pixels)

    Assume M = get_padded_size(data_config, model_config)

    Inputs
    ------
    inputs: torch.Tensor (batch_size, 1, M, M)
    data_config: DataConfig
    model_config: ModelConfig
    N: int, size of last two dimensions (optional override)
    '''

    if N is None:
        N = data_config.N # Use config N

    shape = inputs.shape

    # Ensure dimension matching
    if shape[2] is not None:
        assert int(shape[2]) == int(shape[-1])
    # Calculate M (padded size) using configs
    M = get_padded_size(data_config, model_config)
    diff = M - N
    # Calculating start and end ensures odd/even cases work
    start_clip = diff // 2
    end_clip = diff - start_clip

    #return clipped input
    return inputs[:,:,
                  start_clip:M-end_clip,
                  start_clip:M-end_clip]



#Masking/norm functions
#---------------------
#UNUSED
def mk_centermask(inputs: torch.Tensor, N: int, c: int, kind: str = 'center') -> torch.Tensor:
    '''
    Creates padded object mask to fulfill Nyquist criterion. This way, when we do FFT for
    diffraction image, we don't oversample frequency.
    (I.e. pad image on both sides with N/4 pixels, so total # of
    padded pixels is N/2)).
    The purpose of this mask is to find out how many images are overlapped
    at a specific pixel, so we can normalize the sum of pixel values
    by the number of images on that pixel (e.g. Pixel = 30, images = 3 -> normalized = 10)

    Inputs
    ------
    inputs: torch.Tensor
    N: Image dimension (x and y) in pixels
    c: Number of images per patch (usually 4)
    kind: 'center' or 'border'. Affects how mask is made

    Outputs
    -------
    output: Padded mask
    '''
    n_images = inputs.shape[0]
    count_mask = torch.ones((n_images, c, N, N),
                              dtype = inputs.dtype)
    #count_mask = F.pad(count_mask, (N//4, N//4, N//4, N//4), "constant", 0)

    if kind == 'center':
        return count_mask
    elif kind == 'border':
        return 1 - count_mask
    else:
        raise ValueError

#Pad functions
#---------------------

def pad_patches(input: torch.Tensor, padded_size: Optional[int] = None) -> torch.Tensor:
    '''
    Pad patches to be the same size as the original image
    '''
    if padded_size is None:
        padded_size = get_padded_size()
    pad_dim = (padded_size - input.shape[-1]) // 2

    return F.pad(input, (pad_dim, pad_dim, pad_dim, pad_dim), "constant", 0)

def pad_obj(input: torch.Tensor, h: int, w: int) -> torch.Tensor:
    '''
    Pad object function for Nyquist criterion. An N x M image needs to be padded by N // 4
    pixels on the top and bottom sides, and the M // 4 on the left and right sides
    '''
    # Using ZeroPad2d which pads (left, right, top, bottom)
    return nn.ZeroPad2d((w // 4, w // 4, h // 4, h // 4))(input)

def get_padded_size(data_config: DataConfig, model_config: ModelConfig) -> int: # Added configs
    bigN = get_bigN(data_config, model_config) # Pass configs
    buffer = model_config.max_position_jitter # Use config

    return bigN + buffer

def get_bigN(data_config: DataConfig, model_config: ModelConfig) -> int: # Added configs
    N = data_config.N # Use config
    gridsize = data_config.grid_size # Use config
    offset = math.ceil(data_config.max_neighbor_distance) # Use config
    #Add extra offset if odd, otherwise get weird padding mismatch error
    if offset % 2 == 1:
        offset += 1

    return N + (gridsize[0] - 1) * offset

def trim_and_pad_output(tensor_nxn, data_config, model_config):
    """
    Takes an (N, N) tensor, extracts the central (desired_inner_size, desired_inner_size)
    region, and pads it back to (N, N) with zeros.

    Args:
        tensor_nxn (torch.Tensor): The input tensor with shape (..., N, N).
        desired_inner_size (int): The side length of the central region to keep.
        model_config (ModelConfig): ModelConfig class, used for boundary trim parameter

    Returns:
        torch.Tensor: The (N, N) tensor with only the central region preserved.
    """

    N = data_config.N
    edge_pad = model_config.edge_pad

    #Desired dimension of non-zero square from decoder output
    desired_inner_size = (N - 2 * edge_pad) // 2

    #Sanity check
    if desired_inner_size == N:
        return tensor_nxn # Nothing to do
    if desired_inner_size > N:
        raise ValueError("desired_inner_size cannot be larger than the tensor size N")
    if desired_inner_size <= 0:
         raise ValueError("desired_inner_size must be positive")


    # Calculate padding needed *around* the desired inner size to reach N
    pad_total = N - desired_inner_size
    #Handles odd padding amount
    if pad_total % 2 != 0:
        # Handle odd padding - add extra pixel to right/bottom
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
    else:
        pad_left = pad_right = pad_top = pad_bottom = pad_total // 2

    # Calculate slice indices to extract the inner part
    start = pad_top # or pad_left, assuming square
    end = N - pad_bottom # or n - pad_right

    # Extract the central part
    inner_part = tensor_nxn[..., start:end, start:end]

    # Pad the extracted inner part back to N x N
    # F.pad takes (pad_left, pad_right, pad_top, pad_bottom)
    padded_tensor = F.pad(inner_part, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)

    return padded_tensor

#Translation functions
#---------------------
def Translation(img, offset, jitter_amt=0):
    '''
    Translation function with custom complex number support.
    Uses torch.nn.functional.grid_sample to perform subpixel translation.
    Meant to be performed on a flattened set of inputs.

    Grid_sample takes an input of (N, C, H_in, W_in) and grid of (N, H_out, W_out, 2) to
    output (N, C, H_out, W_out). In our case, C is 1 because we already flattened the channels from (N, C, H_in, W_in)
    to (N * C, H_in, W_in). Offset is (N*C, 1, 2)

    We transform the input solution region to (N, 1, H, W), and the grid to (N, H_out, W_out, 2).
    The grid essentially contains all c possible translations of the input region.

    The output in our case will be (N, 1, H_out, W_out)

    Inputs
    ------
    img: torch.Tensor (N, H, W). Stack of images in a solution region. dtype complex, cfloat 
    offset: torch.Tensor (N, 1, 2). Offset of each image in the solution region

    Outputs
    ------
    out: torch.Tensor (N, 1, H_out, W_out)
    '''
    n, h, w = img.shape

    # Compute normalized offset (much more efficient than creating full identity grid first)
    norm_factor = torch.tensor([2.0/(w-1), 2.0/(h-1)], device=img.device)

    # Optional jitter - only compute if needed
    if jitter_amt > 0: 
        jitter = torch.normal(
            mean=torch.zeros_like(offset),
            std=jitter_amt
        )
        offset = offset + jitter

    # Convert to normalized coordinates (negative because grid coords work in reverse)
    norm_offset = -offset * norm_factor.reshape(1, 1, 2)

    # Create the sampling grid directly with affine_grid (much faster than manual construction)
    # This is the creation of an affine matrix (2x3) for translation only

    theta = torch.zeros(n, 2, 3, device=img.device, dtype=norm_offset.dtype)
    theta[:, 0, 0] = 1.0  # Identity for x-scaling
    theta[:, 1, 1] = 1.0  # Identity for y-scaling
    theta[:, :, 2:] = norm_offset.view(n, 2, 1)  # Translation matrix

    # Generate sampling grid from affine matrix
    grid = F.affine_grid(theta, [n, 1, h, w], align_corners=True)

    # return translated
    if torch.is_complex(img):
        img_expanded = img.unsqueeze(1)
        translated_real = F.grid_sample(img_expanded.real, grid, mode='bilinear', align_corners=True)
        translated_imag = F.grid_sample(img_expanded.imag, grid, mode='bilinear', align_corners=True)
        return torch.complex(translated_real, translated_imag)
    else:
        return F.grid_sample(img.unsqueeze(1), grid, mode='bilinear', align_corners=True)

#Flattening functions
#---------------------
def _flat_to_channel(img: torch.Tensor, channels: int = 4) -> torch.Tensor:
    '''
    Reshapes tensor from flat format to channel format. Useful to batch apply operations such as
    translation on the flat tensor, then reshape back to channel format.

    Inputs
    ------
    img: torch.Tensor (N, H, W)
    channels: int - Number of channels to reshape into

    Outputs
    -------
    out: torch.Tensor (M, C, H, W)
    '''

    _, H, W = img.shape

    img = torch.reshape(img, (-1, channels, H, W))

    return img
    

#Fourier functions for forward pass

def pad_and_diffract(input: torch.Tensor, pad: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Pads channel images and performs Fourier transform, going from real space (object) to
    reciprocal space (diffraction pattern)

    Input
    --------
    input: torch.Tensor (N, C, P, H, W). Does not need to be a flattened tensor.
    pad: Boolean. Whether to pad the input before performing the Fourier transform.
    
    '''

    h, w = input.shape[-2], input.shape[-1]

    if pad:
        input = pad_obj(input, h, w)
    padded = input

    input = torch.fft.fft2(input.to(torch.complex64))
    input = torch.sum(input, dim = 2) # (N, C, P, H, W) -> (N, C, H, W)
    input = torch.real(torch.conj(input) * input) / (h * w)
    input = torch.sqrt(torch.fft.fftshift(input, dim=(-2, -1)))

    return input, padded

import torch

def illuminate_and_diffract(
    input_obj: torch.Tensor,
    probe: torch.Tensor, # Shape (P, H, W) or (H, W) representing scaled modes {α_k P_k}
    nphotons: float = 1e5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Performs multi-modal coherent illumination (assuming probe modes are pre-scaled),
    diffraction, and intensity scaling. Used for simulating synthetic data

    Inputs
    --------
    input_obj: torch.Tensor (N, H, W) - Complex object batch.
    probe: torch.Tensor (P, H, W) or (H, W) - Complex probe mode(s).
                                          Assumed to be the set {α_k P_k},
                                          where P_k are orthogonal modes and α_k
                                          are complex coefficients. Scaling is
                                          baked into this tensor.
                                          If (H, W), treated as a single mode (P=1).
    nphotons: float - Target average photon count (total intensity) per pattern.

    Returns
    --------
    output_intensity: torch.Tensor (N, H, W) - Scaled, fftshifted intensity.
    original_input: torch.Tensor (N, H, W) - The original input object tensor.
    intensity_scale_factor: torch.Tensor (scalar) - Applied scaling factor.
    '''
    # --- Input Validation ---
    if not (input_obj.ndim == 3 and input_obj.is_complex()):
        raise ValueError("input_obj must be complex 3D tensor (N, H, W)")
    if not (probe.ndim in [2, 3] and probe.is_complex()):
        raise ValueError("probe must be complex 2D (H, W) or 3D (P, H, W) tensor")
    if input_obj.shape[-2:] != probe.shape[-2:]:
        raise ValueError("Spatial dimensions (H, W) of input_obj and probe must match.")
    if not isinstance(nphotons, (int, float)) or nphotons <= 0:
         raise ValueError("nphotons must be a positive number")

    N, H, W = input_obj.shape
    original_input_obj = input_obj # Keep original for return

    # --- Handle Probe Shape ---
    if probe.ndim == 2:
        probe = probe.unsqueeze(0) # (H, W) -> (1, H, W)

    # --- Calculation ---
    # Reshape for broadcasting
    obj_reshaped = input_obj.unsqueeze(1)       # (N, 1, H, W)
    probe_b = probe.unsqueeze(0)                # (1, P, H, W)

    # Form the exit wave for each pre-scaled mode (object * [α_k P_k])
    exit_waves_per_mode = obj_reshaped * probe_b  # (N, P, H, W)

    # Perform FFT using 'ortho' norm for Parseval convenience
    fft_waves_per_mode = torch.fft.fft2(exit_waves_per_mode.to(torch.complex64), norm='ortho') # (N, P, H, W)

    # Perform coherent sum across modes (Sum_k FFT{O * [α_k P_k]})
    coherent_fft_sum = torch.sum(fft_waves_per_mode, dim=1) # (N,P,H,W) -> (N, H, W)

    # Calculate the final coherent intensity |Sum_k(FFT{O * [α_k P_k]})|^2
    diffraction_intensity_coherent = torch.abs(coherent_fft_sum)**2 # (N, H, W) (Real float)

    # Total intensity per image in the batch (using Parseval via 'ortho' norm)
    current_total_intensity = torch.sum(diffraction_intensity_coherent, dim=(-2, -1)) # Shape: (N,)

    # Calculate average intensity across the batch
    mean_intensity_per_image = torch.mean(current_total_intensity)

    # Handle potential zero intensity patterns
    # when target nphotons is also very small.
    tolerance = max(1e-9, nphotons * 1e-9)
    if mean_intensity_per_image.item() <= tolerance:
        print(f"Warning: Mean intensity ({mean_intensity_per_image.item()}) is near zero relative to target. Scaling might be unstable or unnecessary. Setting scale factor to 1.0.")
        intensity_scale_factor = torch.tensor(1.0, device=input_obj.device, dtype=mean_intensity_per_image.dtype)
    else:
        # Calculate scaling factor for INTENSITY to reach target average count
        intensity_scale_factor = nphotons / mean_intensity_per_image

    # Apply scaling to the coherent intensity
    # Ensure scale factor has compatible dtype and dimensions for broadcasting
    scaled_intensity = diffraction_intensity_coherent * intensity_scale_factor.to(diffraction_intensity_coherent.dtype)

    # Applying the square root of this scaling to the probe
    scaled_probe = probe_b * torch.sqrt(intensity_scale_factor)
    # Apply fftshift for centered diffraction pattern
    output_intensity = torch.fft.fftshift(scaled_intensity, dim=(-2, -1))

    # Ensure output is non-negative (intensity should be)
    output_intensity = torch.relu(output_intensity)

    # Return the scaled intensity, original input, and the intensity scale factor
    return output_intensity, original_input_obj, scaled_probe

#Photon scaling functions

def derive_intensity_scale_from_amplitudes(x_norm: torch.Tensor, nphotons: float) -> torch.Tensor:
    """
    Derive dataset-level physics scale from normalized amplitudes.

    intensity_scale = sqrt(nphotons / mean(sum(x_norm**2)))
    """
    if not isinstance(x_norm, torch.Tensor):
        x_norm = torch.as_tensor(x_norm)
    if x_norm.ndim < 2:
        raise ValueError("x_norm must have at least 2 dims")
    if not isinstance(nphotons, (int, float)) or nphotons <= 0:
        raise ValueError("nphotons must be positive")

    spatial_dims = tuple(range(x_norm.ndim - 2, x_norm.ndim))
    mean_intensity = torch.mean(torch.sum(x_norm ** 2, dim=spatial_dims))
    if mean_intensity.item() <= 0:
        raise ValueError("mean intensity must be positive")
    return torch.sqrt(torch.tensor(float(nphotons), dtype=mean_intensity.dtype) / mean_intensity)

def normalize_probe(X):
    """
    Normalizes numpy array. Currently only works on 1D
    """
    N = X.shape[0]

    total_intensity = np.sum(np.abs(X)**2)

    scaling_factor = np.sqrt(1/total_intensity)

    X /= np.sqrt(total_intensity)

    return X, scaling_factor



def get_rms_scaling_factor(X: torch.Tensor,
                       data_config: DataConfig) -> torch.Tensor: # Changed h, w to N assuming square
    """
    Scale the photon counts in the diffraction image accounting for zero padding condition
    and the fact that diffraction pixel values are the square of the amplitude, which we're
    looking to scale to

    Multiple options for which groups we normalize over.
    Batch normalizes over the entire experiment
    Group normalizes over each input group (along the channel dimension)
    Each is over each image

    Returns scaling factor

    Inputs
    --------
    X: torch.Tensor (N, H, W) or (N,C,H,W) depending on if batch or group

    Outputs
    --------
    Scaling_factor: tensor shape (B,1,1,1)

    """
    N = data_config.N

    if data_config.data_scaling == 'Parseval':
        #Parseval Scaling
        if data_config.normalize == 'Batch':
            #Sum intensities across H,W dimensions, then take the mean of all images
            scaling_factor = torch.sqrt((N*N) / (torch.mean(torch.sum(X**2, dim = (-2, -1)))))
            scaling_factor = scaling_factor.view(-1,1,1,1) #Reshape to (1,1,1,1)
        elif data_config.normalize == 'Group' and len(X.shape) == 4:
            #(B,C,H,W) -> (B.)
            scaling_factor= torch.sqrt((N*N) / torch.mean(torch.sum(X**2, dim = (-2, -1)),dim = 1))# Sum over last two dims

            scaling_factor = scaling_factor.view(-1,1,1,1) #To (B,1,1,1)
        elif data_config.normalize == 'Each' and len(X.shape) == 4:
            #(B,C,H,W) -> (B.C)
            scaling_factor= torch.sqrt((N * N) / torch.sum(X**2, dim = (-2, -1))) # Sum over last two dims
                
            scaling_factor = scaling_factor.view(-1,-1,1,1)#To (B,C,1,1)
            
    elif data_config.data_scaling == 'Max':
        #Alternative max scaling
        scaling_factor = 1 / torch.max(X.sum(dim=(1,2)))   
        scaling_factor = scaling_factor.view(-1,1,1,1)

    return scaling_factor

def get_physics_scaling_factor(X: torch.Tensor,
                       data_config: DataConfig) -> torch.Tensor: # Changed h, w to N assuming square
    """
    Same as above, but calculates a physics consistent scaling factor that scales the total intensity of the image to 1

    Inputs
    --------
    X: torch.Tensor (N, H, W) or (N,C,H,W) depending on if batch or group

    Outputs
    --------
    Scaling_factor: tensor shape (B,1,1,1)

    """
    N = data_config.N

    if data_config.data_scaling == 'Parseval':
        #Parseval Scaling
        if data_config.normalize == 'Batch':
            #Sum intensities across H,W dimensinos, then take the mean of all images
            scaling_factor = 1 / (torch.mean(torch.sum(X, dim = (-2, -1))))
                
            scaling_factor = scaling_factor.view(-1,1,1,1) #Reshape to (1,1,1,1)
        elif data_config.normalize == 'Group' and len(X.shape) == 4:
            #(B,C,H,W) -> (B.)
            scaling_factor= 1 / torch.mean(torch.sum(X, dim = (-2, -1)),dim = 1)# Sum over last two dims

            scaling_factor = scaling_factor.view(-1,1,1,1) #To (B,1,1,1)
        elif data_config.normalize == 'Each' and len(X.shape) == 4:
            #(B,C,H,W) -> (B.C)
            scaling_factor= 1 / torch.sum(X, dim = (-2, -1)) # Sum over last two dims
                
            scaling_factor = scaling_factor.view(-1,-1,1,1)#To (B,C,1,1)
            
    elif data_config.data_scaling == 'Max':
        #Alternative max scaling
        scaling_factor = 1 / torch.max(X.sum(dim=(1,2)))   
        scaling_factor = scaling_factor.view(-1,1,1,1)

    return scaling_factor

def poisson_scale(input: torch.Tensor):
    """
    Scales input and adds Poisson noise

    Args:
        input: torch.Tensor (N,H,W) - Diffraction patterns that haven't been scaled
    
    Returns:
        noisy_output: torch.Tensor (N,H,W) - Diffraction patterns with Poisson scaling 
    
    """
    #Apply Poisson sampling
    noisy_intensities = torch.poisson(input)

    return noisy_intensities

#Other operations

def center_crop(larger_img, target_size):
    h, w = larger_img.shape[:2]
    start_h = (h - target_size) // 2
    start_w = (w - target_size) // 2
    return larger_img[start_h:start_h+target_size, start_w:start_w+target_size]
