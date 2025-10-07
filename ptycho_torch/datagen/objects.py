#Synthetic object generation is handled here

#Imports
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import noise
import time

#Other functions
import ptycho_torch.helper as hh

from ptycho_torch.config_params import TrainingConfig, DataConfig, ModelConfig
from skimage.draw import line_aa, disk, rectangle, ellipse, circle_perimeter_aa
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter # For blurring the 2D noise
from perlin_noise import PerlinNoise
from scipy.spatial.transform import Rotation # For 3D rotations
import random
import cv2


# Function to handle safe normalization per image in a batch
def normalize_batch_percentile(
    batch_tensor,
    p_min=1.0, # Percentile for minimum (e.g., 1.0 for 1st percentile)
    p_max=99.0, # Percentile for maximum (e.g., 99.0 for 99th percentile)
    epsilon=1e-7
):
    """
    Normalizes each image in a batch tensor to [0, 1] based on percentiles.
    Values outside the percentile range are clipped.

    Specifically for Julia fractal image generation
    """
    B = batch_tensor.shape[0]
    batch_tensor_flat = batch_tensor.view(B, -1).float() # Flatten spatial dims

    # --- *** MODIFIED PART START *** ---
    # Calculate quantiles (0-1 range)
    q_min = p_min / 100.0
    q_max = p_max / 100.0

    # Convert quantiles to scalar tensors for separate calls
    q_min_val = torch.tensor(q_min, dtype=torch.float32, device=batch_tensor.device)
    q_max_val = torch.tensor(q_max, dtype=torch.float32, device=batch_tensor.device)

    # Calculate min and max percentiles separately using scalar q
    # keepdim=False (default) reduces the dimension, result should be shape (B,)
    v_min_flat = torch.quantile(batch_tensor_flat, q=q_min_val, dim=1, interpolation='linear', keepdim=False)
    v_max_flat = torch.quantile(batch_tensor_flat, q=q_max_val, dim=1, interpolation='linear', keepdim=False)

    # Check shapes if needed (debugging)
    # print(f"Inside normalize_batch_percentile:")
    # print(f"  batch_tensor shape: {batch_tensor.shape}")
    # print(f"  v_min_flat shape: {v_min_flat.shape}") # Should be (B,)
    # print(f"  v_max_flat shape: {v_max_flat.shape}") # Should be (B,)
    # print(f"  B: {B}")

    # Reshape the (B,) tensors to (B, 1, 1) for broadcasting
    v_min = v_min_flat.view(B, 1, 1)
    v_max = v_max_flat.view(B, 1, 1)
    # --- *** MODIFIED PART END *** ---

    range_ = v_max - v_min
    # Prevent division by zero or very small numbers
    range_[range_ < epsilon] = epsilon

    # Clip tensor to the percentile range first
    clipped_tensor = torch.clamp(batch_tensor.float(), min=v_min, max=v_max)

    # Normalize the clipped tensor
    normalized = (clipped_tensor - v_min) / range_
    return normalized



def _normalize_np(array, epsilon=1e-7):
    """Helper to normalize a numpy array to [0, 1]."""
    min_val, max_val = np.min(array), np.max(array)
    range_val = max_val - min_val
    if range_val < epsilon:
        # Handle constant field case
        return np.full_like(array, 0.5, dtype=np.float32)
    else:
        return (array - min_val) / range_val
    
## Noise functions

def create_white_noise_object(img_shape, obj_arg):
    """
    Generates a complex object from white noise with material parameters.
    
    Args:
        img_shape (tuple): Shape (H, W) for the output object
        
    Returns:
        np.ndarray: Complex-valued white noise object
    """
    blur = obj_arg['blur']
    # Generate white noise base map [0, 1]
    base_noise = np.random.uniform(0, 1, size=img_shape).astype(np.float32)
    
    # Apply small amount of blur for smoother transitions
    if blur:
        blur_sigma = np.random.uniform(1.0, 3.0)
        base_noise = gaussian_filter(base_noise, sigma=blur_sigma)
    
    # Renormalize after blur
    base_noise = _normalize_np(base_noise)
    
    # Set amplitude range similar to other functions
    amplitude_range = (0.8, 1.2)
    
    # Create complex object using existing infrastructure
    obj = create_complex_object(
        base_noise,
        amplitude_target_range=amplitude_range,
        pure_phase_prob=0,
        pure_amp_prob=0
    )
    
    return obj

def create_simplex_noise_object(img_shape):
    """
    Generates a complex object from simplex noise with material parameters.
    
    Args:
        img_shape (tuple): Shape (H, W) for the output object
        
    Returns:
        np.ndarray: Complex-valued simplex noise object
    """
    # Generate simplex noise parameters
    octaves = np.random.randint(3, 8)
    noise_gen = PerlinNoise(octaves=octaves, seed=np.random.randint(10000))
    
    # Generate base simplex noise map (vectorized)
    i_coords, j_coords = np.meshgrid(np.arange(img_shape[0]), np.arange(img_shape[1]), indexing='ij')
    coords = np.stack([i_coords / img_shape[0], j_coords / img_shape[1]], axis=-1)
    base_noise = np.array([noise_gen(coord.tolist()) for coord in coords.reshape(-1, 2)]).reshape(img_shape).astype(np.float32)
    
    # Normalize to [0, 1]
    base_noise = _normalize_np(base_noise)
    
    # Create complex object using existing infrastructure
    amplitude_range = (0.8, 1.2)
    obj = create_complex_object(
        base_noise,
        amplitude_target_range=amplitude_range,
        pure_phase_prob=0,
        pure_amp_prob=0
    )
    
    return obj

def generate_perlin_object(
    batch_size: int,
    N: int,
    M: int,
    scale_batch: torch.Tensor,       # Shape (B,) - Feature size scaling
    octaves_batch: torch.Tensor,     # Shape (B,) - Int, number of noise layers
    persistence_batch: torch.Tensor, # Shape (B,) - Amplitude falloff per octave
    lacunarity_batch: torch.Tensor,  # Shape (B,) - Frequency increase per octave
    device: torch.device = torch.device('cpu'),
    amplitude_range: tuple[float, float] = (0.8, 1.2), # Target amplitude range
    uncorrelated_amp_noise_std_fraction: float = 0.05, # Std dev of small added noise as fraction of amp range
    seed_offset: int = 0 # Offset for Perlin 'base' seed for reproducibility/variation
) -> torch.Tensor:
    """
    Generates a batch of complex objects using multi-octave Perlin/Simplex noise
    with controlled hyperparameters and physically correlated amplitude/phase.
    Runs generation on CPU then moves to device.

    Phase is mapped to [-pi, pi]. Amplitude is derived from the same Perlin field,
    mapped to a specified narrow range (default [0.8, 1.2]), with optional small
    uncorrelated noise added. Phase contrast remains dominant.

    Args:
        batch_size (int): Number of objects to generate.
        N (int): Height of the object image.
        M (int): Width of the object image.
        scale_batch, octaves_batch, persistence_batch, lacunarity_batch: Tensors (B,)
            controlling the base Perlin noise generation.
        device (torch.device): Target device for the final output tensor.
        amplitude_range (tuple[float, float]): Target min/max for the amplitude map.
        uncorrelated_amp_noise_std_fraction (float): Standard deviation of additive
            Gaussian noise applied to the amplitude, expressed as a fraction of the
            total amplitude_range width. Set to 0 for no added noise.
        seed_offset (int): Integer offset added to batch index for Perlin 'base' seed.

    Returns:
        torch.Tensor: Complex object tensor of shape (B, N, M) on the specified device.
    """
    # --- Input Validation ---
    if not (scale_batch.shape == (batch_size,) and
            octaves_batch.shape == (batch_size,) and
            persistence_batch.shape == (batch_size,) and
            lacunarity_batch.shape == (batch_size,)):
        raise ValueError("Input parameter tensors must have shape (batch_size,)")
    if not (isinstance(amplitude_range, tuple) and len(amplitude_range) == 2 and
            amplitude_range[0] >= 0 and amplitude_range[1] >= amplitude_range[0]):
        raise ValueError("amplitude_range must be a tuple of (min, max) with max >= min >= 0.")
    if not (0 <= uncorrelated_amp_noise_std_fraction <= 1.0):
         raise ValueError("uncorrelated_amp_noise_std_fraction must be between 0 and 1.")

    # --- Prepare parameter lists (CPU) ---
    scales = scale_batch.cpu().tolist()
    octaves = octaves_batch.cpu().int().tolist()
    persistences = persistence_batch.cpu().tolist()
    lacunarities = lacunarity_batch.cpu().tolist()

    # --- Storage for generated maps (NumPy arrays) ---
    all_amplitude_maps_np = []
    all_phase_maps_np = []

    # --- Calculate Amplitude Noise Standard Deviation ---
    target_min_amp, target_max_amp = amplitude_range
    amplitude_width = target_max_amp - target_min_amp
    absolute_amp_noise_std = amplitude_width * uncorrelated_amp_noise_std_fraction

    # --- Loop through batch size (CPU Execution) ---
    start_loop_time = time.time()
    print(f"Starting correlated Perlin generation loop for {batch_size} items on CPU...")
    for b in range(batch_size):
        # --- Get parameters for current item ---
        scale = scales[b]
        octave = octaves[b]
        persistence = persistences[b]
        lacunarity = lacunarities[b]
        base_seed = b + seed_offset

        # --- 1. Generate Base Real Fractal Field F ---
        fractal_field_np = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            for j in range(M):
                fractal_field_np[i, j] = noise.pnoise2(i / scale, j / scale,
                                                       octaves=octave,
                                                       persistence=persistence,
                                                       lacunarity=lacunarity,
                                                       base=base_seed)

        # --- 2. Normalize F -> f_norm_np [0, 1] ---
        # This normalized field dictates the structure for BOTH phase and amplitude
        f_norm_np = _normalize_np(fractal_field_np)

        # --- 3. Create Phase Map [-pi, pi] ---
        phase_map_np = f_norm_np * (2 * np.pi) - np.pi
        all_phase_maps_np.append(phase_map_np)

        # --- 4. Create Correlated Amplitude Map [target_min, target_max] ---
        # 4a. Scale the normalized field directly to the target amplitude range
        amplitude_base_np = target_min_amp + f_norm_np * amplitude_width

        # 4b. Generate and add small uncorrelated noise (optional)
        if absolute_amp_noise_std > 1e-6: # Only add noise if std is significant
             # Use a different seed for this noise if desired, or just np.random
             amp_uncorrelated_noise_np = np.random.normal(
                 loc=0.0,
                 scale=absolute_amp_noise_std,
                 size=(N, M)
             ).astype(np.float32)
             amplitude_final_np = amplitude_base_np + amp_uncorrelated_noise_np
             # Clamp the result to ensure it stays within the target range
             amplitude_final_np = np.clip(amplitude_final_np, target_min_amp, target_max_amp)
        else:
             amplitude_final_np = amplitude_base_np # No noise added

        all_amplitude_maps_np.append(amplitude_final_np)

        if (b + 1) % max(1, batch_size // 10) == 0:
             print(f"  Generated {b+1}/{batch_size}...")

    end_loop_time = time.time()
    print(f"Perlin generation loop finished in {end_loop_time - start_loop_time:.3f} seconds.")

    # --- Stack, Convert to Torch, Move to Device ---
    print("Stacking NumPy arrays...")
    amplitude_maps_batch_np = np.stack(all_amplitude_maps_np, axis=0)
    phase_maps_batch_np = np.stack(all_phase_maps_np, axis=0)

    print(f"Converting to Torch tensors and moving to {device}...")
    amplitude_map = torch.from_numpy(amplitude_maps_batch_np).to(device=device)
    phase_map = torch.from_numpy(phase_maps_batch_np).to(device=device)

    # --- 5. Combine Phase and Amplitude ---
    print("Combining amplitude and phase...")
    complex_object = amplitude_map * torch.exp(1j * phase_map)
    print("Done.")

    return complex_object

def create_complex_layered_procedural_object(img_shape):
    base_layered= generate_base_map_shapes_perlin_layers(
        shape=img_shape,
        shape_types=['disk', 'gauss_blob', 'ellipse'], # Include gauss_blob
        n_instances_per_layer_range=(5, 10),
        layer_intensity_range=(0.5, 0.9), # Base intensity before factor
        layer_skip_probability=0.0,
        max_layer_blur_sigma=1.0,            # Blur for non-gauss layers
        perlin_octaves_range=(4, 7),
        perlin_weight_range=(0.3, 0.4),
        blur_perlin_noise_sigma_range=(0.0, 1.5),
        max_final_global_blur_sigma=1.0,
        background_level_range=(0.0, 0.02),
        shuffle_layer_order=True,
        # --- Gauss blob specific params ---
        gauss_blob_intensity_factor=1.8, # Make them significantly "taller"
        gauss_blob_radius_range=(8, 30), # Moderate sized disks
        gauss_blob_blur_sigma_range=(6.0, 12.0) # High blur sigma
    )

    amp_range = (0.8, 1.2) # Example target range
    obj = create_complex_object(
            base_layered,
            amplitude_target_range=amp_range,
            pure_phase_prob=0, # Keep probabilities
            pure_amp_prob=0
    )

    return obj

# --- Example Usage ---
# batch_size = 8
# height = 256
# width = 256
# amp_range = (0.9, 1.1) # Example tighter amplitude range
# noise_frac = 0.03      # Example: noise std dev is 3% of the amplitude range width
#
# if torch.cuda.is_available():
#     device = torch.device('cuda:0')
# else:
#     device = torch.device('cpu')
# print(f"Target device: {device}")
#
# # Define Hyperparameters for the Batch
# batch_scales = torch.rand(batch_size) * 80 + 20
# batch_octaves = torch.randint(4, 9, (batch_size,))
# batch_persistences = torch.rand(batch_size) * 0.2 + 0.4
# batch_lacunarities = torch.rand(batch_size) * 0.4 + 1.8
#
# print(f"Generating batch of {batch_size} correlated Perlin objects ({height}x{width})...")
# start_time_total = time.time()
# object_batch = generate_perlin_object_batch_correlated(
#     batch_size,
#     height,
#     width,
#     batch_scales,
#     batch_octaves,
#     batch_persistences,
#     batch_lacunarities,
#     device=device,
#     amplitude_range=amp_range,
#     uncorrelated_amp_noise_std_fraction=noise_frac
# )
# end_time_total = time.time()
# print(f"Total generation took {end_time_total - start_time_total:.3f} seconds.")

# Synthetic object generation with proper amplitude/phase generation

def gaussian_2d(coords, center, sigma_x, sigma_y, amplitude=1.0, theta=0.0):
    """Generates a 2D Gaussian distribution."""
    x, y = coords
    cx, cy = center
    # Translate coordinates
    x = x - cx
    y = y - cy
    # Rotate coordinates
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)
    # Calculate Gaussian
    sigma_x2 = sigma_x**2
    sigma_y2 = sigma_y**2
    g = amplitude * np.exp(
        -((x_rot**2) / (2 * sigma_x2) + (y_rot**2) / (2 * sigma_y2))
    )
    return g

# --- NEW: Layered Base Map Generator ---
def generate_base_map_shapes_perlin_layers(
    shape=(256, 256),
    shape_types=['disk', 'rect', 'ellipse', 'ring', 'gauss_blob'],
    n_instances_per_layer_range=(5, 15),
    layer_intensity_range=(0.4, 1.0),
    layer_skip_probability=0.1,
    max_layer_blur_sigma=3.0,        # <<< Default max blur for non-gauss layers
    perlin_octaves_range=(2, 6),
    perlin_weight_range=(0.1, 0.6),
    blur_perlin_noise_sigma_range=(0.0, 1.5),
    max_final_global_blur_sigma=0.5,
    background_level_range=(0.0, 0.1),
    shuffle_layer_order=True,
    # --- Parameters for 'gauss_blob' layer specific behavior ---
    gauss_blob_intensity_factor=1.5, # Multiplier for layer intensity
    gauss_blob_radius_range=(5, 25), # Radius range for disks used
    gauss_blob_blur_sigma_range=(4.0, 10.0) # <<< Much higher blur sigma range
    ):
    """
    Generates a base map using layered shapes (AND logic).
    'gauss_blob' layers use disks, AND logic, but higher intensity and blur.

    Usage example:
    base_and_v2 = generate_base_map_shapes_perlin_layers(
        shape=img_shape,
        shape_types=['disk', 'rect', 'gauss_blob', 'line', 'ellipse'], # Include gauss_blob
        n_instances_per_layer_range=(8, 20),
        layer_intensity_range=(0.5, 0.9), # Base intensity before factor
        layer_skip_probability=0.0,
        max_layer_blur_sigma=2.5,            # Blur for non-gauss layers
        perlin_weight_range=(0.05, 0.25),
        max_final_global_blur_sigma=0.5,
        background_level_range=(0.0, 0.02),
        shuffle_layer_order=True,
        # --- Gauss blob specific params ---
        gauss_blob_intensity_factor=1.8, # Make them significantly "taller"
        gauss_blob_radius_range=(8, 30), # Moderate sized disks
        gauss_blob_blur_sigma_range=(6.0, 12.0) # High blur sigma
    )
    """
    base = np.random.uniform(background_level_range[0], background_level_range[1], size=shape).astype(float)
    max_dim = max(shape)
    scaling_factor = shape[0]/256 #Initial number of objects scaled to 256 pixels
    # yy, xx = np.indices(shape) # Only needed if gaussian_2d were used

    current_shape_types = list(shape_types)
    if shuffle_layer_order:
        random.shuffle(current_shape_types)

    current_shape_types.append('line')

    # --- Iterate through shape types (Layers) ---
    for shape_type in current_shape_types:

        if random.random() < layer_skip_probability:
            continue

        # --- Determine parameters for this LAYER ---
        layer_intensity = np.random.uniform(layer_intensity_range[0], layer_intensity_range[1])
        n_instances = np.random.randint(n_instances_per_layer_range[0], n_instances_per_layer_range[1] + 1)
        n_instances = int(n_instances * scaling_factor)
        if shape_type == 'line':
            n_instances = int(60*scaling_factor)
        elif shape_type == 'ellipse':
            n_instances = int(130 * scaling_factor)
        elif shape_type == 'disk':
            n_instances = int(40 * scaling_factor)
        if n_instances == 0:
            continue

        # --- Apply intensity factor for 'gauss_blob' layers ---
        if shape_type == 'gauss_blob':
            layer_intensity *= gauss_blob_intensity_factor

        # --- AND Logic setup ---
        layer_and_mask = np.zeros(shape, dtype=bool)
        instances_generated_for_layer = 0

        # Generate all instance masks for this layer first
        for inst_idx in range(n_instances):
            instance_mask = np.zeros(shape, dtype=bool)
            instance_generated = False
            try:
                # --- Draw the shape instance onto the boolean instance_mask ---
                if shape_type == 'line':
                    # (line drawing code - unchanged from previous AND version)
                    x1, y1 = np.random.randint(0, shape[1]), np.random.randint(0, shape[0])
                    x2, y2 = np.random.randint(0, shape[1]), np.random.randint(0, shape[0])
                    width = np.random.randint(1, max(2, max_dim // 50))
                    for i in range(width):
                       offset_x, offset_y = np.random.uniform(-width/1.5, width/1.5), np.random.uniform(-width/1.5, width/1.5)
                       rr, cc, val = line_aa(int(y1+offset_y), int(x1+offset_x), int(y2+offset_y), int(x2+offset_x))
                       valid = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
                       rr, cc = rr[valid], cc[valid]
                       if rr.size > 0: instance_mask[rr, cc] = True; instance_generated = True
                elif shape_type == 'disk':
                    # (disk drawing code - unchanged from previous AND version)
                    center_r, center_c = np.random.randint(0, shape[0]), np.random.randint(0, shape[1])
                    min_radius = max(3, max_dim // 40); max_radius = max_dim // 30
                    min_radius *= np.sqrt(scaling_factor); max_radius *= np.sqrt(scaling_factor)
                    if max_radius <= min_radius: max_radius = min_radius + 10
                    radius = np.random.randint(min_radius, max_radius)
                    rr, cc = disk((center_r, center_c), radius, shape=shape)
                    if rr.size > 0: instance_mask[rr, cc] = True; instance_generated = True
                elif shape_type == 'rect':
                    # (rect drawing code - unchanged from previous AND version)
                    r, c = np.random.randint(0, shape[0] - 10), np.random.randint(0, shape[1] - 10)
                    min_dim_s = max(10, max_dim // 25); max_dim_s = max_dim // 4
                    if max_dim_s <= min_dim_s: max_dim_s = min_dim_s + 10
                    height, width = np.random.randint(min_dim_s, max_dim_s), np.random.randint(min_dim_s, max_dim_s)
                    start = (r, c); end_r, end_c = min(r + height, shape[0] - 1), min(c + width, shape[1] - 1)
                    if end_r >= start[0] and end_c >= start[1]:
                         rr, cc = rectangle(start, end=(end_r, end_c), shape=shape)
                         if rr.size > 0: instance_mask[rr, cc] = True; instance_generated = True
                elif shape_type == 'ellipse':
                    # (ellipse drawing code - unchanged from previous AND version)
                    center_r, center_c = np.random.randint(0, shape[0]), np.random.randint(0, shape[1])
                    min_radius_val = max(1, max_dim // 50) * np.sqrt(scaling_factor); max_radius_val = max_dim // 10 * np.sqrt(scaling_factor)
                    if max_radius_val <= min_radius_val: max_radius_val = min_radius_val + 10
                    ry = np.random.randint(min_radius_val, max_radius_val); rx = np.random.randint(min_radius_val, max_radius_val)
                    orientation = np.random.uniform(0, np.pi)
                    rr, cc = ellipse(center_r, center_c, ry, rx, shape=shape, rotation=orientation)
                    if rr.size > 0: instance_mask[rr, cc] = True; instance_generated = True
                elif shape_type == 'ring':
                    # (ring drawing code - unchanged from previous AND version)
                    center_r, center_c = np.random.randint(0, shape[0]), np.random.randint(0, shape[1])
                    min_outer = max(10, max_dim // 20); max_outer = max_dim // 6
                    if max_outer <= min_outer: max_outer = min_outer + 10
                    outer_radius = np.random.randint(min_outer, max_outer)
                    thickness_fraction = np.random.uniform(0.3, 0.7)
                    inner_radius = max(1, int(outer_radius * (1.0 - thickness_fraction)))
                    rr_out, cc_out = disk((center_r, center_c), outer_radius, shape=shape)
                    rr_in, cc_in = disk((center_r, center_c), inner_radius, shape=shape)
                    mask_out = np.zeros(shape, dtype=bool); mask_in = np.zeros(shape, dtype=bool)
                    mask_out[rr_out, cc_out] = True; mask_in[rr_in, cc_in] = True
                    ring_mask = mask_out & (~mask_in)
                    if np.any(ring_mask): instance_mask[ring_mask] = True; instance_generated = True

                # --- NEW: Handle 'gauss_blob' by drawing disks ---
                elif shape_type == 'gauss_blob':
                    center_r, center_c = np.random.randint(0, shape[0]), np.random.randint(0, shape[1])
                    radius = np.random.randint(gauss_blob_radius_range[0], gauss_blob_radius_range[1] + 1)
                    radisu *= scaling_factor
                    rr, cc = disk((center_r, center_c), radius, shape=shape)
                    if rr.size > 0:
                         instance_mask[rr, cc] = True
                         instance_generated = True

                # --- Accumulate the AND mask ---
                if instance_generated:
                    layer_and_mask = layer_and_mask | instance_mask
                    instances_generated_for_layer += 1
                else:
                    pass # Ignore failed instance

            except Exception as e:
                # print(f"Warning: AND Instance {inst_idx} of {shape_type} failed: {e}")
                pass

        # --- Process the final layer AND mask ---
        if instances_generated_for_layer > 0 and np.any(layer_and_mask):
            final_layer_mask = layer_and_mask.astype(float)

            # --- Apply layer-specific blur (Conditional for gauss_blob) ---
            if shape_type == 'gauss_blob':
                layer_blur_sigma = np.random.uniform(gauss_blob_blur_sigma_range[0], gauss_blob_blur_sigma_range[1])
            else:
                layer_blur_sigma = np.random.uniform(0, max_layer_blur_sigma)

            if layer_blur_sigma > 0.1:
                blurred_layer_mask = ndi.gaussian_filter(final_layer_mask, sigma=layer_blur_sigma)
                # Normalization after blur is less critical now intensity is handled separately
            else:
                blurred_layer_mask = final_layer_mask

            # Apply layer intensity (potentially boosted for gauss_blob)
            final_layer_contribution = blurred_layer_mask * layer_intensity

            # Combine with base using addition
            base += final_layer_contribution

    # --- Add Perlin Noise (same as before) ---
    # ... perlin noise code ...
    octaves = np.random.randint(perlin_octaves_range[0], perlin_octaves_range[1] + 1)
    noise = PerlinNoise(octaves=octaves, seed=np.random.randint(10000))
    perlin_map = np.array([[noise([i/shape[0], j/shape[1]]) for j in range(shape[1])] for i in range(shape[0])])
    if np.max(perlin_map) > np.min(perlin_map) + 1e-9: perlin_map = (perlin_map - np.min(perlin_map)) / (np.max(perlin_map) - np.min(perlin_map))
    perlin_blur_sigma = np.random.uniform(blur_perlin_noise_sigma_range[0], blur_perlin_noise_sigma_range[1])
    if perlin_blur_sigma > 0.1:
        perlin_map = ndi.gaussian_filter(perlin_map, sigma=perlin_blur_sigma)
        if np.max(perlin_map) > np.min(perlin_map) + 1e-9: perlin_map = (perlin_map - np.min(perlin_map)) / (np.max(perlin_map) - np.min(perlin_map))
    perlin_weight = np.random.uniform(perlin_weight_range[0], perlin_weight_range[1])
    base = base + perlin_map * perlin_weight


    # --- Optional: Apply subtle final global blur (same as before) ---
    # ... final blur code ...
    final_global_blur_sigma = np.random.uniform(0, max_final_global_blur_sigma)
    if final_global_blur_sigma > 0.1: base = ndi.gaussian_filter(base, sigma=final_global_blur_sigma)


    # --- Final Normalization & Clipping (same as before) ---
    # ... normalization code ...
    min_val, max_val = np.min(base), np.max(base)
    if max_val > min_val + 1e-9: base = (base - min_val) / (max_val - min_val)
    else: base = np.zeros_like(base) + min_val
    base = np.clip(base, 0.0, 1.0)

    return base

# --- Complex Object Creator (Using Physical Models) ---
def create_complex_object(
    base_map,
    amplitude_target_range=(0.8, 1.2),
    pure_phase_prob=0.05,
    pure_amp_prob=0.01
    ):
    """
    Derives complex object from a base map, constraining the final
    amplitude to amplitude_target_range and DIRECTLY mapping base_map [0,1]
    to phase [-pi, pi] without wrapping.

    Args:
        base_map (np.ndarray): Input map, assumed normalized [0, 1].
        amplitude_target_range (tuple): Desired (min, max) for the final amplitude.
        pure_phase_prob (float): Probability of generating a nearly pure phase object
                                 (amplitude forced to 1.0).
        pure_amp_prob (float): Probability of generating a nearly pure amplitude object
                               (phase forced to 0.0).

    Returns:
        np.ndarray: Complex-valued object array.
    """

    # --- Map base_map directly to target Amplitude range ---
    A_min, A_max = amplitude_target_range
    if not (A_min <= A_max):
        raise ValueError(f"amplitude_target_range minimum ({A_min}) must be less than or equal to maximum ({A_max})")
    # Map base_map [0, 1] to [A_max, A_min] (inverted relationship)
    amplitude_final = A_max - base_map * (A_max - A_min)
    amplitude_final = np.clip(amplitude_final, A_min, A_max) # Safeguard clip

    # --- Directly map base_map [0, 1] to Phase [-pi, pi] ---
    base_map = np.nan_to_num(base_map, nan=0.0, posinf=1.0, neginf=0.0)
    base_map = np.clip(base_map, 0, 1)  # Ensure strict [0,1] bounds

    # Skip indirect mapping to direct interpolation-based mapping
    phase_final = np.interp(base_map, [0, 1], [-np.pi, np.pi])

    # --- Combine ---
    complex_object = amplitude_final * np.exp(1j * phase_final)

    return complex_object

#Downscaling larger images for grid continuity
def downscale_complex_image(complex_img, scale_factor):
    # Split into real and imaginary components
    real_part = np.real(complex_img)
    imag_part = np.imag(complex_img)
    
    # Downscale each component separately
    real_downscaled = ndi.zoom(real_part, 1/scale_factor, order=3)  # cubic spline
    imag_downscaled = ndi.zoom(imag_part, 1/scale_factor, order=3)
    
    # Recombine into complex image
    return real_downscaled + 1j * imag_downscaled

# --- 3D nanoparticle generation ---
def create_complex_polyhedra(img_shape):
    canvas_xy_dims = (img_shape[0]+50, img_shape[1] + 50)
    canvas_z_depth = 128
    num_particles = 20 + int((canvas_xy_dims[0]-200)/5)
    physical_dz = 0.1

    target_amp_min, target_amp_max = 0.6, 1.2 # Define desired output amplitude range

    if torch.cuda.is_available(): device_to_use = 'cuda'
    else: device_to_use = 'cpu'

    print(f"Generating 3D polyhedra with background noise, projecting on {device_to_use}...")
    amp_2d, phase_2d_scaled = generate_3d_polyhedra_and_project(
    canvas_dims_xy=canvas_xy_dims, canvas_depth_z_voxels=canvas_z_depth,
    num_particles_target=num_particles, voxel_size_dz_physical=physical_dz,
    max_placement_attempts_factor=10,
    base_scale_range=(0.1, 0.15),
    asymmetry_factors_range=(0.5, 1.5),
    vertex_jitter_factor=0.1,
    hollow_particle_probability=0.3, # Make 60% of particles hollow
    shell_thickness_factor_range=(0.9, 0.97), # Inner void is 65-85% of outer before jitter
    particle_beta_per_voxel_range=(0.001, 0.003), #Abs
    particle_delta_per_voxel_range=(0.005, 0.03), #Phase
    target_min_amplitude=target_amp_min,
    target_max_amplitude=target_amp_max,
    bg_noise_amplitude_std_dev=0.05, # Std dev for amp noise (subtracted from 1)
    bg_noise_phase_std_dev=0.1,     # Std dev for phase noise (radians)
    bg_noise_blur_sigma=15.0,        # Controls smoothness of background noise
    device_str=device_to_use
    )

    amp_cropped = hh.center_crop(amp_2d, img_shape[0])
    phase_cropped = hh.center_crop(phase_2d_scaled, img_shape[0])
    
    #Turn complex
    obj = amp_cropped * np.exp(1j * phase_cropped)
    

    return obj


# --- Helper Functions ---

# --- Polyhedron Definition (Icosahedron) ---
def get_icosahedron_vertices_faces():
    """Returns vertices and triangular faces of a regular icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices_np = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=np.float32)
    faces_np = np.array([ # Ensure counter-clockwise for outward normals usually
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.int64) # Use int64 for PyTorch indexing
    return vertices_np, faces_np

# --- PyTorch Helper Functions ---
def get_face_planes_pytorch(vertices_tensor, faces_tensor):
    """
    Calculates plane equations (normals and d values) for a convex polyhedron.
    Assumes normals point outwards. Plane eq: dot(N, P) - d = 0.
    Point P is inside if dot(N,P)-d <= 0 for all outward normals N.
    """
    face_verts = vertices_tensor[faces_tensor] # (num_faces, 3_verts_per_face, 3_coords)
    v0, v1, v2 = face_verts[:, 0, :], face_verts[:, 1, :], face_verts[:, 2, :]
    
    edge1, edge2 = v1 - v0, v2 - v0
    normals = torch.cross(edge1, edge2, dim=1)
    
    norms_mag = torch.linalg.norm(normals, dim=1, keepdim=True)
    # Prevent division by zero for tiny/degenerate faces
    valid_norms_mask = norms_mag.squeeze() > 1e-9
    if torch.any(valid_norms_mask): # Process only valid normals
        normals[valid_norms_mask] /= norms_mag[valid_norms_mask]
    
    # Ensure normals point outwards from the centroid (assumed near origin for local vertices)
    # For a convex shape, if dot(normal, representative_face_point - centroid) < 0, flip normal.
    # Here, using v0 as a representative point. Assuming centroid is (0,0,0) for local coords.
    # So, if dot(normal, v0) < 0, flip. (This heuristic depends on centroid being near origin)
    # A more robust method might be needed if jitter creates highly non-centrosymmetric local shapes.
    # For now, rely on consistent face winding. If issues, this is a point to refine.
    
    d_values = torch.sum(normals * v0, dim=1) # d = dot(N, P0_face)
    return normals, d_values

def is_inside_convex_polyhedron_pytorch(points_batch_local, face_normals_local, face_d_values_local, epsilon=1e-5):
    """
    Checks if a batch of points (in local coords) is inside a convex polyhedron.
    """
    # plane_evals = dot(points_batch_local, Normal_transposed) - d_values_broadcasted
    plane_evals = torch.matmul(points_batch_local, face_normals_local.T) - face_d_values_local.unsqueeze(0)
    is_inside_mask = torch.all(plane_evals <= epsilon, dim=1)
    return is_inside_mask

# --- Main Generator using PyTorch ---
def generate_3d_polyhedra_and_project(
    canvas_dims_xy=(128, 128), canvas_depth_z_voxels=64,
    num_particles_target=10, voxel_size_dz_physical=0.1,
    max_placement_attempts_factor=15,
    # Particle shape
    base_scale_range=(0.05, 0.15),
    asymmetry_factors_range=(0.7, 1.3),
    vertex_jitter_factor=0.1,
    # --- Shell Parameters ---
    hollow_particle_probability=0.5, # Probability a particle is hollow
    shell_thickness_factor_range=(0.6, 0.9), # Inner radius = outer_radius * factor (smaller factor = thicker shell)
    # Material properties
    particle_beta_per_voxel_range=(0.0005, 0.005),
    particle_delta_per_voxel_range=(0.002, 0.02),
    # --- Output Amplitude/Phase Constraints ---
    target_min_amplitude=0.6, # <<< NEW
    target_max_amplitude=1.2, # <<< NEW
    target_min_phase=-np.pi, target_max_phase=np.pi,
    # --- Background Gaussian Noise Parameters ---
    bg_noise_amplitude_std_dev=0.01,
    bg_noise_phase_std_dev=0.05,
    bg_noise_blur_sigma=2.0,
    device_str='cpu'
    ):

    device = torch.device(device_str)
    print(f"Using device: {device}")

    NX_voxels, NY_voxels = canvas_dims_xy
    NZ_voxels = canvas_depth_z_voxels
    canvas_dims_voxels_np = np.array([NX_voxels, NY_voxels, NZ_voxels])
    smallest_canvas_dim = np.min(canvas_dims_voxels_np)

    beta_canvas_3d_torch = torch.zeros((NX_voxels, NY_voxels, NZ_voxels), dtype=torch.float32, device=device)
    delta_canvas_3d_torch = torch.zeros((NX_voxels, NY_voxels, NZ_voxels), dtype=torch.float32, device=device)
    
    placed_particles_count = 0
    base_vertices_np, base_faces_np = get_icosahedron_vertices_faces()
    base_vertices_torch = torch.from_numpy(base_vertices_np).to(device)
    base_faces_torch = torch.from_numpy(base_faces_np).to(device)

    start_time_generation = time.time()

    for attempt_idx in range(num_particles_target * max_placement_attempts_factor):
        if placed_particles_count >= num_particles_target: break

        # --- A. Determine if particle is hollow ---
        is_hollow = np.random.rand() < hollow_particle_probability
        current_shell_thickness_factor = 1.0 # For solid particles
        if is_hollow:
            current_shell_thickness_factor = np.random.uniform(
                shell_thickness_factor_range[0], shell_thickness_factor_range[1]
            )

        # --- B. Define Outer Polyhedron (Local Coordinates) ---
        current_base_scale = np.random.uniform(base_scale_range[0], base_scale_range[1]) * smallest_canvas_dim
        scale_factors_np = np.random.uniform(asymmetry_factors_range[0], asymmetry_factors_range[1], size=3)
        scale_factors_torch = torch.from_numpy(scale_factors_np).float().to(device)
        
        outer_local_vertices_no_jitter = base_vertices_torch * scale_factors_torch * current_base_scale
        
        avg_radius_outer = torch.mean(torch.linalg.norm(outer_local_vertices_no_jitter, dim=1))
        jitter_magnitude_outer = avg_radius_outer * vertex_jitter_factor
        jitter_outer = (torch.rand_like(outer_local_vertices_no_jitter) - 0.5) * 2.0 * jitter_magnitude_outer
        outer_local_vertices = outer_local_vertices_no_jitter + jitter_outer
        
        rotation_obj_scipy = Rotation.random() # Same rotation for outer and inner
        rotation_matrix_torch = torch.from_numpy(rotation_obj_scipy.as_matrix()).float().to(device)
        outer_local_vertices = torch.matmul(outer_local_vertices, rotation_matrix_torch.T)
        
        outer_face_normals_local, outer_face_d_values_local = get_face_planes_pytorch(outer_local_vertices, base_faces_torch)
        
        # --- C. Define Inner Polyhedron (Void) if hollow ---
        inner_face_normals_local, inner_face_d_values_local = None, None
        if is_hollow:
            # Inner void starts from the same scaled (but not jittered) outer vertices, then scaled down
            # This makes the shell thickness more uniform before jitter.
            # Jitter for inner part could be independent or scaled down version of outer jitter.
            # Let's use independent, smaller jitter for the inner surface.
            inner_local_vertices_no_jitter = outer_local_vertices_no_jitter * current_shell_thickness_factor
            
            avg_radius_inner = torch.mean(torch.linalg.norm(inner_local_vertices_no_jitter, dim=1))
            jitter_magnitude_inner = avg_radius_inner * vertex_jitter_factor # Can use a different factor if desired
            jitter_inner = (torch.rand_like(inner_local_vertices_no_jitter) - 0.5) * 2.0 * jitter_magnitude_inner
            inner_local_vertices = inner_local_vertices_no_jitter + jitter_inner
            inner_local_vertices = torch.matmul(inner_local_vertices, rotation_matrix_torch.T) # Same rotation
            
            inner_face_normals_local, inner_face_d_values_local = get_face_planes_pytorch(inner_local_vertices, base_faces_torch)

        # --- D. Calculate World AABB (based on outer shell) & Center ---
        local_min_coords_outer = torch.min(outer_local_vertices, dim=0)[0]
        local_max_coords_outer = torch.max(outer_local_vertices, dim=0)[0]
        margin_x_min, margin_y_min, margin_z_min = -local_min_coords_outer.cpu().numpy()
        margin_x_max, margin_y_max, margin_z_max = local_max_coords_outer.cpu().numpy()
        center_x = np.random.uniform(margin_x_min, NX_voxels - margin_x_max)
        center_y = np.random.uniform(margin_y_min, NY_voxels - margin_y_max)
        center_z = np.random.uniform(margin_z_min, NZ_voxels - margin_z_max)
        particle_center_world_torch = torch.tensor([center_x, center_y, center_z], dtype=torch.float32, device=device)
        world_aabb_min = particle_center_world_torch + local_min_coords_outer
        world_aabb_max = particle_center_world_torch + local_max_coords_outer
        min_ix, min_iy, min_iz = torch.floor(world_aabb_min).long().clamp(min=0)
        max_ix, max_iy, max_iz = torch.ceil(world_aabb_max).long().clamp(max=torch.tensor(canvas_dims_voxels_np, device=device)-1)
        if not (min_ix <= max_ix and min_iy <= max_iy and min_iz <= max_iz): continue

        # --- E. Material Properties ---
        current_particle_beta = np.random.uniform(particle_beta_per_voxel_range[0], particle_beta_per_voxel_range[1])
        current_particle_delta = np.random.uniform(particle_delta_per_voxel_range[0], particle_delta_per_voxel_range[1])

        # --- F. Voxelization (GPU accelerated) ---
        ix_coords = torch.arange(min_ix, max_ix + 1, device=device, dtype=torch.float32)
        iy_coords = torch.arange(min_iy, max_iy + 1, device=device, dtype=torch.float32)
        iz_coords = torch.arange(min_iz, max_iz + 1, device=device, dtype=torch.float32)
        if ix_coords.numel() == 0 or iy_coords.numel() == 0 or iz_coords.numel() == 0: continue
        grid_x, grid_y, grid_z = torch.meshgrid(ix_coords, iy_coords, iz_coords, indexing='ij')
        voxel_centers_world_flat = torch.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), dim=1) + 0.5
        if voxel_centers_world_flat.shape[0] == 0: continue
        voxel_centers_local_flat = voxel_centers_world_flat - particle_center_world_torch.unsqueeze(0)
        
        is_inside_outer_shell_flat = is_inside_convex_polyhedron_pytorch(
            voxel_centers_local_flat, outer_face_normals_local, outer_face_d_values_local
        )
        
        is_part_of_material_flat = is_inside_outer_shell_flat
        if is_hollow and inner_face_normals_local is not None:
            is_inside_inner_void_flat = is_inside_convex_polyhedron_pytorch(
                voxel_centers_local_flat, inner_face_normals_local, inner_face_d_values_local
            )
            is_part_of_material_flat = is_inside_outer_shell_flat & (~is_inside_inner_void_flat)
            
        inside_indices_global_flat = voxel_centers_world_flat[is_part_of_material_flat, :3].long()
        if inside_indices_global_flat.shape[0] == 0: continue
            
        # --- G. Place particle ("first wins") ---
        temp_particle_mask_3d = torch.zeros_like(beta_canvas_3d_torch, dtype=torch.bool)
        idx_x = inside_indices_global_flat[:,0].clamp(min=0, max=NX_voxels-1)
        idx_y = inside_indices_global_flat[:,1].clamp(min=0, max=NY_voxels-1)
        idx_z = inside_indices_global_flat[:,2].clamp(min=0, max=NZ_voxels-1)
        temp_particle_mask_3d[idx_x, idx_y, idx_z] = True
        empty_slots_for_particle = (beta_canvas_3d_torch == 0) & temp_particle_mask_3d
        beta_canvas_3d_torch[empty_slots_for_particle] = current_particle_beta
        delta_canvas_3d_torch[empty_slots_for_particle] = current_particle_delta
        if torch.any(empty_slots_for_particle): placed_particles_count += 1

    end_time_generation = time.time()
    print(f"Polyhedron/Shell generation loop took: {end_time_generation - start_time_generation:.2f} s")
    print(f"Successfully placed {placed_particles_count} particles/shells.")

    # --- Projection to 2D (GPU) ---
    start_time_projection = time.time()
    integrated_beta_path = torch.sum(beta_canvas_3d_torch, dim=2)
    integrated_delta_path = torch.sum(delta_canvas_3d_torch, dim=2)

    amplitude_2d_projected_raw = torch.exp(-integrated_beta_path * voxel_size_dz_physical)
    raw_phase_2d_projected_raw = -integrated_delta_path * voxel_size_dz_physical

    # --- Add Conditional Smoothed 2D Gaussian Background Noise (Your Block) ---
    epsilon = 1e-5
    empty_region_mask = (torch.abs(amplitude_2d_projected_raw - 1.0) < epsilon) & \
                        (torch.abs(raw_phase_2d_projected_raw) < epsilon)

    amplitude_2d_final_pre_scale = amplitude_2d_projected_raw.clone() # Before final amp scaling
    raw_phase_2d_final = raw_phase_2d_projected_raw.clone()

    if torch.any(empty_region_mask) and (bg_noise_amplitude_std_dev > 0 or bg_noise_phase_std_dev > 0):
        # Generate full-field Gaussian noise N(0,1) first, then scale by std_dev
        full_amp_noise_raw_scaled = torch.randn(NX_voxels, NY_voxels, device=device) * bg_noise_amplitude_std_dev
        full_phase_noise_raw_scaled = torch.randn(NX_voxels, NY_voxels, device=device) * bg_noise_phase_std_dev

        processed_amp_noise = full_amp_noise_raw_scaled
        processed_phase_noise = full_phase_noise_raw_scaled

        if bg_noise_blur_sigma > 0.01:
            amp_noise_np = full_amp_noise_raw_scaled.cpu().numpy()
            phase_noise_np = full_phase_noise_raw_scaled.cpu().numpy()
            smoothed_amp_noise_np = gaussian_filter(amp_noise_np, sigma=bg_noise_blur_sigma)
            smoothed_phase_noise_np = gaussian_filter(phase_noise_np, sigma=bg_noise_blur_sigma)
            processed_amp_noise = torch.from_numpy(smoothed_amp_noise_np).to(device)
            processed_phase_noise = torch.from_numpy(smoothed_phase_noise_np).to(device)
        
        if bg_noise_amplitude_std_dev > 0:
            # Your block implies noise is already scaled. For amplitude, ensure it reduces from 1.
            # Use abs of the (potentially smoothed) scaled noise.
            amp_noise_values_to_subtract = torch.abs(processed_amp_noise[empty_region_mask])
            amplitude_2d_final_pre_scale[empty_region_mask] = 1.0 - amp_noise_values_to_subtract
            # This amplitude_2d_final_pre_scale will then be rescaled to target range.

        if bg_noise_phase_std_dev > 0:
            phase_noise_values = processed_phase_noise[empty_region_mask]
            raw_phase_2d_final[empty_region_mask] = phase_noise_values # Add scaled, possibly smoothed noise
    
    # --- Scale Amplitude to [target_min_amplitude, target_max_amplitude] ---
    # This is applied to the entire image (particles + noisy background)
    current_min_amp = torch.min(amplitude_2d_final_pre_scale)
    current_max_amp = torch.max(amplitude_2d_final_pre_scale)
    # print(f'min amp: {current_min_amp}, max_amp: {current_max_amp}')
    amplitude_2d_final_scaled = torch.zeros_like(amplitude_2d_final_pre_scale)

    if torch.abs(current_max_amp - current_min_amp) < 1e-9: # If amplitude map is flat
        amplitude_2d_final_scaled.fill_((target_min_amplitude + target_max_amplitude) / 2.0)
    else:
        amplitude_2d_final_scaled = target_min_amplitude + \
            (amplitude_2d_final_pre_scale - current_min_amp) * \
            (target_max_amplitude - target_min_amplitude) / (current_max_amp - current_min_amp)
    amplitude_2d_final_scaled = torch.clamp(amplitude_2d_final_scaled, target_min_amplitude, target_max_amplitude)
    
    # --- Scale Phase to target range [-pi, pi] ---
    min_raw_p, max_raw_p = torch.min(raw_phase_2d_final), torch.max(raw_phase_2d_final)
    if torch.abs(max_raw_p - min_raw_p) < 1e-9:
        phase_2d_scaled_torch = torch.full_like(raw_phase_2d_final, (target_min_phase + target_max_phase) / 2.0)
        if target_min_phase <= 0.0 <= target_max_phase and torch.abs(min_raw_p) < 1e-9:
             phase_2d_scaled_torch.fill_(0.0)
    else:
        phase_2d_scaled_torch = target_min_phase + \
            (raw_phase_2d_final - min_raw_p) * \
            (target_max_phase - target_min_phase) / (max_raw_p - min_raw_p)
    
    phase_2d_scaled_torch = torch.clamp(phase_2d_scaled_torch, target_min_phase, target_max_phase)
    end_time_projection = time.time(); print(f"Projection, Noise & Scaling took: {end_time_projection - start_time_projection:.2f} s")

    return amplitude_2d_final_scaled.cpu().numpy(), phase_2d_scaled_torch.cpu().numpy()

# --- Example Usage ---
# if __name__ == "__main__":
    # canvas_xy_dims = (256, 256)
    # canvas_z_depth = 128
    # num_particles = 30 # Reduced slightly due to complexity of shells
    # physical_dz = 0.1

    # if torch.cuda.is_available(): device_to_use = 'cuda'
    # else: device_to_use = 'cpu'

    # target_amp_min, target_amp_max = 0.6, 1.2 # Define desired output amplitude range

    # print(f"Generating 3D shells/polyhedra with BG noise, projecting on {device_to_use}...")
    # amp_2d, phase_2d_scaled = generate_3d_shells_and_project(
    #     canvas_dims_xy=canvas_xy_dims, canvas_depth_z_voxels=canvas_z_depth,
    #     num_particles_target=num_particles, voxel_size_dz_physical=physical_dz,
    #     max_placement_attempts_factor=25, # Shells might need more attempts if inner part makes them "vanish"
    #     base_scale_range=(0.04, 0.12), # Slightly larger to accommodate shells
    #     asymmetry_factors_range=(0.6, 1.4),
    #     vertex_jitter_factor=0.15,
    #     hollow_particle_probability=0.6, # Make 60% of particles hollow
    #     shell_thickness_factor_range=(0.65, 0.85), # Inner void is 65-85% of outer before jitter
    #     particle_beta_per_voxel_range=(0.001, 0.008),
    #     particle_delta_per_voxel_range=(0.005, 0.03),
    #     target_min_amplitude=target_amp_min,
    #     target_max_amplitude=target_amp_max,
    #     bg_noise_amplitude_std_dev=0.015, # This will be std dev BEFORE blur
    #     bg_noise_phase_std_dev=0.1,    # This will be std dev BEFORE blur
    #     bg_noise_blur_sigma=3.0,
    #     device_str=device_to_use
    # )

def create_dead_leaves(img_shape, obj_arg):
    """
    Wrapper for dead leaves function. Only takes square shapes.

    Args:
        img_shape (int, int): Image dimensions in (h,w)
        obj_arg (Dict): Passable dictionary with object generation arguments
    """


    # For testing, reduce iterations to see individual leaves better
    MAX_ITERS = obj_arg.get('max_iters',500) # Original code has 5000

    R_MIN_FRAC = obj_arg.get('r_min_frac', 0.03)
    R_MAX_FRAC = obj_arg.get('r_max_frac', 0.2)
    R_SIGMA = obj_arg.get('r_sigma', 3) # For radii power-law

    # Adjust materials "properties" to change phase and amplitude attenuation
    # These are arbitrarily defined refractive indice values to give the synthetic objects some sort of
    # "meaning"

    BETA_PARETO_ALPHA = 1.5
    BETA_SCALE = 0.001
    DELTA_BETA_RATIO_MEAN = 100
    DELTA_BETA_RATIO_STD = 10
    EFFECTIVE_THICKNESS = 3.0

    MIN_PHASE, MAX_PHASE = -np.pi, np.pi

    MIN_AMP, MAX_AMP = 0.6, 1.1

    amp_2d, phase_2d, _, _ = dead_leaves_ptycho(res=img_shape[0], r_sigma_param=R_SIGMA,
                        max_iters = MAX_ITERS,
                        r_min_frac = R_MIN_FRAC, r_max_frac = R_MAX_FRAC,
                        beta_pareto_alpha = BETA_PARETO_ALPHA, beta_scale = BETA_SCALE,
                        delta_beta_mean = DELTA_BETA_RATIO_MEAN, delta_beta_std = DELTA_BETA_RATIO_STD,
                        thickness = EFFECTIVE_THICKNESS,
                        min_phase = MIN_PHASE, max_phase = MAX_PHASE,
                        min_amp = MIN_AMP, max_amp = MAX_AMP)
    
    obj = amp_2d * np.exp(1j * phase_2d)

    return obj

def get_skewed_random_value(alpha, scale, min_clip=1e-6, max_clip=None):
    """
    Generates a random value from a distribution skewed to small values.
    Example using Pareto: P(x) ~ x^-(alpha+1) for x >= scale
    Here, we'll use 1/U trick for power law or simpler exponential.
    For simplicity, let's use an exponential for now, can be refined.
    A simple way to get skewed values: (random_uniform() ^ power) * scale
    """
    # Using (random_uniform ^ power)
    # power > 1 skews towards 0
    # power < 1 skews towards scale
    power = alpha # Reuse alpha, larger alpha means more skew to 0
    val = (np.random.uniform(0, 1) ** power) * scale
    val = max(val, min_clip)
    if max_clip is not None:
        val = min(val, max_clip)
    return val


def dead_leaves_ptycho(res, r_sigma_param, max_iters,
                       r_min_frac, r_max_frac,
                       beta_pareto_alpha, beta_scale, delta_beta_mean, delta_beta_std,
                       thickness,
                       min_phase, max_phase,
                       min_amp, max_amp):
    # --- Initialize Canvases ---
    # These will store the material properties (beta, delta) of the TOPMOST leaf at each pixel
    beta_map = np.zeros((res, res), dtype=np.float32)
    delta_map = np.zeros((res, res), dtype=np.float32)
    # Optional: keep track of coverage to stop early
    # coverage_mask = np.zeros((res, res), dtype=bool)

    # --- Radii Distribution ---
    rmin_abs = int(res * r_min_frac)
    rmax_abs = int(res * r_max_frac)
    if rmin_abs < 1: rmin_abs = 1
    if rmax_abs <= rmin_abs: rmax_abs = rmin_abs +1

    r_list_abs = np.linspace(rmin_abs, rmax_abs, 200) # Radii in pixels
    if r_sigma_param > 0:
        r_dist = 1. / (r_list_abs ** r_sigma_param)
        r_dist = r_dist - 1. / (rmax_abs ** r_sigma_param) # Normalize tail
    else: # Uniform radius distribution if sigma is 0 or less
        r_dist = np.ones_like(r_list_abs)

    r_dist = np.cumsum(r_dist)
    if r_dist.max() > 1e-9: # Avoid division by zero if all r_dist are zero
        r_dist = r_dist / r_dist.max()
    else: # Fallback if distribution is problematic
        r_dist = np.linspace(0,1, len(r_list_abs))


    # --- Main Loop: Adding Leaves ---
    for i in range(max_iters):
        # 1. Select Shape Type
        available_shapes = ['circle', 'oriented_square', 'rectangle', 'triangle', 'quadrilater']
        shape = random.choice(available_shapes)

        # 2. Select Size (Radius)
        r_p = np.random.uniform(0, 1)
        r_i = np.argmin(np.abs(r_dist - r_p))
        radius_pixels = max(int(r_list_abs[r_i]), 1)

        # 3. Select Material Properties (beta, delta)
        # Skewed towards small values
        current_beta = get_skewed_random_value(alpha=beta_pareto_alpha, scale=beta_scale, min_clip=1e-7)
        
        ratio = np.random.normal(loc=delta_beta_mean, scale=delta_beta_std)
        ratio = max(1.0, ratio) # Ensure delta is at least beta
        current_delta = current_beta * ratio
        
        # Ensure delta is not excessively large if beta is tiny but ratio is huge
        # This can be capped if necessary, e.g., current_delta = min(current_delta, MAX_DELTA_VALUE)

        # 4. Select Position
        center_x, center_y = np.random.randint(0, res, size=2)

        # 5. Create a Mask for the Current Shape
        # We need a temporary 2D boolean mask for the current shape
        # OpenCV drawing functions can draw on a single channel image (e.g., uint8)
        # which can then be converted to a boolean mask.
        temp_mask = np.zeros((res, res), dtype=np.uint8)

        if shape == 'circle':
            cv2.circle(temp_mask, (center_x, center_y), radius=radius_pixels, color=1, thickness=-1)
        else:
            # (Polygon drawing logic from your code, adapted to draw '1' on temp_mask)
            if shape == 'oriented_square':
                side = radius_pixels * np.sqrt(2) # radius_pixels is like circumradius here
                corners_rel = np.array(((-side / 2, -side / 2), (+side / 2, -side / 2),
                                        (+side / 2, +side / 2), (-side / 2, +side / 2)))

                theta = np.random.uniform(0, 2 * np.pi)
                c, s = np.cos(theta), np.sin(theta)
                R_mat = np.array(((c, -s), (s, c)))
                corners_rel = (R_mat @ corners_rel.T).T
            elif shape == 'rectangle':
                a = np.random.uniform(0, 0.5 * np.pi)
                rx, ry = radius_pixels * np.cos(a), radius_pixels * np.sin(a) # radii of rectangle
                corners_rel = np.array(((+rx, +ry), (+rx, -ry), (-rx, -ry), (-rx, +ry)))
                
                theta = np.random.uniform(0, 2 * np.pi)
                c, s = np.cos(theta), np.sin(theta)
                R_mat = np.array(((c, -s), (s, c)))
                corners_rel = (R_mat @ corners_rel.T).T
            else: # triangle or quadrilateral
                num_verts = 3 if shape == 'triangle' else 4
                angles = sorted(np.random.uniform(0, 2 * np.pi, num_verts))
                corners_rel = []
                for ang in angles:
                    corners_rel.append((radius_pixels * np.cos(ang), radius_pixels * np.sin(ang)))
                corners_rel = np.array(corners_rel)
            
            corners_abs = (np.array([center_x, center_y]) + corners_rel).astype(np.int32)
            cv2.fillPoly(temp_mask, [corners_abs], color=1)
        
        current_shape_bool_mask = temp_mask.astype(bool)

        # 6. Update beta_map and delta_map (Occlusion: new leaf replaces old)
        beta_map[current_shape_bool_mask] = current_beta
        delta_map[current_shape_bool_mask] = current_delta
        
        # Optional: Early stopping if fully covered
        # coverage_mask |= current_shape_bool_mask
        # if np.all(coverage_mask):
        # print(f"Fully covered at iteration {i+1}")
        # break
        
    # --- Convert Accumulated Beta/Delta to Amplitude/Phase ---
    # Amplitude = exp(-beta_map * effective_thickness)
    # Let's assume beta_map and delta_map are effective values per "unit depth" of a leaf.
    
    amplitude_2d = np.exp(-beta_map * thickness)
    amplitude_2d = np.clip(amplitude_2d, 0.0, 1.0) # Should be inherent

    raw_phase_2d = -delta_map * thickness # Sign convention

    # --- Normalize/Scale Final Images ---
    # Amplitude: Optionally scale to a target range (e.g. [0.6, 1.0])
    # For now, let's keep it 0-1 as derived from exp.
    if min_amp and max_amp:
      min_amp_raw = np.min(amplitude_2d)
      max_amp_raw = np.max(amplitude_2d)
      if max_amp_raw > min_amp_raw + 1e-9:
          amplitude_2d = min_amp + \
              (amplitude_2d - min_amp_raw) * \
              (max_amp - min_amp) / (max_amp_raw - min_amp_raw)
      else: # Flat amplitude
          amplitude_2d.fill((min_amp + max_amp) / 2.0)
      amplitude_2d = np.clip(amplitude_2d, min_amp, max_amp)

    # Phase: Scale to [TARGET_MIN_PHASE, TARGET_MAX_PHASE]
    min_raw_phase = np.min(raw_phase_2d)
    max_raw_phase = np.max(raw_phase_2d)
    phase_2d_scaled = np.zeros_like(raw_phase_2d)

    if np.abs(max_raw_phase - min_raw_phase) < 1e-9: # If phase map is flat
        if min_phase <= 0.0 <= max_phase and np.abs(min_raw_phase) < 1e-9 :
            phase_2d_scaled.fill(0.0)
        else:
            phase_2d_scaled.fill((min_phase + min_phase) / 2.0)
    else:
        phase_2d_scaled = min_phase + \
            (raw_phase_2d - min_raw_phase) * \
            (max_phase - min_phase) / (max_raw_phase - min_raw_phase)
    phase_2d_scaled = np.clip(phase_2d_scaled, min_phase, max_phase)

    print(f"Amplitude range: {np.min(amplitude_2d):.4f} - {np.max(amplitude_2d):.4f}")
    print(f"Phase range: {np.min(phase_2d_scaled):.4f} - {np.max(phase_2d_scaled):.4f} radians")
    print(f"Beta map range: {np.min(beta_map):.4e} - {np.max(beta_map):.4e}")
    print(f"Delta map range: {np.min(delta_map):.4e} - {np.max(delta_map):.4e}")

    return amplitude_2d, phase_2d_scaled, beta_map, delta_map # Return beta/delta maps for inspection


# --- Example of how to call it for testing ---
# if __name__ == "__main__":
#     # Test generation of one sample
#     RESOLUTION_TEST = 256
#     # For testing, reduce iterations to see individual leaves better
#     MAX_ITERS_TEST = 500 # Original code has 5000

#     # Adjust material properties for testing visibility
#     # Override global constants for this test block
#     _BETA_PARETO_ALPHA = 2.0
#     _BETA_SCALE = 0.01
#     _DELTA_BETA_RATIO_MEAN = 30
#     _DELTA_BETA_RATIO_STD = 10
#     _EFFECTIVE_THICKNESS = 5.0 # Make leaves "thicker" for more contrast

#     # Temporarily assign to globals for dead_leaves_ptycho to pick up
#     BETA_PARETO_ALPHA = _BETA_PARETO_ALPHA
#     BETA_SCALE = _BETA_SCALE
#     DELTA_BETA_RATIO_MEAN = _DELTA_BETA_RATIO_MEAN
#     DELTA_BETA_RATIO_STD = _DELTA_BETA_RATIO_STD
#     EFFECTIVE_THICKNESS = _EFFECTIVE_THICKNESS


#     print(f"Generating one sample with resolution {RESOLUTION_TEST}x{RESOLUTION_TEST}...")
#     amplitude, phase, final_beta_map, final_delta_map = dead_leaves_ptycho(
#         RESOLUTION_TEST, R_SIGMA, MAX_ITERS_TEST
#     )

#     print(f"Amplitude range: {np.min(amplitude):.4f} - {np.max(amplitude):.4f}")
#     print(f"Phase range: {np.min(phase):.4f} - {np.max(phase):.4f} radians")
#     print(f"Beta map range: {np.min(final_beta_map):.4e} - {np.max(final_beta_map):.4e}")
#     print(f"Delta map range: {np.min(final_delta_map):.4e} - {np.max(final_delta_map):.4e}")
