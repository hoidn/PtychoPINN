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
from sklearn.mixture import GaussianMixture

_REF_N = 64

def _pixel_scale(N):
    return N / _REF_N


def _import_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for synthetic data generation. "
            "Install it with: pip install ptychopinn[datagen]"
        ) from None


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

def create_white_noise_object(img_shape, obj_arg, N=_REF_N):
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
        s = _pixel_scale(N)
        blur_sigma = np.random.uniform(2.0, 3.0) * s
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


def create_white_noise_clustered_reim(img_shape, obj_arg, N=_REF_N):
    """
    Generate a complex object by quantizing blurred white noise to GMM clusters.

    Two independent blurred noise fields are generated in [-1.2, 1.2], then
    snapped to the nearest cluster centers from a GMM fit. This produces
    organic spatial boundaries between discrete material types.

    Parameters
    ----------
    img_shape : tuple (H, W)
        Output image dimensions.
    obj_arg : dict
        Required (one of):
        - 'reference_objects': list of complex arrays to fit GMM from.
        - 'gmm_params': dict from fit_gmm_from_objects() (skip fitting).

        Optional:
        - 'n_clusters': int or 'auto' (default 'auto')
        - 'blur': float, base noise field blur sigma (default 3.0)
        - 'quantization_mode': 'hard' or 'soft' (default 'hard')
        - 'softmax_temperature': float (default 0.1)
        - 'amp_std_scale': float, radial variance scale (default 1.0)
        - 'phase_std_scale': float, tangential variance scale (default 1.0)
        - 'rotation_range': tuple (default (0, 2*pi))
        - 'center_jitter_std': float (default 0.05)
        - 'weight_dirichlet_conc': float (default 5.0)
        - 'final_blur': float, light blur after quantization (default 0.3)
        - 'texture_scale': float, spatial correlation length for perturbation
          in pixels (default 0.0 = pixel-level IID)

    Returns
    -------
    np.ndarray : complex64 object of shape img_shape
    """
    cv2 = _import_cv2()
    rng = np.random.default_rng()

    # --- Get or fit GMM parameters ---
    gmm_params = obj_arg.get('gmm_params', None)
    if gmm_params is None:
        ref_objects = obj_arg.get('reference_objects', None)
        if ref_objects is None:
            raise ValueError("Either 'reference_objects' or 'gmm_params' must be provided")
        n_clusters = obj_arg.get('n_clusters', 6) #Used to be 'auto'
        gmm_params = fit_gmm_from_objects(ref_objects, n_clusters=n_clusters)

    means = gmm_params['means']

    # --- Scale pixel-based params to maintain relative feature size ---
    s = _pixel_scale(N)

    # --- Generate blurred noise fields ---
    blur_sigma = obj_arg.get('blur', 2.0) * s
    re_noise = np.random.uniform(-1.2, 1.2, size=img_shape).astype(np.float32)
    im_noise = np.random.uniform(-1.2, 1.2, size=img_shape).astype(np.float32)

    if blur_sigma > 0:
        re_noise = gaussian_filter(re_noise, sigma=blur_sigma)
        im_noise = gaussian_filter(im_noise, sigma=blur_sigma)
        # Rescale back to [-1.2, 1.2] after blur shrinks the range
        re_noise = _normalize_np(re_noise) * 2.4 - 1.2
        im_noise = _normalize_np(im_noise) * 2.4 - 1.2

    # --- Quantize to cluster centers ---
    q_mode = obj_arg.get('quantization_mode', 'hard')
    temperature = obj_arg.get('softmax_temperature', 0.1)

    real_q, imag_q = quantize_reim_to_clusters(
        re_noise, im_noise,
        cluster_re=means[:, 0], cluster_im=means[:, 1],
        mode=q_mode, temperature=temperature,
        cluster_covariances=gmm_params['covariances'],
        cluster_weights=gmm_params['weights'],
        amp_std_scale=obj_arg.get('amp_std_scale', 1.0),
        phase_std_scale=obj_arg.get('phase_std_scale', 1.3), #2.0 TP2
        texture_scale=obj_arg.get('texture_scale', 2.5) * s,
    )

    # --- Optional light final blur for smoother transitions ---
    final_blur = obj_arg.get('final_blur', 0.1) * s
    if final_blur > 0:
        real_q = cv2.GaussianBlur(real_q, (0, 0), final_blur)
        imag_q = cv2.GaussianBlur(imag_q, (0, 0), final_blur)

    real_q = np.clip(real_q, -1.2, 1.2)
    imag_q = np.clip(imag_q, -1.2, 1.2)

    obj = (real_q + 1j * imag_q).astype(np.complex64)
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

    # Scale pixel-based blur/radius params to maintain relative feature size
    max_layer_blur_sigma *= scaling_factor
    max_final_global_blur_sigma *= scaling_factor
    blur_perlin_noise_sigma_range = (blur_perlin_noise_sigma_range[0] * scaling_factor,
                                     blur_perlin_noise_sigma_range[1] * scaling_factor)
    gauss_blob_radius_range = (gauss_blob_radius_range[0] * scaling_factor,
                               gauss_blob_radius_range[1] * scaling_factor)
    gauss_blob_blur_sigma_range = (gauss_blob_blur_sigma_range[0] * scaling_factor,
                                   gauss_blob_blur_sigma_range[1] * scaling_factor)

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
                    radius = np.random.randint(int(gauss_blob_radius_range[0]), int(gauss_blob_radius_range[1]) + 1)
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

def create_dead_leaves(img_shape, obj_arg, histogram = None):
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
    cv2 = _import_cv2()
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

## BEta features
import numpy as np
from typing import Tuple, Optional, Union, List

def create_density_centered_histogram(
    objects: Union[np.ndarray, List[np.ndarray]], 
    bins: int = 256, 
    smoothing_sigma: float = 4.0,
    amp_range: Tuple[float, float] = (0.0, 1.0),
    phase_range: Tuple[float, float] = (-np.pi, np.pi)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregates amplitude/phase statistics from a single tensor or a list of 
    mismatched objects, identifies the vacuum mode, and returns a centered, 
    smoothed material prior.
    """
    # --- 1. Aggregate Statistics ---
    # Handle list of arrays (mismatched sizes) or single B,H,W array
    if isinstance(objects, list):
        all_amps = np.concatenate([np.abs(obj).flatten() for obj in objects])
        all_phases = np.concatenate([np.angle(obj).flatten() for obj in objects])
    else:
        all_amps = np.abs(objects).flatten()
        all_phases = np.angle(objects).flatten()
    
    # --- 2. Initial high-res histogram for peak finding ---
    hist_raw, a_edges, p_edges = np.histogram2d(
        all_amps, all_phases, bins=bins, range=[amp_range, phase_range]
    )
    
    # Smooth to identify the "center of mass" of the most common material (vacuum)
    hist_smoothed = gaussian_filter(hist_raw, sigma=smoothing_sigma)
    
    # --- 3. Identify Vacuum Mode ---
    peak_idx = np.unravel_index(np.argmax(hist_smoothed), hist_smoothed.shape)
    
    # Map index to physical values
    amp_vac = (a_edges[peak_idx[0]] + a_edges[peak_idx[0]+1]) / 2
    phase_vac = (p_edges[peak_idx[1]] + p_edges[peak_idx[1]+1]) / 2
    
    print(f"Detected Vacuum Mode: Amplitude={amp_vac:.4f}, Phase={phase_vac:.4f} rad")

    # --- 4. Re-center Data ---
    # Amplitude scaling is kept as identity per your current snippet logic
    centered_amps = all_amps 
    centered_phases = all_phases - phase_vac
    
    # Wrap phases back to [-pi, pi] to maintain circular continuity
    centered_phases = (centered_phases + np.pi) % (2 * np.pi) - np.pi
    
    # --- 5. Final Material Prior Construction ---
    # Build the final histogram within the generator's expected bounds
    final_hist, x_edges, y_edges = np.histogram2d(
        centered_amps, centered_phases, 
        bins=bins, 
        range=[[0.0, 1.0], [-np.pi, np.pi]]
    )

    # Normalize to create a valid PDF for the rng.choice sampler
    final_hist = final_hist / (np.sum(final_hist) + 1e-12)

    # Final smoothing pass to thicken the material manifold (linear space)
    final_hist = gaussian_filter(final_hist, sigma=2.0)
    
    print("Created density histogram. ")
    return final_hist, x_edges, y_edges


# =====================================================================
# Re-Im Space Synthetic Object Generation
# =====================================================================

def compute_reim_statistics(
    objects: Union[np.ndarray, List[np.ndarray]]
) -> dict:
    """
    Compute Real-Imaginary space statistics from complex-valued objects.

    Parameters
    ----------
    objects : complex array or list of complex arrays
        Single object or list of objects (may have mismatched sizes).

    Returns
    -------
    dict with keys:
        re_mean, im_mean, re_std, im_std, re_min, re_max, im_min, im_max,
        correlation, covariance (2x2), energy_ratio (E_re / E_im)
    """
    if isinstance(objects, list):
        all_re = np.concatenate([np.real(obj).flatten() for obj in objects])
        all_im = np.concatenate([np.imag(obj).flatten() for obj in objects])
    else:
        all_re = np.real(objects).flatten()
        all_im = np.imag(objects).flatten()

    re_mean = float(np.mean(all_re))
    im_mean = float(np.mean(all_im))
    re_std = float(np.std(all_re))
    im_std = float(np.std(all_im))

    # Pearson correlation
    if re_std > 1e-12 and im_std > 1e-12:
        correlation = float(np.corrcoef(all_re, all_im)[0, 1])
    else:
        correlation = 0.0

    # 2x2 covariance matrix
    covariance = np.cov(all_re, all_im)

    # Energy ratio
    e_re = float(np.mean(all_re ** 2))
    e_im = float(np.mean(all_im ** 2))
    energy_ratio = e_re / max(e_im, 1e-12)

    return {
        're_mean': re_mean, 'im_mean': im_mean,
        're_std': re_std, 'im_std': im_std,
        're_min': float(np.min(all_re)), 're_max': float(np.max(all_re)),
        'im_min': float(np.min(all_im)), 'im_max': float(np.max(all_im)),
        'correlation': correlation,
        'covariance': covariance,
        'energy_ratio': energy_ratio,
    }


def create_density_histogram_reim(
    objects: Union[np.ndarray, List[np.ndarray]],
    bins: int = 256,
    smoothing_sigma: float = 4.0,
    re_range: Tuple[float, float] = (-1.4, 1.4),
    im_range: Tuple[float, float] = (-1.4, 1.4),
    origin_threshold: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float]]:
    """
    Build a 2D histogram PDF in (Real, Imag) space from complex objects.

    Mirror of create_density_centered_histogram() but in Re-Im coordinates.
    Range [-1.2, 1.2] matches the decoder output 1.2 * tanh(x).

    Parameters
    ----------
    objects : complex array or list of complex arrays
    bins : int
        Number of histogram bins per axis.
    smoothing_sigma : float
        Gaussian smoothing sigma for peak finding.
    re_range, im_range : tuple
        Histogram axis ranges.
    origin_threshold : float
        Bins with |Re| < threshold AND |Im| < threshold are zeroed out
        to exclude unscanned regions (default: 0.05).

    Returns
    -------
    hist_pdf : np.ndarray
        Normalized 2D histogram (bins x bins).
    re_edges : np.ndarray
        Bin edges for real axis.
    im_edges : np.ndarray
        Bin edges for imaginary axis.
    vacuum_reim : tuple (float, float)
        (re, im) coordinates of the vacuum mode (histogram peak).
    """
    if isinstance(objects, list):
        all_re = np.concatenate([np.real(obj).flatten() for obj in objects])
        all_im = np.concatenate([np.imag(obj).flatten() for obj in objects])
    else:
        all_re = np.real(objects).flatten()
        all_im = np.imag(objects).flatten()

    # Build 2D histogram
    hist_raw, re_edges, im_edges = np.histogram2d(
        all_re, all_im, bins=bins, range=[list(re_range), list(im_range)]
    )

    # Mask out near-origin bins (unscanned/unilluminated regions)
    if origin_threshold > 0:
        re_centers = (re_edges[:-1] + re_edges[1:]) / 2
        im_centers = (im_edges[:-1] + im_edges[1:]) / 2
        re_grid, im_grid = np.meshgrid(re_centers, im_centers, indexing='ij')
        origin_mask = np.sqrt(re_grid**2 + im_grid**2) < origin_threshold
        hist_raw[origin_mask] = 0.0

    # Smooth for peak finding
    hist_smoothed = gaussian_filter(hist_raw, sigma=smoothing_sigma)

    # Find vacuum mode (peak)
    peak_idx = np.unravel_index(np.argmax(hist_smoothed), hist_smoothed.shape)
    vacuum_re = (re_edges[peak_idx[0]] + re_edges[peak_idx[0] + 1]) / 2
    vacuum_im = (im_edges[peak_idx[1]] + im_edges[peak_idx[1] + 1]) / 2

    print(f"Re-Im histogram vacuum mode: Re={vacuum_re:.4f}, Im={vacuum_im:.4f}")

    # Normalize to PDF
    hist_pdf = hist_raw / (np.sum(hist_raw) + 1e-12)

    # Final smoothing to thicken the material manifold
    hist_pdf = gaussian_filter(hist_pdf, sigma=2.0)
    hist_pdf = hist_pdf / (np.sum(hist_pdf) + 1e-12)

    return hist_pdf, re_edges, im_edges


def fit_gmm_from_objects(
    objects: Union[np.ndarray, List[np.ndarray]],
    n_clusters: Union[int, str] = 'auto',
    max_clusters: int = 8,
    subsample: int = 80000,
    origin_mask_radius: float = 0.4,
) -> dict:
    """
    Fit a Gaussian Mixture Model to the (Re, Im) point cloud of complex objects.

    Parameters
    ----------
    objects : complex array or list of complex arrays
        Experimental objects to extract the Re-Im distribution from.
    n_clusters : int or 'auto'
        Number of GMM components. If 'auto', sweeps K=2..max_clusters
        and selects by lowest BIC.
    max_clusters : int
        Upper bound for BIC sweep when n_clusters='auto'.
    subsample : int
        Maximum number of points to fit (random subsample for speed).

    Returns
    -------
    dict with keys:
        means : np.ndarray (K, 2)
        covariances : np.ndarray (K, 2, 2)
        weights : np.ndarray (K,)
        vacuum_reim : np.ndarray (2,)  — cluster closest to (1, 0)
        n_clusters : int
    """
    # Flatten to (Re, Im) point cloud
    if isinstance(objects, list):
        all_re = np.concatenate([np.real(obj).flatten() for obj in objects])
        all_im = np.concatenate([np.imag(obj).flatten() for obj in objects])
    else:
        all_re = np.real(objects).flatten()
        all_im = np.imag(objects).flatten()

    points = np.column_stack([all_re, all_im]).astype(np.float64)

    origin_mask = np.sqrt(all_re**2 + all_im**2) < origin_mask_radius
    points = points[~origin_mask,:]


    # Subsample for speed
    if len(points) > subsample:
        idx = np.random.default_rng().choice(len(points), size=subsample, replace=False)
        points = points[idx]

    print(f"n_clusters is: {n_clusters}")

    # Fit GMM
    if n_clusters == 'auto':
        best_bic, best_gmm = np.inf, None
        for k in range(2, max_clusters + 1):
            gmm = GaussianMixture(n_components=k, covariance_type='full',
                                  n_init=3, random_state=42)
            gmm.fit(points)
            bic = gmm.bic(points)
            if bic < best_bic:
                best_bic, best_gmm = bic, gmm
        gmm = best_gmm
        print(f"GMM auto-selected K={gmm.n_components} (BIC={best_bic:.1f})")
    else:
        gmm = GaussianMixture(n_components=int(n_clusters), covariance_type='full',
                              n_init=3, random_state=42)
        gmm.fit(points)

    means = gmm.means_.astype(np.float64)          # (K, 2)
    covariances = gmm.covariances_.astype(np.float64)  # (K, 2, 2)
    weights = gmm.weights_.astype(np.float64)       # (K,)

    # Detect vacuum mode: cluster closest to (1, 0)
    dists_to_vacuum = np.linalg.norm(means - np.array([1.0, 0.0]), axis=1)
    vacuum_idx = np.argmin(dists_to_vacuum)
    vacuum_reim = means[vacuum_idx].copy()

    print(f"GMM fit: K={len(weights)}, "
          f"centers={(means[:, 0].round(3).tolist(), means[:, 1].round(3).tolist())}, "
          f"weights={weights.round(3).tolist()}, "
          f"vacuum=({vacuum_reim[0]:.3f}, {vacuum_reim[1]:.3f})")

    return {
        'means': means,
        'covariances': covariances,
        'weights': weights,
        'vacuum_reim': vacuum_reim,
        'n_clusters': len(weights),
    }


def _perturb_gmm_config(
    gmm_params: dict,
    rng: np.random.Generator,
    perturbation_mode: str = 'physical',
    rotation_range: Tuple[float, float] = (0.0, 2 * np.pi),
    phase_jitter_std: float = 0.1,
    amplitude_scale_std: float = 0.03,
    center_jitter_std: float = 0.05,
    weight_dirichlet_conc: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perturb fitted GMM parameters for per-object variety.

    Parameters
    ----------
    gmm_params : dict
        Output of fit_gmm_from_objects().
    rng : np.random.Generator
    perturbation_mode : str
        'physical' (default): small phase jitter around origin + amplitude
        scaling, preserving the physical Re-Im structure.
        'rotation': legacy mode with full rotation around centroid.
    rotation_range : tuple (lo, hi)
        Global rotation angle range (only used when perturbation_mode='rotation').
    phase_jitter_std : float
        Std dev of global phase offset in radians (perturbation_mode='physical').
    amplitude_scale_std : float
        Std dev of log-normal amplitude scaling (perturbation_mode='physical').
    center_jitter_std : float
        Per-center Gaussian jitter standard deviation.
    weight_dirichlet_conc : float
        Dirichlet concentration parameter (higher = closer to fitted weights).

    Returns
    -------
    means : np.ndarray (K, 2)
    covs : np.ndarray (K, 2, 2)
    weights : np.ndarray (K,)
    vacuum_reim : np.ndarray (2,)
    """
    means = gmm_params['means'].copy()
    covs = gmm_params['covariances'].copy()
    weights = gmm_params['weights'].copy()
    K = len(weights)

    if perturbation_mode == 'physical':
        # --- Global phase shift (small rotation around origin) ---
        delta_phase = rng.normal(0.0, phase_jitter_std)
        c, s = np.cos(delta_phase), np.sin(delta_phase)
        R = np.array([[c, -s], [s, c]])
        means = (R @ means.T).T
        for k in range(K):
            covs[k] = R @ covs[k] @ R.T

        # --- Amplitude scaling (radial from origin) ---
        scale = np.exp(rng.normal(0.0, amplitude_scale_std))
        means = means * scale
        covs = covs * (scale ** 2)

    elif perturbation_mode == 'rotation':
        # Legacy: full rotation around weighted centroid
        alpha = rng.uniform(rotation_range[0], rotation_range[1])
        c, s = np.cos(alpha), np.sin(alpha)
        R = np.array([[c, -s], [s, c]])
        centroid = np.average(means, axis=0, weights=weights)
        means_centered = means - centroid
        means = (R @ means_centered.T).T + centroid
        for k in range(K):
            covs[k] = R @ covs[k] @ R.T
    else:
        raise ValueError(f"Unknown perturbation_mode: {perturbation_mode!r}")

    # --- Per-center jitter ---
    means += rng.normal(0.0, center_jitter_std, size=means.shape)

    # --- Weight resampling via Dirichlet ---
    alpha_dir = weight_dirichlet_conc * weights + 1e-6
    weights = rng.dirichlet(alpha_dir)

    # Recompute vacuum as the cluster closest to (1, 0) after perturbation
    dists = np.linalg.norm(means - np.array([1.0, 0.0]), axis=1)
    vacuum_reim = means[np.argmin(dists)].copy()

    return means, covs, weights, vacuum_reim


def quantize_reim_to_clusters(
    real_map: np.ndarray,
    imag_map: np.ndarray,
    cluster_re: np.ndarray,
    cluster_im: np.ndarray,
    mode: str = 'hard',
    temperature: float = 0.1,
    perturbation_std: float = 0.0,
    cluster_covariances: np.ndarray = None,
    cluster_weights: np.ndarray = None,
    amp_std_scale: float = 1.0,
    phase_std_scale: float = 1.0,
    texture_scale: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Snap (Re, Im) fields to the nearest cluster centers.

    Parameters
    ----------
    real_map, imag_map : np.ndarray (H, W)
        Continuous-valued Re and Im fields.
    cluster_re, cluster_im : np.ndarray (K,)
        Cluster center coordinates.
    mode : str
        'hard' — assign to nearest center.
        'soft' — softmax-weighted blend with temperature.
    temperature : float
        Softmax temperature for 'soft' mode (lower = sharper).
    perturbation_std : float
        Post-quantization isotropic Gaussian noise std.
        Used as fallback when cluster_covariances is None.
    cluster_covariances : np.ndarray (K, 2, 2), optional
        Per-cluster covariance matrices from GMM fit. When provided,
        perturbation is sampled in polar (amplitude, phase) space as
        independent Gaussians, then mapped to (Re, Im) via the exact
        polar → Cartesian transform. The radial and tangential
        projections of the covariance set the amplitude and phase
        standard deviations respectively.
    cluster_weights : np.ndarray (K,), optional
        GMM mixing weights (should sum to 1). When provided, adds
        log(w_k) bias to the Euclidean assignment score, shifting
        Voronoi boundaries so that higher-weighted clusters capture
        proportionally more area. When None, pure Euclidean (Voronoi)
        assignment is used.
    amp_std_scale : float
        Scale factor for the amplitude (radial) standard deviation.
    phase_std_scale : float
        Scale factor for the phase (tangential) standard deviation.
        The tangential Cartesian variance is divided by the cluster
        amplitude to convert to angular (radian) units.
    texture_scale : float
        Spatial correlation length for perturbation in pixels.
        When > 0, delta_A and delta_phi are generated as spatially
        smooth 2D random fields (Gaussian-blurred white noise) before
        the polar-to-Cartesian transform, producing coherent texture
        islands of ~texture_scale pixels. When 0 (default), per-pixel
        IID sampling is used (original behavior).

    Returns
    -------
    real_q, imag_q : np.ndarray (H, W)
    """
    H, W = real_map.shape
    K = len(cluster_re)

    # Differences: (H, W, K)
    re_diff = real_map[:, :, None] - cluster_re[None, None, :]  # (H, W, K)
    im_diff = imag_map[:, :, None] - cluster_im[None, None, :]
    dists_sq = re_diff ** 2 + im_diff ** 2  # (H, W, K)

    # Assignment scores (higher = better).
    # Use Euclidean distance as the base metric. The full GMM log-posterior
    # (Mahalanobis + log-det) is NOT suitable here: the input is uniform
    # noise spanning [-1.2, 1.2]^2, so the quadratic Mahalanobis term
    # (scaling as 1/sigma^2) overwhelms log-det and log-weight corrections,
    # causing the loosest cluster to capture nearly all pixels.
    # Instead, Euclidean distance preserves Voronoi geometry and log(w_k)
    # shifts the decision boundaries to approximate target area fractions.
    scores = -dists_sq
    if cluster_weights is not None:
        safe_weights = np.clip(cluster_weights, 1e-30, None)
        scores = scores + np.log(safe_weights)[None, None, :]

    if mode == 'hard':
        nearest = np.argmax(scores, axis=2)  # (H, W)
        real_q = cluster_re[nearest]
        imag_q = cluster_im[nearest]
    elif mode == 'soft':
        # Softmax over scores: w_k = exp(score_k / tau) / sum(...)
        logits = scores / (temperature + 1e-12)
        logits -= logits.max(axis=2, keepdims=True)  # numerical stability
        exp_logits = np.exp(logits)
        w = exp_logits / (exp_logits.sum(axis=2, keepdims=True) + 1e-12)
        real_q = np.sum(w * cluster_re[None, None, :], axis=2)
        imag_q = np.sum(w * cluster_im[None, None, :], axis=2)
    else:
        raise ValueError(f"Unknown quantization mode '{mode}'. Use 'hard' or 'soft'.")

    # Post-quantization perturbation
    if cluster_covariances is not None:
        # Polar-space perturbation: sample independent Gaussians in
        # (amplitude, phase) space, then apply exact polar → Cartesian
        # transform.  Equivalent to the previous Cartesian anisotropic
        # method in the linear regime, but geometrically exact (phase
        # perturbations trace arcs, not tangent lines).
        rng = np.random.default_rng()
        nearest = np.argmax(scores, axis=2)  # (H, W)
        nearest_flat = nearest.ravel()

        eps = 1e-8
        for k in range(K):
            A_k = np.sqrt(cluster_re[k] ** 2 + cluster_im[k] ** 2)
            mask_k = (nearest_flat == k)
            n_pixels = mask_k.sum()
            if n_pixels == 0:
                continue

            if A_k < eps:
                # Degenerate at origin — isotropic from covariance trace
                iso_var = 0.5 * np.trace(cluster_covariances[k])
                noise_re = rng.normal(0, np.sqrt(iso_var), size=n_pixels)
                noise_im = rng.normal(0, np.sqrt(iso_var), size=n_pixels)
            else:
                phi_k = np.arctan2(cluster_im[k], cluster_re[k])
                r_hat = np.array([np.cos(phi_k), np.sin(phi_k)])
                t_hat = np.array([-np.sin(phi_k), np.cos(phi_k)])

                Sigma_k = cluster_covariances[k]
                # Amplitude std in amplitude units
                sigma_A = np.sqrt(r_hat @ Sigma_k @ r_hat) * amp_std_scale
                # Phase std in radians (tangential Cartesian variance / A_k)
                sigma_phi = np.sqrt(t_hat @ Sigma_k @ t_hat) / A_k * phase_std_scale

                if texture_scale > 0:
                    # Spatially coherent perturbation: blur in polar space
                    # before the nonlinear polar→Cartesian transform.
                    raw_A = rng.standard_normal((H, W)).astype(np.float32)
                    raw_phi = rng.standard_normal((H, W)).astype(np.float32)
                    smooth_A = gaussian_filter(raw_A, sigma=texture_scale)
                    smooth_phi = gaussian_filter(raw_phi, sigma=texture_scale)
                    # Normalize to unit variance, then scale
                    std_A = smooth_A.std()
                    std_phi = smooth_phi.std()
                    if std_A > 0:
                        smooth_A *= sigma_A / std_A
                    if std_phi > 0:
                        smooth_phi *= sigma_phi / std_phi
                    delta_A = smooth_A.ravel()[mask_k]
                    delta_phi = smooth_phi.ravel()[mask_k]
                else:
                    # Per-pixel IID sampling (original behavior)
                    delta_A = rng.normal(0, sigma_A, size=n_pixels)
                    delta_phi = rng.normal(0, sigma_phi, size=n_pixels)

                # Exact polar → Cartesian transform
                A_new = A_k + delta_A
                phi_new = phi_k + delta_phi
                noise_re = A_new * np.cos(phi_new) - cluster_re[k]
                noise_im = A_new * np.sin(phi_new) - cluster_im[k]

            real_q.ravel()[mask_k] += noise_re
            imag_q.ravel()[mask_k] += noise_im

    elif perturbation_std > 0:
        # Isotropic fallback
        real_q += np.random.default_rng().normal(0, perturbation_std, (H, W))
        imag_q += np.random.default_rng().normal(0, perturbation_std, (H, W))

    return real_q.astype(np.float32), imag_q.astype(np.float32)


def _draw_random_leaf(
    rng: np.random.Generator,
    shapes: list,
    height: int,
    width: int,
    dim_min_px: float,
    dim_max_px: float,
    beta: float,
    coef: float,
) -> Tuple[np.ndarray, float]:
    """
    Draw a single random leaf shape and return its anti-aliased coverage mask.

    Shared helper extracted from generate_dead_leaves_correlated() to avoid
    code duplication across dead leaves variants.

    Parameters
    ----------
    rng : np.random.Generator
    shapes : list of str
        Shape types to choose from (e.g. ['circle', 'oriented_square', 'rectangle', 'triangle']).
    height, width : int
        Canvas dimensions.
    dim_min_px, dim_max_px : float
        Min/max dimension for power-law sampling.
    beta, coef : float
        Pre-computed power-law parameters:
        beta = 1.0 - dimension_power_law_exponent
        coef = (dim_max_px / dim_min_px)^beta - 1.0

    Returns
    -------
    coverage : np.ndarray (height, width), float32 in [0, 1]
        Anti-aliased coverage mask.
    max_extent : float
        Maximum extent of the shape from center (for coverage tracking).
    """
    cv2 = _import_cv2()
    def sample_dim():
        return dim_min_px * np.power(1.0 + rng.uniform() * coef, 1.0 / beta)

    shape = rng.choice(shapes)

    if shape == 'circle':
        radius_px = sample_dim()
        max_extent = radius_px
    elif shape == 'oriented_square':
        side = sample_dim()
        max_extent = side * np.sqrt(2) / 2
    elif shape == 'rectangle':
        rect_w, rect_h = sample_dim(), sample_dim()
        max_extent = np.sqrt(rect_w ** 2 + rect_h ** 2) / 2
    else:  # triangle
        radius_px = sample_dim()
        max_extent = radius_px

    center_x = rng.uniform(-max_extent, width + max_extent)
    center_y = rng.uniform(-max_extent, height + max_extent)

    temp_mask = np.zeros((height, width), dtype=np.uint8)

    if shape == 'circle':
        cv2.circle(temp_mask, (int(center_x), int(center_y)),
                   int(radius_px), 255, -1, cv2.LINE_AA)
    else:
        if shape == 'oriented_square':
            pts = np.array([[-side / 2, -side / 2], [side / 2, -side / 2],
                            [side / 2, side / 2], [-side / 2, side / 2]])
        elif shape == 'rectangle':
            pts = np.array([[rect_w / 2, rect_h / 2], [rect_w / 2, -rect_h / 2],
                            [-rect_w / 2, -rect_h / 2], [-rect_w / 2, rect_h / 2]])
        else:  # triangle
            angles = sorted(rng.uniform(0, 2 * np.pi, 3))
            pts = np.array([[radius_px * np.cos(a), radius_px * np.sin(a)]
                            for a in angles])

        if shape in ['oriented_square', 'rectangle']:
            theta = rng.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]])
            pts = (R @ pts.T).T

        pts_abs = (np.array([center_x, center_y]) + pts).astype(np.int32)
        cv2.fillPoly(temp_mask, [pts_abs], 255, cv2.LINE_AA)

    coverage = temp_mask.astype(np.float32) / 255.0
    return coverage, max_extent


# --- Method 1: Bivariate Gaussian Dead Leaves in Re-Im Space ---

def generate_dead_leaves_reim(
    height: int,
    width: int,
    dim_min_px: float = 2.0,
    dim_max_px: float = 30.0,
    dimension_power_law_exponent: float = 2.0,
    re_mean: float = 0.5,
    im_mean: float = 0.0,
    re_std: float = 0.15,
    im_std: float = 0.20,
    correlation: float = 0.0,
    vacuum_re: float = 0.8,
    vacuum_im: float = 0.0,
    max_iterations: int = 100000,
    coverage_threshold: float = 0.99,
    blur_sigma: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Bivariate Gaussian dead leaves in Re-Im space.

    Canvas is initialized to (vacuum_re, vacuum_im) instead of (amp=1, phase=0).
    Each leaf samples (leaf_re, leaf_im) from a bivariate Gaussian with the
    specified mean, standard deviations, and correlation.

    Parameters
    ----------
    height, width : int
        Canvas dimensions.
    dim_min_px, dim_max_px : float
        Min/max shape dimension for power-law sampling.
    dimension_power_law_exponent : float
        Power-law exponent (higher = more small shapes).
    re_mean, im_mean : float
        Mean of bivariate Gaussian for leaf (Re, Im) values.
    re_std, im_std : float
        Standard deviations of the bivariate Gaussian.
    correlation : float
        Pearson correlation between Re and Im components, in [-1, 1].
    vacuum_re, vacuum_im : float
        Canvas initialization values (vacuum/substrate).
    max_iterations : int
        Maximum number of leaves.
    coverage_threshold : float
        Stop when this fraction of pixels are covered.
    blur_sigma : float
        Post-processing Gaussian blur sigma.
    seed : int or None
        Random seed.

    Returns
    -------
    real_map : np.ndarray (height, width)
    imag_map : np.ndarray (height, width)
    num_leaves : int
    """
    rng = np.random.default_rng(seed)

    cv2 = _import_cv2()
    # Initialize canvas to vacuum
    real_map = np.full((height, width), vacuum_re, dtype=np.float32)
    imag_map = np.full((height, width), vacuum_im, dtype=np.float32)
    cumulative_coverage = np.zeros((height, width), dtype=np.float32)

    # Build 2x2 covariance matrix from stds and correlation
    cov_matrix = np.array([
        [re_std ** 2, correlation * re_std * im_std],
        [correlation * re_std * im_std, im_std ** 2]
    ])

    # Power-law parameters
    beta = 1.0 - dimension_power_law_exponent
    coef = np.power(dim_max_px / dim_min_px, beta) - 1.0
    shapes = ['circle', 'oriented_square', 'rectangle', 'triangle']

    num_leaves = 0
    for iteration in range(max_iterations):
        # Draw leaf shape
        coverage, _ = _draw_random_leaf(
            rng, shapes, height, width, dim_min_px, dim_max_px, beta, coef
        )

        # Sample (re, im) from bivariate Gaussian
        leaf_re, leaf_im = rng.multivariate_normal(
            [re_mean, im_mean], cov_matrix
        )

        # Alpha blend on both channels
        real_map = real_map * (1.0 - coverage) + leaf_re * coverage
        imag_map = imag_map * (1.0 - coverage) + leaf_im * coverage

        # Coverage tracking
        cumulative_coverage = np.maximum(cumulative_coverage, coverage)
        num_leaves += 1

        if num_leaves % 500 == 0:
            cov_frac = np.count_nonzero(
                cumulative_coverage >= coverage_threshold
            ) / cumulative_coverage.size
            if cov_frac >= coverage_threshold:
                break

    # Post-processing: blur and clip to decoder output range [-1.2, 1.2]
    if blur_sigma > 0:
        real_map = cv2.GaussianBlur(real_map, (0, 0), blur_sigma)
        imag_map = cv2.GaussianBlur(imag_map, (0, 0), blur_sigma)

    real_map = np.clip(real_map, -1.2, 1.2)
    imag_map = np.clip(imag_map, -1.2, 1.2)

    return real_map, imag_map, num_leaves


# --- Method 2: Histogram-Based Re-Im Dead Leaves ---

def generate_dead_leaves_reim_histogram(
    height: int,
    width: int,
    dim_min_px: float = 2.0,
    dim_max_px: float = 30.0,
    dimension_power_law_exponent: float = 2.0,
    material_hist: np.ndarray = None,
    re_range: Tuple[float, float] = (-1.2, 1.2),
    im_range: Tuple[float, float] = (-1.2, 1.2),
    vacuum_re: float = 0.8,
    vacuum_im: float = 0.0,
    max_iterations: int = 100000,
    coverage_threshold: float = 0.99,
    blur_sigma: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Histogram-based dead leaves in Re-Im space.

    Uses an empirical 2D histogram from create_density_histogram_reim() to
    sample leaf (Re, Im) values. Structurally identical to
    generate_dead_leaves_correlated() but operates in (Re, Im) instead of
    (Amp, Phase).

    Parameters
    ----------
    height, width : int
        Canvas dimensions.
    dim_min_px, dim_max_px : float
        Min/max shape dimension for power-law sampling.
    dimension_power_law_exponent : float
        Power-law exponent.
    material_hist : np.ndarray
        2D histogram PDF in (Re, Im) space from create_density_histogram_reim().
    re_range, im_range : tuple
        Histogram axis ranges (must match those used to build material_hist).
    vacuum_re, vacuum_im : float
        Canvas initialization values.
    max_iterations : int
        Maximum number of leaves.
    coverage_threshold : float
        Stop when this fraction of pixels are covered.
    blur_sigma : float
        Post-processing Gaussian blur sigma.
    seed : int or None
        Random seed.

    Returns
    -------
    real_map : np.ndarray (height, width)
    imag_map : np.ndarray (height, width)
    num_leaves : int
    """
    cv2 = _import_cv2()
    if material_hist is None:
        raise ValueError("material_hist is required for histogram-based generation")

    rng = np.random.default_rng(seed)

    # Initialize canvas to vacuum
    real_map = np.full((height, width), vacuum_re, dtype=np.float32)
    imag_map = np.full((height, width), vacuum_im, dtype=np.float32)
    cumulative_coverage = np.zeros((height, width), dtype=np.float32)

    # Prepare histogram sampler
    hist_flat = material_hist.flatten()
    hist_indices = np.arange(hist_flat.size)
    re_bins = np.linspace(re_range[0], re_range[1], material_hist.shape[0])
    im_bins = np.linspace(im_range[0], im_range[1], material_hist.shape[1])

    # Power-law parameters
    beta = 1.0 - dimension_power_law_exponent
    coef = np.power(dim_max_px / dim_min_px, beta) - 1.0
    shapes = ['circle', 'oriented_square', 'rectangle', 'triangle']

    num_leaves = 0
    for iteration in range(max_iterations):
        # Draw leaf shape
        coverage, _ = _draw_random_leaf(
            rng, shapes, height, width, dim_min_px, dim_max_px, beta, coef
        )

        # Sample (re, im) from histogram
        idx = rng.choice(hist_indices, p=hist_flat)
        re_idx, im_idx = np.unravel_index(idx, material_hist.shape)
        leaf_re = re_bins[re_idx]
        leaf_im = im_bins[im_idx]

        # Alpha blend
        real_map = real_map * (1.0 - coverage) + leaf_re * coverage
        imag_map = imag_map * (1.0 - coverage) + leaf_im * coverage

        # Coverage tracking
        cumulative_coverage = np.maximum(cumulative_coverage, coverage)
        num_leaves += 1

        if num_leaves % 500 == 0:
            cov_frac = np.count_nonzero(
                cumulative_coverage >= coverage_threshold
            ) / cumulative_coverage.size
            if cov_frac >= coverage_threshold:
                break

    # Post-processing
    if blur_sigma > 0:
        real_map = cv2.GaussianBlur(real_map, (0, 0), blur_sigma)
        imag_map = cv2.GaussianBlur(imag_map, (0, 0), blur_sigma)

    real_map = np.clip(real_map, -1.2, 1.2)
    imag_map = np.clip(imag_map, -1.2, 1.2)

    return real_map, imag_map, num_leaves


# --- Method 2b: GMM-Based Dead Leaves in Re-Im Space ---

def generate_dead_leaves_reim_gmm(
    height: int,
    width: int,
    means: np.ndarray,
    covariances: np.ndarray,
    weights: np.ndarray,
    vacuum_reim: np.ndarray,
    dim_min_px: float = 2.0,
    dim_max_px: float = 20.0,
    dimension_power_law_exponent: float = 2.0,
    max_iterations: int = 100000,
    coverage_threshold: float = 0.99,
    blur_sigma: float = 0.5,
    clip_range: Tuple[float, float] = (-1.2, 1.2),
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    GMM-based dead leaves in Re-Im space.

    Each leaf samples a cluster index k ~ Categorical(weights), then draws
    (leaf_re, leaf_im) from the k-th component's bivariate Gaussian.
    This produces discrete material clusters in the Re-Im plane.

    Parameters
    ----------
    height, width : int
        Canvas dimensions.
    means : np.ndarray (K, 2)
        GMM cluster centers in (Re, Im).
    covariances : np.ndarray (K, 2, 2)
        Per-cluster covariance matrices.
    weights : np.ndarray (K,)
        Cluster mixing weights (must sum to 1).
    vacuum_reim : np.ndarray (2,)
        Canvas initialization (Re, Im) for vacuum/substrate.
    dim_min_px, dim_max_px : float
        Min/max shape dimension for power-law sampling.
    dimension_power_law_exponent : float
        Power-law exponent (higher = more small shapes).
    max_iterations : int
        Maximum number of leaves.
    coverage_threshold : float
        Stop when this fraction of pixels are covered.
    blur_sigma : float
        Post-processing Gaussian blur sigma.
    seed : int or None
        Random seed.

    Returns
    -------
    real_map : np.ndarray (height, width)
    imag_map : np.ndarray (height, width)
    num_leaves : int
    """
    cv2 = _import_cv2()
    rng = np.random.default_rng(seed)

    # Initialize canvas to vacuum
    real_map = np.full((height, width), vacuum_reim[0], dtype=np.float32)
    imag_map = np.full((height, width), vacuum_reim[1], dtype=np.float32)
    cumulative_coverage = np.zeros((height, width), dtype=np.float32)

    K = len(weights)
    # Normalize weights to ensure valid probability vector
    w = weights / (weights.sum() + 1e-12)

    # Power-law parameters
    beta = 1.0 - dimension_power_law_exponent
    coef = np.power(dim_max_px / dim_min_px, beta) - 1.0
    shapes = ['circle', 'oriented_square', 'rectangle', 'triangle']

    num_leaves = 0
    for iteration in range(max_iterations):
        # Draw leaf shape
        coverage, _ = _draw_random_leaf(
            rng, shapes, height, width, dim_min_px, dim_max_px, beta, coef
        )

        # Sample cluster, then draw (re, im) from that cluster's Gaussian
        k = rng.choice(K, p=w)
        leaf_re, leaf_im = rng.multivariate_normal(means[k], covariances[k])

        # Alpha blend on both channels
        real_map = real_map * (1.0 - coverage) + leaf_re * coverage
        imag_map = imag_map * (1.0 - coverage) + leaf_im * coverage

        # Coverage tracking
        cumulative_coverage = np.maximum(cumulative_coverage, coverage)
        num_leaves += 1

        if num_leaves % 500 == 0:
            cov_frac = np.count_nonzero(
                cumulative_coverage >= coverage_threshold
            ) / cumulative_coverage.size
            if cov_frac >= coverage_threshold:
                break

    # Post-processing: blur and clip
    if blur_sigma > 0:
        real_map = cv2.GaussianBlur(real_map, (0, 0), blur_sigma)
        imag_map = cv2.GaussianBlur(imag_map, (0, 0), blur_sigma)

    real_map = np.clip(real_map, clip_range[0], clip_range[1])
    imag_map = np.clip(imag_map, clip_range[0], clip_range[1])

    return real_map, imag_map, num_leaves


# --- Method 3: Correlated Perlin Noise in Re-Im Space ---

def generate_perlin_object_reim(
    batch_size: int,
    N: int,
    M: int,
    re_mean: float = 0.5,
    im_mean: float = 0.0,
    re_std: float = 0.15,
    im_std: float = 0.20,
    correlation: float = 0.0,
    scale_range: Tuple[float, float] = (20.0, 100.0),
    octaves_range: Tuple[int, int] = (4, 8),
    persistence_range: Tuple[float, float] = (0.4, 0.6),
    lacunarity_range: Tuple[float, float] = (1.8, 2.2),
    device: torch.device = torch.device('cpu'),
    seed_offset: int = 0,
) -> torch.Tensor:
    """
    Generate batch of complex objects using correlated Perlin noise in Re-Im space.

    Two independent Perlin noise fields f1, f2 are generated, then correlated:
        g1 = f1
        g2 = rho * f1 + sqrt(1 - rho^2) * f2

    The fields are scaled/shifted to target Re-Im statistics:
        re = re_mean + re_std * normalize(g1)
        im = im_mean + im_std * normalize(g2)

    Parameters
    ----------
    batch_size : int
        Number of objects.
    N, M : int
        Height and width.
    re_mean, im_mean : float
        Target mean values.
    re_std, im_std : float
        Target standard deviations.
    correlation : float
        Target Pearson correlation between Re and Im, in [-1, 1].
    scale_range : tuple
        Range for Perlin noise scale parameter.
    octaves_range : tuple
        Range for number of noise octaves.
    persistence_range : tuple
        Range for amplitude falloff per octave.
    lacunarity_range : tuple
        Range for frequency increase per octave.
    device : torch.device
        Target device for output.
    seed_offset : int
        Offset for reproducibility.

    Returns
    -------
    torch.Tensor : complex tensor of shape (batch_size, N, M)
    """
    import noise as pnoise_module

    all_re_maps = []
    all_im_maps = []

    rho = np.clip(correlation, -0.999, 0.999)
    sqrt_factor = np.sqrt(1.0 - rho ** 2)

    print(f"Starting Re-Im Perlin generation for {batch_size} items on CPU...")
    start_time = time.time()

    for b in range(batch_size):
        scale = np.random.uniform(*scale_range)
        octave = np.random.randint(octaves_range[0], octaves_range[1] + 1)
        persistence = np.random.uniform(*persistence_range)
        lacunarity = np.random.uniform(*lacunarity_range)
        base_seed_1 = b + seed_offset
        base_seed_2 = b + seed_offset + 10000  # different seed for second field

        # Generate two independent Perlin fields
        f1 = np.zeros((N, M), dtype=np.float32)
        f2 = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            for j in range(M):
                f1[i, j] = pnoise_module.pnoise2(
                    i / scale, j / scale, octaves=octave,
                    persistence=persistence, lacunarity=lacunarity,
                    base=base_seed_1
                )
                f2[i, j] = pnoise_module.pnoise2(
                    i / scale, j / scale, octaves=octave,
                    persistence=persistence, lacunarity=lacunarity,
                    base=base_seed_2
                )

        # Create correlated pair
        g1 = f1
        g2 = rho * f1 + sqrt_factor * f2

        # Normalize each to [0, 1] then scale/shift
        g1_norm = _normalize_np(g1)
        g2_norm = _normalize_np(g2)

        # Map to target statistics: center at 0.5 then shift
        re_map = re_mean + re_std * (g1_norm - 0.5) / 0.2887  # 0.2887 ≈ std of U[0,1]
        im_map = im_mean + im_std * (g2_norm - 0.5) / 0.2887

        # Clip to decoder range
        re_map = np.clip(re_map, -1.2, 1.2)
        im_map = np.clip(im_map, -1.2, 1.2)

        all_re_maps.append(re_map)
        all_im_maps.append(im_map)

        if (b + 1) % max(1, batch_size // 10) == 0:
            print(f"  Generated {b + 1}/{batch_size}...")

    print(f"Re-Im Perlin generation finished in {time.time() - start_time:.3f}s.")

    re_batch = torch.from_numpy(np.stack(all_re_maps, axis=0)).to(device)
    im_batch = torch.from_numpy(np.stack(all_im_maps, axis=0)).to(device)

    return re_batch + 1j * im_batch


def create_perlin_reim(
    img_shape: Tuple[int, int],
    obj_arg: dict,
    stats: dict,
    N: int = _REF_N,
) -> np.ndarray:
    """
    Wrapper for Perlin Re-Im object generation, matching the create_dead_leaves
    interface so it can be used as a partial in simulate_synthetic_objects.

    Reads statistics from obj_arg (e.g. from compute_reim_statistics()) and
    falls back to randomized defaults within experimental ranges if keys are
    missing.

    Parameters
    ----------
    img_shape : tuple (H, W)
        Image dimensions.
    obj_arg : dict
        Generation arguments. Optional keys:
        - 're_mean' or 're_mean_range': float or tuple (default range (-0.6, 0.8))
        - 'im_mean' or 'im_mean_range': float or tuple (default range (-0.6, 0.5))
        - 're_std' or 're_std_range': float or tuple (default range (0.03, 0.25))
        - 'im_std' or 'im_std_range': float or tuple (default range (0.04, 0.35))
        - 'correlation' or 'correlation_range': float or tuple (default range (-0.93, 0.84))
        - 'scale_range': tuple (default (20.0, 100.0))
        - 'octaves_range': tuple (default (4, 8))
        - 'persistence_range': tuple (default (0.4, 0.6))
        - 'lacunarity_range': tuple (default (1.8, 2.2))

        If a scalar is provided (e.g. 're_mean': 0.5), that exact value is used.
        If a range is provided (e.g. 're_mean_range': (-0.6, 0.8)), a random
        value is drawn uniformly from that range per call.

    stats: dict
        object stats from analysis function

    Returns
    -------
    np.ndarray : complex64 object of shape img_shape
    """
    height, width = img_shape

    # Resolve Re-Im statistics: scalar overrides range
    def _resolve(key_scalar, key_range, default_range):
        if key_scalar in obj_arg:
            return obj_arg[key_scalar]
        rng = obj_arg.get(key_range, default_range)
        return np.random.uniform(*rng)

    re_mean = stats.get('re_mean', np.random.uniform(0.6, 1.05))
    im_mean = stats.get('im_mean', np.random.uniform(0, 1.00))
    re_std = stats.get('re_std', np.random.uniform(0.03, 0.1))
    im_std = stats.get('im_std', np.random.uniform(0.03, 0.1))
    correlation = stats.get('correlation', np.random.uniform(-0.93, 0.84))

    # Perlin spatial parameters — scale with N to keep relative feature size
    s = _pixel_scale(N)
    raw_scale_range = obj_arg.get('scale_range', (20.0, 100.0))
    scale_range = (raw_scale_range[0] * s, raw_scale_range[1] * s)
    octaves_range = obj_arg.get('octaves_range', (4, 8))
    persistence_range = obj_arg.get('persistence_range', (0.4, 0.6))
    lacunarity_range = obj_arg.get('lacunarity_range', (1.8, 2.2))

    batch = generate_perlin_object_reim(
        batch_size=1, N=height, M=width,
        re_mean=re_mean,
        im_mean=im_mean,
        re_std=re_std,
        im_std=im_std,
        correlation=correlation,
        scale_range=scale_range,
        octaves_range=octaves_range,
        persistence_range=persistence_range,
        lacunarity_range=lacunarity_range,
    )

    return batch[0].numpy()


# --- Method 4: Constrained-Phase Dead Leaves ---

def generate_dead_leaves_constrained_phase(
    height: int,
    width: int,
    dim_min_px: float = 2.0,
    dim_max_px: float = 30.0,
    dimension_power_law_exponent: float = 2.0,
    amplitude_min: float = 0.6,
    amplitude_max: float = 1.2,
    phase_center: Optional[float] = None,
    phase_width: float = 0.8,
    max_iterations: int = 100000,
    coverage_threshold: float = 0.99,
    blur_sigma: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Dead leaves with constrained phase range.

    Identical to generate_dead_leaves_uniform() but phase is restricted to
    [phase_center - phase_width/2, phase_center + phase_width/2] instead of
    the full [-pi, pi] range. This produces objects with concentrated phase
    distributions similar to experimental data.

    Parameters
    ----------
    height, width : int
        Canvas dimensions.
    dim_min_px, dim_max_px : float
        Min/max shape dimension for power-law sampling.
    dimension_power_law_exponent : float
        Power-law exponent.
    amplitude_min, amplitude_max : float
        Amplitude range.
    phase_center : float or None
        Center of the phase range. If None, randomized from [-pi, pi].
    phase_width : float
        Total width of the phase range in radians (default 0.8).
    max_iterations : int
        Maximum number of leaves.
    coverage_threshold : float
        Stop when this fraction of pixels are covered.
    blur_sigma : float
        Post-processing Gaussian blur sigma.
    seed : int or None
        Random seed.

    Returns
    -------
    amplitude_map : np.ndarray (height, width)
    phase_map : np.ndarray (height, width)
    num_leaves : int
    """
    cv2 = _import_cv2()
    rng = np.random.default_rng(seed)

    if phase_center is None:
        phase_center = rng.uniform(-np.pi, np.pi)

    phase_min = phase_center - phase_width / 2
    phase_max = phase_center + phase_width / 2

    # Initialize canvas
    amplitude_map = np.zeros((height, width), dtype=np.float32)
    phase_map = np.zeros((height, width), dtype=np.float32)
    cumulative_coverage = np.zeros((height, width), dtype=np.float32)

    # Power-law parameters
    beta = 1.0 - dimension_power_law_exponent
    coef = np.power(dim_max_px / dim_min_px, beta) - 1.0
    shapes = ['circle', 'oriented_square', 'rectangle', 'triangle']

    num_leaves = 0
    for iteration in range(max_iterations):
        # Draw leaf shape
        coverage, _ = _draw_random_leaf(
            rng, shapes, height, width, dim_min_px, dim_max_px, beta, coef
        )

        # Sample amplitude (uniform in intensity space)
        amp_sq = rng.uniform(amplitude_min ** 2, amplitude_max ** 2)
        leaf_amp = np.sqrt(amp_sq)

        # Sample phase from constrained range
        leaf_phase = rng.uniform(phase_min, phase_max)

        # Alpha blend
        amplitude_map = amplitude_map * (1.0 - coverage) + leaf_amp * coverage
        phase_map = phase_map * (1.0 - coverage) + leaf_phase * coverage

        # Coverage tracking
        cumulative_coverage = np.maximum(cumulative_coverage, coverage)
        num_leaves += 1

        if num_leaves % 500 == 0:
            cov_frac = np.count_nonzero(
                cumulative_coverage >= coverage_threshold
            ) / cumulative_coverage.size
            if cov_frac >= coverage_threshold:
                break

    # Post-processing
    if blur_sigma > 0:
        amplitude_map = cv2.GaussianBlur(amplitude_map, (0, 0), blur_sigma)
        phase_map = cv2.GaussianBlur(phase_map, (0, 0), blur_sigma)

    return amplitude_map, phase_map, num_leaves


# --- Wrapper: create_dead_leaves_v3 ---

def create_dead_leaves_v3(
    img_shape: Tuple[int, int],
    obj_arg: dict,
    histogram: Optional[np.ndarray] = None,
    N: int = _REF_N,
) -> np.ndarray:
    """
    Top-level dispatcher for Re-Im space dead leaves generation.

    Analogous to create_dead_leaves_v2() but produces objects with realistic
    Re-Im distributions matching experimental data.

    Parameters
    ----------
    img_shape : tuple (H, W)
        Image dimensions.
    obj_arg : dict
        Generation arguments. Required key:
        - 'mode': str, one of 'gaussian', 'histogram', 'constrained_phase'

        Optional keys (inherited from existing dead leaves):
        - 'r_min': float (default 2)
        - 'r_max': float (default 30)
        - 'power_exponent': float (default 2)

        Mode-specific optional keys:
        - For 'gaussian': re_mean, im_mean, re_std, im_std, correlation,
          vacuum_re, vacuum_im (if not provided, randomized within
          experimental ranges)
        - For 'histogram': re_range, im_range, vacuum_reim
        - For 'constrained_phase': phase_center, phase_width,
          amplitude_min, amplitude_max
    histogram : np.ndarray or None
        Required for 'histogram' mode. 2D histogram from
        create_density_histogram_reim().

    Returns
    -------
    np.ndarray : complex64 object of shape img_shape
    """
    mode = obj_arg.get('mode', 'gaussian')
    height, width = img_shape[0], img_shape[1]
    s = _pixel_scale(N)
    min_r = obj_arg.get('r_min', 5) * s
    max_r = obj_arg.get('r_max', 20) * s
    power = obj_arg.get('power_exponent', 2)
    scaled_blur = obj_arg.get('blur_sigma', 0.5) * s

    if mode == 'gaussian':
        # Randomize bivariate Gaussian params within experimental ranges
        # unless explicitly provided
        re_mean = obj_arg.get('re_mean', np.random.uniform(-0.6, 0.8))
        im_mean = obj_arg.get('im_mean', np.random.uniform(-0.6, 0.5))
        re_std = obj_arg.get('re_std', np.random.uniform(0.03, 0.25))
        im_std = obj_arg.get('im_std', np.random.uniform(0.04, 0.35))
        corr = obj_arg.get('correlation', np.random.uniform(-0.93, 0.84))
        vac_re = obj_arg.get('vacuum_re', np.random.uniform(0.6, 1.0))
        vac_im = obj_arg.get('vacuum_im', np.random.uniform(-0.2, 0.2))

        real_map, imag_map, n_leaves = generate_dead_leaves_reim(
            height=height, width=width,
            dim_min_px=min_r, dim_max_px=max_r,
            dimension_power_law_exponent=power,
            re_mean=re_mean, im_mean=im_mean,
            re_std=re_std, im_std=im_std,
            correlation=corr,
            vacuum_re=vac_re, vacuum_im=vac_im,
            blur_sigma=scaled_blur,
        )
        obj = (real_map + 1j * imag_map).astype(np.complex64)

    elif mode == 'histogram':
        if histogram is None:
            raise ValueError("histogram required for 'histogram' mode")

        re_range = obj_arg.get('re_range', (-1.2, 1.2))
        im_range = obj_arg.get('im_range', (-1.2, 1.2))
        vacuum_reim = obj_arg.get('vacuum_reim', (0.8, 0.0))

        real_map, imag_map, n_leaves = generate_dead_leaves_reim_histogram(
            height=height, width=width,
            dim_min_px=min_r, dim_max_px=max_r,
            dimension_power_law_exponent=power,
            material_hist=histogram,
            re_range=re_range, im_range=im_range,
            vacuum_re=vacuum_reim[0], vacuum_im=vacuum_reim[1],
            blur_sigma=scaled_blur,
        )
        obj = (real_map + 1j * imag_map).astype(np.complex64)

    elif mode == 'constrained_phase':
        phase_center = obj_arg.get('phase_center', None)
        phase_width = obj_arg.get('phase_width', 0.8)
        amp_min = obj_arg.get('amplitude_min', 0.6)
        amp_max = obj_arg.get('amplitude_max', 1.2)

        amp_map, phase_map, n_leaves = generate_dead_leaves_constrained_phase(
            height=height, width=width,
            dim_min_px=min_r, dim_max_px=max_r,
            dimension_power_law_exponent=power,
            amplitude_min=amp_min, amplitude_max=amp_max,
            phase_center=phase_center, phase_width=phase_width,
            blur_sigma=scaled_blur,
        )
        obj = (amp_map * np.exp(1j * phase_map)).astype(np.complex64)

    elif mode == 'uniform':
        re_min = obj_arg.get('re_min', -1.0)
        re_max = obj_arg.get('re_max', 1.0)
        im_min = obj_arg.get('im_min', -1.0)
        im_max = obj_arg.get('im_max', 1.0)
        vac_re = obj_arg.get('vacuum_re', np.random.uniform(0.6, 1.0))
        vac_im = obj_arg.get('vacuum_im', np.random.uniform(-0.2, 0.2))

        real_map, imag_map, n_leaves = generate_dead_leaves_uniform_reim(
            height=height, width=width,
            dim_min_px=min_r, dim_max_px=max_r,
            dimension_power_law_exponent=power,
            re_min=re_min, re_max=re_max,
            im_min=im_min, im_max=im_max,
            vacuum_re=vac_re, vacuum_im=vac_im,
        )
        obj = (real_map + 1j * imag_map).astype(np.complex64)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'gaussian', 'histogram', 'constrained_phase', or 'uniform'.")

    print(f"create_dead_leaves_v3 (mode={mode}): generated {img_shape} object")
    return obj


def create_dead_leaves_reim_gmm(
    img_shape: Tuple[int, int],
    obj_arg: dict,
    N: int = _REF_N,
) -> np.ndarray:
    """
    Top-level dispatcher for GMM-based dead leaves generation in Re-Im space.

    Fits a GMM to experimental objects (or accepts pre-fitted parameters),
    perturbs cluster parameters for variety, then generates a dead leaves
    object sampling from the GMM components.

    Parameters
    ----------
    img_shape : tuple (H, W)
        Image dimensions.
    obj_arg : dict
        Required (one of):
        - 'reference_objects': list of complex arrays to fit GMM from.
        - 'gmm_params': dict from fit_gmm_from_objects() (skip fitting).

        Optional:
        - 'n_clusters': int or 'auto' (default 'auto')
        - 'perturbation_mode': 'physical' or 'rotation' (default 'physical')
        - 'phase_jitter_std': float (default 0.1) — phase perturbation in rad
        - 'amplitude_scale_std': float (default 0.03) — log-normal scale std
        - 'rotation_range': tuple (default (0, 2*pi)) — only for mode='rotation'
        - 'center_jitter_std': float (default 0.05)
        - 'weight_dirichlet_conc': float (default 5.0)
        - 'r_min': float (default 1)
        - 'r_max': float (default 20)
        - 'power_exponent': float (default 2)
        - 'max_iterations': int (default 100000)
        - 'coverage_threshold': float (default 0.99)
        - 'blur_sigma': float (default 0.5)
        - 'clip_range': tuple (default (-1.2, 1.2))

    Returns
    -------
    np.ndarray : complex64 object of shape img_shape
    """
    rng = np.random.default_rng()
    height, width = img_shape

    # --- Get or fit GMM parameters ---
    gmm_params = obj_arg.get('gmm_params', None)
    if gmm_params is None:
        ref_objects = obj_arg.get('reference_objects', None)
        if ref_objects is None:
            raise ValueError("Either 'reference_objects' or 'gmm_params' must be provided")
        n_clusters = obj_arg.get('n_clusters', 'auto')
        gmm_params = fit_gmm_from_objects(ref_objects, n_clusters=n_clusters)

    # --- Perturb GMM for per-object variety ---
    perturbation_mode = obj_arg.get('perturbation_mode', 'physical')
    rotation_range = obj_arg.get('rotation_range', (0.0, 2 * np.pi))
    phase_jitter_std = obj_arg.get('phase_jitter_std', 0.1)
    amplitude_scale_std = obj_arg.get('amplitude_scale_std', 0.03)
    center_jitter_std = obj_arg.get('center_jitter_std', 0.05)
    weight_dirichlet_conc = obj_arg.get('weight_dirichlet_conc', 5.0)

    means, covs, weights, vacuum_reim = _perturb_gmm_config(
        gmm_params, rng,
        perturbation_mode=perturbation_mode,
        rotation_range=rotation_range,
        phase_jitter_std=phase_jitter_std,
        amplitude_scale_std=amplitude_scale_std,
        center_jitter_std=center_jitter_std,
        weight_dirichlet_conc=weight_dirichlet_conc,
    )

    # --- Generate dead leaves (scale pixel params with N) ---
    s = _pixel_scale(N)
    min_r = obj_arg.get('r_min', 5) * s
    max_r = obj_arg.get('r_max', 20) * s
    power = obj_arg.get('power_exponent', 2)
    max_iter = obj_arg.get('max_iterations', 100000)
    cov_thresh = obj_arg.get('coverage_threshold', 0.99)
    blur = obj_arg.get('blur_sigma', 0.5) * s
    clip_range = obj_arg.get('clip_range', (-1.2, 1.2))

    real_map, imag_map, n_leaves = generate_dead_leaves_reim_gmm(
        height=height, width=width,
        means=means, covariances=covs, weights=weights,
        vacuum_reim=vacuum_reim,
        dim_min_px=min_r, dim_max_px=max_r,
        dimension_power_law_exponent=power,
        max_iterations=max_iter,
        coverage_threshold=cov_thresh,
        blur_sigma=blur,
        clip_range=clip_range,
    )

    obj = (real_map + 1j * imag_map).astype(np.complex64)
    print(f"create_dead_leaves_reim_gmm: generated {img_shape} object, "
          f"K={len(weights)}, {n_leaves} leaves")
    return obj


def generate_dead_leaves_uniform(
    height: int,
    width: int,
    dim_min_px: float,
    dim_max_px: float,
    dimension_power_law_exponent: float,
    amplitude_min: float,
    amplitude_max: float,
    phase_min_rad: float,
    phase_max_rad: float,
    max_iterations: int = 100000,
    coverage_threshold: float = 0.99,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Generate a synthetic object using the dead leaves model with multiple shape types.
    
    Uses independent power-law sampling for each shape dimension to ensure consistent
    spatial frequency content across all shape types. Implements anti-aliasing using
    OpenCV's LINE_AA for smooth edges.
    
    Parameters
    ----------
    height : int
        Image height in pixels
    width : int
        Image width in pixels
    dim_min_px : float
        Minimum dimension (radius, side, width, height) in pixels
    dim_max_px : float
        Maximum dimension (radius, side, width, height) in pixels
    dimension_power_law_exponent : float
        Power-law exponent for dimension distribution. 
        Higher values favor smaller dimensions.
        Distribution: p(d) ∝ d^(-exponent)
    amplitude_min : float
        Minimum amplitude value
    amplitude_max : float
        Maximum amplitude value
    phase_min_rad : float
        Minimum phase value in radians
    phase_max_rad : float
        Maximum phase value in radians
    max_iterations : int, optional
        Maximum number of leaves to place (default: 100000)
    coverage_threshold : float, optional
        Stop when this fraction of pixels are covered (default: 0.99)
    seed : int or None, optional
        Random seed for reproducibility (default: None)
    
    Returns
    -------
    amplitude : np.ndarray
        Amplitude map of shape (height, width)
    phase : np.ndarray
        Phase map in radians of shape (height, width)
    num_leaves : int
        Number of leaves placed before stopping
        
    Notes
    -----
    - Each shape dimension is sampled independently from the same power-law distribution
    - Circle: radius sampled once
    - Square: side length sampled once
    - Rectangle: width and height sampled independently
    - Triangle: circumradius sampled once
    - All shapes have equal selection probability
    - Amplitude is sampled uniformly in intensity space: A = sqrt(U(A_min², A_max²))
    - Phase is sampled uniformly: φ = U(φ_min, φ_max)
    - Uses alpha blending with anti-aliased coverage masks for smooth edges
    """
    cv2 = _import_cv2()

    # Validate inputs
    if dim_min_px >= dim_max_px:
        raise ValueError('dim_min_px must be less than dim_max_px')
    if amplitude_min >= amplitude_max:
        raise ValueError('amplitude_min must be less than amplitude_max')
    if phase_min_rad >= phase_max_rad:
        raise ValueError('phase_min_rad must be less than phase_max_rad')
    if not 0 < coverage_threshold <= 1.0:
        raise ValueError('coverage_threshold must be in (0, 1]')
    
    # Initialize RNG
    rng = np.random.default_rng(seed)
    
    # Initialize amplitude and phase maps
    amplitude_map = np.zeros((height, width), dtype=np.float32)
    phase_map = np.zeros((height, width), dtype=np.float32)
    
    # Track coverage
    cumulative_coverage = np.zeros((height, width), dtype=np.float32)
    
    # Pre-compute power-law sampling parameters
    beta = 1.0 - dimension_power_law_exponent
    coef = np.power(dim_max_px / dim_min_px, beta) - 1.0
    
    def sample_dimension():
        """Sample a single dimension from power-law distribution."""
        u = rng.uniform()
        return dim_min_px * np.power(1.0 + u * coef, 1.0 / beta)
    
    # Shape selection (equal probabilities)
    shapes = ['circle', 'oriented_square', 'rectangle', 'triangle']
    
    # Main loop: place leaves until coverage threshold is met
    num_leaves = 0
    
    for iteration in range(max_iterations):
        # 1. Select shape type (equal probabilities)
        shape = rng.choice(shapes)
        
        # 2. Sample dimensions independently based on shape type
        if shape == 'circle':
            radius_px = sample_dimension()
            max_extent = radius_px
            
        elif shape == 'oriented_square':
            side = sample_dimension()
            # Maximum extent from center to corner
            max_extent = side * np.sqrt(2) / 2
            
        elif shape == 'rectangle':
            rect_width = sample_dimension()
            rect_height = sample_dimension()
            # Maximum extent from center to corner
            max_extent = np.sqrt(rect_width**2 + rect_height**2) / 2
            
        else:  # triangle
            radius_px = sample_dimension()
            max_extent = radius_px
        
        # 3. Sample position (can extend beyond image bounds)
        center_x = rng.uniform(-max_extent, width + max_extent)
        center_y = rng.uniform(-max_extent, height + max_extent)
        
        # 4. Sample amplitude (uniform in intensity space)
        amplitude_squared = rng.uniform(amplitude_min ** 2, amplitude_max ** 2)
        leaf_amplitude = np.sqrt(amplitude_squared)
        
        # 5. Sample phase (uniform)
        leaf_phase = rng.uniform(phase_min_rad, phase_max_rad)
        
        # 6. Create anti-aliased mask for the shape
        temp_mask = np.zeros((height, width), dtype=np.uint8)
        
        if shape == 'circle':
            cv2.circle(
                temp_mask,
                (int(center_x), int(center_y)),
                radius=int(radius_px),
                color=255,
                thickness=-1,
                lineType=cv2.LINE_AA
            )
            
        else:
            # Generate polygon vertices
            if shape == 'oriented_square':
                corners_rel = np.array([
                    [-side / 2, -side / 2],
                    [+side / 2, -side / 2],
                    [+side / 2, +side / 2],
                    [-side / 2, +side / 2]
                ])
                # Random rotation
                theta = rng.uniform(0, 2 * np.pi)
                c, s = np.cos(theta), np.sin(theta)
                R_mat = np.array([[c, -s], [s, c]])
                corners_rel = (R_mat @ corners_rel.T).T
                
            elif shape == 'rectangle':
                corners_rel = np.array([
                    [+rect_width / 2, +rect_height / 2],
                    [+rect_width / 2, -rect_height / 2],
                    [-rect_width / 2, -rect_height / 2],
                    [-rect_width / 2, +rect_height / 2]
                ])
                # Random rotation
                theta = rng.uniform(0, 2 * np.pi)
                c, s = np.cos(theta), np.sin(theta)
                R_mat = np.array([[c, -s], [s, c]])
                corners_rel = (R_mat @ corners_rel.T).T
                
            elif shape == 'triangle':
                num_verts = 3
                angles = sorted(rng.uniform(0, 2 * np.pi, num_verts))
                corners_rel = np.array([
                    [radius_px * np.cos(ang), radius_px * np.sin(ang)]
                    for ang in angles
                ])
            
            # Convert to absolute coordinates
            corners_abs = (np.array([center_x, center_y]) + corners_rel).astype(np.int32)
            
            # Draw filled polygon with anti-aliasing
            cv2.fillPoly(
                temp_mask,
                [corners_abs],
                color=255,
                lineType=cv2.LINE_AA
            )
        
        # 7. Convert mask to float coverage [0, 1]
        coverage = temp_mask.astype(np.float32) / 255.0
        
        # 8. Alpha blend: new_value = old * (1 - coverage) + new * coverage
        amplitude_map = amplitude_map * (1.0 - coverage) + leaf_amplitude * coverage
        phase_map = phase_map * (1.0 - coverage) + leaf_phase * coverage
        
        # 9. Update cumulative coverage
        cumulative_coverage = np.maximum(cumulative_coverage, coverage)
        
        num_leaves += 1
        
        # 10. Check stopping criterion
        num_covered_pixels = np.count_nonzero(cumulative_coverage >= coverage_threshold)
        coverage_fraction = num_covered_pixels / cumulative_coverage.size
        
        # Log progress periodically
        if num_leaves % 100 == 0 or coverage_fraction >= coverage_threshold:
            print(f'Leaves: {num_leaves}, Coverage: {coverage_fraction * 100:.2f}%')
        
        if coverage_fraction >= coverage_threshold:
            print(f'Reached {coverage_threshold * 100}% coverage at {num_leaves} leaves')
            break
    
    if num_leaves >= max_iterations:
        print(f'Warning: Reached max iterations ({max_iterations}) with {coverage_fraction * 100:.2f}% coverage')
    
    return amplitude_map, phase_map, num_leaves


def generate_dead_leaves_uniform_reim(
    height: int,
    width: int,
    dim_min_px: float,
    dim_max_px: float,
    dimension_power_law_exponent: float,
    re_min: float = -1.0,
    re_max: float = 1.0,
    im_min: float = -1.0,
    im_max: float = 1.0,
    vacuum_re: float = 1.0,
    vacuum_im: float = 0.0,
    max_iterations: int = 100000,
    coverage_threshold: float = 0.99,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Generate a synthetic object using the dead leaves model in real/imaginary space.

    Each leaf's real and imaginary values are sampled independently and uniformly,
    ensuring the full complex disc is supported (no amplitude/phase correlation).

    Parameters
    ----------
    height : int
        Image height in pixels
    width : int
        Image width in pixels
    dim_min_px : float
        Minimum dimension (radius, side, width, height) in pixels
    dim_max_px : float
        Maximum dimension in pixels
    dimension_power_law_exponent : float
        Power-law exponent for dimension distribution.
        Higher values favor smaller dimensions.
        Distribution: p(d) ~ d^(-exponent)
    re_min : float
        Minimum real part value (default -1.0)
    re_max : float
        Maximum real part value (default 1.0)
    im_min : float
        Minimum imaginary part value (default -1.0)
    im_max : float
        Maximum imaginary part value (default 1.0)
    vacuum_re : float
        Real part of the vacuum (background) value (default 1.0)
    vacuum_im : float
        Imaginary part of the vacuum (background) value (default 0.0)
    max_iterations : int, optional
        Maximum number of leaves to place (default: 100000)
    coverage_threshold : float, optional
        Stop when this fraction of pixels are covered (default: 0.99)
    seed : int or None, optional
        Random seed for reproducibility (default: None)

    Returns
    -------
    real_map : np.ndarray
        Real part map of shape (height, width)
    imag_map : np.ndarray
        Imaginary part map of shape (height, width)
    num_leaves : int
        Number of leaves placed before stopping

    Notes
    -----
    - Real and imaginary parts are sampled independently from U[re_min, re_max]
      and U[im_min, im_max] respectively, so the joint distribution is uniform
      over the rectangle and the full complex disc is supported.
    - Shape geometry and power-law dimension sampling are identical to
      generate_dead_leaves_uniform.
    """
    cv2 = _import_cv2()

    # Validate inputs
    if dim_min_px >= dim_max_px:
        raise ValueError('dim_min_px must be less than dim_max_px')
    if re_min >= re_max:
        raise ValueError('re_min must be less than re_max')
    if im_min >= im_max:
        raise ValueError('im_min must be less than im_max')
    if not 0 < coverage_threshold <= 1.0:
        raise ValueError('coverage_threshold must be in (0, 1]')

    # Initialize RNG
    rng = np.random.default_rng(seed)

    # Initialize real and imaginary maps to vacuum
    real_map = np.full((height, width), vacuum_re, dtype=np.float32)
    imag_map = np.full((height, width), vacuum_im, dtype=np.float32)

    # Track coverage
    cumulative_coverage = np.zeros((height, width), dtype=np.float32)

    # Pre-compute power-law sampling parameters
    beta = 1.0 - dimension_power_law_exponent
    coef = np.power(dim_max_px / dim_min_px, beta) - 1.0

    def sample_dimension():
        """Sample a single dimension from power-law distribution."""
        u = rng.uniform()
        return dim_min_px * np.power(1.0 + u * coef, 1.0 / beta)

    # Shape selection (equal probabilities)
    shapes = ['circle', 'oriented_square', 'rectangle', 'triangle']

    # Main loop: place leaves until coverage threshold is met
    num_leaves = 0

    for iteration in range(max_iterations):
        # 1. Select shape type
        shape = rng.choice(shapes)

        # 2. Sample dimensions independently based on shape type
        if shape == 'circle':
            radius_px = sample_dimension()
            max_extent = radius_px

        elif shape == 'oriented_square':
            side = sample_dimension()
            max_extent = side * np.sqrt(2) / 2

        elif shape == 'rectangle':
            rect_width = sample_dimension()
            rect_height = sample_dimension()
            max_extent = np.sqrt(rect_width**2 + rect_height**2) / 2

        else:  # triangle
            radius_px = sample_dimension()
            max_extent = radius_px

        # 3. Sample position (can extend beyond image bounds)
        center_x = rng.uniform(-max_extent, width + max_extent)
        center_y = rng.uniform(-max_extent, height + max_extent)

        # 4. Sample real and imaginary parts independently (uniform)
        leaf_re = rng.uniform(re_min, re_max)
        leaf_im = rng.uniform(im_min, im_max)

        # 5. Create anti-aliased mask for the shape
        temp_mask = np.zeros((height, width), dtype=np.uint8)

        if shape == 'circle':
            cv2.circle(
                temp_mask,
                (int(center_x), int(center_y)),
                radius=int(radius_px),
                color=255,
                thickness=-1,
                lineType=cv2.LINE_AA
            )

        else:
            # Generate polygon vertices
            if shape == 'oriented_square':
                corners_rel = np.array([
                    [-side / 2, -side / 2],
                    [+side / 2, -side / 2],
                    [+side / 2, +side / 2],
                    [-side / 2, +side / 2]
                ])
                theta = rng.uniform(0, 2 * np.pi)
                c, s = np.cos(theta), np.sin(theta)
                R_mat = np.array([[c, -s], [s, c]])
                corners_rel = (R_mat @ corners_rel.T).T

            elif shape == 'rectangle':
                corners_rel = np.array([
                    [+rect_width / 2, +rect_height / 2],
                    [+rect_width / 2, -rect_height / 2],
                    [-rect_width / 2, -rect_height / 2],
                    [-rect_width / 2, +rect_height / 2]
                ])
                theta = rng.uniform(0, 2 * np.pi)
                c, s = np.cos(theta), np.sin(theta)
                R_mat = np.array([[c, -s], [s, c]])
                corners_rel = (R_mat @ corners_rel.T).T

            elif shape == 'triangle':
                num_verts = 3
                angles = sorted(rng.uniform(0, 2 * np.pi, num_verts))
                corners_rel = np.array([
                    [radius_px * np.cos(ang), radius_px * np.sin(ang)]
                    for ang in angles
                ])

            # Convert to absolute coordinates
            corners_abs = (np.array([center_x, center_y]) + corners_rel).astype(np.int32)

            # Draw filled polygon with anti-aliasing
            cv2.fillPoly(
                temp_mask,
                [corners_abs],
                color=255,
                lineType=cv2.LINE_AA
            )

        # 6. Convert mask to float coverage [0, 1]
        coverage = temp_mask.astype(np.float32) / 255.0

        # 7. Alpha blend: new_value = old * (1 - coverage) + new * coverage
        real_map = real_map * (1.0 - coverage) + leaf_re * coverage
        imag_map = imag_map * (1.0 - coverage) + leaf_im * coverage

        # 8. Update cumulative coverage
        cumulative_coverage = np.maximum(cumulative_coverage, coverage)

        num_leaves += 1

        # 9. Check stopping criterion
        num_covered_pixels = np.count_nonzero(cumulative_coverage >= coverage_threshold)
        coverage_fraction = num_covered_pixels / cumulative_coverage.size

        # Log progress periodically
        if num_leaves % 100 == 0 or coverage_fraction >= coverage_threshold:
            print(f'Leaves: {num_leaves}, Coverage: {coverage_fraction * 100:.2f}%')

        if coverage_fraction >= coverage_threshold:
            print(f'Reached {coverage_threshold * 100}% coverage at {num_leaves} leaves')
            break

    if num_leaves >= max_iterations:
        print(f'Warning: Reached max iterations ({max_iterations}) with {coverage_fraction * 100:.2f}% coverage')

    return real_map, imag_map, num_leaves


## New experimental stuff (02/10/2026)

from scipy.spatial import Voronoi
from typing import Tuple, Optional, List

def create_dead_leaves_regions(img_shape, obj_arg, histogram):
    """
    Wrapper for modified dead leaves function from Steve

    Args:
        img_shape (int, int): Image dimensions in (h,w)
        obj_arg (Dict): Passable dictionary with object generation arguments
    """

    print("Generating dead leaves...")
    
    HEIGHT, WIDTH = img_shape[0], img_shape[1]
    MIN_R = obj_arg.get('r_min', 3.0) #Minimum radius to sample
    MAX_R = obj_arg.get('r_max', 30.0)
    N_REGIONS = obj_arg.get('n_regions', 5)
    POWER = obj_arg.get('power_exponent', 1.5)
    FLAT_FRACTION = obj_arg.get('flat_fraction', 0.4)
    N_LEAVES = obj_arg.get('n_leaves', 400)

    amp_2d, phase_2d = generate_hierarchical_constrained_object(
                            height = HEIGHT,
                            width = WIDTH,
                            material_hist = histogram,
                            n_regions = N_REGIONS,
                            dim_min_px = MIN_R,
                            dim_max_px = MAX_R,
                            dimension_power_law_exponent=POWER,
                            leaves_per_region = N_LEAVES,
                            bulk_persistence_fraction=FLAT_FRACTION
                            )
    
    obj = amp_2d * np.exp(1j * phase_2d)

    return obj

def generate_hierarchical_constrained_object(
    height: int,
    width: int,
    material_hist: np.ndarray,
    n_regions: int = 8,
    dim_min_px: float = 5.0,
    dim_max_px: float = 64.0,
    dimension_power_law_exponent: float = 1.5,
    amp_range: Tuple[float, float] = (0.0, 1.0),
    phase_range: Tuple[float, float] = (-np.pi, np.pi),
    leaves_per_region: int = 50,
    bulk_persistence_fraction: float = 0.3, # 30% of continent is "Permanent Bulk"
    blur_sigma: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    
    cv2 = _import_cv2()
    rng = np.random.default_rng(seed)

    # --- 1. Sampling Helpers ---
    beta = 1.0 - dimension_power_law_exponent
    coef = np.power(dim_max_px / dim_min_px, beta) - 1.0
    def sample_dim(): return dim_min_px * np.power(1.0 + rng.uniform() * coef, 1.0 / beta)

    amp_bins = np.linspace(amp_range[0], amp_range[1], material_hist.shape[0])
    phase_bins = np.linspace(phase_range[0], phase_range[1], material_hist.shape[1])
    hist_flat = material_hist.flatten()
    hist_indices = np.arange(hist_flat.size)

    def sample_material():
        idx = rng.choice(hist_indices, p=hist_flat)
        a_idx, p_idx = np.unravel_index(idx, material_hist.shape)
        return amp_bins[a_idx], phase_bins[p_idx]

    # --- 2. VORONOI CONTINENTS ---
    points = np.column_stack([rng.uniform(0, width, n_regions), rng.uniform(0, height, n_regions)])
    y_c, x_c = np.indices((height, width))
    pixel_coords = np.stack([x_c.ravel(), y_c.ravel()], axis=1)
    dists = np.linalg.norm(pixel_coords[:, None, :] - points[None, :, :], axis=2)
    region_labels = np.argmin(dists, axis=1).reshape((height, width))

    amplitude_map = np.ones((height, width), dtype=np.float32)
    phase_map = np.zeros((height, width), dtype=np.float32)

    # --- 3. REGION PROCESSING WITH PERSISTENCE ---
    shapes = ['circle', 'square', 'rectangle']
    
    for r_id in range(n_regions):
        region_indices = (region_labels == r_id)
        region_mask = region_indices.astype(np.float32)
        
        # A. Set Bulk Material Baseline
        bulk_amp, bulk_phi = sample_material()
        amplitude_map[region_indices] = bulk_amp
        phase_map[region_indices] = bulk_phi

        # B. CREATE PERSISTENCE MASK (The "Bulk Anchors")
        # Generate random noise and threshold it to pick 'forbidden' pixels
        # Using a low-pass filter on noise creates "Clumps" of persistence
        noise = rng.uniform(0, 1, (height, width)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (7, 7), 0) # Create structural clumps
        persistence_mask = (noise < bulk_persistence_fraction).astype(np.float32)
        
        # Allowance mask: Where leaves are ALLOWED to go
        # (Inside region AND NOT in persistence zone)
        leaf_allowance_mask = region_mask * (1.0 - persistence_mask)

        # C. ADD LEAVES
        for _ in range(leaves_per_region):
            shape = rng.choice(shapes)
            leaf_amp, leaf_phi = sample_material()
            cx, cy = rng.uniform(0, width), rng.uniform(0, height)
            
            temp_mask = np.zeros((height, width), dtype=np.uint8)
            if shape == 'circle':
                cv2.circle(temp_mask, (int(cx), int(cy)), int(sample_dim()/2), 255, -1, cv2.LINE_AA)
            elif shape == 'square':
                side = sample_dim()
                rect = ((cx, cy), (side, side), rng.uniform(0, 360))
                cv2.fillPoly(temp_mask, [cv2.boxPoints(rect).astype(np.int32)], 255, cv2.LINE_AA)
            elif shape == 'rectangle':
                # Independent dimensions
                rect = ((cx, cy), (sample_dim(), sample_dim()), rng.uniform(0, 360))
                cv2.fillPoly(temp_mask, [cv2.boxPoints(rect).astype(np.int32)], 255, cv2.LINE_AA)

            # APPLY CONSTRAINTS
            # Leaf only exists where (Shape Mask) AND (Allowance Mask)
            leaf_alpha = (temp_mask.astype(np.float32) / 255.0) * leaf_allowance_mask
            
            amplitude_map = amplitude_map * (1.0 - leaf_alpha) + leaf_amp * leaf_alpha
            phase_map = phase_map * (1.0 - leaf_alpha) + leaf_phi * leaf_alpha

    if blur_sigma > 0:
        amplitude_map = cv2.GaussianBlur(amplitude_map, (0, 0), blur_sigma)
        phase_map = cv2.GaussianBlur(phase_map, (0, 0), blur_sigma)

    return amplitude_map, phase_map

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
