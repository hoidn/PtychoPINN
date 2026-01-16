"""Central quality assessment and metrics orchestration for ptychographic reconstruction.

This module serves as PtychoPINN's evaluation control center, coordinating multiple 
quality metrics and providing standardized interfaces for training monitoring, model 
comparison, and research analysis. It operates as a logic/control module that 
orchestrates specialized metric calculations while handling complex data preprocessing
and alignment requirements.

**System Role:**
In the PtychoPINN architecture, this module bridges reconstruction outputs with 
quantitative assessment, enabling real-time training validation and systematic model 
comparisons. It integrates with FRC analysis (ptycho.FRC), image registration 
(ptycho.image), and model comparison workflows (scripts/studies/).

**Key Control Logic:**
- **Dual-Component Processing**: Automatically handles amplitude/phase separation
- **Configurable Phase Alignment**: Routes to 'plane' fitting or 'mean' subtraction
- **Adaptive Normalization**: Applies consistent scaling across all amplitude metrics
- **Debug Pipeline Control**: Coordinates visualization output across metric types
- **Format Standardization**: Ensures consistent input/output contracts

**Primary Functions:**
- `eval_reconstruction()`: Main orchestrator returning comprehensive metric suite
- `ms_ssim()`: Multi-scale SSIM with adaptive downsampling control
- `frc50()`: Fourier Ring Correlation with 0.5 threshold detection
- `fit_and_remove_plane()`: Linear phase trend removal via least squares
- `save_metrics()`: Persistent storage integration with params.cfg

**Integration Workflows:**

Training Pipeline:
```python
from ptycho.evaluation import eval_reconstruction, save_metrics
from ptycho.image.cropping import align_for_evaluation

# Real-time validation during training
aligned_recon = align_for_evaluation(reconstruction, ground_truth)
metrics = eval_reconstruction(aligned_recon, ground_truth, 
                            phase_align_method='plane',
                            debug_save_images=False)
validation_loss = metrics['mae'][0]  # Amplitude MAE for early stopping
```

Model Comparison Study:
```python
# Systematic multi-model evaluation with debug visualization
for model_name, reconstruction in models.items():
    metrics = eval_reconstruction(
        reconstruction, ground_truth,
        label=model_name,
        debug_save_images=True,  # Saves preprocessing images for analysis
        ms_ssim_sigma=1.0        # Consistent preprocessing across models
    )
    save_metrics(reconstruction, ground_truth, label=model_name)
    
    # Structured output: metrics['frc50'] = (amp_frc50, phase_frc50)
    print(f"{model_name}: FRC50 = {metrics['frc50']}")
```

**Data Flow & Dependencies:**
- **Input**: Complex reconstructions (3D/4D), ground truth objects (3D)
- **Output**: Structured metric dictionaries with (amplitude, phase) tuples
- **Dependencies**: params.cfg (legacy paths), FRC modules, skimage.metrics
- **Format Requirements**: Amplitude arrays must be normalized consistently

**Critical Behavior Modes:**
- **phase_align_method='plane'**: Removes linear phase gradients (recommended)
- **phase_align_method='mean'**: Simple mean subtraction (legacy compatibility)
- **debug_save_images=True**: Generates diagnostic images in cwd/debug_images_<label>/
- **Amplitude normalization**: Always applied via mean scaling for fair comparison

**Performance Characteristics:**
- FRC calculation requires square images (auto-crops to smaller dimension)
- MS-SSIM scales adaptively with image resolution
- Debug image generation adds ~2-3x evaluation time
- Caching not implemented (metrics recalculated on each call)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from skimage.metrics import structural_similarity
from pathlib import Path

from ptycho import params
from ptycho import misc

def _save_debug_image(image_data, filename_suffix, output_dir, vmin=None, vmax=None):
    """
    Private helper function to save debug images using matplotlib.
    
    Args:
        image_data: 2D numpy array to save as image
        filename_suffix: String suffix for the filename (e.g., 'pinn_phase_for_ms-ssim')
        output_dir: Directory to save the image (will be created if needed)
        vmin: Minimum value for color scaling (optional)
        vmax: Maximum value for color scaling (optional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{filename_suffix}.png"
    filepath = output_dir / filename
    
    # Detect if this is a phase image (typically in [-π, π] or [0, 1] range)
    is_phase = 'phase' in filename_suffix.lower()
    
    # Choose appropriate colormap
    cmap = 'hsv' if is_phase else 'gray'
    
    # Use matplotlib to save the image with proper scaling
    plt.imsave(filepath, image_data, cmap=cmap, vmin=vmin, vmax=vmax)

def ms_ssim(img1, img2, levels=5, sigma=0.0):
    """
    Multi-Scale Structural Similarity (MS-SSIM) metric.
    
    This function implements MS-SSIM by iteratively downsampling the images
    and calculating SSIM at each level using standard weights.
    
    Args:
        img1: First image (2D numpy array)
        img2: Second image (2D numpy array)  
        levels: Number of scale levels (default: 5)
        sigma: Gaussian smoothing sigma before MS-SSIM calculation (default: 0.0, no smoothing)
        
    Returns:
        MS-SSIM value (float)
    """
    # Apply Gaussian smoothing if sigma > 0
    if sigma > 0:
        from scipy.ndimage import gaussian_filter as gf
        img1 = gf(img1, sigma)
        img2 = gf(img2, sigma)
    
    # Standard MS-SSIM weights from the literature
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    
    # Ensure we don't exceed the weights array
    levels = min(levels, len(weights))
    
    # Calculate data range for SSIM
    data_range = max(img1.max() - img1.min(), img2.max() - img2.min())
    
    ms_ssim_val = 1.0
    
    for level in range(levels):
        # Calculate SSIM at current scale
        if level == levels - 1:
            # Use full SSIM for the final level
            ssim_val = structural_similarity(img1, img2, data_range=data_range)
        else:
            # Use only luminance component for intermediate levels
            ssim_val, ssim_map = structural_similarity(img1, img2, data_range=data_range, full=True)
            # For MS-SSIM, we typically use the mean of the contrast and structure components
            # This is a simplified implementation - the full MS-SSIM separates these components
            
        # Apply the weight for this level
        # Handle negative or NaN SSIM values
        if np.isnan(ssim_val):
            import warnings
            warnings.warn(f"NaN SSIM value encountered in MS-SSIM calculation at level {level}. "
                        f"Returning 0.0 for MS-SSIM.", RuntimeWarning)
            return 0.0  # Return 0 if NaN encountered
        elif ssim_val < 0:
            import warnings
            warnings.warn(f"Negative SSIM value ({ssim_val:.4f}) encountered in MS-SSIM calculation. "
                        f"Clamping to 0.0001 to avoid NaN.", RuntimeWarning)
            ssim_val = 0.0001  # Small positive value to avoid fractional power issues
        
        ms_ssim_val *= (ssim_val ** weights[level])
        
        # Check if the result became NaN after the power operation
        if np.isnan(ms_ssim_val):
            import warnings
            warnings.warn(f"MS-SSIM became NaN after power operation at level {level}. "
                        f"SSIM was {ssim_val}, weight was {weights[level]}. Returning 0.0.", RuntimeWarning)
            return 0.0
        
        # Downsample for next level (except on last iteration)
        if level < levels - 1:
            # Simple downsampling by factor of 2
            img1 = img1[::2, ::2]
            img2 = img2[::2, ::2]
            
            # Ensure images are large enough for next level
            if min(img1.shape) < 7 or min(img2.shape) < 7:
                # Not enough resolution for more levels
                break
    
    # Final check for NaN before returning
    if np.isnan(ms_ssim_val):
        import warnings
        warnings.warn(f"Final MS-SSIM value is NaN. Returning 0.0.", RuntimeWarning)
        return 0.0
    
    return ms_ssim_val

def recon_patches(patches):
    """
    chop channel dimension size to 1, then patch together a single image
    """
    from ptycho import generate_data as data
    return data.reassemble(patches[:, :, :, :1])[0]

def symmetrize(arr):
    return (arr + arr[::-1, ::-1]) / 2

def symmetrize_3d(arr):
    return (arr + arr[:, ::-1, ::-1]) / 2

def cropshow(arr, *args, crop = True, **kwargs):
    if crop:
        arr = arr[16:-16, 16:-16]
    plt.imshow(arr, *args, **kwargs)

from scipy.ndimage import gaussian_filter as gf

def summarize(i, a, b, X_test, Y_I_test, Y_phi_test, probe, channel = 0, **kwargs):
    from . import params as cfg
    plt.rcParams["figure.figsize"] = (10, 10)
    vmin = 0
    vmax = np.absolute(b)[i].max()

    heatmaps = {}  # initialize the dictionary to store the heatmaps
    probe = np.absolute(probe)
    aa, bb = 3, 3
    plt.subplot(aa, bb, 1)
    plt.title('True amp.\n(illuminated)')
    true_amp_illuminated = (Y_I_test[i, :, :, channel])
    cropshow(true_amp_illuminated, cmap = 'jet', **kwargs)
    heatmaps['true_amp_illuminated'] = true_amp_illuminated  # add to the dictionary

    plt.subplot(aa, bb, 2)
    plt.title('Reconstructed amp.\n(illuminated)')
    rec_amp_illuminated = (np.absolute(b))[i] * probe[..., None]
    cropshow(rec_amp_illuminated, cmap = 'jet', **kwargs)
    heatmaps['rec_amp_illuminated'] = rec_amp_illuminated  # add to the dictionary

    plt.subplot(aa, bb, 3)
    plt.title('True phase')
    true_phase = ((Y_phi_test * (probe > .01)[..., None]))[i, :, :, channel]
    cropshow(true_phase, cmap = 'jet', **kwargs)
    plt.colorbar()
    heatmaps['true_phase'] = true_phase  # add to the dictionary

    plt.subplot(aa, bb, 4)
    plt.title('True amp.\n(full)')
    true_amp_full = (Y_I_test[i, :, :, channel] / (probe + 1e-9))
    cropshow(true_amp_full, cmap = 'jet', **kwargs)
    heatmaps['true_amp_full'] = true_amp_full  # add to the dictionary

    plt.subplot(aa, bb, 5)
    plt.title('Reconstructed amp. (full)')
    rec_amp_full = (np.absolute(b))[i]
    cropshow(rec_amp_full, cmap = 'jet', **kwargs)
    heatmaps['rec_amp_full'] = rec_amp_full  # add to the dictionary

    plt.subplot(aa, bb, 6)
    plt.title('Reconstructed phase')
    rec_phase = (np.angle(b) * (probe > .01)[..., None])[i]
    rec_phase[np.isclose(rec_phase,  0)] = np.nan
    cropshow(rec_phase, cmap = 'jet', **kwargs)
    plt.colorbar()
    heatmaps['rec_phase'] = rec_phase  # add to the dictionary
    print('phase min:', np.min((np.angle(b) * (probe > .01)[..., None])),
        'phase max:', np.max((np.angle(b) * (probe > .01)[..., None])))

    plt.subplot(aa, bb, 7)
    plt.title('True diffraction')
    true_diffraction = np.log(cfg.get('intensity_scale') * X_test)[i, :, :, channel]
    plt.imshow(true_diffraction, cmap = 'jet')
    plt.colorbar()
    heatmaps['true_diffraction'] = true_diffraction  # add to the dictionary

    plt.subplot(aa, bb, 8)
    plt.title('Recon diffraction')
    rec_diffraction = np.log(a)[i, :, :, channel]
    plt.imshow(rec_diffraction, cmap = 'jet')
    plt.colorbar()
    heatmaps['rec_diffraction'] = rec_diffraction  # add to the dictionary

    return heatmaps

def plt_metrics(history, loss_type = 'MAE', metric2 = 'padded_obj_loss'):
    hist=history
    epochs=np.asarray(history.epoch)+1

    plt.style.use('seaborn-white')
    matplotlib.rc('font',family='Times New Roman')
    matplotlib.rcParams['font.size'] = 12

    f, axarr = plt.subplots(2, sharex=True, figsize=(12, 8))

    axarr[0].set(ylabel='Loss')
    axarr[0].plot(epochs,hist.history['loss'], 'C3o', label='Diffraction {} Training'.format(loss_type))
    axarr[0].plot(epochs,hist.history['val_loss'], 'C3-', label='Diffraction {} Validation'.format(loss_type))
    axarr[0].grid()
    axarr[0].legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

    axarr[1].set(ylabel='Loss')
    axarr[1].plot(epochs,hist.history[metric2], 'C0o', label='Object {} Training'.format(loss_type))
    axarr[1].plot(epochs,hist.history['val_' + metric2], 'C0-', label='Object {} Validation'.format(loss_type))
    axarr[1].legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
    plt.xlabel('Epochs')
    plt.tight_layout()
    #plt.semilogy()
    axarr[1].grid()

import scipy.fftpack as fftpack
fp = fftpack

def trim(arr2d):
    offset = params.get('offset')
    assert not (offset % 2)
    return arr2d[offset // 2:-offset // 2, offset // 2:-offset // 2]

def mae(target, pred, normalize = True):
    """
    mae for an entire (stitched-together) reconstruction.
    """
    if normalize:
        scale = np.mean(target) / np.mean(pred)
    else:
        scale = 1
    print('mean scale adjustment:', scale)
    return np.mean(np.absolute(target - scale * pred))

def mse(target, pred, normalize = True):
    """
    mae for an entire (stitched-together) reconstruction.
    """
    if normalize:
        scale = np.mean(target) / np.mean(pred)
    else:
        scale = 1
    print('mean scale adjustment:', scale)
    return np.mean((target - scale * pred)**2)

def psnr(target, pred, normalize = True, shift = False):
    """
    for phase inputs, assume that global shift has already been taken care off
    """
    import cv2
    target = np.array(target, dtype=np.float32)
    pred = np.array(pred, dtype=np.float32)
    if normalize:
        scale = np.mean(target) / np.mean(pred)
    else:
        scale = 1
    if shift:
        offset = min(np.min(target), np.min(pred))
        target = target - offset
        pred = pred - offset
    pred = scale * pred
    return cv2.PSNR(target, pred)

def fft2d(aphi):
    F1 = fftpack.fft2((aphi).astype(float))
    F2 = fftpack.fftshift(F1)
    return F2

def highpass2d(aphi, n = 2):
    if n == 1:
        print('subtracting mean', np.mean(aphi))
        return aphi - np.mean(aphi)
    F2 = fft2d(aphi)
    (w, h) = aphi.shape
    half_w, half_h = int(w/2), int(h/2)

    # high pass filter

    F2[half_w-n:half_w+n+1,half_h-n:half_h+n+1] = 0

    im1 = fp.ifft2(fftpack.ifftshift(F2)).real
    return im1

def lowpass2d(aphi, n = 2):
    if n == 1:
        print('subtracting mean', np.mean(aphi))
        return aphi - np.mean(aphi)
    F2 = fft2d(aphi)
    (w, h) = aphi.shape
    half_w, half_h = int(w/2), int(h/2)

    # high pass filter
    mask = np.zeros_like(F2)
    mask[half_w-n:half_w+n+1,half_h-n:half_h+n+1] = 1.
    F2 = F2 * mask

    im1 = fp.ifft2(fftpack.ifftshift(F2)).real
    return im1

def fit_and_remove_plane(phase_img, reference_phase=None):
    """
    Fit and remove a plane from phase image to enable fair phase comparison.
    
    Args:
        phase_img: 2D array of phase values
        reference_phase: Optional reference phase for alignment (default: fit to zero plane)
    
    Returns:
        phase_aligned: Phase image with plane removed
    """
    h, w = phase_img.shape
    
    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Flatten for linear system
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    phase_flat = phase_img.flatten()
    
    # Set up linear system A * coeffs = phase for plane z = a*x + b*y + c
    A = np.column_stack([x_flat, y_flat, np.ones(len(x_flat))])
    
    # Solve for plane coefficients
    coeffs, _, _, _ = np.linalg.lstsq(A, phase_flat, rcond=None)
    
    # Compute fitted plane
    fitted_plane = coeffs[0] * x_coords + coeffs[1] * y_coords + coeffs[2]
    
    # Remove the fitted plane
    phase_aligned = phase_img - fitted_plane
    
    # If reference is provided, align to it instead
    if reference_phase is not None:
        ref_coeffs, _, _, _ = np.linalg.lstsq(A, reference_phase.flatten(), rcond=None)
        ref_plane = ref_coeffs[0] * x_coords + ref_coeffs[1] * y_coords + ref_coeffs[2]
        phase_aligned = phase_img - fitted_plane + ref_plane
    
    return phase_aligned

def frc50(target, pred, frc_sigma = 0, debug_save_images=False, debug_dir=None, label=""):
    if np.isnan(pred).all():
        raise ValueError
    if np.max(target) == np.min(target) == 0:
        return None, np.nan
    
    # Ensure images are square for FRC calculation
    target = np.array(target)
    pred = np.array(pred)
    
    if target.shape[0] != target.shape[1]:
        # Center crop to the smaller dimension to make square
        min_dim = min(target.shape[0], target.shape[1])
        h_start = (target.shape[0] - min_dim) // 2
        w_start = (target.shape[1] - min_dim) // 2
        target = target[h_start:h_start + min_dim, w_start:w_start + min_dim]
        pred = pred[h_start:h_start + min_dim, w_start:w_start + min_dim]
    
    # Save debug images if requested
    if debug_save_images and debug_dir:
        # Calculate consistent color scales for FRC images
        frc_vmin = min(target.min(), pred.min())
        frc_vmax = max(target.max(), pred.max())
        
        _save_debug_image(target, f"{label}_target_for_frc" if label else "target_for_frc", debug_dir, vmin=frc_vmin, vmax=frc_vmax)
        _save_debug_image(pred, f"{label}_pred_for_frc" if label else "pred_for_frc", debug_dir, vmin=frc_vmin, vmax=frc_vmax)
    
    from ptycho.FRC import fourier_ring_corr as frc
    shellcorr = frc.FSC(target, pred)
    if frc_sigma > 0:
        shellcorr = gf(shellcorr, frc_sigma)
    
    # Find where FRC drops below 0.5
    below_half = np.where(shellcorr < .5)[0]
    if len(below_half) > 0:
        frc50_value = below_half[0]
    else:
        # If FRC never drops below 0.5, use the length (indicates excellent reconstruction)
        frc50_value = len(shellcorr)
    
    return shellcorr, frc50_value



def eval_reconstruction(stitched_obj, ground_truth_obj, lowpass_n = 1,
        label = '', phase_align_method='plane', frc_sigma=0, debug_save_images=False, ms_ssim_sigma=1.0):
    """
    Evaluate reconstruction quality against ground truth using multiple metrics.
    
    Args:
        stitched_obj: Reconstructed complex object (3D or 4D array)
        ground_truth_obj: Ground truth complex object (3D array)
        lowpass_n: Legacy parameter (deprecated, kept for compatibility)
        label: Label for debug logging
        phase_align_method: Method for phase alignment ('plane' or 'mean')
                          - 'plane': Fit and remove planes from phase images (default)
                          - 'mean': Subtract mean from phase images
        frc_sigma: Gaussian smoothing sigma for FRC calculation (0 = no smoothing)
        debug_save_images: If True, save debug images for MS-SSIM and FRC preprocessing
        ms_ssim_sigma: Gaussian smoothing sigma for MS-SSIM amplitude calculation (default: 1.0)
    
    Returns:
        dict: Dictionary containing evaluation metrics with keys:
            - 'mae': (amplitude_mae, phase_mae) - Mean Absolute Error
            - 'mse': (amplitude_mse, phase_mse) - Mean Squared Error  
            - 'psnr': (amplitude_psnr, phase_psnr) - Peak Signal-to-Noise Ratio
            - 'ssim': (amplitude_ssim, phase_ssim) - Structural Similarity Index
            - 'ms_ssim': (amplitude_ms_ssim, phase_ms_ssim) - Multi-Scale SSIM
            - 'frc50': (amplitude_frc50, phase_frc50) - FRC at 0.5 threshold
            - 'frc': (amplitude_frc_curve, phase_frc_curve) - Full FRC curves
    """
    # Handle shape consistency: convert 4D reconstruction to 3D before assertions
    assert np.ndim(ground_truth_obj) == 3
    assert int(np.ndim(stitched_obj)) in [3, 4]
    if np.ndim(stitched_obj) == 4:
        stitched_obj = stitched_obj[0]
    
    # Now both arrays should be 3D and have consistent shapes
    assert stitched_obj.shape[0] == ground_truth_obj.shape[0]  # height
    assert stitched_obj.shape[1] == ground_truth_obj.shape[1]  # width
    YY_ground_truth = np.absolute(ground_truth_obj)
    YY_phi_ground_truth = np.angle(ground_truth_obj)

    # Extract raw phase data
    phi_pred_raw = trim(np.squeeze(np.angle(stitched_obj)))
    phi_target_raw = trim(np.squeeze(YY_phi_ground_truth))
    
    # Apply configurable phase alignment
    if phase_align_method == 'plane':
        # Use plane fitting alignment - align both images to remove linear phase trends
        phi_pred = fit_and_remove_plane(phi_pred_raw)
        phi_target = fit_and_remove_plane(phi_target_raw)
    elif phase_align_method == 'mean':
        # Use mean subtraction alignment
        phi_pred = phi_pred_raw - np.mean(phi_pred_raw)
        phi_target = phi_target_raw - np.mean(phi_target_raw)
    else:
        raise ValueError(f"Unknown phase_align_method: {phase_align_method}. Use 'plane' or 'mean'.")
    amp_target = tf.cast(trim(YY_ground_truth), tf.float32)
    amp_pred = trim(np.absolute(stitched_obj))

    # Convert to numpy for consistent processing
    amp_target_np = np.array(amp_target[:, :, 0])
    amp_pred_np = np.array(amp_pred[:, :, 0])
    
    # ===== EXPLICIT AMPLITUDE NORMALIZATION =====
    # Calculate scale factor to normalize predicted amplitude to match ground truth mean
    scale_factor = np.mean(amp_target_np) / np.mean(amp_pred_np)
    amp_pred_normalized = amp_pred_np * scale_factor
    print(f'Amplitude normalization scale factor: {scale_factor:.6f}')
    # ============================================
    
    # Now all metrics use the same normalized amplitude data
    mae_amp = mae(amp_target_np, amp_pred_normalized, normalize=False)
    mse_amp = mse(amp_target_np, amp_pred_normalized, normalize=False)
    psnr_amp = psnr(amp_target_np, amp_pred_normalized, normalize=False, shift=False)
    # Calculate SSIM for amplitude using normalized prediction
    amp_data_range = float(np.max(amp_target_np) - np.min(amp_target_np))
    ssim_amp = structural_similarity(amp_target_np, amp_pred_normalized, 
                                   data_range=amp_data_range)
    
    # Prepare debug directory path
    import os
    cwd = os.getcwd()
    debug_dir = os.path.join(cwd, f"debug_images_{label}" if label else "debug_images") if debug_save_images else None
    
    # Debug logging for FRC inputs
    print(f"DEBUG eval_reconstruction [{label}]: amp_target stats: mean={np.mean(amp_target):.6f}, std={np.std(amp_target):.6f}, shape={amp_target.shape}")
    print(f"DEBUG eval_reconstruction [{label}]: amp_pred stats: mean={np.mean(amp_pred):.6f}, std={np.std(amp_pred):.6f}, shape={amp_pred.shape}")
    print(f"DEBUG eval_reconstruction [{label}]: phi_target stats: mean={np.mean(phi_target):.6f}, std={np.std(phi_target):.6f}, shape={phi_target.shape}")
    print(f"DEBUG eval_reconstruction [{label}]: phi_pred stats: mean={np.mean(phi_pred):.6f}, std={np.std(phi_pred):.6f}, shape={phi_pred.shape}")
    
    frc_amp, frc50_amp = frc50(amp_target_np, amp_pred_normalized, frc_sigma=frc_sigma, 
                               debug_save_images=debug_save_images, 
                               debug_dir=debug_dir,
                               label=f"{label}_amp" if label else "amp")

    mae_phi = mae(phi_target, phi_pred, normalize=False) # PINN
    mse_phi = mse(phi_target, phi_pred, normalize=False) # PINN
    psnr_phi = psnr(phi_target, phi_pred, normalize=False, shift=False)  # No hidden shift - use consistent alignment
    
    # Calculate SSIM for phase (scale from [-π,π] to [0,1] after plane alignment)
    phi_target_scaled = (phi_target + np.pi) / (2 * np.pi)
    phi_pred_scaled = (phi_pred + np.pi) / (2 * np.pi)
    print(f'Phase preprocessing: plane-fitted range [{phi_target.min():.3f}, {phi_target.max():.3f}] -> scaled range [{phi_target_scaled.min():.3f}, {phi_target_scaled.max():.3f}]')
    ssim_phi = structural_similarity(phi_target_scaled, phi_pred_scaled, 
                                   data_range=1.0)
    
    # Calculate MS-SSIM for amplitude and phase using the same preprocessing
    ms_ssim_amp = ms_ssim(amp_target_np, amp_pred_normalized, sigma=ms_ssim_sigma)
    ms_ssim_phi = ms_ssim(phi_target_scaled, phi_pred_scaled)
    
    # Save debug images if requested
    if debug_save_images:
        
        # Calculate consistent color scales for phase images (scaled phase: [0, 1])
        phase_vmin = min(phi_pred_scaled.min(), phi_target_scaled.min())
        phase_vmax = max(phi_pred_scaled.max(), phi_target_scaled.max())
        
        # Calculate consistent color scales for amplitude images (using normalized amplitude)
        amp_vmin = min(amp_pred_normalized.min(), amp_target_np.min())
        amp_vmax = max(amp_pred_normalized.max(), amp_target_np.max())
        
        # Save MS-SSIM preprocessing images with consistent color scaling
        _save_debug_image(phi_pred_scaled, f"{label}_phase_pred_for_ms-ssim" if label else "phase_pred_for_ms-ssim", debug_dir, vmin=phase_vmin, vmax=phase_vmax)
        _save_debug_image(phi_target_scaled, f"{label}_phase_target_for_ms-ssim" if label else "phase_target_for_ms-ssim", debug_dir, vmin=phase_vmin, vmax=phase_vmax)
        _save_debug_image(amp_pred_normalized, f"{label}_amp_pred_for_ms-ssim" if label else "amp_pred_for_ms-ssim", debug_dir, vmin=amp_vmin, vmax=amp_vmax)
        _save_debug_image(amp_target_np, f"{label}_amp_target_for_ms-ssim" if label else "amp_target_for_ms-ssim", debug_dir, vmin=amp_vmin, vmax=amp_vmax)
    
    frc_phi, frc50_phi = frc50(phi_target, phi_pred, frc_sigma=frc_sigma,
                               debug_save_images=debug_save_images,
                               debug_dir=debug_dir, 
                               label=f"{label}_phi" if label else "phi")

    return {'mae': (mae_amp, mae_phi),
        'mse': (mse_amp, mse_phi),
        'psnr': (psnr_amp, psnr_phi),
        'ssim': (ssim_amp, ssim_phi),
        'ms_ssim': (ms_ssim_amp, ms_ssim_phi),
        'frc50': (frc50_amp, frc50_phi),
        'frc': (frc_amp, frc_phi)}


import pandas as pd
import os
import dill
def save_metrics(stitched_obj, YY_ground_truth,  label = ''):
    """
    evaluate reconstruction and save the result to disk.
    """
    out_prefix = misc.get_path_prefix()
    os.makedirs(out_prefix, exist_ok=True)
    metrics = eval_reconstruction(stitched_obj, YY_ground_truth, label = label)
    metrics['label'] = label
    d = {**params.cfg, **metrics}
    with open(out_prefix + '/params.dill', 'wb') as f:
        dill.dump(d, f)
    df = pd.DataFrame({k: d[k] for k in ['mae', 'mse', 'psnr', 'frc50']})
    df.to_csv(out_prefix + '/metrics.csv')
    return {k: metrics[k] for k in ['mae', 'mse', 'psnr', 'frc50', 'frc']}


# Unit Tests for Phase 1 Enhancements
# Uncomment and run these to verify functionality:

# def test_ssim_self_comparison():
#     """Test that SSIM self-comparison returns 1.0"""
#     test_img = np.random.rand(64, 64)
#     ssim_val = structural_similarity(test_img, test_img, data_range=1.0)
#     assert np.isclose(ssim_val, 1.0), f"SSIM self-comparison failed: {ssim_val}"
#     print("✓ SSIM self-comparison test passed")

# def test_plane_fitting():
#     """Test that plane fitting removes known planes"""
#     h, w = 32, 32
#     y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
#     # Create known plane: z = 0.1*x + 0.2*y + 0.5
#     known_plane = 0.1 * x + 0.2 * y + 0.5
#     aligned = fit_and_remove_plane(known_plane)
#     # After removing the plane, should be near zero
#     assert np.allclose(aligned, 0, atol=1e-10), f"Plane fitting failed: max residual = {np.max(np.abs(aligned))}"
#     print("✓ Plane fitting test passed")

# def test_frc_self_comparison():
#     """Test that FRC self-comparison returns all 1s"""
#     test_img = np.random.rand(64, 64)
#     frc_curve, frc50_val = frc50(test_img, test_img, frc_sigma=0)
#     assert np.allclose(frc_curve, 1.0, rtol=1e-10), f"FRC self-comparison failed: min FRC = {np.min(frc_curve)}"
#     print("✓ FRC self-comparison test passed")

# To run tests:
# test_ssim_self_comparison()
# test_plane_fitting() 
# test_frc_self_comparison()
