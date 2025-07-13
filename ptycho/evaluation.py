import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from skimage.metrics import structural_similarity

from ptycho import params
from ptycho import misc

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

def frc50(target, pred, frc_sigma = 0):
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
        label = '', phase_align_method='plane', frc_sigma=0):
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
    
    Returns:
        dict: Dictionary containing evaluation metrics with keys:
            - 'mae': (amplitude_mae, phase_mae) - Mean Absolute Error
            - 'mse': (amplitude_mse, phase_mse) - Mean Squared Error  
            - 'psnr': (amplitude_psnr, phase_psnr) - Peak Signal-to-Noise Ratio
            - 'ssim': (amplitude_ssim, phase_ssim) - Structural Similarity Index
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
        # Use plane fitting alignment - align predicted phase to target
        phi_pred = fit_and_remove_plane(phi_pred_raw, phi_target_raw)
        phi_target = fit_and_remove_plane(phi_target_raw)
    elif phase_align_method == 'mean':
        # Use mean subtraction alignment
        phi_pred = phi_pred_raw - np.mean(phi_pred_raw)
        phi_target = phi_target_raw - np.mean(phi_target_raw)
    else:
        raise ValueError(f"Unknown phase_align_method: {phase_align_method}. Use 'plane' or 'mean'.")
    amp_target = tf.cast(trim(YY_ground_truth), tf.float32)
    amp_pred = trim(np.absolute(stitched_obj))

    # TODO complex FRC?
    mae_amp = mae(amp_target, amp_pred) # PINN
    mse_amp = mse(amp_target, amp_pred) # PINN
    psnr_amp = psnr(amp_target[:, :, 0], amp_pred[:, :, 0], normalize = True,
        shift = False)
    
    # Calculate SSIM for amplitude (convert TF tensors to numpy)
    amp_target_np = np.array(amp_target[:, :, 0])
    amp_pred_np = np.array(amp_pred[:, :, 0])
    amp_data_range = float(np.max(amp_target_np) - np.min(amp_target_np))
    ssim_amp = structural_similarity(amp_target_np, amp_pred_np, 
                                   data_range=amp_data_range)
    
    # Debug logging for FRC inputs
    print(f"DEBUG eval_reconstruction [{label}]: amp_target stats: mean={np.mean(amp_target):.6f}, std={np.std(amp_target):.6f}, shape={amp_target.shape}")
    print(f"DEBUG eval_reconstruction [{label}]: amp_pred stats: mean={np.mean(amp_pred):.6f}, std={np.std(amp_pred):.6f}, shape={amp_pred.shape}")
    print(f"DEBUG eval_reconstruction [{label}]: phi_target stats: mean={np.mean(phi_target):.6f}, std={np.std(phi_target):.6f}, shape={phi_target.shape}")
    print(f"DEBUG eval_reconstruction [{label}]: phi_pred stats: mean={np.mean(phi_pred):.6f}, std={np.std(phi_pred):.6f}, shape={phi_pred.shape}")
    
    frc_amp, frc50_amp = frc50(amp_target_np, amp_pred_np, frc_sigma=frc_sigma)

    mae_phi = mae(phi_target, phi_pred, normalize=False) # PINN
    mse_phi = mse(phi_target, phi_pred, normalize=False) # PINN
    psnr_phi = psnr(phi_target, phi_pred, normalize = False, shift = True)
    
    # Calculate SSIM for phase (scale from [-π,π] to [0,1])
    phi_target_scaled = (phi_target + np.pi) / (2 * np.pi)
    phi_pred_scaled = (phi_pred + np.pi) / (2 * np.pi)
    ssim_phi = structural_similarity(phi_target_scaled, phi_pred_scaled, 
                                   data_range=1.0)
    
    frc_phi, frc50_phi = frc50(phi_target, phi_pred, frc_sigma=frc_sigma)

    return {'mae': (mae_amp, mae_phi),
        'mse': (mse_amp, mse_phi),
        'psnr': (psnr_amp, psnr_phi),
        'ssim': (ssim_amp, ssim_phi),
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
