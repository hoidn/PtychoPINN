import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.metrics import structural_similarity
import cv2
from scipy.ndimage import fourier_shift
from scipy.signal.windows import tukey
import warnings

try:
    from skimage.restoration import unwrap_phase as _unwrap_phase_2d
except ImportError:
    _unwrap_phase_2d = None

# Helper function to prepare an image for registration based on image_prop
def _get_alignment_image(img, prop, unwrap_phase_for_gradient=False):
    """
    Prepares an image for the registration algorithm based on the 'image_prop' parameter.
    """
    if prop == 'complex':
        return img
    elif prop == 'phasor':
        # Use the complex phasor (unit amplitude, original phase)
        return np.exp(1j * np.angle(img))
    elif prop == 'phase':
        # Use the phase directly (real-valued)
        return np.angle(img)
    elif prop == 'variation':
        # Use the magnitude of the phase gradient (real-valued)
        phase = np.angle(img)
        if unwrap_phase_for_gradient: # Optional: unwrap phase first for potentially more robust gradient
            # skimage.restoration.unwrap_phase might be an option for more advanced unwrapping
            phase = np.unwrap(np.unwrap(phase, axis=0), axis=1) 
        
        # Compute gradient of phase. Note: MATLAB's get_phase_gradient_2D might have
        # different boundary handling or specific finite difference schemes.
        gy, gx = np.gradient(phase)
        # The MATLAB code example uses sqrt(dX^2+dY^2) for the alignment image.
        return np.sqrt(gx**2 + gy**2)
    else:
        raise ValueError(f"Unknown image_prop: {prop}")

# Helper function for global and linear phase term matching
def _match_phases_least_squares(img_to_correct, img_ref=None, weights_mode='magnitude', verbose = False):
    """
    Matches global and linear phase terms between two complex images, or removes
    the phase ramp from a single image.

    The function fits a 2D plane (a0 + ax*x + ay*y) to the phase difference
    (if img_ref is provided) or to the phase of img_to_correct (if img_ref is None).
    The fitted plane is then subtracted from the phase of img_to_correct.

    Args:
        img_to_correct (np.ndarray): Complex image whose phase will be corrected.
        img_ref (np.ndarray, optional): Complex reference image. If provided,
            img_to_correct will be matched to img_ref. Defaults to None.
        weights_mode (str): How to determine weights for least squares fitting.
            'magnitude': weights are proportional to the magnitude of the complex
                         product (or img_to_correct if no ref).
            'uniform': all weights are 1.

    Returns:
        np.ndarray: The phase-corrected complex image.
    """
    if not np.iscomplexobj(img_to_correct):
        warnings.warn("Attempting to match phases on a real image. Returning original image.")
        return img_to_correct

    if img_ref is not None and not np.iscomplexobj(img_ref):
        warnings.warn("Attempting to use a real image as reference for phase matching. "
                      "Returning original image to correct.")
        return img_to_correct
    
    if img_ref is not None and img_to_correct.shape != img_ref.shape:
        raise ValueError("Images must have the same shape for phase matching.")

    if img_ref is None: # Remove ramp from img_to_correct itself
        phase_to_fit = np.angle(img_to_correct)
        if weights_mode == 'magnitude':
            weights_val = np.abs(img_to_correct)
        else: # 'uniform'
            weights_val = np.ones_like(img_to_correct, dtype=float)
    else: # Match img_to_correct to img_ref
        complex_product = img_to_correct * np.conj(img_ref)
        phase_to_fit = np.angle(complex_product)
        if weights_mode == 'magnitude':
            weights_val = np.abs(complex_product)
        else: # 'uniform'
            weights_val = np.ones_like(complex_product, dtype=float)

    rows, cols = phase_to_fit.shape
    if rows == 0 or cols == 0: # Handle empty images
        return img_to_correct

    # Create coordinate matrices, normalized to [-0.5, 0.5] for numerical stability
    y_coords = np.linspace(-0.5, 0.5, rows, endpoint=True)
    x_coords = np.linspace(-0.5, 0.5, cols, endpoint=True)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # Design matrix for least squares: columns are [1, x, y]
    A = np.vstack([np.ones(rows * cols), x_grid.ravel(), y_grid.ravel()]).T
    b = phase_to_fit.ravel() # Target values (phase differences or phases)
    
    # Prepare weights for weighted least squares
    w = weights_val.ravel()
    min_weight_threshold = 1e-6 # Avoid issues with zero or tiny weights
    w[w < min_weight_threshold] = min_weight_threshold # Cap very small weights
    
    sqrt_w = np.sqrt(w)
    A_w = A * sqrt_w[:, np.newaxis] # Scale rows of A by sqrt(weight)
    b_w = b * sqrt_w               # Scale b by sqrt(weight)
    
    try:
        # Solve (A_w)^T (A_w) params = (A_w)^T b_w
        params, residuals, rank, singular_values = np.linalg.lstsq(A_w, b_w, rcond=None)
        if rank < A.shape[1]: # Check for rank deficiency if parameters are not well-determined
             warnings.warn(f"Rank deficient in phase fitting (rank {rank} for {A.shape[1]} params). "
                           "Results may be unstable.")
    except np.linalg.LinAlgError:
        warnings.warn("Least squares phase fitting failed. Returning original image.")
        return img_to_correct # Return uncorrected image if fit fails
    
    a0, ax_fit, ay_fit = params[0], params[1], params[2]

    # Create the fitted phase ramp
    phase_ramp = a0 + ax_fit * x_grid + ay_fit * y_grid

    if verbose: # Add a verbose flag, pass it from frc_preprocess_images
        print(f"  _match_phases: a0={a0:.4f}, ax_fit={ax_fit:.4f}, ay_fit={ay_fit:.4f}")
        print(f"  _match_phases: phase_ramp min={np.min(phase_ramp):.4f}, max={np.max(phase_ramp):.4f}, mean={np.mean(phase_ramp):.4f}")


    
    # Apply correction: img_corrected = img_original * exp(-1j * fitted_ramp)
    return img_to_correct * np.exp(-1j * phase_ramp)


# ---------------------------------------------------------------------------
# Calibration helpers (post-alignment, pre-tapering)
# ---------------------------------------------------------------------------

def _calibrate_complex_linear(z_pred, z_gt, verbose=False):
    """
    Complex linear scaling: z_corr = alpha * z_pred.

    alpha = <z_gt * conj(z_pred)> / <|z_pred|^2>   (complex scalar, 2 DOF).

    This is the standard "fair" normalization for CDI — corrects global
    amplitude scaling and constant phase rotation simultaneously.

    Args:
        z_pred (np.ndarray): Complex predicted image.
        z_gt   (np.ndarray): Complex ground-truth image.
        verbose (bool): Print fitted parameters.

    Returns:
        (z_corrected, params_dict)
    """
    numerator   = np.sum(z_gt * np.conj(z_pred))
    denominator = np.sum(np.abs(z_pred) ** 2)
    if np.abs(denominator) < 1e-30:
        warnings.warn("_calibrate_complex_linear: near-zero denominator; returning uncorrected.")
        return z_pred.copy(), {'alpha': complex(1, 0)}

    alpha = numerator / denominator
    z_corr = alpha * z_pred

    params = {
        'alpha': alpha,
        'amp_scale': np.abs(alpha),
        'phase_offset_rad': np.angle(alpha),
    }
    if verbose:
        print(f"  complex_linear: |alpha|={params['amp_scale']:.4f}, "
              f"arg(alpha)={np.degrees(params['phase_offset_rad']):.2f} deg")
    return z_corr, params


def _calibrate_phase_affine(z_pred, z_gt, weights_mode='magnitude', verbose=False):
    """
    Phase affine calibration: phi_corr = c * phi_pred + d   (2 DOF).

    Directly stretches the phase dynamic range.  Amplitude is left unchanged.
    Requires 2-D phase unwrapping; falls back to sequential np.unwrap if
    skimage.restoration.unwrap_phase is unavailable.

    Args:
        z_pred (np.ndarray): Complex predicted image.
        z_gt   (np.ndarray): Complex ground-truth image.
        weights_mode (str): 'magnitude' (weight by |z_gt|) or 'uniform'.
        verbose (bool): Print fitted parameters.

    Returns:
        (z_corrected, params_dict)
    """
    # --- unwrap phases ---
    if _unwrap_phase_2d is not None:
        phi_pred = _unwrap_phase_2d(np.angle(z_pred))
        phi_gt   = _unwrap_phase_2d(np.angle(z_gt))
    else:
        warnings.warn("skimage.restoration.unwrap_phase not available; "
                      "using sequential np.unwrap (less robust).")
        phi_pred = np.unwrap(np.unwrap(np.angle(z_pred), axis=0), axis=1)
        phi_gt   = np.unwrap(np.unwrap(np.angle(z_gt),   axis=0), axis=1)

    # --- weights ---
    if weights_mode == 'magnitude':
        w = np.abs(z_gt).ravel()
    else:
        w = np.ones(z_gt.size, dtype=float)
    w_min = 1e-6
    w[w < w_min] = w_min
    sqrt_w = np.sqrt(w)

    # --- weighted least squares:  phi_gt ≈ c * phi_pred + d ---
    A = np.vstack([phi_pred.ravel(), np.ones(phi_pred.size)]).T
    b = phi_gt.ravel()

    A_w = A * sqrt_w[:, np.newaxis]
    b_w = b * sqrt_w

    try:
        params_fit, _, rank, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
        if rank < 2:
            warnings.warn("_calibrate_phase_affine: rank-deficient fit.")
    except np.linalg.LinAlgError:
        warnings.warn("_calibrate_phase_affine: lstsq failed; returning uncorrected.")
        return z_pred.copy(), {'c': 1.0, 'd': 0.0}

    c, d = float(params_fit[0]), float(params_fit[1])
    phi_corr = c * phi_pred + d
    z_corr = np.abs(z_pred) * np.exp(1j * phi_corr)

    params = {'c': c, 'd': d}
    if verbose:
        print(f"  phase_affine: c={c:.4f}  (phase stretch), d={d:.4f} rad")
    return z_corr, params


def _calibrate_reim_affine(z_pred, z_gt, weights_mode='magnitude', verbose=False):
    """
    Re/Im affine calibration (diagnostic, 6 DOF):

        [Re_corr]   =  M  [Re_pred]  +  b
        [Im_corr]        [Im_pred]

    where M is a 2x2 matrix and b is a 2-vector.  Handles independent channel
    scaling, rotation, and shearing.  Too many DOF for publication — diagnostic
    only.

    Args:
        z_pred (np.ndarray): Complex predicted image.
        z_gt   (np.ndarray): Complex ground-truth image.
        weights_mode (str): 'magnitude' (weight by |z_gt|) or 'uniform'.
        verbose (bool): Print fitted parameters.

    Returns:
        (z_corrected, params_dict)
    """
    re_p = np.real(z_pred).ravel()
    im_p = np.imag(z_pred).ravel()
    re_g = np.real(z_gt).ravel()
    im_g = np.imag(z_gt).ravel()

    n = re_p.size

    # --- weights ---
    if weights_mode == 'magnitude':
        w = np.abs(z_gt).ravel()
    else:
        w = np.ones(n, dtype=float)
    w_min = 1e-6
    w[w < w_min] = w_min
    sqrt_w = np.sqrt(w)

    # Design matrix: [re_p, im_p, 1] per pixel  (shared for both output channels)
    A = np.column_stack([re_p, im_p, np.ones(n)])  # (n, 3)
    A_w = A * sqrt_w[:, np.newaxis]

    results = {}
    corrected = np.zeros_like(z_pred)

    for label, target in [('Re', re_g), ('Im', im_g)]:
        b_w = target * sqrt_w
        try:
            p, _, rank, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
            if rank < 3:
                warnings.warn(f"_calibrate_reim_affine [{label}]: rank-deficient fit.")
        except np.linalg.LinAlgError:
            warnings.warn(f"_calibrate_reim_affine [{label}]: lstsq failed.")
            p = np.array([1.0, 0.0, 0.0]) if label == 'Re' else np.array([0.0, 1.0, 0.0])

        corr_channel = (A @ p).reshape(z_pred.shape)
        if label == 'Re':
            corrected += corr_channel
            results['M_row_re'] = p[:2].tolist()
            results['b_re'] = float(p[2])
        else:
            corrected += 1j * corr_channel
            results['M_row_im'] = p[:2].tolist()
            results['b_im'] = float(p[2])

    M = np.array([results['M_row_re'], results['M_row_im']])
    results['M'] = M.tolist()

    if verbose:
        print(f"  reim_affine: M =\n    {M}")
        print(f"  reim_affine: b = ({results['b_re']:.4f}, {results['b_im']:.4f})")

    return corrected, results

def frc_preprocess_images(img1_orig, img2_orig,
                          image_prop='phasor',
                          taper_px=20,
                          remove_ramp_before_register=True,
                          registration_upsample_factor=100,
                          verbose=False,
                          align=True,
                          calibration_mode=None):
    """
    Pre-processes two images for Fourier Ring/Shell Correlation (FRC/FSC).

    This function implements the three core pre-processing steps:
    1. Sub-pixel registration (two-pass: coarse and fine).
    2. Global and linear phase term matching (applied at multiple stages).
    3. Soft-edge mask multiplication (tapering) using a Tukey window.

    Args:
        img1_orig (np.ndarray): First 2D image (can be complex or real).
        img2_orig (np.ndarray): Second 2D image (can be complex or real).
        image_prop (str): Determines how images are treated for registration:
                          'complex': Use the full complex image.
                          'phasor': Use phase with unit amplitude (complex).
                          'phase': Use the phase only (real).
                          'variation': Use magnitude of phase gradient (real).
                          Defaults to 'phasor'.
        taper_px (int): Width of the taper on each side of the image in pixels
                        for the Tukey window. If 0, no tapering is applied.
                        Defaults to 20.
        remove_ramp_before_register (bool): If True, an initial phase ramp
                                           correction is applied to both images
                                           before the first registration step.
                                           Defaults to True.
        registration_upsample_factor (int): Upsampling factor for sub-pixel
                                           registration using phase cross-correlation.
                                           Defaults to 100.
        verbose (bool): If True, print progress messages. Defaults to False.
        calibration_mode (str or None): Post-alignment calibration applied to
            img2 toward img1.  One of:
            - None: no calibration (default, backward-compatible).
            - 'complex_linear': z_corr = alpha * z_pred (2 DOF).
            - 'phase_affine': phi_corr = c * phi + d (2 DOF).
            - 'reim_affine': (Re,Im)_corr = M*(Re,Im) + b (6 DOF, diagnostic).

    Returns:
        tuple: (processed_img1, processed_img2, calibration_info)
               The two pre-processed images, ready for FRC calculation.
               calibration_info is None when no calibration applied, otherwise
               a dict with fitted parameters.
               Returns (None, None, None) if processing fails.
    """
    if verbose: print("Starting FRC pre-processing...")

    img1 = img1_orig.copy()
    img2 = img2_orig.copy()

    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same shape.")
    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError("Input images must be 2D.")

    if align:
        # --- 1. Initial phase ramp removal (optional, typically for complex images) ---
        if remove_ramp_before_register and np.iscomplexobj(img1) and np.iscomplexobj(img2):
            if verbose: print("Step 1a: Initial phase ramp removal on img1.")
            img1 = _match_phases_least_squares(img1, img_ref=None)
            if verbose: print("Step 1b: Initial phase ramp removal on img2 (matched to corrected img1).")
            img2 = _match_phases_least_squares(img2, img_ref=img1)

        # --- 2. Sub-pixel registration (Two rounds) ---
        # Round 1: Initial (coarse) alignment
        if verbose: print("Step 2a: Sub-pixel registration - Round 1 (Initial).")
        align_img1_r1 = _get_alignment_image(img1, image_prop)
        align_img2_r1 = _get_alignment_image(img2, image_prop)
        
        # Calculate shift of align_img2_r1 relative to align_img1_r1
        # skimage's phase_cross_correlation handles real or complex spatial inputs correctly.
        shift_r1, _, _ = phase_cross_correlation(
            align_img1_r1, align_img2_r1, 
            upsample_factor=registration_upsample_factor
        )
        if verbose: print(f"  Initial shift (y,x): {shift_r1}")

        # Create sub-images for fine alignment based on integer part of shift_r1.
        # These are crops from img1 and img2 corresponding to their overlapping region
        # after the coarse integer shift is notionally applied.
        s0, s1 = img1.shape # original shape
        shift_r1_int = np.round(shift_r1).astype(int)

        # Slices for img1 (reference for this cropping)
        crop_y1_start = max(0, shift_r1_int[0]); crop_y1_end = min(s0, s0 + shift_r1_int[0])
        crop_x1_start = max(0, shift_r1_int[1]); crop_x1_end = min(s1, s1 + shift_r1_int[1])
        # Slices for img2 (to match the region sliced from img1)
        crop_y2_start = max(0, -shift_r1_int[0]); crop_y2_end = min(s0, s0 - shift_r1_int[0])
        crop_x2_start = max(0, -shift_r1_int[1]); crop_x2_end = min(s1, s1 - shift_r1_int[1])

        subimg1_for_fine = img1[crop_y1_start:crop_y1_end, crop_x1_start:crop_x1_end]
        subimg2_for_fine = img2[crop_y2_start:crop_y2_end, crop_x2_start:crop_x2_end]

        if subimg1_for_fine.size == 0 or subimg2_for_fine.size == 0:
            warnings.warn("Sub-images for fine alignment are empty after coarse registration. "
                        "This might indicate very large initial misalignment or small images. "
                        "Returning (None, None, None).")
            return None, None, None
            
        # Phase matching for sub-images before fine alignment (mirroring MATLAB's remove_linearphase_v2)
        if np.iscomplexobj(subimg1_for_fine) and np.iscomplexobj(subimg2_for_fine):
            if verbose: print("Step 2b: Phase ramp removal for fine alignment sub-images.")
            subimg1_for_fine = _match_phases_least_squares(subimg1_for_fine, img_ref=None)
            subimg2_for_fine = _match_phases_least_squares(subimg2_for_fine, img_ref=None)

        # Round 2: Fine alignment on (potentially phase-corrected) sub-images
        if verbose: print("Step 2c: Sub-pixel registration - Round 2 (Fine).")
        align_img1_r2 = _get_alignment_image(subimg1_for_fine, image_prop)
        align_img2_r2 = _get_alignment_image(subimg2_for_fine, image_prop)
        
        shift_r2, _, _ = phase_cross_correlation(
            align_img1_r2, align_img2_r2, 
            upsample_factor=registration_upsample_factor
        ) # This is 'deltafine' in MATLAB
        if verbose: print(f"  Fine shift (y,x): {shift_r2}")

        # Apply fine shift to subimg2_for_fine.
        # phase_cross_correlation returns (row_shift, col_shift) needed to move the
        # `moving_image` (align_img2_r2) to align with `reference_image` (align_img1_r2).
        # So, fourier_shift(subimg2_for_fine, -shift_r2) applies this alignment.
        shifted_subimg2_for_fine = fourier_shift(subimg2_for_fine, shift=-shift_r2)#-shift_r2)
        
        # Crop both images to their common valid region after the fine shift.
        # `subimg1_for_fine` is the reference. `shifted_subimg2_for_fine` is now aligned to it.
        # The applied shift to get `shifted_subimg2_for_fine` was `-shift_r2`.
        # Let `eff_shift_y, eff_shift_x = -shift_r2`.
        eff_shift_y, eff_shift_x = -shift_r2 
        h_fine, w_fine = subimg1_for_fine.shape

        # Determine slices for subimg1_for_fine:
        # If eff_shift_y > 0, content of subimg2 effectively moved "down",
        # so subimg1_for_fine's top rows (0 to ceil(eff_shift_y)-1) have no corresponding original data.
        s1_crop_y_start = int(np.ceil(max(0, eff_shift_y)))
        s1_crop_y_end   = int(np.floor(min(h_fine, h_fine + eff_shift_y))) # h_fine + eff_shift_y can be < h_fine if eff_shift_y < 0
        s1_crop_x_start = int(np.ceil(max(0, eff_shift_x)))
        s1_crop_x_end   = int(np.floor(min(w_fine, w_fine + eff_shift_x)))

        print("subimg1, shifted_sub_im2")


        final_subimg1 = subimg1_for_fine[s1_crop_y_start:s1_crop_y_end, s1_crop_x_start:s1_crop_x_end]
        # Apply the same slices to the already shifted `shifted_subimg2_for_fine`
        final_subimg2 = shifted_subimg2_for_fine[s1_crop_y_start:s1_crop_y_end, s1_crop_x_start:s1_crop_x_end]

        if final_subimg1.size == 0 or final_subimg2.size == 0:
            warnings.warn("Final sub-images are empty after fine registration and cropping. "
                        "Returning (None, None, None).")
            return None, None, None
            
        # Special handling for image_prop == 'variation' as in MATLAB:
        # The alignment was done using phase variation, but the final images
        # (before FRC-specific phase matching & tapering) should be phasors
        # derived from the complex-valued sub-images.
        if image_prop == 'variation':
            if np.iscomplexobj(final_subimg1) and np.iscomplexobj(final_subimg2):
                if verbose: print("Step 2d: Converting to phasors for 'variation' property.")
                final_subimg1 = np.exp(1j * np.angle(final_subimg1))
                final_subimg2 = np.exp(1j * np.angle(final_subimg2))
            # else: if inputs were real, they remain real. This case might not be typical for 'variation'.
    else:
        final_subimg1 = img1
        final_subimg2 = img2
    # --- 3. Final phase matching (global and linear term) ---
    # MATLAB: subim1 = utils.stabilize_phase(subim1, subim2, ...); -> matches subim1 to subim2
    if np.iscomplexobj(final_subimg1) and np.iscomplexobj(final_subimg2):
        if verbose: print("Step 3: Final phase matching of img1 to img2.")
        final_subimg1 = _match_phases_least_squares(final_subimg1, img_ref=final_subimg2, verbose=verbose)

    # --- 3b. Calibration (correct img2 toward img1) ---
    calibration_info = None
    _valid_modes = (None, 'complex_linear', 'phase_affine', 'reim_affine')
    if calibration_mode is not None:
        if calibration_mode not in _valid_modes:
            raise ValueError(f"calibration_mode must be one of {_valid_modes}, "
                             f"got '{calibration_mode}'")
        if not (np.iscomplexobj(final_subimg1) and np.iscomplexobj(final_subimg2)):
            warnings.warn("Calibration requires complex images; skipping.")
        else:
            if verbose:
                print(f"Step 3b: Applying '{calibration_mode}' calibration.")
            if calibration_mode == 'complex_linear':
                final_subimg2, calibration_info = _calibrate_complex_linear(
                    final_subimg2, final_subimg1, verbose=verbose)
            elif calibration_mode == 'phase_affine':
                final_subimg2, calibration_info = _calibrate_phase_affine(
                    final_subimg2, final_subimg1, verbose=verbose)
            elif calibration_mode == 'reim_affine':
                final_subimg2, calibration_info = _calibrate_reim_affine(
                    final_subimg2, final_subimg1, verbose=verbose)
            calibration_info['mode'] = calibration_mode

    # --- 4. Soft-edge mask multiplication (Tapering) ---
    if taper_px > 0 and final_subimg1.size > 0 : # Check size again in case they became empty
        if verbose: print(f"Step 4: Applying soft-edge taper ({taper_px} px).")
        h, w = final_subimg1.shape
        
        # Ensure taper_px is not excessively large for the image dimensions to avoid alpha > 1 for Tukey.
        # alpha for Tukey is (2 * taper_width_single_side) / total_length.
        # taper_px is the width of the tapered region on one side.
        actual_taper_y = min(taper_px, h // 2) if h > 0 else 0
        actual_taper_x = min(taper_px, w // 2) if w > 0 else 0

        alpha_y = (2 * actual_taper_y / h) if h > 0 else 0 
        alpha_x = (2 * actual_taper_x / w) if w > 0 else 0
        
        # Create 1D Tukey windows for each dimension
        tukey_y = tukey(h, alpha=alpha_y, sym=True) if h > 0 else np.array([1.0])
        tukey_x = tukey(w, alpha=alpha_x, sym=True) if w > 0 else np.array([1.0])
        
        # Combine into a 2D taper mask by outer product
        taper_mask = tukey_y[:, np.newaxis] * tukey_x[np.newaxis, :]
        
        final_subimg1 = final_subimg1 * taper_mask
        final_subimg2 = final_subimg2 * taper_mask
            
    if verbose: print("FRC pre-processing complete.")
    return final_subimg1, final_subimg2, calibration_info


# ---------------------------------------------------------------------------
# Decomposed FRC
# ---------------------------------------------------------------------------

def compute_decomposed_frc(z_gt, z_pred,
                           preprocess_kwargs=None,
                           calibration_mode=None,
                           verbose=False):
    """
    Compute separate FRC curves for amplitude, phasor, and full complex images.

    This isolates phase resolution from amplitude resolution without any
    calibration.  Optionally applies a calibration mode to the phasor
    component to quantify the gap caused by phase compression.

    Args:
        z_gt   (np.ndarray): Complex ground-truth image (2D).
        z_pred (np.ndarray): Complex predicted image (2D).
        preprocess_kwargs (dict or None): Extra kwargs passed to
            ``frc_preprocess_images`` (e.g. taper_px, image_prop).
            ``calibration_mode`` is handled separately and should NOT
            be included here.
        calibration_mode (str or None): If provided, an *additional*
            phasor FRC is computed after applying this calibration.
        verbose (bool): Print progress.

    Returns:
        dict with keys:
            'complex'   : {'FR_curve', 'x_FR', 'T_curve', 'x_T', 'AUC'}
            'amplitude' : {'FR_curve', 'x_FR', 'T_curve', 'x_T', 'AUC'}
            'phasor'    : {'FR_curve', 'x_FR', 'T_curve', 'x_T', 'AUC'}
            'phasor_calibrated' : (only if calibration_mode is not None)
                           same sub-dict, plus 'calibration_info'.
    """
    from ptycho_torch.eval.eval_metrics import FSC

    if preprocess_kwargs is None:
        preprocess_kwargs = {}
    # Ensure caller doesn't accidentally pass calibration_mode twice
    preprocess_kwargs.pop('calibration_mode', None)

    def _frc_from_pair(img1, img2, label):
        """Run FSC and package results."""
        FR_curve, x_FR, T_curve, x_T = FSC(img1, img2)
        # AUC up to Nyquist (x_FR <= 1)
        mask = x_FR <= 1.0
        if np.any(mask):
            auc = float(np.mean(FR_curve[mask]))
        else:
            auc = 0.0
        if verbose:
            print(f"  {label}: AUC = {auc:.4f}")
        return {'FR_curve': FR_curve, 'x_FR': x_FR,
                'T_curve': T_curve, 'x_T': x_T, 'AUC': auc}

    results = {}

    # --- 1. Full complex FRC ---
    if verbose:
        print("Decomposed FRC: computing complex FRC...")
    gt_c, pred_c, _ = frc_preprocess_images(
        z_gt, z_pred, verbose=verbose, **preprocess_kwargs)
    if gt_c is not None:
        results['complex'] = _frc_from_pair(gt_c, pred_c, 'complex')
    else:
        warnings.warn("compute_decomposed_frc: complex preprocessing failed.")
        return results

    # --- 2. Amplitude FRC  (real-valued, no phase involved) ---
    if verbose:
        print("Decomposed FRC: computing amplitude FRC...")
    amp_gt   = np.abs(z_gt)
    amp_pred = np.abs(z_pred)
    # Preprocess as real images (alignment still useful for shift correction)
    gt_a, pred_a, _ = frc_preprocess_images(
        amp_gt, amp_pred, verbose=verbose, **preprocess_kwargs)
    if gt_a is not None:
        results['amplitude'] = _frc_from_pair(gt_a, pred_a, 'amplitude')

    # --- 3. Phasor FRC  (unit-amplitude complex, phase only) ---
    if verbose:
        print("Decomposed FRC: computing phasor FRC...")
    phasor_gt   = np.exp(1j * np.angle(z_gt))
    phasor_pred = np.exp(1j * np.angle(z_pred))
    gt_p, pred_p, _ = frc_preprocess_images(
        phasor_gt, phasor_pred, verbose=verbose, **preprocess_kwargs)
    if gt_p is not None:
        results['phasor'] = _frc_from_pair(gt_p, pred_p, 'phasor')

    # --- 4. Calibrated phasor FRC (optional) ---
    if calibration_mode is not None:
        if verbose:
            print(f"Decomposed FRC: computing calibrated phasor FRC "
                  f"(mode='{calibration_mode}')...")
        # Apply calibration on full complex images, then extract phasors
        _, pred_cal, cal_info = frc_preprocess_images(
            z_gt, z_pred, calibration_mode=calibration_mode,
            verbose=verbose, **preprocess_kwargs)
        if pred_cal is not None:
            # Re-extract phasors from the calibrated result
            gt_pc, pred_pc, _ = frc_preprocess_images(
                np.exp(1j * np.angle(gt_c)),
                np.exp(1j * np.angle(pred_cal)),
                verbose=verbose, **preprocess_kwargs)
            if gt_pc is not None:
                entry = _frc_from_pair(gt_pc, pred_pc, 'phasor_calibrated')
                entry['calibration_info'] = cal_info
                results['phasor_calibrated'] = entry

    return results


# ---------------------------------------------------------------------------
# Fourier-space metrics: R-factor, PRTF
# ---------------------------------------------------------------------------

import torch

def _to_torch_complex(arr, device=None):
    """Convert numpy array to torch complex tensor on the specified device."""
    if isinstance(arr, torch.Tensor):
        t = arr
    else:
        t = torch.from_numpy(np.ascontiguousarray(arr))
    if not t.is_complex():
        t = t.to(torch.complex64)
    elif t.dtype == torch.complex128:
        t = t.to(torch.complex64)
    if device is not None:
        t = t.to(device)
    return t


def _to_torch_float(arr, device=None):
    """Convert numpy array to torch float tensor on the specified device."""
    if isinstance(arr, torch.Tensor):
        t = arr.float()
    else:
        t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    if device is not None:
        t = t.to(device)
    return t


def extract_object_patches_from_canvas(canvas, scan_positions, patch_size,
                                        device=None):
    """
    Extract object patches from a reconstructed canvas at each scan position.

    Vectorized via PyTorch: computes all integer offsets, builds a validity
    mask, and extracts patches using advanced indexing in a single pass.

    The canvas coordinate system follows the reconstruction convention:
    position (y, x) maps to canvas pixel (y - min_y, x - min_x).

    Args:
        canvas: (H, W) complex — reconstructed object.
            Accepts np.ndarray or torch.Tensor.
        scan_positions: (N, 2) array of (row, col) scan positions.
            Accepts np.ndarray or torch.Tensor.
        patch_size (int): Side length of square patches to extract.
        device: torch device for computation. If None, uses 'cpu'.

    Returns:
        patches (torch.Tensor): (N_valid, patch_size, patch_size) complex.
        valid_mask (torch.Tensor): (N,) bool — True for in-bounds positions.
    """
    if device is None:
        device = torch.device('cpu')

    canvas_t = _to_torch_complex(canvas, device=device)
    pos_t = _to_torch_float(scan_positions, device=device)
    H, W = canvas_t.shape

    # Integer pixel offsets relative to canvas origin
    origin = pos_t.min(dim=0).values  # (2,)
    offsets = torch.round(pos_t - origin).long()  # (N, 2)
    oy, ox = offsets[:, 0], offsets[:, 1]

    # Vectorized bounds check
    valid_mask = (
        (oy >= 0) & (ox >= 0) &
        (oy + patch_size <= H) & (ox + patch_size <= W)
    )

    valid_idx = torch.where(valid_mask)[0]
    n_valid = valid_idx.shape[0]

    if n_valid == 0:
        empty = torch.empty((0, patch_size, patch_size),
                            dtype=canvas_t.dtype, device=device)
        return empty, valid_mask

    # Build vectorized extraction indices
    oy_valid = oy[valid_idx]  # (N_valid,)
    ox_valid = ox[valid_idx]  # (N_valid,)

    # Patch-local row/col grids: (patch_size,)
    patch_rows = torch.arange(patch_size, device=device)
    patch_cols = torch.arange(patch_size, device=device)

    # Global row/col for every pixel of every valid patch
    # rows: (N_valid, patch_size, 1),  cols: (N_valid, 1, patch_size)
    rows = oy_valid[:, None, None] + patch_rows[None, :, None]  # (N_valid, P, 1)
    cols = ox_valid[:, None, None] + patch_cols[None, None, :]   # (N_valid, 1, P)

    # Extract all patches at once via advanced indexing
    patches = canvas_t[rows.expand(-1, patch_size, patch_size),
                       cols.expand(-1, patch_size, patch_size)]  # (N_valid, P, P)

    return patches, valid_mask


def simulate_exit_wave_diffraction(object_patches, probe, batch_size=2000,
                                    device=None):
    """
    Simulate far-field diffraction from object patches illuminated by a probe.

    Vectorized via PyTorch: batched Hadamard product, FFT2, and fftshift
    on GPU following the pattern in ``helper.illuminate_and_diffract``.

    For each patch j:  psi_j = FFT{ O_j * P },   I_j = |psi_j|^2

    Args:
        object_patches: (N, H, W) complex object patches.
            Accepts np.ndarray or torch.Tensor.
        probe: (H, W) complex illumination probe.
            Accepts np.ndarray or torch.Tensor.
        batch_size (int): GPU batch size to limit VRAM. Default 2000.
        device: torch device. If None, uses CUDA when available.

    Returns:
        dict with torch.Tensor values (on *device*):
            'exit_waves':  (N, H, W) complex — O_j * P.
            'psi_calc':    (N, H, W) complex — fftshifted FFT.
            'amplitudes':  (N, H, W) float  — |psi_calc|.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    patches_t = _to_torch_complex(object_patches, device=device)
    probe_t = _to_torch_complex(probe, device=device)

    N, H, W = patches_t.shape
    probe_b = probe_t.unsqueeze(0)  # (1, H, W) for broadcasting

    exit_waves = torch.empty_like(patches_t)
    psi_calc = torch.empty_like(patches_t)
    amplitudes = torch.empty((N, H, W), dtype=torch.float32, device=device)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = patches_t[start:end]

        ew = batch * probe_b                                     # (B, H, W)
        pc = torch.fft.fftshift(torch.fft.fft2(ew), dim=(-2, -1))  # (B, H, W)
        amp = torch.abs(pc)                                       # (B, H, W)

        exit_waves[start:end] = ew
        psi_calc[start:end] = pc
        amplitudes[start:end] = amp

    return {
        'exit_waves': exit_waves,
        'psi_calc': psi_calc,
        'amplitudes': amplitudes,
    }


def compute_amplitude_rfactor(predicted_amplitudes, measured_data,
                               data_is_amplitude=False,
                               optimal_scaling=False, verbose=False):
    """
    Compute the amplitude R-factor (R_ptycho).

    R_F = sum_j sum_q | A_meas_j(q) - alpha * A_pred_j(q) |
          / sum_j sum_q A_meas_j(q)

    where A_meas = sqrt(I_meas) and A_pred = |FFT{O_j * P}|.

    Args:
        predicted_amplitudes (np.ndarray): (N, H, W) predicted amplitudes.
        measured_data (np.ndarray): (N, H, W) measured diffraction data.
            Interpreted as intensities by default; set data_is_amplitude=True
            if already sqrt'd.
        data_is_amplitude (bool): If True, measured_data is already amplitudes.
        optimal_scaling (bool): If True, find alpha that minimizes R.
        verbose (bool): Print diagnostics.

    Returns:
        dict with:
            'R_factor' (float): Global amplitude R-factor.
            'R_per_position' (np.ndarray): (N,) per-position R-factors.
            'scale_factor' (float): Optimal scaling alpha (1.0 if not used).
    """
    A_pred = np.asarray(predicted_amplitudes)
    if data_is_amplitude:
        A_meas = np.asarray(measured_data).copy()
    else:
        A_meas = np.sqrt(np.maximum(np.asarray(measured_data), 0.0))

    # Mask non-finite pixels (NaN/Inf from 0/0 in barycentric accumulation)
    finite_mask = np.isfinite(A_pred) & np.isfinite(A_meas)
    n_bad = int(np.count_nonzero(~finite_mask))
    if n_bad > 0:
        if verbose:
            print(f"  R-factor: masking {n_bad} non-finite pixels ({100*n_bad/finite_mask.size:.2f}%)")
        A_pred = np.where(finite_mask, A_pred, 0.0)
        A_meas = np.where(finite_mask, A_meas, 0.0)

    # Optimal scaling
    if optimal_scaling:
        denom = np.sum(A_pred ** 2)
        alpha = float(np.sum(A_pred * A_meas) / denom) if denom > 0 else 1.0
    else:
        alpha = 1.0

    if verbose:
        print(f"  R-factor scale alpha = {alpha:.6f}")

    # Global R-factor (vectorized)
    residual = np.abs(A_meas - alpha * A_pred)
    R_global = float(np.sum(residual) / np.sum(A_meas))

    # Per-position R-factors (vectorized over spatial dims)
    denom_per = np.sum(A_meas, axis=(1, 2))  # (N,)
    num_per = np.sum(residual, axis=(1, 2))   # (N,)
    R_per = np.where(denom_per > 0, num_per / denom_per, np.nan)

    if verbose:
        print(f"  R_factor = {R_global:.6f}, mean per-position = {np.nanmean(R_per):.6f}")

    return {
        'R_factor': R_global,
        'R_per_position': R_per,
        'scale_factor': float(alpha),
    }


def compute_prtf(predicted_amplitudes, measured_data,
                  data_is_amplitude=False,
                  optimal_scaling=False,
                  n_bins=None, verbose=False):
    """
    Compute the Phase Retrieval Transfer Function (PRTF).

    PRTF(q) = < alpha * |psi_calc(q)| >_j  /  < A_meas(q) >_j

    averaged over scan positions j, then radially binned into frequency
    shells q.  A perfect reconstruction gives PRTF = 1 everywhere.

    Args:
        predicted_amplitudes (np.ndarray): (N, H, W) predicted amplitudes.
        measured_data (np.ndarray): (N, H, W) measured diffraction data.
        data_is_amplitude (bool): If True, measured_data is already amplitudes.
        optimal_scaling (bool): Scale predicted to match measured globally.
        n_bins (int or None): Number of radial bins. Default: H // 2.
        verbose (bool): Print diagnostics.

    Returns:
        dict with:
            'PRTF_curve' (np.ndarray): (n_bins,) PRTF values.
            'spatial_freq' (np.ndarray): (n_bins,) normalized frequencies
                (0 = DC, 1 = Nyquist).
            'resolution_half' (float or None): Frequency where PRTF crosses 0.5.
            'resolution_1e' (float or None): Frequency where PRTF crosses 1/e.
    """
    from ptycho_torch.eval.eval_metrics import spinavej

    A_pred = np.asarray(predicted_amplitudes)
    if data_is_amplitude:
        A_meas = np.asarray(measured_data).copy()
    else:
        A_meas = np.sqrt(np.maximum(np.asarray(measured_data), 0.0))

    # Mask non-finite pixels (NaN/Inf from 0/0 in barycentric accumulation)
    finite_mask = np.isfinite(A_pred) & np.isfinite(A_meas)
    n_bad = int(np.count_nonzero(~finite_mask))
    if n_bad > 0:
        if verbose:
            print(f"  PRTF: masking {n_bad} non-finite pixels ({100*n_bad/finite_mask.size:.2f}%)")
        A_pred = np.where(finite_mask, A_pred, 0.0)
        A_meas = np.where(finite_mask, A_meas, 0.0)

    # Optimal scaling
    if optimal_scaling:
        denom = np.sum(A_pred ** 2)
        alpha = float(np.sum(A_pred * A_meas) / denom) if denom > 0 else 1.0
    else:
        alpha = 1.0

    if verbose:
        print(f"  PRTF scale alpha = {alpha:.6f}")

    # Position-average in 2D Fourier space
    avg_pred = np.mean(alpha * A_pred, axis=0)  # (H, W)
    avg_meas = np.mean(A_meas, axis=0)          # (H, W)

    # Radial binning using spinavej (same as FSC)
    num_radial = np.real(spinavej(avg_pred))
    den_radial = np.real(spinavej(avg_meas))

    eps = np.finfo(float).eps
    PRTF_curve = num_radial / (den_radial + eps)

    H = A_pred.shape[1]
    spatial_freq = np.arange(len(PRTF_curve)) / (H / 2.0)

    # Find resolution crossings
    def _find_crossing(curve, freq, threshold):
        """Find first frequency where curve drops below threshold."""
        below = np.where(curve < threshold)[0]
        if len(below) == 0:
            return None
        idx = below[0]
        if idx == 0:
            return float(freq[0])
        f0, f1 = freq[idx - 1], freq[idx]
        c0, c1 = curve[idx - 1], curve[idx]
        if abs(c0 - c1) < eps:
            return float(f0)
        frac = (threshold - c0) / (c1 - c0)
        return float(f0 + frac * (f1 - f0))

    resolution_half = _find_crossing(PRTF_curve, spatial_freq, 0.5)
    resolution_1e = _find_crossing(PRTF_curve, spatial_freq, 1.0 / np.e)

    if verbose:
        print(f"  PRTF resolution (0.5 crossing): {resolution_half}")
        print(f"  PRTF resolution (1/e crossing): {resolution_1e}")

    return {
        'PRTF_curve': PRTF_curve,
        'spatial_freq': spatial_freq,
        'resolution_half': resolution_half,
        'resolution_1e': resolution_1e,
    }


def evaluate_fourier_metrics(canvas, probe, scan_positions,
                              measured_diffraction, patch_size,
                              data_is_amplitude=False,
                              metrics=('rfactor', 'prtf'),
                              optimal_scaling=True,
                              max_positions=None,
                              device=None,
                              verbose=False):
    """
    Evaluate Fourier-space metrics for a ptychographic reconstruction.

    Extracts object patches from the canvas, simulates diffraction via the
    forward model, and computes the requested metrics against measured data.

    Patch extraction and far-field propagation are vectorized via PyTorch
    and run on GPU when available.

    Args:
        canvas: (H, W) complex reconstructed object (np.ndarray or torch.Tensor).
        probe: (H_probe, W_probe) complex illumination probe.
        scan_positions: (N, 2) scan positions (row, col).
        measured_diffraction: (N, H_probe, W_probe) measured diffraction data
            (intensities by default).
        patch_size (int): Patch side length (must match probe dimensions).
        data_is_amplitude (bool): Whether measured data is amplitude (True)
            or intensity (False, default).
        metrics (tuple of str): Which metrics to compute.
            Options: 'rfactor', 'prtf'.
        optimal_scaling (bool): Auto-scale predicted to match measured.
        max_positions (int or None): Subsample to at most this many positions.
        device: torch device for GPU acceleration. If None, auto-selects.
        verbose (bool): Print progress.

    Returns:
        dict with keys matching requested metrics, each containing
        the corresponding metric function's return dict.
        Also includes 'n_positions_used' and 'valid_fraction'.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print(f"Evaluating Fourier metrics: {metrics} (device={device})")
        canvas_shape = canvas.shape if hasattr(canvas, 'shape') else 'unknown'
        probe_shape = probe.shape if hasattr(probe, 'shape') else 'unknown'
        n_pos = scan_positions.shape[0] if hasattr(scan_positions, 'shape') else len(scan_positions)
        print(f"  Canvas shape: {canvas_shape}, Probe shape: {probe_shape}")
        print(f"  N scan positions: {n_pos}, patch_size: {patch_size}")

    # 1. Extract patches (vectorized PyTorch)
    patches, valid_mask = extract_object_patches_from_canvas(
        canvas, scan_positions, patch_size, device=device
    )
    n_valid = patches.shape[0]
    n_total = valid_mask.shape[0]
    valid_fraction = n_valid / n_total if n_total > 0 else 0.0

    if verbose:
        print(f"  Valid patches: {n_valid}/{n_total} ({valid_fraction:.1%})")

    if n_valid == 0:
        warnings.warn("No valid patches extracted from canvas.")
        return {'n_positions_used': 0, 'valid_fraction': 0.0}

    # Filter measured data to valid positions
    valid_mask_np = valid_mask.cpu().numpy() if isinstance(valid_mask, torch.Tensor) else valid_mask
    meas_valid = np.asarray(measured_diffraction)[valid_mask_np]

    # 1b. Drop patches containing NaN/Inf (edge artifacts from 0/0 in
    #     barycentric accumulation).  A single non-finite pixel propagates
    #     through FFT and poisons the entire amplitude array.
    finite_mask = torch.isfinite(patches.real) & torch.isfinite(patches.imag)
    clean_mask = finite_mask.view(n_valid, -1).all(dim=1)  # (n_valid,)
    n_dirty = int((~clean_mask).sum().item())
    if n_dirty > 0:
        clean_idx = torch.where(clean_mask)[0]
        patches = patches[clean_idx]
        clean_idx_np = clean_idx.cpu().numpy()
        meas_valid = meas_valid[clean_idx_np]
        n_valid = patches.shape[0]
        if verbose:
            print(f"  Dropped {n_dirty} patches with non-finite pixels, "
                  f"{n_valid} clean patches remain")

    if n_valid == 0:
        warnings.warn("All patches dropped (non-finite); cannot compute Fourier metrics.")
        return {'n_positions_used': 0, 'valid_fraction': float(valid_fraction)}

    # 2. Subsample if requested
    if max_positions is not None and n_valid > max_positions:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_valid, size=max_positions, replace=False)
        idx.sort()
        idx_t = torch.from_numpy(idx).to(device)
        patches = patches[idx_t]
        meas_valid = meas_valid[idx]
        n_valid = max_positions
        if verbose:
            print(f"  Subsampled to {n_valid} positions")

    # 3. Simulate diffraction (vectorized PyTorch)
    if verbose:
        print("  Simulating exit wave diffraction...")
    sim = simulate_exit_wave_diffraction(patches, probe, device=device)

    # Move amplitudes to numpy for metric computation (uses spinavej)
    amp_np = sim['amplitudes'].cpu().numpy()

    # 4. Compute requested metrics
    results = {
        'n_positions_used': n_valid,
        'valid_fraction': float(valid_fraction),
    }

    if 'rfactor' in metrics:
        if verbose:
            print("  Computing amplitude R-factor...")
        results['rfactor'] = compute_amplitude_rfactor(
            amp_np, meas_valid,
            data_is_amplitude=data_is_amplitude,
            optimal_scaling=optimal_scaling,
            verbose=verbose,
        )

    if 'prtf' in metrics:
        if verbose:
            print("  Computing PRTF...")
        results['prtf'] = compute_prtf(
            amp_np, meas_valid,
            data_is_amplitude=data_is_amplitude,
            optimal_scaling=optimal_scaling,
            verbose=verbose,
        )

    return results