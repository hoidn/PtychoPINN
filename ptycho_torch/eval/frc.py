import numpy as np
from skimage.registration import phase_cross_correlation
import cv2
from scipy.ndimage import fourier_shift
from scipy.signal.windows import tukey
import warnings

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


def frc_preprocess_images(img1_orig, img2_orig, 
                          image_prop='phasor', 
                          taper_px=20, 
                          remove_ramp_before_register=True,
                          registration_upsample_factor=100,
                          verbose=False,
                          align = True):
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

    Returns:
        tuple: (processed_img1, processed_img2)
               The two pre-processed images, ready for FRC calculation.
               Returns (None, None) if processing fails significantly (e.g. empty subimages).
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
                        "Returning (None, None).")
            return None, None
            
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
                        "Returning (None, None).")
            return None, None
            
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
        final_subimg1 = _match_phases_least_squares(final_subimg1, img_ref=final_subimg2, verbose = verbose)

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
    return final_subimg1, final_subimg2