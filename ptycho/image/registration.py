"""Image registration module for ptychographic reconstruction alignment.

This module provides robust image registration functionality using phase cross-correlation
to detect and correct translational misalignments between reconstructions and ground truth.
The implementation uses scikit-image's phase_cross_correlation for sub-pixel precision.

Key features:
- Sub-pixel precision registration using upsampled FFT
- Complex-valued image support via magnitude extraction
- Fourier-domain shifting for exact sub-pixel alignment
- Border cropping to eliminate wrap-around artifacts

Typical workflow:
1. Use find_translation_offset() to detect misalignment
2. Apply apply_shift_and_crop() to correct and clean the alignment
3. Or use register_and_align() for one-step convenience
"""

from __future__ import annotations
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from typing import Tuple
from skimage.registration import phase_cross_correlation


def _mag(a: np.ndarray) -> np.ndarray:
    """Extract magnitude from complex or real arrays.
    
    Args:
        a: Input array (real or complex)
        
    Returns:
        Magnitude array (real-valued)
    """
    return np.abs(a) if np.iscomplexobj(a) else a


def _fourier_shift(img: np.ndarray, shift: Tuple[float, float]) -> np.ndarray:
    """Apply sub-pixel translation using Fourier-domain phase multiplication.
    
    This function performs exact sub-pixel shifting by multiplying the Fourier
    transform with a linear phase ramp. Preserves the input data type (real/complex).
    
    Args:
        img: Input 2D image array (real or complex)
        shift: Translation as (dy, dx) in pixels
        
    Returns:
        Shifted image with same dtype as input
        
    Note:
        Positive shifts move the image content in the positive direction.
        For example, shift=(1, 0) moves content down by 1 pixel.
    """
    dy, dx = shift
    ny, nx = img.shape
    ky = fftfreq(ny).reshape(-1, 1)
    kx = fftfreq(nx).reshape(1, -1)
    phase = np.exp(-2j * np.pi * (dy * ky + dx * kx))
    out = ifft2(fft2(img) * phase)
    return out.real if np.isrealobj(img) else out


def find_translation_offset(
    image: np.ndarray,
    reference: np.ndarray,
    upsample_factor: int = 50,
) -> Tuple[float, float]:
    """Find translational offset between two images using phase cross-correlation.
    
    This function detects the translational misalignment between an image and reference
    using scikit-image's phase_cross_correlation with sub-pixel precision. Both real
    and complex images are supported via magnitude extraction.
    
    Args:
        image: Moving image to be aligned (2D array, real or complex)
        reference: Fixed reference image (2D array, real or complex)
        upsample_factor: Upsampling factor for sub-pixel precision (default: 50)
                        Higher values give better precision but cost more computation
    
    Returns:
        Translation offset as (dy, dx) tuple in pixels.
        Apply this offset TO the `image` to align it with `reference`.
        
    Raises:
        ValueError: If inputs are not 2D or have mismatched shapes
        
    Example:
        >>> offset = find_translation_offset(reconstruction, ground_truth)
        >>> print(f"Reconstruction is offset by {offset} pixels")
        >>> # offset = (-1.2, 0.8) means reconstruction is 1.2 pixels up, 0.8 pixels right
    
    Note:
        - Positive dy means image content should move down to align
        - Positive dx means image content should move right to align
        - Sub-pixel precision achieved through upsampled FFT correlation
    """
    if image.ndim != 2 or reference.ndim != 2:
        raise ValueError("Both inputs must be 2-D.")
    if image.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {image.shape} vs {reference.shape}")

    shift, _, _ = phase_cross_correlation(
        _mag(reference).astype(np.float32),
        _mag(image).astype(np.float32),
        upsample_factor=upsample_factor,
    )
    return float(shift[0]), float(shift[1])


def apply_shift_and_crop(
    image: np.ndarray,
    reference: np.ndarray,
    offset: Tuple[float, float],
    border_crop: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply translation offset and crop borders to eliminate wrap-around artifacts.
    
    This function applies the detected offset to align the image with the reference,
    then crops both images equally to remove Fourier wrap-around artifacts that can
    occur at image borders during sub-pixel shifting.
    
    Args:
        image: Image to be shifted and cropped (2D array, real or complex)
        reference: Reference image to be cropped identically (2D array, real or complex)
        offset: Translation offset as (dy, dx) in pixels
        border_crop: Number of pixels to crop from each edge (default: 2)
                    Helps eliminate wrap-around artifacts from Fourier shifting
    
    Returns:
        Tuple of (shifted_cropped_image, cropped_reference) with identical shapes
        
    Raises:
        ValueError: If border_crop is negative or too large for the image size
        
    Example:
        >>> offset = find_translation_offset(recon, gt)
        >>> aligned_recon, aligned_gt = apply_shift_and_crop(recon, gt, offset)
        >>> # Now both images are aligned and ready for metric calculation
        
    Note:
        The border cropping is essential when using Fourier-domain shifting as it
        eliminates periodic boundary artifacts that would contaminate evaluation metrics.
    """
    if border_crop < 0:
        raise ValueError("border_crop must be non-negative")

    shifted = _fourier_shift(image, offset)
    y0, y1 = border_crop, image.shape[0] - border_crop
    x0, x1 = border_crop, image.shape[1] - border_crop
    if y0 >= y1 or x0 >= x1:
        raise ValueError("border_crop too large")

    return shifted[y0:y1, x0:x1], reference[y0:y1, x0:x1]


def register_and_align(
    image: np.ndarray,
    reference: np.ndarray,
    upsample_factor: int = 50,
    border_crop: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Complete registration workflow: detect offset, apply shift, and crop borders.
    
    This convenience function combines offset detection and alignment correction in a
    single call. It's the recommended entry point for most registration tasks.
    
    Args:
        image: Moving image to be aligned (2D array, real or complex)
        reference: Fixed reference image (2D array, real or complex)  
        upsample_factor: Upsampling factor for sub-pixel precision (default: 50)
        border_crop: Border pixels to crop after shifting (default: 2)
    
    Returns:
        Tuple of (aligned_image, cropped_reference) ready for metric evaluation
        
    Example:
        >>> # Complete registration in one step
        >>> aligned_recon, aligned_gt = register_and_align(reconstruction, ground_truth)
        >>> mae = np.mean(np.abs(aligned_recon - aligned_gt))
        >>> print(f"Aligned MAE: {mae:.6f}")
        
    Note:
        This function is equivalent to:
        ```python
        offset = find_translation_offset(image, reference, upsample_factor)
        return apply_shift_and_crop(image, reference, offset, border_crop)
        ```
    """
    offset = find_translation_offset(image, reference, upsample_factor)
    return apply_shift_and_crop(image, reference, offset, border_crop)