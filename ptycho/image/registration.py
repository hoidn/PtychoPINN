"""
ptycho_registration.py
======================

Image-registration helpers for ptychographic reconstructions
-----------------------------------------------------------

Public API
----------
find_translation_offset(image, reference, upsample_factor=50) → (dy, dx)
apply_shift_and_crop(image, reference, offset, border_crop=2) → (img_crop, ref_crop)
register_and_align(image, reference, upsample_factor=50, border_crop=2)

Conventions
-----------
* Arrays are 2-D NumPy ndarrays (real or complex), shape (H, W).
* Returned offsets are floats (dy, dx) **in pixels**.  
  Positive dy ⇒ `image` is lower than `reference`.  
  Positive dx ⇒ `image` is to the right of `reference`.
* A fixed `border_crop=2` (default) trims 2 px from every edge after shifting
  to eliminate Fourier wrap-around artefacts.

Dependencies
------------
* Mandatory: NumPy, SciPy
* Optional : scikit-image (for `phase_cross_correlation`)
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# -----------------------------------------------------------------------------#
# Internal helpers
# -----------------------------------------------------------------------------#
def _tukey_window(shape: Tuple[int, int], alpha: float = 0.25) -> np.ndarray:
    """Return a 2-D Tukey window as float32."""
    from scipy.signal.windows import tukey
    wy = tukey(shape[0], alpha, sym=False)
    wx = tukey(shape[1], alpha, sym=False)
    return np.outer(wy, wx).astype(np.float32)


def _magnitude(arr: np.ndarray) -> np.ndarray:
    """Return magnitude for complex input, passthrough for real."""
    return np.abs(arr) if np.iscomplexobj(arr) else arr


def _prep_for_corr(arr: np.ndarray) -> np.ndarray:
    """Magnitude → mean-remove → Tukey taper."""
    arr_f = _magnitude(arr).astype(np.float32, copy=False)
    arr_f -= arr_f.mean(dtype=np.float64)
    return arr_f * _tukey_window(arr_f.shape)


def _fourier_shift(img: np.ndarray, shift: Tuple[float, float]) -> np.ndarray:
    """Exact sub-pixel circular shift using the Fourier shift theorem."""
    dy, dx = shift
    ny, nx = img.shape
    ky = fftfreq(ny).reshape(-1, 1)
    kx = fftfreq(nx).reshape(1, -1)
    phase = np.exp(-2j * np.pi * (dy * ky + dx * kx))
    shifted = ifft2(fft2(img) * phase)
    return shifted.real if np.isrealobj(img) else shifted


# -------------------- FFT phase-correlation fallback ------------------------- #
def _phase_corr_fft(a: np.ndarray, b: np.ndarray, upsample: int = 50) -> Tuple[float, float]:
    """
    Pure-NumPy phase correlation.

    Returns the sub-pixel (dy, dx) to apply to *b* so it matches *a*.
    """
    Fa, Fb = fft2(a), fft2(b)
    R = Fa * Fb.conj()
    R /= np.abs(R, where=(R != 0), out=np.ones_like(R))
    r = ifft2(R)
    peak = np.unravel_index(np.argmax(np.abs(r)), r.shape)

    # Coarse integer estimate
    mid = np.array(a.shape) // 2
    shift = np.array(peak, dtype=float)
    shift[shift > mid] -= np.array(a.shape)[shift > mid]

    if upsample <= 1:
        return float(shift[0]), float(shift[1])

    # Sub-pixel refinement: quadratic fit on 3×3 neighbourhood
    y0, x0 = peak
    nbr = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy = (y0 + dy) % a.shape[0]
            xx = (x0 + dx) % a.shape[1]
            nbr.append(((dy, dx), np.abs(r[yy, xx])))
    # Fit simple 2-D paraboloid
    dz = np.array([v for (_, v) in nbr])
    A = np.array([[1, dx, dy, dx*dx, dx*dy, dy*dy] for ((dy, dx), _) in nbr])
    coeff, *_ = np.linalg.lstsq(A, dz, rcond=None)
    a0, a1, a2, a3, a4, a5 = coeff
    denom = 4*a3*a5 - a4**2
    if denom != 0:
        sub_dx = (a4*a2 - 2*a5*a1) / denom
        sub_dy = (a4*a1 - 2*a3*a2) / denom
        shift += np.array([sub_dy, sub_dx])
    return float(shift[0]), float(shift[1])


# -----------------------------------------------------------------------------#
# Public API
# -----------------------------------------------------------------------------#
try:
    from skimage.registration import phase_cross_correlation

    def find_translation_offset(
        image: np.ndarray,
        reference: np.ndarray,
        upsample_factor: int = 50,
    ) -> Tuple[float, float]:
        """
        Detect sub-pixel translation between `image` and `reference`.

        Returns the shift (dy, dx) that must be **applied to `image`** to
        align it with `reference`.
        """
        if image.ndim != 2 or reference.ndim != 2:
            raise ValueError("Both inputs must be 2-D.")
        if image.shape != reference.shape:
            raise ValueError(f"Shape mismatch: {image.shape} vs {reference.shape}")

        shift, _, _ = phase_cross_correlation(
            _prep_for_corr(reference),
            _prep_for_corr(image),
            upsample_factor=upsample_factor,
        )
        return float(shift[0]), float(shift[1])

except ModuleNotFoundError:  # Fallback to NumPy implementation
    def find_translation_offset(
        image: np.ndarray,
        reference: np.ndarray,
        upsample_factor: int = 50,
    ) -> Tuple[float, float]:
        if image.ndim != 2 or reference.ndim != 2:
            raise ValueError("Both inputs must be 2-D.")
        if image.shape != reference.shape:
            raise ValueError(f"Shape mismatch: {image.shape} vs {reference.shape}")

        return _phase_corr_fft(
            _prep_for_corr(reference), _prep_for_corr(image), upsample=upsample_factor
        )


def apply_shift_and_crop(
    image: np.ndarray,
    reference: np.ndarray,
    offset: Tuple[float, float],
    border_crop: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shift `image` by `offset` (sub-pixel, circular) and return both arrays
    cropped by `border_crop` pixels on each edge.
    """
    if border_crop < 0:
        raise ValueError("border_crop must be non-negative.")

    shifted = _fourier_shift(image, offset)
    y0, y1 = border_crop, image.shape[0] - border_crop
    x0, x1 = border_crop, image.shape[1] - border_crop
    if y0 >= y1 or x0 >= x1:
        raise ValueError("border_crop too large for image size.")

    return shifted[y0:y1, x0:x1], reference[y0:y1, x0:x1]


def register_and_align(
    image: np.ndarray,
    reference: np.ndarray,
    upsample_factor: int = 50,
    border_crop: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper → returns **aligned, identically-sized crops**.

    Example
    -------
    >>> img_aligned, ref_crop = register_and_align(img, gt)
    """
    offset = find_translation_offset(image, reference, upsample_factor)
    return apply_shift_and_crop(image, reference, offset, border_crop)