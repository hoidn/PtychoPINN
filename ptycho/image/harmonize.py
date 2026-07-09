from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom


def resize_complex_to_shape(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize 2D complex data to target shape using cubic interpolation."""
    data = np.squeeze(np.asarray(arr))
    if data.ndim != 2:
        raise ValueError("resize_complex_to_shape expects 2D input after squeeze")
    if not np.iscomplexobj(data):
        data = data.astype(np.complex64)
    if data.shape == target_hw:
        return data.astype(np.complex64)

    zoom_factors = (target_hw[0] / data.shape[0], target_hw[1] / data.shape[1])
    real = zoom(data.real, zoom_factors, order=3)
    imag = zoom(data.imag, zoom_factors, order=3)
    return (real + 1j * imag).astype(np.complex64)

