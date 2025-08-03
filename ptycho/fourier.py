"""Signal processing utilities for frequency domain operations in ptychography.

This module provides foundational mathematical utilities for frequency domain 
analysis and signal conditioning. It implements Gaussian filtering, frequency 
clipping, and complex amplitude processing functions primarily used in data 
exploration, probe conditioning, and signal analysis workflows.

Architecture Role:
    Data Analysis → fourier.py (signal processing) → Conditioned Data
    
    This module provides mathematical building blocks for signal conditioning
    and frequency domain analysis, primarily used in interactive notebooks
    and experimental workflows.

Public Interface:
    `lowpass_g(size, y, sym=False)`
        - Purpose: Generate Gaussian lowpass filter windows.
        - Critical: Filter size controls frequency cutoff relative to array length.
        - Returns: Normalized Gaussian window for frequency domain filtering.
    
    `power(arr)` and `mag(arr)`
        - Purpose: Extract power and magnitude from complex amplitudes.
        - Critical: power() computes |arr|², mag() computes |arr|.
        - Returns: Real-valued power spectra or magnitude arrays.
    
    `clip_high(x, frac_zero)` and `clip_low(x, frac_zero)`
        - Purpose: Zero out frequency components with fractional control.
        - Critical: frac_zero specifies fraction of frequencies to remove.
        - Returns: Modified array with frequency masking applied.

Workflow Usage Example:
    ```python
    import numpy as np
    from ptycho import fourier
    
    # Generate Gaussian filter for probe conditioning
    N = 64
    dummy_array = np.ones(N)
    filter_window = fourier.lowpass_g(0.4, dummy_array, sym=True)
    
    # Analyze complex ptychographic data
    complex_wave = np.random.random((N,)) + 1j * np.random.random((N,))
    power_spectrum = fourier.power(complex_wave)
    magnitude = fourier.mag(complex_wave)
    
    # Apply frequency domain clipping
    clipped_data, mask = fourier.clip_low(complex_wave, frac_zero=0.1)
    ```

Architectural Notes:
- Provides scipy/numpy-based signal processing utilities.
- Stateless functions suitable for interactive analysis and notebooks.
- No dependencies on core PtychoPINN reconstruction pipeline.
- Primarily used for data exploration and experimental signal conditioning.
"""

import pandas as pd
import numpy as np

from scipy.fft import fft, fftfreq, ifft, fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
from scipy.signal import butter
from scipy import signal
from scipy.signal import convolve2d as conv2

from skimage import color, data, restoration
from scipy.ndimage import gaussian_filter as gf

def plot_df(*args):
    df = pd.DataFrame([p for p, _ in args]).T
    df.columns = [l for _, l in args ]
    return df.plot()

def lowpass_g(size, y, sym = False):
    from scipy.signal.windows import gaussian
    L = gaussian(len(y), std = len(y) / (size * np.pi**2), sym = sym)
    L /= L.max()
    return L

def highpass_g(size, y):
    return 1 - lowpass_g(size, y)

def bandpass_g(L, H, y):
    L = lowpass_g(L, y)
    H = highpass_g(H, y)
    return L * H

def clip_high(x, frac_zero):
    N = len(x)
    nz = int(frac_zero * N)
    x2  = x.copy()
    x2[(N - nz) // 2 : (N + nz) // 2] = 0
    #x2[(-nz) // 2:] = 0
    return x2

def clip_low(x, frac_zero, invert = False):
    N = len(x)
    nz = int(frac_zero * N)
    x2  = x.copy()
    mask = np.ones_like(x)
    mask[:( nz) // 2 ] = 0
    mask[(-nz) // 2:] = 0
    if invert:
        mask = 1 - mask
    x2 = x2 * mask

#     x2[:( nz) // 2 ] = 0
#     x2[(-nz) // 2:] = 0
    return x2, mask

def clip_low_window(x, frac_zero):
    N = len(x)
    nz = int(frac_zero * N)
    x2  = np.ones_like(x)
    x2[:( nz) // 2 ] = 0
    x2[(-nz) // 2:] = 0
    return x2

def if_mag(arr, phase = 0, truncate = False, toreal = 'psd', **kwargs):
    #print("arr shape", arr.shape)
    #trunc = len(arr) - unpadded_length
    phase = np.exp(1j * phase)
    tmp = ifft(arr)
    if toreal == 'psd':
        real = np.real(np.sqrt(np.conjugate(tmp) * tmp))
    elif toreal == 'real':
        real = np.real(tmp)
    else:
        raise ValueError
    if truncate:
        raise NotImplementedError
        #return real[trunc // 2: -trunc // 2]
    return real

def power(arr):
    ampsq = arr * np.conjugate(arr)
    return np.real(ampsq)

def mag(x):
    return np.sqrt(power(x))

def lorenz(gamma, x, x0):
    return ( 1. / (np.pi * gamma)) * (gamma**2) / ((x - x0)**2 + gamma**2)

