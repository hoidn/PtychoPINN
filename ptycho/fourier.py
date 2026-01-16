"""Fourier transform utilities and frequency domain processing for ptychography.

This module provides essential Fourier domain operations and filtering utilities
used throughout the PtychoPINN ptychographic reconstruction pipeline. It serves
as the mathematical foundation for frequency domain analysis, probe processing,
and signal filtering operations required for coherent diffraction imaging.

Core Functions:
    Gaussian Filters:
        - lowpass_g(): Gaussian lowpass filtering for probe initialization
        - highpass_g(): Gaussian highpass filtering for frequency analysis
        - bandpass_g(): Combined lowpass/highpass for frequency band selection
    
    Frequency Domain Clipping:
        - clip_high(): Remove high-frequency components with fractional control
        - clip_low(): Remove low-frequency components with masking
        - clip_low_window(): Generate frequency domain windows
    
    Fourier Operations:
        - if_mag(): Inverse FFT with magnitude processing and phase control
        - power(): Complex amplitude to power spectrum conversion
        - mag(): Complex amplitude to magnitude conversion
    
    Analysis Utilities:
        - lorenz(): Lorentzian function for spectral line fitting
        - plot_df(): Pandas DataFrame plotting utility

Architecture Integration:
    - **Probe Module (ptycho/probe.py)**: Uses lowpass_g() for creating default
      disk-shaped scanning probes with controlled frequency content
    - **Physics Simulation**: Provides frequency domain tools for realistic
      modeling of coherent diffraction patterns
    - **Signal Processing**: Core mathematical utilities for ptychographic
      reconstruction algorithms

Mathematical Context:
    The functions implement frequency domain operations essential for ptychography:
    - Gaussian filters model realistic probe shapes and experimental conditions
    - Frequency clipping operations simulate detector limitations and noise
    - Complex amplitude processing handles the phase-sensitive nature of coherent imaging

Example:
    # Create a Gaussian lowpass filter for probe initialization
    >>> import numpy as np
    >>> from ptycho import fourier as f
    >>> N = 64
    >>> scale = 0.4
    >>> filter_1d = f.lowpass_g(scale, np.ones(N), sym=True)
    >>> probe_2d = f.gf(np.outer(filter_1d, filter_1d) > 0.5, sigma=1)
    
    # Process complex amplitude data
    >>> complex_data = np.random.random((128,)) + 1j * np.random.random((128,))
    >>> power_spectrum = f.power(complex_data)
    >>> magnitude = f.mag(complex_data)
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

