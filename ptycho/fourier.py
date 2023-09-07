import pandas as pd
import numpy as np

from scipy.fft import fft, fftfreq, ifft, fft2, ifft2, ifftshift
from scipy.signal import blackman
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
    L = signal.gaussian(len(y), std = len(y) / (size * np.pi**2), sym = sym)
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

