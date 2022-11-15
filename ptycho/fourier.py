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

from xrdc import featurization as feat

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

def spec_fft(patterns, i, pad = 1000, roll = 0, do_conv_window = False, do_window = True, log = False, dat = None):
    if dat is not None:
        pi = dat
    else:
        pi = patterns[i]
    if log:
        y = np.pad(np.log(pi + 1), pad, mode = 'edge')
    else:
        y = np.pad(pi, pad, mode = 'edge')
    y = np.roll(y, roll)
    # Number of sample points
    N = y.shape[0]
    w = blackman(N)
    #w = 1
    #yf = fft(y * w)
    if do_window:
        ywf = fft(y*w)
    else:
        ywf = fft(y)
    if do_conv_window:
        ywf = conv_window(ywf)
    return w, ywf

def power(arr):
    ampsq = arr * np.conjugate(arr)
    return np.real(ampsq)

def mag(x):
    return np.sqrt(power(x))

def lorenz(gamma, x, x0):
    return ( 1. / (np.pi * gamma)) * (gamma**2) / ((x - x0)**2 + gamma**2)

def do_rl(sig, window_width = 4, peak_width = 2, window_type = 'gaussian',
         bwindow = None, norm = False):
    if window_type == 'gaussian':
        gwindow = signal.gaussian(len(y), std = window_width)
        #gwindow = lorenz(peak_width, np.arange(len(sig)), len(sig) // 2)
        L = power(fft(gwindow))
        L /= L.max()
        H = 1 - L
    elif window_type == 'step':
        H = clip_low_window(sig, .001) * bwindow
    else:
        raise ValueError
    
    g = signal.gaussian(len(y), std = peak_width)
    gfft = fft(g)
    
    psf = mag(ifft(gfft * H))[:, None].T
    psf_1d = psf[:, 1275:1324]
    deconvolved_RL = restoration.richardson_lucy((sig[:, None].T) / (10 * sig.max()), psf_1d, iterations=120)
    if not norm:
        return deconvolved_RL[0]
    else:
        return deconvolved_RL[0] / deconvolved_RL[0].mean()
    
def conv_window(sig, mode = 'same'):
    tmp = np.real(np.sqrt(fft(window) * np.conjugate(fft(window))))
    return np.convolve(sig, tmp / tmp.max(), mode =mode)#if_mag

def filter_bg(patterns, i, smooth = 1.5, window_type = 'gaussian', blackman = True,
             deconvolve = False, invert = False, **kwargs):
    cutoff = 4
    window, ywf = spec_fft(patterns, i, 1000)
    if window_type == 'gaussian': #todo inversion
        sig = if_mag(patterns, highpass_g(cutoff, ywf) * ywf, **kwargs)
    elif window_type == 'step': # hard step
        clipped, mask = clip_low(ywf, .001, invert = invert)
        if blackman:
            if invert:
                window = 1 - window
            mask *= window
            sig = if_mag(patterns, clipped * window, **kwargs)
        else:
            sig = if_mag(patterns, clipped, **kwargs)
    else:
        raise ValueError
    if deconvolve:
        sig = do_rl(sig, cutoff, 2.2)
    sig = gf(sig, smooth)
    return sig[1000: -1000]
