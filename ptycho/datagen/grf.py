# credit https://github.com/PabloVD/MapGenerator

import matplotlib.pyplot as plt
import numpy as np
import powerbox as pbox
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

#--- Parameters for GRF---#


# Number of bins per dimension
boxsize = 100#(max(xx.shape) + 1) // 2#xx.shape[0] // 2
# Number of bins per dimension in the high resolution  box

# Define power spectrum as a power law with an spectral index indexlaw
# With lower the spectral indexes, small structures are removed
def powerspec(k,indexlaw):
    return k**indexlaw

# Filter the field with a gaussian window
def smooth_field(field,sigmagauss,gridsize=boxsize):
    ny, nx = field.shape
    y = np.linspace(0.0, float(ny - 1), num=ny)
    x = np.linspace(0.0, float(nx - 1), num=nx)

    # `interp2d` was removed in SciPy 1.14; use regular-grid interpolation.
    interp = RegularGridInterpolator((y, x), field, method="linear", bounds_error=False, fill_value=None)

    qy = np.linspace(0.0, float(ny - 1), num=gridsize)
    qx = np.linspace(0.0, float(nx - 1), num=gridsize)
    qyy, qxx = np.meshgrid(qy, qx, indexing="ij")
    points = np.stack([qyy, qxx], axis=-1)
    resized = interp(points)

    # Filtering
    smooth = ndimage.gaussian_filter(resized, sigmagauss)
    return smooth

# Remove regions below sea level
def mainland(field,threshold):
    for i, row in enumerate(field):
        for j, el in enumerate(row):
            if el<threshold:   field[i,j]=0.
    return field

# Normalize the values of the field between 0 and 1
def normalize_field(field):
    min, max = np.amin(field), np.amax(field)
    newfield = (field-min)/(max-min)
    return newfield

# Generate a map of islands applying different processes:
# 1. Generate a random gaussian field given a power spectrum
# 2. Normalize the field between 0 and 1
# 3. Smooth the field with a gaussian filter
# 4. Retain only the mainland above a certain threshold
def generate_map(indexlaw,sigma,threshold, boxsize):
    # Number of bins per dimension in the high resolution  box
    highboxsize = 2*boxsize
    field = pbox.powerbox.PowerBox(boxsize, lambda k: powerspec(k,indexlaw), dim=2, boxlength=100.).delta_x()
    field = normalize_field(field)
    field = smooth_field(field,sigma,gridsize=highboxsize)
    return field

def mk_grf(N):
    assert not N % 2
    boxsize = N // 2
    # Threshold for the sea level
    threshold = 0.4
    # Sigma for the gaussian smoothing
    sigma = 1
    # Spectral index for the power spectrum
    indexlaw = -.4
    res = np.zeros((N, N, 1))
    res[:, :, :] = generate_map(indexlaw, sigma, threshold, boxsize)[..., None]
    return res
