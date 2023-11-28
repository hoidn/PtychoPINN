import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter as gf

def randones(N, pct = .1):
    """
    Return array whose entries are randomly either 0 or 1.
    """
    rows, cols = N, N

    # define the percentage of entries to increment
    pct = 0.1

    # create a 2D numpy array of all 0s
    arr = np.zeros((rows, cols))

    # determine the number of entries to increment
    num_entries = int(rows * cols * pct)

    # randomly select indices to increment with replacement
    indices = np.random.choice(rows * cols, num_entries)

    # increment the values at the selected indices by 1
    np.add.at(arr, np.unravel_index(indices, (rows, cols)), 1)

    # print the resulting array
    return arr


def mk_points(N, sigma = 1, pct = .15):
    img = randones(N, pct = pct)
    img = gf(img, sigma)
    img = img + gf(img, 10 * sigma) * 5
    img = img[:, :, None]
    return img
