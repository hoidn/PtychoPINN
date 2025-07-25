"""Generate synthetic objects with diagonal line patterns for ptychographic simulation.

This module provides functions for creating test objects containing randomly positioned
vertical, horizontal, and diagonal line patterns, primarily used by ptycho.diffsim.

Usage Example:
    ```python
    from ptycho.datagen.diagonals import mk_diags
    diagonal_pattern = mk_diags(N=64, sigma=0.75)
    ```
"""
import numpy as np

def draw_lines(shape, num):
    num_vertical = num_horizontal = num_diagonal = num
    # Create a 2D NumPy array with zeros
    arr = np.zeros(shape)
    n, m = shape

    # Draw vertical lines
    for i in range(num_vertical):
        x = np.random.randint(0, shape[1])  # Random x coordinate
        arr[:, x] = 1

    # Draw horizontal lines
    for i in range(num_horizontal):
        y = np.random.randint(0, shape[0])  # Random y coordinate
        arr[y, :] = 1

    # Draw diagonal lines
    for i in range(num_diagonal):
        x = np.random.randint(0, shape[1])  # Random x coordinate
        y = np.random.randint(0, shape[0])  # Random y coordinate
        off = min(x, y)
        x = x - off
        y = y - off
        ix = np.arange(x, n - y)
        iy = np.arange(y, m - x)
        arr[ix, iy] = 1
        arr[ix, -iy] = 1

    return arr


from scipy.ndimage import gaussian_filter as gf
def mk_diags(N, sigma = .75):
    img = draw_lines((N, N), 40)
    img = gf(img, sigma)
    img = img + gf(img, 10 * sigma) * 5
    img = img[:, :, None]
    return img
