"""Display utilities for ptychographic reconstruction visualization and debugging.

Lightweight matplotlib-based visualization for the PtychoPINN physics-informed neural
network system. Serves as quick display utility for reconstruction results, diffraction
patterns, and data during development workflows.

Primary Consumers: Interactive development, debugging workflows, notebook analysis
Integration Points: Visualizes outputs from model.py, evaluation.py, data pipeline modules

Public Interface:
    `display_imgs(x, y=None, log=False, cbar=False, figsize=(10, 2), **kwargs)`
        - Display array sequences as horizontal subplot grids with optional labels
        - Input: x: shape (n_images, height, width); y: optional classification labels
"""
import numpy as np
import matplotlib.pyplot as plt

def display_imgs(x, y=None, log = False, cbar = False, figsize=(10, 2), **kwargs):
  if not isinstance(x, (np.ndarray, np.generic)):
    x = np.array(x)
  #plt.ioff()
  n = x.shape[0]
  fig, axs = plt.subplots(1, n, figsize = figsize)
  if y is not None:
    fig.suptitle(np.argmax(y, axis=1))
  for i in range(n):
    if log:
        axs.flat[i].imshow(np.log(.01 + x[i].squeeze()), interpolation='none', cmap='jet', **kwargs)
    else:
        axs.flat[i].imshow((x[i].squeeze()), interpolation='none', cmap='jet', **kwargs)
    axs.flat[i].axis('off')
  if cbar:
    plt.colorbar()
  plt.show()
  plt.close()
  plt.ion()
