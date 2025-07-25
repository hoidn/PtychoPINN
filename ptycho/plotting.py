"""Interactive and static plotting utilities for ptychographic data visualization.

This module provides advanced plotting capabilities for the PtychoPINN physics-informed
neural network system with interactive Jupyter visualization and standardized static
plot generation with automatic layout management.

Primary Consumers: Jupyter notebooks, research workflows, publication-quality figures
Integration Points: Visualizes data from model.py, evaluation.py, reconstruction workflows
Key Capabilities:
    - **Interactive Visualization:** `ishow_imgs()` with widget controls for image series
    - **Plot Standardization:** `@plotting_function` decorator for consistent figure creation
    - **Layout Management:** Automated subfigure composition with flexible grid layouts

Public Interface:
    `ishow_imgs(*arrays, labels=None, styles=None, log=False, height='550px')`
        - Interactive widget-based viewer for image series with navigation controls
    `@plotting_function` decorator - standardizes plot creation with save/display options
"""

import matplotlib.pyplot as plt
from ipywidgets import interactive

def ishow_imgs(*arrs_list, styles = None, labels = None,
              log = False, height = '550px',
              nested_label_callback = None):
    """
    Plot a series of curves interactively.
    """
    plt.rcParams["figure.figsize"]=(12, 9)
    #labels = [label1, label2]
    if labels is None:
        labels = [''] * len(arrs_list)
    def f(i):
        for j, patterns in enumerate(arrs_list):
            if styles is not None:
                extra_args = (styles[j],)
            else:
                extra_args = ()
            try:
                for k in range(len(patterns[i])):
                    len(patterns[i][k]) # TODO hack
                    if nested_label_callback is not None:
                        label = nested_label_callback(patterns[i], k)
                    else:
                        label = k
                    plt.imshow(patterns[i][k], *extra_args, label = label)
            except: # TODO except what?
                if j < 2:
                    plt.imshow(patterns[i], label = labels[j])
                else:
                    plt.imshow(patterns[i], *extra_args)

    interactive_plot = interactive(f, i=(0, len(arrs_list[0]) - 1), step = 1)
    output = interactive_plot.children[-1]
    output.layout.height = height
    return interactive_plot

# Implementing actual plotting functions and the decorator for visual output

import matplotlib.pyplot as plt
import numpy as np

from functools import wraps

def plotting_function(func):
    @wraps(func)
    def wrapper(layout=(1, 1), display: bool = False, save: bool = False, save_path: str = "", *args, **kwargs):
        standalone = 'ax' not in kwargs or kwargs['ax'] is None
        if standalone:
            fig, axs = plt.subplots(layout[0], layout[1], figsize=(layout[1]*3, layout[0]*3))
            if layout == (1, 1):
                axs = np.array([axs])
            else:
                axs = axs.reshape(layout[0], layout[1])
            kwargs['ax'] = axs
        result = func(*args, **kwargs)
        if standalone:
            plt.tight_layout()
            if save:
                plt.savefig(save_path if save_path else "/mnt/data/plot.png")
            if display:
                plt.show()
        return result
    return wrapper

@plotting_function
def plot_subfigure(ax=None, title: str = "Subfigure", *args, **kwargs):
    rows, cols = ax.shape if isinstance(ax, np.ndarray) else (1, 1)
    for i in range(rows):
        for j in range(cols):
            ax[i, j].plot([1, 2, 3], [1, 2, 3])
            ax[i, j].set_title(f"{title} {i+1},{j+1}")

def compose_and_save_figure():
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # Adjusting the plotting function to accept individual Axes
    plot_subfigure(ax=np.array([[ax1]]), title="Plot 1")
    plot_subfigure(ax=np.array([[ax2]]), title="Plot 2")
    plot_subfigure(ax=np.array([[ax3]]), layout=(1, 1), title="Plot 3")

    plt.tight_layout()
    save_path = "/mnt/data/composed_figure.png"
    plt.savefig(save_path)
    plt.show()
## To visually check, we'll call the plot_subfigure function directly with a layout parameter for standalone mode
#plot_subfigure(layout=(2, 2), display=True, save=True, title="Standalone Plot", save_path="/mnt/data/standalone_plot.png")

