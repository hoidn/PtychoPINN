# ptycho_simulation.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Union, Tuple, Dict
from ptycho.loader import RawData
from ptycho import tf_helper as hh
from ptycho import probe
from ptycho import baselines as bl

def load_probe_object(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load object and probe guesses from a .npz file.

    Args:
        file_path (str): Path to the .npz file containing objectGuess and probeGuess.

    Returns:
        tuple: A tuple containing (objectGuess, probeGuess)

    Raises:
        ValueError: If required data is missing from the .npz file or if data is invalid.
        RuntimeError: If an error occurs during file loading.
    """
    try:
        with np.load(file_path) as data:
            if 'objectGuess' not in data or 'probeGuess' not in data:
                raise ValueError("The .npz file must contain 'objectGuess' and 'probeGuess'")
            
            objectGuess = data['objectGuess']
            probeGuess = data['probeGuess']

        # Validate extracted data
        if objectGuess.ndim != 2 or probeGuess.ndim != 2:
            raise ValueError("objectGuess and probeGuess must be 2D arrays")
        if not np.iscomplexobj(objectGuess) or not np.iscomplexobj(probeGuess):
            raise ValueError("objectGuess and probeGuess must be complex-valued")

        return objectGuess, probeGuess

    except Exception as e:
        raise RuntimeError(f"Error loading data from {file_path}: {str(e)}")

from ptycho.misc import memoize_simulated_data

#@memoize_simulated_data
def generate_simulated_data(objectGuess: np.ndarray, probeGuess: np.ndarray, nimages: int, buffer: float, random_seed: int = None, return_patches: bool = True) -> Union[RawData, Tuple[RawData, np.ndarray]]:
    """
    Generate simulated ptychography data using random scan positions.

    Args:
        objectGuess (np.ndarray): Complex-valued 2D array representing the object.
        probeGuess (np.ndarray): Complex-valued 2D array representing the probe.
        nimages (int): Number of scan positions to generate.
        buffer (float): Border size to avoid when generating coordinates.
        random_seed (int, optional): Seed for random number generation. If None, uses system time.
        return_patches (bool): If True, return ground truth patches along with simulated data.

    Returns:
        RawData or tuple: A RawData instance containing the simulated ptychography data,
                          or a tuple of (RawData, ground_truth_patches) if return_patches is True.

    Raises:
        ValueError: If input parameters are invalid.
        RuntimeError: If an error occurs during simulation.
    """
    # Input validation
    if objectGuess.ndim != 2 or probeGuess.ndim != 2:
        raise ValueError("objectGuess and probeGuess must be 2D arrays")
    if not np.iscomplexobj(objectGuess) or not np.iscomplexobj(probeGuess):
        raise ValueError("objectGuess and probeGuess must be complex-valued")
    if nimages <= 0 or buffer < 0:
        raise ValueError("nimages must be positive and buffer must be non-negative")

    # Get object dimensions
    height, width = objectGuess.shape

    # Ensure buffer doesn't exceed image dimensions
    buffer = min(buffer, min(height, width) / 2 - 1)

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate random coordinates (floats)
    xcoords = np.random.uniform(buffer, width - buffer, nimages)
    ycoords = np.random.uniform(buffer, height - buffer, nimages)

    # Create scan_index
    scan_index = np.zeros(nimages, dtype=int)

    # Generate simulated data
    return RawData.from_simulation(xcoords, ycoords, probeGuess, objectGuess, scan_index, return_patches=return_patches)

def simulate_from_npz(file_path: str, nimages: int, buffer: float = None, random_seed: int = None, return_patches: bool = True) -> Union[RawData, Tuple[RawData, np.ndarray]]:
    """
    Load object and probe guesses from a .npz file and generate simulated ptychography data.

    Args:
        file_path (str): Path to the .npz file containing objectGuess and probeGuess.
        nimages (int): Number of scan positions to generate.
        buffer (float, optional): Border size to avoid when generating coordinates. 
                                  If None, defaults to 35% of the smaller dimension of objectGuess.
        random_seed (int, optional): Seed for random number generation. If None, uses system time.
        return_patches (bool): If True, return ground truth patches along with simulated data.

    Returns:
        RawData or tuple: A RawData instance containing the simulated ptychography data,
                          or a tuple of (RawData, ground_truth_patches) if return_patches is True.

    Raises:
        ValueError: If input parameters are invalid.
        RuntimeError: If an error occurs during simulation or file loading.
    """
    # Load guesses from file
    objectGuess, probeGuess = load_probe_object(file_path)

    # Set default buffer if not provided
    if buffer is None:
        buffer = min(objectGuess.shape) * 0.35  # 35% of the smaller dimension

    # Generate simulated data
    return generate_simulated_data(objectGuess, probeGuess, nimages, buffer, random_seed, return_patches=return_patches)

def plot_complex_image(ax: plt.Axes, data: np.ndarray, title: str) -> None:
    """Helper function to plot complex-valued images."""
    im = ax.imshow(np.abs(data), cmap='viridis')
    ax.set_title(f"{title} (Magnitude)")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax_phase = divider.append_axes("bottom", size="100%", pad=0.2, sharex=ax)
    im_phase = ax_phase.imshow(np.angle(data), cmap='hsv')
    ax_phase.set_title(f"{title} (Phase)")
    cax_phase = divider.append_axes("bottom", size="5%", pad=0.5)
    plt.colorbar(im_phase, cax=cax_phase, orientation="horizontal")

def visualize_simulated_data(data: Dict[str, np.ndarray], output_dir: str) -> None:
    """
    Visualize the simulated ptychography data and save all plots in a single image file.

    Args:
        data (dict): Dictionary containing the loaded simulated data.
        output_dir (str): Directory to save the output plot.
    """
    import os

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(24, 30))
    gs = fig.add_gridspec(5, 3, height_ratios=[1, 0.2, 1, 0.2, 1])

    # Plot probe guess
    ax_probe = fig.add_subplot(gs[0, 0])
    plot_complex_image(ax_probe, data['probe_guess'], "Probe Guess")

    # Plot object guess
    ax_object = fig.add_subplot(gs[0, 1])
    plot_complex_image(ax_object, data['object'], "Object Guess")

    # Plot scan positions
    ax_scan = fig.add_subplot(gs[0, 2])
    ax_scan.scatter(data['x_coordinates'], data['y_coordinates'], alpha=0.5)
    ax_scan.set_title("Scan Positions")
    ax_scan.set_xlabel("X Coordinate")
    ax_scan.set_ylabel("Y Coordinate")
    ax_scan.set_aspect('equal')

    # Add title for diffraction patterns
    fig.text(0.5, 0.62, "Sample Diffraction Patterns", ha='center', va='center', fontsize=16)

    # Plot a sample of diffraction patterns
    for i in range(3):
        if i < min(3, data['diffraction_patterns'].shape[0]):
            ax = fig.add_subplot(gs[2, i])
            im = ax.imshow(np.log(data['diffraction_patterns'][i]), cmap='viridis')
            ax.set_title(f"Pattern {i}")
            plt.colorbar(im, ax=ax)

    # Add title for ground truth patches
    fig.text(0.5, 0.22, "Sample Ground Truth Patches", ha='center', va='center', fontsize=16)

    # Plot ground truth patches
    for i in range(3):
        if i < min(3, data['ground_truth_patches'].shape[0]):
            ax = fig.add_subplot(gs[4, i])
            plot_complex_image(ax, data['ground_truth_patches'][i], f"Patch {i}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "simulated_data_visualization.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"All plots have been saved to: {os.path.join(output_dir, 'simulated_data_visualization.png')}")

def plot_random_groups(tmp: RawData, K: int, figsize: Tuple[int, int] = (15, 5), seed: int = None) -> None:
    """
    Plot a random selection of K groups of (diffraction image, Y amplitude, Y phase) from a RawData object.

    Args:
        tmp (RawData): The RawData object containing the ptychography data.
        K (int): Number of groups to plot.
        figsize (tuple): Figure size for each group plot. Default is (15, 5).
        seed (int): Random seed for reproducibility. Default is None.

    Raises:
        ValueError: If K is greater than the number of available diffraction patterns.
    """
    if K > tmp.diff3d.shape[0]:
        raise ValueError(f"K ({K}) cannot be greater than the number of diffraction patterns ({tmp.diff3d.shape[0]})")

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Randomly select K indices
    indices = np.random.choice(tmp.diff3d.shape[0], K, replace=False)

    for idx in indices:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f"Group {idx}")

        # Plot diffraction image (log scale for better visibility)
        diff_img = axes[0].imshow(np.log1p(1 + 100 * tmp.diff3d[idx]), cmap='jet')
        axes[0].set_title("Diffraction (log scale)")
        plt.colorbar(diff_img, ax=axes[0])

        # Plot Y amplitude
        amp_img = axes[1].imshow(np.abs(tmp.Y[idx]), cmap='viridis')
        axes[1].set_title("Y Amplitude")
        plt.colorbar(amp_img, ax=axes[1])

        # Plot Y phase
        phase_img = axes[2].imshow(np.angle(tmp.Y[idx]), cmap='twilight')
        axes[2].set_title("Y Phase")
        plt.colorbar(phase_img, ax=axes[2])

        # Remove axis ticks for cleaner look
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.show()

def compare_reconstructions(obj_tensor_full: np.ndarray, global_offsets: np.ndarray, ground_truth: np.ndarray, ptychonn_tensor: np.ndarray) -> None:
    """
    Compare the reconstructed object with the ground truth and PtychoNN prediction.

    Args:
        obj_tensor_full (np.ndarray): Full reconstructed object tensor.
        global_offsets (np.ndarray): Global offsets for positioning.
        ground_truth (np.ndarray): Ground truth object.
        ptychonn_tensor (np.ndarray): PtychoNN predicted object tensor.
    """
    from ptycho import nbutils
    irange = int(np.max(global_offsets[:, 0, 1, 0]) - np.min(global_offsets[:, 0, 1, 0]))
    trimmed_ground_truth = hh.trim_reconstruction(ground_truth[None, ..., None], irange)[0, :, :, 0]
    
    nbutils.compare(obj_tensor_full, global_offsets, trimmed_ground_truth, ptychonn_tensor=ptychonn_tensor)

# Add any additional helper functions or classes as needed

