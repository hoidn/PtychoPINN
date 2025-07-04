# ptycho/nongrid_simulation.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Union, Tuple, Dict

# Import modern config and legacy params for the adapter
from ptycho.config.config import TrainingConfig
from ptycho import params as p

from ptycho.loader import RawData
from ptycho import tf_helper as hh
from ptycho import probe
from ptycho import baselines as bl


def load_probe_object(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load object and probe guesses from a .npz file.
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


def _generate_simulated_data_legacy_params(config: TrainingConfig, objectGuess: np.ndarray, probeGuess: np.ndarray, buffer: float, random_seed: int = None) -> RawData:
    """
    Internal legacy function that manipulates global state to generate data.
    This is a temporary workaround until raw_data.py and diffsim.py can be refactored.
    """
    height, width = objectGuess.shape
    buffer = min(buffer, min(height, width) / 2 - 1)

    if random_seed is not None:
        np.random.seed(random_seed)

    xcoords = np.random.uniform(buffer, width - buffer, config.n_images)
    ycoords = np.random.uniform(buffer, height - buffer, config.n_images)
    scan_index = np.zeros(config.n_images, dtype=int)

    # This is the non-conforming part: it manipulates global state.
    # It sets N and gridsize for the duration of the call to from_simulation.
    original_N = p.get('N')
    original_gridsize = p.get('gridsize')
    try:
        # Set N to match the probe and gridsize from the modern config
        p.set('N', probeGuess.shape[0])
        p.set('gridsize', config.model.gridsize)
        raw_data = RawData.from_simulation(xcoords, ycoords, probeGuess, objectGuess, scan_index)
    finally:
        # Ensure global state is restored
        p.set('N', original_N)
        p.set('gridsize', original_gridsize)
    
    return raw_data


def generate_simulated_data(config: TrainingConfig, objectGuess: np.ndarray, probeGuess: np.ndarray, buffer: float, return_patches: bool = True) -> Union[RawData, Tuple[RawData, np.ndarray]]:
    """
    CONFORMING: Generate simulated ptychography data using a configuration object.
    
    This function acts as an adapter, calling an internal legacy function
    but exposing a clean, modern interface.
    """
    # TODO: Get seed from config if it gets added there.
    raw_data = _generate_simulated_data_legacy_params(
        config=config,
        objectGuess=objectGuess,
        probeGuess=probeGuess,
        buffer=buffer,
        random_seed=42 
    )

    if return_patches:
        ground_truth_patches = raw_data.Y if hasattr(raw_data, 'Y') else None
        return raw_data, ground_truth_patches
    else:
        return raw_data


def simulate_from_npz(config: TrainingConfig, file_path: str, buffer: float = None, return_patches: bool = True) -> Union[RawData, Tuple[RawData, np.ndarray]]:
    """
    CONFORMING: Load object/probe and generate simulated data using a config object.
    """
    objectGuess, probeGuess = load_probe_object(file_path)

    if buffer is None:
        buffer = min(objectGuess.shape) * 0.35

    # Calls the new, conforming adapter function
    return generate_simulated_data(config, objectGuess, probeGuess, buffer, return_patches=return_patches)


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
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(24, 30))
    gs = fig.add_gridspec(5, 3, height_ratios=[1, 0.2, 1, 0.2, 1])
    ax_probe = fig.add_subplot(gs[0, 0])
    plot_complex_image(ax_probe, data['probe_guess'], "Probe Guess")
    ax_object = fig.add_subplot(gs[0, 1])
    plot_complex_image(ax_object, data['object'], "Object Guess")
    ax_scan = fig.add_subplot(gs[0, 2])
    ax_scan.scatter(data['x_coordinates'], data['y_coordinates'], alpha=0.5)
    ax_scan.set_title("Scan Positions")
    ax_scan.set_xlabel("X Coordinate")
    ax_scan.set_ylabel("Y Coordinate")
    ax_scan.set_aspect('equal')
    fig.text(0.5, 0.62, "Sample Diffraction Patterns", ha='center', va='center', fontsize=16)
    for i in range(3):
        if i < min(3, data['diffraction_patterns'].shape[0]):
            ax = fig.add_subplot(gs[2, i])
            im = ax.imshow(np.log(data['diffraction_patterns'][i] + 1e-9), cmap='viridis')
            ax.set_title(f"Pattern {i}")
            plt.colorbar(im, ax=ax)
    fig.text(0.5, 0.22, "Sample Ground Truth Patches", ha='center', va='center', fontsize=16)
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
    """
    if K > tmp.diff3d.shape[0]:
        raise ValueError(f"K ({K}) cannot be greater than the number of diffraction patterns ({tmp.diff3d.shape[0]})")
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.choice(tmp.diff3d.shape[0], K, replace=False)
    for idx in indices:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f"Group {idx}")
        diff_img = axes[0].imshow(np.log1p(1 + 100 * tmp.diff3d[idx]), cmap='jet')
        axes[0].set_title("Diffraction (log scale)")
        plt.colorbar(diff_img, ax=axes[0])
        amp_img = axes[1].imshow(np.abs(tmp.Y[idx]), cmap='viridis')
        axes[1].set_title("Y Amplitude")
        plt.colorbar(amp_img, ax=axes[1])
        phase_img = axes[2].imshow(np.angle(tmp.Y[idx]), cmap='twilight')
        axes[2].set_title("Y Phase")
        plt.colorbar(phase_img, ax=axes[2])
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.show()


def compare_reconstructions(obj_tensor_full: np.ndarray, global_offsets: np.ndarray, ground_truth: np.ndarray, ptychonn_tensor: np.ndarray) -> None:
    """
    Compare the reconstructed object with the ground truth and PtychoNN prediction.
    """
    from ptycho import nbutils
    irange = int(np.max(global_offsets[:, 0, 1, 0]) - np.min(global_offsets[:, 0, 1, 0]))
    trimmed_ground_truth = hh.trim_reconstruction(ground_truth[None, ..., None], irange)[0, :, :, 0]
    nbutils.compare(obj_tensor_full, global_offsets, trimmed_ground_truth, ptychonn_tensor=ptychonn_tensor)
