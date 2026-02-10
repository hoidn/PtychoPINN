"""Non-grid ptychography simulation module for coordinate-based workflows.

This module implements the modern coordinate-based approach to ptychography simulation,
providing flexibility over traditional grid-based methods. It enables simulation of
arbitrary scan patterns with random or structured coordinate positioning, making it
ideal for realistic experimental conditions and advanced reconstruction algorithms.

Architecture Overview
--------------------
The module serves as a bridge between the modern configuration system and legacy
simulation components, providing a clean interface while maintaining compatibility:

* **Modern Interface**: Uses TrainingConfig dataclasses for consistent parameter management
* **Legacy Adapter**: Safely handles global state manipulation for backward compatibility  
* **Coordinate-Based**: Generates arbitrary scan positions vs fixed grid patterns
* **Simulation Pipeline**: Complete workflow from NPZ data to simulated measurements

Key Concepts
-----------
**Non-Grid vs Grid-Based Approaches:**

* **Grid-Based (Legacy)**: Fixed, regular grid of scan positions with uniform spacing
* **Non-Grid (Modern)**: Arbitrary coordinate positioning enabling:
  - Random scan patterns (more realistic)
  - Non-uniform sampling density
  - Irregular geometric arrangements
  - Better convergence properties for some algorithms

**Coordinate Workflow:**
1. Load object/probe from NPZ files
2. Generate random scan coordinates within object bounds
3. Simulate diffraction patterns at each position
4. Return RawData container with measurements and ground truth

Core Components
--------------
* `generate_simulated_data()`: Main simulation function with modern config interface
* `simulate_from_npz()`: Complete workflow from file to simulated data
* `load_probe_object()`: NPZ data loading with validation
* `visualize_simulated_data()`: Comprehensive visualization of simulation results
* `plot_random_groups()`: Debug visualization for diffraction/reconstruction pairs

Integration Points
-----------------
* **Configuration**: Uses TrainingConfig for all simulation parameters
* **Data Pipeline**: Produces RawData objects compatible with training workflows
* **Baselines**: Integrates with baseline reconstruction methods for comparison
* **Visualization**: Provides debug and analysis tools for simulation validation

Example Usage
------------
Basic simulation from NPZ file:

    >>> from ptycho.config.config import TrainingConfig, ModelConfig
    >>> from ptycho.nongrid_simulation import simulate_from_npz
    >>> 
    >>> # Configure simulation parameters
    >>> model_config = ModelConfig(N=64, gridsize=2)
    >>> training_config = TrainingConfig(
    ...     model=model_config,
    ...     n_images=2000
    ... )
    >>> 
    >>> # Simulate non-grid data from experimental object/probe
    >>> raw_data, ground_truth = simulate_from_npz(
    ...     config=training_config,
    ...     file_path="datasets/fly/fly001_transposed.npz",
    ...     buffer=20.0  # Minimum distance from object edges
    ... )
    >>> 
    >>> print(f"Generated {raw_data.diff3d.shape[0]} diffraction patterns")
    >>> print(f"Scan coordinates: {raw_data.xcoords.shape}")

Advanced simulation with custom parameters:

    >>> # Direct simulation with loaded data
    >>> from ptycho.nongrid_simulation import load_probe_object, generate_simulated_data
    >>> 
    >>> obj, probe = load_probe_object("my_sample.npz")
    >>> raw_data = generate_simulated_data(
    ...     config=training_config,
    ...     objectGuess=obj,
    ...     probeGuess=probe,
    ...     buffer=15.0,
    ...     return_patches=False
    ... )
    >>> 
    >>> # Visualize results
    >>> from ptycho.nongrid_simulation import visualize_simulated_data
    >>> viz_data = {
    ...     'probe_guess': probe,
    ...     'object': obj,
    ...     'x_coordinates': raw_data.xcoords,
    ...     'y_coordinates': raw_data.ycoords,
    ...     'diffraction_patterns': raw_data.diff3d,
    ...     'ground_truth_patches': raw_data.Y
    ... }
    >>> visualize_simulated_data(viz_data, "simulation_output/")

Notes
-----
This module includes legacy compatibility adapters that temporarily modify global
state during simulation calls. This is necessary for integration with existing
simulation components but is isolated to internal functions to maintain a clean
public interface.

The coordinate-based approach provides more realistic simulation conditions compared
to grid-based methods, leading to better training data for neural network models
and more robust reconstruction algorithms.
"""

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

    # Use n_groups (modern) with fallback to n_images (deprecated) for compatibility
    n_positions = config.n_groups if config.n_groups is not None else config.n_images
    if n_positions is None:
        raise ValueError("Either n_groups or n_images must be specified in config")

    xcoords = np.random.uniform(buffer, width - buffer, n_positions)
    ycoords = np.random.uniform(buffer, height - buffer, n_positions)
    scan_index = np.zeros(n_positions, dtype=int)

    # This is the non-conforming part: it manipulates global state.
    # It sets N, gridsize, and nphotons for the duration of the call to from_simulation.
    original_N = p.get('N')
    original_gridsize = p.get('gridsize')
    original_nphotons = p.get('nphotons')
    try:
        # Set parameters to match the modern config
        p.set('N', probeGuess.shape[0])
        p.set('gridsize', config.model.gridsize)
        p.set('nphotons', config.nphotons)  # Critical: Set nphotons for proper scaling
        raw_data = RawData.from_simulation(xcoords, ycoords, probeGuess, objectGuess, scan_index)
    finally:
        # Ensure global state is restored
        p.set('N', original_N)
        p.set('gridsize', original_gridsize)
        p.set('nphotons', original_nphotons)
    
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
