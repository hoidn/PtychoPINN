"""Grid-based data generation using mk_simdata.

Wraps the legacy mk_simdata() function with proper params.cfg setup
and provides a clean interface for generating training/test datasets.

References:
    - ptycho/diffsim.py::mk_simdata()
    - docs/DATA_GENERATION_GUIDE.md
"""
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ptycho import params as p
from ptycho.diffsim import mk_simdata


@dataclass
class GridDataset:
    """Container for grid-simulated data."""
    X: np.ndarray          # Diffraction patterns (batch, N, N, channels)
    Y_I: np.ndarray        # Ground truth amplitude
    Y_phi: np.ndarray      # Ground truth phase
    intensity_scale: float # Normalization factor
    norm_Y_I: np.ndarray   # Amplitude normalization
    YY_full: np.ndarray    # Full object (complex)
    coords: Any            # Coordinate information
    config: Dict[str, Any] # Configuration used


def setup_params_cfg(
    N: int,
    gridsize: int = 1,
    offset: int = 4,
    nphotons: float = 1e9,
    size: int = 500,
    data_source: str = 'lines',
    outer_offset_train: int = 12,
    outer_offset_test: int = 12,
    max_position_jitter: int = 3,
    sim_jitter_scale: float = 0.0,
    set_phi: bool = False,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Set up params.cfg for mk_simdata.

    This follows CONFIG-001: params.cfg must be set before calling legacy modules.

    Returns:
        Dictionary of configuration values (also sets params.cfg)
    """
    config = {
        'N': N,
        'gridsize': gridsize,
        'offset': offset,
        'nphotons': nphotons,
        'size': size,
        'data_source': data_source,
        'outer_offset_train': outer_offset_train,
        'outer_offset_test': outer_offset_test,
        'max_position_jitter': max_position_jitter,
        'sim_jitter_scale': sim_jitter_scale,
        'set_phi': set_phi,
        'batch_size': batch_size,
    }

    # Set all values in params.cfg
    for key, value in config.items():
        p.cfg[key] = value

    return config


def calculate_image_count(n: int, size: int, N: int, outer_offset: int, gridsize: int = 1) -> int:
    """
    Calculate the number of images produced by mk_simdata.

    Formula: n * ((size - bigN) // outer_offset + 1)^2
    where bigN = N * gridsize + (gridsize - 1) * offset
    """
    offset = 4  # default
    bigN = N * gridsize + (gridsize - 1) * offset
    positions_per_dim = (size - bigN) // outer_offset + 1
    patches_per_object = positions_per_dim ** 2
    return n * patches_per_object


def generate_grid_dataset(
    probe: np.ndarray,
    n_objects: int,
    outer_offset: int,
    which: str = 'train',
    intensity_scale: Optional[float] = None,
    **config_overrides
) -> GridDataset:
    """
    Generate a grid-sampled dataset using mk_simdata.

    Args:
        probe: Complex probe array (N, N)
        n_objects: Number of unique objects to generate
        outer_offset: Grid step size (determines image count)
        which: 'train' or 'test'
        intensity_scale: Optional fixed intensity scale (use for test to match train)
        **config_overrides: Override default config values

    Returns:
        GridDataset containing all simulation outputs
    """
    N = probe.shape[0]

    # Default config
    default_config = {
        'N': N,
        'gridsize': 1,
        'offset': 4,
        'nphotons': 1e9,
        'size': 500,
        'data_source': 'lines',
        'outer_offset_train': 12,
        'outer_offset_test': 12,
    }
    default_config.update(config_overrides)

    # Setup params.cfg (CONFIG-001)
    config = setup_params_cfg(**default_config)

    # Ensure probe is the right dtype
    probe_np = probe.astype(np.complex64)

    # Calculate expected image count
    expected_count = calculate_image_count(
        n_objects, config['size'], N, outer_offset, config['gridsize']
    )
    print(f"Generating {which} data: n={n_objects}, outer_offset={outer_offset}")
    print(f"  Expected images: {expected_count}")

    # Call mk_simdata
    # Returns: X, Y_I, Y_phi, intensity_scale, YY_full, norm_Y_I, coords
    result = mk_simdata(
        n=n_objects,
        size=config['size'],
        probe=probe_np,
        outer_offset=outer_offset,
        intensity_scale=intensity_scale,
        which=which,
    )

    X, Y_I, Y_phi, intensity_scale_out, YY_full, norm_Y_I, coords = result

    # Convert to numpy if needed
    if hasattr(X, 'numpy'):
        X = X.numpy()
    if hasattr(Y_I, 'numpy'):
        Y_I = Y_I.numpy()
    if hasattr(Y_phi, 'numpy'):
        Y_phi = Y_phi.numpy()
    if hasattr(norm_Y_I, 'numpy'):
        norm_Y_I = norm_Y_I.numpy()
    if hasattr(YY_full, 'numpy'):
        YY_full = YY_full.numpy()

    print(f"  Actual shape: X={X.shape}, Y_I={Y_I.shape}")
    print(f"  Intensity scale: {intensity_scale_out:.6f}")

    return GridDataset(
        X=X,
        Y_I=Y_I,
        Y_phi=Y_phi,
        intensity_scale=intensity_scale_out,
        norm_Y_I=norm_Y_I,
        YY_full=YY_full,
        coords=coords,
        config=config,
    )


def generate_train_test_data(
    probe: np.ndarray,
    n_train_objects: int = 5,
    n_test_objects: int = 2,
    outer_offset: int = 12,
    **config_overrides
) -> Tuple[GridDataset, GridDataset]:
    """
    Generate paired train and test datasets.

    Uses same intensity_scale for test as train for consistency.

    Args:
        probe: Complex probe array
        n_train_objects: Number of training objects (default: 5 -> 5120 images)
        n_test_objects: Number of test objects (default: 2 -> 2048 images)
        outer_offset: Grid step size
        **config_overrides: Override default config values

    Returns:
        (train_data, test_data)
    """
    print("=" * 60)
    print("Generating Grid-Based Datasets")
    print("=" * 60)

    # Generate training data
    train_data = generate_grid_dataset(
        probe=probe,
        n_objects=n_train_objects,
        outer_offset=outer_offset,
        which='train',
        **config_overrides
    )

    # Generate test data with same intensity_scale
    test_data = generate_grid_dataset(
        probe=probe,
        n_objects=n_test_objects,
        outer_offset=outer_offset,
        which='test',
        intensity_scale=train_data.intensity_scale,
        **config_overrides
    )

    print("=" * 60)
    print(f"Train: {train_data.X.shape[0]} images")
    print(f"Test: {test_data.X.shape[0]} images")
    print("=" * 60)

    return train_data, test_data


if __name__ == "__main__":
    # Test data generation
    from probe_utils import get_probe_for_N

    print("=== Grid Data Generator Test ===\n")

    # Test with N=64
    probe_64 = get_probe_for_N(64)

    # Small test: n=1 objects
    train_data, test_data = generate_train_test_data(
        probe=probe_64,
        n_train_objects=1,
        n_test_objects=1,
        outer_offset=12,
    )

    print(f"\nTrain X stats: mean={train_data.X.mean():.4f}, std={train_data.X.std():.4f}")
    print(f"Train Y_I stats: mean={train_data.Y_I.mean():.4f}, std={train_data.Y_I.std():.4f}")
