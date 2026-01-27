"""
Shared pytest fixtures for ptycho_torch tests.
"""
import pytest
import numpy as np
from pathlib import Path


def pytest_configure(config):
    """
    Register custom markers for torch tests.

    Per docs/development/TEST_SUITE_INDEX.md:
    - pytest.mark.slow: Tests taking > 1 second (e.g., multi-epoch training)
    - pytest.mark.deterministic: Tests requiring reproducible random seeds
    """
    config.addinivalue_line("markers", "slow: Mark test as slow running")
    config.addinivalue_line("markers", "deterministic: Test requires reproducible seed")


@pytest.fixture
def synthetic_ptycho_npz(tmp_path):
    """Create synthetic NPZ files compatible with both RawData and grid-lines workflow.

    Returns:
        Tuple[Path, Path]: (train_path, test_path) to synthetic NPZ files

    NPZ keys provided:
        - diffraction: Input diffraction patterns (n_scans, N, N)
        - xcoords, ycoords: Scan coordinates (n_scans,)
        - Y_I: Amplitude ground truth (n_samples, N, N, gridsize^2)
        - Y_phi: Phase ground truth (n_samples, N, N, gridsize^2)
        - coords_nominal: Nominal scan positions (n_samples * gridsize^2, 2)
        - coords_true: True scan positions (n_samples * gridsize^2, 2)
        - probeGuess: Probe function (N, N) complex64
        - objectGuess: Object guess (N*2, N*2) complex64
    """
    N = 64  # Use standard N=64 for config compatibility
    n_scans = 16  # Number of scan positions for RawData
    n_samples = 4
    gridsize = 1

    # Generate scan coordinates on a grid
    grid_side = int(np.sqrt(n_scans))
    xcoords = np.tile(np.arange(grid_side), grid_side).astype(np.float32) * 10.0
    ycoords = np.repeat(np.arange(grid_side), grid_side).astype(np.float32) * 10.0

    data = {
        # RawData required keys
        "diff3d": np.random.rand(n_scans, N, N).astype(np.float32),
        "diffraction": np.random.rand(n_scans, N, N).astype(np.float32),  # Alias
        "xcoords": xcoords,
        "ycoords": ycoords,
        "probeGuess": np.ones((N, N), dtype=np.complex64),
        "objectGuess": np.ones((N * 2, N * 2), dtype=np.complex64),
        # Grid-lines workflow keys
        "Y_I": np.random.rand(n_samples, N, N, gridsize**2).astype(np.float32),
        "Y_phi": np.random.rand(n_samples, N, N, gridsize**2).astype(np.float32),
        "coords_nominal": np.random.rand(n_samples * gridsize**2, 2).astype(np.float32),
        "coords_true": np.random.rand(n_samples * gridsize**2, 2).astype(np.float32),
    }
    train_path = tmp_path / "train.npz"
    test_path = tmp_path / "test.npz"
    np.savez(train_path, **data)
    np.savez(test_path, **data)
    return train_path, test_path


@pytest.fixture(scope="session", autouse=True)
def create_dummy_npz_files():
    """
    Create dummy NPZ files in /tmp for fixtures that use dummy paths.
    This is a session-scoped autouse fixture that runs once before any tests.

    Phase EB1: Many test fixtures use Path("/tmp/dummy_train.npz") but the factory
    now validates file existence. Rather than updating every fixture individually,
    we create these dummy files once at session start.
    """
    dummy_data = {
        'diffraction': np.random.rand(10, 64, 64).astype(np.float32),
        'xcoords': np.random.rand(10),
        'ycoords': np.random.rand(10),
        'probeGuess': np.ones((64, 64), dtype=np.complex64),
        'objectGuess': np.ones((128, 128), dtype=np.complex64),
    }

    train_path = Path("/tmp/dummy_train.npz")
    test_path = Path("/tmp/dummy_test.npz")

    if not train_path.exists():
        np.savez(str(train_path), **dummy_data)
    if not test_path.exists():
        np.savez(str(test_path), **dummy_data)

    yield

    # Cleanup after all tests
    if train_path.exists():
        train_path.unlink()
    if test_path.exists():
        test_path.unlink()
