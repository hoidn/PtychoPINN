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
