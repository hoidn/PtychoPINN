import pytest
import numpy as np
from pathlib import Path
from ptycho.raw_data import RawData
from ptycho.config.config import TrainingConfig, ModelConfig

@pytest.fixture
def sample_probe():
    """Return a valid probe (2D array to be promoted to 3D)."""
    return np.random.rand(64, 64)

@pytest.fixture
def sample_raw_data(sample_probe):
    """Return a sample RawData instance with minimal required attributes."""
    num_samples = 5
    xcoords = np.random.rand(num_samples)
    ycoords = np.random.rand(num_samples)
    xcoords_start = np.copy(xcoords)
    ycoords_start = np.copy(ycoords)
    diff3d = np.random.rand(num_samples, 64, 64)
    scan_index = np.zeros(num_samples, dtype=int)
    objectGuess = np.random.rand(64, 64)
    return RawData(xcoords, ycoords, xcoords_start, ycoords_start, diff3d, sample_probe, scan_index, objectGuess=objectGuess)

@pytest.fixture
def sample_config():
    """Return a sample TrainingConfig instance with a dummy ModelConfig (with model.N = 64)."""
    # Create a minimal ModelConfig (fill in required fields as appropriate)
    model_config = ModelConfig(N=64, default_probe_scale=0.7, probe_mask=True, probe=None)
    # Create a TrainingConfig; fill other parameters with dummy values if needed.
    return TrainingConfig(model=model_config, train_data_file=Path("dummy_train.npz"), test_data_file=Path("dummy_test.npz"))

@pytest.fixture
def sample_raw_data_list(sample_raw_data):
    """Return a list containing two sample RawData instances."""
    return [sample_raw_data, sample_raw_data]
