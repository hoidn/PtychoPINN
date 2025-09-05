import numpy as np
from ptycho.nongrid_simulation import generate_simulated_data
from ptycho.loader import RawData
import os
import shutil
import unittest

@unittest.skip("Deprecated: generate_simulated_data API changed from (obj,probe,nimages) to (config,obj,probe) and memoization disabled")
def test_memoize_simulated_data():
    # Create sample input data
    objectGuess = np.random.rand(128, 128) + 1j * np.random.rand(128, 128)
    probeGuess = np.random.rand(32, 32) + 1j * np.random.rand(32, 32)
    nimages = 100
    buffer = 10
    random_seed = 42

    # Clear the cache directory before starting the test
    cache_dir = 'memoized_simulated_data'
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    # First call, should compute the result and cache it
    result1 = generate_simulated_data(objectGuess, probeGuess, nimages, buffer, random_seed)
    assert isinstance(result1, tuple), "Result should be a tuple"
    assert len(result1) == 2, "Result tuple should have 2 elements"
    assert isinstance(result1[0], RawData), "First element should be a RawData instance"
    assert isinstance(result1[1], np.ndarray), "Second element should be a numpy array"

    # Second call, should load the result from the cache
    result2 = generate_simulated_data(objectGuess, probeGuess, nimages, buffer, random_seed)
    assert isinstance(result2, tuple), "Result should be a tuple"
    assert len(result2) == 2, "Result tuple should have 2 elements"
    assert isinstance(result2[0], RawData), "First element should be a RawData instance"
    assert isinstance(result2[1], np.ndarray), "Second element should be a numpy array"

    # Check if results are identical
    assert np.array_equal(result1[0].diff3d, result2[0].diff3d), "Cached result differs from original"
    assert np.array_equal(result1[1], result2[1]), "Cached patches differ from original"

    # Third call with different random seed, should compute a new result
    result3 = generate_simulated_data(objectGuess, probeGuess, nimages, buffer, random_seed=123)
    assert isinstance(result3, tuple), "Result should be a tuple"
    assert len(result3) == 2, "Result tuple should have 2 elements"
    assert isinstance(result3[0], RawData), "First element should be a RawData instance"
    assert isinstance(result3[1], np.ndarray), "Second element should be a numpy array"

    # Check if results are different
    assert not np.array_equal(result1[0].diff3d, result3[0].diff3d), "Results with different seeds should differ"

    print("All tests passed successfully!")

if __name__ == "__main__":
    test_memoize_simulated_data()
