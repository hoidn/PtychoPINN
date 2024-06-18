import numpy as np
from ptycho.loader import PtychoDataContainer

def test_slicing():
    # Create dummy data for testing
    X = np.random.rand(10, 64, 64, 1)
    Y_I = np.random.rand(10, 64, 64, 1)
    Y_phi = np.random.rand(10, 64, 64, 1)
    norm_Y_I = np.random.rand(10, 64, 64, 1)
    YY_full = np.random.rand(10, 64, 64, 1)
    coords_nominal = np.random.rand(10, 2)
    coords_true = np.random.rand(10, 2)
    nn_indices = np.random.randint(0, 10, (10, 5))
    global_offsets = np.random.rand(10, 2)
    local_offsets = np.random.rand(10, 2)
    probe = np.random.rand(64, 64)

    # Create an instance of PtychoDataContainer
    original_container = PtychoDataContainer(
        X,
        Y_I,
        Y_phi,
        norm_Y_I,
        YY_full,
        coords_nominal,
        coords_true,
        nn_indices,
        global_offsets,
        local_offsets,
        probe
    )

    # Perform slicing
    sliced_container = original_container[:5]

    # Check that the original container is unchanged
    assert np.array_equal(original_container.X, X)
    assert np.array_equal(original_container.Y_I, Y_I)
    assert np.array_equal(original_container.Y_phi, Y_phi)
    assert np.array_equal(original_container.norm_Y_I, norm_Y_I)
    assert np.array_equal(original_container.YY_full, YY_full)
    assert np.array_equal(original_container.coords_nominal, coords_nominal)
    assert np.array_equal(original_container.coords_true, coords_true)
    assert np.array_equal(original_container.nn_indices, nn_indices)
    assert np.array_equal(original_container.global_offsets, global_offsets)
    assert np.array_equal(original_container.local_offsets, local_offsets)
    assert np.array_equal(original_container.probe, probe)

    # Check that the sliced container has the correct data
    assert np.array_equal(sliced_container.X, X[:5])
    assert np.array_equal(sliced_container.Y_I, Y_I[:5])
    assert np.array_equal(sliced_container.Y_phi, Y_phi[:5])
    assert np.array_equal(sliced_container.norm_Y_I, norm_Y_I[:5])
    assert np.array_equal(sliced_container.YY_full, YY_full[:5])
    assert np.array_equal(sliced_container.coords_nominal, coords_nominal[:5])
    assert np.array_equal(sliced_container.coords_true, coords_true[:5])
    assert np.array_equal(sliced_container.nn_indices, nn_indices[:5])
    assert np.array_equal(sliced_container.global_offsets, global_offsets[:5])
    assert np.array_equal(sliced_container.local_offsets, local_offsets[:5])
    assert np.array_equal(sliced_container.probe, probe)

    print("All tests passed!")

if __name__ == "__main__":
    test_slicing()
