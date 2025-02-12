def test_probe_property():
    import numpy as np
    from ptycho.loader import PtychoDataContainer

    # Create dummy container data.
    X = np.random.rand(5, 64, 64)
    Y_I = np.random.rand(5, 64, 64)
    Y_phi = np.random.rand(5, 64, 64)
    norm_Y_I = np.ones(5)
    YY_full = None
    coords_nominal = np.random.rand(5, 2)
    coords_true = np.random.rand(5, 2)
    nn_indices = np.zeros((5, 3), dtype=int)
    global_offsets = np.random.rand(5, 2)
    local_offsets = np.random.rand(5, 2)
    valid_probe = np.random.rand(64, 64, 1)  # valid probe shape
    probe_indices = np.zeros(5, dtype=np.int64)

    # Instantiate the container (this will call the property setter).
    container = PtychoDataContainer(
        X, Y_I, Y_phi, norm_Y_I, YY_full,
        coords_nominal, coords_true, nn_indices,
        global_offsets, local_offsets, valid_probe, probe_indices=probe_indices
    )

    # Test the getter.
    assert container.probe.shape == valid_probe.shape

    # Test the setter with a valid update.
    new_probe = np.random.rand(64, 64, 1)
    container.probe = new_probe
    assert np.array_equal(container.probe, new_probe)

    # Test that a 2D input is promoted to 3D.
    probe_2d = np.random.rand(64, 64)
    container.probe = probe_2d
    assert container.probe.ndim == 3 and container.probe.shape[-1] == 1

    # Test that an invalid 3D shape (wrong channel size) raises a ValueError.
    try:
        container.probe = np.random.rand(64, 64, 3)
        raise AssertionError("Expected ValueError for invalid probe shape")
    except ValueError:
        pass

    # Optionally, test that non-array inputs raise TypeError.
    try:
        container.probe = "not an array"
        raise AssertionError("Expected TypeError for non-array probe")
    except TypeError:
        pass
