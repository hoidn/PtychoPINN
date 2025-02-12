def test_create_multi_container():
    import numpy as np
    from ptycho.raw_data import RawData, key_coords_offsets, key_coords_relative
    from ptycho.workflows.components import create_multi_container_from_raw_data
    # Create a minimal dummy configuration with an attribute model.N.
    class DummyModel:
        def __init__(self, N):
            self.N = N
    class DummyConfig:
        def __init__(self, N):
            self.model = DummyModel(N)
    config = DummyConfig(64)

    # Define a FakeRawData with the minimal interface.
    class FakeRawData:
        def __init__(self, num_samples=5):
            self.num_samples = num_samples
            # valid 2D probe; setter will promote to 3D.
            self.probeGuess = np.random.rand(64, 64)
        def generate_grouped_data(self, N, K=7, nsamples=1):
            dset = {}
            dset['objectGuess'] = np.random.rand(64, 64)
            dset['X_full'] = np.random.rand(self.num_samples, 64, 64)
            dset[key_coords_offsets] = np.random.rand(self.num_samples, 2)
            dset[key_coords_relative] = np.random.rand(self.num_samples, 2)
            dset['Y'] = np.random.rand(self.num_samples, 64, 64)
            dset['nn_indices'] = np.zeros((self.num_samples, 3), dtype=int)
            dset['coords_offsets'] = np.random.rand(self.num_samples, 2)
            dset['coords_relative'] = np.random.rand(self.num_samples, 2)
            return dset

    # Create two FakeRawData instances.
    raw1 = FakeRawData(num_samples=3)
    raw2 = FakeRawData(num_samples=4)
    multi_container = create_multi_container_from_raw_data([raw1, raw2], config)

    # Verify merged container sample count is 3+4 = 7.
    assert multi_container.X.shape[0] == 7, "Merged container should have 7 samples"
    # Verify merged probes has correct 4D shape: [num_probes, H, W, 1]
    assert multi_container.probes.ndim == 4, "Probes should be a 4D array"
    # Check probe_indices length.
    assert multi_container.probe_indices.shape[0] == 7, "Probe indices should match the number of samples"

    # --- Test error case for an invalid probe shape ---
    class FakeRawDataInvalidProbe(FakeRawData):
        def __init__(self):
            super().__init__(num_samples=3)
            # Create a probe with invalid channel size.
            self.probeGuess = np.random.rand(64, 64, 3)
    raw_invalid = FakeRawDataInvalidProbe()
    try:
        _ = create_multi_container_from_raw_data([raw1, raw_invalid], config)
        assert False, "Expected ValueError for invalid probe shape in container"
    except ValueError:
        pass
