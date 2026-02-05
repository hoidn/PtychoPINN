import numpy as np

from ptycho_torch.data_container_bridge import PtychoDataContainerTorch


def test_rawdata_container_flips_coords_relative_sign():
    N = 4
    coords_relative = np.array([[[[1.0], [2.0]]]], dtype=np.float32)  # (1, 1, 2, 1)
    grouped_data = {
        "X_full": np.zeros((1, N, N, 1), dtype=np.float32),
        "Y": np.zeros((1, N, N, 1), dtype=np.complex64),
        "coords_relative": coords_relative,
        "coords_offsets": np.zeros((1, 1, 2, 1), dtype=np.float64),
        "nn_indices": np.zeros((1, 1), dtype=np.int32),
    }
    probe = np.zeros((N, N), dtype=np.complex64)

    container = PtychoDataContainerTorch(grouped_data, probe)

    assert np.allclose(container.coords_nominal.cpu().numpy(), -coords_relative)
