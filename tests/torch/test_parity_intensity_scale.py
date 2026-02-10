import numpy as np
import pytest


def test_intensity_scale_parity():
    tf = pytest.importorskip("tensorflow")
    import torch

    from ptycho import params
    from ptycho.diffsim import scale_nphotons
    from ptycho_torch.helper import derive_intensity_scale_from_amplitudes

    nphotons = 1e5
    arr = np.linspace(1.0, 2.0, 16, dtype=np.float32).reshape(2, 4, 2)

    old_nphotons = params.get("nphotons")
    try:
        params.set("nphotons", nphotons)
        tf_scale = scale_nphotons(tf.convert_to_tensor(arr)).numpy()
    finally:
        params.set("nphotons", old_nphotons)

    torch_scale = derive_intensity_scale_from_amplitudes(torch.tensor(arr), nphotons).numpy()

    assert np.allclose(tf_scale, torch_scale, rtol=1e-6, atol=1e-6)
