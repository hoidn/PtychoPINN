import numpy as np
import pytest


def test_probe_normalization_parity():
    pytest.importorskip("tensorflow")

    from ptycho import params
    from ptycho import probe as tf_probe
    from ptycho_torch.helper import normalize_probe_like_tf

    probe_guess = (np.ones((8, 8), dtype=np.float32) * 2.0).astype(np.complex64)

    old_n = params.get("N")
    old_probe_scale = params.get("probe_scale")
    old_probe = params.get("probe", None)
    try:
        params.set("N", probe_guess.shape[0])
        params.set("probe_scale", 4.0)

        tf_probe.set_probe_guess(None, probe_guess)
        tf_norm = params.get("probe").numpy()
        torch_norm, _ = normalize_probe_like_tf(probe_guess, probe_scale=4.0)
    finally:
        params.set("N", old_n)
        params.set("probe_scale", old_probe_scale)
        params.set("probe", old_probe)

    tf_norm = tf_norm[..., 0]
    assert np.allclose(tf_norm, torch_norm, rtol=1e-6, atol=1e-6)
