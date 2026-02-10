import numpy as np

from ptycho import params as tf_params
from ptycho import probe as tf_probe
from ptycho_torch.helper import normalize_probe_like_tf


def test_probe_normalization_matches_tf():
    np.random.seed(123)
    N = 16
    probe_scale = 4.0
    probe_guess = (
        np.random.rand(N, N) + 1j * np.random.rand(N, N)
    ).astype(np.complex64)

    old_n = tf_params.get('N')
    old_probe_scale = tf_params.get('probe_scale')
    old_probe = tf_params.get('probe', None)
    try:
        tf_params.set('N', N)
        tf_params.set('probe_scale', probe_scale)
        tf_probe.set_probe_guess(probe_guess=probe_guess)
        tf_norm = tf_params.get('probe').numpy()[..., 0]
    finally:
        tf_params.set('N', old_n)
        tf_params.set('probe_scale', old_probe_scale)
        tf_params.set('probe', old_probe)

    torch_norm, _ = normalize_probe_like_tf(probe_guess, probe_scale=probe_scale)

    assert np.allclose(tf_norm, torch_norm, atol=1e-6)
