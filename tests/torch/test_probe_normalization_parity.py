import numpy as np

from ptycho import params as tf_params
from ptycho import probe as tf_probe
from ptycho_torch.helper import normalize_probe_like_tf


def test_probe_normalization_matches_tf_with_explicit_hard_mask():
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

    hard_mask = np.abs(tf_probe.get_probe_mask(N).numpy()[..., 0]).astype(np.float32)
    torch_norm, _ = normalize_probe_like_tf(
        probe_guess,
        probe_scale=probe_scale,
        probe_mask=True,
        probe_mask_tensor=hard_mask,
        probe_mask_sigma=0.0,
    )

    assert np.allclose(tf_norm, torch_norm, atol=1e-6)


def test_probe_normalization_default_soft_mask_differs_from_tf_hard_mask():
    np.random.seed(321)
    N = 16
    probe_guess = (
        np.random.rand(N, N) + 1j * np.random.rand(N, N)
    ).astype(np.complex64)

    hard_mask = np.abs(tf_probe.get_probe_mask(N).numpy()[..., 0]).astype(np.float32)
    torch_soft, _ = normalize_probe_like_tf(probe_guess, probe_scale=4.0)
    torch_hard, _ = normalize_probe_like_tf(
        probe_guess,
        probe_scale=4.0,
        probe_mask=True,
        probe_mask_tensor=hard_mask,
        probe_mask_sigma=0.0,
    )

    assert not np.allclose(torch_soft, torch_hard, atol=1e-6)
