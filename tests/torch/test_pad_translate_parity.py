"""Parity between ``ptycho_torch.pad_translate`` and the TF originals.

The torch rail's ``objectGuess``-without-``Y`` ground-truth path in
``ptycho.raw_data.get_image_patches`` now uses the numpy ``pad``/``translate``
ports.  These tests pin them against the TensorFlow originals
(``ptycho.tf_helper.pad`` and the projective-warp ``translate_xla`` code path
that ``ptycho.tf_helper.translate`` uses by default) on complex64 fixtures,
including a zero-background fill case, plus a cold-subprocess check that the
whole ``generate_grouped_data`` path stays TF-free.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ptycho_torch.pad_translate import pad, translate


def _tf_helpers():
    tf = pytest.importorskip("tensorflow")
    from ptycho.tf_helper import pad as tf_pad
    from ptycho.tf_helper import translate_xla

    return tf, tf_pad, translate_xla


def _complex64_fixture(B=3, H=12, W=12):
    rng = np.random.default_rng(0)
    img = np.zeros((B, H, W, 1), dtype=np.complex64)
    for i in range(B):
        img[i, 3:9, 4:10, 0] = (
            rng.uniform(0.1, 0.9) + 1j * rng.uniform(-0.5, 0.5)
        ) * (i + 1)
    return img


def test_pad_matches_zero_padding_2d():
    tf, tf_pad, _ = _tf_helpers()
    img = _complex64_fixture()
    for size in (0, 3, 6):
        got = pad(img, size)
        want = tf_pad(tf.constant(img), size).numpy()
        np.testing.assert_allclose(got, want, rtol=0, atol=0)
        assert got.dtype == want.dtype


def test_translate_bilinear_matches_xla():
    tf, _, translate_xla = _tf_helpers()
    img = _complex64_fixture()
    offsets = np.array([[0.4, -0.6], [-1.1, 0.2], [0.0, 0.0]], dtype=np.float32)

    got = translate(img, offsets, interpolation="bilinear")
    want = translate_xla(
        tf.constant(img),
        tf.constant(offsets, dtype=tf.float32),
        interpolation="bilinear",
        use_jit=False,
    ).numpy()

    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)
    assert got.dtype == want.dtype


def test_translate_nearest_matches_xla():
    tf, _, translate_xla = _tf_helpers()
    img = _complex64_fixture()
    offsets = np.array([[0.4, -0.6], [-1.1, 0.2], [0.0, 0.0]], dtype=np.float32)

    got = translate(img, offsets, interpolation="nearest")
    want = translate_xla(
        tf.constant(img),
        tf.constant(offsets, dtype=tf.float32),
        interpolation="nearest",
        use_jit=False,
    ).numpy()

    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)


def test_translate_zero_fills_vacated_background():
    tf, _, translate_xla = _tf_helpers()
    img = np.zeros((1, 8, 8, 1), dtype=np.complex64)
    img[0, 1, 1, 0] = 0.5 + 0.5j
    offsets = np.array([[2.0, 2.0]], dtype=np.float32)

    got = translate(img, offsets, interpolation="bilinear")
    want = translate_xla(
        tf.constant(img),
        tf.constant(offsets, dtype=tf.float32),
        interpolation="bilinear",
        use_jit=False,
    ).numpy()

    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)
    # Content moved down-right by (2,2): destination holds the source value and
    # the vacated home is zero-filled (background fill, not wrapped).
    assert abs(got[0, 3, 3, 0]) > 0.6
    assert got[0, 1, 1, 0] == 0


def test_generate_grouped_data_object_guess_without_y_is_tf_free():
    """Cold subprocess: the objectGuess-without-Y path never loads TensorFlow.

    ``ptycho.raw_data.get_image_patches`` is reached from ``generate_grouped_data``
    when an input has ``objectGuess`` but no ``Y``; that path used to call the
    TF-only ``hh.pad``/``hh.translate``.  It now routes through numpy.
    """
    repo_root = Path(__file__).resolve().parents[2]
    script = """\
import sys
import numpy as np
from ptycho.raw_data import RawData

N = 8
M = 6
rng = np.random.default_rng(0)
x = np.arange(M, dtype=np.float64)
y = np.arange(M, dtype=np.float64)
diff = rng.random((M, N, N)).astype(np.float32)
probe = (rng.random((N, N)) + 1j * rng.random((N, N))).astype(np.complex64)
obj = (rng.random((16, 16)) + 1j * rng.random((16, 16))).astype(np.complex64)

raw = RawData(x, y, x, y, diff, probe, None, objectGuess=obj, Y=None)
grouped = raw.generate_grouped_data(N=N, K=4, nsamples=3, gridsize=1, seed=0)
assert grouped["Y"] is not None, "objectGuess-without-Y must produce Y patches"
assert grouped["Y"].shape == (3, N, N, 1)
assert "tensorflow" not in sys.modules, "generate_grouped_data loaded tensorflow"
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        f"objectGuess-without-Y probe failed (rc={completed.returncode}):\n"
        f"{completed.stderr}"
    )
