import numpy as np

from ptycho.datagen.grf import mk_grf


def test_mk_grf_generates_finite_expected_shape() -> None:
    out = mk_grf(64)
    assert out.shape == (64, 64, 1)
    assert np.isfinite(out).all()
