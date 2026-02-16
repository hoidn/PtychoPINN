import numpy as np

from scripts.studies.grid_lines_torch_runner import _reassemble_with_coords_offsets


def test_large_external_n128_uses_batched_backend_without_tf_shift_sum(monkeypatch):
    pred = np.ones((4096, 128, 128, 1), dtype=np.complex64)
    test_data = {"coords_offsets": np.zeros((4096, 1, 2, 1), dtype=np.float32)}

    def fail_shift_sum(*args, **kwargs):
        _ = (args, kwargs)
        raise AssertionError("shift-sum backend should not be used for large N=128 auto routing")

    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner._reassemble_position_shift_sum",
        fail_shift_sum,
    )
    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner._reassemble_position_batched",
        lambda patches, offsets_b12c, M, batch_size: np.full((M, M), 2.0 + 0.0j, dtype=np.complex64),
    )

    out = _reassemble_with_coords_offsets(
        pred,
        test_data,
        M=128,
        backend="auto",
        batch_size=16,
    )

    assert out.shape == (128, 128)
    assert np.allclose(out, np.full((128, 128), 2.0 + 0.0j, dtype=np.complex64))
