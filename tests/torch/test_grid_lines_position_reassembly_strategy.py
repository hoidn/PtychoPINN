import numpy as np

from scripts.studies.grid_lines_torch_runner import _reassemble_with_coords_offsets


def test_large_external_n128_auto_prefers_shift_sum(monkeypatch):
    pred = np.ones((4096, 128, 128, 1), dtype=np.complex64)
    test_data = {"coords_offsets": np.zeros((4096, 1, 2, 1), dtype=np.float32)}

    def fake_shift_sum(patches, offsets_b112, M):
        _ = (patches, offsets_b112)
        return np.full((M, M), 2.0 + 0.0j, dtype=np.complex64)

    def fail_batched(*args, **kwargs):
        _ = (args, kwargs)
        raise AssertionError("batched backend should not be selected by auto")

    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner._reassemble_position_shift_sum",
        fake_shift_sum,
    )
    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner._reassemble_position_batched",
        fail_batched,
    )

    out = _reassemble_with_coords_offsets(
        pred,
        test_data,
        M=128,
        backend="auto",
        batch_size=16,
        position_crop_border=0,
    )

    assert out.shape == (128, 128)
    assert np.allclose(out, np.full((128, 128), 2.0 + 0.0j, dtype=np.complex64))


def test_explicit_batched_backend_still_uses_batched(monkeypatch):
    pred = np.ones((4096, 128, 128, 1), dtype=np.complex64)
    test_data = {"coords_offsets": np.zeros((4096, 1, 2, 1), dtype=np.float32)}

    def fail_shift_sum(*args, **kwargs):
        _ = (args, kwargs)
        raise AssertionError("shift_sum should not be used when backend='batched'")

    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner._reassemble_position_shift_sum",
        fail_shift_sum,
    )
    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner._reassemble_position_batched",
        lambda patches, offsets_b12c, M, batch_size: np.full((M, M), 3.0 + 0.0j, dtype=np.complex64),
    )

    out = _reassemble_with_coords_offsets(
        pred,
        test_data,
        M=128,
        backend="batched",
        batch_size=16,
        position_crop_border=0,
    )

    assert out.shape == (128, 128)
    assert np.allclose(out, np.full((128, 128), 3.0 + 0.0j, dtype=np.complex64))


def test_auto_default_applies_nonzero_crop(monkeypatch):
    pred = np.ones((4096, 128, 128, 1), dtype=np.complex64)
    test_data = {"coords_offsets": np.zeros((4096, 1, 2, 1), dtype=np.float32)}

    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner._reassemble_position_shift_sum",
        lambda patches, offsets_b112, M: np.full((M, M), 4.0 + 0.0j, dtype=np.complex64),
    )
    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner._reassemble_position_batched",
        lambda patches, offsets_b12c, M, batch_size: np.full((M, M), 8.0 + 0.0j, dtype=np.complex64),
    )

    out = _reassemble_with_coords_offsets(
        pred,
        test_data,
        M=128,
        backend="auto",
        batch_size=16,
        position_crop_border=None,
    )

    assert out.shape == (64, 64)


def test_explicit_zero_crop_preserves_legacy_patch_size(monkeypatch):
    pred = np.ones((4096, 128, 128, 1), dtype=np.complex64)
    test_data = {"coords_offsets": np.zeros((4096, 1, 2, 1), dtype=np.float32)}

    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner._reassemble_position_shift_sum",
        lambda patches, offsets_b112, M: np.full((M, M), 2.0 + 0.0j, dtype=np.complex64),
    )

    out = _reassemble_with_coords_offsets(
        pred,
        test_data,
        M=128,
        backend="shift_sum",
        batch_size=16,
        position_crop_border=0,
    )

    assert out.shape == (128, 128)
