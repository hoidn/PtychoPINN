import numpy as np


def test_helper_center_crop_uses_shared_center_crop(monkeypatch):
    import ptycho_torch.helper as helper

    called = {}

    def _fake_center_crop_spatial(arr, target_h, target_w):
        called["args"] = (int(target_h), int(target_w))
        return arr[:target_h, :target_w]

    monkeypatch.setattr(helper, "center_crop_spatial", _fake_center_crop_spatial)

    src = np.arange(64).reshape(8, 8)
    out = helper.center_crop(src, 4)

    assert called["args"] == (4, 4)
    np.testing.assert_array_equal(out, src[:4, :4])


def test_helper_center_crop_matches_shared_spatial_center_crop():
    from ptycho.image.cropping import center_crop_spatial
    from ptycho_torch.helper import center_crop

    arr = np.arange(64).reshape(8, 8)
    expected = center_crop_spatial(arr, 4, 4)
    actual = center_crop(arr, 4)
    np.testing.assert_array_equal(actual, expected)
