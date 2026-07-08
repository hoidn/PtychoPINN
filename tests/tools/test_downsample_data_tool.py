import numpy as np


def test_downsample_tool_crop_center_uses_shared_center_crop(monkeypatch):
    from scripts.tools import downsample_data_tool as module

    called = {}

    def _fake_center_crop_spatial(arr, target_h, target_w):
        called["args"] = (int(target_h), int(target_w))
        return arr[:target_h, :target_w]

    monkeypatch.setattr(module, "center_crop_spatial", _fake_center_crop_spatial)

    src = np.arange(36).reshape(6, 6)
    out = module.crop_center(src, 4, 4)

    assert called["args"] == (4, 4)
    np.testing.assert_array_equal(out, src[:4, :4])


def test_downsample_tool_crop_center_matches_shared_spatial_crop():
    from ptycho.image.cropping import center_crop_spatial
    from scripts.tools.downsample_data_tool import crop_center

    src = np.arange(36).reshape(6, 6)
    expected = center_crop_spatial(src, 4, 4)
    actual = crop_center(src, 4, 4)
    np.testing.assert_array_equal(actual, expected)
