"""Tests for architecture-neutral raster patch tiling."""

import numpy as np


def test_stitch_raster_patches_crops_tiles_and_restores_object_scale():
    from ptycho.image.stitching import stitch_raster_patches

    patches = np.stack(
        [np.full((4, 4), value + 1j * value) for value in (1, 2, 3, 4)]
    ).astype(np.complex64)

    stitched = stitch_raster_patches(
        patches,
        outer_offset=4,
        normalization=3.0,
    )

    assert stitched.shape == (4, 4)
    expected = np.block(
        [
            [np.full((2, 2), 3 + 3j), np.full((2, 2), 6 + 6j)],
            [np.full((2, 2), 9 + 9j), np.full((2, 2), 12 + 12j)],
        ]
    )
    np.testing.assert_array_equal(stitched, expected)


def test_historical_tiled_geometry_produces_a_270_square_canvas():
    from ptycho.image.stitching import stitch_raster_patches

    patches = np.ones((729, 128, 128), dtype=np.complex64)
    stitched = stitch_raster_patches(
        patches,
        outer_offset=20,
        normalization=1.0,
    )

    assert stitched.shape == (270, 270)
