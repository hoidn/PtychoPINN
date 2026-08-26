"""NumPy ports of the TensorFlow-only image helpers used by the
``objectGuess``-without-``Y`` ground-truth path in
``ptycho.raw_data.get_image_patches``.

These replicate the exact semantics of the originals so the torch rail can
extract ground-truth patches without loading TensorFlow:

* ``pad`` matches ``ptycho.tf_helper.pad`` (``tf.keras.layers.ZeroPadding2D``).
* ``translate`` matches ``ptycho.tf_helper.translate`` on its default XLA path
  (``projective_warp_xla``): output pixel ``(y, x)`` samples input
  ``(y - dy, x - dx)`` with bilinear/nearest interpolation and zero fill
  out of bounds.
"""
from __future__ import annotations

import numpy as np


def pad(imgs, size):
    """Zero-pad H and W of an NHWC array by ``size`` on each side.

    Matches ``tf.keras.layers.ZeroPadding2D((size, size))`` exactly,
    preserving dtype.
    """
    imgs = np.asarray(imgs)
    return np.pad(
        imgs, ((0, 0), (size, size), (size, size), (0, 0)), mode="constant"
    )


def translate(imgs, offsets, interpolation="bilinear"):
    """Translate an NHWC image stack by ``offsets`` (B, 2) = [dx, dy] pixels.

    Matches ``ptycho.tf_helper.translate``'s projective-warp convention
    (positive offsets move content in the positive direction).  Returns an
    array with the same dtype as ``imgs``.
    """
    imgs = np.asarray(imgs)
    offsets = np.asarray(offsets, dtype=np.float64)
    B, H, W, C = imgs.shape
    dx = -offsets[:, 0]
    dy = -offsets[:, 1]

    y = np.arange(H, dtype=np.float64)
    x = np.arange(W, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    sx = xx[None] + dx[:, None, None]  # (B, H, W) source x
    sy = yy[None] + dy[:, None, None]  # (B, H, W) source y

    b = np.arange(B)[:, None, None]

    def _gather(yi, xi):
        return imgs[b, yi.astype(np.int32), xi.astype(np.int32), :]

    if interpolation == "nearest":
        ix = np.clip(np.round(sx), 0, W - 1)
        iy = np.clip(np.round(sy), 0, H - 1)
        mask = ((sx >= 0) & (sx <= W - 1) & (sy >= 0) & (sy <= H - 1))[..., None]
        return (_gather(iy, ix) * mask).astype(imgs.dtype)

    if interpolation != "bilinear":
        raise ValueError("interpolation must be 'bilinear' or 'nearest'")

    x0 = np.floor(sx)
    y0 = np.floor(sy)
    x1 = x0 + 1.0
    y1 = y0 + 1.0

    in_x0 = (x0 >= 0) & (x0 <= W - 1)
    in_x1 = (x1 >= 0) & (x1 <= W - 1)
    in_y0 = (y0 >= 0) & (y0 <= H - 1)
    in_y1 = (y1 >= 0) & (y1 <= H - 1)

    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)

    Ia = _gather(y0c, x0c) * (in_x0 & in_y0)[..., None]
    Ib = _gather(y0c, x1c) * (in_x1 & in_y0)[..., None]
    Ic = _gather(y1c, x0c) * (in_x0 & in_y1)[..., None]
    Id = _gather(y1c, x1c) * (in_x1 & in_y1)[..., None]

    wx = (sx - x0)[..., None]
    wy = (sy - y0)[..., None]
    wa = (1.0 - wx) * (1.0 - wy)
    wb = wx * (1.0 - wy)
    wc = (1.0 - wx) * wy
    wd = wx * wy

    return (wa * Ia + wb * Ib + wc * Ic + wd * Id).astype(imgs.dtype)
