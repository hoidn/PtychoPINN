"""
XLA‑friendly projective image warp for TensorFlow (GPU/TPU/CPU).

- Pure TensorFlow ops; no TensorFlow Addons dependency.
- Works with @tf.function(jit_compile=True) to fuse grid + sampling math.
- NHWC layout. Supports batched 3x3 homographies or TFA-style 8‑param vectors.
- Interpolation: "nearest" or "bilinear".
- Fill mode: "edge" (duplicate/clamp) or "zeros".

Typical perf (guidance, not a guarantee):
  A100, B=8, 1024x1024, C=1, bilinear+edge: ~2.5–4 ms/iter after compile.
"""
from __future__ import annotations
import tensorflow as tf

# ------------------------------
# Public API
# ------------------------------

def tfa_params_to_3x3(params: tf.Tensor) -> tf.Tensor:
    """Convert TFA 8‑parameter projective transform vectors to 3x3 matrices.

    Args:
      params: [B, 8] with order [a0, a1, a2, b0, b1, b2, c0, c1].
    Returns:
      mats: [B, 3, 3]
    """
    params = tf.convert_to_tensor(params)
    if params.shape.rank != 2 or params.shape[-1] != 8:
        raise ValueError("params must have shape [B,8]")
    a0, a1, a2, b0, b1, b2, c0, c1 = tf.unstack(params, axis=-1)
    one = tf.ones_like(a0)
    row0 = tf.stack([a0, a1, a2], axis=-1)
    row1 = tf.stack([b0, b1, b2], axis=-1)
    row2 = tf.stack([c0, c1, one], axis=-1)
    return tf.stack([row0, row1, row2], axis=-2)


def projective_warp_xla(
    images: tf.Tensor,
    transforms: tf.Tensor,
    *,
    interpolation: str = "bilinear",
    fill_mode: str = "edge",
) -> tf.Tensor:
    """Projective warp (homography) with XLA‑friendly ops.

    Mapping convention matches TFA/TFG: output pixel (x, y) samples source (x', y')
    computed by applying the matrix to homogeneous output coordinates.

    Args:
      images: [B,H,W,C] NHWC, any float/half/bfloat16 type.
      transforms: either [B,3,3] homographies or [B,8] TFA parameters.
      interpolation: "nearest" or "bilinear".
      fill_mode: "edge" (duplicate/clamp) or "zeros".

    Returns:
      Warped images with shape [B,H,W,C] and same dtype as `images`.
    """
    if images.shape.rank != 4:
        raise ValueError("images must be [B,H,W,C] NHWC")

    B = tf.shape(images)[0]
    H = tf.shape(images)[1]
    W = tf.shape(images)[2]
    C = tf.shape(images)[3]

    img_dtype = images.dtype
    compute_dtype = tf.float32
    if img_dtype.is_integer:
        images = tf.cast(images, compute_dtype)
    elif img_dtype in (tf.float16, tf.bfloat16):
        images = tf.cast(images, compute_dtype)
    elif img_dtype == tf.float64:
        # Keep float64 precision for computation
        compute_dtype = tf.float64

    if transforms.shape.rank == 2 and transforms.shape[-1] == 8:
        M = tfa_params_to_3x3(transforms)
    elif transforms.shape.rank == 3 and transforms.shape[-1] == 3 and transforms.shape[-2] == 3:
        M = tf.convert_to_tensor(transforms, dtype=compute_dtype)
    else:
        raise ValueError("transforms must be [B,3,3] or [B,8]")

    M = tf.cast(M, compute_dtype)

    # Build output grid in pixel coordinates (x in [0,W-1], y in [0,H-1])
    y = tf.range(H, dtype=compute_dtype)
    x = tf.range(W, dtype=compute_dtype)
    yy, xx = tf.meshgrid(y, x, indexing="ij")  # [H,W]
    ones = tf.ones_like(xx)
    grid = tf.stack([xx, yy, ones], axis=-1)    # [H,W,3]
    grid = tf.expand_dims(grid, 0)              # [1,H,W,3]
    grid = tf.tile(grid, [B, 1, 1, 1])         # [B,H,W,3]

    # Apply homography: src = M @ [x, y, 1]^T
    src = tf.einsum("bij,bhwj->bhwi", M, grid)  # [B,H,W,3]
    sx = src[..., 0] / src[..., 2]
    sy = src[..., 1] / src[..., 2]

    if interpolation not in ("nearest", "bilinear"):
        raise ValueError("interpolation must be 'nearest' or 'bilinear'")
    if fill_mode not in ("edge", "zeros"):
        raise ValueError("fill_mode must be 'edge' or 'zeros'")

    if interpolation == "nearest":
        ix = tf.round(sx)
        iy = tf.round(sy)

        if fill_mode == "edge":
            ix = tf.clip_by_value(ix, 0.0, tf.cast(W - 1, compute_dtype))
            iy = tf.clip_by_value(iy, 0.0, tf.cast(H - 1, compute_dtype))
            gathered = _gather_bhw(images, ix, iy, H, W)
            out = gathered
        else:  # zeros
            in_x = (sx >= 0.0) & (sx <= tf.cast(W - 1, compute_dtype))
            in_y = (sy >= 0.0) & (sy <= tf.cast(H - 1, compute_dtype))
            mask = tf.cast(in_x & in_y, compute_dtype)
            ix = tf.clip_by_value(ix, 0.0, tf.cast(W - 1, compute_dtype))
            iy = tf.clip_by_value(iy, 0.0, tf.cast(H - 1, compute_dtype))
            gathered = _gather_bhw(images, ix, iy, H, W)
            out = gathered * tf.expand_dims(tf.reshape(mask, [B, H, W]), -1)
    else:
        # Bilinear
        x0 = tf.floor(sx)
        y0 = tf.floor(sy)
        x1 = x0 + 1.0
        y1 = y0 + 1.0

        if fill_mode == "edge":
            x0c = tf.clip_by_value(x0, 0.0, tf.cast(W - 1, compute_dtype))
            y0c = tf.clip_by_value(y0, 0.0, tf.cast(H - 1, compute_dtype))
            x1c = tf.clip_by_value(x1, 0.0, tf.cast(W - 1, compute_dtype))
            y1c = tf.clip_by_value(y1, 0.0, tf.cast(H - 1, compute_dtype))

            Ia = _gather_bhw(images, x0c, y0c, H, W)
            Ib = _gather_bhw(images, x1c, y0c, H, W)
            Ic = _gather_bhw(images, x0c, y1c, H, W)
            Id = _gather_bhw(images, x1c, y1c, H, W)

            wx = tf.expand_dims(sx - x0, -1)
            wy = tf.expand_dims(sy - y0, -1)
            # Cast weights to same dtype as images for multiplication
            wx = tf.cast(wx, images.dtype)
            wy = tf.cast(wy, images.dtype)
            wa = (1.0 - wx) * (1.0 - wy)
            wb = wx * (1.0 - wy)
            wc = (1.0 - wx) * wy
            wd = wx * wy
            out = wa * Ia + wb * Ib + wc * Ic + wd * Id
        else:  # zeros
            in_x0 = (x0 >= 0.0) & (x0 <= tf.cast(W - 1, compute_dtype))
            in_x1 = (x1 >= 0.0) & (x1 <= tf.cast(W - 1, compute_dtype))
            in_y0 = (y0 >= 0.0) & (y0 <= tf.cast(H - 1, compute_dtype))
            in_y1 = (y1 >= 0.0) & (y1 <= tf.cast(H - 1, compute_dtype))

            x0c = tf.clip_by_value(x0, 0.0, tf.cast(W - 1, compute_dtype))
            y0c = tf.clip_by_value(y0, 0.0, tf.cast(H - 1, compute_dtype))
            x1c = tf.clip_by_value(x1, 0.0, tf.cast(W - 1, compute_dtype))
            y1c = tf.clip_by_value(y1, 0.0, tf.cast(H - 1, compute_dtype))

            Ia = _gather_bhw(images, x0c, y0c, H, W)
            Ib = _gather_bhw(images, x1c, y0c, H, W)
            Ic = _gather_bhw(images, x0c, y1c, H, W)
            Id = _gather_bhw(images, x1c, y1c, H, W)

            Ia *= tf.cast(tf.expand_dims(tf.reshape(in_x0 & in_y0, [B, H, W]), -1), images.dtype)
            Ib *= tf.cast(tf.expand_dims(tf.reshape(in_x1 & in_y0, [B, H, W]), -1), images.dtype)
            Ic *= tf.cast(tf.expand_dims(tf.reshape(in_x0 & in_y1, [B, H, W]), -1), images.dtype)
            Id *= tf.cast(tf.expand_dims(tf.reshape(in_x1 & in_y1, [B, H, W]), -1), images.dtype)

            wx = tf.expand_dims(sx - x0, -1)
            wy = tf.expand_dims(sy - y0, -1)
            # Cast weights to same dtype as images for multiplication
            wx = tf.cast(wx, images.dtype)
            wy = tf.cast(wy, images.dtype)
            wa = (1.0 - wx) * (1.0 - wy)
            wb = wx * (1.0 - wy)
            wc = (1.0 - wx) * wy
            wd = wx * wy
            out = wa * Ia + wb * Ib + wc * Ic + wd * Id

    out = tf.cast(out, img_dtype) if img_dtype.is_floating else tf.cast(out, img_dtype)
    return out


@tf.function(jit_compile=True)
def projective_warp_xla_jit(images: tf.Tensor, transforms: tf.Tensor,
                            interpolation: str = "bilinear",
                            fill_mode: str = "edge") -> tf.Tensor:
    """JIT‑compiled wrapper to encourage XLA fusion.
    Note: `interpolation` and `fill_mode` are treated as constants in the graph.
    """
    return projective_warp_xla(images, transforms,
                               interpolation=interpolation,
                               fill_mode=fill_mode)


# ------------------------------
# Internal helpers
# ------------------------------

def _gather_bhw(images: tf.Tensor, ix: tf.Tensor, iy: tf.Tensor, H: tf.Tensor, W: tf.Tensor) -> tf.Tensor:
    """Gather pixels at integer indices (broadcasted over batch/H/W).

    Args:
      images: [B,H,W,C] float32.
      ix, iy: [B,H,W] float32 (will be cast to int32 inside).
    Returns:
      gathered: [B,H,W,C]
    """
    B = tf.shape(images)[0]
    C = tf.shape(images)[3]
    ix = tf.cast(ix, tf.int32)
    iy = tf.cast(iy, tf.int32)
    flat_idx = iy * tf.cast(W, tf.int32) + ix              # [B,H,W]
    flat_idx = tf.reshape(flat_idx, [B, -1])               # [B,HW]
    flat_img = tf.reshape(images, [B, -1, C])              # [B,HW,C]
    gathered = tf.gather(flat_img, flat_idx, batch_dims=1) # [B,HW,C]
    return tf.reshape(gathered, [B, H, W, C])


# ------------------------------
# PtychoPINN Integration
# ------------------------------

def translate_xla(images: tf.Tensor, translations: tf.Tensor, 
                  interpolation: str = 'bilinear',
                  use_jit: bool = True) -> tf.Tensor:
    """PtychoPINN-compatible wrapper for XLA projective warp.
    
    Args:
        images: [B,H,W,C] (can be complex)
        translations: [B,2] with [dx,dy] order (PtychoPINN convention)
        interpolation: 'bilinear' or 'nearest'
        use_jit: Whether to use JIT-compiled version
        
    Returns:
        Translated images with same shape and dtype as input
    """
    # Handle complex images by splitting, processing, and recombining
    if images.dtype in [tf.complex64, tf.complex128]:
        real_dtype = tf.float32 if images.dtype == tf.complex64 else tf.float64
        real_part = tf.cast(tf.math.real(images), real_dtype)
        imag_part = tf.cast(tf.math.imag(images), real_dtype)
        
        real_out = translate_xla(real_part, translations, interpolation, use_jit)
        imag_out = translate_xla(imag_part, translations, interpolation, use_jit)
        
        return tf.complex(real_out, imag_out)
    
    # Ensure translations has correct shape
    translations = tf.ensure_shape(translations, [None, 2])
    
    # Build translation-only homography matrices
    B = tf.shape(translations)[0]
    # Use same dtype as images for consistency
    matrix_dtype = images.dtype.real_dtype if images.dtype.is_complex else images.dtype
    
    # Convert translations to homography matrices
    # PtychoPINN convention: negate dx,dy (positive values move content in positive direction)
    dx = -translations[:, 0]
    dy = -translations[:, 1]
    # Cast to matrix dtype
    dx = tf.cast(dx, matrix_dtype)
    dy = tf.cast(dy, matrix_dtype)
    ones = tf.ones([B], dtype=matrix_dtype)
    zeros = tf.zeros([B], dtype=matrix_dtype)
    
    # Translation matrix: [[1,0,dx],[0,1,dy],[0,0,1]]
    row0 = tf.stack([ones, zeros, dx], axis=-1)
    row1 = tf.stack([zeros, ones, dy], axis=-1)
    row2 = tf.stack([zeros, zeros, ones], axis=-1)
    M = tf.stack([row0, row1, row2], axis=-2)  # [B,3,3]
    
    # Use XLA-friendly warp (with or without JIT)
    # Match original implementation: use 'zeros' fill mode
    if use_jit:
        return projective_warp_xla_jit(images, M, 
                                      interpolation=interpolation,
                                      fill_mode='zeros')  # Match original: CONSTANT with 0
    else:
        return projective_warp_xla(images, M, 
                                  interpolation=interpolation,
                                  fill_mode='zeros')  # Match original: CONSTANT with 0
