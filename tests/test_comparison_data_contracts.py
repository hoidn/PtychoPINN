"""Regression contracts for live comparison and reassembly helpers."""

from types import SimpleNamespace

import numpy as np
import pytest


def test_prepare_baseline_inference_data_flattens_grouped_channels():
    from scripts.compare_models import prepare_baseline_inference_data

    container = SimpleNamespace(
        X=np.zeros((2, 16, 16, 4), dtype=np.float32),
        global_offsets=np.zeros((2, 1, 2, 1), dtype=np.float32),
        local_offsets=None,
    )

    diffraction, offsets = prepare_baseline_inference_data(container)

    assert diffraction.shape == (8, 16, 16, 1)
    assert offsets.shape == (8, 1, 2, 1)

    container.X = np.zeros((2, 16, 16, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="must be a perfect square"):
        prepare_baseline_inference_data(container)


def test_reassemble_patches_layer_handles_more_than_one_batch():
    import tensorflow as tf

    from ptycho.custom_layers import ReassemblePatchesLayer

    # 17 groups * 4 channels crosses the layer's 64-patch batching boundary.
    patches = tf.complex(
        tf.ones((17, 16, 16, 4), dtype=tf.float32),
        tf.zeros((17, 16, 16, 4), dtype=tf.float32),
    )
    positions = tf.zeros((17, 1, 2, 4), dtype=tf.float32)

    result = ReassemblePatchesLayer(
        padded_size=20,
        N=16,
        gridsize=2,
    )([patches, positions])

    assert result.shape == (1, 20, 20, 1)
    assert bool(tf.reduce_all(tf.math.is_finite(tf.math.real(result))))
    assert bool(tf.reduce_all(tf.math.is_finite(tf.math.imag(result))))
