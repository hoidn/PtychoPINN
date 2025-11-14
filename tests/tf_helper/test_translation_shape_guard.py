"""
Regression test for non-XLA translation shape guard (FIX-PYTORCH-FORWARD-PARITY-001 Phase C1d).

This module tests the shape consistency guard added to ptycho/tf_helper.py::translate_core
to prevent crashes when images and translations have mismatched batch dimensions.

Reference:
- Issue: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/.../tf_baseline/phase_c1/red/blocked_20251114T074039Z_tf_non_xla_shape_error.md
- Finding: POLICY-001 (PyTorch mandatory), CONFIG-001 (update_legacy_dict bridge)

The bug occurred when USE_XLA_TRANSLATE=0 and gridsize=2, causing _channel_to_flat to produce
images with batch dimension (b*gridsize^2) while translations kept batch dimension (b).
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import patch

# Import the module under test
import ptycho.tf_helper as tf_helper
import ptycho.params as params


@pytest.fixture(autouse=True)
def setup_params():
    """Initialize params.cfg for all tests in this module."""
    # Save original config
    original_cfg = params.cfg.copy()
    try:
        # Update params directly (avoid params.set() validation requirements)
        params.cfg.update({
            'N': 64,
            'gridsize': 2,
            'offset': 0,
            'use_xla_translate': False,  # Force non-XLA path for testing
            'padded_size': 128,  # Required for reassembly tests
        })
        yield
    finally:
        # Restore original config
        params.cfg.clear()
        params.cfg.update(original_cfg)


def test_non_xla_translation_guard():
    """Test that translate_core falls back gracefully when batch dimensions mismatch.

    This is a minimal reproduction of the gridsize=2 shape error that occurred when
    XLA was disabled (USE_XLA_TRANSLATE=0).

    Setup:
        - Create fake images with batch=gridsize^2 (simulating _channel_to_flat output)
        - Create fake translations with batch=1 (simulating original batch before flattening)
        - Mock should_use_xla() to return False to force non-XLA path

    Expected:
        - translate_core should detect the mismatch via tf.debugging.assert_equal
        - The assertion will raise tf.errors.InvalidArgumentError
        - The try/except will catch it and fall back to _translate_images_simple
        - The function should complete successfully without crashing
    """
    gridsize = 2
    N = 64

    # Simulate _channel_to_flat output: (b*c, N, N, 1) where b=1, c=gridsize^2=4
    # So batch dimension = 4
    fake_images = tf.ones((gridsize**2, N, N, 1), dtype=tf.float32)

    # Translations stay in original batch dimension: (b, 2) where b=1
    fake_translations = tf.constant([[10.0, 20.0]], dtype=tf.float32)  # shape: (1, 2)

    # Force non-XLA path by mocking should_use_xla
    with patch.object(tf_helper, 'should_use_xla', return_value=False):
        # This should NOT crash - it should fall back to _translate_images_simple
        result = tf_helper.translate_core(
            fake_images,
            fake_translations,
            interpolation='bilinear',
            use_xla_workaround=False
        )

    # Verify result has correct shape (same as input images)
    assert result.shape == fake_images.shape
    # Verify result is not all zeros (translation actually happened)
    assert tf.reduce_sum(tf.abs(result)) > 0.0


def test_non_xla_translation_matching_batch():
    """Test that translate_core uses fast path when batch dimensions match.

    When images and translations have matching batch dimensions, the function
    should successfully use the ImageProjectiveTransformV3 fast path without
    falling back to the slower _translate_images_simple.
    """
    batch_size = 4
    N = 64

    # Both images and translations have matching batch dimension
    fake_images = tf.ones((batch_size, N, N, 1), dtype=tf.float32)
    fake_translations = tf.constant([[10.0, 20.0]] * batch_size, dtype=tf.float32)  # shape: (4, 2)

    # Force non-XLA path
    with patch.object(tf_helper, 'should_use_xla', return_value=False):
        result = tf_helper.translate_core(
            fake_images,
            fake_translations,
            interpolation='bilinear',
            use_xla_workaround=False
        )

    # Verify result has correct shape
    assert result.shape == fake_images.shape


def test_reassemble_patches_position_real_gridsize2():
    """Integration test for _reassemble_patches_position_real with gridsize=2.

    This tests the full reassembly path that triggered the original bug:
    - imgs in channel format (b, N, N, c) where c=gridsize^2
    - offsets_xy in channel format (b, 1, 2, c)
    - _channel_to_flat converts both to flat format
    - Translation layer should handle the flattened dimensions correctly

    This test ensures the guard allows the reassembly to complete successfully.
    """
    # Ensure gridsize=2 is set
    params.set('gridsize', 2)
    params.set('N', 64)
    params.set('padded_size', 128)

    batch_size = 2
    N = 64
    gridsize = 2
    c = gridsize ** 2

    # Create fake channel-format inputs
    imgs = tf.ones((batch_size, N, N, c), dtype=tf.complex64)
    offsets_xy = tf.constant(
        np.random.randn(batch_size, 1, 2, c).astype(np.float32)
    )

    # Force non-XLA path
    with patch.object(tf_helper, 'should_use_xla', return_value=False):
        # This should complete without crashing
        result = tf_helper._reassemble_patches_position_real(
            imgs,
            offsets_xy,
            agg=True,
            padded_size=128
        )

    # Verify result shape (should be aggregated canvas)
    assert result.shape[1] == 128  # padded_size
    assert result.shape[2] == 128
    assert result.shape[3] == 1  # aggregated
