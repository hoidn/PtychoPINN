"""Tests for model factory functions.

Tests MODULE-SINGLETON-001 fix: models created with different N values
must work correctly in a single process.

This module sets USE_XLA_TRANSLATE=0 BEFORE any ptycho imports to prevent
XLA trace caching bugs when creating models with different N values.

Ref: REFACTOR-MODEL-SINGLETON-001, TF-NON-XLA-SHAPE-001, CONFIG-001
"""
import os
import pytest

# CRITICAL: Set environment BEFORE any ptycho imports to avoid XLA trace caching
# See docs/findings.md MODULE-SINGLETON-001 and TF-NON-XLA-SHAPE-001
os.environ['USE_XLA_TRANSLATE'] = '0'

# Also disable TensorFlow's XLA JIT to prevent compile-time constant errors
# in tf.repeat during graph execution (the non-XLA translate path)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

import tensorflow as tf
import numpy as np

# Force eager execution to avoid XLA compilation of the graph
# This is necessary because Keras 3.x uses XLA JIT by default for model.predict()
tf.config.run_functions_eagerly(True)


def _init_params_for_model(N: int, gridsize: int = 2):
    """Initialize params.cfg with all values required by ptycho.model at import time.

    The ptycho.model module has extensive module-level code that accesses params.cfg.
    This function sets up all the required params before importing model.

    Required by MODULE-SINGLETON-001: ptycho.model.py lines 144-239 access:
    - N, gridsize, offset (line 144-146)
    - probe (line 149)
    - intensity_scale (line 239)
    - nphotons (line 235)
    """
    from ptycho import params as p
    from ptycho import probe

    # Set core params
    p.cfg['N'] = N
    p.cfg['gridsize'] = gridsize
    p.cfg['offset'] = 4  # default offset

    # Probe initialization requires default_probe_scale
    if 'default_probe_scale' not in p.cfg or p.cfg['default_probe_scale'] is None:
        p.cfg['default_probe_scale'] = 4.0

    # Create and set default probe
    default_probe = probe.get_default_probe(N, fmt='tf')
    p.cfg['probe'] = default_probe

    # Set intensity_scale (required by model.py:239)
    if 'intensity_scale' not in p.cfg:
        p.cfg['intensity_scale'] = 1.0

    # nphotons (required by model.py:235)
    if 'nphotons' not in p.cfg:
        p.cfg['nphotons'] = 1e9


class TestMultiNModelCreation:
    """Test that models with different N values can coexist."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear session before each test."""
        tf.keras.backend.clear_session()
        yield
        tf.keras.backend.clear_session()

    def test_multi_n_model_creation(self):
        """Verify models with different N values don't cause shape mismatch.

        This test reproduces the bug from dose_response_study.py where creating
        models with N=128 then N=64 caused XLA trace shape conflicts.

        The root cause was:
        1. `from ptycho import model` triggers module-level code that creates models
        2. During model construction, Translation layers call should_use_xla() -> True
        3. projective_warp_xla_jit traces with N=128 shapes
        4. XLA traces persist at Python module level (clear_session doesn't clear them)
        5. Later create_model_with_gridsize(N=64) sets use_xla_translate=False but old traces exist
        6. Translation layer executes with stale XLA trace expecting N=128 shapes, crashes on N=64

        Fix: Set USE_XLA_TRANSLATE=0 environment variable BEFORE any ptycho imports.

        Ref: REFACTOR-MODEL-SINGLETON-001, TF-NON-XLA-SHAPE-001
        """
        # Initialize params BEFORE importing model (required by MODULE-SINGLETON-001)
        _init_params_for_model(N=128, gridsize=2)

        from ptycho import params as p
        from ptycho.model import create_model_with_gridsize

        gridsize = 2

        # First model: N=128
        p.cfg['N'] = 128
        p.cfg['gridsize'] = gridsize
        tf.keras.backend.clear_session()

        autoenc_128, d2o_128 = create_model_with_gridsize(gridsize, N=128)

        # Verify shapes
        assert autoenc_128.input_shape[0] == (None, 128, 128, gridsize**2), \
            f"Expected input shape (None, 128, 128, {gridsize**2}), got {autoenc_128.input_shape[0]}"

        # Run a forward pass to trigger any lazy tracing
        dummy_input_128 = [
            np.random.randn(1, 128, 128, gridsize**2).astype(np.float32),
            np.random.randn(1, 1, 2, gridsize**2).astype(np.float32)
        ]
        _ = autoenc_128.predict(dummy_input_128, verbose=0)

        # Second model: N=64 (THIS SHOULD NOT CRASH)
        p.cfg['N'] = 64
        tf.keras.backend.clear_session()

        autoenc_64, d2o_64 = create_model_with_gridsize(gridsize, N=64)

        # Verify shapes
        assert autoenc_64.input_shape[0] == (None, 64, 64, gridsize**2), \
            f"Expected input shape (None, 64, 64, {gridsize**2}), got {autoenc_64.input_shape[0]}"

        # Run a forward pass - this is where the XLA shape mismatch would occur
        dummy_input_64 = [
            np.random.randn(1, 64, 64, gridsize**2).astype(np.float32),
            np.random.randn(1, 1, 2, gridsize**2).astype(np.float32)
        ]
        try:
            _ = autoenc_64.predict(dummy_input_64, verbose=0)
        except tf.errors.InvalidArgumentError as e:
            pytest.fail(f"XLA shape mismatch bug not fixed: {e}")

        # Verify different models have different shapes
        assert autoenc_128.input_shape[0][1] == 128
        assert autoenc_64.input_shape[0][1] == 64
