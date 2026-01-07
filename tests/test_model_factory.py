"""Tests for model factory functions.

Tests MODULE-SINGLETON-001 fix: models created with different N values
must work correctly in a single process.

Lazy loading in ptycho/model.py (via __getattr__) prevents import-time model
creation, eliminating XLA trace caching conflicts when changing N values.
No environment variable workarounds are needed.

Ref: REFACTOR-MODEL-SINGLETON-001, CONFIG-001
"""
import os
import pytest

import tensorflow as tf
import numpy as np


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

        This test validates that creating models with N=128 then N=64 works
        correctly without XLA trace shape conflicts.

        Fix (REFACTOR-MODEL-SINGLETON-001 Phase B): Lazy loading via __getattr__
        in ptycho/model.py prevents import-time model creation. Models are only
        created when explicitly requested via create_model_with_gridsize() or
        when module-level singletons are accessed. This eliminates XLA trace
        conflicts because each model creation starts fresh.

        Ref: REFACTOR-MODEL-SINGLETON-001, CONFIG-001
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


class TestImportSideEffects:
    """Test that importing ptycho.model doesn't create models.

    Exit Criterion for Phase B (REFACTOR-MODEL-SINGLETON-001):
    Importing the module should NOT:
    1. Create Keras models
    2. Instantiate tf.Variable
    3. Execute model graph construction
    """

    def test_import_no_side_effects(self):
        """Verify importing ptycho.model doesn't create models.

        This test runs in a subprocess to ensure a clean import state.

        Ref: REFACTOR-MODEL-SINGLETON-001 Phase B, ANTIPATTERN-001
        """
        import subprocess
        import sys
        from pathlib import Path

        # Run a subprocess that imports ptycho.model and checks for side effects
        # No XLA environment workarounds needed - lazy loading prevents import-time side effects
        code = '''
import sys

# Import the module without any environment workarounds
# Lazy loading should prevent any model creation at import time
from ptycho import model

# Check that no models were created at import time
# The _lazy_cache should be empty if no singletons were accessed
assert model._lazy_cache == {}, f"Models created at import: {list(model._lazy_cache.keys())}"
assert not model._model_construction_done, "Model construction ran at import time"

print("PASS: No side effects at import")
'''
        result = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )

        if result.returncode != 0:
            pytest.fail(f"Import side-effect test failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

        assert "PASS" in result.stdout


class TestXLAReenablement:
    """Test that XLA can be re-enabled after lazy loading fix.

    Phase C spike test for REFACTOR-MODEL-SINGLETON-001.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear session before each test."""
        tf.keras.backend.clear_session()
        yield
        tf.keras.backend.clear_session()

    def test_multi_n_with_xla_enabled(self):
        """Verify models with different N values work with XLA translation enabled.

        This test verifies the hypothesis that lazy loading (Phase B) fixes the
        XLA shape mismatch bug, allowing XLA to be re-enabled.

        Approach:
        1. Run in subprocess with clean Python state (no env var workarounds)
        2. Import ptycho.model (no side effects due to lazy loading)
        3. Create model with N=128, run forward pass
        4. Create model with N=64, run forward pass
        5. Verify no XLA shape mismatch errors

        If this test passes, Phase A workarounds can be removed.
        If it fails, document the specific error and Phase C is blocked.

        Ref: REFACTOR-MODEL-SINGLETON-001 Phase C
        """
        import subprocess
        import sys
        from pathlib import Path

        # CRITICAL: Do NOT set USE_XLA_TRANSLATE=0 - we want to test WITH XLA
        code = '''
import sys
import os

# Clear any XLA workaround env vars to test with XLA enabled
os.environ.pop('USE_XLA_TRANSLATE', None)
os.environ.pop('TF_XLA_FLAGS', None)

import tensorflow as tf

# Allow eager for Keras 3.x compatibility but don't disable XLA JIT
# tf.config.run_functions_eagerly(True)  # Intentionally commented out

print(f"XLA test starting, TF version: {tf.__version__}")

# Initialize params before importing model
from ptycho import params as p
from ptycho import probe

# Set up params for N=128 first
N1, N2 = 128, 64
gridsize = 2

p.cfg['N'] = N1
p.cfg['gridsize'] = gridsize
p.cfg['offset'] = 4
p.cfg['default_probe_scale'] = 4.0
p.cfg['probe'] = probe.get_default_probe(N1, fmt='tf')
p.cfg['intensity_scale'] = 1.0
p.cfg['nphotons'] = 1e9

print(f"Params initialized for N={N1}")

# Import model - should NOT create models (lazy loading)
from ptycho import model

# Verify lazy loading worked
assert model._lazy_cache == {}, f"Models created at import: {list(model._lazy_cache.keys())}"
print("Lazy loading verified: no models at import")

from ptycho.model import create_model_with_gridsize
import numpy as np

# Create first model with N=128
print(f"Creating model with N={N1}...")
tf.keras.backend.clear_session()
autoenc_128, d2o_128 = create_model_with_gridsize(gridsize, N=N1)

# Run forward pass to trigger any XLA tracing
dummy_128 = [
    np.random.randn(1, N1, N1, gridsize**2).astype(np.float32),
    np.random.randn(1, 1, 2, gridsize**2).astype(np.float32)
]
print(f"Running forward pass for N={N1}...")
out_128 = autoenc_128.predict(dummy_128, verbose=0)
print(f"Forward pass N={N1} succeeded, output shapes: {[o.shape for o in out_128]}")

# Create second model with N=64 (the bug scenario)
print(f"Creating model with N={N2}...")
p.cfg['N'] = N2
p.cfg['probe'] = probe.get_default_probe(N2, fmt='tf')
tf.keras.backend.clear_session()

autoenc_64, d2o_64 = create_model_with_gridsize(gridsize, N=N2)

# Run forward pass - THIS IS WHERE THE XLA BUG WOULD OCCUR
dummy_64 = [
    np.random.randn(1, N2, N2, gridsize**2).astype(np.float32),
    np.random.randn(1, 1, 2, gridsize**2).astype(np.float32)
]
print(f"Running forward pass for N={N2}...")
try:
    out_64 = autoenc_64.predict(dummy_64, verbose=0)
    print(f"Forward pass N={N2} succeeded, output shapes: {[o.shape for o in out_64]}")
    print("PASS: XLA re-enablement spike test succeeded")
except tf.errors.InvalidArgumentError as e:
    print(f"FAIL: XLA shape mismatch error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FAIL: Unexpected error: {type(e).__name__}: {e}")
    sys.exit(1)
'''

        result = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            timeout=120
        )

        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")

        if result.returncode != 0:
            pytest.fail(
                f"XLA re-enablement spike failed:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        assert "PASS" in result.stdout, "Expected PASS in output"
