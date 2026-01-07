Mode: Implementation
Focus: REFACTOR-MODEL-SINGLETON-001 — Phase C (XLA Re-enablement Spike)
Selector: tests/test_model_factory.py::TestMultiNModelCreation::test_multi_n_model_creation

## Summary

Phase C investigates whether XLA translation can be re-enabled now that lazy loading (Phase B) prevents import-time model creation. This is a **spike test** to verify the hypothesis before removing workarounds.

## Goal

Determine whether the XLA workarounds (Phase A) are still necessary, or if lazy loading alone fixes the multi-N shape mismatch bug.

**Hypothesis:** With lazy loading in place, `create_model_with_gridsize()` can be called multiple times with different N values in the same process, and XLA translation will work because:
1. No models are created at import time (lazy loading)
2. TensorFlow's XLA polymorphic compilation handles different shapes
3. Each model creation starts fresh without stale traces

## Tasks

### C-SPIKE-1: Create XLA spike test

**File:** `tests/test_model_factory.py`

Add a new test class `TestXLAReenablement` with a spike test that verifies XLA works for multi-N:

```python
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
```

### C-SPIKE-2: Run the spike test

```bash
# Run just the XLA spike test
pytest tests/test_model_factory.py::TestXLAReenablement::test_multi_n_with_xla_enabled -vv 2>&1 | tee plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T050000Z/pytest_phase_c_spike.log
```

### C-SPIKE-3: Decision gate

Based on the spike test result:

**If PASS:**
- The lazy loading fix is sufficient
- Proceed to C1-C4 (remove workarounds from dose_response_study.py and tests)
- Update implementation.md Phase C checklist

**If FAIL:**
- Document the specific error in `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T050000Z/blocker_xla_spike.md`
- Phase C is blocked pending further investigation
- The XLA workarounds remain necessary even with lazy loading
- Update fix_plan.md with blocker status

## Pitfalls To Avoid

1. **DO NOT** remove the XLA workarounds from dose_response_study.py until the spike passes
2. **DO NOT** set `USE_XLA_TRANSLATE=0` in the spike test - the whole point is to test WITH XLA
3. **DO** run the spike in a subprocess to ensure clean Python state
4. **DO** verify `model._lazy_cache == {}` after import to confirm lazy loading works
5. **DO** test both N=128→N=64 sequence (the original bug scenario)
6. **DO** capture full stdout/stderr for debugging if it fails

## Artifacts

- Reports: `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T050000Z/`
- Spike test log: `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T050000Z/pytest_phase_c_spike.log`
- Blocker (if fail): `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T050000Z/blocker_xla_spike.md`

## Findings Applied

- **MODULE-SINGLETON-001**: This spike tests whether lazy loading is the complete fix
- **TF-NON-XLA-SHAPE-001**: The non-XLA path issues should not affect this test (XLA enabled)
- **CONFIG-001**: Params must be set before model creation

## Pointers

- Implementation plan: `plans/active/REFACTOR-MODEL-SINGLETON-001/implementation.md` (Phase C checklist)
- Lazy loading implementation: `ptycho/model.py:867-890` (`__getattr__`)
- XLA JIT function: `ptycho/projective_warp_xla.py:202` (`projective_warp_xla_jit`)
- Translation layer XLA check: `ptycho/tf_helper.py:154` (`should_use_xla()`)
- Factory function: `ptycho/model.py:681-811` (`create_model_with_gridsize`)

## If Blocked

If the spike fails:
1. Log the specific error message and stack trace
2. Create `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T050000Z/blocker_xla_spike.md`
3. Analyze whether the issue is:
   - XLA trace persistence (module-level `@tf.function` cache)
   - Translation layer instantiation order
   - Something else in the model construction path
4. Mark Phase C as blocked in implementation.md
5. Consider whether alternative approaches exist (e.g., XLA only for single-N workflows)

## Next Up (Optional)

If spike passes, proceed with:
- C1: Remove XLA workarounds from `scripts/studies/dose_response_study.py`
- C2: Remove workarounds from `tests/test_model_factory.py` (keep new XLA test)
- C3: Performance verification
- C4: Update `docs/findings.md` MODULE-SINGLETON-001 to mark fully resolved
