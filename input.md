Mode: Implementation
Focus: REFACTOR-MODEL-SINGLETON-001 — Phase A (Environment Variable Fix + Test)
Selector: tests/test_model_factory.py::test_multi_n_model_creation

## Summary

Fix the XLA trace caching bug by setting `USE_XLA_TRANSLATE=0` before importing `ptycho.model`. The root cause is that module-level model creation at import time creates XLA traces that persist even after `tf.keras.backend.clear_session()`.

## Root Cause (from supervisor analysis)

1. `from ptycho import model` triggers module-level code (lines 554-562) that creates `autoencoder`, `diffraction_to_obj`
2. During model construction, `Translation` layers call `should_use_xla()` → defaults to `True`
3. `projective_warp_xla_jit` (line 202, decorated with `@tf.function(jit_compile=True)`) traces with N=128 shapes
4. XLA traces persist at Python module level — `tf.keras.backend.clear_session()` doesn't clear them
5. Later `create_model_with_gridsize(N=64)` sets `use_xla_translate=False` but old traces still exist
6. When Translation layer executes, the stale XLA trace expects N=128 shapes, crashes on N=64 input

**Error signature:**
```
InvalidArgumentError: Input to reshape is a tensor with 389376 values, but the requested shape has 24336
  389376 = 78 × 78 × 64 (padded_size=78 for N=64)
  24336 = 156 × 156 (padded_size=156 for N=128 — stale trace)
```

## Fix Strategy

**Two-part fix:**
1. **Immediate stabilization:** Set `USE_XLA_TRANSLATE=0` environment variable at the START of any script that needs multiple N values (before any ptycho imports)
2. **Test coverage:** Create a regression test that verifies multi-N model creation works

## Tasks

### A0: Create regression test (RED → GREEN)
**File:** `tests/test_model_factory.py`

```python
"""Tests for model factory functions.

Tests MODULE-SINGLETON-001 fix: models created with different N values
must work correctly in a single process.
"""
import os
import pytest

# Set environment BEFORE any ptycho imports to avoid XLA trace caching
os.environ['USE_XLA_TRANSLATE'] = '0'

import tensorflow as tf
import numpy as np


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

        Ref: REFACTOR-MODEL-SINGLETON-001, TF-NON-XLA-SHAPE-001
        """
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
```

### A1: Update dose_response_study.py with environment fix
**File:** `scripts/studies/dose_response_study.py`

Add at the very top of the file (BEFORE any imports):
```python
#!/usr/bin/env python
"""Dose Response Study: Compare high vs low dose reconstructions.
...
"""
import os
# CRITICAL: Disable XLA translation to avoid shape caching issues
# when creating models with different N values. See MODULE-SINGLETON-001.
os.environ['USE_XLA_TRANSLATE'] = '0'

# Rest of imports follow...
```

### A2: Run test and verify
```bash
pytest tests/test_model_factory.py::test_multi_n_model_creation -vv 2>&1 | tee plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-06T180000Z/pytest_model_factory.log
```

Expected: PASS (GREEN)

### A3 (optional): Run dose_response_study.py to verify end-to-end
If A0-A2 pass, run the study script:
```bash
cd scripts/studies && python dose_response_study.py --dry-run 2>&1 | head -100
```

## Pitfalls To Avoid

1. **DO NOT** import any ptycho module before setting `USE_XLA_TRANSLATE=0`
2. **DO NOT** rely on `params.cfg['use_xla_translate'] = False` alone — this doesn't prevent import-time XLA tracing
3. **DO NOT** modify `projective_warp_xla.py` — changing the XLA decorator is out of scope
4. **DO** ensure the environment variable is set at module level (top of file), not inside functions
5. **DO** clear Keras session between model creations with `tf.keras.backend.clear_session()`

## Artifacts

- Reports: `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-06T180000Z/`
- Test log: `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-06T180000Z/pytest_model_factory.log`

## Findings Applied

- **MODULE-SINGLETON-001**: Module-level singletons capture shapes at import time; this fix prevents XLA tracing at import by disabling XLA before import.
- **TF-NON-XLA-SHAPE-001**: Non-XLA translation path is safer for multi-N scenarios.
- **CONFIG-001**: Environment variable takes precedence over params.cfg in `should_use_xla()`.

## Pointers

- Implementation plan: `plans/active/REFACTOR-MODEL-SINGLETON-001/implementation.md` (Phase A checklist)
- Translation layer: `ptycho/tf_helper.py:817-850` (uses `should_use_xla()`)
- XLA toggle: `ptycho/tf_helper.py:154-175` (`should_use_xla()` — checks env var first!)
- XLA decorator: `ptycho/projective_warp_xla.py:202` (`@tf.function(jit_compile=True)`)
- Model factory: `ptycho/model.py:681` (`create_model_with_gridsize`)
- Module-level model: `ptycho/model.py:554-562` (root cause of XLA trace pollution)

## Next Up (Optional)

If A0-A3 complete successfully:
- Update `plans/active/REFACTOR-MODEL-SINGLETON-001/implementation.md` Phase A checklist
- Move to Phase B (Module Variable Inventory + Lazy Loading)
