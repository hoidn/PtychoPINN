Mode: Implementation
Focus: REFACTOR-MODEL-SINGLETON-001 — Phase B (Lazy Loading + Import Cleanup)
Selector: tests/test_model_factory.py::test_multi_n_model_creation
Additional: tests/test_model_factory.py::test_import_no_side_effects (NEW)

## Summary

Refactor `ptycho/model.py` to eliminate import-time side effects by implementing lazy loading for the legacy module-level singletons (`autoencoder`, `diffraction_to_obj`).

## Goal

After Phase B, importing `ptycho.model` MUST NOT:
1. Create Keras models
2. Instantiate tf.Variable objects
3. Execute model graph construction

Legacy code doing `from ptycho.model import autoencoder` will still work via `__getattr__` lazy loading, but with a DeprecationWarning.

## Current State (Post Phase A)

- **✅ Phase A complete**: XLA workaround applied, `test_multi_n_model_creation` passes
- **Problem remaining**: Lines 464-562 of model.py execute at import time, creating:
  - `normed_input` (line 464)
  - `obj` (line 473)
  - `padded_obj_2` (lines 480-494)
  - `trimmed_obj` (line 498)
  - `padded_objs_with_offsets` (lines 502-509)
  - `pred_diff` (line 517)
  - `pred_amp_scaled` (line 520)
  - `pred_intensity_sampled` (line 527)
  - `autoencoder` (line 554)
  - `autoencoder_no_nll` (line 556)
  - `diffraction_to_obj` (line 561)

## Tasks

### B4: Implement Lazy Loading via `__getattr__`

**File:** `ptycho/model.py`

**Pattern:** Move ALL model construction (lines 464-593) into a conditional block that only executes on first access to the singletons.

**Step 1:** Add at module level (near top, after imports but before any model code):

```python
# Module-level cache for lazy-loaded singletons
_lazy_cache = {}
_model_construction_done = False
```

**Step 2:** Wrap the model construction block (lines 464-593) in a function:

```python
def _build_module_level_models():
    """Build module-level models on first access (lazy loading).

    This function encapsulates all the model construction code that previously
    ran at import time. It's called by __getattr__ on first access to
    autoencoder, diffraction_to_obj, or autoencoder_no_nll.

    Warning: This function modifies module-level state. It should only be
    called once per process via the _model_construction_done guard.
    """
    global _model_construction_done
    if _model_construction_done:
        return

    # Get current params (whatever is set when models are first accessed)
    N = p.get('N')
    gridsize = p.get('gridsize')
    n_filters_scale = p.get('n_filters_scale')

    # === Begin model construction (moved from module level) ===
    input_img = Input(shape=(N, N, gridsize**2), name='input')
    input_positions = Input(shape=(1, 2, gridsize**2), name='input_positions')

    normed_input = IntensityScaler(name='intensity_scaler')([input_img])
    decoded1, decoded2 = create_autoencoder(normed_input, n_filters_scale, gridsize, p.get('object.big'))

    obj = CombineComplexLayer(name='obj')([decoded1, decoded2])

    # ... (rest of model construction from lines 475-527)

    # Create models
    autoencoder = Model([input_img, input_positions], [trimmed_obj, pred_amp_scaled, pred_intensity_sampled])
    diffraction_to_obj = tf.keras.Model(inputs=[input_img, input_positions], outputs=[trimmed_obj])
    autoencoder_no_nll = Model(inputs=[input_img, input_positions], outputs=[pred_amp_scaled])

    # Store in cache
    _lazy_cache['autoencoder'] = autoencoder
    _lazy_cache['diffraction_to_obj'] = diffraction_to_obj
    _lazy_cache['autoencoder_no_nll'] = autoencoder_no_nll

    # Compile autoencoder (moved from module level)
    mae_weight = p.get('mae_weight')
    nll_weight = p.get('nll_weight')
    realspace_weight = p.get('realspace_weight')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    use_xla_compile = os.environ.get('USE_XLA_COMPILE', '').lower() in ('1', 'true', 'yes')
    try:
        use_xla_compile = use_xla_compile or p.get('use_xla_compile')
    except KeyError:
        pass

    autoencoder.compile(
        optimizer=optimizer,
        loss=[hh.realspace_loss, 'mean_absolute_error', negloglik],
        loss_weights=[realspace_weight, mae_weight, nll_weight],
        jit_compile=use_xla_compile
    )

    _model_construction_done = True
```

**Step 3:** Add `__getattr__` at module bottom:

```python
def __getattr__(name):
    """Lazy load module-level singletons on first access.

    Implements MODULE-SINGLETON-001 fix: model construction is deferred until
    first access, allowing params.cfg to be configured before models are built.

    Emits DeprecationWarning for legacy singleton access.
    """
    import warnings

    if name in ('autoencoder', 'diffraction_to_obj', 'autoencoder_no_nll'):
        if name not in _lazy_cache:
            warnings.warn(
                f"Accessing deprecated module-level singleton '{name}'. "
                "Use create_compiled_model() or create_model_with_gridsize() instead. "
                "See docs/findings.md MODULE-SINGLETON-001 for migration guide.",
                DeprecationWarning,
                stacklevel=2
            )
            _build_module_level_models()
        return _lazy_cache[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

**Step 4:** Delete the module-level model construction code (lines 464-593) that now lives in `_build_module_level_models()`.

**Step 5:** Also delete the module-level probe initialization (lines 148-165) and move it inside `_build_module_level_models()` or make it lazy as well.

### B-TEST: Create test for import side-effects

**File:** `tests/test_model_factory.py`

Add new test:

```python
def test_import_no_side_effects():
    """Verify importing ptycho.model doesn't create models.

    Exit Criterion for Phase B: importing the module should NOT:
    1. Create Keras models
    2. Instantiate tf.Variable
    3. Execute model graph construction

    Ref: REFACTOR-MODEL-SINGLETON-001 Phase B, ANTIPATTERN-001
    """
    import subprocess
    import sys

    # Run a subprocess that imports ptycho.model and checks for side effects
    code = '''
import sys
import tensorflow as tf

# Count existing variables before import
initial_vars = len(tf.Variable._variable_count_by_device) if hasattr(tf.Variable, '_variable_count_by_device') else 0

# Import the module
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
```

### B-VERIFY: Run tests

```bash
# Run both tests
pytest tests/test_model_factory.py -vv 2>&1 | tee plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T040000Z/pytest_phase_b.log
```

Expected:
- `test_multi_n_model_creation` — PASS (existing Phase A test still works)
- `test_import_no_side_effects` — PASS (new Phase B criterion)

## Pitfalls To Avoid

1. **DO NOT** leave any tf.Variable creation at module level (e.g., `log_scale`, `initial_probe_guess`)
2. **DO NOT** break backward compatibility — existing `from ptycho.model import autoencoder` must still work
3. **DO** emit DeprecationWarning so users know to migrate
4. **DO** ensure `_build_module_level_models()` only runs once (guard with `_model_construction_done`)
5. **DO** preserve compilation settings (optimizer, loss weights, jit_compile) in the lazy builder
6. **DO NOT** modify any layers in `custom_layers.py` — this is a model.py-only refactor

## Artifacts

- Reports: `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T040000Z/`
- Test log: `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T040000Z/pytest_phase_b.log`

## Findings Applied

- **MODULE-SINGLETON-001**: This is the definitive fix — lazy loading prevents import-time model creation.
- **ANTIPATTERN-001**: Import-time side effects are eliminated by deferring to first access.
- **CONFIG-001**: params.cfg can now be configured BEFORE models are built (lazy loading honors current config).

## Pointers

- Implementation plan: `plans/active/REFACTOR-MODEL-SINGLETON-001/implementation.md` (Phase B checklist items B0-B5)
- Module-level model construction to refactor: `ptycho/model.py:464-593`
- Module-level probe init to refactor: `ptycho/model.py:148-165`
- Module-level log_scale init: `ptycho/model.py:239-243`
- Factory function (reference): `ptycho/model.py:681-793` (`create_model_with_gridsize`)
- `__getattr__` pattern: Python 3 module `__getattr__` (PEP 562)

## If Blocked

If you encounter issues:
1. Log the specific error to `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T040000Z/blocker.md`
2. Run `pytest tests/test_model_factory.py::test_multi_n_model_creation -vv` to verify Phase A still works
3. Check that all module-level variable assignments are moved inside `_build_module_level_models()`

## Next Up (Optional)

If Phase B completes:
- Update `plans/active/REFACTOR-MODEL-SINGLETON-001/implementation.md` Phase B checklist
- Unblock STUDY-SYNTH-DOSE-COMPARISON-001 (depends on this fix)
