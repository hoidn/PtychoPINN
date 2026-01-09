# BUG-TF-REASSEMBLE-MIXED-TYPES-001: Mixed Types in tf.zeros Shape Construction

**Status:** Open
**Severity:** High
**Affected Component:** `ptycho/tf_helper.py`, `_reassemble_position_batched()`
**Reporter:** Claude Code
**Date:** 2026-01-09

---

## Summary

The batched patch reassembly function `_reassemble_position_batched()` fails with `ValueError: Can't convert Python sequence with mixed types to Tensor` when constructing tensor shapes. The error occurs at line 936 in `tf_helper.py` where `tf.zeros((1, padded_size, padded_size, 1), ...)` is called with a `padded_size` value that has an incompatible type (likely numpy scalar or tensor) mixed with Python int literals.

---

## Symptoms

- `test_pinn_reconstruction_reassembles_full_train_split` fails
- Error: `ValueError: Can't convert Python sequence with mixed types to Tensor`
- Traceback points to `ptycho/tf_helper.py:936` in `batched_approach()` function
- Occurs when processing large datasets (159 batches × 4 channels = 636 patches) that trigger batched processing

---

## Root Cause

### The Error Location: `tf_helper.py:936`
```python
def batched_approach():
    # Initialize the canvas with zeros
    final_canvas = tf.zeros((1, padded_size, padded_size, 1), dtype=imgs_flat.dtype)
```

### The Problem

TensorFlow's `tf.zeros()` requires shape elements to be of consistent types. The tuple `(1, padded_size, padded_size, 1)` contains:
- Python int literals: `1`, `1`
- `padded_size`: potentially a numpy scalar (`numpy.int64`) or TensorFlow tensor

When `padded_size` is a numpy scalar rather than a Python `int`, TensorFlow cannot construct a valid shape from the mixed-type sequence.

### Type Flow Analysis

1. `ReassemblePatchesLayer.__init__()` accepts `padded_size: Optional[int] = None`
2. When `None`, `mk_reassemble_position_batched_real()` calls `get_padded_size()` which returns a Python int
3. When explicitly passed, the value could be:
   - A numpy scalar from array indexing (e.g., `array[0]` returns `numpy.int64`)
   - A TensorFlow tensor dimension from dynamic shape operations
   - A Python int (correct)

### Related Failures

The same underlying issue affects `TestReassemblePosition` tests:
- `test_batched_reassembly_memory_efficiency`
- `test_batched_reassembly_shape_consistency`
- `test_reassembly_deterministic_across_methods`
- `test_shift_and_sum_shape_handling`

All these failures involve `shift_and_sum` / `_reassemble_position_batched` and the same mixed-type tensor shape construction.

---

## Impact

- Batched patch reassembly is unusable for large datasets
- PINN reconstruction tests fail
- The batched reassembly path was designed to prevent OOM errors on large datasets; its failure forces fallback to the non-batched path which may cause memory issues

---

## Recommended Fix

**Option A: Explicit Type Conversion (Preferred)**

Cast `padded_size` to Python `int` before use in shape construction:

```python
def batched_approach():
    # Ensure padded_size is a Python int for tensor shape construction
    ps = int(padded_size)  # Convert numpy scalar/tensor to Python int
    final_canvas = tf.zeros((1, ps, ps, 1), dtype=imgs_flat.dtype)
```

Apply the same fix to all shape constructions in the function:
- Line 936: `tf.zeros` shape
- Line 976: `tf.image.resize_with_crop_or_pad` dimensions

**Option B: Validate at Entry Point**

Add type validation in `_reassemble_position_batched()`:

```python
def _reassemble_position_batched(imgs: tf.Tensor, offsets_xy: tf.Tensor,
                                  padded_size: int, ...):
    # Ensure padded_size is a Python int
    if not isinstance(padded_size, int):
        padded_size = int(padded_size)
```

**Option C: Fix at Source**

Ensure callers always pass Python `int`:
- `custom_layers.py:168`: Cast `self.padded_size` before passing
- `tf_helper.py:1259`: Cast `size` before passing

---

## Test Plan

1. Run `pytest tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_full_train_split -v`
2. Run `pytest tests/ -k "TestReassemblePosition" -v`
3. Verify all 5 affected tests pass after fix
4. Verify no regressions in other reassembly tests

---

## Related Files

- `ptycho/tf_helper.py` - `_reassemble_position_batched()`, `mk_reassemble_position_batched_real()`
- `ptycho/custom_layers.py` - `ReassemblePatchesLayer`
- `ptycho/params.py` - `get_padded_size()`
- `tests/study/test_dose_overlap_comparison.py` - Failing test
- `tests/test_tf_helper.py` - `TestReassemblePosition` class

---

## References

- TensorFlow documentation on shape specifications
- `docs/DATA_NORMALIZATION_GUIDE.md` - Related data handling conventions
- NumPy 2.x type behavior changes (numpy scalars vs Python ints)
