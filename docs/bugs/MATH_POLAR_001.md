# Bug Report: [MATH-POLAR-001] CombineComplexLayer Incorrectly Applies Cartesian Logic to Polar Inputs

**Status:** FIXED (2026-01-26)
**Severity:** Critical - Fundamental mathematical error affecting reconstruction physics
**Discovered:** 2026-01-26
**Affects:** `ptycho/custom_layers.py`, `ptycho/model.py`, all PINN training and inference

---

## Summary

The `CombineComplexLayer` class is intended to merge the network's two output heads into a single complex field. The upstream model produces **Amplitude** (via Sigmoid/Swish) and **Phase** (via Tanh scaled by π).

However, `CombineComplexLayer` merges these using `tf.complex(amp, phase)`, which treats them as **Real** and **Imaginary** components (Z = A + iφ). This is mathematically incorrect for ptychography, which requires a Polar-to-Cartesian conversion (Z = A · e^(iφ)).

## Impact

### 1. Phase Averaging Failure in Stitching

The downstream `ReassemblePatchesLayer` and `reassemble_patches()` function average overlapping pixels by:
1. Extracting real and imaginary parts: `real = tf.math.real(channels)`, `imag = tf.math.imag(channels)`
2. Averaging each separately
3. Recombining: `tf.dtypes.complex(assembled_real, assembled_imag)`

With the current bug:
- `real(obj) = amplitude` (correct)
- `imag(obj) = phase` (WRONG - phase should not be averaged linearly)

**Example:** Two overlapping pixels with phases +π and -π (same physical phase due to 2π periodicity):
- **Buggy:** Average of imag parts = (π + (-π))/2 = 0 → WRONG phase
- **Correct:** Vector average of e^(iπ) and e^(-iπ) = average of (-1, -1) = -1 → phase = π ✓

### 2. Potential Artifacts at Patch Boundaries

Phase wrapping at ±π causes destructive interference when patches are stitched, potentially creating "null" pixels or discontinuities at boundaries.

### 3. Trained Models Have Learned Compensating Behavior

**Important:** Existing trained models have learned to produce outputs that minimize the physics loss despite this bug. The "amplitude" head does NOT output true physical amplitude—it outputs whatever value works when combined as (real, imag). Models would need **retraining** after fixing this bug.

## Location

**Primary File:** `ptycho/custom_layers.py` (lines 14-52)
**Related:** `ptycho/model.py` (lines 529-533, 759-762)
**Downstream:** `ptycho/tf_helper.py` `reassemble_patches()` (lines 1088-1112)

## Code Analysis

### Current Implementation (BUGGY)

```python
# ptycho/custom_layers.py
class CombineComplexLayer(layers.Layer):
    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        real_part, imag_part = inputs  # Actually (amplitude, phase)!
        # ...
        return tf.complex(real_part, imag_part)  # WRONG: treats as Re+Im
```

### Model Context

```python
# ptycho/model.py
def create_autoencoder(input_tensor, n_filters_scale, gridsize, big):
    decoded_amp = create_decoder_amp(...)    # Sigmoid output → Amplitude
    decoded_phase = create_decoder_phase(...) # Tanh*π output → Phase [-π, π]
    return decoded_amp, decoded_phase

# Later:
decoded1, decoded2 = create_autoencoder(...)  # (amp, phase)
obj = CombineComplexLayer()([decoded1, decoded2])  # BUG: amp+i*phase
```

### Downstream Impact

```python
# ptycho/tf_helper.py
def reassemble_patches(channels, ...):
    real = tf.math.real(channels)  # With bug: = amplitude
    imag = tf.math.imag(channels)  # With bug: = phase (averaged LINEARLY!)
    assembled_real = fn_reassemble_real(real, ...)  # OK
    assembled_imag = fn_reassemble_real(imag, ...)  # BAD: linear phase averaging
    return tf.dtypes.complex(assembled_real, assembled_imag)
```

## Proposed Fix

Update `CombineComplexLayer.call` to apply Euler's formula:

```python
def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
    """Combine Amplitude and Phase into Complex tensor using Euler's formula."""
    amp, phase = inputs

    # Cast to float32 for computation
    amp = tf.cast(amp, tf.float32)
    phase = tf.cast(phase, tf.float32)

    # Apply Euler's formula: Z = A * exp(i * φ) = A * (cos(φ) + i*sin(φ))
    real_part = amp * tf.cos(phase)
    imag_part = amp * tf.sin(phase)

    return tf.complex(real_part, imag_part)
```

## Validation Test

```python
import tensorflow as tf
import numpy as np

def test_phase_averaging():
    """Verify correct behavior at phase wrap boundary."""
    # Two patches with phase at ±π (same physical phase)
    amp = tf.constant([[1.0], [1.0]])
    phase = tf.constant([[np.pi], [-np.pi]])

    # BUGGY: amp + i*phase
    buggy = tf.complex(amp, phase)
    buggy_avg = tf.reduce_mean(buggy)
    # Result: 1 + 0j (wrong - phase averaged to 0)

    # CORRECT: amp * exp(i*phase)
    correct = amp * tf.exp(tf.complex(0.0, phase))
    correct_avg = tf.reduce_mean(correct)
    # Result: -1 + 0j (correct - both were at phase π)

    assert tf.abs(tf.math.angle(correct_avg) - np.pi) < 0.01, "Phase should be π"
```

## Migration Considerations

1. **Breaking Change:** All existing trained models assume the buggy behavior
2. **Retraining Required:** Models must be retrained after fix
3. **Version Flag:** Consider adding a `polar_combine=True/False` flag for backward compatibility
4. **Gradual Migration:** Could keep both modes during transition period

## Files Involved

- `ptycho/custom_layers.py` - CombineComplexLayer implementation
- `ptycho/model.py` - Model construction using CombineComplexLayer
- `ptycho/tf_helper.py` - reassemble_patches affected by this representation
- All training scripts - Would need model retraining

## References

- Euler's formula: https://en.wikipedia.org/wiki/Euler%27s_formula
- Complex number representation in ptychography literature
- `docs/specs/spec-ptycho-core.md` - Physics specification (should document correct form)
