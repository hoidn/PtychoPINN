# Complex Dtype Handling in PtychoPINN

## Overview

This document describes critical dtype handling requirements discovered during the TensorFlow 2.18 migration and XLA implementation. These issues can cause silent failures, incorrect results, or runtime errors if not handled properly.

## The Problem

When working with complex-valued tensors in TensorFlow, especially with XLA compilation, dtype mismatches can occur between:
- Complex tensors and their real/imaginary components
- Float32 and Float64 precision levels
- Grid computations and image data

### Example Bug
The following code would fail with a dtype mismatch error:
```python
# Bug: Grid computed in float32, but image is float64
images = tf.constant(data, dtype=tf.float64)
grid = tf.meshgrid(tf.range(H, dtype=tf.float32), tf.range(W, dtype=tf.float32))
# This multiplication fails: float32 * float64
result = grid * images
```

## Key Principles

### 1. Complex to Real Conversion

Always explicitly handle complex dtypes when converting to real:

```python
# Correct approach
if images.dtype in [tf.complex64, tf.complex128]:
    real_dtype = tf.float32 if images.dtype == tf.complex64 else tf.float64
    real_part = tf.cast(tf.math.real(images), real_dtype)
    imag_part = tf.cast(tf.math.imag(images), real_dtype)
    
    # Process real and imaginary separately
    real_result = process(real_part)
    imag_result = process(imag_part)
    
    # Recombine
    result = tf.complex(real_result, imag_result)
```

### 2. Compute Dtype Consistency

When performing mathematical operations, ensure all operands have consistent dtypes:

```python
# Determine compute precision based on input
img_dtype = images.dtype
compute_dtype = tf.float32
if img_dtype == tf.float64:
    compute_dtype = tf.float64
elif img_dtype == tf.complex128:
    compute_dtype = tf.float64

# Use compute_dtype for all intermediate computations
grid_x = tf.range(width, dtype=compute_dtype)
grid_y = tf.range(height, dtype=compute_dtype)
weights = tf.ones([batch_size], dtype=compute_dtype)
```

### 3. XLA Compilation Requirements

XLA is particularly strict about dtype consistency. Common issues and solutions:

```python
# Problem: Mixed precision in multiplication
wa = (1.0 - wx) * (1.0 - wy)  # wx, wy might be float32
result = wa * image  # image might be float64

# Solution: Explicit casting
wx = tf.cast(wx, images.dtype)
wy = tf.cast(wy, images.dtype)
wa = (1.0 - wx) * (1.0 - wy)
result = wa * image
```

## Common Pitfalls and Solutions

### Pitfall 1: Assuming tf.math.real() Returns Float32
- **Issue**: `tf.math.real()` preserves precision (complex64 → float32, complex128 → float64)
- **Solution**: Always explicitly cast to desired dtype

### Pitfall 2: Hardcoding Float32 in Grid Computations
- **Issue**: Creating coordinate grids with hardcoded float32 when images are float64
- **Solution**: Use a compute_dtype variable that matches the input precision

### Pitfall 3: Mixing Dtypes in Mathematical Operations
- **Issue**: TensorFlow's automatic dtype promotion doesn't work in XLA-compiled code
- **Solution**: Explicitly cast all operands to the same dtype before operations

## Testing for Dtype Issues

Add these test cases to catch dtype problems:

```python
def test_float64_support(self):
    """Test that operations work with float64 inputs."""
    image_f64 = tf.constant(data, dtype=tf.float64)
    result = your_operation(image_f64)
    self.assertEqual(result.dtype, tf.float64)

def test_complex128_support(self):
    """Test that operations work with complex128 inputs."""
    image_c128 = tf.complex(
        tf.constant(real_data, dtype=tf.float64),
        tf.constant(imag_data, dtype=tf.float64)
    )
    result = your_operation(image_c128)
    self.assertEqual(result.dtype, tf.complex128)

def test_xla_compilation_mixed_precision(self):
    """Test XLA compilation with different precisions."""
    @tf.function(jit_compile=True)
    def compiled_op(x):
        return your_operation(x)
    
    # Should not raise dtype errors
    result_f32 = compiled_op(tf.constant(data, dtype=tf.float32))
    result_f64 = compiled_op(tf.constant(data, dtype=tf.float64))
```

## Best Practices Checklist

Before committing code that handles numerical operations:

- [ ] Check all complex number operations explicitly handle dtypes
- [ ] Verify float32/float64 consistency throughout the computation
- [ ] Test with complex64, complex128, float32, and float64 inputs
- [ ] Ensure XLA compilation succeeds without dtype errors
- [ ] Add test cases for all supported dtype combinations

## References

- See `projective_warp_xla.py` for a complete implementation example
- See `tests/test_projective_warp_xla.py` for comprehensive dtype testing