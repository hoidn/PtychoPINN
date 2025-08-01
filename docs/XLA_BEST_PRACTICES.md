# XLA Best Practices for PtychoPINN

## Overview

XLA (Accelerated Linear Algebra) compilation can provide significant performance improvements (40%+ in our tests, up to 100x+ for inference). However, it requires careful implementation to avoid compilation failures and ensure correct results.

## Performance Benefits

In our implementation:
- Training step time: 24ms → 14-15ms (40% improvement)
- Inference can be 100x+ faster with proper batching
- Memory usage is more efficient due to kernel fusion

## Key Guidelines

### 1. Dtype Consistency is Critical

XLA is extremely strict about dtype matching. Operations that work in eager mode may fail during compilation.

```python
# BAD: This works in eager mode but fails in XLA
@tf.function(jit_compile=True)
def bad_example(images_f64, grid_f32):
    return images_f64 * grid_f32  # TypeError in XLA!

# GOOD: Explicit dtype management
@tf.function(jit_compile=True)
def good_example(images):
    dtype = images.dtype
    grid = tf.meshgrid(
        tf.range(height, dtype=dtype),
        tf.range(width, dtype=dtype)
    )
    return images * grid[0]
```

### 2. Avoid Dynamic Shapes

XLA performs best with static shapes. Use `tf.ensure_shape()` where possible:

```python
# Help XLA understand shapes
def process_batch(images, positions):
    # Explicitly declare expected shapes
    images = tf.ensure_shape(images, [None, 64, 64, 1])
    positions = tf.ensure_shape(positions, [None, 2])
    
    # Now XLA can optimize better
    return transform(images, positions)
```

### 3. Pure TensorFlow Operations

Replace external library calls with pure TF ops:

```python
# BAD: Using TensorFlow Addons
import tensorflow_addons as tfa
transformed = tfa.image.transform(image, matrix)

# GOOD: Pure TensorFlow implementation
from ptycho.projective_warp_xla import projective_warp_xla
transformed = projective_warp_xla(image, matrix)
```

### 4. Complex Number Handling

XLA doesn't directly support complex operations in all cases. Split and recombine:

```python
@tf.function(jit_compile=True)
def process_complex(complex_tensor):
    # Split complex into real and imaginary
    real = tf.math.real(complex_tensor)
    imag = tf.math.imag(complex_tensor)
    
    # Process separately
    real_out = some_operation(real)
    imag_out = some_operation(imag)
    
    # Recombine
    return tf.complex(real_out, imag_out)
```

## Implementation Patterns

### Pattern 1: Conditional JIT Compilation

Allow users to control JIT compilation:

```python
def translate_xla(images, translations, use_jit=True):
    if use_jit:
        return _translate_jit(images, translations)
    else:
        return _translate_eager(images, translations)

@tf.function(jit_compile=True)
def _translate_jit(images, translations):
    return projective_warp_xla(images, translations)

def _translate_eager(images, translations):
    return projective_warp_xla(images, translations)
```

### Pattern 2: Dtype-Aware Operations

Create operations that handle multiple dtypes correctly:

```python
def dtype_aware_operation(tensor):
    # Determine compute precision
    if tensor.dtype == tf.float64:
        compute_dtype = tf.float64
    elif tensor.dtype == tf.complex128:
        compute_dtype = tf.float64
    else:
        compute_dtype = tf.float32
    
    # Create constants in correct dtype
    one = tf.constant(1.0, dtype=compute_dtype)
    zero = tf.constant(0.0, dtype=compute_dtype)
    
    # Perform operations
    return tensor * one + zero
```

### Pattern 3: Efficient Batching

XLA excels at batched operations:

```python
# BAD: Processing one at a time
results = []
for i in range(batch_size):
    results.append(process_single(data[i]))
output = tf.stack(results)

# GOOD: Vectorized operations
@tf.function(jit_compile=True)
def process_batch(data):
    # Process entire batch at once
    return vectorized_operation(data)
```

## Testing XLA Compilation

### Basic Compilation Test

```python
def test_xla_compilation():
    """Verify function compiles with XLA."""
    @tf.function(jit_compile=True)
    def test_fn(x):
        return your_operation(x)
    
    # Test with different dtypes
    test_inputs = [
        tf.constant([1.0], dtype=tf.float32),
        tf.constant([1.0], dtype=tf.float64),
        tf.complex(tf.constant([1.0]), tf.constant([0.0]))
    ]
    
    for input_tensor in test_inputs:
        try:
            result = test_fn(input_tensor)
            print(f"✓ XLA compilation successful for {input_tensor.dtype}")
        except Exception as e:
            print(f"✗ XLA compilation failed for {input_tensor.dtype}: {e}")
```

### Performance Comparison

```python
import time

def benchmark_xla():
    """Compare performance with and without XLA."""
    data = tf.random.normal([32, 64, 64, 1])
    
    # Without XLA
    @tf.function(jit_compile=False)
    def without_xla(x):
        return your_operation(x)
    
    # With XLA
    @tf.function(jit_compile=True)
    def with_xla(x):
        return your_operation(x)
    
    # Warmup
    without_xla(data)
    with_xla(data)
    
    # Benchmark
    n_iterations = 100
    
    start = time.time()
    for _ in range(n_iterations):
        without_xla(data)
    time_without = time.time() - start
    
    start = time.time()
    for _ in range(n_iterations):
        with_xla(data)
    time_with = time.time() - start
    
    print(f"Without XLA: {time_without:.3f}s")
    print(f"With XLA: {time_with:.3f}s")
    print(f"Speedup: {time_without/time_with:.2f}x")
```

## Common XLA Errors and Solutions

### Error 1: Dtype Mismatch
```
TypeError: Input 'y' of 'Mul' Op has type float64 that does not match type float32 of argument 'x'.
```
**Solution**: Ensure all operands have the same dtype (see DTYPE_HANDLING.md)

### Error 2: Dynamic Shape
```
InvalidArgumentError: XLA compilation requires fixed tensor shapes
```
**Solution**: Use `tf.ensure_shape()` or avoid operations that produce dynamic shapes

### Error 3: Unsupported Operation
```
InvalidArgumentError: Operation not supported by XLA
```
**Solution**: Replace with equivalent pure TensorFlow operations

## Debugging Tips

1. **Disable JIT temporarily**: Set `jit_compile=False` to identify if issue is XLA-specific
2. **Check dtypes**: Print tensor dtypes at each step
3. **Use tf.print**: Works inside XLA-compiled functions for debugging
4. **Enable XLA logs**: Set `TF_XLA_FLAGS=--tf_xla_enable_xla_devices` for detailed logs

## When NOT to Use XLA

- Operations with highly dynamic shapes
- Code that requires Python control flow
- When debugging (eager mode gives better error messages)
- Small operations where compilation overhead exceeds benefits

## References

- [TensorFlow XLA Documentation](https://www.tensorflow.org/xla)
- See `projective_warp_xla.py` for a complete XLA-compatible implementation
- See `tests/test_projective_warp_xla.py` for XLA testing examples