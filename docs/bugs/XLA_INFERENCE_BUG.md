# XLA Inference Bug: PINN Model Fails with Dynamic Batch Sizes

**Status:** FIXED
**Severity:** High - Blocked PINN inference in grid_lines_workflow
**Discovered:** 2026-01-26
**Fixed:** 2026-01-26
**Affects:** `ptycho/workflows/grid_lines_workflow.py`, any PINN inference with variable batch sizes

---

## Summary

PINN models trained with XLA compilation fail during inference when the batch size differs from training or varies between batches. The error occurs in the Translation layer's FFT and tile operations which require compile-time constant shapes under XLA.

## Fix Applied

The fix involved two changes:

### 1. Force Translation Layer to Use XLA-Safe Path (`ptycho/tf_helper.py`)

The Translation layer now always uses the `translate_xla` function which employs `tf.gather` with modular indexing instead of `tf.repeat`/`tf.tile`:

```python
@tf.keras.utils.register_keras_serializable(package='ptycho')
class Translation(tf.keras.layers.Layer):
    def __init__(self, jitter_stddev: float = 0.0, use_xla: bool = False, **kwargs) -> None:
        super(Translation, self).__init__(**kwargs)
        self.jitter_stddev = jitter_stddev
        # CRITICAL FIX: Always use XLA-compatible path
        self.use_xla = True  # Force XLA path regardless of constructor argument
```

### 2. Disable object.big for grid_lines_workflow (`ptycho/workflows/grid_lines_workflow.py`)

Set `object_big=False` in the ModelConfig to ensure consistent output format between PINN and baseline models, enabling proper stitching:

```python
model=ModelConfig(N=cfg.N, gridsize=cfg.gridsize, object_big=False),
```

---

## Symptoms

```
tensorflow.python.framework.errors_impl.UnimplementedError: Graph execution error:
Asked to propagate a dynamic dimension from hlo set-dimension-size.6716@{}@0
to hlo %fft.6717 = c64[<=32,64,64]{2,1,0} fft(...)
```

Or:

```
Input 1 to node `...translation_19_1...` with op Tile must be a compile-time constant.
XLA compilation requires that operator arguments that represent shapes or dimensions
be evaluated to concrete values at compile time.
```

## Root Cause Analysis

### 1. XLA Compilation During Training

When `USE_XLA_COMPILE=1` (default), the model is compiled with XLA which traces and optimizes the computation graph. The training batch size becomes "baked in" to certain operations.

### 2. Affected Operations

The following operations in `ptycho/tf_helper.py` don't support dynamic shapes under XLA:

**a) Translation Layer (`translate_core`, line ~803)**
```python
def broadcast_translations(translations, target_shape):
    return tf.tile(translations, [...])  # Tile requires static shape
```

**b) FFT Operations (throughout `tf_helper.py`)**
```python
tf.signal.fft2d(...)  # FFT requires static spatial dimensions
```

**c) Reassemble Patches (`_reassemble_position_batched`, line ~1019)**
```python
tf.while_loop(...)  # Loop bounds may be dynamic
```

### 3. Why Training Works but Inference Fails

- **Training:** Fixed batch size (e.g., 16) → XLA compiles for batch=16
- **Inference:** Variable batch sizes (last batch may be smaller) → XLA fails

## Reproduction

```python
# Train PINN with default settings (XLA enabled)
from ptycho import train_pinn
model, history = train_pinn.train(train_data)

# Inference fails with non-training batch size
X_test = np.random.randn(100, 64, 64, 4)  # 100 != training batch size
coords = np.random.randn(100, 1, 2, 4)
model.predict([X_test, coords])  # FAILS with XLA error
```

## Previous Workaround (No Longer Needed)

The error handling in `grid_lines_workflow.py` that catches XLA errors and returns `None` is now a fallback safety net. With the fix applied, PINN inference completes successfully.

## Potential Fixes (Historical Reference)

### Option 1: Disable XLA for Inference Only

Rebuild model without XLA after training:
```python
# After training, create fresh model without XLA
os.environ['USE_XLA_COMPILE'] = '0'
fresh_model = create_model_with_gridsize(gridsize, N)
fresh_model.set_weights(trained_model.get_weights())
# Use fresh_model for inference
```

**Pros:** Preserves XLA speedup during training
**Cons:** Requires model recreation, may have weight compatibility issues

### Option 2: Fixed Batch Size with Padding

Pad inference batches to training batch size:
```python
def predict_with_padding(model, X, coords, batch_size=16):
    n = X.shape[0]
    pad_size = (batch_size - n % batch_size) % batch_size
    if pad_size > 0:
        X = np.concatenate([X, np.zeros((pad_size,) + X.shape[1:])])
        coords = np.concatenate([coords, np.zeros((pad_size,) + coords.shape[1:])])
    pred = model.predict([X, coords], batch_size=batch_size)
    return pred[:n]  # Remove padding
```

**Pros:** Simple, no model changes
**Cons:** Wastes computation on padding

### Option 3: Disable XLA Globally

Set `USE_XLA_COMPILE=0` before any TensorFlow imports:
```python
import os
os.environ['USE_XLA_COMPILE'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
```

**Pros:** Simplest fix
**Cons:** Loses XLA training speedup (~2-3x slower training)

### Option 4: Make Translation Layer XLA-Safe

Modify `tf_helper.py` to use static shapes:
```python
def translate_core(...):
    # Use tf.ensure_shape() to guarantee static shapes
    imgs = tf.ensure_shape(imgs, [BATCH_SIZE, H, W, C])
    ...
```

**Pros:** Fixes root cause
**Cons:** Requires significant refactoring, may break other code paths

## Files Involved

- `ptycho/tf_helper.py` - Translation layer, FFT operations
- `ptycho/model.py` - Model compilation with XLA
- `ptycho/custom_layers.py` - Custom Keras layers using tf_helper
- `ptycho/workflows/grid_lines_workflow.py` - Affected workflow

## Testing Plan

Verify the fix with:
```bash
python scripts/studies/grid_lines_workflow.py \
    --N 64 --gridsize 2 \
    --output-dir /tmp/xla_fix_test \
    --nimgs-train 2 --nimgs-test 2 --nepochs 2
```

Expected: Both PINN and Baseline metrics in `metrics.json`, no zeros in comparison PNG.

## Verification Results (2026-01-26)

Fix verified successfully:
- PINN inference completed without XLA errors
- PINN metrics: SSIM=0.74 (amplitude), 0.86 (phase)
- Baseline metrics: SSIM=0.17 (amplitude)
- Comparison visualization generated successfully

## References

- TensorFlow XLA documentation: https://www.tensorflow.org/xla
- Related TF issue: Dynamic shapes in XLA (tensorflow/tensorflow#47725)
- `docs/plans/2026-01-26-grid-resolution-study.md` - Documents discovery of this bug
