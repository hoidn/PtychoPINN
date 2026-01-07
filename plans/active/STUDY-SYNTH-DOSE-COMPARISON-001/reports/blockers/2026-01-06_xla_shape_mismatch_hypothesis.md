# Blocker Analysis: XLA Shape Mismatch in Dose Response Study

**Date:** 2026-01-06
**Status:** Active Blocker - REVISED HYPOTHESIS
**Initiative:** STUDY-SYNTH-DOSE-COMPARISON-001
**Symptom:** `Input to reshape is a tensor with 778752 values, but the requested shape has 24336`

---

## Executive Summary

The dose response study fails during model forward pass with an XLA reshape error. Root cause analysis reveals this is **NOT** a regrouping failure or model singleton issue. The regrouping architecture works correctly.

**REVISED HYPOTHESIS (2026-01-06):** The bug is likely **UPSTREAM** of `projective_warp_xla.py`, not in the XLA code itself. The XLA code has been tested and works correctly. The issue is in the **data preparation layer** before the XLA function is called - specifically in how `_channel_to_flat` and `flatten_offsets` prepare data for the Translation layer.

---

## Verified Working Components

| Component | Status | Evidence |
|-----------|--------|----------|
| Late-binding regrouping | ✅ Works | `RawData(200 patterns) → Container(50, 64, 64, 4)` |
| Model factory | ✅ Works | `create_compiled_model()` produces 4-channel model |
| params.cfg synchronization | ✅ Works | `gridsize=2` maintained through container creation |
| XLA workaround placement | ✅ Correct | Set at line 369, before `train_cdi_model()` |

---

## The Error

```
File "projective_warp_xla.py", line 145, in projective_warp_xla
Input to reshape is a tensor with 778752 values, but the requested shape has 24336
```

### Numerical Analysis

| Value | Factorization | Interpretation |
|-------|---------------|----------------|
| 778752 | 32 × 78 × 78 × 4 | (n_groups, padded_H, padded_W, channels) |
| 778752 | 128 × 78 × 78 | (n_groups × channels, padded_H, padded_W) flattened |
| 24336 | 156 × 156 | (2 × 78)² = reassembled output size |
| Ratio | 32 | 778752 / 24336 = 32 = n_groups when C=4 |

The XLA function receives `(32, 78, 78, 4)` but attempts to reshape into `(156, 156)`.

---

## Call Stack Analysis

```
train_pinn.train_eval()
  └─ model.train()
       └─ autoencoder.fit()
            └─ ReassemblePatchesLayer.call()
                 └─ reassemble_patches_position_batched_real()  [tf_helper.py:1320]
                      └─ _reassemble_position_batched()         [tf_helper.py:1003]
                           └─ original_approach()               [tf_helper.py:914]
                                └─ TranslationLayer.call()      [tf_helper.py:836]
                                     └─ translate()             [tf_helper.py:800]
                                          └─ translate_core()   [tf_helper.py:722]
                                               └─ translate_xla()        [projective_warp_xla.py:294]
                                                    └─ projective_warp_xla_jit()
                                                         └─ projective_warp_xla()   [line 145 - CRASH]
```

---

## Hypotheses

### H1: ORIGINAL - XLA Translation Layer Assumes C=1 (DEMOTED)

**Confidence: LOW (20%) - DEMOTED based on user feedback**

~~The `projective_warp_xla` function was written with single-channel (`gridsize=1`) assumptions.~~

**USER FEEDBACK:** `projective_warp_xla.py` has been tested and works correctly. The bug is upstream.

---

### H1-REVISED: Input Data Shape Mismatch Between Images and Offsets (PRIME SUSPECT)

**Confidence: HIGH (80%)**

The issue is that `imgs` and `offsets_xy` have **inconsistent channel dimensions** when passed to `_reassemble_position_batched`. The data preparation in lines 901-902 expects:

```python
offsets_flat = flatten_offsets(offsets_xy)  # Expects (B, 1, 2, C) → (B*C, 2)
imgs_flat = _channel_to_flat(imgs)          # Expects (B, N, N, C) → (B*C, N, N, 1)
```

If `offsets_xy` has shape `(B, 2)` instead of `(B, 1, 2, C)`, then:
- `flatten_offsets` will produce `(B, 2)` (NOT flattened)
- `imgs_flat` will be `(B*C, N, N, 1)` (correctly flattened)
- Translation receives **mismatched batch dimensions**: images `(128, 78, 78, 1)` vs offsets `(32, 2)`

The XLA code then fails because `B = tf.shape(translations)[0] = 32` but images have batch 128.

**Numerical Analysis:**
- 778752 = 128 × 78 × 78 × 1 (flattened images batch)
- 24336 = 156 × 156 (reassembly canvas size - irrelevant red herring)
- The 32x ratio = batch size before flattening, suggesting offsets weren't flattened

**Evidence:**
- The error 778752 vs 24336 has a ratio of 32, which equals the batch size B
- The XLA code uses `B = tf.shape(translations)[0]` for grid creation
- If translations shape is `(32, 2)` but images shape is `(128, ...)`, the grid will be `[32, H, W]` but images are `[128, ...]`

**Investigation Required:** Check the shape of `input_positions` (offsets_xy) passed to `ReassemblePatchesLayer`.

---

### H2-REVISED: TensorFlow Graph Tracing Captures Wrong B Dimension

**Confidence: MEDIUM (50%)**

During TensorFlow model tracing, symbolic tensor shapes are used. The `_gather_bhw` function at `projective_warp_xla.py:235` performs:

```python
return tf.reshape(gathered, [B, H, W, C])
```

Where `B = tf.shape(images)[0]`. During tracing, TensorFlow might infer an incorrect symbolic batch size if:
- The model is traced with a different batch size than runtime
- There's shape inference ambiguity between flattened (B*C) and original (B) batch sizes

**Supporting Evidence:**
- 24336 = 156 × 156 = (padded_size × gridsize)² - this is the canvas size
- If B=1 during tracing (singleton batch for shape inference), the reshape would target 24336
- But actual data has 778752 elements (32 × 78 × 78 × 4 or 128 × 78 × 78)

**Test:** Add explicit `tf.ensure_shape` calls to lock down expected shapes during tracing.

---

### H3-REVISED: Code Path Static Analysis

**Analysis Status:** VERIFIED - Data Formats Appear Correct

Traced the data flow from raw data through to model:

1. `RawData.generate_grouped_data()`: Produces `coords_start_relative` shape `(nsamples, 1, 2, gridsize²)` ✓
2. `loader.load()`: Converts to tensor, preserves shape ✓
3. `PtychoDataContainer`: Stores as `coords_nominal` ✓
4. `model.py`: Defines `input_positions = Input(shape=(1, 2, gridsize**2))` ✓
5. `prepare_inputs()`: Returns `[train_data.X, train_data.coords]` ✓

The documented shapes are consistent. The bug must be either:
- A runtime shape transformation not captured in static analysis
- A TensorFlow tracing/graph compilation issue
- An edge case in the XLA code path for specific shape combinations

---

### H2: XLA Workaround Not Applied Due to Graph Caching

**Confidence: MEDIUM (60%)**

TensorFlow may cache the computation graph (including XLA decisions) during:
- Model import (module-level code in `model.py`)
- First call to `create_ptycho_data_container()`
- Keras model compilation

If any of these happen before `p.cfg['use_xla_translate'] = False` is set, the XLA path gets baked into the cached graph.

**Current Code Flow:**
```python
# Line 362: update_legacy_dict(p.cfg, config)
# --> use_xla_translate is True (default from params.py:74)

# Line 369: p.cfg['use_xla_translate'] = False
# --> Now False, but is it too late?

# Line 376: train_cdi_model(train_data, test_data, config)
# --> Model graph may already be cached with XLA=True
```

**Test:** Move `p.cfg['use_xla_translate'] = False` to the very start of `main()`, before any TensorFlow imports.

---

### H3: should_use_xla() Called Before Workaround

**Confidence: MEDIUM (50%)**

The `should_use_xla()` function at `tf_helper.py:160` is called when:
1. A layer's `call()` method executes
2. During Keras model tracing (first call with symbolic tensors)

If model tracing happens during `create_ptycho_data_container()` or `probe.set_probe_guess()`, the XLA decision is captured before the workaround.

**Investigation Required:** Add logging to `should_use_xla()` to trace when it's called.

---

### H4: Environment Variable Override

**Confidence: LOW (20%)**

The `should_use_xla()` function checks `os.environ.get('USE_XLA_TRANSLATE')` first:

```python
use_xla_env = os.environ.get('USE_XLA_TRANSLATE', '').lower() in ('1', 'true', 'yes')
```

If this environment variable is set to `'1'` or `'true'`, it would override the config setting.

**Test:** `echo $USE_XLA_TRANSLATE` in the shell, or add `os.environ['USE_XLA_TRANSLATE'] = '0'` at script start.

---

## Recommended Fixes (Priority Order) - REVISED

### Fix 1: Verify Input Position Format in ReassemblePatchesLayer (DIAGNOSTIC)

Before fixing, verify the hypothesis by adding shape logging:

```python
# In ReassemblePatchesLayer.call() at custom_layers.py:147
patches, positions = inputs
tf.print("[DEBUG] patches.shape:", tf.shape(patches))    # Expected: (B, N, N, C)
tf.print("[DEBUG] positions.shape:", tf.shape(positions))  # Expected: (B, 1, 2, C)
```

If positions has shape `(B, 2)` instead of `(B, 1, 2, C)`, this confirms H1-REVISED.

### Fix 2: Ensure Consistent Position Format (ROOT CAUSE FIX)

If H1-REVISED is confirmed, fix the position tensor format where it's created:

```python
# Wherever coords_nominal is created, ensure it has channel format
# Wrong: coords_nominal shape (B, 2)
# Right: coords_nominal shape (B, 1, 2, C) where C = gridsize²

# Example conversion if needed:
def ensure_channel_format(coords, gridsize):
    """Convert (B, 2) → (B, 1, 2, C) by broadcasting."""
    B = tf.shape(coords)[0]
    C = gridsize ** 2
    # Reshape: (B, 2) → (B, 1, 2) → tile → (B, 1, 2, C)
    coords = tf.expand_dims(coords, axis=1)  # (B, 1, 2)
    coords = tf.expand_dims(coords, axis=-1)  # (B, 1, 2, 1)
    coords = tf.tile(coords, [1, 1, 1, C])    # (B, 1, 2, C)
    return coords
```

### Fix 3: Add Shape Guard in flatten_offsets (DEFENSIVE)

Add explicit shape checking to catch mismatched input early:

```python
def flatten_offsets(channels: tf.Tensor) -> tf.Tensor:
    """Convert (B, 1, 2, C) → (B*C, 2)."""
    # Shape guard: ensure 4D input
    if len(channels.shape) != 4:
        raise ValueError(f"flatten_offsets expects 4D tensor (B, 1, 2, C), got shape {channels.shape}")
    return _channel_to_flat(channels)[:, 0, :, 0]
```

### Fix 4: Early XLA Disable (WORKAROUND - USE IF ROOT CAUSE FIX IS BLOCKED)

```python
# At the very top of dose_response_study.py, after imports
import os
os.environ['USE_XLA_TRANSLATE'] = '0'  # Before ANY TensorFlow operations

from ptycho import params as p
p.cfg['use_xla_translate'] = False
```

---

## Verification Plan - REVISED

### Phase 1: Confirm Root Cause (DIAGNOSTIC)

1. **Add shape logging** to `ReassemblePatchesLayer.call()` to capture positions shape
2. **Run dose study** with minimal epochs to trigger the error
3. **Analyze logs** to confirm positions shape is `(B, 2)` vs expected `(B, 1, 2, C)`

### Phase 2: Trace Position Creation

If Phase 1 confirms shape mismatch:

1. **Find position source**: Trace where `coords_nominal` is created for the model
2. **Identify format inconsistency**: Check if it uses flat format vs channel format
3. **Document finding**: Update this hypothesis with confirmed root cause

### Phase 3: Implement Fix

Based on findings:
- If positions format wrong → Fix 2 (ensure channel format)
- If format conversion missing → Add conversion in model/data pipeline
- If blocked → Use Fix 4 (XLA disable workaround)

### Phase 4: Verify Fix

1. Re-run dose study end-to-end
2. Confirm all 4 arms produce valid reconstructions
3. Generate 6-panel figure
4. Archive logs as evidence

---

## Related Findings

- `WORKAROUND-001`: XLA translate disable for shape mismatches (referenced in dose_response_study.py:368)
- `MODULE-SINGLETON-001`: Model singleton gridsize issue (FIXED in REFACTOR-MODEL-SINGLETON-001)
- `CONFIG-001`: update_legacy_dict timing requirements

---

## Appendix: Diagnostic Commands

```bash
# Check if XLA env var is set
echo $USE_XLA_TRANSLATE

# Run with XLA explicitly disabled
USE_XLA_TRANSLATE=0 python scripts/studies/dose_response_study.py --nepochs 2

# Add debug logging to should_use_xla()
# In tf_helper.py:160, add:
#   print(f"DEBUG should_use_xla: config={get('use_xla_translate')}, returning={result}")
```

### Shape Debugging Script

```python
# Quick diagnostic to check positions shape in model
import tensorflow as tf
from ptycho import params as p
from ptycho.config.config import TrainingConfig, update_legacy_dict

# Setup minimal config
config = TrainingConfig(gridsize=2, N=64)
update_legacy_dict(p.cfg, config)

# Create sample tensors matching expected model input
batch_size = 32
N = p.cfg['N']
C = p.cfg['gridsize'] ** 2

# Check what the model receives
patches = tf.zeros((batch_size, N, N, C))
positions_channel_format = tf.zeros((batch_size, 1, 2, C))  # CORRECT
positions_flat_format = tf.zeros((batch_size, 2))           # WRONG

print(f"patches.shape: {patches.shape}")
print(f"positions (channel): {positions_channel_format.shape}")
print(f"positions (flat): {positions_flat_format.shape}")

# Test flatten_offsets with both formats
from ptycho.tf_helper import flatten_offsets, _channel_to_flat

try:
    flat_correct = flatten_offsets(positions_channel_format)
    print(f"flatten_offsets(channel) → {flat_correct.shape}")  # Should be (128, 2)
except Exception as e:
    print(f"flatten_offsets(channel) ERROR: {e}")

try:
    flat_wrong = flatten_offsets(positions_flat_format)
    print(f"flatten_offsets(flat) → {flat_wrong.shape}")  # Will fail or give wrong shape
except Exception as e:
    print(f"flatten_offsets(flat) ERROR: {e}")
```
