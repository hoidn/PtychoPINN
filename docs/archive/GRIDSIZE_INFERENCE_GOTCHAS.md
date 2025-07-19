# GridSize Inference Gotchas and Solutions

This document captures critical issues and their solutions discovered during the implementation of gridsize=2 inference support in PtychoPINN.

## Summary

Enabling gridsize>1 inference revealed several interconnected initialization order and configuration management issues that prevented models trained with gridsize=2 from loading correctly during inference. This document provides a detailed breakdown of each issue and its solution.

## Issue 1: Configuration Loading Order Bug

### Problem
The inference script's `setup_inference_configuration` function used training-focused configuration logic that ignored YAML parameters, causing gridsize to default to 1 even when YAML specified gridsize=2.

### Root Cause
```python
# BUGGY: Used training config loader that prioritized CLI args over YAML
config = setup_configuration(args, yaml_path)  # Ignored YAML gridsize
```

### Solution
```python
# FIXED: Proper YAML-first configuration loading
model_defaults = {f.name: f.default for f in fields(ModelConfig)}
if yaml_path:
    yaml_data = load_yaml_config(Path(yaml_path))
    model_defaults.update(yaml_data.get('model', {}))
final_model_config = ModelConfig(**model_defaults)
```

**Files Changed:** `scripts/inference/inference.py:setup_inference_configuration()`

## Issue 2: Multi-Channel Data Loading Bug

### Problem
The `ptycho/loader.py` had duplicate `load()` functions with incorrect logic that flattened multi-channel data into single-channel tensors, causing shape mismatches for gridsize>1 models.

### Root Cause
```python
# BUGGY: Incorrect handling of multi-channel arrays
Y = tf.ones_like(X)  # Created wrong channel dimensions
# Missing proper channel validation and splitting logic
```

### Solution
```python
# FIXED: Preserve channel dimensions for both X and Y
X = tf.convert_to_tensor(X_full_split, dtype=tf.float32)
if dset['Y'] is None:
    Y = tf.ones_like(X, dtype=tf.complex64)  # Same channel shape as X
else:
    Y = tf.convert_to_tensor(Y_split, dtype=tf.complex64)

# Validate channel consistency
if X.shape[-1] != Y.shape[-1]:
    raise ValueError(f"Channel mismatch between X ({X.shape[-1]}) and Y ({Y.shape[-1]})")
```

**Files Changed:** `ptycho/loader.py:load()`

## Issue 3: Model Loading Parameter Order Bug

### Problem
The ModelManager loaded TensorFlow models before updating global parameters, causing lambda layers to see default gridsize=1 during model validation.

### Root Cause
```python
# BUGGY: Model loaded before params updated
model = tf.keras.models.load_model(model_dir, custom_objects=custom_objects)
# params.cfg.update(loaded_params)  # Too late!
```

### Solution
```python
# FIXED: Parameters loaded and applied before model loading
with open(params_path, 'rb') as f:
    loaded_params = dill.load(f)
params.cfg.update(loaded_params)  # Set BEFORE model loading
print(f"DEBUG: Global params updated. gridsize is now: {params.get('gridsize')}")

# Now load model with correct parameters available
model = tf.keras.models.load_model(model_dir, custom_objects=custom_objects)
```

**Files Changed:** `ptycho/model_manager.py:load_model()`

## Issue 4: Initialization Order Bug (The Core Issue)

### Problem
Python modules imported `ptycho.model` at the top level, causing the model graph to be constructed during import time with default gridsize=1, before configuration could be updated.

### Root Cause
```python
# BUGGY: Top-level import causes premature model construction
from ptycho import train_pinn  # This imports model at module level!
# Model graph built with gridsize=1 before config is set
```

### Solution
```python
# FIXED: Delay model imports until after configuration
# Remove top-level imports that trigger model construction
# from ptycho import train_pinn  # REMOVED

# Move model imports inside functions
def reconstruct_image(test_data, diffraction_to_obj=None):
    from ptycho import model  # Import delayed until function call
    # Now model is built with correct gridsize
```

**Files Changed:** 
- `scripts/inference/inference.py` (removed `train_pinn` import)
- `ptycho/nbutils.py:reconstruct_image()` (delayed model import)

## Issue 5: TensorFlow Model Validation Tensor Shape Mismatch

### Problem  
Even after all fixes, TensorFlow's model loading process creates validation tensors with shape `(1,64,64,1)` for models that expect `(1,64,64,4)`, causing reshape failures.

### Root Cause
**Fundamental Architectural Issue**: The model has an implicit dependency on global state (`params.cfg['gridsize']`) that is not preserved during model serialization/deserialization. Lambda layers depend on global state that may not be set correctly when TensorFlow reconstructs the model.

### Solution Required
🔧 **ARCHITECTURAL FIX NEEDED** - Requires eliminating global state dependency by making gridsize an explicit model parameter. See <doc-ref type="architecture">docs/GRIDSIZE_ARCHITECTURAL_FIX.md</doc-ref> for the complete solution design.

### Current Status
📋 **AWAITING PERMISSION** - Solution requires modifying core protected files (`ptycho/model.py`, `ptycho/tf_helper.py`) which requires explicit permission per project directives.

**Diagnosis Tools Added:**
```python
# Debug logging in tf_helper.py
def _flat_to_channel(img: tf.Tensor, N: Optional[int] = None) -> tf.Tensor:
    gridsize = params()['gridsize'] 
    print(f"DEBUG: gridsize={gridsize}, img.shape={img.shape}, expected={(-1, gridsize**2, N, N)}")
```

## Key Lessons Learned

### 1. Configuration Management Anti-Patterns
- **Never use training config loaders for inference** - They have different parameter priorities
- **Always validate YAML parameter loading** with debug output
- **Ensure configuration flows from source → inference config → legacy params**

### 2. Initialization Order Anti-Patterns  
- **Never import model-constructing modules at top level** in inference scripts
- **Always delay model imports** until after configuration is finalized
- **Be aware of transitive imports** that trigger model construction (e.g., `train_pinn`)

### 3. Multi-Channel Data Handling Anti-Patterns
- **Never assume single-channel tensor shapes** in data loading logic
- **Always preserve channel dimensions** when converting between data formats
- **Validate channel consistency** between related tensors (X and Y)

### 4. Model Loading Anti-Patterns
- **Never load models before setting configuration** parameters
- **Always update global state before TensorFlow operations** that depend on it
- **Add debug logging** to verify parameter loading sequence

## Debugging Strategies

### Configuration Issues
```bash
# Check YAML loading
grep "Loading configuration from YAML" output.log
grep "Loaded gridsize=" output.log

# Verify final config  
grep "Final inference config" output.log
```

### Initialization Order Issues
```bash
# Check model import timing
grep "DEBUG: Global params updated" output.log
grep "Loading sub-model" output.log

# Verify parameter flow
grep "gridsize" output.log | head -10
```

### Data Loading Issues
```bash
# Check tensor shapes
grep "PtychoDataContainer" output.log
grep "Channel mismatch" output.log

# Verify multi-channel handling
grep "loader: setting dummy Y ground truth" output.log
```

## Future Considerations

### Architecture Improvements
1. **Eliminate global state dependency** - Pass configuration explicitly to all functions
2. **Implement proper dependency injection** for model construction  
3. **Create configuration validation layer** before any model operations

### Testing Improvements  
1. **Add integration tests** for gridsize>1 workflows
2. **Mock TensorFlow model loading** to test configuration flow
3. **Add shape validation tests** for multi-channel data pipelines

### Documentation Improvements
1. **Document initialization order requirements** in developer guide
2. **Create configuration troubleshooting guide** with common error patterns
3. **Add model loading best practices** to prevent similar issues

## Related Issues

- Configuration loading: `scripts/inference/inference.py:84-122`
- Multi-channel data: `ptycho/loader.py:150-217` 
- Model loading order: `ptycho/model_manager.py:71-94`
- Initialization timing: Import statements in inference workflow
- TensorFlow validation: Ongoing investigation needed

---

**Author**: Generated during gridsize=2 inference implementation  
**Date**: 2025-01-17  
**Status**: All issues resolved except TensorFlow model validation mismatch