# Architectural Fix for GridSize Dependency Issue

## The Fundamental Problem

The current model architecture has an **implicit dependency on global state** that causes failures during model loading:

1. **During Training**: Model built correctly with `params.cfg['gridsize'] = 2`
2. **During Saving**: Keras saves structure but lambda layer's dependency on global `params.cfg['gridsize']` is not saved
3. **During Loading**: `tf.keras.models.load_model()` sees default `gridsize=1`, builds incorrect validation graph, crashes

## Root Cause Analysis

The problematic code pattern in `ptycho/model.py`:
```python
# PROBLEMATIC: Lambda layer depends on global state
pred_diff = Lambda(lambda x: hh._flat_to_channel(x), name='pred_diff_channels')(pred_diff)
```

And in `ptycho/tf_helper.py`:
```python
def _flat_to_channel(img: tf.Tensor, N: Optional[int] = None) -> tf.Tensor:
    gridsize = params()['gridsize']  # IMPLICIT GLOBAL DEPENDENCY!
    # ...
```

**The Issue**: When TensorFlow loads the model, it reconstructs lambda layers, but the global state dependency is not preserved.

## The Correct Architectural Solution

### 1. Make GridSize an Explicit Parameter

**Update `ptycho/tf_helper.py`:**
```python
def _flat_to_channel(img: tf.Tensor, N: Optional[int] = None, gridsize: Optional[int] = None) -> tf.Tensor:
    if gridsize is None:
        gridsize = params()['gridsize']  # Fallback for backward compatibility
    if N is None:
        N = params()['N']
    img = tf.reshape(img, (-1, gridsize**2, N, N))
    img = tf.transpose(img, [0, 2, 3, 1], conjugate=False)
    return img
```

### 2. Create Model Factory Function

**Update `ptycho/model.py`:**
```python
def create_model_with_gridsize(gridsize: int, N: int, **kwargs):
    """
    Create model with explicit gridsize parameter to eliminate global state dependency.
    
    Args:
        gridsize: Grid size for neighbor patch processing
        N: Image size parameter
        **kwargs: Other model configuration parameters
    
    Returns:
        Tuple of (autoencoder, diffraction_to_obj) models
    """
    # ... existing model construction code ...
    
    # FIXED: Lambda layer uses local gridsize variable
    pred_diff = Lambda(
        lambda x: hh._flat_to_channel(x, N=N, gridsize=gridsize), 
        name='pred_diff_channels'
    )(pred_diff)
    
    # ... rest of model construction ...
    
    autoencoder = Model([input_img, input_positions], [trimmed_obj, pred_amp_scaled, pred_intensity_sampled])
    diffraction_to_obj = Model(inputs=[input_img, input_positions], outputs=[trimmed_obj])
    
    return autoencoder, diffraction_to_obj

# Backward compatibility: existing global construction
def _create_models_from_global_config():
    """Create models using global configuration (for backward compatibility)."""
    gridsize = cfg.get('gridsize')
    N = cfg.get('N')
    return create_model_with_gridsize(gridsize, N)

# Global model instances (maintained for backward compatibility)
autoencoder, diffraction_to_obj = _create_models_from_global_config()
```

### 3. Update ModelManager for Explicit Architecture

**Update `ptycho/model_manager.py`:**
```python
@staticmethod
def save_model_with_config(models_dict: Dict[str, tf.keras.Model], base_path: str, 
                          custom_objects: Dict[str, Any], model_config: Dict[str, Any]) -> None:
    """Save models with explicit configuration including gridsize."""
    
    # Save model configuration alongside weights
    config_path = f"{base_path}_config.json"
    with open(config_path, 'w') as f:
        json.dump(model_config, f)
    
    # Save models normally
    ModelManager.save_multiple_models(models_dict, base_path, custom_objects, model_config['intensity_scale'])

@staticmethod
def load_model_with_config(base_path: str) -> Tuple[Dict[str, tf.keras.Model], Dict[str, Any]]:
    """Load models by first reading configuration and rebuilding architecture."""
    
    # Load model configuration first
    config_path = f"{base_path}_config.json"
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    # Update global parameters
    params.cfg.update(model_config)
    
    # Rebuild model with correct architecture
    from ptycho.model import create_model_with_gridsize
    autoencoder, diffraction_to_obj = create_model_with_gridsize(
        gridsize=model_config['gridsize'],
        N=model_config['N']
    )
    
    # Load weights into correctly-structured models
    # ... weight loading logic ...
    
    return {'autoencoder': autoencoder, 'diffraction_to_obj': diffraction_to_obj}, model_config
```

### 4. Update Training and Inference Scripts

**Training Scripts:**
```python
from ptycho.model import create_model_with_gridsize
from ptycho import params

# After configuration is loaded
gridsize = params.get('gridsize')
N = params.get('N')
autoencoder, diffraction_to_obj = create_model_with_gridsize(gridsize, N)
```

**Inference Scripts:**
```python
# Use new ModelManager methods
models_dict, model_config = ModelManager.load_model_with_config(model_path)
diffraction_to_obj = models_dict['diffraction_to_obj']
```

## Benefits of This Solution

1. **Eliminates Global State Dependency**: Model architecture is self-contained
2. **Robust Model Loading**: TensorFlow can reconstruct models correctly
3. **Backward Compatibility**: Existing code continues to work
4. **Explicit Configuration**: Model requirements are clearly documented
5. **Testable Architecture**: Models can be created with different parameters for testing

## Implementation Requirements

### Core File Changes Required:
- `ptycho/tf_helper.py`: Update `_flat_to_channel` signature
- `ptycho/model.py`: Add model factory function
- `ptycho/model_manager.py`: Add configuration-aware save/load methods

### Script Updates Required:
- Training scripts: Use new factory function
- Inference scripts: Use new ModelManager methods

## Migration Strategy

1. **Phase 1**: Add new functions alongside existing ones (backward compatible)
2. **Phase 2**: Update scripts to use new functions
3. **Phase 3**: Deprecate old global-state-dependent functions
4. **Phase 4**: Remove deprecated functions

## Permission Required

This fix requires modifying core files protected by project directives:
- `ptycho/model.py` (core model architecture)
- `ptycho/tf_helper.py` (core helper functions)

**Request**: Explicit permission to implement this architectural fix to resolve the gridsize inference issue permanently.

## Alternative: Workaround Solutions

If core file modification is not permitted, potential workarounds include:
1. **Model Conversion Tool**: Convert saved models to remove lambda layer dependencies
2. **Custom Model Loading**: Bypass TensorFlow's standard loading process
3. **Configuration Injection**: Modify global state during model loading (fragile)

However, these workarounds do not address the fundamental architectural issue and may create maintenance burdens.

---

**Status**: Awaiting permission to implement core architectural changes  
**Priority**: High - Required for gridsize>1 inference support  
**Impact**: Resolves fundamental global state dependency issue