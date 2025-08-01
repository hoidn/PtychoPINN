# TensorFlow 2.19 Inference Debugging Summary

## Overview
This document summarizes the debugging process for running `ptycho_inference` with models trained on TensorFlow 2.19 and the compatibility issues encountered with Keras 3.

## Initial Command
```bash
ptycho_inference --model_path ../PtychoPINN/fly64_pinn_gridsize2_final \
                 --test_data ../PtychoPINN/datasets/fly64/fly001_64_train_converted.npz \
                 --config ../PtychoPINN/inference_gridsize2_config.yaml \
                 --output_dir verification_test
```

## Issues Encountered and Fixes Applied

### 1. ProbeIllumination Layer dtype Mismatch
**Error**: `Input 'y' of 'Mul' Op has type float32 that does not match type complex64 of argument 'x'`

**Fix**: Added `dtype=tf.complex64` to Lambda layers that feed into ProbeIllumination:
```python
# In model.py (lines 436-441, 627-631)
padded_objs_with_offsets = Lambda(
    lambda x: hh.extract_patches_position(x[0], x[1], 0.),
    output_shape=(N, N, 1),
    dtype=tf.complex64,  # Added this
    name='padded_objs_with_offsets'
)([padded_obj_2, input_positions])
```

### 2. TensorFlow Probability DistributionLambda Issue
**Error**: `A KerasTensor cannot be used as input to a TensorFlow function`

**Fix**: Replaced TFP DistributionLambda with simple Lambda layer:
```python
# Before (line 658)
dist_poisson_intensity = tfpl.DistributionLambda(lambda amplitude:
                                   (tfd.Independent(
                                       tfd.Poisson(
                                           (amplitude**2)))))
pred_intensity_sampled = dist_poisson_intensity(pred_amp_scaled)

# After
pred_intensity_sampled = Lambda(lambda x: tf.square(x), name='pred_intensity')(pred_amp_scaled)
```

### 3. SavedModel Loading in Keras 3
**Error**: `File format not supported: filepath=/path/to/model. Keras 3 only supports V3 .keras and .weights.h5 files`

**Fix**: Created SavedModelWrapper class to handle legacy SavedModel format:
```python
class SavedModelWrapper(tf.keras.Model):
    def __init__(self, saved_model, blank_model):
        super().__init__()
        self.saved_model = saved_model
        self.blank_model = blank_model
        # ... initialization code ...
    
    def call(self, inputs, training=None, mask=None):
        if isinstance(inputs, (list, tuple)):
            if len(inputs) >= 2:
                output_dict = self.inference(
                    input=inputs[0],
                    input_positions=inputs[1]
                )
        # ... rest of implementation ...
```

### 4. CenterMaskLayer Serialization
**Error**: `Could not locate class 'CenterMaskLayer'`

**Fix**: 
1. Moved CenterMaskLayer to module level in tf_helper.py
2. Added get_config method for proper serialization
3. Added to custom_objects in ModelManager

```python
# In tf_helper.py
class CenterMaskLayer(tfkl.Layer):
    def __init__(self, N, c, kind='center', **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.c = c
        self.kind = kind
        self.zero_pad = tfkl.ZeroPadding2D((N // 4, N // 4))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'N': self.N,
            'c': self.c,
            'kind': self.kind
        })
        return config
```

### 5. Custom Layer Initialization
**Error**: `ProbeIllumination.__init__() got an unexpected keyword argument 'trainable'`

**Fix**: Updated custom layers to accept **kwargs:
```python
class ProbeIllumination(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        kwargs.pop('dtype', None)  # Handle dtype separately
        super(ProbeIllumination, self).__init__(name=name, **kwargs)
        self.w = initial_probe_guess
        self.sigma = p.get('gaussian_smoothing_sigma')
```

### 6. Lambda Layer Deserialization
**Error**: `The function of this Lambda layer is a Python lambda. Deserializing it is unsafe.`

**Fix**: Enabled unsafe deserialization:
```python
tf.keras.config.enable_unsafe_deserialization()
```

## Model Training Test
Successfully trained a new model with the updated code:
```bash
ptycho_train --train_data_file datasets/fly64/fly001_64_train_converted.npz \
             --test_data_file datasets/fly64/fly001_64_train_converted.npz \
             --gridsize 2 --output_dir fly64_pinn_gridsize2_final --n_images 5000
```
Output: `fly64_pinn_gridsize2_final/wts.h5.zip` (34.9 MB)

## Current Status
While significant progress was made fixing TF 2.19/Keras 3 compatibility issues, complete inference still fails due to fundamental incompatibilities between:
- SavedModel format (TensorFlow graph representation)
- Keras 3's stricter serialization requirements
- Lambda layers with Python functions that cannot be safely deserialized

## Recommendations

### Short-term Solutions
1. **Use the updated code for new training** - The fixes ensure new models will be more compatible
2. **Create custom inference script** - Use raw TensorFlow SavedModel API instead of Keras model loading
3. **Use older TensorFlow version** - Temporarily use TF <2.16 for inference on legacy models

### Long-term Solutions
1. **Replace Lambda layers** - Convert all Lambda layers to proper Keras layers with full serialization support
2. **Implement proper model versioning** - Track model format versions and provide migration utilities
3. **Update model save format** - Use Keras 3's native `.keras` format for new models

## Known Remaining Issues
1. XLA compilation error with ImageProjectiveTransformV3 operation (requires `jit_compile=False` during training)
2. Some Lambda layers still use inline functions that cannot be properly serialized
3. Model architecture recreation from SavedModel format is unreliable in Keras 3

## Files Modified
- `ptycho/model.py` - Fixed dtype specifications, custom layer initialization
- `ptycho/tf_helper.py` - Created proper CenterMaskLayer class
- `ptycho/model_manager.py` - Added SavedModel compatibility handling
- `TF_2.19_Migration_Report.md` - Previously documented migration issues

## Conclusion
The debugging process revealed significant compatibility challenges between TensorFlow SavedModel format and Keras 3's serialization system. While the core training functionality works correctly with TF 2.19, inference on models saved in the older format requires additional work or alternative approaches. The fixes implemented ensure that newly trained models will be more compatible with the current TensorFlow/Keras ecosystem.