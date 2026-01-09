# PtychoPINN Implementation Analysis

This document summarizes the findings from analyzing the PtychoPINN codebase implementation details that extend beyond the documentation in `ptychopinnv2.tex`.

## 1. Model Architecture

### Adaptive Network Architecture
The model adapts its architecture based on the input size N:
- **N=64**: 3-layer encoder/decoder with filters [32, 64, 128] × `n_filters_scale`
- **N=128**: 4-layer encoder/decoder with filters [16, 32, 64, 128] × `n_filters_scale`  
- **N=256**: 5-layer encoder/decoder with filters [8, 16, 32, 64, 128] × `n_filters_scale`

This adaptive design ensures appropriate receptive fields for different reconstruction resolutions.

### Activation Functions
- **Amplitude decoder**: Configurable via `params['amp_activation']`
  - Options: sigmoid, swish, softplus, relu
  - Default: sigmoid
- **Phase decoder**: Fixed at π × tanh(x) to constrain phase to [-π, π]

## 2. Data Normalization Pipeline

The codebase implements a multi-stage normalization process:

### Stage 1: Initial Normalization (raw_data.py)
```python
X_full_norm = np.sqrt(((N / 2)**2) / np.mean(tf.reduce_sum(diffraction**2, axis=[1, 2])))
normalized_data = X_full_norm * diffraction
```
This ensures average total intensity per diffraction pattern equals (N/2)².

### Stage 2: Photon Scaling (diffsim.py)
Calculates a normalization factor based on desired photon count:
```python
norm = tf.math.sqrt(params['nphotons'] / mean_photons)
```

### Stage 3: External Scaling (model.py)
Before model input: `x_ext = params['intensity_scale'] * x`

### Stage 4: Internal Scaling (model.py)
Inside model: `x_in = x_ext / exp(α_log)` where α_log is trainable

## 3. Probe Boundary Handling

### Circular Support Constraint
When `probe.mask=True`:
- Probe confined to circular region with radius N/4
- Mask applied: `P_eff(r) = P(r) × mask(r)` where mask(r) = 1 if |r - center| < N/4, else 0
- Center offset by 0.5 pixels for proper pixel centering

### Optional Gaussian Smoothing
- Controlled by `gaussian_smoothing_sigma` parameter
- Applies complex-valued Gaussian filter to reduce sharp edges
- Only active when sigma ≠ 0

### Probe Initialization Methods
1. **Default disk**: Smooth circular probe via low-pass filtering
2. **Data-driven**: Estimated from average diffraction pattern
3. **User-provided**: Custom complex-valued probe

## 4. Object Reconstruction Boundary Handling

### Adaptive Boundary Learning (`pad_object=True`)
The decoder implements a sophisticated dual-pathway architecture:

1. **Central pathway**: 
   - Processes inner N/2 × N/2 region
   - Zero-padded by N/4 on each side
   
2. **Boundary pathway**:
   - Uses last 4 feature channels (`c_outer=4`)
   - Additional upsampling layers
   - Border mask ensures contribution only to outer region
   
3. **Combination**: `O_patch = O_central + O_boundary × border_mask`

This allows the network to learn physically plausible boundary values rather than assuming zeros.

### Zero Padding Mode (`pad_object=False`)
Simple zero-padding applied symmetrically around patches.

### Overlap Handling
For `object.big=True`:
- Patches padded to size M = N + (gridsize-1) × offset
- Overlapping regions averaged with normalization
- Small epsilon (1e-3) prevents division by zero

## 5. Solution Region Grouping Mechanism

The framework uses a sophisticated **nearest-neighbor clustering approach** to group individual scan points into overlapping solution regions, enabling reconstruction from irregular scan patterns.

### Core Algorithm

#### 1. K-Nearest Neighbor Finding
**Function**: `get_neighbor_indices()` (raw_data.py:365-385)
- Uses scipy's `cKDTree` for efficient spatial queries
- For each scan point (x,y), finds K+1 nearest neighbors (including itself)
- Creates index array of shape (N_scans, K+1) where each row contains neighbor indices

#### 2. Random Sampling for Solution Regions
**Function**: `sample_rows()` (raw_data.py:388-404)
- From each point's K+1 neighbors, randomly samples C points (where C = gridsize²)
- Generates `nsamples` different random selections per scan point
- Results in shape (N_scans, nsamples, C) → reshaped to (N_scans × nsamples, C)

#### 3. Coordinate Transformation
**Function**: `get_relative_coords()` (raw_data.py:333-346)
- **Global coordinates**: `coords_offsets = mean(coords_nn, axis=3)` - centroid of each solution region
- **Local coordinates**: `coords_relative = (coords_nn - coords_offsets)` - relative positions within each region

#### 4. Data Grouping
**Function**: `get_neighbor_diffraction_and_positions()` (raw_data.py:407-454)
- Groups diffraction patterns according to neighbor indices: `diff4d_nn = diff3d[nn_indices]`
- Transposes to channel format: `(batch, height, width, C)` where C = gridsize²
- Creates coordinate arrays for both global offsets and relative positions

### Key Parameters
- **K**: Number of nearest neighbors to consider (default: 6-7)
- **C**: Patches per solution region = gridsize² (typically 4 for gridsize=2)
- **nsamples**: Number of random samplings per scan point (default: 1-10)

### Special Cases
1. **Single patch mode** (C=1): Uses `get_neighbor_self_indices()` - each point forms its own "solution region"
2. **Regular grids**: Still uses neighbor finding but results in more predictable groupings

### Example Workflow
For a scan with 100 points, K=6, gridsize=2, nsamples=5:
1. Find 7 nearest neighbors for each of 100 points
2. Randomly sample 4 points from each neighborhood, 5 times
3. Results in 500 solution regions, each containing 4 overlapping patches
4. Each solution region has global position (centroid) and 4 relative patch positions

### Benefits
- **Irregular scan support**: Handles arbitrary scan patterns without grid assumptions
- **Overlap constraints**: Neighboring solution regions share scan points, providing consistency constraints
- **Data augmentation**: Multiple random samplings increase training data diversity
- **Scalability**: KDTree queries scale efficiently with scan size

## 6. Additional Configuration Parameters

### Not Documented in TEX
- `intensity_scale.trainable` (default: False) - Whether internal scaling is learnable
- `pad_object` (default: True) - Enables adaptive boundary learning
- `probe.mask` (default: True) - Applies circular probe support
- `gaussian_smoothing_sigma` (default: 0.0) - Probe smoothing parameter
- `probe.big` - Whether probe varies across scan positions
- `max_position_jitter` - Position uncertainty during training

### Implementation Details
- Physics model integrated into model.py rather than separate physics.py module
- Complex convolution mentioned in TODO but not implemented
- Supports position correction during training via jitter parameter

## 7. Key Implementation Insights

### Strengths
1. **Flexible boundary handling**: Adaptive learning vs zero-padding options
2. **Robust normalization**: Multi-stage approach handles diverse datasets
3. **Physical constraints**: Probe masking and phase constraints ensure physical validity
4. **Memory efficient**: Overlap averaging computed via gradient operations

### Limitations
1. **Fixed probe assumption**: No per-position probe variation in standard mode
2. **Rectangular patches only**: Non-rectangular scan patterns require preprocessing
3. **Zero boundary assumption**: Even adaptive mode assumes zero outside full reconstruction region

### Recommendations
1. Use `pad_object=True` for better boundary behavior
2. Enable `probe.mask=True` for physical probe constraints
3. Consider `gaussian_smoothing_sigma` > 0 if probe edges cause artifacts
4. Monitor `intensity_scale` parameter for convergence issues

## 8. Code Organization

The implementation spreads key functionality across multiple files:
- **model.py**: Main model definition, forward pass, loss computation
- **tf_helper.py**: Low-level operations (padding, FFT, patch extraction)
- **probe.py**: Probe initialization and constraints
- **loader.py**: Data loading and normalization
- **physics.py**: Currently empty (functionality in model.py)

This modular structure allows flexibility but can make the full pipeline harder to trace.

## Summary

The PtychoPINN implementation is more sophisticated than the documentation suggests, with adaptive architectures, flexible boundary handling, and multi-stage normalization. The code includes several experimental features and optimizations not described in the paper, providing additional flexibility for different use cases while maintaining the core physics-informed approach.