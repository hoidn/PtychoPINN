# PC-CCNF: Position-Conditioned Shared Canvas + Neural Field Decoder

## Overview

The PC-CCNF architecture is a hybrid neural network for ptychographic reconstruction that replaces the standard UNet autoencoder in PtychoPINN. It combines two key ideas:

1. **Position-Conditioned Shared Canvas (PC-SCD)**: A weight-shared CNN encoder processes each diffraction pattern independently, FiLM conditioning injects spatial coordinates into the bottleneck, and the conditioned latents are assembled onto a shared spatial canvas. Overlapping regions share the same canvas representation, providing structural phase consistency.

2. **Coordinate-Conditioned Neural Field (CCNF)**: Instead of a convolutional decoder that upsamples via transposed convolutions, a small MLP evaluates $f_\theta(\mathbf{r}, z_{\text{canvas}}) \to \mathbb{C}$ at each query coordinate. Fourier positional encoding at the coordinate level provides high-frequency representational capacity, directly counteracting the phase compression problem observed in the standard UNet decoder.

### Motivation

The standard UNet autoencoder in PtychoPINN suffers from three linked limitations:

- **No positional conditioning**: Four overlapping diffraction patterns enter as anonymous channels `(B, C=4, 64, 64)`. The network must implicitly learn fixed quadrant relationships (TL/TR/BL/BR) from channel ordering alone.
- **No explicit overlap encoding**: Overlap consistency is only enforced *after* the decoder, via the physics forward model (reassemble, re-extract, FFT, compare to measured diffraction). The latent space itself has no mechanism for ensuring that overlapping patches share a coherent representation.
- **Phase compression**: Convolutional decoders have an intrinsic smoothing bias from transposed convolution / upsampling operations. Combined with implicit averaging across channels, the model consistently compresses the dynamic phase range, producing unrealistically moderate phase values.

### Key Properties

- **Structural phase consistency**: Two patches sharing a physical region are decoded from the same canvas features. The neural field is a single-valued function of position, so identical canvas coordinates produce identical complex values. Phase consistency error is at float32 precision (~$10^{-7}$).
- **No smoothing bias**: Fourier positional encoding enables the MLP to represent sharp phase transitions without the frequency attenuation inherent in convolutional upsampling.
- **Flexible scan geometry**: The architecture handles arbitrary numbers of overlapping patches (any C) and arbitrary scan patterns. It is not locked to a rectangular grid.
- **Compatible with existing physics**: Output shape is `(B, C, N, N)` complex64, identical to the standard autoencoder. The `ForwardModel`, `RectangularScaledDiffraction`, `VarProScaler`, and all reassembly/inference code remain completely unchanged.

---

## Architecture Diagram

```
Input: (B, C, N, N) diffraction patterns + (B, C, 1, 2) coords_relative + (B, C, P, N, N) probe

  [1] Weight-Shared Encoder
      Reshape to (B*C, 1, N, N) -> ConvPoolBlocks -> (B*C, D, 8, 8)
      Reshape to (B, C, D, 8, 8)
              |
  [2] FiLM Positional Conditioning
      coords_relative -> Fourier Encode -> MLP -> (gamma, beta)
      z = z * sigmoid(gamma) + beta
      Output: (B, C, D, 8, 8)
              |
  [3] Latent Canvas Assembly
      Translate each patch's latent by coords/8 onto shared canvas
      Probe-weighted averaging in overlap regions
      Output: (B, D, M_lat, M_lat)
              |
  [4] Multi-Resolution Feature Volume
      Three parallel convolutions (1x1, 3x3, 3x3 dilated)
      Output: (B, 3*F, M_lat, M_lat)
              |
  [5] Neural Field Decoder
      For each output pixel: Fourier(coords) + grid_sample(features) -> MLP -> (amp, phase)
      Output: (B, C, N, N) complex64
              |
  [ForwardModel] (unchanged)
      Reassemble -> Probe x Object -> FFT -> Predicted Intensity
```

---

## Module Reference

### `FourierPositionalEncoding`

**Purpose**: Maps 2D spatial coordinates to a high-dimensional feature space using sinusoidal functions at geometrically-spaced frequency bands. This encoding allows the downstream MLP to represent high-frequency spatial variations without the spectral bias that standard MLPs exhibit toward smooth, low-frequency functions.

**Mathematical formulation**:

$$\gamma(\mathbf{r}) = \bigl[\sin(2^0 \pi r_x),\; \cos(2^0 \pi r_x),\; \ldots,\; \sin(2^{L-1} \pi r_x),\; \cos(2^{L-1} \pi r_x),\; \sin(2^0 \pi r_y),\; \cos(2^0 \pi r_y),\; \ldots,\; \sin(2^{L-1} \pi r_y),\; \cos(2^{L-1} \pi r_y)\bigr]$$

where $L$ = `num_bands` and $\mathbf{r} = (r_x, r_y)$.

**Parameters**:
- `num_bands` (int): Number of frequency octaves. Output dimension = $4 \times L$ (2 coordinates $\times$ sin/cos $\times$ $L$ bands).
- `max_freq_log2` (int, optional): Maximum frequency exponent. Defaults to `num_bands - 1`.

**Progressive annealing**: Call `set_active_bands(n)` to mask higher-frequency bands during early training. Inactive bands produce zero-valued features. This prevents high-frequency oscillations while the latent canvas stabilizes, and can be progressively increased during training (coarse-to-fine curriculum).

**Usage in the architecture**: Used in two places:
- `FiLMConditioning`: Encodes patch-level coordinates ($L = 32$ bands by default via `fourier_bands_film`).
- `NeuralFieldDecoder`: Encodes per-pixel query coordinates ($L = 10$ bands by default via `fourier_bands_coord`).

---

### `FiLMConditioning`

**Purpose**: Feature-wise Linear Modulation (FiLM) injects spatial position information into each patch's latent representation at the bottleneck. This tells the encoder's output WHERE each patch belongs in the solution region before canvas assembly.

**Mechanism**:

1. Fourier-encode `coords_relative` (B, C, 2) into positional features.
2. Pass through a 2-layer MLP to produce per-patch scale ($\gamma$) and shift ($\beta$) vectors.
3. Modulate the latent feature maps: $\mathbf{z}_i = \mathbf{z}_i \odot \sigma(\gamma_i) + \beta_i$

The sigmoid on $\gamma$ ensures multiplicative scaling is bounded in $[0, 1]$, preventing training instabilities from large scale factors while still allowing selective suppression of channels.

**Tensor flow**:
```
coords_relative: (B, C, 2)
    -> FourierPositionalEncoding -> (B, C, 4*L)
    -> Linear(4*L, film_dim) + ReLU -> (B, C, film_dim)
    -> Linear(film_dim, 2*D) -> (B, C, 2*D)
    -> split -> gamma: (B, C, D, 1, 1), beta: (B, C, D, 1, 1)
    -> z * sigmoid(gamma) + beta -> (B, C, D, H, W)
```

**Config parameters**: `fourier_bands_film` (default 32), `film_dim` (default 256).

---

### `LatentCanvasAssembler`

**Purpose**: Assembles position-conditioned per-patch latents onto a unified spatial canvas. This is the mechanism that creates a shared joint latent for overlapping regions: patches whose spatial positions overlap contribute to the same canvas pixels, and their contributions are averaged with probe-based weighting.

**Mechanism**:

1. Scale coordinates from pixel space to latent resolution (divide by 8, since the encoder downsamples 3 times with stride-2 pooling).
2. Pad each patch's `(D, 8, 8)` latent to `(D, M_lat, M_lat)` where `M_lat = ceil(get_padded_size() / 8)`.
3. Translate each padded latent to its correct canvas position using the existing `Translation()` function from `helper.py`.
4. Accumulate a weighted sum (numerator) and weight sum (denominator) across all C patches.
5. Normalize: `canvas = numerator / denominator`.

**Probe weighting**: When a probe tensor is provided, the weight for each patch is derived from the probe intensity $w_i = \sum_p |P_p|^2$ (summed over incoherent modes), downsampled to the latent resolution via adaptive average pooling. This provides physically-motivated confidence weighting: regions with stronger probe illumination contribute more to the canvas. When no probe is provided, uniform weighting is used.

**Reuse of existing infrastructure**: The assembly uses `helper.Translation()` (the same sub-pixel affine grid translation used throughout PtychoPINN) applied at 1/8 resolution. The latent feature channels are reshaped into the batch dimension for efficient batched translation.

**Tensor flow**:
```
latents: (B, C, D, 8, 8) + coords: (B, C, 1, 2) + probe: (B, C, P, N, N)

For each patch i in 0..C-1:
    Pad: (B, D, 8, 8) -> (B, D, M_lat, M_lat)
    Reshape D into batch: (B*D, M_lat, M_lat)
    Translate by coords[i]/8 using Translation()
    Reshape back: (B, D, M_lat, M_lat)
    Weight by probe intensity at latent resolution

Accumulate and normalize -> canvas: (B, D, M_lat, M_lat)
```

**Config dependency**: Canvas size is derived from `get_padded_size(data_config, model_config)`, which depends on `N`, `grid_size`, `max_neighbor_distance`, and `max_position_jitter`.

---

### `MultiResolutionFeatureVolume`

**Purpose**: Extracts features from the latent canvas at three different receptive field scales, providing the neural field decoder with both fine-grained local detail and broader spatial context.

**Architecture**: Three parallel convolutions applied to the same canvas:
- **Level 0**: `Conv2d(D, F, kernel=1)` -- pointwise features, no spatial mixing.
- **Level 1**: `Conv2d(D, F, kernel=3, padding=1)` -- local neighborhood ($3 \times 3$ receptive field).
- **Level 2**: `Conv2d(D, F, kernel=3, padding=2, dilation=2)` -- wider context ($5 \times 5$ effective receptive field).

The three outputs are concatenated along the channel dimension, producing a feature volume of depth $3F$.

**Tensor flow**:
```
canvas: (B, D, M_lat, M_lat)
    -> level0: (B, F, M_lat, M_lat)
    -> level1: (B, F, M_lat, M_lat)
    -> level2: (B, F, M_lat, M_lat)
    -> concat: (B, 3*F, M_lat, M_lat)
```

**Config parameter**: `feature_volume_channels` (default 64), so default output depth = 192.

---

### `NeuralFieldDecoder`

**Purpose**: The core decoder. Instead of convolutional upsampling, it evaluates a coordinate-conditioned MLP at every query pixel. This eliminates the smoothing bias of transposed convolutions and provides perfect phase consistency by construction.

**Mechanism**:

For each output patch $i$ (of $C$ total):

1. **Generate query coordinates**: Create an $N \times N$ meshgrid centered at the origin (local pixel coordinates). Add the patch's `coords_relative[i]` offset to get canvas-space coordinates.

2. **Sample feature volume**: Use `F.grid_sample` with bilinear interpolation to sample the multi-resolution feature volume at each query coordinate. This gives each pixel a spatially-localized conditioning vector from the canvas.

3. **Fourier encode coordinates**: Apply `FourierPositionalEncoding` to the query coordinates in pixel space. This provides the MLP with high-frequency positional information that enables sharp phase transitions.

4. **Concatenate and run MLP**: The input to the MLP is `[Fourier(r), sampled_features]`. The MLP is a sequence of `Linear + SiLU` layers, ending with a 2-output linear layer producing `(amplitude, phase)`.

5. **Apply activations**: Amplitude uses `sigmoid` (output in $[0, 1]$). Phase uses `1.2 * tanh` (output in $[-1.2, 1.2]$), matching the conventions of the existing UNet decoder.

6. **Combine to complex**: `result = amplitude + j * phase` (rectangular representation, matching `CombineComplexRectangular`).

**Phase consistency guarantee**: If two patches $i$ and $j$ both contain physical position $\mathbf{r}_0$, then:
- Patch $i$: query coord = $\mathbf{r}_0 - \text{offset}_i$ in local space, maps to canvas position $(\mathbf{r}_0 - \text{offset}_i) + \text{offset}_i = \mathbf{r}_0$ in canvas space.
- Patch $j$: query coord = $\mathbf{r}_0 - \text{offset}_j$ in local space, maps to canvas position $(\mathbf{r}_0 - \text{offset}_j) + \text{offset}_j = \mathbf{r}_0$ in canvas space.

Both sample the same canvas features at the same Fourier-encoded coordinate. The MLP is deterministic, so $f_\theta(\mathbf{r}_0, z) = f_\theta(\mathbf{r}_0, z)$ identically. Measured consistency error: ~$10^{-7}$ (float32 bilinear interpolation precision).

**Tensor flow**:
```
feature_volume: (B, 3*F, M_lat, M_lat) + coords: (B, C, 1, 2)

For all C patches simultaneously:
    grid_local: (1, 1, N*N, 2) meshgrid in pixel units
    r_canvas = grid_local + offsets: (B, C, N*N, 2) canvas coordinates
    Normalize to [-1,1] for grid_sample
    Reshape to (B, C*N*N, 1, 2) and sample feature_volume
    -> sampled: (B, C*N*N, 3*F)

    Fourier encode r_canvas: (B, C*N*N, 4*L_coord)
    Concat: (B, C*N*N, 4*L_coord + 3*F)
    MLP: -> (B, C*N*N, 2)
    Reshape: amplitude (B, C, N, N), phase (B, C, N, N)
    Combine: (B, C, N, N) complex64
```

**Config parameters**: `fourier_bands_coord` (default 10), `neural_field_hidden` (default `[512, 256, 128]`), `feature_volume_channels` (default 64).

---

### `AutoencoderCCNF`

**Purpose**: Composes all five modules into a single autoencoder that replaces the standard `Autoencoder` class.

**Pipeline**:

1. **Weight-shared encoder**: Reshapes input from `(B, C, N, N)` to `(B*C, 1, N, N)` and processes each patch independently through the same `Encoder` instance (from `model.py`). The encoder uses `C_model=1` (single-channel input) regardless of the actual number of overlapping patches. Optional CBAM attention at the bottleneck.

2. **FiLM conditioning**: Injects `coords_relative` into each patch's bottleneck representation.

3. **Canvas assembly**: Assembles conditioned latents onto a shared spatial canvas.

4. **Feature volume**: Extracts multi-scale features from the canvas.

5. **Neural field decoder**: Evaluates the coordinate-conditioned MLP at each output pixel.

**Encoder configuration**: The encoder is instantiated with a deep-copied `ModelConfig` where `C_model=1` and `object_big=False`, so its first convolution layer expects single-channel input. This allows weight sharing across all C patches regardless of how many overlapping positions are grouped together.

**Interface**: 
```python
def forward(self, x, coords, probe=None) -> (complex_out, amplitude, phase)
```
- `x`: `(B, C, N, N)` diffraction patterns (already RMS-scaled)
- `coords`: `(B, C, 1, 2)` relative coordinates per patch
- `probe`: `(B, C, P, N, N)` complex probe (optional, used for canvas weighting)
- Returns: `(B, C, N, N)` complex64 patches, plus separate amplitude and phase tensors

---

### `PtychoPINN_CCNF`

**Purpose**: Full model module that wraps `AutoencoderCCNF` with the intensity scaler and physics forward model, providing the same interface as the standard `PtychoPINN`.

**Forward pass**:
1. Scale input diffraction patterns by `input_scale_factor` (RMS normalization).
2. Run `AutoencoderCCNF(x, positions, probe)` to get complex object patches.
3. Pass patches through `ForwardModel.forward()` (reassemble, probe illumination, FFT, scaling).
4. Return predicted intensities, amplitude, and phase.

**Interface compatibility**: `PtychoPINN_CCNF.forward()` and `PtychoPINN_CCNF.forward_predict()` have identical signatures to `PtychoPINN`, so `PtychoPINN_Lightning` can use either model interchangeably based on `model_config.architecture`.

**Additional methods** (matching `PtychoPINN` interface):
- `get_encoder_bottom_params()` / `get_encoder_top_params()`: For staged fine-tuning with discriminative learning rates.
- `freeze_encoder()`: Freezes all encoder parameters for decoder-only fine-tuning.

---

## Configuration

All new parameters are added to `ModelConfig` in `config_params.py`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `architecture` | `'unet'` or `'ccnf'` | `'unet'` | Selects between standard UNet and PC-CCNF architecture. Frozen field. |
| `fourier_bands_film` | int | 32 | Number of Fourier frequency bands for FiLM positional encoding. |
| `fourier_bands_coord` | int | 10 | Number of Fourier frequency bands for the neural field's coordinate encoding. |
| `neural_field_hidden` | List[int] | [512, 256, 128] | Hidden layer dimensions for the neural field MLP. |
| `film_dim` | int | 256 | Hidden dimension of the FiLM conditioning MLP. |
| `feature_volume_channels` | int | 64 | Output channels per level in the multi-resolution feature volume (total = 3x this). |

**Example config JSON**:
```json
{
    "architecture": "ccnf",
    "fourier_bands_film": 32,
    "fourier_bands_coord": 10,
    "neural_field_hidden": [512, 256, 128],
    "film_dim": 256,
    "feature_volume_channels": 64
}
```

Setting `architecture` to `"unet"` (or omitting it) uses the standard UNet autoencoder with no changes to existing behavior.

---

## Integration with Existing Code

### Training pipeline

`PtychoPINN_Lightning.__init__` in `model.py` selects the model based on `model_config.architecture`:

```python
if getattr(model_config, 'architecture', 'unet') == 'ccnf':
    from ptycho_torch.beta_modules.latent_model import PtychoPINN_CCNF
    self.model = PtychoPINN_CCNF(model_config, data_config, training_config)
else:
    self.model = PtychoPINN(model_config, data_config, training_config)
```

The import is deferred (lazy) to avoid circular imports, since `latent_model.py` imports `Encoder`, `ForwardModel`, and `IntensityScalerModule` from `model.py`.

### Forward model

`ForwardModel.forward()` is completely unchanged. It receives `(B, C, N, N)` complex64 patches from the autoencoder and applies:
1. `reassemble_patches_position_real_probe()` -- assembles patches into a solution region (provides *additional* consistency enforcement on top of the architectural guarantee).
2. `extract_channels_from_region()` -- re-extracts patches from the assembled region.
3. `RectangularScaledDiffraction` -- applies `exit_wave = s1*(probe*real) + j*s2*(probe*imag)`, FFT, incoherent mode summation.

### Loss computation

`PtychoPINN_Lightning.compute_loss()` requires no changes. It already passes `positions` and `probe` through to `self.model()`, and `PtychoPINN_CCNF.forward()` has an identical signature to `PtychoPINN.forward()`.

### Inference

`forward_predict()` returns `(B, C, N, N)` complex64, compatible with all existing reconstruction functions (`reconstruct_image_barycentric`, `reconstruct_image_barycentric_weighted`, etc.).

---

## Training Recommendations

### Fourier frequency annealing

The neural field benefits from a coarse-to-fine training schedule:

1. **Epochs 0-10**: Set `decoder.coord_encoder.set_active_bands(4)` (of 10 total). This restricts the neural field to low-frequency outputs while the encoder and canvas assembly stabilize.
2. **Epochs 10-30**: Progressively increase active bands. By epoch 30, all bands should be active.
3. **Epochs 30+**: Full training with all frequencies. Standard LR scheduling applies.

This can be implemented in a Lightning callback:

```python
class FourierAnnealingCallback(L.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        total_bands = pl_module.model.autoencoder.decoder.coord_encoder.num_bands
        active = min(4 + epoch // 3, total_bands)
        pl_module.model.autoencoder.decoder.coord_encoder.set_active_bands(active)
```

### Loss function

The standard physics-based loss (Poisson or MAE on predicted vs. measured diffraction intensities) is sufficient. No additional overlap consistency loss is needed because the architecture enforces consistency structurally.

Existing regularizers (amplitude/phase mean deviation, total variation) apply unchanged to the amplitude and phase outputs.

---

## Computational Profile

**Parameter count** (~1.9M total):
- Weight-shared encoder: ~1.1M (ConvPoolBlocks with `n_filters_scale=2`, single-channel input)
- FiLM conditioning: ~130K (Fourier encoding + 2-layer MLP)
- Canvas assembly: 0 learnable parameters (uses `Translation()`)
- Multi-resolution feature volume: ~150K (three small convolutions)
- Neural field MLP: ~500K (three hidden layers + output layer)

For comparison, the standard UNet autoencoder has ~3.5M parameters (dual convolutional decoders).

**Forward pass cost**: The main computational expense is the MLP evaluation at every output pixel. For B=16, C=4, N=64: $16 \times 4 \times 64^2 = 262{,}144$ MLP forward passes per batch. The MLP is small (4 layers), so this is comparable in wall-clock time to a single convolutional decoder pass.

**Memory**: The multi-resolution feature volume `(B, 192, M_lat, M_lat)` is small ($M_\text{lat} \approx 12$). The MLP operates pointwise with no large intermediate feature maps. Peak memory is dominated by the encoder, as with the standard UNet.

---

## File Structure

```
ptycho_torch/
    config_params.py          # ModelConfig: new architecture selection fields
    model.py                  # PtychoPINN_Lightning: lazy import for CCNF model selection
    beta_modules/
        latent_model.py       # All PC-CCNF modules (this file)
        Latent_Model.md       # This documentation
```

**Dependencies from `model.py`**: `Encoder`, `ForwardModel`, `IntensityScalerModule`
**Dependencies from `model_attention.py`**: `CBAM`
**Dependencies from `helper.py`**: `Translation()`, `get_padded_size()`
