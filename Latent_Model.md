# PC-CCNF: Position-Conditioned Shared Canvas + Neural Field Decoder

## Overview

The PC-CCNF architecture is a hybrid neural network for ptychographic reconstruction that replaces the standard UNet autoencoder in PtychoPINN. It combines two key ideas:

1. **Position-Conditioned Shared Canvas**: A weight-shared CNN encoder processes each diffraction pattern independently. The resulting latents are fused onto a shared spatial canvas via geometry-tagged cross-attention, which uses Fourier-encoded local offsets to learn non-linear combination of overlapping patch observations.

2. **Coordinate-Conditioned Neural Field (CCNF)**: Instead of a convolutional decoder that upsamples via transposed convolutions, a small MLP evaluates $f_\theta(\mathbf{r}, z_{\text{canvas}}) \to \mathbb{C}$ at each query coordinate. Fourier positional encoding at the coordinate level provides high-frequency representational capacity, directly counteracting the phase compression problem observed in the standard UNet decoder.

### Motivation

The standard UNet autoencoder in PtychoPINN suffers from three linked limitations:

- **No positional conditioning**: Four overlapping diffraction patterns enter as anonymous channels `(B, C=4, 64, 64)`. The network must implicitly learn fixed quadrant relationships (TL/TR/BL/BR) from channel ordering alone.
- **No explicit overlap encoding**: Overlap consistency is only enforced *after* the decoder, via the physics forward model (reassemble, re-extract, FFT, compare to measured diffraction). The latent space itself has no mechanism for ensuring that overlapping patches share a coherent representation.
- **Phase compression**: Convolutional decoders have an intrinsic smoothing bias from transposed convolution / upsampling operations. Combined with implicit averaging across channels, the model consistently compresses the dynamic phase range, producing unrealistically moderate phase values.

### Key Properties

- **Structural phase consistency**: Two patches sharing a physical region are decoded from the same canvas features. The neural field is a single-valued function of position, so identical canvas coordinates produce identical complex values. Phase consistency error is at float32 precision (~$10^{-7}$).
- **No smoothing bias**: Fourier positional encoding enables the MLP to represent sharp phase transitions without the frequency attenuation inherent in convolutional upsampling.
- **Flexible scan geometry**: The architecture handles arbitrary numbers of overlapping patches (any C) and arbitrary scan patterns. Dynamic per-axis canvas scaling automatically adapts to both isotropic and rectangular scan geometries.
- **Compatible with existing physics**: Output shape is `(B, C, N, N)` complex64, identical to the standard autoencoder. The `ForwardModel`, `RectangularScaledDiffraction`, `VarProScaler`, and all reassembly/inference code remain completely unchanged.

---

## Architecture Diagram

```
Input: (B, C, N, N) diffraction patterns + (B, C, 1, 2) coords_relative + (B, C, P, N, N) probe

  [1] Weight-Shared Encoder (per-patch, single-channel)
      Reshape to (B*C, 1, N, N)

      encoder_type='cnn':
        ConvPoolBlocks -> (B*C, D, 8, 8)

      encoder_type='fno_cnn':
        Lifting Conv1x1 -> (B*C, W, N, N)
        FNO Blocks (full resolution, global context) -> (B*C, W, N, N)
        CNN ConvPoolBlocks (spatial compression) -> (B*C, D, 16, 16)
        Final MaxPool -> (B*C, D, 8, 8)

      Reshape to (B, C, D, 8, 8)
              |
  [2] Dynamic Canvas Scaling
      compute_canvas_scale(coords, N, M_lat)
      Per-batch, per-axis scale: (B, 1, 2)
              |
  [3] Geometry-Tagged Cross-Attention Fusion
      Bilinear splatting with dynamic scale
      Fourier geometry tags + multi-head cross-attention
      Output: (B, D, M_lat, M_lat)
              |
  [4] Multi-Resolution Feature Volume
      Three parallel convolutions (1x1, 3x3, 3x3 dilated)
      Output: (B, 3*F, M_lat, M_lat)
              |
  [5] Neural Field Decoder
      For each output pixel: Fourier(coords) + grid_sample(features) -> MLP -> (x_real, x_imag)
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
- `GeometryTaggedAttentionFusion`: Encodes local geometry offset tags ($L = 3$ bands by default via `fusion_tag_bands`).
- `NeuralFieldDecoder`: Encodes per-pixel query coordinates ($L = 10$ bands by default via `fourier_bands_coord`).

---

### `compute_canvas_scale`

**Purpose**: Computes per-batch, per-axis scale factors that dynamically map coordinate bounding boxes to fill the latent canvas. This replaces the previous hardcoded `spatial_size=8` divisor, enabling automatic adaptation to arbitrary scan geometries.

**Motivation**: With a fixed divisor, the mapping from pixel coordinates to canvas positions was static. This caused two problems:
- **Small coordinate spread** (e.g., y-axis of a rectangular scan with 2 $\mu$m step): patches occupied a tiny fraction of the canvas, wasting most of the spatial capacity.
- **Large coordinate spread**: patches could extend beyond the canvas boundary, causing feature clipping.

Dynamic per-axis scaling solves both: each axis of the coordinate bounding box (plus patch extent) is scaled to fill the canvas independently.

**Mathematical formulation**:

$$\text{scale}_x = \frac{M_\text{lat} - \text{margin}}{\text{bbox}_x + N}, \quad \text{scale}_y = \frac{M_\text{lat} - \text{margin}}{\text{bbox}_y + N}$$

where $\text{bbox}$ is the max-min coordinate spread across the $C$ patches in each batch element, $N$ is the diffraction pattern size (patch extent), and margin (default 2) prevents edge clipping.

**Signature**:
```python
def compute_canvas_scale(coords, N, M_lat, margin=2) -> torch.Tensor:
    # coords: (B, C, 1, 2) relative coordinates in pixel units
    # Returns: (B, 1, 2) canvas pixels per real pixel, per axis
```

**Example values** (N=64, M_lat=16):

| Scan geometry | bbox (px) | scale | Normalized max coord |
|--------------|-----------|-------|---------------------|
| Isotropic (20, 20) | (20, 20) | (0.167, 0.167) | ~0.86 |
| Rectangular (24, 4) | (24, 4) | (0.159, 0.206) | ~0.86 |
| Degenerate (0, 0) | (0, 0) | (0.219, 0.219) | ~0.68 |

The scale is clamped so `total_extent >= N`, ensuring the degenerate case (all coords identical) produces valid scaling.

**Invariant**: The Fourier positional encoding in the decoder operates on raw pixel-unit coordinates *before* scaling, so the MLP sees physically meaningful frequencies regardless of the canvas mapping.

---

### `SpectralConv2d`

**Purpose**: Core learnable Fourier-domain convolution layer. Performs a pointwise multiplication of the input's Fourier transform by a learnable complex weight tensor, truncated to the lowest `modes` frequencies in each spatial dimension. This is the building block of FNO layers -- multiplication in Fourier space is equivalent to convolution in real space with a kernel that spans the entire input, providing global receptive field from a single layer.

**Physical interpretation**: Applying FFT to a diffraction pattern (which is already $|\mathcal{F}\{P \cdot O\}|^2$) yields the autocorrelation of $P \cdot O$ -- the Patterson function in crystallography. The learnable spectral filter $R(\mathbf{k})$ therefore operates on inter-feature distance information, which is physically meaningful for ptychographic reconstruction.

**Mechanism**:

$$\text{SpectralConv2d}(x) = \mathcal{F}^{-1}\bigl[R(\mathbf{k}) \cdot \mathcal{F}[x]\bigr]$$

where $R(\mathbf{k})$ is a learnable complex weight tensor of shape `(in_ch, out_ch, modes, modes)`.

Implementation details:
- Uses `torch.fft.rfft2` (real FFT) with `norm='ortho'` for energy-preserving round-trips.
- The rfft2 output has shape `(H, W//2+1)` due to Hermitian symmetry of real signals. For 64x64 input, this gives `(64, 33)` complex values.
- Two spectral regions are multiplied: positive-$k_y$ corner `[:modes, :modes]` and negative-$k_y$ (wrapped) corner `[-modes:, :modes]`.
- Weight stored as real tensor with shape `(in_ch, out_ch, modes, modes, 2)`, converted to complex via `torch.view_as_complex` in the forward pass (avoids optimizer issues with complex parameters).
- Initialized with scale `1 / (in_ch * out_ch)` -- small perturbation so the FNO block starts near-identity when combined with the residual connection.

**Tensor flow**:
```
x: (B, C_in, H, W)
  -> rfft2 -> (B, C_in, H, W//2+1) complex
  -> truncate to modes, einsum with R(k) -> (B, C_out, H, W//2+1) complex
  -> irfft2 -> (B, C_out, H, W) real
```

---

### `FNOBlock`

**Purpose**: Single Fourier Neural Operator processing block that combines a global spectral path with a local pointwise path, normalization, activation, and a residual connection.

**Architecture**:

$$\text{FNOBlock}(x) = x + \text{GELU}\bigl(\text{InstanceNorm}\bigl(\text{SpectralConv2d}(x) + \text{Conv1x1}(x)\bigr)\bigr)$$

Components:
- **Spectral path** (`SpectralConv2d`): Global frequency-domain processing. Captures long-range correlations (Friedel conjugate pairs, radial ring structure, speckle correlations across the full detector).
- **Local path** (`Conv2d` with `kernel_size=1`): Pointwise features that complement the spectral path. The spectral path truncates to `modes` frequencies, so it cannot represent features with spatial frequency above the cutoff -- the 1x1 conv fills this gap.
- **InstanceNorm2d** (affine=True): Per-sample normalization. Chosen over BatchNorm because the effective batch size can be small during weight-shared encoding (B*C elements, but patterns can have very different intensity scales).
- **GELU activation**: Smoother than ReLU, standard for transformer/FNO architectures.
- **Residual connection**: When `in_channels == out_channels` (always true after lifting). The block starts near-identity due to the small spectral weight initialization.

**Tensor flow**: `(B, C, H, W) -> (B, C, H, W)` (spatial dimensions preserved).

---

### `FNOCNNEncoder`

**Purpose**: Hybrid encoder that combines FNO blocks at full spatial resolution for global context extraction with CNN blocks for spatial compression to the 8x8 bottleneck. This replaces the pure CNN `Encoder` when `encoder_type='fno_cnn'`.

**Motivation**: Diffraction patterns have fundamentally non-local correlations -- a single object feature affects every detector pixel. A CNN with stacked 3x3 kernels only achieves global receptive field at the deepest layer, where spatial resolution is already compressed to 8x8 and fine-grained long-range correlations are lost. FNO blocks provide global context from the first layer via spectral multiplication (equivalent to a full-image convolution).

**Architecture** (for N=64, `fno_width=32`, `fno_blocks=2`, `n_filters_scale=2`):

```
Input: (B*C, 1, 64, 64)

Lifting Conv1x1:  (B*C, 1, 64, 64)   -> (B*C, 32, 64, 64)    project to working width
FNO Block 1:      (B*C, 32, 64, 64)   -> (B*C, 32, 64, 64)    global context  [skip]
FNO Block 2:      (B*C, 32, 64, 64)   -> (B*C, 32, 64, 64)    global context  [skip]
ConvPoolBlock 0:  (B*C, 32, 64, 64)   -> (B*C, 128, 32, 32)   compress        [skip]
ConvPoolBlock 1:  (B*C, 128, 32, 32)  -> (B*C, 256, 16, 16)   compress        [skip]
Final MaxPool:    (B*C, 256, 16, 16)  -> (B*C, 256, 8, 8)      bottleneck

Output: (B*C, 256, 8, 8)
```

**Generalized filter list**: The CNN filter progression is computed programmatically for any power-of-two $N \geq 16$:
```python
n_cnn_blocks = int(math.log2(N / 8)) - 1
cnn_filter_list = [fno_width] + [nfs * (128 >> i) for i in range(n_cnn_blocks - 1, -1, -1)]
```

For N=64: `[W, nfs*64, nfs*128]` (2 CNN blocks).
For N=128: `[W, nfs*32, nfs*64, nfs*128]` (3 CNN blocks).
For N=256: `[W, nfs*16, nfs*32, nfs*64, nfs*128]` (4 CNN blocks).

**API compatibility**: Exposes the same interface as the standard `Encoder`:
- `forward(x) -> (bottleneck, skips)` with bottleneck shape `(B*C, D, 8, 8)`
- `self.blocks` (ModuleList): concatenation of FNO blocks + CNN blocks, used by `get_encoder_bottom_params()` / `get_encoder_top_params()` for staged fine-tuning. The split naturally separates FNO blocks (global, freeze first) from CNN blocks (compression, adapt more).
- `self.filters` (list): channel counts per stage, used by `AutoencoderCCNF` to determine `bottleneck_channels`.

**Config parameters**: `encoder_type` (`'cnn'` or `'fno_cnn'`), `fno_width` (default 32), `fno_modes` (default 16), `fno_blocks` (default 2). All frozen fields.

---

### `GeometryTaggedAttentionFusion`

**Purpose**: Performs relational fusion of overlapping patch latents via geometry-aware cross-attention. Instead of value-based averaging (which discards multi-view structure), this module uses attention to learn how to weight and combine observations from different patches at each canvas position, conditioned on each patch's local geometry offset.

**Dynamic canvas scaling**: Accepts a per-batch, per-axis `scale` tensor (from `compute_canvas_scale`) that maps pixel-unit coordinates to canvas-pixel positions. This replaces the previous fixed `spatial_size=8` divisor and enables automatic adaptation to any scan geometry, including rectangular scans where $\Delta x \gg \Delta y$.

**Mechanism**:

1. **Bilinear splatting**: For each patch $k$, use `F.grid_sample` to sample its $8 \times 8$ feature map at every canvas position. Positions outside the patch's spatial extent receive zero features (`padding_mode='zeros'`). The grid_sample normalization accounts for dynamic scaling:

$$\text{norm\_local} = \frac{\text{local\_coords} \times 2}{N \times \text{scale}}$$

This is algebraically equivalent to $2 \times \text{real\_distance} / N$, making the splatting invariant to the canvas scale.

2. **Geometry tags**: For each canvas position $(i, j)$ and each patch $k$, compute the local offset $\tau_k = (i, j) - \text{center}_k$ in canvas-pixel units. Fourier-encode: $\gamma(\tau_k) \in \mathbb{R}^{\text{tag\_dim}}$. This tells the attention mechanism *where in the probe* this evidence originated.

3. **Multi-head cross-attention** at each of the $S = M_\text{lat}^2$ canvas positions:

$$Q = W_Q \cdot \text{mean}(\{z_k\}_{k=1}^C) \quad \text{(mean-pooled query)}$$

$$K_k = W_K \cdot [z_k \;;\; \gamma(\tau_k)] \quad \text{(features concatenated with geometry tag)}$$

$$V_k = W_V \cdot z_k$$

$$\alpha_k = \text{softmax}\left(\frac{Q \cdot K_k^\top}{\sqrt{d_\text{head}}}\right), \quad \text{out} = \sum_k \alpha_k \cdot V_k$$

4. **Output projection + residual**: $\text{canvas} = \text{LayerNorm}(W_\text{out} \cdot \text{out} + Q)$

The residual connection from $Q$ and the zero-initialization of $W_\text{out}$ ensure the module starts close to mean-pooling behavior, providing a smooth training transition.

**Tensor flow**:
```
latents: (B, C, D, 8, 8) + coords: (B, C, 1, 2) + scale: (B, 1, 2)

Bilinear splatting:
  patch_centers = coords * scale                            (B, C, 2)
  local_coords = canvas_grid - centers (broadcast)          (B, C, S, 2)
  norm_local = local_coords * 2 / (N * scale)               (B, C, S, 2)
  F.grid_sample per patch -> splatted                       (B, C, D, M_lat, M_lat)
  reshape -> features                                       (B, S, C, D)

Geometry tags:
  FourierEncode(local_coords) -> tags                       (B, S, C, tag_dim)

Cross-attention:
  Q = proj(mean(features))                                  (B, S, D)
  K = proj([features ; tags])                               (B, S, C, D)
  V = proj(features)                                        (B, S, C, D)
  Multi-head attention + residual + LayerNorm               (B, S, D)

Reshape -> canvas                                           (B, D, M_lat, M_lat)
```

**Edge cases**:
- Canvas positions with no patch coverage: all splatted features = 0, output $\approx$ 0 (correct).
- $C = 1$: softmax over 1 element = 1.0, output = single patch's features (identity).
- Variable $C$: the module reads $C$ from the tensor shape, not a fixed config value. Works for $C = 4$ during training and $C = 10+$ during inference.
- **Rectangular scans**: per-axis scaling stretches the tighter axis to fill the canvas, giving effective anisotropic resolution without changing tensor shapes.

**Config parameters**: `fusion_heads` (default 4), `fusion_tag_bands` (default 3).

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

**Dynamic canvas scaling**: Like the fusion module, the decoder accepts a per-batch, per-axis `scale` tensor. Query coordinates in pixel units are scaled to canvas units before normalization for `grid_sample`, ensuring the feature volume is sampled at the correct positions regardless of scan geometry.

**Output representation**: The MLP outputs two values per pixel that are combined in rectangular (Cartesian) form:
- `x_real = 0.2 + tanh(output[..., 0])` -- real part, range $[-0.8, 1.2]$
- `x_imag = 1.2 * tanh(output[..., 1])` -- imaginary part, range $[-1.2, 1.2]$
- `result = x_real + j \cdot x_imag`

This matches the rectangular decomposition used by the standard autoencoder (`Decoder_amp` + `Decoder_phase` combined via addition, not polar form).

**Mechanism**:

For each output patch $i$ (of $C$ total):

1. **Generate query coordinates**: Create an $N \times N$ meshgrid centered at the origin (local pixel coordinates). Add the patch's `coords_relative[i]` offset to get canvas-space coordinates.

2. **Dynamic scale normalization**: Apply per-axis scale and normalize to $[-1, 1]$:

$$r_\text{normalized} = \frac{r_\text{canvas} \times \text{scale}}{M_\text{lat} / 2}$$

3. **Sample feature volume**: Use `F.grid_sample` with bilinear interpolation to sample the multi-resolution feature volume at each normalized query coordinate. This gives each pixel a spatially-localized conditioning vector from the canvas.

4. **Fourier encode coordinates**: Apply `FourierPositionalEncoding` to the query coordinates in *pixel space* (pre-scaling). This provides the MLP with physically meaningful high-frequency positional information, invariant to the canvas scaling.

5. **Concatenate and run MLP**: The input to the MLP is `[Fourier(r_pixel), sampled_features]`. The MLP is a sequence of `Linear + SiLU` layers, ending with a 2-output linear layer producing `(x_real, x_imag)`.

6. **Apply activations and combine**: $0.2 + \text{tanh}$ for real, $1.2 \times \text{tanh}$ for imaginary, combined as rectangular complex.

**Phase consistency guarantee**: If two patches $i$ and $j$ both contain physical position $\mathbf{r}_0$, then:
- Patch $i$: query coord = $\mathbf{r}_0 - \text{offset}_i$ in local space, maps to canvas position $(\mathbf{r}_0 - \text{offset}_i) + \text{offset}_i = \mathbf{r}_0$ in canvas space.
- Patch $j$: query coord = $\mathbf{r}_0 - \text{offset}_j$ in local space, maps to canvas position $(\mathbf{r}_0 - \text{offset}_j) + \text{offset}_j = \mathbf{r}_0$ in canvas space.

Both sample the same canvas features at the same Fourier-encoded coordinate. The MLP is deterministic, so $f_\theta(\mathbf{r}_0, z) = f_\theta(\mathbf{r}_0, z)$ identically. Measured consistency error: ~$10^{-7}$ (float32 bilinear interpolation precision).

**Tensor flow**:
```
feature_volume: (B, 3*F, M_lat, M_lat) + coords: (B, C, 1, 2) + scale: (B, 1, 2)

For all C patches simultaneously:
    grid_local: (1, 1, N*N, 2) meshgrid in pixel units
    r_canvas = grid_local + offsets: (B, C, N*N, 2) pixel-unit canvas coordinates
    r_scaled = r_canvas * scale: (B, C, N*N, 2) canvas-pixel coordinates
    r_normalized = r_scaled / (M_lat / 2): (B, C, N*N, 2) in [-1, 1]
    Reshape to (B, C*N*N, 1, 2) and sample feature_volume
    -> sampled: (B, C*N*N, 3*F)

    Fourier encode r_canvas (pixel units): (B, C*N*N, 4*L_coord)
    Concat: (B, C*N*N, 4*L_coord + 3*F)
    MLP: -> (B, C*N*N, 2)
    Reshape: x_real (B, C, N, N), x_imag (B, C, N, N)
    Combine: (B, C, N, N) complex64
```

**Config parameters**: `fourier_bands_coord` (default 10), `neural_field_hidden` (default `[512, 256, 128]`), `feature_volume_channels` (default 64).

---

### `AutoencoderCCNF`

**Purpose**: Composes all modules into a single autoencoder that replaces the standard `Autoencoder` class.

**Pipeline**:

1. **Dynamic canvas scaling**: Computes per-batch, per-axis scale factors from the coordinate bounding box and patch extent via `compute_canvas_scale(coords, N, M_lat)`.

2. **Weight-shared encoder**: Reshapes input from `(B, C, N, N)` to `(B*C, 1, N, N)` and processes each patch independently through the same encoder instance. When `encoder_type='fno_cnn'`, uses `FNOCNNEncoder` (FNO blocks for global context + CNN blocks for compression). When `encoder_type='cnn'` (default), uses the standard `Encoder` from `model.py`. Both produce `(B*C, D, 8, 8)` bottleneck output. Optional CBAM attention at the bottleneck.

3. **Geometry-tagged cross-attention fusion**: Bilinearly splats each patch's latent onto the shared canvas using the dynamic scale, tags contributions with local geometry offsets, and fuses via multi-head cross-attention.

4. **Feature volume**: Extracts multi-scale features from the fused canvas.

5. **Neural field decoder**: Evaluates the coordinate-conditioned MLP at each output pixel using the dynamic scale for feature volume sampling.

**Encoder configuration**: The encoder is instantiated with a deep-copied `ModelConfig` where `C_model=1` and `object_big=False`, so its first layer expects single-channel input. The `encoder_type` field selects between `Encoder` (pure CNN) and `FNOCNNEncoder` (FNO+CNN hybrid). This allows weight sharing across all C patches regardless of how many overlapping positions are grouped together.

**Interface**: 
```python
def forward(self, x, coords, probe=None) -> (complex_out, x_real, x_imag)
```
- `x`: `(B, C, N, N)` diffraction patterns (already RMS-scaled)
- `coords`: `(B, C, 1, 2)` relative coordinates per patch (pixel units)
- `probe`: `(B, C, P, N, N)` complex probe (optional)
- Returns: `(B, C, N, N)` complex64 patches, plus separate real and imaginary tensors

---

### `PtychoPINN_CCNF`

**Purpose**: Full model module that wraps `AutoencoderCCNF` with the intensity scaler and physics forward model, providing the same interface as the standard `PtychoPINN`.

**Forward pass**:
1. Scale input diffraction patterns by `input_scale_factor` (RMS normalization).
2. Run `AutoencoderCCNF(x, positions, probe)` to get complex object patches.
3. Pass patches through `ForwardModel.forward()` (reassemble, probe illumination, FFT, scaling).
4. Return predicted intensities, real component, and imaginary component.

**Interface compatibility**: `PtychoPINN_CCNF.forward()` and `PtychoPINN_CCNF.forward_predict()` have identical signatures to `PtychoPINN`, so `PtychoPINN_Lightning` can use either model interchangeably based on `model_config.architecture`.

**Additional methods** (matching `PtychoPINN` interface):
- `get_encoder_bottom_params()` / `get_encoder_top_params()`: For staged fine-tuning with discriminative learning rates.
- `freeze_encoder()`: Freezes all encoder parameters for decoder-only fine-tuning.

---

## Configuration

All new parameters are added to `ModelConfig` in `config_params.py`:

| Parameter | Type | Default | Frozen | Description |
|-----------|------|---------|--------|-------------|
| `architecture` | `'unet'` / `'ccnf'` | `'unet'` | Yes | Selects between standard UNet and PC-CCNF architecture. |
| `encoder_type` | `'cnn'` / `'fno_cnn'` | `'cnn'` | Yes | Encoder variant. `'cnn'` uses standard ConvPoolBlocks. `'fno_cnn'` uses FNO blocks at full resolution + CNN compression. |
| `fno_modes` | int | 16 | Yes | Number of truncated Fourier modes per dimension in `SpectralConv2d`. For N=64, rfft2 gives 33 bins; 16 retains ~half the spectrum. |
| `fno_width` | int | 32 | Yes | Channel width of FNO layers. Controls parameter count quadratically (spectral weight is `W x W x modes x modes x 2`). |
| `fno_blocks` | int | 2 | Yes | Number of full-resolution FNO blocks before CNN compression. |
| `latent_canvas_size` | int | 16 | No | Size of the square latent canvas ($M_\text{lat} \times M_\text{lat}$). Dynamic scaling maps the coordinate bounding box to fill this canvas. |
| `fusion_heads` | int | 4 | No | Number of attention heads in geometry-tagged fusion. |
| `fusion_tag_bands` | int | 3 | No | Fourier frequency bands for geometry offset tags in the fusion module. |
| `fourier_bands_coord` | int | 10 | No | Fourier frequency bands for the neural field's coordinate encoding. |
| `neural_field_hidden` | List[int] | [512, 256, 128] | No | Hidden layer dimensions for the neural field MLP. |
| `feature_volume_channels` | int | 64 | No | Output channels per level in the multi-resolution feature volume (total = 3x this). |

**Example config JSON** (FNO+CNN encoder):
```json
{
    "architecture": "ccnf",
    "encoder_type": "fno_cnn",
    "fno_modes": 16,
    "fno_width": 32,
    "fno_blocks": 2,
    "latent_canvas_size": 16,
    "fourier_bands_coord": 10,
    "neural_field_hidden": [512, 256, 128],
    "feature_volume_channels": 64
}
```

Setting `architecture` to `"unet"` (or omitting it) uses the standard UNet autoencoder with no changes to existing behavior. Setting `encoder_type` to `"cnn"` (or omitting it) uses the standard CNN encoder within the CCNF architecture.

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

Existing regularizers (real/imaginary mean deviation, total variation) apply unchanged to the `x_real` and `x_imag` outputs.

---

## Computational Profile

### With CNN encoder (`encoder_type='cnn'`, ~1.8M total):
- Weight-shared encoder: ~1.1M (ConvPoolBlocks with `n_filters_scale=2`, single-channel input)
- Geometry-tagged attention fusion: ~270K (projections + tag encoder)
- Multi-resolution feature volume: ~150K (three small convolutions)
- Neural field MLP: ~500K (three hidden layers + output layer)

### With FNO+CNN encoder (`encoder_type='fno_cnn'`, ~2.8M total):
- FNO lifting + 2 FNO blocks (`fno_width=32`, `fno_modes=16`): ~1.05M
  - Spectral weights dominate: `32 x 32 x 16 x 16 x 2 = 524K` per block
- CNN compression blocks: ~1.1M (ConvPoolBlocks for 64->32->16, MaxPool to 8)
- Fusion + feature volume + MLP: ~920K (unchanged)

Parameter count scales quadratically with `fno_width`:

| `fno_width` | FNO blocks | CNN blocks | Rest | **Total** |
|-------------|-----------|------------|------|-----------|
| 32          | ~1.05M    | ~1.1M      | ~0.92M | **~3.1M** |
| 64          | ~4.2M     | ~1.1M      | ~0.92M | **~6.2M** |

For comparison, the standard UNet autoencoder has ~3.5M parameters (dual convolutional decoders).

**Forward pass cost**: The FNO blocks add FFT/iFFT operations at full resolution (O($N^2 \log N$) per block), which is cheaper than self-attention (O($N^4$)) and comparable to a single conv layer at the same resolution. The MLP decoder evaluates $B \times C \times N^2$ points per batch (262K for B=16, C=4, N=64) -- comparable to a single convolutional decoder pass.

**Memory**: The FNO blocks operate at full resolution, so the rfft2 intermediate `(B*C, W, N, N//2+1)` complex tensor is the main addition. For B=16, C=4, `fno_width=32`, N=64: ~69MB per block. The multi-resolution feature volume `(B, 192, M_lat, M_lat)` remains small ($M_\text{lat} = 16$).

---

## File Structure

```
ptycho_torch/
    config_params.py          # ModelConfig: architecture selection fields
    model.py                  # PtychoPINN_Lightning: lazy import for CCNF model selection
    beta_modules/
        latent_model.py       # All PC-CCNF modules (this file)
Latent_Model.md               # This documentation
```

**Dependencies from `model.py`**: `Encoder`, `ForwardModel`, `IntensityScalerModule`, `ConvPoolBlock`
**Dependencies from `model_attention.py`**: `CBAM`
