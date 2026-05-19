# Overlap Baseline: Pairwise-Offset Gated Mixing

## Overview

The Overlap Baseline is a minimal architecture for testing whether latent-space overlap fusion improves ptychographic reconstruction, independent of the more complex machinery in the full PC-CCNF (`latent_model.py`). It isolates a single question: **does a learned, position-aware mixing of encoder bottleneck features across overlapping patches help the decoder?**

The architecture shares the same FNO+CNN encoder and PixelShuffle+ConvNeXt decoder structure as the PC-CCNF, inserting a pairwise offset gate at the bottleneck as the sole fusion mechanism. Everything else (encoder structure, decoder structure, forward model, loss computation) is shared or reused from the PC-CCNF and base PtychoPINN.

### What this model tests

The standard PtychoPINN encodes all C overlapping patches through the same encoder (with `object_big=True`, the input is `(B, C, N, N)` treated as C channels). Overlap consistency is enforced only *after* decoding, via the physics forward model: patches are reassembled, re-extracted, probe-multiplied, and FFT'd. The encoder and decoder never explicitly see overlap geometry.

This baseline asks: does giving the bottleneck a simple, position-conditioned view of neighboring patches' features improve reconstruction quality? The pairwise offset gate provides this view with minimal architectural assumptions -- no spatial canvas, no attention mechanism, no coordinate-conditioned decoder. If this baseline outperforms the standard model, it confirms that latent-space overlap encoding is beneficial. The full PC-CCNF can then be compared against it to quantify the additional value of the canvas, cross-attention, and neural field decoder.

### Comparison to PC-CCNF (`latent_model.py`)

| Component | Overlap Baseline | PC-CCNF |
|-----------|-----------------|---------|
| Encoder | FNOCNNEncoder (interleaved) | FNOCNNEncoder (interleaved) |
| Overlap mechanism | Pairwise offset gate (~272--480 params) | Geometry-tagged cross-attention (~270K params) |
| Feature extraction | None (raw bottleneck) | MultiResolutionFeatureVolume |
| Spatial representation | None (bottleneck features only) | Shared latent canvas with dynamic scaling |
| Decoder | PerPatchPixelShuffleDecoder | CNNCanvasDecoder (grid_sample from canvas) |
| Decoder architecture | stem + PixelShuffle+ConvNeXt + dual heads | stem + PixelShuffle+ConvNeXt + dual heads |
| Skip injection | Direct (per-patch skips at matching res) | Direct (per-patch skips at matching res) |
| Phase consistency | Not guaranteed (enforced via forward model) | Structural guarantee (single-valued canvas function) |
| Total parameters | ~2.1M | ~1.8M--2.8M |

---

## Architecture Diagram

```
Input: (B, C, N, N) diffraction patterns + (B, C, 1, 2) coords_relative

  [1] Weight-Shared FNO+CNN Encoder (per-patch, single-channel, interleaved)
      Reshape: (B, C, N, N) -> (B*C, 1, N, N)
      Lifting conv -> FNO block -> [CNN+pool, BottleneckFNO]xK -> final_pool
      Output: (B*C, D, H, W)   [e.g. D=256, H=W=8 for N=64]
      Produces skip connections: list of (B*C, D_k, H_k, W_k)
      Optional CBAM at bottleneck
      Reshape: (B*C, D, H, W) -> (B, C, D, H, W)
              |
  [2] Pairwise Offset Gate
      Compute pairwise offsets: delta_ij = (coord_i - coord_j) / N
      Encode offsets (Fourier or Hybrid)
      Project to G group gates: sigmoid(proj(enc)) in [0,1]^G
      Broadcast groups to D channels
      Normalized weighted sum: z'_i = sum_j(g_ij * z_j) / sum_j(g_ij)
      Output: (B, C, D, H, W)
              |
  [3] PerPatch PixelShuffle Decoder
      Reshape: (B, C, D, H, W) -> (B*C, D, H, W)
      Stem: Conv2d(D, base_ch, 1) + LayerNorm2d + GELU
      Stages: PixelShuffle(2x) + ConvNeXt refinement + skip injection
        Stage 0 (8->16):  no skip
        Stage 1 (16->32): skip from encoder CNN conv (256 ch)
        Stage 2 (32->64): skip from encoder CNN conv (128 ch)
      Head_real: Conv2d -> 0.2 + tanh(...)   [real component]
      Head_imag: Conv2d -> 1.2 * tanh(...)   [imaginary component]
      Output: (B*C, 1, N, N) each
      Reshape: (B*C, 1, N, N) -> (B, C, N, N) each
      Combine: x_real + j * x_imag -> (B, C, N, N) complex64
              |
  [ForwardModel] (unchanged from standard PtychoPINN)
      Reassemble -> Probe x Object -> FFT -> Predicted Intensity
```

---

## Module Reference

### `PairwiseOffsetGate`

**File**: `ptycho_torch/beta_modules/overlap_baseline.py`, class at line 81

**Purpose**: The core fusion module. Computes a learned, position-dependent mixing weight for every pair of patches in a batch, then applies a gated weighted average across the channel (patch) dimension of the bottleneck features.

**Design principles**:
- **C-agnostic**: The gate function operates on pairs `(i, j)`, not on a fixed number of slots. The same learned weights apply whether C=4 (training) or C=10+ (inference). No slot-specific parameters.
- **Pairwise, not global**: Each pair's gate depends only on their relative offset, not on the full batch geometry. This avoids the positional slot bias of a global MLP (where input/output slot ordering becomes an implicit learned convention).
- **Group gating**: Instead of learning D independent gate values from a 2D input (which would require the gate to represent a complex function on a 2D manifold in $\mathbb{R}^D$), channels are partitioned into G groups. Each group receives one scalar gate. This provides a low-rank constraint that is easier to optimize and sufficient for a baseline.

**Encoding modes**: The gate supports two offset encoding schemes selected by `overlap_encoding`:

#### Encoding mode: `'fourier'` (default)

Uses `FourierPositionalEncoding` with log-linear frequency bands. Simple and lightweight but limited at small offsets and anisotropic scans.

$$\delta_{ij} = \frac{\mathbf{c}_i - \mathbf{c}_j}{N} \in [-1, 1]^2$$

$$\gamma(\delta_{ij}) = \bigl[\sin(2^0 \pi \delta_x), \cos(2^0 \pi \delta_x), \ldots, \sin(2^{L-1} \pi \delta_x), \ldots, \cos(2^{L-1} \pi \delta_y)\bigr] \in \mathbb{R}^{4L}$$

$$g_{ij}^{(\text{group})} = \sigma\!\bigl(W \cdot \gamma(\delta_{ij}) + b\bigr) \in [0, 1]^G$$

**Submodules**: `self.enc` = `FourierPositionalEncoding`, `self.gate_proj` = `nn.Linear(4L, G)`.

**Parameter count**: $4L \times G + G$. For defaults ($L=4$, $G=16$): 272 parameters.

#### Encoding mode: `'hybrid'` (recommended for anisotropic scans)

Uses `OverlapHybridEncoding`: physics-informed overlap fractions combined with targeted high-frequency Fourier features, fed through a 2-layer MLP. All features use $|\delta|$ for structurally symmetric gates ($g_{ij} = g_{ji}$).

$$\delta_{ij} = \frac{\mathbf{c}_i - \mathbf{c}_j}{N}$$

Per-axis overlap features ($a \in \{x, y\}$):

$$\text{ov}_a = \max\!\bigl(0,\; 1 - |\delta_a|\bigr) \qquad \text{ov}_a^2 = \bigl[\max(0, 1 - |\delta_a|)\bigr]^2$$

Per-axis Fourier features with frequencies $f \in \{8, 16\}$:

$$\text{fourier}_a = \bigl[\sin(f \pi |\delta_a|),\; \cos(f \pi |\delta_a|)\bigr]_{f \in \{8, 16\}}$$

Concatenated: $\gamma_\text{hybrid} \in \mathbb{R}^{12}$ (4 overlap + 8 Fourier features).

Gate MLP with cross-axis mixing:

$$h = \text{SiLU}\!\bigl(W_1 \cdot \gamma_\text{hybrid} + b_1\bigr) \in \mathbb{R}^{16}$$

$$g_{ij}^{(\text{group})} = \sigma\!\bigl(W_2 \cdot h + b_2\bigr) \in [0, 1]^G$$

**Submodules**: `self.enc` = `OverlapHybridEncoding`, `self.gate_proj` = `nn.Sequential(Linear(12,16), SiLU, Linear(16,G))`.

**Parameter count**: $(12 \times 16 + 16) + (16 \times 16 + 16) = 480$ parameters.

**Common formulation** (both encoding modes):

$$g_{ij} = \text{broadcast}(g_{ij}^{(\text{group})}) \in [0, 1]^D \quad \text{(each group value repeated } D/G \text{ times)}$$

$$z'_i = \frac{\sum_j g_{ij} \odot z_j}{\sum_j g_{ij}}$$

where $\odot$ is elementwise multiplication broadcast over the spatial dimensions $(H, W)$.

**Parameters**:
- `bottleneck_channels` (int): D, the channel depth of the encoder bottleneck. Determined automatically from `FNOCNNEncoder.filters[-1]`.
- `num_bands` (int): $L$, number of Fourier frequency octaves (only used when `encoding='fourier'`). Default 4.
- `num_groups` (int): $G$, number of channel groups for the gate. Default 16. Clamped to $\min(G, D)$.
- `encoding` (str): `'fourier'` or `'hybrid'`. Default `'fourier'`.

---

### `PerPatchPixelShuffleDecoder`

**File**: `ptycho_torch/beta_modules/overlap_baseline.py`, class at line 155

**Purpose**: Per-patch decoder adapted from `CNNCanvasDecoder` (latent_model.py). Upsamples bottleneck features from H_enc to N resolution via PixelShuffle stages with ConvNeXt refinement and encoder skip injection. No canvas crop step -- takes per-patch features directly.

**Architecture** (N=64, H_enc=8, base_ch=128, nfs=2):

| Stage | Input | Output | Skip |
|-------|-------|--------|------|
| Stem | (B\*C, 256, 8, 8) | (B\*C, 128, 8, 8) | -- |
| Stage 0 | (B\*C, 128, 8, 8) | (B\*C, 128, 16, 16) | none |
| Stage 1 | (B\*C, 128, 16, 16) | (B\*C, 64, 32, 32) | skip[3]: (256 ch, 32x32) |
| Stage 2 | (B\*C, 64, 32, 32) | (B\*C, 32, 64, 64) | skip[1]: (128 ch, 64x64) |
| head_real | (B\*C, 32, 64, 64) | (B\*C, 1, 64, 64) | -- |
| head_imag | (B\*C, 32, 64, 64) | (B\*C, 1, 64, 64) | -- |

Each `PixelShuffleUpsampleStage` consists of:
1. `Conv2d(ch_in, 4*ch_out, 1)` -- channel expansion for PixelShuffle
2. `PixelShuffle(2)` -- 2x spatial upsampling
3. Optional `skip_proj(skip) + x` -- 1x1 conv to match channels, then add
4. `ConvNeXtBlock(ch_out)` -- depthwise conv + inverted bottleneck + layer scale refinement

**Output activations**:
- Real: $0.2 + \tanh(h_{\text{real}}(x))$ -- range $[-0.8, 1.2]$, centered at 0.2
- Imaginary: $1.2 \cdot \tanh(h_{\text{imag}}(x))$ -- range $[-1.2, 1.2]$

---

### `AutoencoderOverlapBaseline`

**File**: `ptycho_torch/beta_modules/overlap_baseline.py`, class at line 241

**Purpose**: Composes the FNO+CNN encoder, pairwise offset gate, and PixelShuffle decoder into a single autoencoder module.

**Encoder**: `FNOCNNEncoder` with `fno_interleave=True` forced regardless of config (ensures interleaved mode). A deep-copied `ModelConfig` with `C_model=1` and `object_big=False` is used. All C patches are flattened into the batch dimension `(B*C, 1, N, N)` before encoding, ensuring weight sharing.

**Decoder construction**: At init time, a dummy forward pass through the encoder determines `H_enc` and builds `encoder_skip_info = [(resolution, channels)]` for each skip. This info is passed to `PerPatchPixelShuffleDecoder` for automatic skip matching by spatial resolution.

**Skip connections**: The encoder produces skip connections from FNO and CNN stages. These skips are from the *unmixed* encoding pass. They are passed to the decoder alongside the *mixed* bottleneck features. This means the decoder receives mixed global information through the bottleneck and local per-patch detail through the skips.

**Forward signature**:
```python
def forward(self, x: torch.Tensor, coords: torch.Tensor,
            probe: Optional[torch.Tensor] = None) -> tuple:
    # x: (B, C, N, N) diffraction patterns
    # coords: (B, C, 1, 2) relative coordinates in pixel units
    # probe: unused (interface compatibility with AutoencoderCCNF)
    # Returns: (complex_out, x_real, x_imag), all (B, C, N, N)
```

---

### `PtychoPINN_OverlapBaseline`

**File**: `ptycho_torch/beta_modules/overlap_baseline.py`, class at line 329

**Purpose**: Top-level model module wrapping the autoencoder with the intensity scaler and physics forward model. Provides the same interface as `PtychoPINN` (from `model.py`) and `PtychoPINN_CCNF` (from `latent_model.py`).

**Forward pass**:
1. Scale input diffraction patterns by `input_scale_factor` via `IntensityScalerModule`.
2. Run `AutoencoderOverlapBaseline(x, positions, probe)` to get complex object patches.
3. Pass patches through `ForwardModel.forward()` -- reassemble, probe illumination, FFT, scaling.
4. Return `(predicted_intensities, x_real, x_imag)`.

**Interface methods** (matching `PtychoPINN`):
- `forward(x, positions, probe, input_scale_factor, output_scale_factor, ...)` -- full forward with physics model
- `forward_predict(x, positions, probe, input_scale_factor)` -- decode only, returns complex patches
- `get_encoder_bottom_params()` / `get_encoder_top_params()` -- for staged fine-tuning with discriminative LR
- `freeze_encoder()` -- freezes all encoder parameters for decoder-only fine-tuning

---

## Configuration

All parameters are in `ModelConfig` in `ptycho_torch/config_params.py`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `architecture` | `Literal[..., 'overlap_baseline']` | `'unet'` | Set to `'overlap_baseline'` to select this model. Frozen. |
| `encoder_type` | `Literal['cnn', 'fno_cnn']` | `'cnn'` | Ignored by overlap baseline (always uses FNO+CNN). Frozen. |
| `fno_interleave` | `bool` | `False` | Forced to `True` by overlap baseline. Frozen. |
| `fno_modes` | `int` | `16` | Number of Fourier modes for FNO blocks. Frozen. |
| `fno_width` | `int` | `32` | Channel width in FNO blocks. Frozen. |
| `fno_blocks` | `int` | `2` | Number of FNO blocks. Frozen. |
| `cnn_decoder_base_ch` | `int` | `128` | Base channel count for PixelShuffle decoder. |
| `overlap_fourier_bands` | `int` | `4` | Number of Fourier frequency octaves for pairwise offsets. Only used when `overlap_encoding='fourier'`. |
| `overlap_gate_groups` | `int` | `16` | Number of channel groups for the gate. Each group gate controls D/G channels. |
| `overlap_encoding` | `Literal['fourier', 'hybrid']` | `'fourier'` | Offset encoding scheme. |

All other `ModelConfig` fields that affect the encoder (`n_filters_scale`, `cbam_encoder`, `cbam_bottleneck`, `batch_norm`, etc.) apply unchanged.

**Example config JSON**:
```json
{
    "architecture": "overlap_baseline",
    "encoder_type": "fno_cnn",
    "fno_interleave": true,
    "fno_modes": 16,
    "fno_width": 32,
    "fno_blocks": 2,
    "cnn_decoder_base_ch": 128,
    "overlap_encoding": "fourier",
    "overlap_gate_groups": 16
}
```

---

## Integration

### Architecture selection

`PtychoPINN_Lightning.__init__` in `ptycho_torch/model.py`:

```python
elif arch == 'overlap_baseline':
    from ptycho_torch.beta_modules.overlap_baseline import PtychoPINN_OverlapBaseline
    self.model = PtychoPINN_OverlapBaseline(model_config, data_config, training_config)
```

### Forward model

`ForwardModel.forward()` is completely unchanged. It receives `(B, C, N, N)` complex64 patches from the autoencoder and applies reassembly, probe illumination, and FFT.

### Loss computation

`PtychoPINN_Lightning.compute_loss()` requires no changes. It calls `self.model()` expecting `(pred, x_real, x_imag)`, which `PtychoPINN_OverlapBaseline.forward()` provides.

### Inference

`forward_predict()` returns `(B, C, N, N)` complex64, compatible with all existing reconstruction functions.

---

## Dependencies

```
overlap_baseline.py
  imports from:
    ptycho_torch/config_params.py          -> ModelConfig, DataConfig, TrainingConfig
    ptycho_torch/model.py                  -> ForwardModel, IntensityScalerModule,
                                              CombineComplexRectangular
    ptycho_torch/model_attention.py        -> CBAM
    ptycho_torch/beta_modules/latent_model.py -> FourierPositionalEncoding,
                                                 FNOCNNEncoder,
                                                 LayerNorm2d,
                                                 PixelShuffleUpsampleStage
```

---

## Computational Profile

### Parameter count (~2.1M total with `n_filters_scale=2`, N=64)

- FNOCNNEncoder (interleaved): ~1.75M (lifting + FNO blocks + CNN blocks + spectral weights)
- Pairwise offset gate: **272** (`encoding='fourier'`) or **480** (`encoding='hybrid'`)
- PerPatchPixelShuffleDecoder: ~347K (stem + 3 PixelShuffle stages + ConvNeXt blocks + dual heads)
- Forward model (scaler, etc.): ~100K

The overlap gate adds negligible parameters in either mode.

### Forward pass cost

The pairwise offset gate computes a `(B, C, C)` interaction matrix, which for C=8 is $8 \times 8 = 64$ pairs. The Fourier encoding and linear projection operate on these 64 pairs. The einsum for the weighted sum is `O(B * C^2 * D * H * W)`, which for typical values (B=16, C=8, D=256, H=W=8) is ~33M multiply-adds -- negligible compared to the FNO spectral convolutions and decoder upsampling.

### Memory

No additional feature volumes or canvases. The only new tensor is the gate matrix `(B, C, C, D)`, which for B=16, C=8, D=256 is 512KB in float32.

---

## File Structure

```
ptycho_torch/
    config_params.py                          # architecture='overlap_baseline' + config fields
    model.py                                  # PtychoPINN_Lightning: lazy import for model selection
    beta_modules/
        overlap_baseline.py                   # All overlap baseline modules (this file)
        latent_model.py                       # FNOCNNEncoder, PixelShuffleUpsampleStage (imported)
        Baseline_latent_model.md              # This documentation
```
