# Latent Space Expansion — FNO-CNN Encoder Bottleneck Scaling

**Date:** 2026-07-02
**Branch:** `latent_experimental`
**Author:** Albert Vong (with Claude Code)
**Status:** Experimental — revertible. This document records the exact edits so
they can be undone cleanly if the ablation results are not favorable.

---

## Motivation

The original `FNOCNNEncoder` pools every input size down to a fixed **8×8**
bottleneck. For `N >= 128` this is likely a genuine information bottleneck:

- **Capacity is N-invariant while object DOF grows as N².** The per-patch latent
  is always `8 × 8 × (n_filters_scale·128)`; the object patch it must represent is
  `N × N` complex. Compression ratio worsens quadratically (≈0.5× at N=64,
  2:1 at N=128, 8:1 at N=256).
- **FNO mode collapse.** A spectral conv on an 8×8 grid can retain at most
  `8//2 = 4` modes (`rfft` gives `H/2+1` bins). Diffraction data is broadband
  (speckle carries phase), so 4 modes is far too few. FNO layers deep in the
  network stop behaving like operators.

**Fix (compound scaling):** hold the downsampling *depth* fixed (3 pools) rather
than the latent *size*, so the latent grows with N (8/16/32 for N=64/128/256),
and add a few **residual** FNO blocks at the enlarged bottleneck where the mode
budget is now meaningful.

Grounding: vanilla FNO keeps resolution fixed (Li et al., ICLR 2021; Kovachki
et al., JMLR 2023); deep FNO needs residual/factorized blocks (FFNO, Tran et al.,
ICLR 2023); U-NO downsamples but grows co-domain to preserve capacity (Rahman
et al., TMLR 2023); balanced resolution/depth/width scaling (EfficientNet,
Tan & Le, ICML 2019).

---

## Design summary

- **Backward compatible by default.** New config fields default to the original
  behavior; existing checkpoints/configs are unaffected (fields read via
  `getattr` for legacy-config safety).
- **`H_enc` is auto-detected downstream.** `AutoencoderCCNF` already infers
  `H_enc` from a dummy forward pass and propagates it to `cnn_decoder_crop_size`,
  so **no decoder/fusion edits are required**.
- **Added depth is zero-init residual** (`BottleneckFNOBlock.proj_up` is
  zero-initialized), so the encoder output is bit-identical to the
  no-bottleneck case at initialization.

---

## Exact edits

### 1. `ptycho_torch/config_params.py` — new `ModelConfig` fields (after `fno_interleave`, ~lines 103-110)

Added:

```python
    # FNO-CNN latent bottleneck scaling
    # Target spatial size H_enc of the encoder bottleneck. Default 8 reproduces
    # the original fixed 8x8 latent for all N. Set <=0 to auto-scale as N//8
    # (-> 8/16/32 for N=64/128/256), holding downsampling depth fixed at 3.
    encoder_latent_size: int = field(default=8, metadata={'frozen': True})
    # Residual bottleneck FNO blocks applied at H_enc (adds depth, no pooling).
    # 0 = disabled (original behavior). A few (2-3) is the intended range.
    fno_bottleneck_blocks: int = field(default=0, metadata={'frozen': True})
```

**Revert:** delete these two fields and their comment block.

### 2. `ptycho_torch/beta_modules/latent_model.py` — `FNOCNNEncoder.__init__`, pool-count derivation (~lines 655-671)

**Before:**

```python
        if N < 16 or (N & (N - 1)) != 0:
            raise ValueError(f"N must be a power of 2 >= 16, got N={N}")
        n_cnn_pools = int(math.log2(N / 8))
        n_cnn_blocks = n_cnn_pools - 1
```

**After:**

```python
        if N < 16 or (N & (N - 1)) != 0:
            raise ValueError(f"N must be a power of 2 >= 16, got N={N}")

        # Resolve target bottleneck size H_enc (<=0 => auto-scale as N//8)
        H_enc = getattr(model_config, 'encoder_latent_size', 8)
        if H_enc <= 0:
            H_enc = N // 8
        if (H_enc & (H_enc - 1)) != 0 or N % H_enc != 0:
            raise ValueError(
                f"encoder_latent_size={H_enc} must be a power of 2 dividing N={N}"
            )
        n_pools_total = int(math.log2(N / H_enc))
        n_cnn_blocks = n_pools_total - 1          # last reduction is final_pool
        if n_cnn_blocks < 1:
            raise ValueError(
                f"encoder_latent_size={H_enc} too large for N={N}: need "
                f"H_enc <= N//4 so channels can ramp to n_filters_scale*128"
            )
```

Note: `H_enc=8` reproduces `n_cnn_blocks` exactly as the old `log2(N/8) - 1`.
The variable `n_cnn_pools` was removed (only used to derive `n_cnn_blocks`).

**Revert:** restore the "Before" block.

### 3. `ptycho_torch/beta_modules/latent_model.py` — `FNOCNNEncoder.__init__`, bottleneck block construction (inserted before `def forward`, ~lines 741-753)

Inserted after the `if self._interleave: ... else: ...` block that sets
`self.blocks` / `self.filters`:

```python
        # Residual bottleneck FNO blocks at H_enc resolution: extra depth with
        # no further downsampling. Modes scale with the bottleneck grid.
        self.bottleneck_fno_blocks = nn.ModuleList()
        n_bottleneck = getattr(model_config, 'fno_bottleneck_blocks', 0)
        if n_bottleneck > 0:
            bottleneck_ch = cnn_filter_list[-1]          # n_filters_scale * 128
            bottleneck_modes = min(fno_modes, H_enc // 2)
            for _ in range(n_bottleneck):
                self.bottleneck_fno_blocks.append(
                    BottleneckFNOBlock(bottleneck_ch, fno_width, bottleneck_modes)
                )
            # expose to get_encoder_*_params / fine-tuning splitters
            self.blocks = nn.ModuleList(list(self.blocks) + list(self.bottleneck_fno_blocks))
            self.filters = self.filters + [bottleneck_ch] * n_bottleneck
```

**Revert:** delete this block entirely. (`self.bottleneck_fno_blocks` will no
longer exist; edit 4 must be reverted too.)

### 4. `ptycho_torch/beta_modules/latent_model.py` — `FNOCNNEncoder.forward`, apply bottleneck blocks (~lines 788-792)

**Before:**

```python
        x = self.final_pool(x)

        return x, skips
```

**After:**

```python
        x = self.final_pool(x)

        for fno_block in self.bottleneck_fno_blocks:
            x = fno_block(x)

        return x, skips
```

**Revert:** remove the `for fno_block ...` loop.

---

## How to enable (experimental configs)

In a `ccnf_configs/*.json` (or equivalent `ModelConfig` overrides):

```json
{ "encoder_type": "fno_cnn", "encoder_latent_size": -1, "fno_bottleneck_blocks": 2 }
```

- `encoder_latent_size: -1` → auto `H_enc = N//8` (8/16/32 for N=64/128/256).
  (Can also set an explicit power-of-2 int, e.g. `16`.)
- `fno_bottleneck_blocks: 2` → 2 residual FNO blocks at the bottleneck.

Defaults (`encoder_latent_size: 8`, `fno_bottleneck_blocks: 0`) = original model.

---

## Shape trace (N=128, B=16, C=8; effective batch B·C=128; n_filters_scale=2)

Encoder input after patch-flatten: `(128, 1, 128, 128)`.

| | OLD (H_enc=8, bott=0) | NEW (H_enc=16, bott=2) |
|---|---|---|
| downsampling pools | 4 | 3 |
| cnn conv blocks | 3 (ramp 32→64→128→256) | 2 (ramp 32→128→256) |
| bottleneck output | `(128, 256, 8, 8)` | `(128, 256, 16, 16)` |
| bottleneck FNO modes | n/a | 8 |
| encoder skips | 5 | 4 |
| latent capacity / patch | 16,384 | 65,536 (4×) |

Each `BottleneckFNOBlock` at 16×16: `256 →(1×1)→ 32 →FNOBlock(m=8)→ 32 →(1×1, zero-init)→ 256`, residual; spatial size unchanged.

---

## Verification performed (2026-07-02)

Ran with `/local/miniconda3/envs/PtychoPINN_torch/bin/python`:

1. **Backward compat:** `H_enc=8, bott=0`, N=128 → `(128, 256, 8, 8)`, 5 skips (unchanged). ✅
2. **Scaled:** `encoder_latent_size=-1, fno_bottleneck_blocks=2`, N=128 → `(128, 256, 16, 16)`, 4 skips, 2 bottleneck blocks. ✅
3. **Identity-at-init:** encoder with vs without 2 bottleneck blocks (shared weights) → `max|diff| = 0.0`. ✅
4. **N=256:** auto → `(·, 256, 32, 32)`, modes=16. ✅
5. **Guard:** `encoder_latent_size=32, N=64` raises `ValueError`. ✅
6. **End-to-end:** `AutoencoderCCNF` (N=128, auto latent, 2 bottleneck blocks) → output `(2, 8, 128, 128)` complex64; `H_enc=16` auto-detected; `NeuralFieldDecoder` wiring intact. ✅

---

## Reversion checklist

To fully revert (in any order that leaves the file syntactically valid; do 4→3
before or together with the others to avoid a dangling attribute):

- [ ] `config_params.py`: remove `encoder_latent_size` and `fno_bottleneck_blocks` (edit 1).
- [ ] `latent_model.py`: restore original `n_cnn_pools`/`n_cnn_blocks` lines (edit 2).
- [ ] `latent_model.py`: delete the bottleneck-block construction block (edit 3).
- [ ] `latent_model.py`: remove the bottleneck loop in `forward` (edit 4).

Or simply revert the single commit that introduced this document and these edits.
