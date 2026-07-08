# Hybrid ResNet‑6 + FNO Vanilla Generators (PyTorch) — Design

**Date:** 2026-01-30  
**Target:** `ptycho_torch/` generator registry and grid-lines workflows  
**Scope:** New generator architectures (`fno_vanilla`, `hybrid_resnet`) + docs/tests wiring  

## 1. Problem Statement
The current Hybrid U‑NO decoder uses non‑overlapping transposed convolutions (kernel=2, stride=2), which can introduce blocky checkerboard artifacts. We also lack a pure spectral baseline to measure whether the multiscale Hybrid path is helping or hurting stability.

## 2. Goals
- Add **`fno_vanilla`**: constant‑resolution spectral baseline (no down/upsampling) to isolate multiscale effects.
- Add **`hybrid_resnet`**: FNO encoder + **CycleGAN‑style ResNet‑6 bottleneck + CycleGAN upsamplers** to reduce artifacts while preserving coarse‑to‑fine synthesis.
- Keep physics pipeline and output contracts unchanged for `PtychoPINN_Lightning`.

## 3. Non‑Goals
- No changes to TensorFlow backend or core physics modules.
- No changes to loss formulation or data contracts.
- No changes to legacy `ptycho/` model internals.

## 4. Architecture Summary
### 4.1 `fno_vanilla`
- **Lifter:** `SpatialLifter` (2× 3×3 conv + GELU)
- **Body:** stack of `PtychoBlock` (or stable variant if needed), constant resolution
- **Output:** 1×1 projection to `real_imag` or `amp_phase` (same contract as existing generators)

### 4.2 `hybrid_resnet`
- **Encoder:** FNO blocks + downsampling (N → N/4)
- **Bottleneck:** 6× CycleGAN ResNet blocks at constant low resolution
- **Decoder:** CycleGAN upsampling (ConvTranspose2d kernel=3, stride=2, padding=1, output_padding=1 + InstanceNorm + ReLU) to N
- **Output:** 1×1 projection to `real_imag` or `amp_phase`

**Key correction:** This replaces the **entire** old Hybrid bottleneck + decoder with the CycleGAN backend (ResNet‑6 + CycleGAN upsamplers). No Hybrid U‑NO upsamplers remain.

## 5. Output Modes
- **Default:** `real_imag` (no sigmoid/tanh), matching `TorchRunnerConfig.generator_output_mode = "real_imag"`.
- **Optional:** `amp_phase` uses the **same scaling** as `HybridUNOGenerator`:
  - `amp = sigmoid(Conv1x1)`
  - `phase = π * tanh(Conv1x1)`

## 6. Channel Bridging
The CycleGAN ResNet blocks expect a fixed channel width (often 256). The FNO encoder width may vary with `fno_width`, `fno_blocks`, or `max_hidden_channels`. Add a **1×1 adapter conv** between the FNO encoder output and the ResNet bottleneck **only when widths differ**; otherwise use an identity.

## 7. Data Flow (Shapes)
- Input: `X` in `B×C×N×N`, `C = gridsize²`
- Encoder: downsample to N/4 (two stride‑2 steps)
- Bottleneck: ResNet‑6 at N/4
- Decoder: N/4 → N/2 → N via CycleGAN upsamplers
- Output: `real_imag` tensor `(B, H, W, C, 2)` or `amp_phase` tensors `(B, C, H, W)`

## 8. Integration Points
- Generator registry (`ptycho_torch/generators/registry.py`)
- Torch runner + compare wrapper (grid‑lines workflows)
- Config literals + validation (`ptycho/config/config.py`, `ptycho_torch/config_params.py`, bridge spec)
- Docs: `docs/CONFIGURATION.md`, `docs/architecture_torch.md`, `docs/workflows/pytorch.md`, `ptycho_torch/generators/README.md`

## 9. Validation & Testing
- Unit tests: registry resolution, forward shape sanity, `amp_phase` bounds.
- Workflow proof: single‑epoch run via `scripts/studies/grid_lines_compare_wrapper.py` for `fno_vanilla` and `hybrid_resnet`.

## 10. Risks & Mitigations
- **Width mismatch**: 1×1 adapter prevents runtime crashes.
- **Output mode misuse**: tests enforce correct scaling in `amp_phase`.
- **Artifact regression**: CycleGAN upsampler overlap reduces checkerboarding vs old 2×2 transposed convs.

