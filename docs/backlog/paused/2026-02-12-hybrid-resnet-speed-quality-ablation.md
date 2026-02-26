# Backlog: Hybrid ResNet Speed/Quality Ablation

**Created:** 2026-02-12  
**Status:** Open  
**Priority:** Medium  
**Related:** `ptycho_torch/generators/hybrid_resnet.py`, `ptycho_torch/generators/fno.py`, `scripts/studies/grid_lines_compare_wrapper.py`, `scripts/studies/grid_lines_torch_runner.py`, `docs/performance/2026-02-12-session-findings-hybrid-resnet-sharp-lsqml.md`  
**Impacts:** Grid-lines Torch studies using `pinn_hybrid_resnet` at `N=64` and `N=128`

## Problem

Current default Hybrid ResNet appears conv-heavy in runtime composition, and may be larger than needed for the quality level required in grid-lines studies.

## Goal

Determine whether the **CNN/ResNet portions** of Hybrid ResNet can be made materially faster/lighter while preserving reconstruction quality.

## Proposed Ablation

Run controlled ablations against the current default (`fno_width=32`, `fno_blocks=4`, default ResNet bottleneck), with priority on conv-path reductions:

1. Reduce ResNet bottleneck width via `--torch-resnet-width` (for example: `96`, `64`).
2. Cap encoder channel growth via `--torch-max-hidden-channels` (for example: `64`).
3. Add and ablate a `resnet_blocks` knob (for example: `6 -> 4 -> 2`) to directly reduce bottleneck conv depth.
4. Prioritize reducing CNN feature-map depth in the encoder local-conv path (within `PtychoBlock`) while keeping spectral branch settings fixed:
   - introduce a local-branch width multiplier (for example: `1.0 -> 0.5 -> 0.25`),
   - and/or project to reduced local channels before the local conv and re-project back.
5. Secondary encoder-local ablations (optional after width ablation):
   - `3x3 -> 1x1`,
   - depthwise-separable local conv.
6. Keep FNO spectral settings fixed during the primary ablation to isolate CNN/ResNet effects.
7. Optionally run a secondary FNO ablation only after conv-focused results are established.

Keep dataset, seed, epochs, optimizer/scheduler, and eval protocol fixed per comparison.

## Measurements

For each variant, collect:

1. Quality: MAE, MSE, PSNR, SSIM, FRC50 (amp/phase).
2. Speed: training wall time/epoch, inference throughput, forward latency.
3. Capacity: trainable parameter count.

## Acceptance Criteria

1. At least one reduced-conv-capacity variant shows a clear runtime improvement (`>=20%` inference speedup or equivalent training-time reduction).
2. Quality regression is small relative to default (target: within `~2-5%` on primary metrics, with phase metrics explicitly checked).
3. Results are documented in a study artifact directory and linked from `docs/studies/index.md`.

## Notes

- Prior performance profiling indicates Hybrid ResNet runtime is dominated by convolution-heavy stages; this ablation is meant to verify whether that capacity is necessary for quality in this workload.
