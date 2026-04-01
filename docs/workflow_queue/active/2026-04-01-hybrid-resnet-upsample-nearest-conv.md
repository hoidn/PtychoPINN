# Workflow Idea: Hybrid ResNet Nearest-Conv Upsampling

## Hypothesis
- The current decoder upsamples with transposed convolutions, which may be injecting mild checkerboard or structural artifacts that hurt amplitude SSIM.
- Replacing the decoder upsampling path with a narrow `nearest -> 3x3 conv -> norm -> GELU` block could preserve the accepted model's overall capacity while improving structural fidelity in the final reconstruction.

## Why Now
- The recent `lines_256` evidence points more strongly at routing/fusion issues than at missing coarse context, but decoder upsampling is still an untested architecture lever.
- This is a clean apples-to-apples ablation against the current fixed `CycleGanUpsampler` path without changing modes, width, depth, skips, or training budget.

## Suggested Knobs
- add an explicit upsampling-method knob such as `hybrid_upsample_op`
- first experiment should stay narrow:
  - baseline/current `convtranspose`
  - challenger `nearest_conv`
- keep the accepted `32/32/5`, gated-skip, stride-conv, `10`-block topology unchanged otherwise

## Constraints
- do not bundle bilinear or pixelshuffle into the first experiment
- do not combine this with skip-routing or scheduler changes
- keep the first implementation as a small architecture ablation, not a decoder redesign
