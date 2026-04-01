# Workflow Idea: Hybrid ResNet `32x32` CNN Detour Branch

## Hypothesis
- Add an optional pure-CNN detour branch around the existing `64x64` Hybrid ResNet bottleneck for `N=256` studies.
- The detour should add cheap coarse real-space context without forcing the whole model through a third global downsampling stage.

## Why Now
- The current `lines_256` winner appears to preserve detail reasonably well, but may still under-model coarse real-space consensus at `256`.
- A `32x32` pure-CNN detour is a broader architectural idea that is distinct from the local capacity and scheduler tweaks the workflow has already explored.

## Suggested Knobs
- branch from the post-adapter `64x64` bottleneck input
- one CNN downsample `64x64 -> 32x32`
- one or two pure CNN residual blocks at `32x32`
- one CNN upsample `32x32 -> 64x64`
- fuse back into the main bottleneck output with a learned gated residual add
- narrow experimental config surface first:
  - `hybrid_cnn_detour_enabled`
  - `hybrid_cnn_detour_blocks`
  - `hybrid_cnn_detour_gate_init`

## Constraints
- keep the main spectral trunk unchanged
- do not turn this into a full third global downsampling stage
- keep the first experiment narrow and attributable
- if the branch does not help, prefer not expanding the stable architecture surface
