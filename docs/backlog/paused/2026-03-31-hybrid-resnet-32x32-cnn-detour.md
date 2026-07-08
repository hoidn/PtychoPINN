# Backlog: Hybrid ResNet `32x32` CNN Detour Branch

## Summary
Add an optional pure-CNN detour branch around the existing `64x64` Hybrid ResNet bottleneck for `N=256` studies.

The main spectral trunk stays unchanged:
- encoder with the current `2` downsampling stages
- adapter
- existing `ResnetBottleneck`
- decoder and skip fusion

The new branch is auxiliary:
- branch from the post-adapter `64x64` bottleneck input
- downsample to `32x32` using a CNN downsampler
- process at `32x32` with pure CNN residual blocks only
- upsample back to `64x64`
- fuse back into the main trunk with a learned gated residual add

The goal is to add cheap coarse real-space context without forcing the whole model through a third global downsampling stage.

## Why This May Help
- The canonical `lines_256` contract keeps the informative probe support relatively small inside the `256x256` field; the current `64x64` bottleneck may preserve detail but still under-model coarse real-space consensus.
- Phase/boundary quality at `256` appears to suffer from patch-level consistency drift rather than a simple lack of local detail.
- A `32x32` pure-CNN detour gives the model a larger receptive-field path without further constraining or blunting the main spectral path.

## Why Not A Full Third Downsampling Stage
- Forcing the whole architecture to `32x32` would coarsen the winning spectral trunk and likely hurt detail preservation.
- The current `hybrid_resnet` topology and `lines_256` winner are already centered around the `64x64` bottleneck regime.
- A full third downsample would introduce a larger topology change, more confounds, and a higher risk of degrading the accepted behavior.

## Recommended Design

### Main Principle
Treat the `32x32` path as an optional auxiliary context branch, not as the new core bottleneck.

### Branch Placement
Tap the branch from the same `64x64` tensor that currently feeds `self.resnet`:
- after the encoder
- after `self.adapter`
- before the main `ResnetBottleneck`

This preserves the current main bottleneck input exactly.

### Branch Structure
First implementation:
- one CNN downsample from `64x64 -> 32x32`
- one or two pure CNN residual blocks at `32x32`
- one CNN upsample from `32x32 -> 64x64`

Branch body requirements:
- no spectral/FNO operators
- no new skip topology
- no decoder rewiring
- same channel width as the bottleneck path for the first experiment

### Fusion
Fuse the detour back after the main bottleneck:

`x = main_bottleneck(x) + gate * detour(x)`

Recommended details:
- learned scalar gate
- small positive init such as `0.1`
- additive fusion only for v1

This keeps the current architecture as the dominant path at initialization and lets training scale detour influence up or down.

## Scope

### In Scope
- `hybrid_resnet` module support for an optional auxiliary `32x32` CNN detour
- model-config plumbing sufficient to toggle the branch for experiments
- forward/shape/metadata tests for the branch-enabled path
- one targeted `lines_256` ablation to determine whether the branch is useful

### Out of Scope
- making `3` full downsampling stages the new default
- adding spectral ops to the detour branch
- adding multiple cross-scale exchange points
- redesigning the decoder or skip hierarchy
- promoting the branch to a stable public baseline before evidence exists

## Suggested Config Surface
Use narrow, explicitly experimental knobs first:
- `hybrid_cnn_detour_enabled: bool = False`
- `hybrid_cnn_detour_blocks: int = 1` or `2`
- `hybrid_cnn_detour_gate_init: float = 0.1`

Optional later:
- `hybrid_cnn_detour_downsample_op`
- `hybrid_cnn_detour_width`

Do not widen the config surface further unless the branch demonstrates benefit.

## Rollout Plan

### Phase 1: Experimental Support
- Implement the branch behind an opt-in config flag.
- Keep default behavior bit-for-bit aligned with the current architecture when disabled.
- Add targeted tests covering:
  - branch disabled parity
  - branch enabled forward-shape invariance
  - gate creation and initialization

### Phase 2: Focused `lines_256` Trial
- Evaluate the branch only against the current accepted `lines_256` champion regime.
- Compare against the accepted `32/32/5`, gated-skip, stride-conv, `10`-block topology.
- Keep dataset and epoch contract fixed so the ablation is attributable.

### Phase 3: Promotion Decision
Only if it shows meaningful value:
- document it as a supported knob
- widen runbook/workflow exposure
- consider whether to sweep branch depth or branch width

If it does not help:
- keep it out of the stable surface or remove it entirely

## Risks
- The branch may simply add complexity and cost without fixing the real consistency failure mode.
- A poorly initialized or overpowered fusion path could destabilize the current winning bottleneck.
- The branch may improve coarse coherence while softening fine detail, producing no net `amp_ssim` gain.
- If exposed too broadly too early, it could clutter the architecture surface without proof of value.

## Acceptance Criteria
- With the branch disabled, current `hybrid_resnet` tests and behavior remain unchanged.
- With the branch enabled, forward pass preserves output shape and existing skip/decoder invariants.
- The branch can be toggled from config without invasive wrapper changes.
- A controlled `lines_256` ablation determines whether the branch is worth keeping.

## Suggested Next Steps
- Write an implementation plan for the minimal branch-enabled experiment path.
- Implement the branch with TDD and targeted forward/shape regressions.
- Run one controlled `lines_256` ablation before deciding whether this should become a permanent supported knob.
