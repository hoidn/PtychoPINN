# Workflow Idea: Hybrid ResNet Conditioned Skip Gating

## Hypothesis
- The current `gated_add` skip fusion uses one learned scalar per skip stage, which is likely too coarse for `N=256`.
- A tiny conditioned gate that predicts channel-wise skip strength from the decoder/skip features could preserve helpful edge/detail channels while suppressing skip channels that blur amplitude structure.

## Why Now
- Skip routing is clearly important in `lines_256`: `gated_add` beat plain `add`, while disabling skips hurt badly.
- The failed `32x32` CNN detour suggests the missing piece is not more coarse CNN depth, but better control over which existing features are allowed through.

## Suggested Knobs
- add a narrow skip-gate mode such as `hybrid_skip_gate_mode=channel_sigmoid`
- first experiment should stay small:
  - 1x1 gate head
  - sigmoid output
  - zero-init or near-neutral init so it starts close to current `gated_add`
- keep the main skip projection path intact and only change the gating mechanism

## Constraints
- do not add new detour branches or extra downsampling
- keep the first experiment attributable and close to the current champion
- prefer a single new gate mode over a large skip-fusion redesign
