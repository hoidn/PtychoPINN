# Workflow Idea: Hybrid ResNet Encoder Branch Routing

## Hypothesis
- Inside each encoder block, the spectral and local branches are always summed with equal weight before activation.
- A lightweight routing mechanism that learns how much spectral vs. local contribution to use per block or per stage could improve amplitude SSIM by shifting capacity without globally changing modes, width, or depth.

## Why Now
- Global branch-capacity sweeps have not helped cleanly: stronger spectral scaling regressed, and simple local-branch widening did not beat the champion.
- That pattern suggests the problem may be branch mixing rather than raw branch size.

## Suggested Knobs
- add a default-off routing option such as `hybrid_encoder_branch_gating`
- keep the first version narrow:
  - scalar gates for spectral and local terms
  - initialized near neutral (`0.5/0.5` or equivalent equal routing)
  - stage-shared or block-local, whichever is simpler to attribute
- first scored run should keep the accepted `32/32/5`, gated-skip, stride-conv, `10`-block topology unchanged apart from the routing change

## Constraints
- do not widen branches or add extra blocks in the first experiment
- preserve the existing spectral trunk and local-conv branch structure
- avoid coupling this with scheduler or loss changes
