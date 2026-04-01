# Workflow Idea: Hybrid ResNet Single-Scale Skip Routing

## Hypothesis
- The current champion always fuses both encoder skip taps once skip connections are enabled, but one skip scale may be helping while the other washes out amplitude structure.
- Routing only the more useful skip stage could keep the structural benefit of skips while reducing harmful feature injection.

## Why Now
- `gated_add` clearly helped relative to plain `add`, but that still assumes both skip stages should always be present.
- The failed coarse detour and other local-capacity tweaks point toward information-routing mistakes rather than missing CNN depth.

## Suggested Knobs
- add a narrow skip-selection control such as `hybrid_skip_keys`
- first experiment should stay minimal and attributable:
  - champion config unchanged
  - keep `gated_add`
  - try only the higher-resolution skip first, then the lower-resolution skip only if needed

## Constraints
- do not add new skip branches
- do not change skip fusion style in the first experiment
- keep this as a routing ablation, not a broader architecture rewrite
