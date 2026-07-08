# Backlog: Expose Probe Masking in Torch Training/Inference Interfaces

**Created:** 2026-02-18  
**Status:** Open  
**Priority:** Medium  
**Related:** `ptycho_torch/config_params.py`, `ptycho_torch/model.py`, `ptycho_torch/train.py`, `ptycho_torch/config_factory.py`, `ptycho_torch/workflows/components.py`

## Problem
Torch already supports probe masking internally via `ModelConfig.probe_mask`, but this is not exposed through the primary Torch-facing CLI/factory workflow.

- Mask support exists in config/model:
  - `ptycho_torch/config_params.py` (`probe_mask: Optional[TensorType] = None`)
  - `ptycho_torch/model.py` (`ProbeIllumination` applies provided mask, otherwise uses all-ones)
- Main Torch entry path does not currently thread a probe-mask option:
  - `ptycho_torch/train.py` builds `overrides` without a probe-mask field
  - `ptycho_torch/config_factory.py` creates `PTModelConfig` without probe-mask wiring

## Goal
Add a supported, user-facing way to enable probe masking in Torch workflows, with behavior that is explicit and reproducible.

## Proposed Implementation
1. Add Torch CLI flags for mask control (minimum: on/off; optionally diameter/custom path).
2. Thread new values through `create_training_payload(..., overrides=...)`.
3. Populate `PTModelConfig.probe_mask` from overrides in `ptycho_torch/config_factory.py`.
4. Ensure workflow entry points that build Torch configs (`ptycho_torch/train.py`, and Torch workflow adapters) preserve the option.
5. Document the option in Torch workflow docs and command references.

## Acceptance Criteria
1. Users can enable probe masking from Torch CLI without code edits.
2. Training payload contains non-`None` `pt_model_config.probe_mask` when enabled.
3. `ProbeIllumination` receives and applies the mask during forward pass.
4. Default behavior remains unchanged (masking off unless explicitly enabled).
5. Docs include at least one concrete command example.

## Suggested Tests
1. CLI/factory test: flag/override sets `pt_model_config.probe_mask`.
2. Model test: masked vs unmasked `ProbeIllumination` outputs differ as expected on synthetic input.
3. Regression test: default Torch training path remains unmasked and unchanged.
