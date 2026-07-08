# Backlog: Canonical Ownership for Torch-Only Model Knobs

**Created:** 2026-02-24
**Status:** Open
**Priority:** High
**Related:** `ptycho/config/config.py`, `ptycho_torch/config_params.py`, `ptycho_torch/config_bridge.py`, `ptycho/config/config.py::update_legacy_dict`, `docs/specs/spec-ptycho-config-bridge.md`
**Impacts:** `tests/torch/test_config_bridge.py`, `tests/torch/test_grid_lines_torch_runner.py`, `tests/test_model_config_architecture.py`, study runbooks under `scripts/studies/runbooks/`

## Summary
The codebase needs an explicit ownership policy for Torch-only architecture knobs so we can add search controls without repeatedly expanding cross-backend coupling and `params.cfg` bridge surface area.

Current and incoming knobs in scope include:
- `hybrid_skip_connections`
- `hybrid_downsample_steps`
- `hybrid_downsample_op`
- `hybrid_resnet_blocks`
- `hybrid_skip_style`

## Why This Matters
- The canonical config bridge currently defines TensorFlow dataclasses as source-of-truth across backends.
- `update_legacy_dict` writes dataclass fields into `params.cfg`, so adding shared dataclass fields can implicitly grow legacy state.
- We already carry migration debt around `params.cfg`; new Torch-only knobs should not silently increase that debt.

## Relationship to Active Plan
- This backlog item is a control-plane follow-up from:
  - `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md`
  - `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md`

## Decision Items
1. Decide canonical ownership for Torch-only knobs:
   - Keep in shared `ptycho/config/config.py::ModelConfig`, or
   - Move to Torch-local config and keep bridge projections explicit.
2. Define whether Torch-only knobs are permitted in `params.cfg` at all.
3. Define validation boundaries: bridge-level validation vs Torch workflow/generator validation.
4. Define compatibility policy for existing wrappers (`grid_lines_compare_wrapper.py`, `grid_lines_torch_runner.py`, and runbooks).

## Acceptance Criteria
1. `docs/specs/spec-ptycho-config-bridge.md` explicitly documents ownership and mapping policy for Torch-only knobs.
2. A conformance test matrix exists for each knob:
   - bridge behavior,
   - runner/wrapper propagation,
   - rejection behavior for invalid values.
3. `update_legacy_dict` behavior is documented and tested for Torch-only fields (allowlist/denylist or equivalent explicit rule).
4. Architecture docs and workflow docs are synchronized with the new ownership model.

## Suggested Direction
- Treat this as an ADR-level architecture decision, then implement in one focused initiative.
- Prefer explicit boundaries over convenience pass-through behavior.
