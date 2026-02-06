# Backlog: Implement Global State & Config Flow Refactor

**Created:** 2026-02-06
**Status:** Open
**Priority:** Medium
**Related:** `scripts/studies/grid_lines_compare_wrapper.py`, `scripts/studies/grid_lines_torch_runner.py`, `ptycho_torch/workflows/components.py`, `ptycho_torch/data_container_bridge.py`, `ptycho_torch/legacy_bridge.py`
**Impacts:** `tests/torch/test_grid_lines_compare_wrapper.py`, `tests/torch/test_grid_lines_torch_runner.py`, `tests/torch/test_workflows_components.py`

## Summary
The global state refactor plan calls for a single config builder shared by the torch runner and compare wrapper, plus a centralized legacy bridge that is the only place allowed to mutate `params.cfg` via `update_legacy_dict`. This reduces config pinball and makes the wrapper/runner boundary explicit without changing core physics modules.

## Impact
- **Consistency:** One config builder prevents wrapper and runner defaults from drifting.
- **Safety:** Centralizing `update_legacy_dict` keeps legacy mutations explicit and auditable.
- **Maintainability:** Reduced duplication and a clearer config flow lower regression risk.

## Evidence
- Implementation plan: `docs/plans/2026-02-06-global-state-refactor-synthesis.md`.

## Outstanding Issues
1. Add failing tests to enforce the shared config builder and centralized legacy bridge.
2. Implement `build_torch_runner_config` and update the compare wrapper to call it.
3. Add `ptycho_torch/legacy_bridge.py` and update call sites to use it.
4. Update architecture/workflow docs to describe the refined config flow.
5. Run the targeted pytest selectors listed in the plan.

## Suggested Direction
Execute the plan in `docs/plans/2026-02-06-global-state-refactor-synthesis.md` and validate via the specified pytest selectors.

## Related Artifacts
- Plan: `docs/plans/2026-02-06-global-state-refactor-synthesis.md`
