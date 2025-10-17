# Phase D Review — Override Matrix Preparation

**Initiative:** INTEGRATE-PYTORCH-001
**Timestamp:** 2025-10-17T062820Z (UTC)
**Supervisor:** galph (planning/review)

## What Changed
- Verified Attempt #24 artifacts (`summary.md`, `pytest_baseline.log`, `pytest_parity_full.log`) — baseline comparison test passes and confirms config bridge translations for all spec §5.1-§5.3 fields.
- Updated plan checklists (implementation + parity_green_plan) to mark Phase D.D1 complete and point D2 at this review.
- Selected next evidence target: document the override matrix (required overrides, defaults, and warning expectations) before adding warning tests.

## Key Observations
- Baseline snapshot (`reports/2025-10-17T041908Z/baseline_params.json`) reflects the post-inference `params.cfg` state; training overrides that differ (e.g., `n_groups=1024`, `n_subsample=2048`) are overwritten. Override matrix must call out which values survive the layered update.
- Existing parity suite covers per-field transformations; remaining risk is undefined behaviour when overrides are omitted (nphotons, probe_mask, path conversions). We need explicit documentation + tests so future changes surface regressions immediately.

## Next Actions for Engineering (Attempt #25)
1. Author `override_matrix.md` in this reports folder summarizing, per field, whether adapter defaults, warns, or requires overrides (reference spec §§5.1–5.3 and Attempt #24 config values).
2. Capture red-phase evidence for missing override warnings (e.g., drop `nphotons` override, assert ValueError guidance) to inform Phase D.D3 test additions.
3. Update docs/fix_plan.md with findings and link to new artifact(s).

## Dependencies & References
- specs/ptychodus_api_spec.md §§5.1–5.3 — authoritative field list
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/{baseline_params.json,field_matrix.md}
- tests/torch/test_config_bridge.py — location for upcoming warning assertions
- ptycho_torch/config_bridge.py — adapter implementation inspected for defaults & validation paths

## Risks / Open Questions
- Need confirmation on desired behaviour for training-only overrides (n_groups, n_subsample) since inference update overwrites them; override matrix should clarify expected final state.
- Verify whether future persistence work (Phase D3 / Phase E) requires snapshot refresh when adapter gains new keys.

**Artifacts:** This file (`plan_update.md`).
