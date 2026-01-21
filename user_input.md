# Reviewer Follow-Up — 2026-01-21T01:38Z

## Summary
1. PINN-CHUNKED-001 entry in `docs/findings.md` still says the fix is “calculate_intensity_scale prefers `_X_np`”, but D4f moved the priority to `dataset_intensity_stats`. The knowledge base now contradicts the code and the active plan.
2. Manual `PtychoDataContainer` constructors (dose_response_study, data_preprocessing, inspect_ptycho_data, etc.) never populate `dataset_intensity_stats`, so any workflow that bypasses `loader.load` immediately falls back to the old 988.21 constant. That silently undoes the D4f fix for cached/model-ready containers.

## Evidence
- Doc mismatch: `docs/findings.md:90-98` vs the new D4f implementation in `ptycho/train_pinn.py:165-237`
- Manual constructors without stats: `scripts/studies/dose_response_study.py:395-423`, `ptycho/data_preprocessing.py:202-203`, `scripts/inspect_ptycho_data.py:13-28`

## Plan / Update Needed
- Update the PINN-CHUNKED-001 finding (and any dependent plan text) so it documents the actual priority order: dataset stats → `_X_np` → closed-form fallback. 
- Extend DEBUG-SIM-LINES-DOSE-001 Phase D4f (or open a follow-on task) so every `PtychoDataContainer` creation path attaches raw diffraction stats before calling `calculate_intensity_scale`. Either call `loader.load` in those helpers or compute/stash the same `batch_mean_sum_intensity` so spec compliance applies outside the loader.

## Next Steps for Supervisor
1. Decide whether to update the existing plan vs open a short doc-hygiene task for the finding rewrite; assign owner.
2. Create a follow-up task (likely under DEBUG-SIM-LINES-DOSE-001) to audit all `PtychoDataContainer` constructors and either route them through `loader.load` or add equivalent dataset-stat plumbing so D4f’s fix covers cached/model-ready containers as well.
