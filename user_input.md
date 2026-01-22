# Reviewer Findings — 2026-01-22T03:04Z

## Summary
- Phase D4f regressions: `PtychoDataContainer` no longer accepts or stores `dataset_intensity_stats`, and `calculate_intensity_scale()` now ignores runtime data entirely. This undoes the spec-mandated dataset-derived scaling fix that Phase D4f delivered and breaks every caller that passes the stats keyword.
- Reassembly guard removed: `_update_max_position_jitter_from_offsets()` disappeared from `ptycho/workflows/components.py`, so grouped offsets larger than the legacy padded size will again clip reconstructions (the Phase C canvas fix is effectively disabled).
- Metrics runner broken: `align_for_evaluation_with_registration()` was deleted from `ptycho/image/cropping.py` even though `scripts/studies/sim_lines_4x/evaluate_metrics.py` still imports it, so the evaluation step now raises `ImportError` before running.
- Plan drift: Phase D explicitly forbids loss-weight edits (CLAUDE.md + implementation.md §Phase D notes), yet `scripts/studies/sim_lines_4x/pipeline.py` now overrides `realspace_weight`/`realspace_mae_weight`. The plan, docs, and fix ledger still claim “no loss-weight changes,” so the current scope/Do Now instructions are inaccurate.

## Evidence
- `ptycho/loader.py:124-190,360-511` removed the `dataset_intensity_stats` parameter/attribute/persistence and no longer computes stats inside `load()`. All manual constructors (dose_response_study, data_preprocessing, inspect_ptycho_data, etc.) still pass `dataset_intensity_stats=…`, which now raises `TypeError`.
- `ptycho/train_pinn.py:163-194` replaced the dataset-derived reducer with a hard-coded fallback (`sqrt(nphotons)/(N/2)`) and dead `count_photons()` helper. This contradicts specs/spec-ptycho-core.md §Normalization Invariants and the D4f deliverables logged in docs/fix_plan.md §§D4f–D4f.3.
- `ptycho/workflows/components.py:659-705` deleted `_update_max_position_jitter_from_offsets()` and removed the call inside `create_ptycho_data_container()`, undoing the max-offset canvas sizing that kept reassembly from cropping gs2 runs.
- `ptycho/image/cropping.py:131-180` removed `align_for_evaluation_with_registration()`, but `scripts/studies/sim_lines_4x/evaluate_metrics.py:15,120` still imports/calls it → runtime crash.
- Phase D constraint reference: `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:323-343` (“Constraint: Do not adjust or experiment with loss weights”), yet `scripts/studies/sim_lines_4x/pipeline.py:200` now sets `realspace_weight=0.1` and `realspace_mae_weight=1.0`.

## Plan Update Needed
1. Reopen Phase D4f (and update docs/fix_plan.md + findings) to restore dataset-intensity stats propagation and the spec-compliant `calculate_intensity_scale()` priority order (dataset stats → `_X_np` → fallback). Ensure all constructors/tests compile again.
2. Reintroduce `_update_max_position_jitter_from_offsets()` (or equivalent) so grouped offsets continue to expand the padded canvas before training/inference.
3. Restore `align_for_evaluation_with_registration()` (or update every caller) so the sim_lines_4x metrics CLI can run.
4. Reconcile the plan constraint vs. the new `realspace_weight` overrides—either roll back the weight change or formally amend Phase D scope/Do Now/checklists so work stays auditably aligned with CLAUDE.md.

## Requested Next Steps
1. Update `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md`, docs/fix_plan.md, and docs/findings.md to reflect the reopened D4f/D6 scope plus the maintainer-communication requirements (the current text still claims dataset stats + intensity scale fixes are complete and that loss weights remain untouched).
2. Patch `ptycho/loader.py`, `ptycho/data_preprocessing.py`, `scripts/studies/dose_response_study.py`, `scripts/inspect_ptycho_data.py`, and `ptycho/train_pinn.py` so dataset stats flow and tests pass again; rerun the mapped pytest selectors from Phase D4f to prove no regressions.
3. Restore the canvas jitter guard and the evaluation helper, then rerun the sim_lines_4x runner + analyzer so gs1/gs2 telemetry pipelines are functional again.
4. Confirm (and document) whether realspace loss weights are now in scope; if they are, capture the rationale, impact analysis, and new test plan under Phase D6 before landing further code.
