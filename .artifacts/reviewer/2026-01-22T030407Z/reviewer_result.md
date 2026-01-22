# Reviewer Result — 2026-01-22T030407Z

## Issues Identified
1. **Dataset-intensity telemetry ripped out** (`ptycho/loader.py:124-190,360-511`, `ptycho/data_preprocessing.py:199-204`, `ptycho/train_pinn.py:165-180`). `PtychoDataContainer` no longer accepts/records `dataset_intensity_stats`, all manual constructors still pass the keyword (now raising `TypeError`), and `calculate_intensity_scale()` reverted to the closed-form fallback. This fully undoes Phase D4f, violates specs/spec-ptycho-core.md §Normalization Invariants, and contradicts docs/fix_plan.md + plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md which both claim dataset-derived scaling is complete.
2. **Reassembly canvas guard removed** (`ptycho/workflows/components.py:659-705`). `_update_max_position_jitter_from_offsets()` disappeared and `create_ptycho_data_container()` no longer bumps `max_position_jitter`, reintroducing the padding shortfall that Phase C fixed.
3. **Metrics entrypoint now crashes** (`ptycho/image/cropping.py:131-200`, `scripts/studies/sim_lines_4x/evaluate_metrics.py:15,120`). `align_for_evaluation_with_registration()` was deleted even though the sim_lines metrics runner still imports/calls it, so the CLI fails before producing any metrics.
4. **Plan/design drift** (`plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:323-343` vs `scripts/studies/sim_lines_4x/pipeline.py:190-205`). Phase D explicitly forbids loss-weight edits, but the pipeline now forces `realspace_weight=0.1` and `realspace_mae_weight=1.0` with no plan update, rationale, or tests. Documentation and Do Now instructions are now inaccurate.
5. **Doc/spec map regressions** (`ptycho/raw_data.py:96-103`, `ptycho/io/ptychodus_product_io.py:5-12`). Both files now point to `docs/specs/...`, which does not exist (the authoritative specs live under `specs/`).

A `user_input.md` has been written summarizing these blockers and requesting that Phase D4f and the plan be reopened.

## Integration Test
- Outcome: **PASS**
- Command: `RUN_TS=$(date -u +%Y-%m-%dT%H%M%SZ); RUN_TS=$RUN_TS RUN_LONG_INTEGRATION=1 INTEGRATION_OUTPUT_DIR=.artifacts/integration_manual_1000_512/${RUN_TS}/output pytest tests/test_integration_manual_1000_512.py -v`
- Output Dir: `.artifacts/integration_manual_1000_512/2026-01-22T025957Z/output/`
- Key Log Excerpt: `tests/test_integration_manual_1000_512.py::test_train_infer_cycle_1000_train_512_test PASSED (1 passed in 91.60s)`

## Review Window & References
- review_every_n: 3 (from orchestration.yaml)
- State file: `sync/state.json`
- Logs dir: `logs/`
- Materials reviewed: `CLAUDE.md`, `docs/fix_plan.md`, `docs/index.md`, `plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,plan/parity_logging_spec.md}`, `galph_memory.md`, `orchestration.yaml`, `ptycho/{loader.py,data_preprocessing.py,train_pinn.py,workflows/components.py,image/cropping.py,raw_data.py,io/ptychodus_product_io.py}`, `scripts/studies/sim_lines_4x/pipeline.py`, `user_input.md`, plus integration output under `.artifacts/integration_manual_1000_512/2026-01-22T025957Z/`.
