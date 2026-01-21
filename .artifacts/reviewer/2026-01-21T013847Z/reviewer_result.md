# Reviewer Result — 2026-01-21T013847Z

## Issues Identified
1. `docs/findings.md:90-98` still documents PINN-CHUNKED-001 as "calculate_intensity_scale prefers `_X_np`" even though the new D4f work added a higher-priority dataset statistics path. The doc now disagrees with the implementation and the plan (which says the loader attaches raw stats). Readers will keep chasing `_X_np` bugs instead of verifying that `dataset_intensity_stats` exists. Update the finding to describe the actual priority order (dataset stats → `_X_np` → closed-form) so the knowledge base matches behavior.
2. The new dataset-derived intensity scale only applies to containers built via `loader.load`. Manual `PtychoDataContainer` constructors never populate `dataset_intensity_stats`, so `train_pinn.calculate_intensity_scale()` immediately falls back to the normalized `_X_np` path (i.e., the closed-form constant). Current examples include `scripts/studies/dose_response_study.py:395-423`, `ptycho/data_preprocessing.py:202-203`, and `scripts/inspect_ptycho_data.py:13-27`. Any workflow that loads cached `PtychoDataContainer` NPZ files or creates containers directly will silently lose the fix and continue to log the 988.21 fallback. Those call sites need to run the same raw-diffraction reducer (or go through `loader.load`) before invoking `calculate_intensity_scale`.

## Integration Test
- Command: `RUN_TS=$(date -u +%Y-%m-%dT%H%M%SZ) RUN_LONG_INTEGRATION=1 INTEGRATION_OUTPUT_DIR=.artifacts/integration_manual_1000_512/${RUN_TS}/output pytest tests/test_integration_manual_1000_512.py -v`
- Outcome: PASS on first attempt; outputs under `.artifacts/integration_manual_1000_512/2026-01-21T012912Z/output/`

## Review Window and Context
- Commits inspected: 45412d4e → 70a60428
- `review_every_n`: 3 (from `orchestration.yaml`)
- `state_file`: `sync/state.json`
- `logs_dir`: `logs/`
- Logs consulted: none beyond the passing integration output log
