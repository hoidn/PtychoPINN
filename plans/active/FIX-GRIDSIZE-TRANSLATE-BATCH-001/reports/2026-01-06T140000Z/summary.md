### Turn Summary
Confirmed root cause of gridsize>1 Translation layer failure: `translate_xla` builds homography matrices M with batch dimension from translations, but `projective_warp_xla` tiles the grid using images batch dimension — these diverge when channel flattening is applied.
Designed fix: add `tf.repeat` broadcast in `translate_xla` before building M matrices, same pattern as the non-XLA fix at tf_helper.py:779-794.
Next: Ralph implements the broadcast fix, adds regression test, and verifies dose_response_study.py runs with gridsize=2.
Artifacts: plans/active/FIX-GRIDSIZE-TRANSLATE-BATCH-001/reports/2026-01-06T140000Z/ (implementation.md)
