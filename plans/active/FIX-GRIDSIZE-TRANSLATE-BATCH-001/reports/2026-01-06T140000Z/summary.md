### Turn Summary
Implemented XLA-compatible batch broadcast in `translate_xla` using modular indexing with `tf.gather` to fix gridsize>1 shape mismatch.
Initial `tf.repeat`/`tf.cond` approach failed XLA compilation; modular indexing avoids compile-time constant requirement.
Added 2 regression tests (`test_translate_xla_gridsize_broadcast`, `test_translate_xla_gridsize_broadcast_jit`); all 8 tests pass.
Next: Run dose_response_study.py to verify e2e fix works (in STUDY-SYNTH-DOSE-COMPARISON-001).
Artifacts: plans/active/FIX-GRIDSIZE-TRANSLATE-BATCH-001/reports/2026-01-06T140000Z/ (pytest_all_tests.log)
