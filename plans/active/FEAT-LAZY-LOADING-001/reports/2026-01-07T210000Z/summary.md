### Turn Summary
Selected FEAT-LAZY-LOADING-001 as next focus after STUDY-SYNTH-DOSE-COMPARISON-001 completion; analyzed loader.py to identify eager tf.convert_to_tensor calls causing OOM.
Designed Phase A1 test: create tests/test_lazy_loading.py with memory scaling test (safe) and OOM reproduction test (skipped by default).
Next: Ralph creates test file, runs memory scaling test, verifies test collection.
Artifacts: plans/active/FEAT-LAZY-LOADING-001/reports/2026-01-07T210000Z/
