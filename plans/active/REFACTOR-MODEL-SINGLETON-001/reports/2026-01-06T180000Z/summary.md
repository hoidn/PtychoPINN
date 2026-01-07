### Turn Summary
Refined the root cause analysis: XLA traces are created at module import time (model.py:554-562), not at factory call time, so setting `use_xla_translate=False` in `create_model_with_gridsize()` is too late.
The fix requires setting `USE_XLA_TRANSLATE=0` environment variable BEFORE any ptycho imports; this bypasses XLA tracing entirely during module-level model creation.
Next: Ralph implements test file `tests/test_model_factory.py` and updates `dose_response_study.py` with the env var fix.
Artifacts: plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-06T180000Z/ (input.md updated)
