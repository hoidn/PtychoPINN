### Turn Summary
Analyzed the XLA trace caching bug blocking dose_response_study.py; root cause is `@tf.function(jit_compile=True)` on `projective_warp_xla_jit` persisting shape traces across model creations.
Identified that `create_model_with_gridsize()` sets `use_xla_translate=False` in params.cfg but Translation layers may still use XLA due to timing of `should_use_xla()` evaluation.
Wrote input.md with Phase A tasks: create test reproducing bug (A0), fix XLA toggle propagation in model factory (A1), verify test passes (A2).
Next: Ralph implements the test and fix per input.md checklist.
Artifacts: plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-06T173000Z/ (summary.md)
