### Turn Summary
Phase G dense pipeline failed during Phase C validation because `validate_dataset_contract()` no longer accepts the legacy `dataset_path` kwargs; captured stack trace and CLI logs under this hub.
Framed recovery plan to load NPZ splits before validation, refresh Phase C tests to exercise the real validator, and keep highlights preview guard green.
Next: Ralph patches `generate_dataset_for_dose`, updates tests, and re-runs `bin/run_phase_g_dense.py --clobber` to resume the dense evidence run.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T150500Z/phase_g_dense_full_execution_real_run/ (plan/plan.md, evidence/)
