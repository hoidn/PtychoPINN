### Turn Summary
Identified the dense Phase C subprocess aborting with `ValueError: Object arrays cannot be loaded when allow_pickle=False` before any data/analysis artifacts were produced.
Scoped a TDD fix to load metadata-bearing NPZ files via MetadataManager inside build_simulation_plan and load_data_for_sim, with new pytest guards plus pipeline rerun instructions.
Next: implement the metadata-aware loaders, prove the new tests RED→GREEN, relaunch the Phase C→G pipeline with --clobber, and regenerate the metrics digest evidence.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T210500Z/phase_g_dense_full_execution_real_run/ (plan/phase_c_generation_failure.log)
