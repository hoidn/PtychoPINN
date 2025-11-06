### Turn Summary
Diagnosed the dense relaunch failure: the Phase C metadata guard still demands legacy dose_*_train/test directories and stops the pipeline at [1/8].
Confirmed the blocker via the relaunch log plus blocker.log and updated docs/fix_plan.md alongside the findings ledger with the guard gap.
Next: rewrite validate_phase_c_metadata for the patched_{train,test} layout, rerun the dense pipeline with --clobber, and finish highlights/digest plus pytest proof.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T081826Z/phase_c_metadata_guard_blocker/ (cli/blocker_phase_c_metadata.log, cli/run_phase_g_dense_relaunch_tail.txt)
