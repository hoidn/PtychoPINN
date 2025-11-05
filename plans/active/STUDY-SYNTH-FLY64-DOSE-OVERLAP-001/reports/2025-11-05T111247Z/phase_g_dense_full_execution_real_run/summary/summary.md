### Turn Summary
Reaudited the 2025-11-09T190500Z dense hub and found only legacy Phase C logs with no metrics artifacts, so the pipeline evidence remains missing.
Provisioned a fresh 2025-11-05T111247Z hub with plan/summary scaffolding and reiterated guardrails (AUTHORITATIVE_CMDS_DOC, process sanity checks, artifact verification) for the relaunch.
Updated docs/fix_plan.md status + attempts and rewrote input.md to hand Ralph a ready-for-implementation pipeline run Do Now tied to the new hub.
Next: Ralph runs the mapped regression selector, executes run_phase_g_dense.py with --clobber, verifies the metrics/highlights bundle, and records MS-SSIM/MAE deltas in summary/docs.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T111247Z/phase_g_dense_full_execution_real_run/ (plan/plan.md, summary/summary.md)
