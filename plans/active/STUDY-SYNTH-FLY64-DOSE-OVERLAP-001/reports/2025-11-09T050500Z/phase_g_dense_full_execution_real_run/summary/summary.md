### Turn Summary
Provisioned the 050500Z dense pipeline hub and refreshed plan.md plus fix_plan to target automatic analyze_dense_metrics invocation inside run_phase_g_dense.py.
Rewrote input.md with TDD guardrails so Ralph adds the new orchestrator digest integration test, updates run_phase_g_dense.py, and reruns the dense Phase C→G pipeline with --clobber while capturing logs.
Next: Ralph executes the orchestrator changes, confirms GREEN selectors, runs the pipeline end-to-end, and records MS-SSIM/MAE deltas with digest links in summary/docs.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T050500Z/phase_g_dense_full_execution_real_run/ (plan/plan.md, summary/summary.md)
