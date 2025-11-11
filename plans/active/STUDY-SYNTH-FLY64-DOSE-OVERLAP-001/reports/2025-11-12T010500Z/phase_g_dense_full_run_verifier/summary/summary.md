### Turn Summary
Implemented --post-verify-only flag for run_phase_g_dense.py enabling verification-only reruns (SSIM grid + verifier + highlights) without Phase C→F re-execution; added mutual exclusivity guards, collect-only mode support with hub-relative paths, and artifact inventory regeneration.
Delivered TDD cycle with 2/2 new pytest selectors PASSED (collect-only assertions + execution chain validation) and comprehensive suite GREEN (453 passed, 1 pre-existing failure).
Next: execute full Phase C→G counted dense pipeline with --clobber, then run --post-verify-only mode against fresh artifacts to validate rerun workflow.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/ (pytest_collect_orchestrator_post_verify_only.log, pytest_post_verify_only.log)
