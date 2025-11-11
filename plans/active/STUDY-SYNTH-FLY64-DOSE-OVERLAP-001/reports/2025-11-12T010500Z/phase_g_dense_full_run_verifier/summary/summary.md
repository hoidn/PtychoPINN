### Turn Summary
Reframed the Phase G Do Now so Ralph first fixes the run_phase_g_dense success-banner paths to hub-relative strings and extends the orchestrator test, then reruns the dense pipeline with fresh evidence.
Updated docs/fix_plan.md, implementation.md, and the hub plan to spotlight the relative-path guard plus counted dense run + post-verify-only rerun, and rewrote input.md with the exact commands/log targets.
Next: land the relative-path change + tests, capture collect/green logs, execute the dense `--clobber` + `--post-verify-only` commands, and publish MS-SSIM/MAE + preview evidence under the active hub.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/ (plan/plan.md, summary/summary.md)

### Turn Summary
Re-scoped the Phase G focus now that `--post-verify-only` merged: Ralph must run the counted dense Phase C→G pipeline into the 2025-11-12 hub and capture SSIM grid/verifier/highlights payloads plus a verification-only rerun via the new flag.
Documented the required evidence (analysis/cli contents, artifact inventory refresh, MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, CLI/test logs) across implementation.md, plan.md, docs/fix_plan.md, and refreshed the hub summary metadata.
Next: execute the dense run with `--clobber`, rerun `--post-verify-only`, archive CLI/test logs, and update summary/docs/fix_plan with metrics + verification outputs.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/ (plan/plan.md, summary/summary.md, summary.md)

### Turn Summary
Implemented --post-verify-only flag for run_phase_g_dense.py enabling verification-only reruns (SSIM grid + verifier + highlights) without Phase C→F re-execution; added mutual exclusivity guards, collect-only mode support with hub-relative paths, and artifact inventory regeneration.
Delivered TDD cycle with 2/2 new pytest selectors PASSED (collect-only assertions + execution chain validation) and comprehensive suite GREEN (453 passed, 1 pre-existing failure).
Next: execute full Phase C→G counted dense pipeline with --clobber, then run --post-verify-only mode against fresh artifacts to validate rerun workflow.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/ (pytest_collect_orchestrator_post_verify_only.log, pytest_post_verify_only.log)
