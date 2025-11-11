### Turn Summary
Reality check: commit 74a97db5 landed the default-on post-verify automation + pytest proofs, yet the 2025-11-12 hub still has empty `{analysis,cli}` folders, so we re-scoped the plan around delivering the counted dense run plus a `--post-verify-only` mode to revalidate hubs without rerunning Phase C→F.
Updated plan.md, docs/fix_plan.md, input.md, and the hub metadata to target the new workflow (add the flag in `run_phase_g_dense.py`, extend orchestrator tests, run the dense pipeline with `--clobber`, then rerun the orchestrator in post-verify-only mode for verification evidence).
Next: implement the flag/tests, archive RED/GREEN logs for the new selectors, execute the dense Phase C→G run, rerun `--post-verify-only`, and capture MS-SSIM/MAE deltas + preview verdict + CLI/log references under this hub.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/ (plan/plan.md, summary/summary.md, summary.md)

### Turn Summary
Wired post-verify automation into run_phase_g_dense.py with default-on --skip-post-verify flag, invoking verify_dense_pipeline_artifacts.py and check_dense_highlights_match.py after ssim_grid.
Extended orchestrator tests (collect-only + post-verify hooks) with monkeypatched run_command assertions; both targeted tests GREEN validating CLI strings, invocation order, and hub-relative log paths.
Next: run counted dense Phase C→G pipeline to validate end-to-end automation and capture verification/highlights artifacts in the 2025-11-12 hub.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/ (pytest_collect_only.log, pytest_post_verify_hooks.log)
