### Turn Summary
Wired post-verify automation into run_phase_g_dense.py with default-on --skip-post-verify flag, invoking verify_dense_pipeline_artifacts.py and check_dense_highlights_match.py after ssim_grid.
Extended orchestrator tests (collect-only + post-verify hooks) with monkeypatched run_command assertions; both targeted tests GREEN validating CLI strings, invocation order, and hub-relative log paths.
Next: run counted dense Phase C→G pipeline to validate end-to-end automation and capture verification/highlights artifacts in the 2025-11-12 hub.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/ (pytest_collect_only.log, pytest_post_verify_hooks.log)
