# Phase 2 PDEBench SWE Longer Execution Review-Fix Report

## Completed In This Pass

- Fixed implementation-review `H1` for the remaining output-root locking and selected-root freshness bypasses.
- Added RED/GREEN regressions for:
  - live `logs/longer.pid` markers being rejected even when stale `logs/longer.exit_code=0` exists;
  - stale per-run `logs/longer.exit_code` being removed when a fresh run starts;
  - `validate_fresh_artifacts()` rejecting invocation metadata whose parsed `output_root` does not resolve to the validated run root.
- Updated `scripts/studies/pdebench_swe/longer.py` so live PID evidence always blocks root reuse, stale completion evidence is invalidated before fresh start markers are written, and freshness validation checks invocation `output_root` identity.
- Updated the durable execution plan, PDE execution summary, and studies index to describe the strengthened lock/freshness contract.

## Completed Current-Scope Work

- Preserved the Phase 2 summary decision, `Decision: pivot to OpenFWI FlatVel-A`.
- Preserved the selected historical SWE run as unseeded observed pivot evidence; no provenance or metrics were retroactively edited.
- Did not launch a new long run, execute OpenFWI fallback work, start CDI work, edit YAML/prompt files, create a worktree, or touch stable core physics/model modules.

## Verification

- RED check before the code fix: `python -m pytest tests/studies/test_pdebench_swe_longer_cli.py -k 'live_pid_even_with_stale_exit_code or invalidates_stale_exit_code or output_root_mismatch' -v`
  - Result: `3 failed, 8 deselected` for the intended review reproductions.
- Targeted GREEN check after the code fix: same selector.
  - Result: `3 passed, 8 deselected`.
- Longer-run CLI regression suite: `python -m pytest tests/studies/test_pdebench_swe_longer_cli.py -v`
  - Result: `11 passed in 10.41s`.
- Focused PDEBench SWE suite: `python -m pytest tests/studies/test_pdebench_swe_manifest.py tests/studies/test_pdebench_swe_splits_data.py tests/studies/test_pdebench_swe_metrics.py tests/studies/test_pdebench_swe_models.py tests/studies/test_pdebench_swe_run_config.py tests/studies/test_pdebench_swe_longer_cli.py tests/studies/test_pdebench_swe_reporting.py tests/studies/test_pdebench_swe_smoke_cli.py -v`
  - Result: `52 passed in 16.02s`.
  - Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/review_h1_fix_pytest.log`.
- Structural summary/discoverability check:
  - Result: `PDE execution summary and discoverability checks passed`.
- Selected-run strict freshness helper:
  - Result: selected longer run verified with `run_id=20260420T115509.961336393Z`, tracked PID `3052378`, root `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/runs/20260420T115509.961336393Z`.
  - Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/review_h1_fix_selected_run_check.log`.
- Plan pointer check:
  - Result: `plan_path pointer is valid`.

## Follow-Up Work

- Execute the OpenFWI FlatVel-A fallback in a separate approved tranche.
- Rerun PDEBench SWE only if a fixed-seed SWE primary record is still needed despite the documented pivot; use the corrected seed-recording contract and the strengthened root-locking/freshness checks.

## Residual Risks

- The selected SWE run remains unseeded for model/training RNGs and should not be promoted as fixed-seed reproducible or paper-facing evidence.
- The Phase 2 pivot is based on one observed one-step SWE budget, not multi-seed robustness.
- The OpenFWI fallback still needs storage/access preflight and its own implementation review.
