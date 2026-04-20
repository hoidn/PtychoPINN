# Phase 2 PDEBench SWE Longer Execution Second Review-Fix Report

## Completed In This Pass

- Fixed implementation-review `H1` for longer-run locking and freshness evidence.
- Added RED/GREEN regressions for stale `logs/longer.pid` markers without `logs/longer.exit_code`, missing exit-code evidence during freshness validation, and nonzero exit-code evidence.
- Updated the runner so any per-run `logs/longer.pid` marker without `logs/longer.exit_code` is a hard output-root error, regardless of PID liveness or `--allow-existing-output-root`.
- Updated freshness validation to require fresh `logs/longer.run_id`, `logs/longer.started_at_ns`, `logs/longer.pid`, and `logs/longer.exit_code == "0"`.
- Updated durable plan, summary, and studies-index wording so the documented contract matches the stricter implementation.

## Completed Current-Scope Work

- Preserved the Phase 2 decision, `Decision: pivot to OpenFWI FlatVel-A`.
- Preserved the selected historical SWE run as unseeded observed pivot evidence; no provenance was retroactively edited.
- Did not launch a new long run, execute OpenFWI fallback work, start CDI work, edit YAML/prompt files, create a worktree, or touch stable core physics/model modules.

## Verification

- RED check before the code fix: `python -m pytest tests/studies/test_pdebench_swe_longer_cli.py -k 'prelaunch or requires_run_markers' -v`
  - Result: failed as expected on stale PID marker acceptance and missing `longer.exit_code` freshness acceptance.
- Targeted GREEN check after the code fix: same selector.
  - Result: `2 passed, 6 deselected`.
- Focused PDEBench SWE suite: `python -m pytest tests/studies/test_pdebench_swe_manifest.py tests/studies/test_pdebench_swe_splits_data.py tests/studies/test_pdebench_swe_metrics.py tests/studies/test_pdebench_swe_models.py tests/studies/test_pdebench_swe_run_config.py tests/studies/test_pdebench_swe_longer_cli.py tests/studies/test_pdebench_swe_reporting.py tests/studies/test_pdebench_swe_smoke_cli.py -v`
  - Result: `49 passed in 15.99s`.
  - Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/second_review_fix_pytest.log`.
- Selected-run freshness check with the stricter helper:
  - Result: selected longer run verified with `run_id=20260420T115509.961336393Z`, tracked PID `3052378`, root `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/runs/20260420T115509.961336393Z`, and `logs/longer.exit_code=0`.
  - Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/second_review_fix_selected_run_check.log`.

## Follow-Up Work

- Execute the OpenFWI FlatVel-A fallback in a separate approved tranche.
- Rerun PDEBench SWE only if a fixed-seed SWE primary record is still needed despite the documented pivot; use the corrected budget, seed-recording contract, and stricter run-marker validation.

## Residual Risks

- The selected SWE run remains unseeded for model/training RNGs and should not be promoted as fixed-seed reproducible or paper-facing evidence.
- The Phase 2 pivot is based on one observed one-step SWE budget, not multi-seed robustness.
- The OpenFWI fallback still needs storage/access preflight and its own implementation review.
