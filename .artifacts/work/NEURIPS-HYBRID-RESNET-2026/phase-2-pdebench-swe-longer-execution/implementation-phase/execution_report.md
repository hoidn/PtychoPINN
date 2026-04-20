# Phase 2 PDEBench SWE Longer Execution Review-Fix Report

## Completed In This Pass

- Fixed implementation-review `H1` by rejecting live `logs/longer.pid` roots without `logs/longer.exit_code` regardless of `--allow-existing-output-root`; only `longer.run_id` and `longer.started_at_ns` remain prelaunch markers.
- Fixed implementation-review `H2` for future runs by adding `--training-seed`, requiring `training_seed` in run budgets, seeding Python/NumPy/Torch/CUDA before each profile build, and recording the seed in invocation/profile metrics/profile provenance.
- Fixed implementation-review `M1` by requiring both `fno_base` and `unet_base` in reporting and rejecting budget-backed profile overrides that omit required primary profiles outside inspect-only mode.
- Updated the durable plan/runbook, studies index, docs index, and PDE execution summary to document the tightened seed/PID/baseline contracts.

## Completed Current-Scope Work

- Added regression coverage for live PID output roots, training-seed CLI/provenance, run-budget seed validation, budget-backed profile override rejection, and reporting completeness when a baseline is omitted.
- Explicitly downgraded selected run `20260420T115509.961336393Z` in `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md` as unseeded observed evidence; no long rerun was launched in this pass.
- Preserved the existing Phase 2 decision, `Decision: pivot to OpenFWI FlatVel-A`, as a conservative observed SWE pivot signal rather than a fixed-seed paper-facing claim.

## Verification

- `python -m pytest tests/studies/test_pdebench_swe_manifest.py tests/studies/test_pdebench_swe_splits_data.py tests/studies/test_pdebench_swe_metrics.py tests/studies/test_pdebench_swe_models.py tests/studies/test_pdebench_swe_run_config.py tests/studies/test_pdebench_swe_longer_cli.py tests/studies/test_pdebench_swe_reporting.py tests/studies/test_pdebench_swe_smoke_cli.py -v`
  - Result: `48 passed in 15.91s`
  - Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/final_pytest.log`
- `python scripts/studies/run_pdebench_swe_longer.py --help`
  - Result: help includes `--training-seed`
  - Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/longer_help.log`
- Structural summary/discoverability check:
  - Result: `PDE execution summary and discoverability checks passed`
- Plan pointer check:
  - Result: `plan_path pointer is valid`
- Selected-run identity check:
  - Result: `selected longer run verified: run_id=20260420T115509.961336393Z pid=3052378 root=.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/runs/20260420T115509.961336393Z`

## Follow-Up Work

- Execute the OpenFWI FlatVel-A fallback in a separate approved tranche.
- Rerun PDEBench SWE only if a fixed-seed SWE primary record is still needed despite the documented pivot; use `training_seed=20260420` and the revised tmux/run-budget contract.

## Residual Risks

- The selected SWE run remains unseeded for model/training RNGs and should not be promoted as fixed-seed reproducible or paper-facing evidence.
- The Phase 2 pivot is based on one observed one-step SWE budget, not multi-seed robustness.
- The OpenFWI fallback still needs storage/access preflight and its own implementation review.
