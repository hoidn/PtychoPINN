# Phase 2 PDEBench SWE Longer Execution Review-Fix Report

## Completed In This Pass

- Fixed implementation-review `H1` by schema-migrating the delivered reusable run budget at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/run_budget.json` to include `training_seed=20260420`.
- Updated the durable Phase 2 execution summary so the selected run remains clearly labeled as historical unseeded evidence while the corrected budget is valid for future reruns under the current runner contract.
- Updated the tranche execution plan with the review-fix resolution and documents-read record for this pass.

## Completed Current-Scope Work

- Preserved the selected run's historical `invocation.sh`/`invocation.json` provenance instead of retroactively adding a seed to files produced by the unseeded run.
- Added final artifact validation evidence that calls `load_run_budget()` on the shipped `.artifacts/.../run_budget.json` and confirms `training_seed=20260420` plus required primary profiles.
- Preserved the Phase 2 decision, `Decision: pivot to OpenFWI FlatVel-A`; no long rerun, OpenFWI fallback execution, CDI work, YAML/prompt edit, or stable core physics/model edit was performed.

## Verification

- RED reproduction before the fix: `load_run_budget(.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/run_budget.json)` failed with `ValueError: run budget missing required field: training_seed`.
- Budget validation after the fix: `python - <<'PY' ... load_run_budget(...) ... PY`
  - Result: passed; `training_seed=20260420`; primary profiles are `hybrid_resnet_base,fno_base,unet_base`.
  - Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/review_fix_budget_validation.log`.
- Focused PDEBench SWE tests: `python -m pytest tests/studies/test_pdebench_swe_manifest.py tests/studies/test_pdebench_swe_splits_data.py tests/studies/test_pdebench_swe_metrics.py tests/studies/test_pdebench_swe_models.py tests/studies/test_pdebench_swe_run_config.py tests/studies/test_pdebench_swe_longer_cli.py tests/studies/test_pdebench_swe_reporting.py tests/studies/test_pdebench_swe_smoke_cli.py -v`
  - Result: `48 passed in 16.05s`.
  - Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/review_fix_pytest.log`.
- Structural summary/discoverability check:
  - Result: `PDE execution summary and discoverability checks passed`; decision remains `pivot to OpenFWI FlatVel-A`.
  - Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/review_fix_structural_check.log`.
- Plan pointer check:
  - Result: `plan_path pointer is valid`.
  - Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/review_fix_plan_pointer_check.log`.
- Selected-run freshness/PID check:
  - Result: selected longer run verified with `run_id=20260420T115509.961336393Z`, tracked PID `3052378`, and root `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/runs/20260420T115509.961336393Z`.
  - Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/review_fix_selected_run_check.log`.

## Follow-Up Work

- Execute the OpenFWI FlatVel-A fallback in a separate approved tranche.
- Rerun PDEBench SWE only if a fixed-seed SWE primary record is still needed despite the documented pivot; use the corrected budget with `training_seed=20260420` and the revised tmux/run-budget contract.

## Residual Risks

- The selected SWE run remains unseeded for model/training RNGs and should not be promoted as fixed-seed reproducible or paper-facing evidence.
- The Phase 2 pivot is based on one observed one-step SWE budget, not multi-seed robustness.
- The OpenFWI fallback still needs storage/access preflight and its own implementation review.
