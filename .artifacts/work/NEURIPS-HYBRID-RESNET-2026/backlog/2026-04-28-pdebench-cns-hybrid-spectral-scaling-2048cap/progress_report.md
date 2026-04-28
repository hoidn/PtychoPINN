# Progress Report

## Active Work

- Audited the frozen `512 / 64 / 64` and `1024 / 128 / 128` finalist references and wrote:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/reference_runs_512cap_40ep.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/reference_runs_1024cap_40ep.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/inspect-20260428T190521Z/inspection_summary.json`
- Added the missing split-cap scaling reporting helper in `scripts/studies/pdebench_image128/reporting.py`.
- Added TDD coverage for the new helper in `tests/studies/test_pdebench_image128_runner.py`.
- Archived passing verification for the new helper and the required study selectors:
  - `verification/preflight_pytest.log`
  - `verification/preflight_compileall.log`
  - `verification/runner_scaling_trend.log`
  - `verification/workflow_fix_pytest.log`
  - `verification/workflow_fix_compileall.log`

## Current Status

- Implementation is blocked before Task 3, so the fresh `2048 / 256 / 256` finalist run was not launched.
- The required workflow gate `pytest -v -m integration` failed after the reporting change even though the study-targeted suites passed.
- Failure site: `tests/test_integration_workflow.py::TestFullWorkflow::test_train_save_load_infer_cycle`
- Immediate error:
  - `scripts/training/train.py` raised `ModuleNotFoundError: No module named 'ptycho'`
- Evidence:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/integration_after_workflow_fix.log`

## Next Resume Condition

- Resume once the repo-level integration gate `pytest -v -m integration` passes again, or once a reviewed follow-up explicitly authorizes proceeding despite this unrelated integration failure.

## Blocker

- The approved plan requires stopping before the fresh GPU run when a production-workflow integration rerun fails.
- The current blocker is the repo-level import failure in `scripts/training/train.py`, not the new scaling helper itself.
