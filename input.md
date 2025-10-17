Summary: Capture fixture alignment and red tests for Phase E2 integration parity
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 – Phase E2 Integration Regression & Parity Harness
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_integration_workflow_torch.py -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T213500Z/{phase_e_fixture_sync.md,red_phase.md,phase_e_red_integration.log}

Do Now:
1. INTEGRATE-PYTORCH-001 E2.A1+E2.A2 @ plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md + reports/2025-10-17T212500Z/phase_e2_plan.md — document TEST-PYTORCH-001 fixture inventory and minimal dataset/CLI parameters in phase_e_fixture_sync.md (tests: none)
2. INTEGRATE-PYTORCH-001 E2.B1+E2.B2 @ plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md + reports/2025-10-17T212500Z/phase_e2_plan.md — author torch-optional integration test skeleton, run pytest tests/torch/test_integration_workflow_torch.py -vv, and capture red evidence in red_phase.md + phase_e_red_integration.log (tests: targeted)

If Blocked: Log the blocker and partial findings inside phase_e_fixture_sync.md, save pytest output as phase_e_red_integration.log even if collection fails, and notify the supervisor via docs/fix_plan.md Attempts history.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md:84 — Phase E2 checklist now requires fixture sync before implementation.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T212500Z/phase_e2_plan.md §E2.A–B — authoritative guidance for this loop.
- plans/pytorch_integration_test_plan.md:1 — defines shared fixtures with TEST-PYTORCH-001, must stay in lockstep.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T173826Z/phase_e_callchain/summary.md — backend callchain expectations for new tests.
- docs/findings.md:8 (POLICY-001) — torch must be available; red tests should assume mandatory PyTorch baseline.

How-To Map:
- Create artifact directory: `mkdir -p plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T213500Z`.
- Fixture audit template: reuse table structure from phase_e2_plan.md; include columns for dataset path, n_groups, gridsize, owner, runtime budget, notes (CONFIG-001/DATA-001 compliance).
- New test file location: `tests/torch/test_integration_workflow_torch.py`; follow pytest style (no unittest.TestCase). Use subprocess helpers similar to `tests/test_integration_workflow.py` but guarded with torch availability checks.
- Update `tests/conftest.py` whitelist if necessary to allow new test execution without torch (mirroring config bridge pattern).
- Pytest command: `pytest tests/torch/test_integration_workflow_torch.py -vv` (expect FAILURE). Save output to `phase_e_red_integration.log` using shell redirection or `tee`.
- Summarize failures, selectors, and next steps in `red_phase.md`; note any skips, error messages, and expected remediation in Phase E2.C.

Pitfalls To Avoid:
- Do not implement PyTorch wiring yet; keep this loop red-phase only.
- Avoid editing stable TensorFlow workflow files (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Ensure new tests remain torch-optional by design; gate imports with try/except as documented.
- Keep pytest logs concise—capture the command and stderr/stdout exactly once.
- Maintain params.cfg initialization (CONFIG-001) inside tests before invoking workflows.
- Use deterministic fixture sizes (≤10 groups) to keep runtime manageable for CI.
- Do not delete or relocate existing artifacts; append new notes under the timestamped report directory.
- Ensure red tests use pytest idioms (fixtures, asserts) without unittest inheritance.
- When mocking subprocess calls, avoid global state mutations outside test scope.
- Document all assumptions in phase_e_fixture_sync.md to prevent drift with TEST-PYTORCH-001.

Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md:80
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T212500Z/phase_e2_plan.md:14
- plans/pytorch_integration_test_plan.md:1
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T173826Z/phase_e_callchain/summary.md:10
- tests/test_integration_workflow.py:1

Next Up: Prepare Phase E2.C wiring tasks once red tests and fixture inventory are committed.
