# Phase E2 Execution Plan — Integration Regression & Parity Harness

## Context
- Initiative: INTEGRATE-PYTORCH-001
- Current Phase: E2 (Integration Regression & Parity Harness)
- Dependencies: Phase E1 backend dispatcher blueprint (`phase_e_backend_design.md`), TEST-PYTORCH-001 integration plan (`plans/pytorch_integration_test_plan.md`), DATA-001 finding, CONFIG-001 gate.
- Findings to honour: POLICY-001 (PyTorch mandatory), CONFIG-001 (params.cfg initialization), DATA-001 (complex64 Y patches).

## Goals
1. Produce torch-optional failing tests that expose the missing PyTorch integration path without requiring immediate implementation.
2. Align fixture strategy with TEST-PYTORCH-001 so integration tests reuse canonical datasets/artifacts.
3. Implement PyTorch backend wiring in Ptychodus once red tests are in place, ensuring CONFIG-001 sequencing and fail-fast error handling.
4. Capture TensorFlow vs PyTorch parity evidence (logs, metrics, qualitative comparison) to close Phase E2.

## Phase Structure
### E2.A — Fixture Alignment (Evidence / Documentation)
- Deliverables: `phase_e_fixture_sync.md` capturing fixture inventory, dataset parameters, CLI knobs, runtime budget.
- Commands: none (documentation only). Use `rg`/`grep` for references, confirm dataset paths exist under `datasets/` or synthetic generator scripts.
- Acceptance: Table lists dataset name, file path, owner, sample count, gridsize, notes about reuse with TEST-PYTORCH-001.

### E2.B — Red Test Authoring (TDD Stage)
- Deliverables: `tests/torch/test_integration_workflow_torch.py`, `phase_e_red_integration.log`, `red_phase.md` summarizing failures.
- Commands: `pytest tests/torch/test_integration_workflow_torch.py -vv`; optional `pytest tests/torch/test_backend_selection.py -k integration -vv` to validate dispatcher hooks.
- Expectations: Tests should fail due to missing dispatcher wiring/torch workflow. Ensure tests are torch-optional via `tests/conftest.py` whitelist.

### E2.C — Implementation (Green Stage)
- Deliverables: Updated dispatcher code (Ptychodus repo), adapter invocations, green pytest logs stored as `phase_e_integration_green.log` and `phase_e_backend_green.log`.
- Commands: `pytest tests/torch/test_backend_selection.py -vv`, `pytest tests/torch/test_integration_workflow_torch.py -vv`.
- Guardrails: Maintain CONFIG-001 updates, raise actionable RuntimeError when torch unavailable, ensure persistence uses Phase D3 shims.

### E2.D — Parity Verification (Evidence)
- Deliverables: `phase_e_tf_baseline.log`, `phase_e_torch_run.log`, `phase_e_parity_summary.md` (metrics + qualitative comparison).
- Commands: `pytest tests/test_integration_workflow.py -k full_cycle -vv`; torch counterpart defined in E2.B tests.
- Acceptance: Logs show successful runs; summary documents parity metrics, runtime comparison, and residual gaps (if any).

## Artifact Discipline
- Planning artifacts (this document) stored under `reports/2025-10-17T212500Z/`.
- Future execution loops must create fresh ISO timestamp directories (e.g., `reports/2025-10-17T213500Z/`) for red/green runs.
- Each artifact referenced from docs/fix_plan.md Attempts history and relevant plan rows.

## Risks & Mitigations
| Risk | Impact | Mitigation |
| --- | --- | --- |
| Fixture mismatch with TEST-PYTORCH-001 | Redundant work, diverging datasets | Align via E2.A inventory before touching code; reuse planned fixture names. |
| PyTorch subprocess startup cost | Long test runtime | Limit integration tests to minimal dataset (<20 groups); document runtime budget in fixture sync note. |
| Dispatcher regression in TensorFlow path | Legacy breakage | Add regression assertion to PyTorch tests ensuring default backend remains TensorFlow when unspecified; reuse existing TensorFlow integration tests as guard. |
| CI environment missing torch | Test failure noise | Policy mandates torch availability; however, keep `pytest --collect-only tests/torch/` in handoff to detect environment regressions quickly. |

## Next Supervisor Actions
1. Update `phase_e_integration.md` checklist with E2 sub-tasks (done in this loop).
2. Direct engineering loop to execute E2.A1–E2.B2 (fixture alignment + red tests) next, capturing evidence under `reports/2025-10-17T213500Z/`.
3. After red phase captured, plan follow-up loops for E2.C (implementation) and E2.D (parity metrics).

*Generated: 2025-10-17T212500Z*
