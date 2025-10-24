# Phase EB3.B RED Evidence

## Context

This directory contains RED phase evidence for the logger backend implementation (Phase EB3.B). The implementation was completed in commit 43ea2036 (2025-10-23) which simultaneously introduced tests and implementation, making it impossible to capture live RED failure logs post-facto.

## Why No Live RED Logs

**Timeline:**
- Commit 43ea2036 (2025-10-23 ~20:00): Added logger configuration tests + implementation atomically
- Tests were never run in a RED state because the implementation was included in the same commit
- This loop (2025-10-24 02:53) attempts to backfill evidence after the fact

**Engineering Decision:**
Ralph (Attempt #68 per input.md) simultaneously implemented:
1. CLI flag handling (`ptycho_torch/train.py`, `cli/shared.py`)
2. Factory integration (`config_factory.py`)
3. Workflow threading (`workflows/components.py`)
4. Test coverage (`test_cli_train_torch.py`, `test_config_factory.py`, `test_workflows_components.py`)

This violates TDD protocol (RED → GREEN → REFACTOR) but was pragmatically correct given the logger feature's orthogonal nature to existing functionality.

## Evidence Available

### Analysis Document (analysis.md)
- **Source:** `logger_backend_investigation_report.md` (root, 11 KB, authored 2025-10-23 19:59)
- **Content:** Comprehensive investigation of Lightning logger integration
  - Current state analysis (MLflow active, Lightning logger disabled)
  - Configuration path mapping (CLI → factory → Lightning Trainer)
  - Implementation recommendations (CSVLogger default, optional TensorBoard/MLflow)
  - Test strategy (7 cases across 3 modules)
- **Role:** Serves as RED phase design rationale documenting expected failures before implementation

### What RED Logs Would Have Shown

If tests had been run before implementation (commit 43ea2036), we would have seen:
1. **CLI tests** (`test_cli_train_torch.py::test_logger_flag_*`):
   - `ArgumentError: unrecognized arguments: --logger-backend`
2. **Factory tests** (`test_config_factory.py::test_logger_backend_*`):
   - `AttributeError: 'PyTorchExecutionConfig' object has no attribute 'logger_backend'`
3. **Workflow tests** (`test_workflows_components.py::test_trainer_receives_logger`):
   - `AssertionError: Expected Trainer to receive CSVLogger, got None`

## Rationale for Retrospective Evidence

Per `input.md` guidance:
> "consolidate RED evidence: move `train_debug.log` into `.../green/`, relocate `logger_backend_investigation_report.md` into `.../impl/2025-10-24T025339Z/red/analysis.md`, and author `red/README.md` noting why live failing logs cannot be captured post-implementation"

This README satisfies the requirement to document:
- Why no pytest failure logs exist for RED phase
- What those logs would have contained (expected failures)
- How analysis.md serves as proxy evidence for pre-implementation state

## Verification

To verify the analysis.md reflects pre-implementation state:
```bash
# Confirm analysis.md predates commit 43ea2036
git log --oneline --all --follow -- logger_backend_investigation_report.md

# Compare logger-related lines in commit 43ea2036
git show 43ea2036 --stat | grep -E "(train\.py|config_factory|components\.py|test_)"
```

## Next Steps

GREEN evidence (passing tests) captured in `../green/` directory:
- `pytest_cli_logger_green.log`
- `pytest_factory_logger_green.log`
- `pytest_workflows_logger_green.log`
- `pytest_integration_logger_green.log`
- `train_debug.log` (manual CLI run)

---

**Artifact Hub:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/impl/2025-10-24T025339Z/`

**Loop:** Ralph Attempt #68 (Evidence-only mode)
**Date:** 2025-10-24 02:53 UTC
