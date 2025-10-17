# ADR-003 Backend API Planning Summary (2025-10-17T224444Z)

## Inputs & References
- ADR draft: `docs/architecture/adr/ADR-003.md` (proposed)
- Spec alignment: `specs/ptychodus_api_spec.md` §4
- Workflow guidance: `docs/workflows/pytorch.md`
- Existing config bridge artifacts under `plans/active/INTEGRATE-PYTORCH-001/`

## Key Decisions
1. Maintain canonical TensorFlow dataclasses as the shared contract while introducing a `PyTorchExecutionConfig` for backend-specific knobs.
2. Centralise `TF*Config` construction via factories in `ptycho_torch/config_factory.py` to eliminate duplicated overrides.
3. Refactor `ptycho_torch/workflows/components.py` as the authoritative programmatic API and reduce CLI scripts to thin wrappers.
4. Deprecate the legacy `ptycho_torch/api/` surface once the new workflow API ships.

## Plan Highlights
- Five phased rollout covering architecture inventory, factory implementation, workflow refactor, CLI wrapping, and legacy deprecation/governance.
- Checklist IDs (A1–E3) capture actionable tasks with reporting expectations and artefact storage rules.
- Explicit cross-plan coordination with `INTEGRATE-PYTORCH-001` to avoid redundant work and ensure parity evidence.

## Next Steps
- Log new fix-plan entry `[ADR-003-BACKEND-API]` in `docs/fix_plan.md` referencing this plan and artefact path.
- Prepare supervisor directives for execution loops focusing on Phase A tasks.

