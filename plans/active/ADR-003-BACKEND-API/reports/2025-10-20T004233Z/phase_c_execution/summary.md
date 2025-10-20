# Phase C Planning Summary — PyTorch Execution Config Integration

**Timestamp:** 2025-10-20T004233Z  
**Initiative:** ADR-003-BACKEND-API  
**Scope:** Phase C (Execution Config integration across factories, workflows, CLI)

## What We Planned
- Promote `PyTorchExecutionConfig` to a canonical dataclass in `ptycho/config/config.py` (Option A) with full field coverage and documentation updates.
- Replace placeholder execution-config wiring inside `ptycho_torch/config_factory.py` so payloads emit concrete dataclasses and honour override precedence.
- Thread execution-config data through workflow helpers (`_train_with_lightning`, inference loaders) and extend workflow tests to assert Trainer kwargs + deterministic behaviour.
- Collapse CLI entry points onto factories, expose runtime knobs via argparse, and update docs/specs to describe the new surface.
- Enforce strict reporting discipline under `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/` with RED/GREEN logs for each sub-phase.

## Key References
- `factory_design.md` §2.2 — canonical field list + Option A decision for dataclass placement.
- `override_matrix.md` §2 — override precedence across execution knobs.
- `phase_b3_implementation/summary.md` — confirms factory payloads now GREEN and ready for execution-config wiring.
- `specs/ptychodus_api_spec.md` §4, §6 and `docs/workflows/pytorch.md` §§5–13 — authoritative contracts that must be updated during C1/C4.

## Risks & Mitigations
- **Field Drift:** If Lightning knobs diverge from dataclass defaults, document delta in `design_delta.md` and update override matrix at the same time.
- **Trainer Signature Changes:** Wrap Lightning Trainer invocation in helper to minimise blast radius; tests in `tests/torch/test_workflows_components.py` will catch regressions.
- **CLI Surface Growth:** Introduce flags incrementally (starting with accelerator/deterministic/num_workers). Defer long-tail knobs by logging TODOs in summary + docs.
- **Artifact Discipline:** CLI runs may emit `train_debug.log`; mandate relocation to this plan directory at the end of each loop.

## Next Supervisor Checkpoints
1. Confirm C1 RED→GREEN evidence (new test module + dataclass export) and that spec/workflow docs were updated.
2. Ensure factories now emit real `PyTorchExecutionConfig` instances and that execution overrides appear in audit trails.
3. Verify workflow tests assert Trainer kwargs (accelerator, deterministic) and pass on CPU-only runs.
4. Review CLI changes + docs updates before approving Phase D (legacy API deprecation) work.

## Open Questions To Track
- Do we expose MLflow/Logger control in Phase C or defer to Phase D/ADR governance? (Flag in `design_delta.md` during C1.)
- Should execution config support user-provided dataclass overrides from YAML? Consider adding to Phase D backlog if needed.
- Confirm whether existing CLI smoke tests cover inference-only scenarios; if not, add new test case in Phase C4.D3.

## Action State
- <Action State>: [planning]
