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

## Status Update (2025-10-20T010816Z)
- **C1 COMPLETE:** `PyTorchExecutionConfig` now lives in `ptycho/config/config.py` with `__all__` export; defaults match `design_delta.md` inventory. Tests captured RED→GREEN cycle via `pytest_execution_config_{red,green}.log` (17 cases). Spec §4.8/§6 and workflow guide §12 refreshed to describe execution-config contract.
- **Artifacts Relocated:** All logs/docs for C1 reside under `reports/2025-10-20T004233Z/phase_c_execution/`. No stray `train_debug.log` remaining at repo root.
- **C2 ENTRY CONDITIONS:** Factories already GREEN from Phase B3 (`create_*_payload` returning canonical structs). Ready to inject execution-config wiring without breaking CONFIG-001 bridge.

## Status Update (2025-10-20T032500Z)
- **C3 COMPLETE:** `_train_with_lightning` and inference helpers now accept `PyTorchExecutionConfig` and thread Trainer/DataLoader kwargs per `reports/2025-10-20T025643Z/phase_c3_workflow_integration/summary.md`. RED→GREEN evidence captured in `pytest_workflows_execution_{red,green}.log`; full regression (`pytest_full_suite.log`) passed clean.
- **Exports Restored:** `ptycho/config/config.py` `__all__` list reinstated so downstream imports (Ptychodus integration) resolve `PyTorchExecutionConfig`.
- **Hygiene:** Root-level `train_debug.log` relocated to the C3 report directory; plan/ledger entries updated (docs/fix_plan.md Attempt #95).
- **C4 ENTRY CONDITIONS:** CLI wrappers still build configs manually; execution config knobs (scheduler, logger_backend, checkpoint callbacks) remain deferred. C4 can proceed once CLI flag surface + test plan are finalised.

## Status Update (2025-10-20T033100Z)
- **C4.A–C4.B COMPLETE (RED):** Design artefacts (`cli_flag_inventory.md`, `flag_selection_rationale.md`, `flag_naming_decisions.md`, `argparse_schema.md`) authored under `reports/2025-10-20T033100Z/phase_c4_cli_integration/`. RED pytest scaffolds (`tests/torch/test_cli_train_torch.py`, `tests/torch/test_cli_inference_torch.py`) landed with logs `pytest_cli_{train,inference}_red.log`; `red_baseline.md` documents failure signatures (argparse `unrecognized arguments`, mock-call assertions) and GREEN criteria.
- **Plan Sync:** Phase C4 checklist rows marked `[x]` for design + RED tasks; `docs/fix_plan.md` Attempt #18 records deliverables and artefact paths.
- **Implementation Ready:** Remaining work is C4.C/D/E/F — refactor CLI argparse + factory wiring, GREEN the new tests, update documentation, and close ledger entries. No code changes yet; current CLI still manual.

## Risks & Mitigations
- **Field Drift:** If Lightning knobs diverge from dataclass defaults, document delta in `design_delta.md` and update override matrix at the same time.
- **Trainer Signature Changes:** Wrap Lightning Trainer invocation in helper to minimise blast radius; tests in `tests/torch/test_workflows_components.py` will catch regressions.
- **CLI Surface Growth:** Introduce flags incrementally (starting with accelerator/deterministic/num_workers). Defer long-tail knobs by logging TODOs in summary + docs.
- **Artifact Discipline:** CLI runs may emit `train_debug.log`; mandate relocation to this plan directory at the end of each loop.

## Next Supervisor Checkpoints
1. Execute Phase C4.C implementation: update argparse surfaces, call factories, and pass execution config through to workflows while honouring CONFIG-001 ordering.
2. GREEN the new CLI pytest modules (`tests/torch/test_cli_train_torch.py`, `tests/torch/test_cli_inference_torch.py`) and capture logs (`pytest_cli_{train,inference}_green.log`), plus smoke factory selector reused from C2.
3. Refresh documentation (`docs/workflows/pytorch.md` §13, `specs/ptychodus_api_spec.md` CLI tables) and summarise C4 close-out once implementation/test cycle completes.
4. Maintain artifact hygiene — relocate CLI-generated logs under `phase_c4_cli_integration/` and update ledger immediately after execution.

## Open Questions To Track
- Do we expose MLflow/Logger control in Phase C or defer to Phase D/ADR governance? (Flag in `design_delta.md` during C1.)
- Should execution config support user-provided dataclass overrides from YAML? Consider adding to Phase D backlog if needed.
- Confirm whether existing CLI smoke tests cover inference-only scenarios; if not, add new test case in Phase C4.D3.

## Action State
- <Action State>: [planning]
