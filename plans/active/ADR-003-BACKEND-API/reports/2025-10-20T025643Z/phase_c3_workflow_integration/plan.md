# Phase C3 Workflow Integration Plan — PyTorch Execution Config (2025-10-20T025643Z)

## Context
- Initiative: ADR-003-BACKEND-API — Standardize PyTorch backend API
- Phase Goal: Thread `PyTorchExecutionConfig` through workflow helpers so Lightning `Trainer` and inference paths honour execution knobs captured by the factories.
- Dependencies:
  - `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/plan.md` (umbrella phase plan + exit criteria)
  - `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/summary.md` (C1–C2 decisions, override precedence rules)
  - `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/{factory_design.md,override_matrix.md}` (field inventory + precedence levels)
  - `specs/ptychodus_api_spec.md` §4.8, §6 (backend selection + CONFIG-001 handshake)
  - `docs/workflows/pytorch.md` §§10–13 (Lightning orchestration, regression expectations)
- Reporting Discipline: Store all C3 artefacts under `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T025643Z/phase_c3_workflow_integration/<ISO8601>/` (e.g., `pytest_workflows_execution_red.log`, `pytest_workflows_execution_green.log`, `summary.md`).
- Targeted Tests:
  - `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestTrainWithLightningGreen::test_execution_config_overrides_trainer -vv`
  - `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestTrainWithLightningGreen::test_execution_config_controls_determinism -vv`
  - `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestInferenceExecutionConfig::test_inference_uses_execution_batch_size -vv`

## Phase Checklist

### Phase C3.A — Trainer Integration
Goal: Allow `_train_with_lightning` to accept `PyTorchExecutionConfig` objects and forward Lightning-specific knobs (accelerator, strategy, deterministic, gradient clipping).
Prereqs: Factories returning execution config (Phase C2); design delta approved.
Exit Criteria: `_train_with_lightning` signature updated, call sites adjusted, Lightning `Trainer(...)` receives values from execution config, RED→GREEN evidence captured.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C3.A1 | Restore `__all__` export list in `ptycho/config/config.py` | [ ] | Regression from Phase C2 removed `__all__`. Reintroduce list so `PyTorchExecutionConfig` remains exported alongside existing configs; update docstring comment explaining export role. Verify via `python -c "from ptycho.config.config import PyTorchExecutionConfig"`. |
| C3.A2 | Update `_train_with_lightning` signature | [ ] | Modify `ptycho_torch/workflows/components.py` to accept `execution_config: PyTorchExecutionConfig`. Maintain CONFIG-001 ordering: legacy params must be populated via factories before this function runs. |
| C3.A3 | Thread trainer kwargs | [ ] | Map execution config fields (`accelerator`, `strategy`, `deterministic`, `gradient_clip_val`, `accum_steps`) to Lightning `Trainer(...)` call. Use `dataclasses.replace` if mutation required. Guard GPU-only options with CPU skip logic. |
| C3.A4 | Capture RED log | [ ] | Before wiring, author failing pytest in `tests/torch/test_workflows_components.py` asserting Trainer receives values. Run selector and capture RED output to `.../phase_c3_workflow_integration/pytest_workflows_execution_red.log`. |

### Phase C3.B — Inference Integration
Goal: Ensure inference helpers respect execution config for dataloaders and runtime overrides.
Prereqs: C3.A signature in place.
Exit Criteria: Inference path uses execution config for `num_workers`, `pin_memory`, `inference_batch_size`; tests cover behavior.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C3.B1 | Propagate execution config through inference helper | [ ] | Update `run_inference_with_lightning` (or equivalent) to accept execution config from payload. Ensure dataloaders use `num_workers`, `pin_memory`; record applied knobs in summary. |
| C3.B2 | Support inference batch size override | [ ] | Allow execution config to change dataloader batch size when `inference_batch_size` is set. Document CPU-only constraints (no GPU). |
| C3.B3 | Author failing inference test | [ ] | Add pytest `TestInferenceExecutionConfig` covering batch size/num_workers; capture RED log (extend same file). |

### Phase C3.C — Workflow Tests & GREEN Pass
Goal: GREEN the new tests and confirm deterministic behaviour.
Prereqs: C3.A/B tests failing as expected.
Exit Criteria: Targeted selectors GREEN; deterministic flag ensures reproducible seeds.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C3.C1 | Implement Trainer wiring to satisfy tests | [ ] | Adjust workflow code until RED tests pass; collect GREEN log (`pytest_workflows_execution_green.log`). |
| C3.C2 | Validate deterministic behaviour | [ ] | Extend tests to assert seeding logic triggered (`torch.use_deterministic_algorithms(True)` or Lightning `deterministic=True`). Document any skipped assertions for GPU-only flags. |
| C3.C3 | Full regression smoke | [ ] | If code touched beyond workflows, run `pytest tests/torch/test_workflows_components.py -vv` (only once) and attach log. |

### Phase C3.D — Documentation & Ledger Updates
Goal: Record decisions, update summaries, and enforce log hygiene.
Prereqs: C3.A–C3.C complete.
Exit Criteria: Summary + implementation plan updated; root `train_debug.log` removed; fix_plan attempt logged.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C3.D1 | Update `phase_c_execution/summary.md` | [ ] | Document trainer/inference wiring, outstanding knobs (scheduler, logger_backend). Include log paths and deterministic validation notes. |
| C3.D2 | Refresh implementation checklist | [ ] | Mark `plans/active/ADR-003-BACKEND-API/implementation.md` C3 row `[x]` with evidence references. |
| C3.D3 | Ledger + hygiene | [ ] | Append Attempt entry in `docs/fix_plan.md` with artifact paths. Move or delete root `train_debug.log` (duplicate) so only timestamped copies remain. |

## Artifact Expectations
- RED log: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T025643Z/phase_c3_workflow_integration/pytest_workflows_execution_red.log`
- GREEN log: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T025643Z/phase_c3_workflow_integration/pytest_workflows_execution_green.log`
- Summary: `.../summary.md` capturing deterministic behaviour + open questions
- Optional: `trainer_kwargs_snapshot.json` capturing Lightning Trainer kwargs for traceability

## Open Risks
- CPU-only environment cannot validate GPU accelerators; tests should skip or mock accordingly.
- Lightning version upgrades may change Trainer kwargs; keep assertions resilient (access `.accelerator` attr over internal dicts).
- Execution knobs not yet exposed via CLI (defer to Phase C4/D) — document placeholders instead of patching CLI now.

## Verification Gate
Phase C3 considered complete when:
1. Trainer/inference helpers accept execution config and pass targeted pytest selectors with logs stored under this directory.
2. Deterministic flag toggles Lightning deterministic mode (asserted in tests).
3. `__all__` export restored and import regression verified.
4. Root-level logs removed; summary + implementation plan + fix ledger updated.
