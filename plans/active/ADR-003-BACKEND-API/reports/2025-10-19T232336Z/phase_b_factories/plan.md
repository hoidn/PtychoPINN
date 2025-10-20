# ADR-003 Phase B Execution Plan — Configuration Factories (2025-10-19T232336Z)

## Context
- Initiative: ADR-003-BACKEND-API — Standardize PyTorch backend API
- Phase Goal: Introduce centralized factory helpers that translate canonical TensorFlow configs + PyTorch execution overrides into the objects consumed by the backend, eliminating duplicated wiring in CLI/workflow entry points.
- Dependencies: `specs/ptychodus_api_spec.md` §4 (reconstructor lifecycle), `docs/workflows/pytorch.md` §§5–12, `plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md`, `plans/active/TEST-PYTORCH-001/implementation.md`, canonical config dataclasses (`ptycho/config/config.py`), translation adapters (`ptycho_torch/config_bridge.py`).
- Artifact Discipline: Store Phase B1 artefacts under `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/`. Subsequent B2/B3 execution logs and summaries move to `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T234600Z/phase_b2_skeleton/`. Provide per-task `summary.md` updates and capture pytest logs via `tee`.

### Phase B1 — Factory Design Blueprint
Goal: Define the factory architecture, including inputs, outputs, override strategy, and the execution config surface. Deliver design documentation before touching code.
Prereqs: Phase A inventories (`cli_inventory.md`, `execution_knobs.md`, `overlap_notes.md`) reviewed; confirm POLICY-001/FORMAT-001 constraints.
Exit Criteria:
- Design doc (`factory_design.md`) detailing module structure, exported functions, and how they interact with `config_bridge` + `update_legacy_dict`.
- Override matrix (`override_matrix.md`) enumerating which fields come from canonical configs vs PyTorchExecutionConfig vs runtime overrides.
- Decision log documenting spec/ADR updates required (e.g., new §6 for PyTorch execution knobs).

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B1.a | Author high-level architecture narrative | [x] | ✅ `factory_design.md` (420L) describes module layout, exported helpers, CLI/workflow call sites, CONFIG-001 checkpoints, and TDD strategy. Stored under `reports/2025-10-19T232336Z/phase_b_factories/`. |
| B1.b | Build override matrix | [x] | ✅ `override_matrix.md` (584L) maps 80+ fields across canonical + execution configs with precedence notes and file:line citations. Same artifact hub as B1.a. |
| B1.c | Capture open questions & spec deltas | [x] | ✅ `open_questions.md` documents decisions (Q1 Option A resolved 2025-10-19T234458Z), spec updates, and governance follow-ups. Summary captured in `reports/2025-10-19T232336Z/phase_b_factories/summary.md`. |

### Phase B2 — Factory Module & TDD Scaffold
Goal: Establish the factory module skeleton and RED tests encoding the desired behaviour before implementation.
Prereqs: Phase B1 documentation approved by supervisor; confirm test strategy with TEST-PYTORCH-001 runtime guardrails (≤90s).
Exit Criteria:
- Factory module skeleton (`ptycho_torch/config_factory.py`) added with NotImplementedError placeholders.
- Pytest suite capturing expected behaviour fails with clear assertion/NotImplemented errors (`pytest_factory_red.log`).
- Plan + fix_plan updated with RED evidence and artifact paths.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B2.a | Create factory module skeleton | [x] | ✅ `ptycho_torch/config_factory.py` skeleton (367L) committed; NotImplementedError stubs document Phase B3 follow-ups. See `reports/2025-10-19T234600Z/phase_b2_skeleton/`. |
| B2.b | Author failing pytest coverage | [x] | ✅ `tests/torch/test_config_factory.py` (463L) encodes 19 RED cases. Guards removed 2025-10-20 so selector now fails on NotImplementedError; logs at `reports/2025-10-20T000736Z/phase_b2_redfix/pytest_factory_redfix.log`. |
| B2.c | Update implementation plan & ledger | [x] | ✅ Implementation plan + docs/fix_plan updated through Attempt #7; RED evidence summarised in `reports/2025-10-20T000736Z/phase_b2_redfix/summary.md`. |

### Phase B3 — Factory Implementation & Bridge Integration
Goal: Implement factories, wire workflows/CLI to use them, and update config bridge tests to assert new behaviour.
Prereqs: B2 RED tests in place; design doc resolved outstanding questions (or documented blockers).
Exit Criteria:
- Factories produce canonical configs + execution config objects without duplicating CLI logic.
- Workflows (`ptycho_torch/workflows/components.py`) and CLI (`ptycho_torch/train.py`, `ptycho_torch/inference.py`) delegate to factories.
- Updated pytest suite GREEN with logs stored under `pytest_factory_green.log` and config bridge parity tests adjusted.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B3.a | Implement factory logic | [ ] | Fill in `config_factory.py`: consume modern `TrainingConfig`/`InferenceConfig`, apply overrides from CLI, instantiate PyTorch dataclasses, and produce execution config. Ensure CONFIG-001 compliance (call update_legacy_dict within factory or explicit call site). |
| B3.b | Integrate factories into workflows/CLI | [ ] | Refactor `_train_with_lightning`, `_reassemble_cdi_image_torch`, `cli_main()` to call factory helpers rather than constructing configs manually. Maintain POLICY-001 guardrails (`update_legacy_dict` before data loading). |
| B3.c | Extend tests & capture GREEN evidence | [ ] | Update `tests/torch/test_config_bridge.py` to assert factories supply the bridge, add new test modules if needed, and rerun targeted selectors. Capture log: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -vv | tee .../pytest_factory_green.log`. Promote plan rows to `[x]`, update docs/fix_plan Attempt, refresh `summary.md`. |

### Reporting & Governance
- Maintain running `summary.md` in the phase directory capturing progress, exit-criteria validation, and open questions.
- Link artefacts from docs/fix_plan Attempts and `plans/active/ADR-003-BACKEND-API/implementation.md` (Phase B rows).
- Coordinate with TEST-PYTORCH-001 maintainers if factory changes affect integration runtime or fixtures.
- Notify specs team when PyTorchExecutionConfig schema stabilizes (update ADR + spec per B1.c outcomes).

## Verification Checklist (Supervisor)
- [x] `factory_design.md`, `override_matrix.md`, and (if needed) `open_questions.md` exist with citations.
- [x] RED pytest log stored under timestamped phase directory.
- [x] Implementation + fix plan updated in same loop as status transitions.
- [ ] GREEN pytest log demonstrates factories + workflows functioning within runtime guardrails.
- [ ] Summary notes document compliance with POLICY-001, CONFIG-001, and DATA-001.
