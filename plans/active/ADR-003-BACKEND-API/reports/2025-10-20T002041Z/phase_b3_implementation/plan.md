# ADR-003 Phase B3 Implementation Plan — Configuration Factories (2025-10-20T002041Z)

## Context
- Initiative: ADR-003-BACKEND-API — Standardize PyTorch backend API
- Phase Goal: Turn the config factory RED scaffold into a GREEN implementation that produces canonical TensorFlow configs, PyTorch config objects, and execution overrides without duplicating CLI logic.
- Dependencies:
  - `plans/active/ADR-003-BACKEND-API/implementation.md` (Phase B rows)
  - `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/{factory_design.md,override_matrix.md,open_questions.md,summary.md}`
  - `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T000736Z/phase_b2_redfix/{summary.md,pytest_factory_redfix.log}`
  - `specs/ptychodus_api_spec.md` §4 (CONFIG-001, backend lifecycle)
  - `docs/workflows/pytorch.md` §§5–7, 12 (PyTorch workflow + execution knobs)
  - `docs/findings.md` (POLICY-001, CONFIG-001)
- Reporting: Store all B3 artefacts under `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T002041Z/phase_b3_implementation/` using ISO timestamps (e.g., `pytest_factory_green.log`, `summary.md`, `design_delta.md`).
- Test Selector: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -vv`

### Phase B3.A — Training Payload Implementation
Goal: Implement `create_training_payload()` and `populate_legacy_params()` so the training tests in `tests/torch/test_config_factory.py` pass.
Prereqs: RED baseline captured (`pytest_factory_redfix.log`); design decisions in `factory_design.md` §3 and override precedence rules in `override_matrix.md` §4 understood.
Exit Criteria:
- Training factory returns fully-populated `TrainingPayload` dataclass with TF + PT configs and overrides audit trail.
- `populate_legacy_params()` updates `ptycho.params.cfg` with CONFIG-001 ordering and optional `force` behaviour.
- Tests `TestTrainingPayloadStructure`, `TestConfigBridgeTranslation`, `TestLegacyParamsPopulation`, and `TestOverridePrecedence` pass without modifying stubs for inference.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| A1 | Implement canonical TF config assembly | [ ] | Use `TrainingConfig` dataclass copy (`factory_design.md` §3.1). Accept `train_data_file`, `output_dir`, `overrides`. Call `update_legacy_dict(params.cfg, tf_config)` **before** deriving PyTorch configs (CONFIG-001). |
| A2 | Build PyTorch config objects | [ ] | Translate TF config via `ptycho_torch.config_bridge.create_training_configs` (or equivalent). Ensure grid-size tuple/int, epochs/nepochs, and neighbor_count conversions match tests. |
| A3 | Apply override precedence | [ ] | Merge runtime overrides using hierarchy defined in `override_matrix.md` §4 (override dict → execution config → CLI defaults → PT defaults → TF defaults). Record applied values in `overrides_applied`. |
| A4 | Implement `populate_legacy_params()` helper | [ ] | Thin wrapper around `update_legacy_dict`. Honour `force` flag (reapply even if params already populated). Add logging hook if needed (`factory_design.md` §3.4). |
| A5 | Capture GREEN diagnostics | [ ] | Run selector (above) focusing on training tests (`-k training or config_bridge or legacy_params or override`). Store log as `pytest_factory_training_green.log`. Summarise outcomes + any follow-up in `summary.md`. |

### Phase B3.B — Inference Payload & Helpers
Goal: Implement `create_inference_payload()` and `infer_probe_size()` to satisfy inference + validation tests.
Prereqs: Phase B3.A complete; checkpoint validation rules from `factory_design.md` §3.2 and `override_matrix.md` §5 reviewed.
Exit Criteria:
- Inference payload returns TF + PT inference configs and overrides audit trail.
- `infer_probe_size()` reads NPZ metadata, applies fallback + FORMAT-001 guardrails.
- Validation tests for missing files/checkpoints raise the documented exceptions.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B1 | Implement probe size inference | [ ] | Load NPZ lazily (`numpy.load`) and read `probeGuess`. Handle missing file (fallback N=64) and legacy axis order per FORMAT-001. Unit guidance in `factory_design.md` §3.3. |
| B2 | Assemble inference configs | [ ] | Mirror training flow using `config_bridge` helpers. Ensure `wts.h5.zip` exists in `model_path`; raise `ValueError` if absent (`factory_design.md` §5.4). Populate overrides dict with applied values. |
| B3 | Validate inputs and errors | [ ] | Enforce required `n_groups` override, file existence checks, and checkpoint validation. Provide descriptive error messages referencing plan section for quick diagnosis. |
| B4 | Capture GREEN diagnostics | [ ] | Re-run selector without `-k` filter, store output as `pytest_factory_green.log`. Update `summary.md` with runtime + key assertions. |

### Phase B3.C — Parity & Integration Hooks
Goal: Ensure factories integrate cleanly with workflows and config bridge tests.
Prereqs: Phase B3.A/B GREEN; design doc §4 (integration flow) and `open_questions.md` decisions reviewed.
Exit Criteria:
- `tests/torch/test_config_factory.py` all passing.
- `tests/torch/test_config_bridge.py` updated (if necessary) to consume factories or assert parity.
- Workflow/CLI refactor deferred to Phase C/D but add TODO hooks documenting upcoming changes.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Update summary & artefact log | [ ] | Expand this plan’s `summary.md` with exit-criteria checklist, runtime comparison (RED 2.1s → GREEN ?s), and CONFIG-001 validation evidence. |
| C2 | Review config bridge tests | [ ] | Determine whether additional assertions/workflow harness is required now or in Phase C. If new coverage is needed immediately, document scope and update plan accordingly. |
| C3 | Prep workflow handoff notes | [ ] | Draft TODOs/section in `summary.md` outlining touchpoints for Phase C (`_train_with_lightning`, CLI scripts). Reference relevant file:line anchors for engineer handoff. |

### How-To Map (authoritative commands & artefact rules)
- Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` in shell before running pytest.
- Use `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -vv` for full run; targeted filters (`-k training`, `-k inference`) acceptable during implementation but capture final GREEN log without filters.
- Store logs and supporting markdown under `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T002041Z/phase_b3_implementation/`.
- Record config diffs (e.g., pretty-print `payload.tf_training_config` → json) as needed in `diagnostics/` subfolder; reference them in `summary.md`.
- Maintain CONFIG-001 ordering: call `update_legacy_dict` before data loading helpers (`RawData.from_file`).
- Respect POLICY-001: do not introduce torch-optional pathways; raise actionable errors if torch unavailable.

### Risks & Mitigations
- **PyTorchExecutionConfig placeholder:** Document that execution_config may be `None` until Phase C1 defines dataclass. Leave TODO with pointer to `open_questions.md`.
- **params.cfg global state:** Tests clear `ptycho.params.cfg`; ensure factories do not rely on pre-existing state.
- **File I/O performance:** NPZ fixtures already small; avoid loading entire file multiple times. Use context manager for `numpy.load` (allow_pickle=False).
- **Error messaging drift:** Keep exceptions aligned with test expectations (see comments in `tests/torch/test_config_factory.py` for exact strings).

### Verification Checklist (to be completed at GREEN)
- [ ] `tests/torch/test_config_factory.py` passes (GREEN log stored).
- [ ] `summary.md` updated with exit criteria + runtime delta.
- [ ] docs/fix_plan.md Attempt appended referencing GREEN artefacts.
- [ ] Implementation plan Phase B3 rows updated to `[x]` (with artefact links).
- [ ] Follow-up tasks for Phase C/D documented (workflow integration, CLI refactor).
