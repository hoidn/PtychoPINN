# Phase D — PyTorch Workflow & Persistence Plan

## Context
- Initiative: INTEGRATE-PYTORCH-001 (PyTorch backend integration with Ptychodus)
- Phase Goal: Deliver PyTorch orchestration and persistence surfaces that satisfy the reconstructor lifecycle defined in `specs/ptychodus_api_spec.md` §4.
- Dependencies: Phase B config bridge (complete), Phase C data pipeline (complete), canonical strategy in `plans/ptychodus_pytorch_integration_plan.md` Phase 4–6, and TEST-PYTORCH-001 fixture roadmap.
- Artifact Storage: Capture design notes, decision logs, and validation evidence under `plans/active/INTEGRATE-PYTORCH-001/reports/<timestamp>/phase_d_*`. Reference each artifact from docs/fix_plan.md Attempts History.

---

### Phase D1 — Workflow Entry Point Design
Goal: Specify how PyTorch workflows map onto the reconstructor contract before writing adapters.
Prereqs: Review `ptycho/workflows/components.py::run_cdi_example`, `ptycho/model_manager.py`, `ptycho_torch/train.py`, `ptycho_torch/inference.py`, and `docs/workflows/pytorch.md`.
Exit Criteria: Approved design note documenting call flow, module responsibilities, and backend-selection handshake, plus a decision on API layer reuse vs low-level orchestration.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1.A | Capture current TensorFlow orchestration callchain | [x] | Completed 2025-10-17 — see `reports/2025-10-17T085431Z/phase_d_callchain.md` for annotated call flow and config bridge touchpoints. |
| D1.B | Inventory PyTorch workflow assets | [x] | Completed 2025-10-17 — inventory and reuse scores captured in `reports/2025-10-17T085431Z/phase_d_asset_inventory.md`. |
| D1.C | Decide orchestration surface | [x] | Completed 2025-10-17 — decision documented in `reports/2025-10-17T085431Z/phase_d_decision.md` selecting Option B (orchestration shims). |

---

### Phase D2 — Orchestration Adapter Implementation
Goal: Implement PyTorch equivalents of `run_cdi_example` + helper routines that consume dataclass configs and update `params.cfg`.
Prereqs: Completed D1 design artifacts; config bridge + data adapters green.
Exit Criteria: New module(s) providing `train_cdi_model_torch`, `run_cdi_example_torch`, and helper functions callable from Ptychodus with no global-state drift; smoke-tested via targeted unit tests.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D2.A | Scaffold orchestration module | [x] | Completed 2025-10-17 — see `reports/2025-10-17T091450Z/phase_d2_scaffold.md` for torch-optional scaffold implementation with CONFIG-001 parity guard. Test selector: `pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsScaffold::test_run_cdi_example_calls_update_legacy_dict -vv` (1/1 PASSED). Git commit: 10be6913. |
| D2.B | Implement training path | [x] | ✅ 2025-10-17 — Stub implementation landed (`reports/2025-10-17T094500Z/phase_d2_training.md`). `_ensure_container` + `train_cdi_model_torch` now normalize inputs and delegate to Lightning stub per TDD plan; full Trainer integration + probe handling deferred (tracked in Next Steps). |
| D2.C | Implement inference + stitching path | [x] | ✅ 2025-10-17 — Orchestration logic implemented (`reports/2025-10-17T101500Z/phase_d2c_green.md`). `run_cdi_example_torch` now invokes `train_cdi_model_torch` + conditional `_reassemble_cdi_image_torch` stub per TF baseline parity. Test selector: `pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_invokes_training -vv` (1/1 PASSED). Full regression: 191 passed, 0 failed. `_reassemble_cdi_image_torch` stub deferred to Phase D3 (full inference impl). Git commit: pending. |

---

### Phase D3 — Persistence Bridge
Goal: Produce Lightning/MLflow persistence shim compatible with TensorFlow `ModelManager` archives.
Prereqs: D2 adapters callable end-to-end.
Exit Criteria: PyTorch training run emits archives the reconstructor can consume, with documented schema and restoration procedure.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D3.A | Document persistence delta | [ ] | Run callchain per `prompts/callchain.md` to map TensorFlow training → ModelManager.save_multiple_models → wts.h5.zip → load_inference_bundle, then contrast with PyTorch Lightning checkpoint/MLflow outputs. Capture artifacts under `reports/<timestamp>/phase_d3_callchain/` (static.md, summary.md) outlining schema/touchpoints. |
| D3.B | Implement archive writer | [ ] | Either extend `ModelManager` or add `TorchModelManager` to package Lightning checkpoint + params dump into `.h5.zip`. Respect CONFIG-001 by bundling `params.cfg`. Record implementation summary + sample archive tree in `reports/<timestamp>/phase_d3_writer.md`. |
| D3.C | Implement loader & validation | [ ] | Provide load routine returning ready-to-run Lightning module, verifying `update_legacy_dict` call. Document round-trip test + checksum comparison in `reports/<timestamp>/phase_d3_loader.md`. |

---

### Phase D4 — Regression Hooks & Tests
Goal: Add automated coverage ensuring PyTorch orchestration stays parity-aligned and torch-optional.
Prereqs: D2 + D3 functional.
Exit Criteria: Targeted pytest selectors codified in `plans/pytorch_integration_test_plan.md`, capturing train→save→load workflow and backend toggle smoke tests.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D4.A | Update test plan linkage | [ ] | Activate `plans/active/TEST-PYTORCH-001` with checklists referencing D2/D3 outputs. Provide pointer from fix ledger Attempt log. |
| D4.B | Author failing integration test | [ ] | Extend `tests/torch/test_integration_workflow.py` (or new file) in red phase using minimal fixture; ensure skip rules keep torch-optional behaviour. Record selector + failure log under `reports/<timestamp>/phase_d4_red.md`. |
| D4.C | Turn tests green | [ ] | After D2/D3 finalize, update test to pass, capturing pytest logs + artifact manifests under `reports/<timestamp>/phase_d4_green.md`. |

---

### Risks & Coordination Notes
- MLflow optionality must be resolved in D1.C to keep CI lightweight.
- Coordination with TEST-PYTORCH-001 is mandatory once D4.A starts; share fixtures and skip rules per `docs/workflows/pytorch.md`.
- Any decision that alters archive semantics requires synchronized spec update (Phase 8 of canonical plan).
