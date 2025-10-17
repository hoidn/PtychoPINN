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
| D1.A | Capture current TensorFlow orchestration callchain | [ ] | Run `prompts/callchain.md` against `ptycho.workflows.components.run_cdi_example` using minimal ROI (single Fly sample). Save under `reports/<timestamp>/phase_d_callchain.md`. Highlight entry/exit signatures that must mirror in PyTorch. |
| D1.B | Inventory PyTorch workflow assets | [ ] | Summarize reusable modules (`ptycho_torch.train.main`, `PtychoDataModule`, `train_utils`) vs gaps (missing inference CLI) in `reports/<timestamp>/phase_d_asset_inventory.md`. Tie findings back to `plans/ptychodus_pytorch_integration_plan.md` Delta-2. |
| D1.C | Decide orchestration surface | [ ] | Author decision log (`reports/<timestamp>/phase_d_decision.md`) comparing (A) wrapping `ptycho_torch/api/*` vs (B) building thin shims around low-level modules. Include pros/cons, Lightning dependency policy, and persistence implications. |

---

### Phase D2 — Orchestration Adapter Implementation
Goal: Implement PyTorch equivalents of `run_cdi_example` + helper routines that consume dataclass configs and update `params.cfg`.
Prereqs: Completed D1 design artifacts; config bridge + data adapters green.
Exit Criteria: New module(s) providing `train_cdi_model_torch`, `run_cdi_example_torch`, and helper functions callable from Ptychodus with no global-state drift; smoke-tested via targeted unit tests.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D2.A | Scaffold orchestration module | [ ] | Create `ptycho_torch/workflows/components.py` (or equivalent) exposing mirrored signatures. Enforce torch-optional import guards and call `update_legacy_dict` at entry. Document structure in `reports/<timestamp>/phase_d2_scaffold.md`. |
| D2.B | Implement training path | [ ] | Wrap Lightning trainer to accept `TrainingConfig` + NPZ paths, respecting overrides decided in D1.C. Ensure deterministic seeds + optional MLflow disable flag. Record validation in `reports/<timestamp>/phase_d2_training.md`. |
| D2.C | Implement inference + stitching path | [ ] | Provide function aligning with spec §4.5 (save outputs, optional stitching). Reuse RawDataTorch + PtychoDataContainerTorch. Log parity checks vs TensorFlow metrics in `reports/<timestamp>/phase_d2_inference.md`. |

---

### Phase D3 — Persistence Bridge
Goal: Produce Lightning/MLflow persistence shim compatible with TensorFlow `ModelManager` archives.
Prereqs: D2 adapters callable end-to-end.
Exit Criteria: PyTorch training run emits archives the reconstructor can consume, with documented schema and restoration procedure.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D3.A | Document persistence delta | [ ] | Compare TensorFlow `ModelManager` outputs (`wts.h5.zip`) to PyTorch checkpoints. Capture schema diff in `reports/<timestamp>/phase_d3_persistence_gap.md`. |
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
