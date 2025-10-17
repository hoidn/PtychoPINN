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
| D2.C | Implement inference + stitching path | [x] | ✅ 2025-10-17 — Orchestration logic implemented (`reports/2025-10-17T101500Z/phase_d2c_green.md`). `run_cdi_example_torch` now invokes `train_cdi_model_torch` + conditional `_reassemble_cdi_image_torch` stub per TF baseline parity. Test selector: `pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_invokes_training -vv` (1/1 PASSED). Full regression: 191 passed, 0 failed. `_reassemble_cdi_image_torch` stub deferred to Phase D3 (full inference impl). Ensure associated logs (e.g., `train_debug.log`) reside under the same report directory per artifact storage rules. |

---

### Phase D3 — Persistence Bridge
Goal: Produce Lightning/MLflow persistence shim compatible with TensorFlow `ModelManager` archives.
Prereqs: D2 adapters callable end-to-end.
Exit Criteria: PyTorch training run emits archives the reconstructor can consume, with documented schema and restoration procedure.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D3.A | Document persistence delta | [x] | ✅ 2025-10-17 — Callchain evidence captured under `reports/2025-10-17T104700Z/phase_d3_callchain/{static.md,summary.md,tap_points.md,trace_env.json}`. Summary enumerates TensorFlow dual-model archive schema, CONFIG-001 gates, and PyTorch gaps (no params snapshot, MLflow coupling). Ready for D3.B implementation per summary next actions. |
| D3.B | Implement archive writer | [x] | ✅ 2025-10-17 — save_torch_bundle landed in Attempt #51 (`reports/2025-10-17T110500Z/{phase_d3b_summary.md,pytest_archive_structure_red.log,pytest_params_snapshot_red.log,pytest_green.log}`). Archive format matches TensorFlow contract (manifest.dill + dual-model dirs with model.pth + params.dill), CONFIG-001 snapshots verified via new pytest bundle selectors. |
| D3.C | Implement loader & validation | [x] | ✅ 2025-10-17 — Loader tests + CONFIG-001 params restoration implemented (`reports/2025-10-17T113200Z/phase_d3c_summary.md`). Test selectors: `pytest tests/torch/test_model_manager.py::TestLoadTorchBundle -vv` (2/2 PASSED). Full regression: 195 passed, 13 skipped. CONFIG-001 gate (params.cfg.update) verified via round-trip test; error handling validated via malformed archive test. Model reconstruction deferred to Phase D4 (NotImplementedError stub acceptable per TDD incremental approach). |

---

### Phase D4 — Regression Hooks & Tests
Goal: Deliver torch-optional regression coverage for PyTorch orchestration + persistence and prepare the TEST-PYTORCH-001 initiative hand-off.
Prereqs: D2 + D3 functional.
Exit Criteria: Tasks in `phase_d4_regression.md` (D4.A1–D4.C3) completed with artifacts stored per naming guidance and linked from docs/fix_plan.md.

**Reference Plan:** `plans/active/INTEGRATE-PYTORCH-001/phase_d4_regression.md` (phased checklist + artifact map).

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D4.A | Planning alignment & selector map | [x] | ✅ 2025-10-17 — Alignment narrative + selector map captured under `reports/2025-10-17T111700Z/{phase_d4_alignment.md,phase_d4_selector_map.md}`; implementation + ledger links refreshed. |
| D4.B | Author failing regression tests (TDD red) | [x] | ✅ 2025-10-17 — Torch-optional red tests + summary logged at `reports/2025-10-17T112849Z/phase_d4_red_*`; see `phase_d4_regression.md` D4.B table for selectors and failure modes. |
| D4.C | Turn regression tests green & hand off | [ ] | Complete D4.C1–C3. Implement required fixes, capture green logs, and assemble handoff summary (`phase_d4_handoff.md`) feeding TEST-PYTORCH-001 activation. |

---

### Risks & Coordination Notes
- MLflow optionality must be resolved in D1.C to keep CI lightweight.
- Coordination with TEST-PYTORCH-001 is mandatory once D4.A starts; share fixtures and skip rules per `docs/workflows/pytorch.md`.
- Any decision that alters archive semantics requires synchronized spec update (Phase 8 of canonical plan).
