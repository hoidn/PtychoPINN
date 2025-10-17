# Phase E — Ptychodus Integration & Parity Validation

## Context
- Initiative: INTEGRATE-PYTORCH-001
- Phase Goal: Enable Ptychodus to select and execute the PyTorch backend with parity safeguards that match the TensorFlow reconstructor contract defined in `specs/ptychodus_api_spec.md` §4.
- Dependencies: Phase D handoff (`reports/2025-10-17T121930Z/phase_d4c_summary.md`), canonical initiative plan (`plans/ptychodus_pytorch_integration_plan.md` Phase 6–8), PyTorch workflow guide (`docs/workflows/pytorch.md`), TEST-PYTORCH-001 test strategy (`plans/pytorch_integration_test_plan.md`), CONFIG-001 finding (docs/findings.md ID CONFIG-001).
- Artifact Storage: Capture all Phase E deliverables under `plans/active/INTEGRATE-PYTORCH-001/reports/<ISO8601>/phase_e_*` with descriptive filenames (e.g., `phase_e_callchain.md`, `phase_e_selector_map.md`, `phase_e_parity_summary.md`). Reference each artifact in docs/fix_plan.md.

---

### Phase E1 — Backend Selection & Orchestration Bridge
Goal: Design and implement the reconstructor-selection handshake so Ptychodus can delegate to PyTorch workflows without breaking the existing TensorFlow path.
Prereqs: Review spec §4.1–4.6, TensorFlow baseline call flow (`ptycho/workflows/components.py`, `ptycho/model_manager.py`), and Phase D4 handoff notes.
Exit Criteria: Documented callchain diff, TDD red tests in the Ptychodus repo covering backend selection, and implementation plan for wiring PtychoPINN reconstructor hooks to `ptycho_torch/workflows/components.py`.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| E1.A | Map current Ptychodus reconstructor callchain | [ ] | Run `prompts/callchain.md` with `analysis_question="How does Ptychodus select and invoke the TensorFlow backend?"`; capture output under `phase_e_callchain/static.md`. Anchor entries to `ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py` and `ptycho/workflows/components.py`. |
| E1.B | Author backend-selection failing tests | [ ] | In the Ptychodus repo, add pytest (torch-optional) cases asserting: (1) config flag selects PyTorch backend; (2) fallback to TensorFlow remains default; (3) `update_legacy_dict` triggered before workflow dispatch (CONFIG-001). Store red logs at `phase_e_red_backend_selection.log`. |
| E1.C | Draft implementation blueprint | [ ] | Produce `phase_e_backend_design.md` describing the minimal changes to `PtychoPINNReconstructorLibrary` (spec §4.1) and CLI/config plumbing to surface a `backend='pytorch'` selector. Include decision rules for guarding imports and preserving legacy behaviour. |

---

### Phase E2 — Integration Regression & Parity Harness
Goal: Provide runnable parity tests that exercise the PyTorch backend end-to-end alongside TensorFlow.
Prereqs: E1 design complete, TEST-PYTORCH-001 coordination notes reviewed (`phase_d4_alignment.md`, `phase_d4_selector_map.md`).
Exit Criteria: Torch-optional failing tests documenting the integration gap, green implementation wiring PyTorch workflows, and parity report comparing TensorFlow vs PyTorch outputs on shared fixtures.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| E2.A | Align with TEST-PYTORCH-001 fixtures | [ ] | Meet with TEST-PYTORCH-001 plan owners; capture fixture availability + runtime constraints in `phase_e_fixture_sync.md`. Reference `plans/pytorch_integration_test_plan.md` §Prerequisites. |
| E2.B | Author integration red tests | [ ] | Extend `tests/torch/test_workflows_components.py` or new `tests/torch/test_integration_workflow_torch.py` with subprocess-style tests that mirror `tests/test_integration_workflow.py`. Record red logs under `phase_e_red_integration.log`; ensure torch-optional skip rules documented. |
| E2.C | Implement integration wiring | [ ] | Once red tests exist, wire Ptychodus configuration to call `run_cdi_example_torch` / `load_inference_bundle_torch`. Document code changes and targeted green selectors in `phase_e_integration_green.md`. |
| E2.D | Capture parity metrics | [ ] | Execute both TensorFlow and PyTorch integration tests; summarize runtime, key metrics, and archive diffs in `phase_e_parity_summary.md`. Include commands: `pytest tests/test_integration_workflow.py -k tf` and torch counterpart selector defined in selector map. |

---

### Phase E3 — Documentation, Spec Sync, and Handoff
Goal: Ensure documentation, specs, and downstream initiatives reflect the new backend, and hand off to TEST-PYTORCH-001/production teams.
Prereqs: E1–E2 tasks completed with successful regression runs.
Exit Criteria: Updated documentation and spec excerpts, ledger entries recorded, and follow-up actions queued for TEST-PYTORCH-001.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| E3.A | Update workflow documentation | [ ] | Revise `docs/workflows/pytorch.md` and author Ptychodus user-facing notes (target `architectural_context.md` or new doc). Store diff summary in `phase_e_docs_update.md`; highlight backend toggle instructions. |
| E3.B | Sync specs & findings | [ ] | Draft spec amendments for §4.1–4.6 describing dual-backend behaviour; log CONFIG-XXX follow-up in `docs/findings.md`. Capture working copy in `phase_e_spec_patch.md`. |
| E3.C | Prepare TEST-PYTORCH-001 handoff | [ ] | Summarize remaining risks, test selectors, and ownership matrix in `phase_e_handoff.md`. Include checklist verifying that TEST-PYTORCH-001 can bootstrap without re-discovering Phase E decisions. |

---

### Exit Checklist
- [ ] Phase E1 artifacts stored and referenced; backend-selection tests failing as expected before implementation.
- [ ] Phase E2 integration tests capture PyTorch gaps then pass with wiring in place; parity report archived.
- [ ] Documentation, spec updates, and TEST-PYTORCH-001 handoff complete with ledger entries and risk notes.
