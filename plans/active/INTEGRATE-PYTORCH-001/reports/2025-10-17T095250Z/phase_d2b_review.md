# Phase D2.B — Review & Hand-off Notes (2025-10-17T095250Z)

**Initiative:** INTEGRATE-PYTORCH-001  
**Focus:** Phase D2.B (training orchestration parity)  
**Mode:** Parity / Review  

---

## Status Check
- Verified Attempt #45 artifacts (`reports/2025-10-17T094500Z/`) and updated `phase_d_workflow.md` to mark D2.B COMPLETE.  
- `_ensure_container` + `train_cdi_model_torch` provide full normalization + delegation scaffolding; `_train_with_lightning` remains a stub returning placeholder history (per documented scope).  
- Targeted selector still green: `pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_train_cdi_model_torch_invokes_lightning -vv`.  
- Full regression (minus known legacy skips) previously captured at Attempt #45; no additional runtime executed this loop.

## Gaps / Follow-through
- Lightning trainer still stubbed — replace with real orchestration when ready to consume torch runtime (must remain torch-optional at import time).  
- Probe initialization strategy unresolved (needs coordination with `ptycho.probe` vs PyTorch-local equivalent).  
- MLflow disable/telemetry toggles required for CI friendliness (documented in `phase_d2_training_analysis.md`).

## Next Actions (Phase D2.C focus)
1. **Implement inference+stitching path** (`run_cdi_example_torch`, `reassemble_cdi_image_torch`, `load_inference_bundle_torch` placeholders) following spec §4.5 and TF baseline `ptycho/workflows/components.py:615-710`.  
   - Reuse `_ensure_container` for test data normalization.  
   - Mirror returned result keys (`reconstructed_obj`, `recon_amp`, `recon_phase`).
2. **Author red-phase pytest** in `tests/torch/test_workflows_components.py` (e.g., `TestWorkflowsComponentsRun::test_run_cdi_example_invokes_training_and_optional_stitching`) with skip guard respecting `TORCH_AVAILABLE`.  
   - Assert `update_legacy_dict` already covered (existing test) then validate delegation to `train_cdi_model_torch` + optional inference helper.  
   - Selector: `pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_invokes_training -vv` (name TBD at authoring time).
3. **Scope persistence hand-off** — capture acceptance criteria for Phase D3 once inference path returns tensors + checkpoint handle.  
4. **Document MLflow/Lightning decisions** — when replacing stub, ensure config-driven toggles recorded in plan + findings if behaviour diverges from TF backend.

## References
- `specs/ptychodus_api_spec.md` §4 (reconstructor lifecycle expectations)  
- `plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md` (updated D2 table)  
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T093500Z/phase_d2_training_analysis.md` (baseline analysis)  
- `ptycho/workflows/components.py:535-710` (TensorFlow orchestration reference)  
- `tests/torch/test_workflows_components.py` (existing scaffolding + training parity test)

---

*Recorded by galph (supervisor) to support next engineering loop.*
