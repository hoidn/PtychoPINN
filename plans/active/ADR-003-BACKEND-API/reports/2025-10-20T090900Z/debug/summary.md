# Phase C4.D Debug Notes — 2025-10-20T090900Z

## Focus
- Initiative: ADR-003-BACKEND-API
- Phase: C4.D (Bundle loader + CLI integration)
- Problem: Integration selector now reaches inference but crashes with `AttributeError: 'dict' object has no attribute 'eval'` during bundle load in `ptycho_torch/inference.py`.

## Evidence Reviewed
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/pytest_integration_phase_a.log` — confirms CLI now loads `wts.h5.zip` but retrieves a dict for `'diffraction_to_obj'`.
- Source inspection: `ptycho_torch/model_manager.py::load_torch_bundle` returns a `PtychoPINN_Lightning` instance per manifest entry; targeted test `TestLoadTorchBundle::test_reconstructs_models_from_bundle` (GREEN) verifies contract.
- Manual CLI attempt (`manual_train.log`) shows PyTorch training CLI refuses to run without generated fixture NPZ; indicates our local repo lacks `minimal_dataset_v1.npz` (fixture generation required).

## Hypotheses
1. **Saved bundle lacks real diffraction model.** Training workflows may still stash sentinel dictionaries under `'diffraction_to_obj'`, so loader reconstructs module with random weights and CLI receives sentinel. Needs confirmation by inspecting actual `model.pth`.
2. **Plan/Repo divergence.** Logs reference `/home/ollie/Documents/PtychoPINN2`; ensure Ralph is operating in the same repo revision where loader changes landed—stale environment could still emit dict payloads.

## Recommended Next Step
- Re-run the integration selector, then inspect the generated bundle (`training_outputs/wts.h5.zip`) with `torch.load` to confirm the type stored under `diffraction_to_obj/model.pth`. Capture findings under this report hub for follow-up.
