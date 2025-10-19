# Phase D Kickoff — 2025-10-19T095900Z

## Context
- Focus: INTEGRATE-PYTORCH-001-STUBS Phase D2.D (parity verification & documentation)
- C1–C4 are green as of Attempt #28 (`reports/2025-10-19T092448Z/.../summary.md`).
- The PyTorch integration pytest still fails during Lightning checkpoint load (`TypeError: PtychoPINN_Lightning.__init__()` missing configs) per baseline log `reports/2025-10-17T233109Z/.../pytest_integration_baseline.log`.

## Objectives For This Loop
1. **Re-run integration workflow** after C4 fixes to capture the current failure signature under this timestamp:
   - Command: `pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv`
   - Artifact target: `pytest_integration_current.log` (store here).
2. **Document observed failure** in `diagnostics.md` (create alongside this summary) noting stack trace, checkpoint path, and any deltas vs. the 2025-10-17 baseline.
3. **Update plan + ledger** once new evidence is captured; use this summary to drive remediation plan for missing hyperparameters.

## Notes
- Expect the current failure to remain the Lightning checkpoint hyperparameter gap unless `_train_with_lightning` recently started calling `save_hyperparameters()`. If the failure changes, capture full traceback.
- Ensure `pip install -e .[torch]` environment still active; the workflow spins up both training and inference subprocesses.
- Remember to relocate stray `train_debug.log` (currently at repo root) into this directory alongside the new log if it originates from the run.

## Next Steps Checklist
- [ ] Capture fresh integration log (`pytest_integration_current.log`)
- [ ] Draft `diagnostics.md` summarising failure + remediation hypotheses
- [ ] Reference artifacts from docs/fix_plan.md Attempt #29 and update plan checklist D1 once evidence is in place
