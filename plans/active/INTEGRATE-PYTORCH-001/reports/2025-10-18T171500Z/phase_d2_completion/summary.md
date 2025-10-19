# Phase B.B4 Lightning Regression Prep — Supervisor Notes (2025-10-18T171500Z)

## Context
- Initiative: INTEGRATE-PYTORCH-001 — PyTorch backend integration
- Phase: D2.B4 (turn Lightning regression suite fully green)
- Purpose: Document supervisor observations ahead of B4 implementation loop so engineer can resolve the remaining test failure and capture fresh evidence.

## Key Findings
1. `_train_with_lightning` implementation is complete and verified (Attempts #10–#19); targeted selector shows stable pass/fail profile (2/3 passing) with fixture-induced failure.
2. `TestTrainWithLightningRed.test_train_with_lightning_instantiates_module` still fails because the monkeypatched stub returned by the test is not a subclass of `LightningModule`. Lightning validates module types before `trainer.fit`, causing the RuntimeError.
3. Exit criteria for Phase B.B4 require all TestTrainWithLightning* cases to pass and the green log to be captured under this timestamp. The failure must be resolved by adjusting the test harness (provide a Lightning-compliant stub or exercise the real module) rather than changing production code behaviour.

## Supervisor Guidance for Attempt #20
- Update `tests/torch/test_workflows_components.py::TestTrainWithLightningRed.test_train_with_lightning_instantiates_module` so the monkeypatched object inherits from `lightning.pytorch.core.LightningModule` and implements the tiny interface (`training_step`, `configure_optimizers`) required by Trainer.
- Ensure the stub still records constructor arguments to validate all four config objects; keep assertions intact.
- After adjusting the fixture, rerun:
  ```bash
  pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv \
    | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T171500Z/phase_d2_completion/pytest_train_green.log
  ```
- If green, update `phase_d2_completion.md` checklist B4 and append results to this summary (expected runtime ≤6s). If failure persists, capture traceback in `pytest_train_green.log` and log blocker in docs/fix_plan.md.

## Next Steps After B4
- Phase C (stitching implementation) can start once the Lightning regression suite is fully green.
- Integration selector `pytest tests/torch/test_integration_workflow_torch.py -k train_save -vv` should be queued after B4 to ensure persistence path remains stable.
