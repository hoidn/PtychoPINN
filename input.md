Summary: Fix Lightning regression fixture so TestTrainWithLightningRed runs fully green
Mode: TDD
Focus: INTEGRATE-PYTORCH-001-STUBS — Finish PyTorch workflow stubs deferred from Phase D2
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T171500Z/phase_d2_completion/{summary.md,pytest_train_green.log}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS B4 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Update TestTrainWithLightningRed fixture to use a LightningModule-compatible stub (tests: none)
2. INTEGRATE-PYTORCH-001-STUBS B4 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Run pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T171500Z/phase_d2_completion/pytest_train_green.log (tests: targeted)
3. INTEGRATE-PYTORCH-001-STUBS B4 checklist @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Update summary.md, mark checklist/log attempts, and record docs/fix_plan entry (tests: none)

If Blocked: Capture the failing traceback in the tee log above, leave B4 as [P], append a blocker note in summary.md with hypotheses, and log the obstruction in docs/fix_plan.md Attempts before exiting the loop.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:40 — Phase B4 exit criteria require the Lightning regression suite to turn green with logs captured under the new timestamp.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T171500Z/phase_d2_completion/summary.md — Supervisor notes explain the LightningModule inheritance fix needed to unblock the remaining test.
- specs/ptychodus_api_spec.md:191 — Reconstructor contract mandates Lightning training return persistence-ready handles, so keeping the regression suite green is non-negotiable before Phase C.
- docs/workflows/pytorch.md:114 — Workflow guide details deterministic Lightning configuration that the test suite safeguards; a green run proves B2 implementation remains compliant.

How-To Map:
- Edit tests/torch/test_workflows_components.py:820 to wrap the monkeypatched constructor in a stub class inheriting from lightning.pytorch.core.LightningModule, implementing minimal `training_step` and `configure_optimizers` while preserving the constructor spy.
- Ensure the stub returns deterministic tensors (e.g., torch.zeros) so Trainer completes immediately; retain assertions that all four config objects were captured.
- Command: `pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T171500Z/phase_d2_completion/pytest_train_green.log`
- After the run, append the pass/fail summary to summary.md in the artifact directory, update B4 state to `[x]`, and add Attempt #20 with artifact references to docs/fix_plan.md.

Pitfalls To Avoid:
- Do not modify `_train_with_lightning`; focus solely on test harness adjustments.
- Keep monkeypatch cleanup automatic via pytest fixtures; no global mutation of Lightning classes.
- Avoid importing lightning at module scope outside tests to preserve optionality in production modules.
- Ensure tee captures the entire pytest output; rerun if the log is missing or truncated.
- Preserve existing assertions verifying all four config objects; the stub must still validate constructor arity.
- Do not delete prior artifact directories or logs; keep historical evidence intact.

Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:40
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T171500Z/phase_d2_completion/summary.md
- tests/torch/test_workflows_components.py:713
- specs/ptychodus_api_spec.md:191
- docs/workflows/pytorch.md:114

Next Up: C1 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md (design `_reassemble_cdi_image_torch`) once B4 evidence is green.
