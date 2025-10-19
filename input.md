Summary: Capture the post-C4 integration-test failure and prep Phase D diagnostics
Mode: TDD
Focus: INTEGRATE-PYTORCH-001-STUBS — Finish PyTorch workflow stubs deferred from Phase D2
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T095900Z/phase_d2_completion/{summary.md,pytest_integration_current.log}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS D1 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Run `pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T095900Z/phase_d2_completion/pytest_integration_current.log` (tests: targeted)
2. INTEGRATE-PYTORCH-001-STUBS D1 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Create `diagnostics.md` in the same report directory summarising the failure signature (stack trace snippet, checkpoint path, differences vs. 2025-10-17 baseline) and relocate any new `train_debug.log` into that directory (tests: none)
3. INTEGRATE-PYTORCH-001-STUBS D3 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Update `summary.md` checklist, keep D1 `[ ]` until the test passes, and record Attempt #30 in docs/fix_plan.md with the new log and diagnostics links (tests: none)

If Blocked: Store whatever portion of the pytest output you captured into `pytest_integration_current.log`, jot the failure mode in `diagnostics.md`, and note the blocker plus log path in docs/fix_plan.md before pausing.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:65 — D1 now demands a fresh integration run + diagnostics before remediation.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T095900Z/phase_d2_completion/summary.md:8 — Defines the immediate objectives and artifact expectations for this timestamp.
- specs/ptychodus_api_spec.md:205 — Persistence contract requires loading checkpoints without extra constructor args; present failure violates this spec clause.
- docs/workflows/pytorch.md:260 — Troubleshooting section documents the current TypeError and reminds us to refresh checkpoints with hyperparameters.
- docs/findings.md:8 — POLICY-001 keeps PyTorch mandatory; ensure the integration run continues to honour torch-enabled pathways.

How-To Map:
- Run the targeted pytest command from repo root: `pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T095900Z/phase_d2_completion/pytest_integration_current.log`
- After the run, create diagnostics via `cat > plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T095900Z/phase_d2_completion/diagnostics.md` (include summary, stack trace excerpt, checkpoint path, comparison vs. baseline log).
- If `train_debug.log` appears at repo root, move it with `mv train_debug.log plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T095900Z/phase_d2_completion/`.
- Update `summary.md` checklist items to reflect captured artifacts, then adjust docs/fix_plan.md attempts (add Attempt #30) citing both the log and diagnostics file.
- Keep plan checklist D1 `[ ]` until the test goes green; add a note in diagnostics about next hypotheses (e.g., `save_hyperparameters` payload inspection) for continuity.

Pitfalls To Avoid:
- Do not mark D1 complete or flip plan/state flags until the integration test runs green.
- Avoid deleting or overwriting the baseline 2025-10-17 log; we need it for regression comparison.
- Keep all new artifacts under `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T095900Z/phase_d2_completion/`; no loose files at repo root.
- Maintain CPU-only execution paths; don’t enable CUDA-only flags in the integration test.
- Refrain from editing stable physics modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`) while triaging.
- Don’t silence the failing test; we rely on the red state to guide remediation.
- Ensure pytest command uses exact selector listed—no extra modules or env flags.
- Capture the full traceback in diagnostics; partial snippets make future debugging harder.
- Preserve TDD discipline: only move to implementation after documenting red evidence.
- Keep Git history clean—stage and commit only once artifacts and plan updates are in place.

Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:65
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T095900Z/phase_d2_completion/summary.md:8
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T233109Z/phase_d2_completion/pytest_integration_baseline.log
- specs/ptychodus_api_spec.md:205
- docs/workflows/pytorch.md:260
- docs/findings.md:8

Next Up:
- Begin persisting Lightning hyperparameters (or adapters) so `PtychoPINN_Lightning.load_from_checkpoint` no longer raises the missing-arguments TypeError once diagnostics are captured.
