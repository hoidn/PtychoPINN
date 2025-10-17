Summary: Capture Phase D2 PyTorch workflow baseline and log current Lightning failure
Mode: Docs
Focus: INTEGRATE-PYTORCH-001-STUBS — Finish PyTorch workflow stubs deferred from Phase D2
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_integration_workflow_torch.py::TestPytorchWorkflow::test_pytorch_train_save_load_infer_cycle -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T233109Z/phase_d2_completion/{baseline.md,pytest_integration_baseline.log}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS A.A1+A.A3 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Catalogue current stubs and findings alignment in baseline.md (tests: none)
2. INTEGRATE-PYTORCH-001-STUBS A.A2 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Run pytest tests/torch/test_integration_workflow_torch.py::TestPytorchWorkflow::test_pytorch_train_save_load_infer_cycle -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T233109Z/phase_d2_completion/pytest_integration_baseline.log (tests: targeted)

If Blocked: Capture whatever portion of the integration log you obtained under the artifact directory, summarize the blocker and observed stack trace in baseline.md, and append a partial Attempt entry in docs/fix_plan.md noting why the run could not complete.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:24 — Phase A tasks unblock Lightning/stitching implementation by freezing the current stub state.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T231500Z/parity_summary.md:1 — Latest parity run documents the new Lightning failure we must reference in baseline.md.
- ptycho_torch/workflows/components.py:153 — Stubbed Lightning path currently returns placeholders; we need precise notes before implementation begins.
- docs/findings.md:8 — POLICY-001 enforces torch dependency; confirm baseline respects it and mention in summary.
- specs/ptychodus_api_spec.md:231 — Reconstructor contract dictates outputs the completed workflow must eventually satisfy.

How-To Map:
- export TS=2025-10-17T233109Z; mkdir -p plans/active/INTEGRATE-PYTORCH-001/reports/$TS/phase_d2_completion
- Re-read plan/table rows A.A1–A.A3 and annotate `baseline.md` with: current stub functions, open TODOs, planned resolution phases, and citations to POLICY-001 / FORMAT-001.
- Run pytest tests/torch/test_integration_workflow_torch.py::TestPytorchWorkflow::test_pytorch_train_save_load_infer_cycle -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/$TS/phase_d2_completion/pytest_integration_baseline.log
- After the run, append to `baseline.md`: command executed, environment knobs (torch install status, device), failure stack, and hypotheses for remediation.
- Update docs/fix_plan.md Attempts for INTEGRATE-PYTORCH-001-STUBS once artifacts exist (note checklist IDs touched) before handing back the loop.

Pitfalls To Avoid:
- Do not modify implementation files during this documentation loop.
- Keep all artifacts under the timestamped phase_d2_completion directory; no loose logs.
- Ensure pytest runs from project root with editable install active.
- Avoid truncating the integration log; redirect full output via tee.
- Reference findings by ID (POLICY-001, FORMAT-001) rather than prose summaries.
- Note GPU/cuda availability explicitly—parity expectations depend on it.
- Do not mark plan checklist items complete; leave them `[ ]` after documenting baseline.
- Skip rerunning full test suite; only the targeted selector is allowed this loop.
- Preserve CONFIG-001 ordering in write-ups—call out params.cfg update requirement.
- Capture exact stack trace lines for the Lightning failure; paraphrasing is insufficient.

Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:22
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T231500Z/parity_summary.md:1
- ptycho_torch/workflows/components.py:153
- docs/findings.md:8
- specs/ptychodus_api_spec.md:231

Next Up: Phase B.B1 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md (author Lightning red tests once baseline captured)
