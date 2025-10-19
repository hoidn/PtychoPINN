Summary: Capture PyTorch integration baseline evidence for TEST-PYTORCH-001 Phase A
Mode: TDD
Focus: TEST-PYTORCH-001 — PyTorch integration regression plan (Phase A)
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv
Artifacts: plans/active/TEST-PYTORCH-001/reports/2025-10-19T115303Z/{baseline/inventory.md,baseline/pytest_integration_current.log,baseline/summary.md}

Do Now:
1. TEST-PYTORCH-001 A1 @ plans/active/TEST-PYTORCH-001/implementation.md#L16 — Review existing PyTorch integration coverage (charter + current unittest) and record blockers in `baseline/inventory.md` (tests: none).
2. TEST-PYTORCH-001 A2 @ plans/active/TEST-PYTORCH-001/implementation.md#L23 — Run baseline selector `pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv` and capture full output with `tee` to `baseline/pytest_integration_current.log`.
3. TEST-PYTORCH-001 A3 @ plans/active/TEST-PYTORCH-001/implementation.md#L24 — Summarize environment prerequisites (flags, dataset path, runtime, exit status) in `baseline/summary.md`, highlighting gaps that block a ≤120s CPU regression (tests: none).

If Blocked: If the selector fails before hitting CLI (e.g., missing torch extras), document the failure, traceback, and environment info in `baseline/summary.md`, mark the command as `BLOCKED` with exit code, and notify in docs/fix_plan Attempts rather than modifying workflows.

Priorities & Rationale:
- plans/active/TEST-PYTORCH-001/implementation.md:1 anchors the phased plan; Phase A must be complete before fixture work.
- tests/torch/test_integration_workflow_torch.py:1 documents current unittest harness that still reflects red-phase assumptions.
- specs/ptychodus_api_spec.md:191 enforces train→infer contract that the regression must cover; gaps become plan risks.
- docs/workflows/pytorch.md:120 captures deterministic Lightning settings we must obey during baseline run.
- docs/findings.md:8 (POLICY-001) reminds us torch>=2.2 is mandatory; record if environment deviates.

How-To Map:
- `mkdir -p plans/active/TEST-PYTORCH-001/reports/2025-10-19T115303Z/baseline`
- Review sources for inventory: `plans/pytorch_integration_test_plan.md`, `tests/torch/test_integration_workflow_torch.py`, prior parity logs under `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T111855Z/phase_d2_completion/`
- Baseline run: `CUDA_VISIBLE_DEVICES=\"\" pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv | tee plans/active/TEST-PYTORCH-001/reports/2025-10-19T115303Z/baseline/pytest_integration_current.log`
- Record runtime (wall clock) and checkpoint paths in `baseline/summary.md`; note whether command already finishes within ≤120s.
- If logs exceed guidance, compress with `gzip` after tee (optional) but keep plaintext copy per artifact policy.

Pitfalls To Avoid:
- Do not edit production code or tests this loop—evidence only.
- Keep artifacts under the timestamped baseline directory; nothing at repo root.
- Avoid truncating pytest output; ensure `tee` captures full trace.
- Respect POLICY-001: fail fast if torch unavailable instead of skipping.
- Note deterministic knobs (`--disable_mlflow`, seeds) even if already default; missing notes break Phase B.
- Do not assume dataset small enough; measure and document.
- Ensure temporary directories created by test are within tmpfs; no lingering artifacts outside baseline dir.
- Leave unittest-to-pytest migration decisions for later phases; only describe impacts now.
- Do not mark fix_plan tasks complete until evidence is recorded in docs/fix_plan Attempts.

Pointers:
- plans/active/TEST-PYTORCH-001/implementation.md:16
- tests/torch/test_integration_workflow_torch.py:58
- plans/pytorch_integration_test_plan.md:1
- specs/ptychodus_api_spec.md:191
- docs/workflows/pytorch.md:180
- docs/findings.md:8

Next Up: If Phase A closes early, prepare fixture minimization outline (B1–B3) using insights from baseline runtime.
