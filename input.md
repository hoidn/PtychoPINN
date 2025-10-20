Summary: Fix the coords_relative shape mismatch so the PyTorch integration workflow passes C4.D3.
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C4 CLI integration
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py::TestInferenceCLI -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T061500Z/phase_c4_cli_integration_debug/{analysis.md,pytest_integration_red.log,pytest_integration_green.log,pytest_cli_train_green.log,pytest_cli_inference_green.log}

Do Now:
1. ADR-003-BACKEND-API C4.D3 diagnostics @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — rerun the PyTorch integration selector, capture the failing log, and record observed tensor shapes in `analysis.md`; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv (expect FAIL).
2. ADR-003-BACKEND-API C4.D3 fix @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — modify the PyTorch dataloader so `coords_relative` written to the mmap matches the target shape contract; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv (expect PASS).
3. ADR-003-BACKEND-API C4.D3 regression @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — rerun the targeted CLI selectors to confirm no regressions in execution-config wiring; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py::TestInferenceCLI -vv.
4. ADR-003-BACKEND-API C4.F1+C4.F2 wrap-up @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — update summary.md, plan.md statuses, and docs/fix_plan.md with Attempt #22 including new artifact links; tests: none.

If Blocked: Capture the failing stack trace to `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T061500Z/phase_c4_cli_integration_debug/blocker.log`, mark C4.D3 `[P]` with notes in plan.md, append a blocker note to docs/fix_plan.md, and stop.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md:98 — C4.D3 cannot close while the integration workflow fails.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T050500Z/phase_c4_cli_integration/summary.md — documents the coords_relative mismatch surfaced during full-suite run.
- specs/data_contracts.md:28 — contraction demands grouped coordinate tensors obey the `(nsamples, 1, 2, gridsize²)` shape.
- tests/torch/test_integration_workflow_torch.py:45 — orchestrates the failing workflow; use it to confirm behaviour after the fix.
- ptycho_torch/dataloader.py:500 — mmap population path currently writing coords_relative with an extra dimension.

How-To Map:
- Capture RED evidence: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T061500Z/phase_c4_cli_integration_debug/pytest_integration_red.log`.
- While the test runs, add temporary shape logging via `analysis.md` (summaries only) — remove any inline print/debug code before yielding.
- Compare PyTorch coords pipeline to TensorFlow reference: inspect `ptycho/raw_data.py` and `pytests` expecting `(nsamples, 1, 2, gridsize**2)`; adjust PyTorch path accordingly (likely reshape or transpose before writing to mmap).
- After implementing the fix, rerun the selector and store success log: `.../pytest_integration_green.log`.
- Re-run CLI guards, teeing outputs to `pytest_cli_train_green.log` and `pytest_cli_inference_green.log`.
- Summarise findings, decisions, and directory contents in `analysis.md`; update plan statuses and docs/fix_plan.md once tests are green.

Pitfalls To Avoid:
- Do not mask the integration failure by altering or skipping the test; the fix must address the mmap shape bug.
- Avoid editing protected cores (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`); keep changes scoped to PyTorch dataloader/workflow code.
- Ensure CONFIG-001 sequencing remains intact; do not bypass `update_legacy_dict`.
- Preserve dtype contracts (coords tensors should stay float32/int64 as documented).
- Keep temporary debug statements out of the final diff; use `analysis.md` for notes.
- Do not delete or move existing artifacts from prior loops; add new logs under the fresh timestamp only.
- Run only the mapped selectors; no full-suite reruns unless the plan explicitly requests it.
- Confirm regenerated artifacts (if any) remain DATA-001 compliant before writing to repo.
- Maintain ASCII-only edits; follow existing formatting in dataloader and plan files.
- Remember to add docs/fix_plan Attempt #22 with artifact paths once work completes.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md:90
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T050500Z/phase_c4_cli_integration/summary.md:34
- specs/data_contracts.md:28
- tests/torch/test_integration_workflow_torch.py:45
- ptycho_torch/dataloader.py:500

Next Up: 1) Once C4.D3 is green, close C4.D4 manual smoke and progress to C4.E documentation updates.
