Summary: Author Phase C2 red tests for PyTorch stitching path
Mode: TDD
Focus: INTEGRATE-PYTORCH-001-STUBS — Finish PyTorch workflow stubs deferred from Phase D2
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_workflows_components.py::TestReassembleCdiImageTorchRed -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T081500Z/phase_d2_completion/{inference_design.md,pytest_stitch_red.log}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS C2 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Add new pytest class `TestReassembleCdiImageTorchRed` covering stitching entry path (tests: author)
2. INTEGRATE-PYTORCH-001-STUBS C2 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Run pytest tests/torch/test_workflows_components.py::TestReassembleCdiImageTorchRed -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T081500Z/phase_d2_completion/pytest_stitch_red.log (tests: targeted, expect failure on NotImplementedError)
3. INTEGRATE-PYTORCH-001-STUBS C2 checklist @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Update summary.md with failure analysis, set checklist state, and log Attempt in docs/fix_plan.md (tests: none)

If Blocked: Capture the failing traceback in `pytest_stitch_red.log`, leave C2 `[P]`, summarize blockers + hypotheses in `summary.md`, and record the obstruction in docs/fix_plan.md Attempts.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:54 — C2 requires failing coverage before implementation; red tests guard `_reassemble_cdi_image_torch`.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T081500Z/phase_d2_completion/inference_design.md — Defines expected Lightning prediction flow, reassembly helpers, and flip/transpose behaviors the tests must codify.
- specs/ptychodus_api_spec.md:191 — Reconstructor lifecycle mandates stitched amplitude/phase outputs; red tests translate contract into executable acceptance criteria.
- docs/workflows/pytorch.md:210 — Workflow guide highlights current stitching gap; closing it starts with alarm-raising regression coverage.

How-To Map:
- Extend `tests/torch/test_workflows_components.py` by introducing `TestReassembleCdiImageTorchRed` with fixtures that call `_reassemble_cdi_image_torch` via `run_cdi_example_torch(..., do_stitching=True)` on deterministic dummy data; assert that NotImplementedError is raised until Phase C3 lands.
- Add parametrized case covering flip/transpose flags so the failure message surfaces context for future implementation (e.g., include pytest `with pytest.raises(NotImplementedError, match="stitching path not yet implemented")`).
- Keep new tests pure pytest (no unittest mix); rely on `torch.manual_seed(0)` inside fixtures for deterministic tensors.
- Command: `pytest tests/torch/test_workflows_components.py::TestReassembleCdiImageTorchRed -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T081500Z/phase_d2_completion/pytest_stitch_red.log`
- After the run, append failure summary + selector to `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T081500Z/phase_d2_completion/summary.md`, set C2 row to `[P]` or `[x]` per outcome, and add Attempt entry in docs/fix_plan.md referencing the log.

Pitfalls To Avoid:
- Do not touch `_reassemble_cdi_image_torch` implementation in this loop—goal is RED tests only.
- Avoid importing Lightning globally in tests; use local imports or pytest fixtures to retain torch-optional semantics.
- Ensure new tests clean up monkeypatches via pytest fixture scopes; no lingering state across suites.
- Capture full pytest output with `tee`; rerun if log missing or truncated.
- Keep deterministic seeds/config consistent with design doc to simplify future green runs.
- Do not relocate existing artifact directories or overwrite prior logs from Attempts #0–#21.

Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:54
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T081500Z/phase_d2_completion/inference_design.md
- tests/torch/test_workflows_components.py:560
- specs/ptychodus_api_spec.md:191
- docs/workflows/pytorch.md:210

Next Up: C3 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — implement `_reassemble_cdi_image_torch` once red coverage is in place.
