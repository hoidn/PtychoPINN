Summary: Align PyTorch stitching tests with the implemented `_reassemble_cdi_image_torch`
Mode: TDD
Focus: INTEGRATE-PYTORCH-001-STUBS — Finish PyTorch workflow stubs deferred from Phase D2
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_workflows_components.py -k ReassembleCdiImageTorch -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T084016Z/phase_d2_completion/{summary.md,pytest_stitch_green.log,train_debug.txt}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS C4 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Update `TestReassembleCdiImageTorch*` to assert the new Lightning stitching behavior (supply `train_results` fixtures, validate amplitude/phase outputs, preserve a focused NotImplemented guard case) (tests: targeted TDD)
2. INTEGRATE-PYTORCH-001-STUBS C4 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Run `pytest tests/torch/test_workflows_components.py -k ReassembleCdiImageTorch -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T084016Z/phase_d2_completion/pytest_stitch_green.log` (tests: targeted)
3. INTEGRATE-PYTORCH-001-STUBS C4 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Update summary.md + docs/fix_plan Attempt entry with green evidence (tests: none)

If Blocked: Capture the failing pytest output in `pytest_stitch_green.log`, roll C4 back to `[P]` in phase_d2_completion.md, note the obstacle (including assertion text) in summary.md, and log the issue in docs/fix_plan.md with hypotheses before stopping.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:44 — C4 now requires modernizing the stitching tests after C3 shipped; they currently hard-fail because they still expect NotImplemented.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T084016Z/phase_d2_completion/summary.md — Next Steps section calls for green pytest evidence once tests assert stitched outputs.
- docs/fix_plan.md:26 (Attempt #26) — Documents the supervisor review and directs this loop to close out C4 with updated tests and logs.
- tests/torch/test_workflows_components.py:1076 — Existing “Red” class encodes NotImplemented expectations that now need to be upgraded to success-path assertions.
- specs/ptychodus_api_spec.md:185 — Stitching contract requires `(recon_amp, recon_phase, results)` outputs; the green tests must verify this behavior.

How-To Map:
- Refactor `TestReassembleCdiImageTorchRed` into a green-phase suite (rename optional, keep selector stable for now). Introduce fixtures:
  * `mock_lightning_module` — subclass/minimal object with `eval()` and `__call__(self, X)` returning a deterministic `torch.ones` complex tensor shaped `(batch, 1, N, N)`.
  * `stitch_train_results` — returns `{"models": {"lightning_module": mock_lightning_module}}`; reuse in both direct `_reassemble_cdi_image_torch` and `run_cdi_example_torch` tests.
- Update the primary tests to:
  * Call `_reassemble_cdi_image_torch(..., train_results=stitch_train_results)` and assert that amplitude/phase outputs are numpy arrays with expected shapes and finite values.
  * Verify flip_x / flip_y / transpose by comparing offsets (e.g., check sign flips in `results["global_offsets"]`).
  * For the orchestration test, monkeypatch `_train_with_lightning` to return the same `stitch_train_results` so `run_cdi_example_torch(do_stitching=True)` hits the real stitching path and returns non-None amplitude/phase.
- Preserve a single guard test that still exercises the `train_results=None` NotImplemented branch (update regex to match the new message text).
- Keep pytest style pure (no unittest mixins); prefer helper functions/fixtures inside the module rather than external scripts.
- After edits, run the selector exactly as listed, capture the log with `tee`, and inspect for zero failures before proceeding.
- Update `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T084016Z/phase_d2_completion/summary.md` (append validation notes) and append the same evidence to docs/fix_plan.md Attempts (C4 entry).

Pitfalls To Avoid:
- Do not regress the `train_results=None` guard; keep the NotImplemented path covered by a focused regression test.
- Avoid invoking real Lightning training in tests; rely on lightweight stubs so the selector stays fast.
- Ensure stub outputs use `torch.complex64` so downstream amplitude/phase calculations remain valid.
- Keep tensors on CPU inside stubs; don’t introduce CUDA requirements.
- Preserve deterministic seeds/fixtures already defined in the test module.
- Do not rewrite production code — stay within tests and documentation updates for this loop.
- Ensure pytest log is stored under the timestamped reports directory (no artifacts at repo root).
- Keep class/function names unique to avoid pytest collection conflicts.
- Remember to run selectors from repo root so relative imports stay valid.
- Leave integration test expectations (Phase D) untouched until C4 is green.

Pointers:
- tests/torch/test_workflows_components.py:1076
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:51
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T084016Z/phase_d2_completion/summary.md
- docs/fix_plan.md:26
- specs/ptychodus_api_spec.md:185

Next Up: Run `pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv` once stitching tests are green to re-evaluate the checkpoint-loading failure.
