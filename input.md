Summary: Align PyTorch Decoder_last with TensorFlow parity so probe_big=True no longer throws shape mismatch
Mode: TDD
Focus: INTEGRATE-PYTORCH-001-D1E — Resolve Lightning decoder shape mismatch (Phase D1e)
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_workflows_components.py::TestDecoderLastShapeParity -vv; pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T111855Z/phase_d2_completion/{summary.md,pytest_decoder_shape_red.log,pytest_decoder_shape_green.log,pytest_integration_shape_green.log}

Do Now:
1. INTEGRATE-PYTORCH-001-D1E D1e.B1 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/d1e_shape_plan.md — Revise `TestDecoderLastShapeParity.test_probe_big_shape_alignment` to assert successful forward pass + matching spatial dims (no RuntimeError). Capture the RED failure via `pytest tests/torch/test_workflows_components.py::TestDecoderLastShapeParity -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T111855Z/phase_d2_completion/pytest_decoder_shape_red.log` (tests: targeted selector).
2. INTEGRATE-PYTORCH-001-D1E D1e.B2 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/d1e_shape_plan.md — Implement the decoder alignment fix in `ptycho_torch/model.py` (center-crop x2 to x1 dims, mirroring TensorFlow `trim_and_pad_output`) and keep dtype/device intact (tests: none).
3. INTEGRATE-PYTORCH-001-D1E D1e.B3 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/d1e_shape_plan.md — Rerun `pytest tests/torch/test_workflows_components.py::TestDecoderLastShapeParity -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T111855Z/phase_d2_completion/pytest_decoder_shape_green.log` and ensure both probe_big=True/False cases pass (tests: targeted selector).
4. INTEGRATE-PYTORCH-001-D1E D1e.C1 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/d1e_shape_plan.md — Run the integration selector `pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T111855Z/phase_d2_completion/pytest_integration_shape_green.log` to confirm the end-to-end workflow clears the decoder stage (tests: targeted selector).
5. INTEGRATE-PYTORCH-001-D1E D1e.C2+C3 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/d1e_shape_plan.md — Update plan checklist states, `shape_mismatch_triage.md`, and `summary.md`; author new `summary.md` under 2025-10-19T111855Z capturing fix results; log docs/fix_plan Attempt #40 with artifacts and mark D1e rows `[x]` once green (tests: none).

If Blocked: Capture the failing selector output to the 2025-10-19T111855Z hub (`pytest_decoder_shape_blocked.log` or `pytest_integration_shape_blocked.log`), revert any partial code, and document the blocker plus fresh hypotheses in `shape_mismatch_triage.md` before stopping.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md D1e row gates Phase D completion and downstream parity docs.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/d1e_shape_plan.md lays out evidence→fix→validation workflow.
- specs/ptychodus_api_spec.md §4.6 mandates decoder parity across backends.
- docs/workflows/pytorch.md §§6–7 define Lightning inference expectations we must keep intact.
- shape_mismatch_triage.md captures validated root cause (upsample vs padding) guiding the required crop logic.

How-To Map:
- Red run: after updating the test, run `pytest tests/torch/test_workflows_components.py::TestDecoderLastShapeParity -vv` (expect RuntimeError) and store stdout with `tee` in the new timestamp directory.
- Decoder fix: mirror TensorFlow’s `trim_and_pad_output` (see `ptycho/model.py:360-410`) by computing required crop offsets from `data_config.N` and slicing `x2` to match `x1`; keep tensors on the incoming device and maintain complex pair handling for amp/phase paths.
- Post-fix validation: rerun the class-level pytest selector (should now pass) and inspect log for both test methods; ensure the test checks final `outputs.shape` equals `(batch, out_channels, N, N + 2*(N//4))` or equivalent parity rule you codify.
- Integration: run the targeted integration selector once after unit tests pass; monitor for downstream regressions and keep logs under the 2025-10-19T111855Z hub.
- Documentation: mark checklist boxes in `d1e_shape_plan.md`, update `phase_d2_completion.md` D1e row to `[x]`, append GREEN evidence to `shape_mismatch_triage.md`, write `summary.md` for the new timestamp, and record docs/fix_plan Attempt #40 referencing all stored artifacts.

Pitfalls To Avoid:
- Do not leave temporary logging or instrumentation in `ptycho_torch/model.py` after capturing evidence.
- Keep dtype and device consistent—no implicit `.cpu()` or `.double()` conversions in the crop logic.
- Avoid editing TensorFlow baseline files; all changes stay in PyTorch modules/tests.
- Ensure pytest tests remain native pytest style (no unittest mixins) and deterministic (set seeds if generating tensors).
- Store every new log or summary under the 2025-10-19T111855Z directory—nothing at repo root.
- Don’t skip updating docs/fix_plan.md and plan checklist states once work is complete.
- Verify run directories are cleaned up between test runs to prevent stale checkpoints from masking failures.
- Keep crop math symmetric for even/odd width differences; add assertions if needed.
- Maintain TDD cadence: confirm the test fails before applying the fix.
- Leave `TORCH_DECODER_TRACE` instrumentation disabled unless explicitly re-enabled for debugging.

Pointers:
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/d1e_shape_plan.md
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/shape_mismatch_triage.md
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:60
- tests/torch/test_workflows_components.py:1752
- ptycho/model.py:360
- specs/ptychodus_api_spec.md:4.6
- docs/workflows/pytorch.md:§6

Next Up: D1e.C2 documentation polish (parity narrative) or D2 parity summary once integration log is green.
