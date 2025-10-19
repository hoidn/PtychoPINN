Summary: Capture decoder shape evidence and author RED test for Lightning probe_big merge mismatch before implementing fix
Mode: TDD
Focus: INTEGRATE-PYTORCH-001-D1E — Resolve Lightning decoder shape mismatch (Phase D1e)
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv; pytest tests/torch/test_workflows_components.py::TestDecoderLastShapeParity::test_probe_big_shape_alignment -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/{summary.md,d1e_shape_plan.md,shape_mismatch_triage.md,pytest_integration_shape_red.log,shape_trace.md,pytest_decoder_shape_red.log}

Do Now:
1. INTEGRATE-PYTORCH-001-D1E D1e.A1 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/d1e_shape_plan.md — Reproduce the failing integration selector via `pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/pytest_integration_shape_red.log` (tests: targeted selector)
2. INTEGRATE-PYTORCH-001-D1E D1e.A2 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/d1e_shape_plan.md — Add temporary env-gated logging around `Decoder_last.forward` to record `x`, `x1`, and `x2` shapes; run `TORCH_DECODER_TRACE=1 pytest tests/torch/test_workflows_components.py::TestReassembleCdiImageTorchFloat32::test_batches_remain_float32 -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/shape_trace.md` then remove instrumentation before finishing (tests: targeted selector)
3. INTEGRATE-PYTORCH-001-D1E D1e.A3 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/d1e_shape_plan.md — Update `shape_mismatch_triage.md` with captured shapes, hypotheses, and proposed fix direction (tests: none)
4. INTEGRATE-PYTORCH-001-D1E D1e.B1 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/d1e_shape_plan.md — Author failing pytest `TestDecoderLastShapeParity::test_probe_big_shape_alignment` and capture RED run via `pytest tests/torch/test_workflows_components.py::TestDecoderLastShapeParity::test_probe_big_shape_alignment -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/pytest_decoder_shape_red.log` (tests: targeted selector)

If Blocked: Preserve failing selector output in the artifact directory (`.../pytest_integration_shape_blocked.log` or `.../pytest_decoder_shape_blocked.log`), keep D1e rows `[P]`, and document blockers + hypotheses in `shape_mismatch_triage.md` before stopping.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md (D1e row): checklist gates remaining Phase D milestones.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/d1e_shape_plan.md: phased plan defines evidence → TDD workflow.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/shape_mismatch_triage.md: working hypotheses for mismatch.
- specs/ptychodus_api_spec.md §4.6: decoder/stitching contract that PyTorch must mirror.
- docs/workflows/pytorch.md §§6–7: Lightning inference expectations (probe handling, stitching).

How-To Map:
- Temporary instrumentation: wrap `Decoder_last.forward` with env check `if os.getenv("TORCH_DECODER_TRACE"):` and log tensor shapes via `logger.info`. Remove the code (or guard behind env check) before concluding the loop.
- Integration reproduction: use dataset referenced in previous runs (no new fixtures) and ensure `update_legacy_dict` executes before pipeline (already handled by CLI).
- Shape trace: after instrumentation run, copy relevant log lines into `shape_trace.md` (include tensor names, shapes, dtype) and note whether probe_big branch triggered.
- Red test scaffolding: mirror TensorFlow decoder behaviour by constructing mock tensors shaped `(batch, channels, N, N)` with `probe_big=True`, assert `outputs.shape` equals expected `(batch, n_filters_scale * 32, N, N)` (or parity result). Use fixtures from existing Lightning tests where possible.
- Artifact hygiene: Keep all logs and notes under `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/`; delete temporary checkpoints once shape evidence captured.

Pitfalls To Avoid:
- Do not leave debug logging enabled without env guard; remove instrumentation before committing.
- Avoid mutating TensorFlow baseline files (`ptycho/model.py`, `ptycho/tf_helper.py`).
- Keep dtype enforcement from D1d intact; do not revert recent float32 casts.
- Ensure new pytest uses native pytest style (no unittest mix) and references plan artifacts.
- When adding tests, avoid hard-coding absolute paths; reuse fixtures/configs from existing tests.
- Do not skip params.cfg bridge — verify `update_legacy_dict` runs in tests via existing fixtures.
- Limit logging noise; shape trace file should summarize essential data, not entire pytest output.
- Maintain deterministic seeds when generating mock tensors to keep tests stable.
- Revert instrumentation patches before running red test to prevent unintended logging assertions.

Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:68
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/d1e_shape_plan.md
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/shape_mismatch_triage.md
- ptycho_torch/model.py:312
- tests/torch/test_workflows_components.py:1530

Next Up: Implement decoder crop/pad parity (D1e.B2) once red test and evidence land.
