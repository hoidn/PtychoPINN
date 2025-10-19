Summary: Fix PyTorch stitching channel order and turn the C4 tests green
Mode: TDD
Focus: INTEGRATE-PYTORCH-001-STUBS — Finish PyTorch workflow stubs deferred from Phase D2
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_workflows_components.py -k ReassembleCdiImageTorch -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T092448Z/phase_d2_completion/{debug_shape_triage.md,pytest_stitch_green.log}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS C4 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Update `_reassemble_cdi_image_torch` to emit channel-last tensors before calling the TensorFlow helper (tests: none)
2. INTEGRATE-PYTORCH-001-STUBS C4 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Refresh `TestReassembleCdiImageTorch*` fixtures/assertions and run `pytest tests/torch/test_workflows_components.py -k ReassembleCdiImageTorch -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T092448Z/phase_d2_completion/pytest_stitch_green.log` (tests: targeted)
3. INTEGRATE-PYTORCH-001-STUBS C4 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Update `phase_d2_completion/summary.md` + docs/fix_plan.md Attempt history with the 2025-10-19T092448Z green evidence, and relocate any stray `train_debug.log` into the same report directory (tests: none)

If Blocked: Capture the failing pytest output in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T092448Z/phase_d2_completion/pytest_stitch_green.log`, roll C4 back to `[P]`, note the observed tensor shapes + error text in `debug_shape_triage.md`, and log the blocker in docs/fix_plan.md before pausing.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:53 — C4 remains open; guidance now calls out the channel-order fix plus updated tests.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T092448Z/phase_d2_completion/debug_shape_triage.md — Documents confirmed root cause (channel-first tensor) and required remediation steps.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T084016Z/phase_d2_completion/summary.md — Next Steps point back to the triage note and expect fresh green evidence once fixed.
- tests/torch/test_workflows_components.py:1076 — Existing green-phase test scaffolding that now needs channel-aware fixtures/assertions.
- specs/ptychodus_api_spec.md:185 — Stitching contract requires `(recon_amp, recon_phase, results)` with correct tensor shapes for downstream consumers.

How-To Map:
- Apply the channel-last conversion inside `_reassemble_cdi_image_torch`: after concatenating predictions, use `torch.moveaxis(obj_tensor_full, 1, -1)` (or equivalent) when tensors arrive channel-first; call `np.moveaxis` as a final guard before handing data to `tf_helper.reassemble_position`.
- Ensure the mock Lightning module in tests returns deterministic `torch.ones` complex tensors shaped `(batch, gridsize**2, N, N)` so that moveaxis logic is exercised; keep one guard test explicitly asserting the `train_results=None` NotImplemented path.
- Expand the tests to validate that `results['obj_tensor_full'].shape[-1] == config.model.gridsize ** 2`, amplitude/phase arrays are finite via `np.isfinite`, and that flip/transpose cases still succeed.
- Run `pytest tests/torch/test_workflows_components.py -k ReassembleCdiImageTorch -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T092448Z/phase_d2_completion/pytest_stitch_green.log` from repo root.
- Move the prior root-level `train_debug.log` into `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T092448Z/phase_d2_completion/` (or delete if superseded) so all artifacts live under the initiative directory.
- After tests pass, append the new evidence (timestamp + log path) to `summary.md` and docs/fix_plan.md, marking C4 as `[x]` only when the log shows zero failures.

Pitfalls To Avoid:
- Do not bypass the TensorFlow reassembly helper; we need parity proof before migrating to native PyTorch stitching.
- Keep tensors on CPU for the mock; avoid introducing CUDA-only code in tests.
- Preserve complex64 dtypes for model outputs so amplitude/phase calculations remain valid.
- Do not drop the `train_results=None` guard test—this regression coverage is still required.
- Ensure pytest selector stays exactly as mapped; no ad-hoc wildcards or extra modules.
- Don’t overwrite existing artifacts—append new logs under the fresh timestamp directory.
- Avoid touching stable physics files (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Keep plan checklists synchronized; mark C4 complete only after documentation and logs are updated.
- Leave integration-test (Phase D) steps untouched this loop.

Pointers:
- ptycho_torch/workflows/components.py:608
- tests/torch/test_workflows_components.py:1076
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:53
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T092448Z/phase_d2_completion/debug_shape_triage.md
- specs/ptychodus_api_spec.md:185

Next Up:
- Run `pytest tests/torch/test_integration_workflow_torch.py -vv` once stitching goes green to unblock Phase D. 
