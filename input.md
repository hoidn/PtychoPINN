Summary: Turn PyTorch stitching path green by implementing `_reassemble_cdi_image_torch`
Mode: TDD
Focus: INTEGRATE-PYTORCH-001-STUBS — Finish PyTorch workflow stubs deferred from Phase D2
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_workflows_components.py::TestReassembleCdiImageTorchRed -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T084016Z/phase_d2_completion/{phase_c3_playbook.md,summary.md,pytest_stitch_green.log}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS C3 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Implement `_reassemble_cdi_image_torch` per phase_c3_playbook (fix RawDataTorch dataset_path kwarg, add inference dataloader, wire Lightning predict, stitch outputs) (tests: targeted TDD)
2. INTEGRATE-PYTORCH-001-STUBS C4 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Run pytest tests/torch/test_workflows_components.py::TestReassembleCdiImageTorchRed -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T084016Z/phase_d2_completion/pytest_stitch_green.log (tests: targeted)
3. INTEGRATE-PYTORCH-001-STUBS C3+C4 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Update summary.md + docs/fix_plan Attempt #25 with green evidence referencing the 2025-10-19T084016Z artifact hub (tests: none)

If Blocked: Capture the failing traceback in pytest_stitch_green.log, roll C3 back to [P] in phase_d2_completion.md, summarize the blocker in summary.md, and log the obstruction (with hypotheses) in docs/fix_plan.md Attempts.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:52 — C3 checklist requires full Lightning inference + stitching implementation before Phase D can proceed.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T084016Z/phase_d2_completion/phase_c3_playbook.md — Documents the dataset_path fix and Lightning predict flow needed for green tests.
- specs/ptychodus_api_spec.md:185 — Reconstructor contract mandates stitched amplitude/phase outputs for `run_cdi_example_torch`.
- tests/torch/test_workflows_components.py:1076 — Red tests already encode the stitching contract; turning them green validates the implementation.
- docs/workflows/pytorch.md:205 — Workflow guide highlights the current stitching gap; closing it restores parity.

How-To Map:
- Patch `ptycho_torch/raw_data_bridge.py` so `RawDataTorch.generate_grouped_data` accepts `dataset_path` and forwards it to TensorFlow RawData (maintain docstring note that it’s ignored for caching).
- Add `_build_inference_dataloader` near `_build_lightning_dataloaders` in ptycho_torch/workflows/components.py to produce a deterministic TensorDictDataLoader (batch_size=config.batch_size or 1, shuffle=False, drop_last=False).
- Implement `_reassemble_cdi_image_torch` using _ensure_container for normalization, reuse the trained Lightning module (set eval + no_grad), iterate over the inference dataloader to collect predictions/offets, apply flip_x/flip_y/transpose adjustments, and stitch via `ptycho_torch.reassembly` helpers before returning `(recon_amp, recon_phase, results)`.
- Ensure tensors are moved to CPU before NumPy conversion; include `obj_tensor_full`, `global_offsets`, and container references in the results dict for downstream consumers.
- Run `pytest tests/torch/test_workflows_components.py::TestReassembleCdiImageTorchRed -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T084016Z/phase_d2_completion/pytest_stitch_green.log` and inspect for zero failures; rerun targeted selectors if additional fixes are required.
- Update `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T084016Z/phase_d2_completion/summary.md`, set C3/C4 states to `[x]`, and append Attempt #25 in docs/fix_plan.md with log references.

Pitfalls To Avoid:
- Do not remove torch-optional guards; keep imports inside functions where necessary.
- Avoid refactoring training code paths—limit changes to inference/stitching helpers.
- Ensure `_ensure_container` still works for RawDataTorch and existing containers after the dataset_path change.
- Keep Lightning tensors on the correct device until just before NumPy conversion; move to CPU before `np.array` usage.
- Maintain deterministic behaviour (respect existing seeds and sequential sampling flags).
- Preserve existing logging levels (debug flag) and avoid introducing noisy prints.
- Do not delete or overwrite prior artifact directories or red logs.
- Keep new tests strictly in pytest style; do not add unittest mixins.
- Remember to tee pytest output so the green log includes full stack traces if something fails mid-run.

Pointers:
- ptycho_torch/raw_data_bridge.py:107
- ptycho_torch/workflows/components.py:195
- tests/torch/test_workflows_components.py:1076
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T081500Z/phase_d2_completion/inference_design.md
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T084016Z/phase_d2_completion/phase_c3_playbook.md
- specs/ptychodus_api_spec.md:185

Next Up: Run `pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv` once stitching is green to evaluate the checkpoint-loading blocker.
