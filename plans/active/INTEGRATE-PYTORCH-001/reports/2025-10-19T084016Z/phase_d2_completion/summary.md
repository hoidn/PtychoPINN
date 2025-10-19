# Phase C3 Implementation Staging (Attempt #24)

## Snapshot
- Consolidated C3 playbook (`phase_c3_playbook.md`) to guide implementation of `_reassemble_cdi_image_torch`.
- Identified upstream blocker: `_ensure_container` passes `dataset_path` to `RawDataTorch.generate_grouped_data`, but the adapter signature omits the kwarg — causes the failing TypeError seen in C2 red run (`pytest_stitch_red.log`).
- Recorded verification expectations (targeted pytest selectors, artifact paths) for the upcoming green loop.

## Next Steps for Engineer (Ralph)
1. Patch `RawDataTorch.generate_grouped_data` to accept `dataset_path` and forward it to TF RawData (keeps compatibility; restore parity with TensorFlow helper).
2. Implement `_reassemble_cdi_image_torch` per design doc, including new inference dataloader helper and Lightning prediction path.
3. Run targeted pytest selector(s) and capture `pytest_stitch_green.log` under this timestamp directory.
4. Update docs/fix_plan.md Attempt log + plan checklist when tests pass.

## Key References
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T081500Z/phase_d2_completion/inference_design.md`
- `specs/ptychodus_api_spec.md` §4.5–§4.6
- `docs/workflows/pytorch.md` §§5–7
- TensorFlow baseline: `ptycho/workflows/components.py:582-666`

## Risks / Watch Items
- Ensure all tensors moved to CPU before converting to NumPy for amplitude/phase outputs.
- Confirm lightning module retrieval works for both live training results and persisted inference bundles (if available).
- Maintain torch-optional import guards so TensorFlow-only environments continue to skip tests cleanly.
