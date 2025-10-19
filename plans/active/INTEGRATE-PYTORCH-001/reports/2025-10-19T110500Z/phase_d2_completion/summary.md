# Phase D1d Planning Snapshot

- **Focus:** Enforce float32 tensors throughout PyTorch inference/stitching path so Lightning convolutions receive `torch.float32` inputs.
- **Trigger:** `pytest_integration_workflow_torch` now progresses past checkpoint load but fails with `RuntimeError: Input type (double) and bias type (float)`.
- **Evidence:** See `dtype_triage.md` and integration log under `2025-10-19T134500Z` for full stack trace. `train_debug.log` relocated here for hygiene.
- **Pending Tasks:**
  1. Author RED tests asserting `_build_inference_dataloader` and `_reassemble_cdi_image_torch` preserve float32 tensors (use pytest selectors under `tests/torch`).
  2. Implement dtype enforcement fix (loader cast or upstream conversion) once tests fail as expected.
  3. Capture new green logs (`pytest_float32_red.log` / `pytest_float32_green.log` / rerun integration) in this directory.
  4. Update `phase_d2_completion.md` D1d row and fix_plan attempts after remediation.
- **Dependencies:** Phase D1c complete; D2 parity summary remains blocked pending D1d green evidence.
