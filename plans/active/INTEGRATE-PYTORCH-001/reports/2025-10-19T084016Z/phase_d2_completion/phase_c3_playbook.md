# Phase C3 Implementation Playbook — `_reassemble_cdi_image_torch`

## Context
- **Initiative:** INTEGRATE-PYTORCH-001 — PyTorch backend integration
- **Phase/Checklist:** D2.C / Row C3 in `phase_d2_completion.md`
- **Objective:** Ship a working PyTorch stitching path so `run_cdi_example_torch(..., do_stitching=True)` returns `(recon_amp, recon_phase, results)` and all TestReassembleCdiImageTorch* selectors pass.
- **Blocking defect surfaced by C2 red tests:** `_ensure_container` calls `RawDataTorch.generate_grouped_data(..., dataset_path=...)`, but the adapter signature drops that kwarg. This triggers the current TypeError before `_reassemble_cdi_image_torch` is reached.

## Prereqs & References
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T081500Z/phase_d2_completion/inference_design.md`
- TensorFlow baseline: `ptycho/workflows/components.py:582-666`
- PyTorch helpers: `ptycho_torch/reassembly.py`, `ptycho_torch/helper.py`, `ptycho_torch/lightning_module.py`
- Spec contract: `specs/ptychodus_api_spec.md` §4.5–§4.6 (stitching lifecycle)

## Implementation Checklist
1. **Unblock `_ensure_container`**
   - Update `RawDataTorch.generate_grouped_data` (ptycho_torch/raw_data_bridge.py) to accept `dataset_path` kwarg and forward it to TensorFlow RawData.
   - Mirror TensorFlow docstring note that the parameter is kept for compatibility but ignored.
   - Add targeted unit coverage in `tests/torch/test_workflows_components.py` (or new test) verifying `_ensure_container` handles RawDataTorch after the change.
2. **Introduce `_build_inference_dataloader` helper**
   - Deterministic `TensorDictDataLoader` for inference path (`shuffle=False`, `batch_size=config.batch_size or 1`).
   - Place near `_build_lightning_dataloaders`; keep torch-optional import guards.
3. **Implement `_reassemble_cdi_image_torch`** (follow design doc)
   - Normalize input via `_ensure_container` (train/test) and reuse configured Lightning module from `train_results` / `models` dict.
   - Switch module to `eval()` and guard with `torch.no_grad()`.
   - Iterate over inference dataloader, accumulating predictions + global offsets.
   - Apply `flip_x`, `flip_y`, `transpose` transformations to offsets/predictions before stitching.
   - Call appropriate helper (`reassemble_multi_channel` or `reassemble_single_channel`) to build complex canvas, then derive amplitude/phase numpy arrays.
   - Return `(recon_amp, recon_phase, results)` where `results` contains at minimum `obj_tensor_full`, `global_offsets`, and `containers` reference for downstream consumers.
4. **Document behavior**
   - Update function docstrings (+ module-level TODO removal) and ensure logging aligns with POLICY-001.
   - Record any residual limitations (e.g., device assumptions) in `summary.md` for Phase D documentation.

## Verification Steps
- `pytest tests/torch/test_workflows_components.py::TestReassembleCdiImageTorchRed -vv` (expect **GREEN** after implementation)
- `pytest tests/torch/test_workflows_components.py::TestRunCdiExampleTorch::test_stitching_path -vv` (if new selector added)
- Optional regression: `pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv --maxfail=1`
- Capture green log: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T084016Z/phase_d2_completion/pytest_stitch_green.log`

## Artifact Expectations
- Green pytest log(s) under this directory (at minimum `pytest_stitch_green.log`).
- Updated `summary.md` describing implementation decisions and linking to logs.
- docs/fix_plan.md Attempt #24 entry referencing this playbook and artifacts.

## Open Risks to Monitor
- Complex tensor handling: ensure reassembly helper receives `torch.complex64` or split real/imag as required.
- Device semantics: move tensors to CPU before NumPy conversion to avoid CUDA → NumPy errors.
- Probe handling path: confirm `RawDataTorch` continues to surface `probeGuess` to dataloader for Lightning prediction.
