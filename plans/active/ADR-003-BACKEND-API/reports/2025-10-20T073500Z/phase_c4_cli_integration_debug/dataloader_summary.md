# C4.D3 Lightning Dataloader Investigation (2025-10-20T073500Z)

## Context
- Initiative: ADR-003-BACKEND-API
- Focus: Phase C4.D3 regression in `tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer`
- Symptom: Lightning validation loop crashes with `IndexError: too many indices for tensor of dimension 4` at `ptycho_torch/model.py:1123` (`x = batch[0]['images']`).
- Artifacts referenced: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T060955Z/phase_c4_cli_integration_debug/pytest_integration_green.log` (latest failure log).

## Findings
1. `_build_lightning_dataloaders()` currently wraps the container in `TensorDataset(train_X, train_coords)`. Batches therefore look like `(Tensor[batch, H, W], Tensor[batch, 2])` with no mapping (`dict`/`TensorDict`). See probe output `dataloader_probe.txt` in this directory (first batch length=2, items are plain tensors).
2. `PtychoPINN_Lightning.compute_loss()` assumes the data loader yields a tuple `(tensor_dict, probe, scaling)` where the first element is a dict-like object exposing `'images'`, `'coords_relative'`, `'rms_scaling_constant'`, `'physics_scaling_constant'`, and optional supervised labels. Because `_build_lightning_dataloaders()` supplies plain tensors, indexing with keys raises the observed `IndexError`.
3. Legacy Lightning path (`PtychoDataModule` + `TensorDictDataLoader` in `ptycho_torch/train_utils.py`) still constructs the correct structure via `TensorDict` and `Collate_Lightning`. The MVP shortcut in `_build_lightning_dataloaders()` is the only remaining stub that bypasses this contract.
4. Integration CLI executed via new workflow path now exercises `_build_lightning_dataloaders()` directly, so the mismatch consistently reproduces in CI/cache-free environments.

## Next-Step Recommendations (hand-off to engineer loop)
- TDD: add a regression in `tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining` asserting `_build_lightning_dataloaders()` returns batches where `batch[0]` supports `['images', 'coords_relative', 'rms_scaling_constant', 'physics_scaling_constant']` and where `batch[1]`/`batch[2]` supply probe + scaling tensors.
- Implementation sketch: reuse `TensorDictDataLoader` + `Collate_Lightning` from `ptycho_torch.dataloader` instead of `TensorDataset`, or construct an equivalent `Dataset` that yields the `(tensor_dict, probe, scaling)` tuple expected by `compute_loss()`.
- After structuring batches correctly, rerun `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_dataloader_tensor_dict_structure -vv` (new test) followed by the integration selector.

## Files Generated This Loop
- `dataloader_probe.txt` — raw inspection of current batch tuple output.
- `dataloader_summary.md` — this narrative.
