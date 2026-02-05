# Bug Report: PyTorch `object_big` uses absolute coords (breaks TF contract)

## Summary
PyTorch training/inference treats `object_big=True` as requiring **absolute** scan coordinates (e.g., `coords_nominal`), but the TF reference expects **relative offsets** (`coords_relative`, centered per-group with sign flip). When `object_big=True`, this causes incorrect reassembly/extraction in the forward model and degrades metrics. The issue is masked when `object_big=False`, so the regression surfaced after `config_factory` began defaulting `object_big=True`.

## Impact
- **Integration regression:** `tests/torch/test_grid_lines_hybrid_resnet_integration.py` fails with MAE amp above tolerance.
- **Behavioral mismatch with TF:** `object_big` is a no-op for `gridsize=1` in TF (relative offsets are zero), but **not** in PyTorch due to absolute positions.
- **Potentially affects all PyTorch runs** where `object_big=True` and positions are not relative offsets.

## Evidence
- Failing commit (split bisect): `291e9a780254d82b68f410cd363b382aad0538d8`
- Integration test FAIL log: `.artifacts/bisect_split/test_291e9a780254d82b68f410cd363b382aad0538d8.log`
- Integration test PASS when forcing `object_big=False`: `.artifacts/bisect_split/test_object_big_false.log`

## Expected Behavior (TF contract)
- `object_big=True` should operate on **relative offsets**:
  - `coords_relative = local_offset_sign * (coords_nn - coords_offsets)`
  - For `C=1` (gridsize=1), `coords_relative == 0`, making reassemble/extract a no-op.
- See TF reference:
  - `ptycho/raw_data.py:get_relative_coords`
  - `ptycho/model.py` uses `input_positions` (relative offsets)

## Actual Behavior (PyTorch grid_lines runner)
- `scripts/studies/grid_lines_torch_runner.py` passes **absolute** `coords_nominal` into the Lightning path.
- `ptycho_torch/model.py:ForwardModel.forward` uses these positions directly for reassembly/extraction.
- This violates the TF contract and changes physics behavior, even at `gridsize=1`.

## Root Cause
The PyTorch grid_lines path supplies **absolute coordinates** where the model expects **relative offsets**. This becomes a regression when `object_big=True` is defaulted in `config_factory` (commit `291e9a78`).

## Reproduction
```bash
python -m pytest -v -m integration tests/torch/test_grid_lines_hybrid_resnet_integration.py
```
- Fails when `object_big=True` (default).
- Passes when `object_big=False` (verified via temporary override).

## Proposed Fixes (pick one)
1. **Normalize coords to relative offsets** before `ForwardModel` when `object_big=True`:
   - `coords_relative = -1 * (coords - coords.mean(axis=channel))`
   - Matches TF `local_offset_sign` behavior.
2. **Grid-lines runner**: build and pass `coords_relative` instead of `coords_nominal`.
3. **Guard in ForwardModel**: if `C==1`, bypass reassemble/extract or normalize to zeros.

## Files Involved
- `ptycho_torch/model.py` (`ForwardModel.forward`)
- `scripts/studies/grid_lines_torch_runner.py` (passes coords)
- `ptycho/raw_data.py` (`get_relative_coords`, TF reference)
- `ptycho/model.py` (TF model contract)


## Resolution (2026-02-05)
- **Data boundary fix:** `save_split_npz()` now stamps `coords_type="relative"` in dataset metadata (grid-lines workflow), and the torch runner selects `coords_relative` explicitly, deriving TF-style offsets when metadata says coords are nominal. This keeps physics code unchanged while enforcing the TF contract at the ingestion boundary.
- **Hard guard:** Lightning dataloader now raises when `object_big=True` but `coords_relative` is missing, preventing silent misuse of absolute coordinates.
- **Regression coverage:**
  - `tests/torch/test_coords_relative_contract.py`
  - `tests/test_grid_lines_workflow.py::TestDatasetPersistence::test_metadata_includes_coords_type`
  - `tests/torch/test_grid_lines_torch_runner.py::TestCoordsRelativeSelection`
  - `tests/torch/test_lightning_dataloader_coords_guard.py`
- **Evidence:**
  - `.artifacts/object_big_relative_offsets/pytest_coords_relative_contract.green.log`
  - `.artifacts/object_big_relative_offsets/pytest_grid_lines_metadata_coords_type.green.log`
  - `.artifacts/object_big_relative_offsets/pytest_grid_lines_coords_relative_selection.log`
  - `.artifacts/object_big_relative_offsets/pytest_lightning_coords_guard.green.log`
  - `.artifacts/object_big_relative_offsets/pytest_grid_lines_hybrid_resnet_integration.log`
- **Parity evidence:** `tmp/patch_parity/object_big_relative_offsets/tensorflow_epoch0.png` and `tmp/patch_parity/object_big_relative_offsets/pytorch_epoch0.png` generated via `patch_parity_helper.py` using:
  - TF source: `outputs/grid_lines_gs1_n128_e50_phi_all/recons/pinn/recon.npz` → `tmp/tf_patch_parity_amp_phase.npz`
  - Torch source: `.artifacts/integration/grid_lines_hybrid_resnet/recons/pinn_hybrid_resnet/recon.npz` → `tmp/torch_patch_parity_amp_phase.npz`
  - Log: `.artifacts/object_big_relative_offsets/patch_parity_helper.log`
- **Integration marker:** `pytest -v -m integration` passes after cherry-picking `ce9743b9` (torch-fix) to initialize `intensity_scale` before TF generator build. Log: `.artifacts/object_big_relative_offsets/pytest_integration_marker.log`
