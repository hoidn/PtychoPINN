## C4.D3 Debug Notes — coords_relative / C\_forward mismatch

- **Date:** 2025-10-20
- **Supervisor:** galph (planning/debug loop)
- **Focus:** ADR-003-BACKEND-API Phase C4.D3 (integration workflow failure)
- **Artifacts:** reuse hub `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T061500Z/phase_c4_cli_integration_debug/`

### Failure Signal
- Selector: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv`
- Error: `RuntimeError: shape '[16, 2, 1]' is invalid for input of size 8` in `ptycho_torch/helper.py:425` (`Translation` during validation pass).
- Context: occurs after Phase C4.C CLI integrations; regression survived CLI selectors but integration breaks in Lightning validation loop.

### Findings from Triage
- **Factory output inspection:** Running `create_training_payload()` with `gridsize=1` (CLI parity parameters) yields:
  - `payload.pt_data_config.C == 1`
  - `payload.pt_model_config.C_forward == 4`
  - `payload.pt_model_config.object_big == True`
  - (Captured via `python -m ptycho_torch.config_factory` harness; see transcript in supervisor shell history.)
- **Helper expectation:** `hh.reassemble_patches_position_real()` uses `model_config.C_forward` to determine channel count (`C`) before flattening offsets (`offsets_xy.flatten(start_dim=0, end_dim=1)`).
- **Dataset reality:** `PtychoDataset` (memmap path) writes `coords_relative` with shape `(N, C, 1, 2)`. For `gridsize=1`, this becomes `(N, 1, 1, 2)`—only a single channel.
- **Resulting mismatch:** Lightning batches have `inputs` shaped `(B, 1, N, N)` but helper assumes `C = 4`, so `imgs_flat = inputs.flatten(start_dim=0, end_dim=1)` produces `n = B * 4`, whereas `offsets_xy.flatten(...)` only contains `B * 1 * 2` elements. When reshaping to `(n, 2, 1)` PyTorch raises the observed runtime error.
- **Legacy precedent:** TensorFlow pipeline keeps `ModelConfig.C_forward == gridsize**2`. PyTorch defaults rely on dataclass default (`DataConfig.C = 4`), so without explicit override the two configs diverge whenever CLI sets `gridsize=1`.

### Hypotheses
1. **Primary (CONFIRMED):** `create_training_payload()` must synchronize the PyTorch model config's `C_forward` (and `C_model`) with the inferred data-channel count (`C = grid_size[0] * grid_size[1]`). Absence of this override leaves Lightning helpers believing there are four overlapping patches even when CLI requests `gridsize=1`, producing the reshape failure inside `Translation`.  
   - *Evidence:* Scripted payload inspection; helper code path (`ptycho_torch/helper.py:66-90`); dataclass defaults (`ptycho_torch/config_params.py:30-86`).
2. **Secondary (TO WATCH):** Memmap tensors advertise coordinates as `(N, C, 1, 2)` while the TF bridge expects `(N, 1, 2, C)` (`ptycho_torch/data_container_bridge.py:135-200`). This orientation is legacy Torch behaviour and does not trigger the current failure, but once `C_forward` is corrected we should re-evaluate parity against DATA-001 to avoid future mismatches when `C > 1`.

### Next Step Recommendation
1. Update `create_training_payload()` to pass `C_forward = C` (and align `C_model`) when instantiating `PTModelConfig`.  
2. Re-run the targeted integration selector under the CLI (`test_run_pytorch_train_save_load_infer`) to confirm the reshape error disappears.  
3. If regression persists, log the new failure and revisit memmap orientation vs. bridge contract before modifying helper logic.
