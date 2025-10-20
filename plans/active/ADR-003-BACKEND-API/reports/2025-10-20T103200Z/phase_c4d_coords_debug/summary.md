# Phase C4.D Offset Tensor Debug Notes (2025-10-20T103200Z)

## Context
- Focus: ADR-003-BACKEND-API Phase C4.D — unblock B2/B3 by fixing Lightning training failure in `_train_with_lightning`.
- Symptom: `RuntimeError: shape '[4, 2, 1]' is invalid for input of size 16` raised inside `ptycho_torch.helper.Translation` during CLI-driven tests (`test_bundle_loader_returns_modules`, `test_run_pytorch_train_save_load_infer`).
- Prior assumption: CLI failed to forward `neighbor_count` override. Factory defaults now emit `K=4`, so remaining failure stems from tensor shape mismatch.

## Evidence Collected
- Reproduced failure via Python harness (no code edits) calling `_train_with_lightning` with factory-produced `TrainingConfig` (`gridsize=1`).
- Instrumented `reassemble_patches_position_real` and `Translation` helpers to capture tensor shapes at failure point.
  - `inputs` (complex patches): `torch.Size([4, 1, 64, 64])` → `n = 4` (channels collapse to 1).
  - `offsets_xy` arriving from dataloader: `torch.Size([4, 1, 2, 1])` (axes order = batch, singleton, xy, grid²).
  - After flattening with current code: `offsets_flat = offsets_xy.flatten(0, 1)` → `torch.Size([4, 2, 1])`.
  - Multiplication with `norm_factor.reshape(1, 1, 2)` broadcasts last dimension from 1 → 2, producing `norm_offset` with **16 elements**.
  - `Translation` then attempts `norm_offset.view(n, 2, 1)` using `n=4`, i.e., target shape `4×2×1 = 8`, causing the observed runtime error.
- Verified that reshaping failure persists even when `gridsize=1`; root cause is axis ordering, not gridsize magnitude.
- Confirmed dataloader emits `coords_relative` tensors shaped `(batch, 1, 2, gridsize²)` (x/y axis before channel axis). Forward model expects `(batch, gridsize², 1, 2)` so that flattening yields `(B·C, 1, 2)` without broadcasting.

## Hypotheses
1. **Primary** — *Axis ordering bug*: Lightning dataloader surfaces `coords_relative` with axes `(batch, 1, 2, C)`. Helper functions assume `(batch, C, 1, 2)`. This misalignment forces broadcasting during normalization, inflates the last dimension, and breaks `view`. Fix: permute to `(batch, C, 1, 2)` (e.g., `coords_rel.permute(0, 3, 1, 2).contiguous()`) inside `_build_lightning_dataloaders` before batches flow into the model. **Confidence: high.**
   - *Next confirming step*: Add targeted unit test that pulls one batch from `_build_lightning_dataloaders` and asserts `coords_relative.shape == (batch_size, gridsize**2, 1, 2)`.
2. **Secondary** — *Legacy container contract drift*: `PtychoDataContainerTorch` currently mirrors TensorFlow ordering `(nsamples, 1, 2, C)`. If we prefer to keep dataloader lean, we could revise the container bridge to transpose once after creation, ensuring all consumers observe the expected shape. Requires auditing other callsites (`extract_channels_from_region`, reassembly beta). **Confidence: medium** (contingent on preferred abstraction layer).

## Recommendations
- Implement Phase C4.D.B2 patch by normalizing `coords_relative` axis order when constructing the tensor dict in `_build_lightning_dataloaders` (and ensure `.contiguous()` before view).
- Introduce a fast regression test (pytest) that:
  1. Builds a training payload via factory (gridsize=2 to exercise multi-channel).
  2. Calls `_build_lightning_dataloaders` and inspects the first batch.
  3. Asserts `batch[0]['coords_relative']` matches `(batch, gridsize**2, 1, 2)` and `.numel() == batch * gridsize**2 * 2` (no broadcasting).
- After fix, rerun targeted selectors already mapped in `input.md` (gridsize parity + integration) and update plan rows B2/B3 accordingly.

## Artifacts
- Reproduction harness (no committed code) logged via supervisor notebook commands.
- Instrumented print traces captured in terminal during this loop (see session transcript; no files added).

## Open Questions
- Should the axis permutation live in `PtychoDataContainerTorch` constructor (`coords_relative = coords_relative.transpose(0,3,1,2)`), guaranteeing all downstream consumers see consistent layout?
- Does any other module depend on the existing `(batch, 1, 2, C)` ordering? Need audit before adjusting bridge layer; dataloader-scoped permutation is the safer short-term fix.
