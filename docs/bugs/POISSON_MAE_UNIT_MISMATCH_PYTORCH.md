# Poisson/MAE Unit Mismatch in PyTorch Losses

**Date:** 2026-01-27  
**Status:** Resolved  
**Category:** PyTorch, loss functions, unit consistency

## Summary
In the PyTorch backend (`ptycho_torch/model.py`), Poisson and MAE losses were computed with **unit mismatch**: inputs are diffraction **amplitudes**, but the losses assumed **intensity**. The Poisson loss squared the **predictions** to form the Poisson rate λ but did **not** square the observed input before computing `log_prob`. The MAE loss compared `pred**2` to raw amplitude, mixing intensity and amplitude. This contradicted the documented contract and TensorFlow behavior (where MAE operates on amplitude).

## Evidence
- **Diffraction data is amplitude**
  - TF simulation uses `observe_amplitude` (Poisson on amplitude² → sqrt), so diffraction `X` is amplitude. This is saved as `diffraction` and read by Torch.  
  - Paths: `ptycho/diffsim.py`, `ptycho/workflows/grid_lines_workflow.py`, `ptycho_torch/dataloader.py`

- **PoissonIntensityLayer mismatch**
  - `PoissonIntensityLayer` squared predicted amplitudes to build λ but used observed `x` **without squaring** in `log_prob`, despite docstring stating both should be squared.  
  - Path: `ptycho_torch/model.py`

- **MAELoss mismatch**
  - `MAELoss` computed `L1(pred**2, raw)`, comparing intensity to amplitude if `raw` is amplitude.  
  - TensorFlow uses `mean_absolute_error` on amplitude outputs (per `ptycho/model.py` loss wiring).  
  - Path: `ptycho_torch/model.py`, `ptycho/model.py`

## Expected Behavior
- Poisson loss should compare **intensity**: both observed and predicted amplitudes must be squared before `log_prob`.
- MAE should operate on **amplitude** (as in `ptycho/model.py`), so it must compare like units without squaring one side only.

## Resolution
- `PoissonIntensityLayer.forward` now squares observed amplitudes before `log_prob`.
- `MAELoss.forward` now computes MAE directly on amplitude.
- Added unit tests in `tests/torch/test_loss_units.py` to enforce both contracts.

## Tests
- `pytest tests/torch/test_loss_units.py -v`

## References
- `ptycho_torch/model.py` (PoissonIntensityLayer, PoissonLoss, MAELoss)
- `ptycho/diffsim.py` (diffraction amplitude generation)
- `ptycho/workflows/grid_lines_workflow.py` (saved diffraction data)
- `ptycho/model.py` (TF negloglik + MAE behavior)
