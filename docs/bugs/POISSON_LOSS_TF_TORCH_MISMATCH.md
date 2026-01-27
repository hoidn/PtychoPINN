# POISSON_LOSS_TF_TORCH_MISMATCH

**Date:** 2026-01-27  
**Status:** Open  
**Category:** Loss definition parity / Training comparability  
**Systems:** TensorFlow (`ptycho/`), PyTorch (`ptycho_torch/`)

## Summary
TensorFlow and PyTorch Poisson loss implementations are not equivalent in value or scale. Both use intensity (squared amplitude), but:

1) **TF drops the `log(y!)` constant term** via `compute_full_loss=False` in `tf.nn.log_poisson_loss`, while
2) **Torch includes `log(y!)`** by using `torch.distributions.Poisson.log_prob`.
3) **Torch additionally normalizes the total loss by `mean(x)`**, which TF does not.

This causes loss values and gradient scales to be non‑comparable between backends, complicating cross‑model comparisons and hyperparameter tuning.

## Evidence
### TensorFlow
- `ptycho/model.py`:
  - `negloglik()` uses `tf.nn.log_poisson_loss(y_true, log(y_pred), compute_full_loss=False)`
  - Model outputs intensity via `SquareLayer` and trains against `(intensity_scale * X)**2`

### PyTorch
- `ptycho_torch/model.py`:
  - `PoissonIntensityLayer` squares **both** prediction and observation before `log_prob`
  - `PoissonLoss` uses `torch.distributions.Poisson.log_prob` (includes `log(y!)`)
  - `compute_loss()` divides total loss by `intensity_norm_factor = mean(x) + 1e-8`

## Impact
- Loss curves and magnitudes are not comparable between TF and Torch runs.
- Torch gradients are scaled relative to TF due to the extra normalization.
- Hyperparameter tuning (learning rate, loss weights) does not transfer cleanly across backends.

## Non‑Issue (Confirmed)
There is **no amplitude/intensity unit mismatch** between TF and Torch Poisson losses; both operate on intensity by squaring amplitude.

## Suggested Follow‑ups
- Decide whether to align TF and Torch Poisson definitions and scaling.
- If alignment is required, update one side to match (including constant-term handling and normalization) and add a parity test.

