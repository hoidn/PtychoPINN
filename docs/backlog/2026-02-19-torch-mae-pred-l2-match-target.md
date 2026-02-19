# Backlog: Torch MAE Prediction L2 Match to Target Intensity

**Created:** 2026-02-19
**Status:** Open
**Priority:** Medium
**Related:** `ptycho_torch/model.py`, `ptycho_torch/config_params.py`, `ptycho/config/config.py`, `docs/backlog/2026-02-10-amplitude-l2-loss-normalization.md`
**Impacts:** Torch `torch_loss_mode=mae` training behavior for `fno`/`hybrid`/`hybrid_resnet`/`cnn`

## Summary
Add an optional Torch MAE loss mode that rescales each predicted diffraction-amplitude sample so its amplitude L2 norm matches the corresponding target sample before MAE is computed.

Target behavior per sample `b`:
`alpha_b = sqrt(sum(target_b^2) / (sum(pred_b^2) + eps))`
`pred_b_norm = alpha_b * pred_b`
`loss = MAE(pred_b_norm, target_b)`

This enforces equal integrated intensity (`sum(amp^2)`) between prediction and target while keeping target scaling unchanged.

## Why
Current MAE can spend optimization capacity on global gain mismatch instead of spatial structure mismatch.

Per-sample prediction-to-target L2 matching may reduce scale confounds and improve training signal when reconstructions are shape-correct but intensity-scaled.

## Proposed Contract
1. Add a new config flag for Torch MAE path, default `False`:
`torch_mae_pred_l2_match_target`.
2. Apply only when `torch_loss_mode == "mae"`.
3. Compute normalization per sample over non-batch dimensions.
4. Use numerical guard `eps` and optional scale clamping to avoid instability on near-zero predictions.
5. Do not modify Poisson/NLL path behavior.
6. Log aggregate scale stats (`alpha` mean/std/min/max) when enabled for diagnostics.

## Acceptance Criteria
1. Feature is opt-in and default-off.
2. Unit tests verify `sum(pred_norm^2) ~= sum(target^2)` within tolerance.
3. Unit tests cover near-zero prediction stability and finite gradients.
4. Poisson path remains unchanged by this flag.
5. A/B experiments show whether this improves patch quality and stitched quality on at least one external-dataset run.
6. Run relevant integration tests after implementation and before defaulting; if metrics/visual quality regress versus baseline, keep the feature disabled and record evidence.

## Risks / Open Questions
1. May hide true calibration errors by making MAE more scale-invariant.
2. Whether `alpha` should be detached from autograd graph or fully differentiable.
3. Interaction with existing `physics_scale` and `intensity_norm_factor` in `compute_loss`.
4. Whether this should apply to supervised MAE path or unsupervised MAE path only.

## Suggested Direction
Run non-invasive evidence-first A/B in `tmp/` with fixed checkpoint, fixed data, and fixed reassembly backend:
1. Baseline MAE.
2. MAE with prediction L2 match to target.

Compare patch-level visuals, stitched visuals, and metrics before promoting to default or broader rollout.
Also run integration suites that exercise `hybrid_resnet`/`pinn_ptychovit` study paths to catch cross-workflow regressions early.
