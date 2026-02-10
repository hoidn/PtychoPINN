# Backlog: Amplitude-L2 Loss Normalization Across Generator Models

**Created:** 2026-02-10
**Status:** Open
**Priority:** Medium
**Related:** `ptycho_torch/model.py`, `ptycho_torch/dataloader.py`, `ptycho/model.py`, `docs/DATA_NORMALIZATION_GUIDE.md`
**Impacts:** `tests/torch/`, `tests/`, training/inference loss behavior for CNN-PINN and generator models

## Summary
Add an optional loss path that applies **amplitude-domain L2 normalization** to both prediction and target for MAE-style objectives in PtychoPINN (CNN-PINN) and other generator models. The normalization is per sample and enforces unit amplitude L2 norm, i.e. unit total intensity (`sum(amp**2)`), so MAE compares structure/shape independent of global probe-driven scale.

For NLL objectives, keep the current physical scaling path: compare absolute, non-normalized intensities after applying the same `nphotons`-derived intensity scale that is already used in the pipeline.

## Why
- Current behavior is sensitive to probe-scale conventions and cross-path scaling differences.
- L2-normalized amplitude MAE can reduce scale-coupling while preserving physically meaningful relative amplitude structure.
- NLL should remain in absolute intensity space to preserve photon-count semantics.

## Proposed Contract
1. MAE path:
   - Convert to amplitude domain as needed.
   - Apply per-sample L2 normalization to input/target and output/prediction.
   - Compute MAE on normalized amplitudes.
2. NLL path:
   - Do not L2-normalize.
   - Apply existing absolute intensity scaling derived from `nphotons`.
   - Compute NLL on scaled, non-normalized intensities.
3. Rollout:
   - Guard new MAE behavior behind an explicit config flag.
   - Do not deprecate current MAE behavior until validation criteria pass.

## Validation Requirements Before Deprecation
1. Add targeted unit tests for loss-path math (MAE normalized vs legacy MAE, NLL unchanged semantics).
2. Run A/B experiments on at least one synthetic and one real-like dataset.
3. Confirm no regression in baseline metrics where scale sensitivity is not the dominant error source.
4. Document migration guidance and default-value policy after evidence review.

## Risks / Open Questions
1. Whether normalization should be per patch, per batch element, or per reconstructed object view.
2. Interaction with existing probe normalization/scaling in TF vs Torch paths.
3. Metric comparability across historic runs trained with legacy MAE.

## Suggested Direction
Create an implementation plan that defines exact insertion points in both TF and Torch loss codepaths, config surface, and test matrix, then execute via RED/GREEN test-driven rollout.
