# Torch Physics Scaling Alignment Design (2026-02-05)

## Overview
This design aligns the PyTorch backend with the TensorFlow physics-scaling semantics. The model should always consume normalized diffraction amplitudes while using a derived, dataset-level `intensity_scale` to map normalized outputs back to photon counts at the physics loss boundary and during inference output scaling.

## Goals
- Derive a single dataset-level `intensity_scale` from the final training subset and target `nphotons`.
- Enforce metadata precedence for `nphotons` (metadata first, then config).
- Preserve existing RMS/statistical normalization for model inputs.
- Use physics scaling only at the physics boundary (Poisson loss and output scaling).
- Persist the derived scale in the run bundle and `hparams.yaml`.

## Non-Goals
- Do not write derived scales back into NPZ metadata.
- Do not change existing RMS/statistical normalization semantics.
- Do not introduce per-batch or per-sample physics scaling.

## Precedence Rules
- `nphotons` comes from dataset metadata when present.
- If metadata is missing `nphotons`, fall back to config.
- Log the source of `nphotons` for auditability.

## Derivation Formula
Let `X_norm` be the normalized diffraction amplitudes used by the model (after sampling and RMS/statistical normalization). The derived scale must satisfy:

```text
intensity_scale = sqrt(nphotons / mean(sum(X_norm**2)))
```

This guarantees:

```text
mean(sum((X_norm * intensity_scale)**2)) == nphotons
```

## Data Flow and Wiring
Training flow:
- Finalize the training subset and apply RMS/statistical normalization.
- Compute a single `intensity_scale` using the formula above.
- Use the derived scale in the Poisson loss boundary.
- If `intensity_scale_trainable` is enabled, apply a trainable multiplicative factor initialized to 1.0 on top of the derived scale.
- Persist `intensity_scale` in the bundle and `hparams.yaml`.

Inference flow:
- Load the stored `intensity_scale` from the bundle.
- Apply the same RMS/statistical normalization as training.
- Use the stored physics scale to map outputs back to photon counts.
- Do not re-derive scale from inference data.

## Sampling-Aware Scale Derivation
- Compute the scale after sampling decisions are finalized.
- If training uses the full dataset, the subset is the full dataset.
- The computation is data-driven and independent of sampling strategy.

## Logging and Auditability
Log the following at training time:
- Resolved `nphotons` and its source (metadata or config).
- The derived `intensity_scale`.
- The number of samples used in the scale computation.
- The normalization mode used for `X_norm`.

## Tests
- Unit test for metadata precedence and deterministic scale derivation.
- Integration test confirming training and inference use the same stored scale.
- Parity check against TensorFlow scaling for the same dataset/subset.

## Implementation Notes
- Compute physics scale from the already-normalized amplitudes used as model input.
- Keep RMS/statistical normalization unchanged.
- Apply physics scaling only at the physics loss boundary and output mapping.
- Persist scale in the run bundle and `hparams.yaml` only.

## Open Questions
None. The trainable scale multiplies the derived scale (baseline physics-aligned value).
