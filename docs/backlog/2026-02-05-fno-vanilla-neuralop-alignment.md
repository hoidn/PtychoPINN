# Backlog: Align `fno_vanilla` With NeuralOperator Baseline

## Summary
The current `fno_vanilla` implementation is not a standard NeuralOperator-style FNO. It uses Ptycho-specific `SpatialLifter` and `PtychoBlock` modules, plus an optional fallback FFT spectral conv path. This diverges from the intended “vanilla” baseline and makes comparisons to standard FNO implementations misleading.

## Impact
- **Baseline ambiguity:** “Vanilla FNO” results are not comparable to NeuralOperator baselines or published FNO results.
- **Experiment confusion:** Reports and plots labeled `fno_vanilla` imply a standard model but are actually using custom blocks.
- **Reproducibility risk:** The optional fallback path can change behavior depending on environment.

## Evidence
- `ptycho_torch/generators/fno_vanilla.py` currently imports `SpatialLifter` and `PtychoBlock` from `ptycho_torch/generators/fno.py`.
- The plan `docs/plans/2026-02-05-fno-vanilla-neuralop-alignment.md` specifies the intended standard FNO stack and test coverage.

## Outstanding Issues
1. Replace the `fno_vanilla` block stack with a standard NeuralOperator-style FNO (1×1 lift, spectral+1×1 blocks, GELU).
2. Add coordinate grid channels and padding/crop semantics per the standard FNO baseline.
3. Make `neuralop` a hard dependency for `fno_vanilla` (no fallback spectral path).
4. Update tests and docs to reflect the new definition.

## Suggested Direction
Implement the plan in `docs/plans/2026-02-05-fno-vanilla-neuralop-alignment.md` and validate via `tests/torch/test_fno_generators.py`.

## Related Artifacts
- Plan: `docs/plans/2026-02-05-fno-vanilla-neuralop-alignment.md`
