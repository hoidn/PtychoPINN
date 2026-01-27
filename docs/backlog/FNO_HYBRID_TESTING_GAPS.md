# FNO/Hybrid Generator Testing Gaps

**Created:** 2026-01-26
**Status:** Open
**Priority:** Medium
**Related:** `ptycho_torch/generators/fno.py`, `tests/torch/test_fno_generators.py`

## Summary

The FNO and Hybrid U-NO generator implementations have unit tests covering basic functionality but lack end-to-end validation. This document tracks the testing gaps that should be addressed before production use.

## Current Test Coverage

| Component | Tests | Status |
|-----------|-------|--------|
| `SpatialLifter` | Shape preservation, custom hidden | ✅ Passing |
| `PtychoBlock` | Shape preservation, residual connection | ✅ Passing |
| `HybridUNOGenerator` | Output shape, real/imag contract | ✅ Passing |
| `CascadedFNOGenerator` | Output shape, differentiability | ✅ Passing |
| Generator Registry | Resolve + build_model | ✅ Passing |

**Total:** 12 unit tests passing

## Testing Gaps

### 1. End-to-End Training with Real Data

**Gap:** No test verifies that FNO/hybrid models can train on actual ptychography data and converge.

**What's needed:**
- Integration test with small synthetic dataset (e.g., 4 images, 10 epochs)
- Verify loss decreases over training
- Verify model produces non-trivial reconstructions

**Suggested test location:** `tests/torch/test_fno_integration.py`

### 2. Physics Correctness / Reconstruction Quality

**Gap:** No test verifies that FNO/hybrid reconstructions are physically meaningful or comparable to CNN baseline.

**What's needed:**
- Compare FNO/hybrid SSIM/MSE to CNN baseline on identical test data
- Verify amplitude/phase outputs are in expected ranges
- Test with known ground truth (synthetic data with known object)

**Suggested test location:** `tests/torch/test_fno_reconstruction_quality.py`

### 3. Integration with Lightning Trainer

**Gap:** The `grid_lines_torch_runner.py` scaffold doesn't fully integrate with Lightning training loop.

**What's needed:**
- Test that FNO/hybrid generators work with `_train_with_lightning()`
- Verify checkpoint saving/loading works
- Test MLflow logging integration (if applicable)

**Suggested test location:** `tests/torch/test_fno_lightning_integration.py`

### 4. Actual Spectral Convolution (neuraloperator)

**Gap:** Tests use `_FallbackSpectralConv2d` when `neuraloperator` is not installed. The actual `SpectralConv` from neuraloperator is untested.

**What's needed:**
- CI environment with neuraloperator installed
- Test that `SpectralConv` import path works
- Verify numerical equivalence between fallback and real implementation (within tolerance)

**Suggested approach:**
```python
@pytest.mark.skipif(not HAS_NEURALOPERATOR, reason="neuraloperator not installed")
def test_spectral_conv_with_neuraloperator():
    ...
```

## Acceptance Criteria

Before marking FNO/hybrid as production-ready:

- [ ] End-to-end training test passes on synthetic data
- [ ] Reconstruction quality within 20% of CNN baseline (SSIM)
- [ ] Lightning integration test passes
- [ ] CI includes neuraloperator and tests pass

## Related Files

- `ptycho_torch/generators/fno.py` - FNO/Hybrid implementations
- `ptycho_torch/generators/registry.py` - Generator registry
- `scripts/studies/grid_lines_torch_runner.py` - CLI runner (scaffold)
- `tests/torch/test_fno_generators.py` - Current unit tests
- `docs/architecture_torch.md` §4.1 - Architecture documentation

## Notes

The current implementation uses a fallback spectral convolution (`_FallbackSpectralConv2d`) that performs FFT-based spectral mixing. This is functionally similar to neuraloperator's `SpectralConv` but may differ in:
- Numerical precision
- Memory efficiency
- Support for higher modes

The fallback is suitable for development and testing but production use should verify with the actual neuraloperator library.
