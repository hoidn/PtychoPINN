# Backlog: probe_big=True Degrades Hybrid ResNet Metrics

**Created:** 2026-02-05
**Status:** Open
**Priority:** Medium
**Related:** `scripts/studies/grid_lines_torch_runner.py`, `ptycho_torch/generators/hybrid_resnet.py`, `ptycho_torch/model.py`
**Impacts:** `tests/torch/test_grid_lines_hybrid_resnet_integration.py::test_grid_lines_hybrid_resnet_metrics`

## Summary

The grid-lines Hybrid ResNet integration test shows a large metric regression when `probe_big=True`, even though the Hybrid ResNet generator should not depend on `probe_big`. This suggests a hidden coupling (e.g., unexpected fallback to the CNN path, or a side effect from config propagation) that should be investigated.

## Evidence

- Regression started when `probe_big` began propagating into the PyTorch training payload (commit `0cf58b67`).
- Observed delta (probe_big=True vs probe_big=False) in the integration test:
  - MAE amplitude +0.0779
  - MAE phase +0.0106
  - SSIM amplitude -0.2582
  - SSIM phase -0.0622
- Logs:
  - `.artifacts/PYTEST-TRIAGE-001/2026-02-05T212836Z/bisect_0cf58b67_retry.log` (probe_big=True)
  - `.artifacts/PYTEST-TRIAGE-001/2026-02-05T212836Z/pytest_probe_big_false.log` (probe_big=False)

## Why This Is Suspicious

- `probe_big` is only referenced in the CNN autoencoder (`Decoder_last.forward` in `ptycho_torch/model.py`).
- Hybrid ResNet uses the generator module (`ptycho_torch/generators/hybrid_resnet.py`) and should bypass that decoder path.

## Hypotheses

1. **Fallback to CNN path:** The Hybrid ResNet generator might not be used in this run (wrong architecture propagation or registry mismatch).
2. **Config side effect:** `probe_big` may be altering `params.cfg` via `populate_legacy_params()`, indirectly affecting stitching or evaluation.
3. **Probe handling coupling:** The dataloader or probe normalization might branch on `probe_big` outside the generator path (needs verification).
4. **Non-determinism / seed drift:** The run might be unstable enough that `probe_big` change correlates with a different outcome rather than causes it.

## Next Steps

1. Add a one-line debug log/assert to confirm the generator class used in the integration test (e.g., `type(model.model.autoencoder)` or registry entry for `hybrid_resnet`).
2. Diff the runtime configs produced by `create_training_payload` when `probe_big` toggles (including params.cfg) to identify other field changes.
3. Search for any `probe_big`-conditioned paths outside `Decoder_last` (dataloader, forward model, stitching).
4. Run a controlled A/B with fixed seed and identical configs to confirm causality.
