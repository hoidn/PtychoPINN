# Backlog: probe_big=True Degrades Hybrid ResNet Metrics

**Created:** 2026-02-05
**Status:** Open
**Priority:** Medium
**Related:** `scripts/studies/grid_lines_torch_runner.py`, `ptycho_torch/generators/hybrid_resnet.py`, `ptycho_torch/model.py`
**Impacts:** `tests/torch/test_grid_lines_hybrid_resnet_integration.py::test_grid_lines_hybrid_resnet_metrics`

## Summary

Initial triage suggested a large metric regression when `probe_big=True`, but follow-up instrumentation shows Hybrid ResNet always uses the generator path and `probe_big` does not affect metrics. The earlier “probe_big effect” was likely a misattribution caused by confounding config changes (notably `object_big`) or baseline drift.

## Evidence

- Regression started when `probe_big` began propagating into the PyTorch training payload (commit `0cf58b67`).
- Observed delta (initial triage, later judged confounded): MAE amp +0.0779, MAE phase +0.0106, SSIM amp -0.2582, SSIM phase -0.0622.
- Logs:
  - `.artifacts/PYTEST-TRIAGE-001/2026-02-05T212836Z/bisect_0cf58b67_retry.log` (probe_big=True)
  - `.artifacts/PYTEST-TRIAGE-001/2026-02-05T212836Z/pytest_probe_big_false.log` (probe_big=False)
- Follow-up instrumentation (same seed, same dataset):
  - `.artifacts/PYTEST-TRIAGE-001/2026-02-05T224903Z/pytest_hybrid_resnet_probe_big_false_debug.log`
  - `.artifacts/PYTEST-TRIAGE-001/2026-02-05T225226Z/pytest_hybrid_resnet_probe_big_true_debug.log`
  - Both runs report: `autoencoder=HybridResnetGeneratorModule generator=HybridResnetGeneratorModule generator_output=real_imag` and identical metrics.

## Why This Is Suspicious

- `probe_big` is only referenced in the CNN autoencoder (`Decoder_last.forward` in `ptycho_torch/model.py`).
- Hybrid ResNet uses the generator module (`ptycho_torch/generators/hybrid_resnet.py`) and should bypass that decoder path.

## Updated Trace

The regression tied to `0cf58b67` coincided with propagating **object_big/probe_big/pad_object** into the PyTorch payload. The grid-lines runner was using TF defaults (`object_big=True`), and `object_big` **does** affect the physics forward model (reassembly path), even for Hybrid ResNet. Later A/B runs with `probe_big` toggled (and `object_big=False` pinned) show no metric change and confirm the generator path is active, so the earlier probe_big attribution was likely confounded by `object_big` (or baseline drift).

## Next Steps

1. Reclassify this as an `object_big`/forward-model parity issue unless new evidence shows `probe_big` coupling.
2. If desired, add a small regression test asserting `probe_big` toggles do not change Hybrid ResNet metrics when `object_big=False`.
3. Close this backlog item if no further probe_big-specific effects are observed.
