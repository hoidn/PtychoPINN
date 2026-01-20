# Capture Summary

**Scenario:** dose_legacy_gs2
**Timestamp:** 2026-01-20T12:51:27.505993+00:00

## Configuration

- Total images: 1024
- Group count: 64
- Neighbor count: 5
- Gridsize: 2
- nphotons: 1.00e+09
- Probe mode: custom
- Probe scale: 4.0

## Intensity Scale Summary

- Dataset-derived: 262.776066
- Closed-form fallback: 988.211769
- Ratio: 0.265911

## CONFIG-001 Bridge

Per SIM-LINES-CONFIG-001 and CONFIG-001, `update_legacy_dict(params.cfg, config)` was called
before grouping and container creation to ensure legacy modules see correct parameters.

## Spec Citation

Per `specs/spec-ptycho-core.md §Normalization Invariants`:

> Dataset-level `intensity_scale` `s` is a learned or fixed parameter used symmetrically.
> Two compliant calculation modes are allowed:
> 1) Dataset-derived mode (preferred): `s = sqrt(nphotons / E_batch[Σ_xy |Ψ|²])`
> 2) Closed-form fallback: `s ≈ sqrt(nphotons) / (N/2)`

## Artifacts

- `capture_config.json`: Full capture configuration
- `dose_normalization_stats.json`: Intensity scale and stage statistics (JSON)
- `dose_normalization_stats.md`: Intensity scale and stage statistics (Markdown)
- `intensity_stats.json`: Detailed stage telemetry (JSON)
- `intensity_stats.md`: Detailed stage telemetry (Markdown)