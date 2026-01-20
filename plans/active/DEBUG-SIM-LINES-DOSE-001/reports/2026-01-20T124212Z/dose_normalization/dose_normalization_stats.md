# Dose Normalization Statistics

**Scenario:** dose_legacy_gs2
**Timestamp:** 2026-01-20T13:03:46.034605+00:00

## Spec Reference

Per `specs/spec-ptycho-core.md §Normalization Invariants`:

- Dataset-derived mode (preferred): `s = sqrt(nphotons / E_batch[Σ_xy |Ψ|²])`
- Closed-form fallback: `s ≈ sqrt(nphotons) / (N/2)`

In both modes symmetry SHALL hold:
- Training inputs: `X_scaled = s · X`
- Labels: `Y_amp_scaled = s · X` (amplitude), `Y_int = (s · X)^2` (intensity)

## Intensity Scales

| Source | Value |
| --- | ---: |
| Dataset-derived | 262.776066 |
| Closed-form fallback | 988.211769 |
| Ratio (dataset/closedform) | 0.265911 |

## Stage Flow

| Stage | Mean |
| --- | ---: |
| Raw diffraction | 1.412345 |
| Grouped diffraction | 1.450075 |
| Grouped X (normalized) | 0.376111 |
| Container X | 0.377452 |

## Stage Ratios

| Transition | Ratio |
| --- | ---: |
| Raw diffraction → Grouped diffraction | 1.026714 |
| Grouped diffraction → Grouped X (normalized) | 0.259373 |
| Grouped X (normalized) → Container X | 1.003567 |

## Largest Drop

**Grouped diffraction → Grouped X (normalized)** (ratio=0.259373)