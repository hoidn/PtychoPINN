# spec-ptycho-tracing.md — Tracing and Debug (Normative)

Overview (Normative)
- Purpose: Require sufficient observability and trace hooks to diagnose physics/math divergences and shape/scale regressions.

Tracing Requirements (Normative)
- Forward physics trace:
  - Expose or log intermediate tensors on demand (via debug flags/callbacks):
    • Object field `O`, illuminated field `Ψ`, FFT magnitude `|F|²/(N²)`, amplitude `A`, sampled amplitude `A_obs` (if enabled).
  - Traces SHALL be produced by the same code paths used in production (no re‑derived physics).
- Coordinate/translation trace:
  - Record offset tensors (`coords_offsets`, `coords_relative`, combined `offsets_xy`) and flattened `(B×C,2)` translations for reproduction.
- Scaling invariants:
  - Log `intensity_scale` used and assert X/Y_I symmetric scaling in simulation; emit diagnostics if violated.

First‑Divergence Workflow (Normative)
1) Freeze Poisson sampling and `jit_compile`; compare analytic FFT chain vs `PadAndDiffract` outputs for a single patch and probe.
2) Validate inputs: `N`, `gridsize`, `offset`, padding, amplitude positivity, dtype.
3) Enable stepwise dumps to locate first divergence; verify `fftshift` placement and energy normalization.
4) Only after root cause fix, re‑enable Poisson and XLA to confirm end‑to‑end parity.

Artifacts (Informative)
- Save small npz bundles containing inputs/outputs at each step for offline reproduction.
- For stitching failures, persist per‑channel offsets and border math for the failing case.

