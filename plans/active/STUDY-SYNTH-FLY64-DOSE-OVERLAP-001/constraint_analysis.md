**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Constraint Analysis

## Goals & Scope
- Synthetic-from-fly64 dose sweep; group-level overlap manipulation; PINN vs pty-chi (LSQML) comparison.

## Constraints
- Time: multi-condition runs; prioritize a small matrix first (e.g., 3 doses × 2 views × {gs1, gs2}).
- Compute: CPU acceptable for simulation; training benefits from GPU but not required for scaffolding; pty-chi prefers CUDA when available.
- Data: Start from `datasets/fly64/fly64_shuffled.npz`; enforce y-axis split to prevent leakage.
- Dependencies: pty-chi not vendored; document environment/version where used.
- Reproducibility: Fix seeds (sim, grouping); record in metadata and logs.

## Risks
- Oversampling misuse or doc mismatch: code implements K-choose-C; some docs say “planned”. Mitigate by logging branch choice and documenting in summary.
- Spatial leakage: rigorously apply y-axis split; verify ranges.
- Run time creep: parameterize epochs; start with 100 for pty-chi; PINN epochs sized to budget.

## Success Criteria
- Valid datasets for all planned doses and views, with contract compliance.
- PINN and pty-chi reconstructions produced for each condition (or documented SKIP where not feasible).
- Comparison CSVs + plots produced; phase MS-SSIM reported.
- All evidence paths recorded in docs/fix_plan.md attempts.

## Phase D Policy Update (Spec-Forward)
- Control sampling explicitly via `s_img` and `n_groups`; do not use spacing/packing acceptance gates.
- Report measured overlaps using `specs/overlap_metrics.md` Metric 1/2/3; for `gridsize=1`, skip Metric 1.
- Default `neighbor_count=6` (excluding seed) is acceptable; parameterize in tests/CLI.
- Dense/sparse labels are deprecated for this study; manifests must carry explicit parameters and measured metrics.
