# Initiative: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001

## Problem Statement
We want to study PtychoPINN performance on synthetic datasets derived from the fly64 reconstructed object/probe across multiple photon doses, while manipulating inter-group overlap between solution regions. We will compare against a maximum-likelihood iterative baseline (pty-chi LSQML) using MS-SSIM (phase emphasis, amplitude reported) and related metrics. The study must prevent spatial leakage between train and test.

## Objectives
- Generate synthetic datasets from existing fly64 object/probe for multiple photon doses (e.g., 1e3, 1e4, 1e5).
- Construct two overlap views per dose: dense (high inter-group overlap) and sparse (low inter-group overlap), while keeping intra-group neighborhoods tight via K-NN grouping.
- Train PtychoPINN on each condition (gs=1 baseline, gs=2 grouped with K≥C for K-choose-C).
- Reconstruct with pty-chi LSQML (100 epochs to start; parameterizable).
- Run three-way comparisons (PINN, baseline, pty-chi) emphasizing phase MS-SSIM; also report amplitude MS-SSIM, MAE/MSE/PSNR/FRC.
- Record all artifacts under plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/<timestamp>/ and update docs/fix_plan.md per loop.

## Deliverables
1. Dose-swept synthetic datasets with spatially separated train/test splits.
2. Group-level overlap filtering logic (documented and reproducible) and resulting dataset views (dense vs sparse).
3. Trained PINN models per condition (gs1 and gs2 variants) with fixed seeds.
4. pty-chi LSQML reconstructions per condition (100 epochs baseline; tunable).
5. Comparison outputs (plots, aligned NPZs, CSVs) with MS-SSIM (phase, amplitude) and summary tables.
6. Study summary.md aggregating findings per dose/view.

## Phases

### Phase A — Design & Constraints (COMPLETE)
**Status:** Complete — constants encoded in `studies/fly64_dose_overlap/design.py::get_study_design()`

**Dose sweep:**
- Dose list: [1e3, 1e4, 1e5] photons per exposure

**Grouping parameters:**
- Gridsizes: {1, 2}
- Neighbor count K=7 for gs2 (satisfies K ≥ C=4 for K-choose-C)

**Inter-group overlap control:**
- Dense view: f_overlap=0.7 → S ≈ 38.4 pixels
- Sparse view: f_overlap=0.2 → S ≈ 102.4 pixels
- Rule: S = (1 − f_group) × N where N=128 pixels (patch size)
- Filter groups by minimum center spacing to achieve target overlap

**Patch geometry:**
- Patch size N=128 pixels (nominal from fly64 reconstructions)

**Train/test split:**
- Axis: 'y' (top vs bottom halves)
- Ensures spatial separation to prevent leakage

**RNG seeds (reproducibility):**
- Simulation: 42
- Grouping: 123
- Subsampling: 456

**Metrics configuration:**
- MS-SSIM sigma: 1.0
- Emphasize phase: True
- Report amplitude: True
- FRC threshold: 0.5

**Test coverage:**
- `tests/study/test_dose_overlap_design.py::test_study_design_constants` (PASSED)
- `tests/study/test_dose_overlap_design.py::test_study_design_validation` (PASSED)
- `tests/study/test_dose_overlap_design.py::test_study_design_to_dict` (PASSED)

### Phase B — Test Infrastructure Design
- Author test_strategy.md covering dataset contract checks, dose sanity checks, group filtering invariants, and execution proofs (pytest/logs/CSV under reports/).

### Phase C — Dataset Generation (Dose Sweep)
- Use scripts/simulation/simulate_and_save.py with fly64 object/probe to generate synthetic_full.npz per dose with fixed seed.
- Split to train/test with scripts/tools/split_dataset_tool.py --split-axis y; validate contracts.

### Phase D — Group-Level Overlap Views
- For gs2 (C=4), set neighbor_count=7.
- Build dense vs sparse group views by filtering groups using min center spacing S; ensure intra-group neighborhoods remain tight via K-NN grouping.
- When needed, trigger K-choose-C by requesting n_groups > n_subsample (oversampling branch).

### Phase E — Train PtychoPINN
- Run gs1 baseline and gs2 grouped per dose/view with fixed seeds; store configs and logs under reports/.

### Phase F — pty-chi LSQML Baseline
- Run scripts/reconstruction/ptychi_reconstruct_tike.py with algorithm='LSQML', num_epochs=100 per test set; capture outputs.

### Phase G — Comparison & Analysis
- Use scripts/compare_models.py three-way comparisons with --ms-ssim-sigma 1.0 and registration; produce plots, CSVs, aligned NPZs; write per-condition summaries.

### Phase H — Documentation & Gaps
- Note the current code/doc oversampling status and any deviations; update docs/fix_plan.md with artifact paths and outcomes.

## Risks & Mitigations
- pty-chi dependency not vendored: document version and environment; cache outputs.
- Oversampling misuse: confirm K≥C and log branch choice; include spacing histograms.
- Data leakage: enforce y-axis split; avoid mixed halves in train/test.

## Evidence & Artifacts
- All runs produce logs/plots/CSV in reports/<timestamp>/ per condition. Summaries collected in summary.md.

