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

## Backend Selection (Policy for this Study)
- PINN training/inference: use the TensorFlow backend (`backend='tensorflow'`).
- Iterative baseline (pty-chi): uses PyTorch internally — acceptable and expected.
- Do not switch the study’s PINN runs to `ptycho_torch/`; use TF workflows (`ptycho_train` / workflows.components) and ensure CONFIG-001 bridge (`update_legacy_dict`) precedes legacy consumers.

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

### Phase B — Test Infrastructure Design (COMPLETE)
**Status:** COMPLETE — validator + 11 tests PASSED with RED/GREEN evidence
- Working Plan: `reports/2025-11-04T025541Z/phase_b_test_infra/plan.md`
- Deliverables:
  - `studies/fly64_dose_overlap/validation.py::validate_dataset_contract` enforcing DATA-001 keys/dtypes, amplitude requirement, spacing thresholds vs design constants, and y-axis split integrity. ✅
  - pytest coverage in `tests/study/test_dose_overlap_dataset_contract.py` (11 tests, all PASSED) with logged red/green runs. ✅
  - Updated documentation (`implementation.md`, `test_strategy.md`, `summary.md`) recording validator scope and findings references (CONFIG-001, DATA-001, OVERSAMPLING-001). ✅
- Artifact Hub: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T025541Z/phase_b_test_infra/`
- Test Summary: 11/11 PASSED, 0 FAILED, 0 SKIPPED
- Execution Proof: red/pytest.log (1 FAILED stub), green/pytest.log (11 PASSED), collect/pytest_collect.log (11 collected)

### Phase C — Dataset Generation (Dose Sweep) (COMPLETE)
**Status:** Complete — orchestration pipeline with 5 tests PASSED (RED→GREEN evidence captured)

**Deliverables:**
- `studies/fly64_dose_overlap/generation.py` orchestrates: simulate → canonicalize → patch → split → validate for each dose ✅
- CLI entry point: `python -m studies.fly64_dose_overlap.generation --base-npz <path> --output-root <path>` ✅
- pytest coverage in `tests/study/test_dose_overlap_generation.py` (5 tests, all PASSED with monkeypatched dependencies) ✅
- Updated documentation and test artifacts ✅

**Pipeline Workflow:**
1. `build_simulation_plan(dose, base_npz_path, design_params)` → constructs dose-specific `TrainingConfig`/`ModelConfig` with n_images from base dataset
2. `generate_dataset_for_dose(...)` → orchestrates 5 stages:
   - Stage 1: Simulate diffraction with `simulate_and_save()` (CONFIG-001 bridge handled internally)
   - Stage 2: Canonicalize with `transpose_rename_convert()` (DATA-001 NHW layout enforced)
   - Stage 3: Generate Y patches with `generate_patches()` (K=7 neighbors per design)
   - Stage 4: Split train/test on y-axis with `split_dataset()` (spatial separation)
   - Stage 5: Validate both splits with `validate_dataset_contract()` (Phase B validator)
3. CLI entry iterates over `StudyDesign.dose_list`, captures logs, writes `run_manifest.json`

**Artifact Hub:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T032018Z/phase_c_dataset_generation/`
**Test Summary:** 5/5 PASSED, 0 FAILED, 0 SKIPPED
**Execution Proof:**
- red/pytest.log (3 FAILED with TypeError on TrainingConfig.gridsize)
- green/pytest.log (5 PASSED after fixing ModelConfig.gridsize separation)
- collect/pytest_collect.log (5 collected)

**Findings Applied:**
- CONFIG-001: `simulate_and_save()` handles `update_legacy_dict(p.cfg, config)` internally
- DATA-001: Canonical NHW layout enforced by `transpose_rename_convert_tool`
- OVERSAMPLING-001: K=7 neighbor_count preserved in patch generation for Phase D

### Phase D — Group-Level Overlap Views (COMPLETE)
**Status:** Complete — D1-D4 delivered with metrics bundle workflow, pytest coverage, and CLI artifact integration.

**Delivered Components:**
- `studies/fly64_dose_overlap/overlap.py:23-89` implementing spacing utilities (`compute_spacing_matrix`, `build_acceptance_mask`) and `generate_overlap_views:124-227` that materializes dense/sparse outputs for each dose (gridsize=2, neighbor_count=7) and emits consolidated `metrics_bundle.json` with per-split metrics paths.
- CLI entry `python -m studies.fly64_dose_overlap.overlap` (lines 230-327) batches Phase C artifacts into `{dense,sparse}_{train,test}.npz` files with spacing metrics + manifest; when `--artifact-root` is set, copies `metrics_bundle.json` into the reports hub for traceability.
- Pytest coverage in `tests/study/test_dose_overlap_overlap.py` with 10 tests validating spacing math (`test_compute_spacing_matrix_*`), orchestration (`test_generate_overlap_views_*`), metrics bundle structure (`test_generate_overlap_views_metrics_manifest`), and failure handling (RED→GREEN evidence: `reports/2025-11-04T034242Z/phase_d_overlap_filtering/{red,green,collect}/`).
- Documentation synchronized: Phase D sections updated in this file, `test_strategy.md`, and `plan.md` D4 marked `[x]`; ledger Attempt #10/11 referencing logs & metrics bundle artifacts.

**Metrics Bundle Workflow (Attempt #10):**
- `generate_overlap_views()` returns `metrics_bundle_path` pointing to aggregated JSON containing `train` and `test` keys, each with `{output_path, spacing_stats_path}` entries.
- CLI main copies bundle from temp output to `<artifact_root>/metrics/<dose>/<view>.json` for archival.
- Test guard `test_generate_overlap_views_metrics_manifest` asserts bundle contains required keys and paths exist on disk.
- Execution proof: `reports/2025-11-04T045500Z/phase_d_cli_validation/` (RED→GREEN logs, CLI transcript, copied bundle).

**Key Constraints & References:**
- Spacing thresholds derived from StudyDesign: dense overlap 0.7 → S≈38.4 px, sparse overlap 0.2 → S≈102.4 px (`docs/GRIDSIZE_N_GROUPS_GUIDE.md:154-172`).
- Oversampling guardrail: neighbor_count=7 ≥ C=4 for gridsize=2 (`docs/SAMPLING_USER_GUIDE.md:112-140`, `docs/findings.md` OVERSAMPLING-001).
- CONFIG-001 boundaries maintained: `overlap.py` loads NPZ via `np.load` only; no params.cfg mutation; validator invoked with `view` parameter.
- DATA-001 compliance ensured via Phase B validator (`validate_fly64_dose_overlap_dataset`) called from `generate_overlap_views`.

**Artifact Hubs:**
- Spacing filter implementation: `reports/2025-11-04T034242Z/phase_d_overlap_filtering/`
- Metrics bundle + CLI validation: `reports/2025-11-04T045500Z/phase_d_cli_validation/`
- Documentation sync: `reports/2025-11-04T051200Z/phase_d_doc_sync/`

**Findings Applied:** CONFIG-001 (pure NPZ loading; legacy bridge deferred to training), DATA-001 (validator enforces canonical NHW layout + dtype/key contracts), OVERSAMPLING-001 (K=7 ≥ C=4 preserved in group construction).

### Phase E — Train PtychoPINN
- Backend: TensorFlow.
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
