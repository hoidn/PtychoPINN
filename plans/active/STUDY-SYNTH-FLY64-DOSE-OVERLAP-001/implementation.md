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

### Phase E — Train PtychoPINN (COMPLETE)
**Status:** COMPLETE — PyTorch training runner integration delivered with skip-aware manifest, skip summary persistence, and deterministic CLI evidence

**Backend:** PyTorch (via `ptycho_torch.train.train_cdi_model_torch`) for this phase; TensorFlow backend still supported for production workflows.

**Deliverables:**
- **E1-E2 Job Builder (Attempt #13):** `studies/fly64_dose_overlap/training.py::build_training_jobs()` enumerating 9 jobs per dose (3 doses × 3 variants: baseline gs1 + dense/sparse gs2) with dataset path validation and artifact directory derivation. Test coverage: `test_build_training_jobs_matrix` (1/1 PASSED). CONFIG-001 guard: no params.cfg mutation in builder.
- **E3 Run Helper (Attempt #14-15):** `run_training_job()` orchestrating single job execution with CONFIG-001 bridge (`update_legacy_dict` called before runner invocation), directory creation, runner delegation, and dry-run mode. Test coverage: `test_run_training_job_invokes_runner`, `test_run_training_job_dry_run` (2/2 PASSED with spy-based validation).
- **E4 CLI Integration (Attempt #16):** `studies/fly64_dose_overlap/training.py::main()` with argparse CLI, job filtering (`--dose`, `--view`, `--gridsize`), manifest emission, and artifact orchestration. Test coverage: `test_training_cli_filters_jobs`, `test_training_cli_manifest_and_bridging` (2/2 PASSED).
- **E5 MemmapDatasetBridge Wiring (Attempts #20-22):** Replaced `load_data` stub with `MemmapDatasetBridge` instantiation in `execute_training_job()` (training.py:373-387), extracting `raw_data_torch` payload for trainer delegation. Path alignment: fixed Phase D layout to `dose_{dose}/{view}/{view}_{split}.npz` and introduced `allow_missing_phase_d` flag so CLI can skip absent overlap views while tests stay strict. Test coverage: `test_execute_training_job_delegates_to_pytorch_trainer`, `test_build_training_jobs_skips_missing_view` (3/3 PASSED).
- **E5.5 Skip Reporting (Attempts #23-25):** Enhanced skip reporting with structured metadata: `build_training_jobs()` accepts optional `skip_events` list parameter; when `allow_missing_phase_d=True` and views missing, appends `{dose, view, reason}` dicts (training.py:196-213). CLI `main()` emits skip summary to `skip_summary.json` with schema `{timestamp, skipped_views, skipped_count}`, records path in `training_manifest.json`, and prints human-readable skip count (training.py:692-731). Test coverage: `test_training_cli_manifest_and_bridging` validates skip summary file existence, schema, and manifest consistency (3/3 CLI selectors PASSED).

**Artifact Hubs:**
- E1 Job Builder: `reports/2025-11-04T060200Z/phase_e_training_e1/`
- E3 Run Helper: `reports/2025-11-04T070000Z/phase_e_training_e3/`, `reports/2025-11-04T080000Z/phase_e_training_e3_cli/`
- E4 CLI Integration: `reports/2025-11-04T090000Z/phase_e_training_e4/`
- E5 MemmapDatasetBridge: `reports/2025-11-04T133500Z/phase_e_training_e5/`, `reports/2025-11-04T150500Z/phase_e_training_e5_path_fix/`
- E5.5 Skip Summary: `reports/2025-11-04T161500Z/phase_e_training_e5_real_run/`, `reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/`
- E5 Documentation Sync: `reports/2025-11-04T084850Z/phase_e_training_e5_doc_sync/`

**Key Constraints & References:**
- CONFIG-001 compliance: `update_legacy_dict(params.cfg, config)` called in `execute_training_job` before any data loading or model construction.
- DATA-001 compliance: Phase D NPZ layout enforced via `build_training_jobs` path construction; canonical contract assumed.
- OVERSAMPLING-001: neighbor_count=7 satisfies K≥C=4 for gridsize=2 throughout.
- POLICY-001: PyTorch backend required for Phase E workflows; `backend='pytorch'` set in training config.

**Test Coverage (8 tests, all PASSED):**
- `test_build_training_jobs_matrix` — Job enumeration (9 jobs per dose)
- `test_run_training_job_invokes_runner` — Runner delegation + CONFIG-001 bridge
- `test_run_training_job_dry_run` — Dry-run mode behavior
- `test_training_cli_filters_jobs` — CLI job filtering logic
- `test_training_cli_manifest_and_bridging` — Manifest + skip summary persistence
- `test_execute_training_job_delegates_to_pytorch_trainer` — MemmapDatasetBridge instantiation
- `test_training_cli_invokes_real_runner` — Real runner integration
- `test_build_training_jobs_skips_missing_view` — Graceful skip handling with `allow_missing_phase_d=True`

**Deterministic CLI Baseline Command:**
```bash
python -m studies.fly64_dose_overlap.training \
  --phase-c-root tmp/phase_c_training_evidence \
  --phase-d-root tmp/phase_d_training_evidence \
  --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/real_run \
  --dose 1000 \
  --dry-run
```

**Findings Applied:** CONFIG-001 (builder stays pure; skip_events accumulated client-side), DATA-001 (Phase C/D regeneration reuses canonical contract), POLICY-001 (PyTorch runner default), OVERSAMPLING-001 (skip reasons cite spacing threshold rejections).

#### Execution Guardrails (2025-11-12)
- Reuse `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/` for all dense gs2 and baseline gs1 real-run artifacts until both bundles + SHA256 proofs exist. Do **not** mint new timestamped hubs for this evidence gap.
- Before proposing new manifest/test tweaks, promote the Phase C/D regeneration + training CLI steps into `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_e_job.py` (T2 script) and reference it from future How-To Maps. Prep-only loops are not permitted once the script exists.
- Each new Attempt touching Phase E/G must deliver at least one of:
  * A successful dense gs2 or baseline gs1 training CLI run (stdout must include `bundle_path` + `bundle_sha256`) with artifacts stored under the hub above.
  * A `python -m studies.fly64_dose_overlap.comparison --dry-run=false ...` execution whose manifest captures SSIM/MS-SSIM metrics plus `n_success/n_failed`.
- Training loss guardrail: once a run is visually validated, copy its manifest to `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/reference/training_manifest.json` and treat it as the “golden” baseline. Every new run must execute `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_training_loss.py --reference plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/reference/training_manifest.json --candidate <current_manifest> --dose <value> --view <value> --gridsize <value>` (with optional `--tolerance` override) and archive the checker output next to the manifest. The guardrail fails if `final_loss` is missing, non-finite, or exceeds the reference loss by more than the configured tolerance.

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
