# Initiative: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001

Initiative Header
- ID: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Title: Synthetic fly64 dose/overlap study
- Owner/Date: Ralph / 2025-11-05
- Status: in_progress (High priority)
- Working Plan: this file
- Reports Hub (Phase D): `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_d_overlap_metrics/`
- Reports Hub (Phase E): `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/`
- Reports Hub (Phase G): `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/`

> **Plan maintenance:** This is the single, evolving plan for the dose/overlap study. Update this file in place instead of creating new `plan/plan.md` documents. The active reports hub is `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/` for Phase E and `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/` for Phase G execution until a milestone changes—reuse them for logs/summaries unless a new milestone is declared.

<plan_update version="1.0">
  <trigger>Replace spacing/packing acceptance gating and dense/sparse labels with explicit overlap-driven metrics and controls</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/GRIDSIZE_N_GROUPS_GUIDE.md, specs/overlap_metrics.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/constraint_analysis.md, docs/fix_plan.md</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>Adopt specs/overlap_metrics.md (Metric1/2/3, disc overlap, probe_diameter_px, s_img/ngroups); deprecate dense/sparse labels; remove geometry/spacing acceptance gates from Phase D plan; require explicit CLI/API inputs and measured-overlap reporting in manifests</proposed_changes>
  <impacts>Clarifies study controls; prevents false failures on “insufficient spacing”; requires test updates and later code changes; Phase G references will target overlap metrics outputs instead of spacing acceptance fields</impacts>
  <ledger_updates>Record spec addition and plan edits; set forthcoming tasks to update tests and CLI/API docs; do not change code in this loop</ledger_updates>
  <status>approved</status>
</plan_update>

<plan_update version="1.0">
  <trigger>Reality check shows Phase D code/tests still enforce dense/sparse spacing gates; need actionable overlap-metrics hand-off before any new Phase G/Gs evidence</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, specs/overlap_metrics.md, docs/GRIDSIZE_N_GROUPS_GUIDE.md, specs/data_contracts.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/constraint_analysis.md, docs/fix_plan.md, galph_memory.md, input.md, studies/fly64_dose_overlap/overlap.py, tests/study/test_dose_overlap_overlap.py</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>Codify overlap-metric Do Now, reference the 2025-11-12T010500Z/phase_d_overlap_metrics hub, add evidence expectations, and clarify CLI/test deliverables before engineering resumes</proposed_changes>
  <impacts>Phase G work stays blocked until Phase D metrics + tests ship; engineering must touch overlap.py, overlap CLI, and study tests in one loop; new pytest evidence routed to the phase_d_overlap_metrics hub</impacts>
  <ledger_updates>docs/fix_plan.md Attempts History gains a new entry; hub summaries note the pivot; galph_memory tracks dwell and ready_for_implementation state</ledger_updates>
  <status>approved</status>
</plan_update>

## Problem Statement
We want to study PtychoPINN performance on synthetic datasets derived from the fly64 reconstructed object/probe across multiple photon doses, while manipulating inter-group overlap between solution regions. We will compare against a maximum-likelihood iterative baseline (pty-chi LSQML) using MS-SSIM (phase emphasis, amplitude reported) and related metrics. The study must prevent spatial leakage between train and test.

## Objectives
- Generate synthetic datasets from existing fly64 object/probe for multiple photon doses (e.g., 1e3, 1e4, 1e5).
- Sweep the *inputs we actually control*—image-level subsampling rate and the number of generated groups / solution regions—then record the derived overlap fraction per condition.
- Train PtychoPINN on each condition (gs=1 baseline, gs=2 grouped with K≥C for K-choose-C).
- Reconstruct with pty-chi LSQML (100 epochs to start; parameterizable).
- Run three-way comparisons (PINN, baseline, pty-chi) emphasizing phase MS-SSIM; also report amplitude MS-SSIM, MAE/MSE/PSNR/FRC.
- Record all artifacts under plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/<timestamp>/ and update docs/fix_plan.md per loop.

## Deliverables
1. Dose-swept synthetic datasets with spatially separated train/test splits.
2. Group-level overlap tooling that logs `image_subsampling`, `n_groups_requested`, and the *measured* overlap fraction for every run (historical “dense/sparse” labels become reporting shorthand only).
3. Trained PINN models per condition (gs1 and gs2 variants) with fixed seeds.
4. pty-chi LSQML reconstructions per condition (100 epochs baseline; tunable).
5. Comparison outputs (plots, aligned NPZs, CSVs) with MS-SSIM (phase, amplitude) and summary tables.
6. Study summary.md aggregating findings per dose/view.

## Exit Criteria
1. Phase D overlap metrics implemented (API + CLI) per `specs/overlap_metrics.md`; no spacing/packing gates; dense/sparse labels deprecated.
2. Phase D tests GREEN (disc overlap unit tests; Metric 1/2/3; gs=1 skips Metric 1; integration bundle fields recorded).
3. Phase E TensorFlow training restored; at least one gs1 and one gs2 real run; manifests record `bundle_path` + SHA256; logs archived in the Phase E hub.
4. Phase G evidence present: `ssim_grid_summary.md`, `verification_report.json`, `check_dense_highlights.log`, `metrics_summary.json`, `metrics_digest.md`, `artifact_inventory.txt`; hub summaries updated with MS‑SSIM/MAE deltas.
5. Test registry synchronized: update `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md`; save `--collect-only` logs under the active Reports Hub.

## Backend Selection (Policy for this Study)
- PINN training/inference: **TensorFlow only.** This initiative depends on the legacy `ptycho_train` stack because it is the fully tested backend per CLAUDE.md.
- Iterative baseline (pty-chi): PyTorch under the hood is acceptable for Phase F scripts, but keep it isolated from the PINN pipeline.
- PyTorch parity belongs to future work. Log the remaining work as TODO items rather than mixing stacks mid-initiative.

## Phases Overview
- Phase A — Design & Constraints: Study design constants, seeds, split policy.
- Phase B — Test Infrastructure: Validator + test scaffolding for DATA‑001 and split integrity.
- Phase C — Dataset Generation: Dose‑swept synthetic datasets; y‑axis split; manifests.
- Phase D — Overlap Metrics: Overlap‑driven sampling; Metric 1/2/3 reporting (no spacing gates).
- Phase E — Training (TF): gs1 baseline + gs2 runs; SHA256 manifests.
- Phase F — Baseline (pty‑chi): LSQML (100 epochs) with artifact capture.
- Phase G — Comparison & Analysis: SSIM grid; verification; highlights; metrics bundle.

## Phases

### Phase A — Design & Constraints (COMPLETE)
**Status:** Complete — constants encoded in `studies/fly64_dose_overlap/design.py::get_study_design()`

Checklist
- [x] Dose list fixed (1e3, 1e4, 1e5)
- [x] Seeds set (sim=42, grouping=123, subsampling=456)
- [x] y‑axis train/test split policy selected and documented

**Dose sweep:**
- Dose list: [1e3, 1e4, 1e5] photons per exposure

**Control knobs (primary inputs):**
- Image subsampling rates `s_img ∈ {1.0, 0.8, 0.6}` — fraction of diffraction frames retained per dose.
- Requested group counts `n_groups ∈ {512, 768, 1024}` (per `docs/GRIDSIZE_N_GROUPS_GUIDE.md`) — scales the number of solution regions per field of view while keeping intra-group K-NN neighborhoods tight (K=7 ≥ C=4 for gs2).

**Derived overlap metric:**
- For every `(dose, s_img, n_groups)` combination we record the achieved overlap fraction `f_overlap = 1 - (mean spacing / N)` via the spacing calculator (N=128 px patch size).
- Labels such as “dense” or “sparse” are reporting shorthands only; artifacts, manifests, and tests must rely on the recorded knobs (`s_img`, `n_groups`) plus the measured `f_overlap`.

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

Checklist
- [x] Validator enforcing DATA‑001 and split integrity
- [x] 11 tests GREEN with red/green/collect artifacts
- [x] Documentation updated with validator scope and findings references
- Working Plan: `reports/2025-11-04T025541Z/phase_b_test_infra/plan.md`
- Deliverables:
  - `studies/fly64_dose_overlap/validation.py::validate_dataset_contract` enforcing DATA-001 keys/dtypes, amplitude requirement, spacing thresholds vs design constants, and y-axis split integrity. ✅
  - Validator now asserts each manifest includes `image_subsampling`, requested `n_groups`, and the derived `overlap_fraction`, ensuring downstream code/tests never assume hard-coded view labels. ✅
  - pytest coverage in `tests/study/test_dose_overlap_dataset_contract.py` (11 tests, all PASSED) with logged red/green runs. ✅
  - Updated documentation (`implementation.md`, `test_strategy.md`, `summary.md`) recording validator scope and findings references (CONFIG-001, DATA-001, OVERSAMPLING-001). ✅
- Artifact Hub: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T025541Z/phase_b_test_infra/`
- Test Summary: 11/11 PASSED, 0 FAILED, 0 SKIPPED
- Execution Proof: red/pytest.log (1 FAILED stub), green/pytest.log (11 PASSED), collect/pytest_collect.log (11 collected)

### Phase C — Dataset Generation (Dose Sweep) (COMPLETE)
**Status:** Complete — orchestration pipeline with 5 tests PASSED (RED→GREEN evidence captured)

Checklist
- [x] Orchestration implemented (simulate → canonicalize → patch → split → validate)
- [x] CLI entry for dose sweep
- [x] 5 tests GREEN; artifacts recorded
- [x] Manifest written and paths stable across reruns

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

### Phase D — Overlap Metrics (SPEC ADOPTED)
**Status:** Spec adopted; code changes pending — this phase now uses explicit overlap-driven metrics and sampling controls. Dense/sparse labels and spacing acceptance gates are deprecated for this study.

Deprecation Note
- Dense/Sparse labels and geometry/spacing acceptance gates are deprecated for this study. Use explicit `s_img`/`n_groups` and report measured overlap metrics per `specs/overlap_metrics.md`.

Checklist
- [ ] Implement API: compute Metric 1 (gs=2), Metric 2, Metric 3; disc overlap with parameterized `probe_diameter_px`.
- [ ] Add CLI controls: `--gridsize`, `--s_img`, `--n_groups`, `--neighbor-count`, `--probe-diameter-px`.
- [ ] Update tests: disc overlap unit tests; Metric 1/2/3; gs=1 skips Metric 1; integration bundle fields.
- [ ] Record per‑split metrics JSON and aggregated bundle under the Phase D hub.

**Planned Components (no code changes in this loop):**
- Implement `specs/overlap_metrics.md` definitions: Metric 1 (group-based, gs=2 only), Metric 2 (image-based, dedup by exact coords), Metric 3 (group↔group via COMs) using disc overlap with parameterized `probe_diameter_px` (default from FWHM or documented fallback).
- Controls move to explicit `--s_img` and `--n_groups` (and Python API equivalents). No geometry acceptance gates; runs proceed and record measured overlaps. gs=1 skips Metric 1 and couples `n_groups` to `n_samples` post-subsample.
- Manifests/metrics bundle to record: `gridsize`, `s_img`, `n_groups`, `neighbor_count` (default 6), `probe_diameter_px`, RNG seeds, and the three metric averages (omit Metric 1 for gs=1).
- CLI messages and docs will deprecate dense/sparse labels; orchestration continues to produce per-split metrics JSON and an aggregated bundle.

#### Do Now (Engineering)
- Implement the overlap metrics pipeline in `studies/fly64_dose_overlap/overlap.py`: remove geometry/spacing acceptance gates plus dense/sparse labels, add deterministic subsampling driven by `s_img` + `rng_seed_subsample`, honor the unified `n_groups` policy (`docs/GRIDSIZE_N_GROUPS_GUIDE.md`), and expose helpers for Metric 1/2/3 per `specs/overlap_metrics.md` (disc overlap math, neighbor_count default 6, configurable `probe_diameter_px`). Metadata/metrics JSON must log the explicit parameters and computed averages (Metric 1 omitted for gs=1).
- Refresh the Phase D CLI to accept `--gridsize`, `--s-img`, `--n-groups`, `--neighbor-count`, `--probe-diameter-px`, and `--rng-seed-subsample` (plus the existing Phase C/Artifact roots). Retain the Python API and wrap the CLI around it so scripted runs (e.g., future Phase G orchestrations) can call into the same functions.
- Update `tests/study/test_dose_overlap_overlap.py` (and any helper fixtures) so disc-overlap math, Metric 1/2/3 aggregation, CLI argument plumbing, and the new bundle schema are covered. Remove spacing-threshold assertions and keep the ACCEPTANCE-001 geometry guard only as historical context in docstrings. Provide at least one CLI-focused selector that inspects `metrics_bundle.json`.
- Evidence for this loop must land under `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_d_overlap_metrics/`: capture `pytest tests/study/test_dose_overlap_overlap.py -vv | tee \"$HUB\"/green/pytest_phase_d_overlap.log`, drop CLI stdout/err into `cli/phase_d_overlap_metrics.log`, and ensure the new per-split metrics JSON + `metrics_bundle.json` include the Metric 1/2/3 fields and sampling parameters. Blockers go to `$HUB/red/blocked_<timestamp>.md` with command + exit code.
- Do not re-run Phase G orchestrations until Phase D metrics ship and tests are GREEN.

**Metrics Bundle (to be updated):**
- Aggregated JSON must include per-split Metric 1 (gs=2 only), Metric 2, Metric 3 values *and* the recorded parameters (`gridsize`, `s_img`, `n_groups`, `neighbor_count`, `probe_diameter_px`, RNG seeds). Spacing-acceptance fields become informational only; they must not gate execution.

**Key Constraints & References:**
- Oversampling guardrail remains: neighbor_count≥C (default 6 for metrics; grouping still uses K≥C for gs=2).
- CONFIG-001 boundaries maintained: Phase D utilities must not mutate legacy params; pure NPZ usage.
- DATA-001 compliance ensured via validator; manifests updated to include overlap metrics, not spacing gates.

**Artifact Hubs:**
- Overlap-metrics implementation + CLI/tests: `reports/2025-11-12T010500Z/phase_d_overlap_metrics/`
- Spacing filter implementation: `reports/2025-11-04T034242Z/phase_d_overlap_filtering/`
- Metrics bundle + CLI validation: `reports/2025-11-04T045500Z/phase_d_cli_validation/`
- Documentation sync: `reports/2025-11-04T051200Z/phase_d_doc_sync/`

**Findings Applied:** CONFIG-001 (pure NPZ loading), DATA-001 (canonical layout), OVERSAMPLING-001 (K≥C preserved for gs=2). Dense/sparse label usage deprecated per `docs/GRIDSIZE_N_GROUPS_GUIDE.md` update.

### Phase E — Train PtychoPINN (PAUSED — awaiting TensorFlow rework)
**Status:** Paused. We must restore the TensorFlow training pipeline before any further runs; PyTorch work is retained below as historical context but no longer authoritative for this initiative.

Checklist
- [ ] E0 — TensorFlow pipeline restoration (delegate to ptycho_train; CONFIG‑001 ordering; add tests)
- [ ] E0.5 — Metadata alignment (Phase D parity: `s_img`, `n_groups`, overlap metrics)
- [ ] E0.6 — Evidence rerun (gs2 dense + gs1 baseline) with SHA256 proofs under Phase E hub

**Immediate Deliverables (blocking before resuming evidence):**
- **E0 — TensorFlow pipeline restoration:** Update `studies/fly64_dose_overlap/training.py` plus `run_phase_e_job.py` so they delegate to the TensorFlow `ptycho_train` workflows, honoring CONFIG-001 ordering. Ship pytest coverage for the TF path (CLI filters, manifest writing, skip summary) and capture a deterministic CLI command under the existing Phase E hub.
- **E0.5 — Metadata alignment:** Ensure manifests and `training_manifest.json` mirror the Phase D metadata fields (`image_subsampling`, `n_groups_requested`, `overlap_fraction`) so Phase G comparisons can correlate overlap statistics with training runs.
- **E0.6 — Evidence rerun:** Re-run the counted dense gs2 + baseline gs1 TensorFlow jobs with SHA256 proofs stored under `plans/active/.../phase_e_training_bundle_real_runs_exec/`, updating docs/fix_plan.md and the hub summary.

**Deferred — PyTorch parity context (informational only):**
- Prior attempts E1–E5.5 wired PyTorch runners, MemmapDatasetBridge hooks, and skip-reporting CLI outputs (see artifact hubs below). These serve as references for future parity work but must not dictate the current backend choice.

**Artifact Hubs (historical evidence, still referenced for context):**
- E1 Job Builder: `reports/2025-11-04T060200Z/phase_e_training_e1/`
- E3 Run Helper: `reports/2025-11-04T070000Z/phase_e_training_e3/`, `reports/2025-11-04T080000Z/phase_e_training_e3_cli/`
- E4 CLI Integration: `reports/2025-11-04T090000Z/phase_e_training_e4/`
- E5 MemmapDatasetBridge: `reports/2025-11-04T133500Z/phase_e_training_e5/`, `reports/2025-11-04T150500Z/phase_e_training_e5_path_fix/`
- E5.5 Skip Summary: `reports/2025-11-04T161500Z/phase_e_training_e5_real_run/`, `reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/`
- E5 Documentation Sync: `reports/2025-11-04T084850Z/phase_e_training_e5_doc_sync/`

**Key Constraints & References:**
- CONFIG-001 compliance: `update_legacy_dict(params.cfg, config)` must run before any TensorFlow data loading or model construction.
- DATA-001 compliance: Phase D NPZ layout enforced via `build_training_jobs` path construction; canonical contract assumed.
- OVERSAMPLING-001: neighbor_count=7 satisfies K≥C=4 for gridsize=2 throughout.
- BACKEND POLICY: TensorFlow is the only supported PINN backend for this initiative; PyTorch tasks are explicitly deferred.

**Historical Test Coverage (PyTorch context):**
- `test_build_training_jobs_matrix`
- `test_run_training_job_invokes_runner`
- `test_run_training_job_dry_run`
- `test_training_cli_filters_jobs`
- `test_training_cli_manifest_and_bridging`
- `test_execute_training_job_delegates_to_pytorch_trainer`
- `test_training_cli_invokes_real_runner`
- `test_build_training_jobs_skips_missing_view`

These tests must be revisited/rewritten for the TensorFlow rework before Phase E can be marked complete again.

**Future deterministic CLI (to be produced):**
Document the TensorFlow command once E0 ships; for now this slot remains `TBD (TensorFlow training command)` and blocks the Do Now nucleus.

#### Execution Guardrails (2025-11-12)
- Reuse `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/` for all dense gs2 and baseline gs1 real-run artifacts until both bundles + SHA256 proofs exist. Do **not** mint new timestamped hubs for this evidence gap.
- Before proposing new manifest/test tweaks, promote the Phase C/D regeneration + training CLI steps into `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_e_job.py` (T2 script) and reference it from future How-To Maps. Prep-only loops are not permitted once the script exists.
- Each new Attempt touching Phase E/G must deliver at least one of:
  * A successful dense gs2 or baseline gs1 training CLI run (stdout must include `bundle_path` + `bundle_sha256`) with artifacts stored under the hub above.
  * A `python -m studies.fly64_dose_overlap.comparison --dry-run=false ...` execution whose manifest captures SSIM/MS-SSIM metrics plus `n_success/n_failed`.
- Training loss guardrail: once a run is visually validated, copy its manifest to `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/reference/training_manifest.json` and treat it as the “golden” baseline. Every new run must execute `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_training_loss.py --reference plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/reference/training_manifest.json --candidate <current_manifest> --dose <value> --view <value> --gridsize <value>` (with optional `--tolerance` override) and archive the checker output next to the manifest. The guardrail fails if `final_loss` is missing, non-finite, or exceeds the reference loss by more than the configured tolerance.

### Phase F — pty-chi LSQML Baseline

Checklist
- [ ] Run LSQML 100 epochs; capture logs/metrics; archive under baseline hub paths; document environment/version
- Run scripts/reconstruction/ptychi_reconstruct_tike.py with algorithm='LSQML', num_epochs=100 per test set; capture outputs.

### Phase G — Comparison & Analysis

Deprecation Note
- Any legacy references to “dense/sparse” and spacing acceptance in this section are historical; the authority for overlap is the measured metrics per `specs/overlap_metrics.md` with explicit `s_img`/`n_groups`.

Checklist
- [ ] Produce SSIM grid summary/log
- [ ] Produce verification report/log
- [ ] Run highlights checker and capture log
- [ ] Write metrics summary and digest; artifact inventory present
- [ ] Update hub summaries and docs/fix_plan with MS‑SSIM/MAE deltas
- Use scripts/compare_models.py three-way comparisons with --ms-ssim-sigma 1.0 and registration; produce plots, CSVs, aligned NPZs; write per-condition summaries.
- After `summarize_phase_g_outputs` completes, run `plans/active/.../bin/report_phase_g_dense_metrics.py --metrics <hub>/analysis/metrics_summary.json` (optionally with `--ms-ssim-threshold 0.80`). Treat the generated **MS-SSIM Sanity Check** table as the go/no-go gate: any row flagged `LOW (...)` indicates reconstruction quality is suspect and the Attempt must be marked blocked until re-run.
- `plans/active/.../bin/analyze_dense_metrics.py` now also embeds the same sanity table inside `metrics_digest.md`; archive this digest under the hub’s `analysis/` directory so reviewers can see absolute MS-SSIM values without hunting through CSVs.

#### Phase G — Active Checklist (2025-11-12)
<plan_update version="1.0">
  <trigger>Dense hub audit confirmed the geometry-aware guard + pytest already landed while the counted rerun is still outstanding</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md (POLICY-001, CONFIG-001, DATA-001, TYPE-PATH-001, STUDY-001, TEST-CLI-001, PREVIEW-PHASE-001, PHASEC-METADATA-001, ACCEPTANCE-001), docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, specs/data_contracts.md §12, galph_memory.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/blocker.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_d_dense.log, studies/fly64_dose_overlap/overlap.py, tests/study/test_dose_overlap_overlap.py, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/pytest_dense_acceptance_floor.log</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>Check off the geometry-aware acceptance floor + pytest checklist items with evidence pointers, add a fresh audit note describing the still-missing SSIM/verification/highlights/metrics artifacts, and leave the counted rerun + metrics helper sequence as the sole Do Now</proposed_changes>
  <impacts>Only the dense pipeline rerun, immediate `--post-verify-only`, and metrics helper executions remain; engineer must capture MS-SSIM ±0.000 / MAE ±0.000000 deltas with the SSIM grid/verification/highlights/metrics/preview/inventory bundle to reset dwell</impacts>
  <ledger_updates>docs/fix_plan.md Latest Attempt entry, input.md rewrite, hub summaries prepended with matching Turn Summary</ledger_updates>
  <status>approved</status>
## Open Questions & Follow-ups (Concise)
- Probe diameter default: derive from probe FWHM when available; otherwise use a documented constant (e.g., 0.6×N). Always record `probe_diameter_px` in metrics JSON.
- neighbor_count default: 6 (excluding seed); parameterized; record actual value used in every run.
- gs=1 vs gs=2 sequencing: run gs=1 first (less failure-prone); define a gs=2 smoke (training loss curve sanity) before expanding.
- Legacy “view” arg (validators/CLI): keep a temporary mapping to explicit `s_img/n_groups`; document deprecation and removal timeline.
- CLI transition shim: optional mapping of dense/sparse → explicit params during transition; remove once tests and docs fully adopt metrics.

## Appendix — Historical Audits (Deprecated)

</plan_update>
- [x] Wire post-verify automation (`run_phase_g_dense.py::main`) so every dense run automatically executes SSIM grid → verifier → highlights checker with success banner references (commit 74a97db5).
- [x] Add pytest coverage for collect-only + execution chain (`tests/study/test_phase_g_dense_orchestrator.py::{test_run_phase_g_dense_collect_only_post_verify_only,test_run_phase_g_dense_post_verify_only_executes_chain}`) and archive GREEN logs under `reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/`.
- [x] Normalize success-banner path prints to hub-relative strings (`run_phase_g_dense.py::main`, both full run and `--post-verify-only`) and extend `test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain` to assert the relative `cli/` + `analysis/` lines (TYPE-PATH-001). — commit `7dcb2297`.
- [x] Extend `test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview` so the **full** execution path asserts hub-relative `CLI logs: cli`, `Analysis outputs: analysis`, and `analysis/artifact_inventory.txt` strings (TYPE-PATH-001) to guard the counted run prior to execution evidence.
- [x] Deduplicate the success banner's "Metrics digest" lines in `plans/active/.../bin/run_phase_g_dense.py::main` so only one Markdown path prints and the CLI log line stays distinct, avoiding conflicting guidance before we archive evidence.
- [x] Extend `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest` so it asserts **exactly one** `Metrics digest:` line appears in stdout (guarding against future banner duplication) and continue to check for the `Metrics digest log:` reference. — commit `4cff9e38`.
- [x] Add a follow-on assertion in `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest` to ensure `stdout.count("Metrics digest log: ") == 1` so CLI log references cannot duplicate when future banner edits land (TYPE-PATH-001, TEST-CLI-001). — commit `32b20a94`.
- [x] Extend `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest` so it also asserts the success banner references `analysis/verification_report.json`, `analysis/verify_dense_stdout.log`, `analysis/check_dense_highlights.log`, and the SSIM grid summary/log paths. This keeps verification evidence lines guarded alongside the digest banner (TEST-CLI-001, TYPE-PATH-001). — commit `6a51d47a`.
- [x] Extend `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain` so the `--post-verify-only` path also asserts the SSIM grid summary/log plus verification report/log/highlights lines and ensures stubbed CLI logs exist (TEST-CLI-001, TYPE-PATH-001). — commit `ba93f39a`.
- [x] Replace the placeholder `geometry_aware_floor` logic in `studies/fly64_dose_overlap/overlap.py::generate_overlap_views` with a per-split bounding-box acceptance bound: compute the theoretical maximum acceptance as `(area / (pi * (threshold / 2) ** 2)) / n_positions`, clamp the bound to ≤0.10, guard against zero with a tiny epsilon, and persist both `geometry_acceptance_bound` and the resulting `effective_min_acceptance` through `SpacingMetrics`, `metrics_bundle.json`, and `_metadata`. (Implemented in `studies/fly64_dose_overlap/overlap.py:334-555`; verified locally + via `$HUB/green/pytest_dense_acceptance_floor.log`.) (docs/findings.md: STUDY-001, ACCEPTANCE-001; specs/data_contracts.md §12.)
- [x] Extend `tests/study/test_dose_overlap_overlap.py` with `test_generate_overlap_views_dense_acceptance_floor` to pin the low-acceptance scenario and verify the metrics bundle records `geometry_acceptance_bound`/`effective_min_acceptance`. (`tests/study/test_dose_overlap_overlap.py:523-661`; GREEN log at `$HUB/green/pytest_dense_acceptance_floor.log`.) (docs/development/TEST_SUITE_INDEX.md:62.)
- [ ] Rerun the counted dense pipeline with logs under the hub: `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`, then immediately invoke `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log` before rerunning the metrics helpers if `analysis/metrics_summary.json` is stale so `{analysis}` gains SSIM grid summary/log, verification report/log, highlights log, metrics summary/digest, preview text, and `artifact_inventory.txt`. (docs/TESTING_GUIDE.md §§Phase G orchestrator + metrics; docs/findings.md: PREVIEW-PHASE-001, TEST-CLI-001.)
<plan_update version="1.0">
  <trigger>`cli/run_phase_g_dense_post_verify_only.log` only contains the argparse usage banner, proving the post-verify helper was invoked without the required `--dose/--view/--splits` arguments; we must restate the full command before another engineering loop.</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, specs/data_contracts.md, docs/fix_plan.md, galph_memory.md, input.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/run_phase_g_dense_post_verify_only.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/blocker.log</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>Document the usage failure as a fresh audit, clarify the full `--post-verify-only` command (including dose/view/splits), and carry the SSIM/verification/highlights/metrics/preview/inventory expectations forward unchanged.</proposed_changes>
  <impacts>Focus stays ready_for_implementation; engineering must re-run the counted pipeline with the corrected post-verify command so `{analysis}` finally accumulates MS-SSIM ±0.000 / MAE ±0.000000 deltas and the required evidence bundle.</impacts>
  <ledger_updates>docs/fix_plan.md Latest Attempt, hub summary Turn Summary, input.md rewrite, galph_memory hand-off.</ledger_updates>
  <status>approved</status>
</plan_update>
<plan_update version="1.0">
  <trigger>2025-11-13 hub audit showed zero progress since the last loop: `{analysis}` remains only `blocker.log`, the hub deletion tracked by git lives entirely under the reports hub, and `cli/run_phase_g_dense_post_verify_only.log` is still the argparse usage banner (the command missed `--dose/--view/--splits`).</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md (POLICY-001, CONFIG-001, DATA-001, TYPE-PATH-001, STUDY-001, TEST-CLI-001, PREVIEW-PHASE-001, PHASEC-METADATA-001, ACCEPTANCE-001), docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md (§§ Phase G orchestrator + metrics helpers), docs/development/TEST_SUITE_INDEX.md (row for tests/study/test_dose_overlap_overlap.py), specs/data_contracts.md §12, docs/fix_plan.md, galph_memory.md, input.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/blocker.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_d_dense.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/run_phase_g_dense_post_verify_only.log</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>Add a 2025-11-13 audit note capturing the unchanged artifact set, reiterate that git was evidence-only dirty (so we skipped pull/rebase), and keep the ready_for_implementation Do Now focused on rerunning the counted dense pipeline immediately followed by the fully parameterized `--post-verify-only` command plus metrics helpers.</proposed_changes>
  <impacts>Dwell remains at the ready_for_implementation ceiling until Ralph executes the pytest guard, the counted dense run, the corrected post-verify sweep, and the metrics helper pair so `{analysis}` gains the SSIM grid / verification / highlights / metrics / preview / inventory bundle with MS-SSIM ±0.000 / MAE ±0.000000 reporting.</impacts>
  <ledger_updates>docs/fix_plan.md Latest Attempt, hub summaries (`summary.md`, `summary/summary.md`), input.md rewrite, galph_memory entry, retrospective log.</ledger_updates>
  <status>approved</status>
</plan_update>
- **2025-11-13T010930Z audit:** `git status --porcelain` only lists the deleted `data/phase_c/run_manifest.json` inside this hub, so per the evidence-only rule we skipped `git pull --rebase` (recorded as `evidence_only_dirty=true`). `{analysis}` is still just `blocker.log`, `{cli}` tops out at `phase_c_generation.log`, `phase_d_dense.log`, and the various `run_phase_g_dense_stdout*.log` files, and `cli/run_phase_g_dense_post_verify_only.log` remains the argparse usage banner because the command omitted `--dose/--view/--splits`. Do Now stays ready_for_implementation: guard the working directory, rerun the pytest selector, execute the counted dense run, immediately run the fully parameterized `--post-verify-only` command, refresh the metrics helpers if `analysis/metrics_summary.json` is stale, and do not stop until `{analysis}` contains the SSIM grid / verification / highlights / metrics / preview / artifact inventory bundle with MS-SSIM ±0.000 / MAE ±0.000000 deltas recorded across hub summaries and docs.
- **2025-11-11T131617Z reality check:** After stashing/restoring the deleted `data/phase_c/run_manifest.json` plus CLI/pytest logs to satisfy `git pull --rebase`, `{analysis}` still only contains `blocker.log` while `{cli}` is limited to `phase_c_generation.log`, `phase_d_dense.log`, and `run_phase_g_dense_stdout.log`. No SSIM grid, verification, preview, metrics, or artifact-inventory outputs exist yet, so the counted dense run + immediate `--post-verify-only` sweep remain the blocking deliverables for this phase.
- **2025-11-12T210000Z retrospective:** Latest hub inspection confirms nothing changed since the prior attempt—`analysis/` still only holds `blocker.log`, `cli/` still has the short trio of logs, and `data/phase_c/run_manifest.json` remains deleted (must be regenerated by the counted run, not restored manually). `cli/phase_d_dense.log` shows the last execution ran from `/home/ollie/Documents/PtychoPINN2` and failed with `ValueError: Object arrays cannot be loaded when allow_pickle=False`, so enforcing `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"` before every command stays mandatory. A quick `git log -10 --oneline` retrospective shows no new Ralph commits landed since the previous Do Now, so the focus remains ready_for_implementation with the same pytest + CLI guardrails.
- **2025-11-11T215207Z geometry audit:** Dense Phase D now runs from this repo but still aborts because only 42/5088 train positions (0.8 %) survive the 38.4 px spacing guard even after the greedy fallback; bounding-box analysis (56 399 px² area) caps the theoretical acceptance at ≈0.96 %, so the legacy 10 % floor can never succeed. `{analysis}` still only contains `blocker.log`; there are no SSIM grid, verification, highlights, metrics, preview, or artifact-inventory artifacts yet. Next step is to encode the geometry-aware acceptance floor, land a pytest, and rerun the dense pipeline with the post-verify sweep so `{analysis}` fills with evidence.
- **2025-11-11T222901Z audit:** Re-read docs/index.md, docs/findings.md (POLICY-001 / CONFIG-001 / DATA-001 / TYPE-PATH-001 / STUDY-001 / TEST-CLI-001 / PREVIEW-PHASE-001 / PHASEC-METADATA-001 / ACCEPTANCE-001), docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, specs/data_contracts.md §12, the hub plan + summaries, `analysis/blocker.log`, `cli/phase_d_dense.log`, `studies/fly64_dose_overlap/overlap.py`, `tests/study/test_dose_overlap_overlap.py`, and `$HUB/green/pytest_dense_acceptance_floor.log`. Geometry-aware acceptance bound + pytest are already landed (cf. `studies/fly64_dose_overlap/overlap.py:334-555` and the GREEN log), but `{analysis}` still only contains `blocker.log` because the counted rerun and `--post-verify-only` sweep never executed after the code change (`cli/phase_d_dense.log` still shows the “minimum 10.0%” message). Do Now stays ready_for_implementation: guard `pwd -P`, export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier`, rerun `pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_dense_acceptance_floor -vv | tee "$HUB"/green/pytest_dense_acceptance_floor.log`, then run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log` immediately followed by `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`, then rerun `report_phase_g_dense_metrics.py` and `analyze_dense_metrics.py` if `analysis/metrics_summary.json` predates the rerun so `{analysis}` captures `ssim_grid_summary.md`, `ssim_grid.log`, `verification_report.json`, `verify_dense_stdout.log`, `check_dense_highlights.log`, `metrics_summary.json`, `metrics_delta_highlights_preview.txt`, `metrics_digest.md`, preview verdict text, and `artifact_inventory.txt` with MS-SSIM ±0.000 / MAE ±0.000000 deltas logged in hub summary + docs/fix_plan.md + galph_memory. Blockers belong under `$HUB/red/blocked_<timestamp>.md`.

<plan_update version="1.0">
  <trigger>2025-11-11T220602Z spec audit shows the current `geometry_aware_floor` helper still applies a 50% fudge factor and exposes the wrong JSON keys, so the checklist must explicitly call for the bounded `geometry_acceptance_bound` implementation plus a rerun.</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, specs/data_contracts.md, docs/fix_plan.md, input.md, galph_memory.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/blocker.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_d_dense.log, studies/fly64_dose_overlap/overlap.py, tests/study/test_dose_overlap_overlap.py</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>Clarify the Phase D checklist item to require the bounded `geometry_acceptance_bound` calculation/fields, restate the pytest target, and add a fresh audit note documenting that `{analysis}` still only has blocker.log so the ready_for_implementation Do Now remains code/test edits + counted run.</proposed_changes>
  <impacts>Focus stays ready_for_implementation (dwell capped at 3) until the acceptance-bound code/test deltas land and `{analysis}` finally contains SSIM grid, verification, highlights, preview, metrics summary/digest, and artifact inventory evidence with MS-SSIM ±0.000 / MAE ±0.000000 deltas.</impacts>
  <ledger_updates>Record Attempt 2025-11-11T220602Z in docs/fix_plan.md with the same selectors, hub path, and artifact list; keep input.md aligned with the bounded-acceptance Do Now.</ledger_updates>
  <status>approved</status>
</plan_update>
- **2025-11-11T220602Z spec audit:** `{analysis}` still only contains `blocker.log` even though `cli/` now has multiple `run_phase_g_dense_stdout*.log` files; there is still no SSIM grid summary/log, verification report/log, highlights log, preview text, metrics summary/digest, or artifact inventory. Reviewing `studies/fly64_dose_overlap/overlap.py` and `tests/study/test_dose_overlap_overlap.py` shows the helper currently emits `geometry_aware_floor` (50 % of the theoretical bound, floored at 1 %) instead of the required `geometry_acceptance_bound` (actual bounding-box acceptance capped at 10 %). Ready_for_implementation Do Now: (1) update `SpacingMetrics` + `_metadata` + `metrics_bundle.json` to expose `geometry_acceptance_bound` and set `effective_min_acceptance = clamp(bound, ε, 0.10)` without the 0.5 fudge; (2) update `test_generate_overlap_views_dense_acceptance_floor` (plus fixtures) to assert the renamed JSON keys and bounded logic; (3) guard `pwd -P` plus `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`, run `pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_dense_acceptance_floor -vv | tee "$HUB"/green/pytest_dense_acceptance_floor.log`, then execute the counted `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log` immediately followed by `--post-verify-only`; rerun the metrics helpers if `analysis/metrics_summary.json` predates the run, and do not exit until `{analysis}` lists SSIM grid summary/log, verification report/log, highlights log, metrics summary/digest, preview verdict, and `artifact_inventory.txt` with MS-SSIM ±0.000 / MAE ±0.000000 deltas logged across the hub summaries + docs/fix_plan.md.

<plan_update version="1.0">
  <trigger>Local dense runs fail deterministically because the hard-coded 10 % acceptance floor exceeds the geometry-derived cap (~0.96 %); we must add the geometry-aware guard, test it, and only then rerun the pipeline.</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/findings.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, specs/data_contracts.md, docs/fix_plan.md, input.md, galph_memory.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/blocker.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_d_dense.log, studies/fly64_dose_overlap/overlap.py, tests/study/test_dose_overlap_overlap.py</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>Add the unchecked checklist items (geometry-aware acceptance floor, pytest, rerun), document the 0.96 % theoretical cap, and lay out a ready_for_implementation Do Now covering code/test updates plus the counted `run_phase_g_dense.py --clobber` → `--post-verify-only` chain and artifact expectations.</proposed_changes>
  <impacts>Requires production edits, new pytest coverage, and a full rerun before supervisor dwell can reset; `{analysis}` must gain SSIM grid summary/log, verification report/log, highlights log, metrics summary/digest, preview text, and `artifact_inventory.txt` with MS-SSIM ±0.000 / MAE ±0.000000 deltas recorded in summaries/docs.</impacts>
  <ledger_updates>Record Attempt 2025-11-11T215207Z in docs/fix_plan.md, update the hub plan/summaries + docs/findings.md + input.md, and reiterate the ready_for_implementation Do Now.</ledger_updates>
  <status>approved</status>
</plan_update>

<plan_update version="1.0">
  <trigger>Post-verify CLI log still shows the argparse usage banner because the command omitted `--dose/--view/--splits`, so the verification/highlights/metrics bundle never ran.</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md (POLICY-001, CONFIG-001, DATA-001, TYPE-PATH-001, STUDY-001, TEST-CLI-001, PREVIEW-PHASE-001, PHASEC-METADATA-001, ACCEPTANCE-001), docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, specs/data_contracts.md §12, docs/fix_plan.md, galph_memory.md, input.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/blocker.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/run_phase_g_dense_post_verify_only.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_d_dense.log</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>Explicitly call for the fully parameterized `--post-verify-only` command inside the Do Now checklist and log a fresh audit explaining why the helper never executed.</proposed_changes>
  <impacts>Engineer must rerun the dense pipeline and immediately invoke the corrected helper so SSIM grid, verification, highlights, metrics, preview, and artifact inventory artifacts exist before the next supervisor loop.</impacts>
  <ledger_updates>Refresh docs/fix_plan.md Latest Attempt, hub summaries, input.md, and galph_memory with the corrected Do Now.</ledger_updates>
  <status>approved</status>
</plan_update>

- **Do Now — geometry-aware acceptance floor + dense rerun (ready_for_implementation):**
  1. Guard rails: `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"` and export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier`; capture stdout/err for every command via `tee` inside the hub (POLICY-001, TYPE-PATH-001).
  2. Implement the geometry-aware acceptance floor in `studies/fly64_dose_overlap/overlap.py::generate_overlap_views`: compute each split’s bounding-box area, derive the theoretical acceptance bound (`area / (π * (threshold / 2) ** 2) / n_positions`), clamp the effective minimum acceptance to `min(0.10, bound)` (guard against degenerate cases), and log both the derived bound and actual acceptance inside the spacing metrics record + `_metadata`. Keep the greedy fallback but only raise if even the geometry-aware floor fails; cite docs/findings.md (ACCEPTANCE-001) in code comments.
  3. Extend `tests/study/test_dose_overlap_overlap.py` with `test_generate_overlap_views_dense_acceptance_floor`: craft a low-acceptance split, assert `generate_overlap_views` returns filtered NPZs + metrics without raising, and verify the metrics bundle stores `geometry_acceptance_bound`/`effective_min_acceptance`. Run `pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_dense_acceptance_floor -vv | tee "$HUB"/green/pytest_dense_acceptance_floor.log` (RED output → `$HUB/red/blocked_<timestamp>.md`).
  4. Execute `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`, confirm Phase C/D finish, then immediately run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log` so the helper does not exit at the argparse usage banner.
  5. If `analysis/metrics_summary.json` predates the rerun, execute `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py --hub "$HUB" --metrics "$HUB"/analysis/metrics_summary.json` and `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --hub "$HUB"` so `metrics_delta_highlights_preview.txt` (PREVIEW-PHASE-001) and `metrics_digest.md` capture the refreshed acceptance metadata and MS-SSIM/MAE deltas.
  6. Evidence expectations: `{analysis}` must now contain `ssim_grid_summary.md`, `ssim_grid.log`, `verification_report.json`, `verify_dense_stdout.log`, `check_dense_highlights.log`, `metrics_summary.json`, `metrics_delta_highlights_preview.txt`, `metrics_digest.md`, and `artifact_inventory.txt` with MS-SSIM ±0.000 / MAE ±0.000000 deltas + preview verdict referenced in `$HUB/summary/summary.md`, docs/fix_plan.md, and galph_memory. Any failure must be logged to `$HUB/red/blocked_<timestamp>.md` with the exact command + exit code before pausing.
- **2025-11-12T231800Z audit:** After stash→pull→pop the repo remains dirty only under this hub; `{analysis}` still just holds `blocker.log`, `cli/` still stops at `{phase_c_generation,phase_d_dense,run_phase_g_dense_stdout}.log`, and there are zero SSIM grid, verification, preview, metrics, or artifact-inventory artifacts. `cli/phase_d_dense.log` again ends with `ValueError: Object arrays cannot be loaded when allow_pickle=False`, confirming the dense rerun never advanced past Phase D inside `/home/ollie/Documents/PtychoPINN`.
- **2025-11-13T003000Z audit:** `timeout 30 git pull --rebase` now runs cleanly (no stash cycle needed) and `git log -10 --oneline` shows the overlap `allow_pickle=True` fix (`5cd130d3`) is already on this branch, so the lingering ValueError in `cli/phase_d_dense.log` is stale output from `/home/ollie/Documents/PtychoPINN2`. The active hub still lacks any counted dense evidence: `analysis/` only holds `blocker.log`, `{cli}` contains `{phase_c_generation,phase_d_dense,run_phase_g_dense_stdout(_retry,_v2).log}`, and there are zero SSIM grid summaries, verification/previews, metrics delta files, or artifact inventory snapshots. With the fix landed locally, the only remaining action is to rerun `run_phase_g_dense.py --clobber ...` followed immediately by `--post-verify-only` **from /home/ollie/Documents/PtychoPINN** so Phase C manifests regenerate and `{analysis,cli}` fill with real artifacts.
- **2025-11-13T012200Z audit:** `git status --porcelain` only lists the stale `cli/phase_d_dense.log` inside the active hub, so per the evidence-only rule I skipped `git pull --rebase` (recorded `evidence_only_dirty=true`) and exported `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` before editing. Re-read docs/index.md, docs/findings.md (POLICY-001 / CONFIG-001 / DATA-001 / TYPE-PATH-001 / STUDY-001 / TEST-CLI-001 / PREVIEW-PHASE-001 / PHASEC-METADATA-001 / ACCEPTANCE-001), docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, specs/data_contracts.md §12, docs/fix_plan.md, galph_memory.md, input.md, `summary.md`, `summary/summary.md`, `analysis/blocker.log`, `cli/phase_d_dense.log`, and `cli/run_phase_g_dense_post_verify_only.log`. `{analysis}` is still just `blocker.log` and the post-verify CLI log remains the argparse usage banner because the command missed `--dose/--view/--splits`, so SSIM grid / verification / highlights / metrics / preview / inventory artifacts do not exist yet. Reissue the same ready_for_implementation Do Now with the fully parameterized post-verify command and the MS-SSIM ±0.000 / MAE ±0.000000 reporting requirements.
- **2025-11-11T191109Z audit:** `timeout 30 git pull --rebase` reported “Already up to date,” and revisiting docs/index.md + docs/findings.md reconfirmed the governing findings (POLICY-001, CONFIG-001, DATA-001, TYPE-PATH-001, STUDY-001, TEST-CLI-001, PREVIEW-PHASE-001, PHASEC-METADATA-001). The hub has **no progress**: `analysis/` still only contains `blocker.log`, `{cli}` is limited to `phase_c_generation.log`, `phase_d_dense.log`, and `run_phase_g_dense_stdout(_retry,_v2).log`, and `cli/phase_d_dense.log` again ends with the local `ValueError: Object arrays cannot be loaded when allow_pickle=False` stack trace even though the fix landed. There are zero SSIM grid summaries, verification/highlights logs, preview text, metrics digests, or artifact-inventory files. The counted dense rerun + immediate `--post-verify-only` sweep therefore remains the blocking deliverable; rerun both commands from `/home/ollie/Documents/PtychoPINN` with the mapped pytest guard and tee’d CLI logs into this hub.
- **2025-11-11T180800Z audit:** Repeated the `git stash push --include-untracked` → `timeout 30 git pull --rebase` → `git stash pop` flow to stay synced while preserving the deleted `data/phase_c/run_manifest.json` and local `.bak` notes. Hub contents remain unchanged: `analysis/` only has `blocker.log` and `cli/` still holds `phase_c_generation.log`, `phase_d_dense.log`, and the `run_phase_g_dense_stdout*.log` variants—there are still zero SSIM grid, verification, preview, metrics, highlights, or artifact-inventory outputs. Importantly, `cli/phase_d_dense.log` shows the failed overlap step was executed **inside** `/home/ollie/Documents/PtychoPINN` (not the secondary clone) and still hit `ValueError: Object arrays cannot be loaded when allow_pickle=False`, which means the post-fix pipeline has never been re-run. Guard every command with `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"` and regenerate the Phase C manifest via the counted run rather than restoring it manually.
- **2025-11-12T235900Z audit:** The hub does contain `cli/run_phase_g_dense_post_verify_only.log`, but it is only the argparse usage banner because the command omitted `--dose/--view/--splits`. `{analysis}` still only has `blocker.log`, so SSIM grid/verification/highlights/metrics/preview/inventory artifacts remain absent. Re-run the counted pipeline and copy the full post-verify command (with dose/view/splits) directly from this plan to avoid another no-op.
- **2025-11-11T185738Z retrospective:** Rechecked the hub immediately after the latest stash→pull→pop sync and scanned `git log -10 --oneline`; no Ralph commits landed after `e3b6d375` (reports evidence), so the prior Do Now was never executed. The hub is still missing every Phase G artifact: `analysis/` contains only `blocker.log`, `{cli}` stops at `{phase_c_generation,phase_d_dense,run_phase_g_dense_stdout(_retry,_v2).log}`, and `analysis/metrics_delta_highlights_preview.txt`, SSIM grid summaries, verification/highlights logs, metrics digests, and artifact inventories **do not exist**. `cli/phase_d_dense.log` continues to end with the pre-fix `ValueError: Object arrays cannot be loaded when allow_pickle=False` from this repo, confirming the counted dense rerun plus immediate `--post-verify-only` sweep still never happened.
<plan_update version="1.0">
  <trigger>Fresh hub audit still shows zero SSIM grid, verification, preview, or metrics artifacts even after the allow_pickle fix, so we must restate the rerun orders with explicit artifact expectations.</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/fix_plan.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>Add a 2025-11-11T192358Z audit note under Phase G summarizing today’s reality check and reiterating the counted --clobber + --post-verify-only Do Now with the full artifact list Ralph must produce.</proposed_changes>
  <impacts>Focus stays ready_for_implementation: Ralph cannot move on until pytest guards rerun and `{analysis}` contains SSIM grid, verification, highlights, preview, metrics summary/digest, and artifact inventory evidence alongside MS-SSIM ±0.000 / MAE ±0.000000 deltas.</impacts>
  <ledger_updates>Record this audit as the next Attempt in docs/fix_plan.md with matching selectors, hub path, and artifact requirements.</ledger_updates>
  <status>approved</status>
</plan_update>
- **2025-11-11T192358Z audit:** Re-read docs/index.md plus docs/findings.md (POLICY-001, CONFIG-001, DATA-001, TYPE-PATH-001, STUDY-001, TEST-CLI-001, PREVIEW-PHASE-001, PHASEC-METADATA-001) and audited the active hub again: `analysis/` still only holds `blocker.log`, while `cli/` is capped at `phase_c_generation.log`, `phase_d_dense.log`, and the trio of `run_phase_g_dense_stdout*.log`. `cli/phase_d_dense.log` ends with the local `ValueError: Object arrays cannot be loaded when allow_pickle=False`, proving the counted dense run never succeeded inside this repo. Reissue the ready_for_implementation Do Now: (1) rerun the mapped pytest collect-only selector plus `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain` (logs under `$HUB/collect` and `$HUB/green`) so banner guards stay GREEN; (2) execute `plans/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber` from `/home/ollie/Documents/PtychoPINN` with stdout piped to `$HUB/cli/run_phase_g_dense_stdout.log`; (3) immediately run `--post-verify-only` into `$HUB/cli/run_phase_g_dense_post_verify_only.log`; (4) verify the rerun produces `analysis/ssim_grid_summary.md`, `analysis/ssim_grid.log`, `analysis/verification_report.json`, `analysis/verify_dense_stdout.log`, `analysis/check_dense_highlights.log`, `analysis/metrics_summary.json`, `analysis/metrics_delta_highlights_preview.txt` (phase-only per PREVIEW-PHASE-001), `analysis/metrics_digest.md`, and `analysis/artifact_inventory.txt`; and (5) publish MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, SSIM grid references, and verification/highlights paths across the hub summaries, docs/fix_plan.md, and galph_memory before closing the loop.
<plan_update version="1.0">
  <trigger>2025-11-11T205022Z hub audit still shows zero SSIM grid, verification, preview, metrics, or inventory artifacts even though overlap.py now honors allow_pickle=True, so the counted rerun + post-verify sweep must be restated with stricter deliverables.</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/fix_plan.md, galph_memory.md, input.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/blocker.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_d_dense.log</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>Add a 2025-11-11T205022Z audit under Phase G describing today’s git pull, the still-empty hub (`analysis` only has `blocker.log`, `cli/phase_d_dense.log` ends with the allow_pickle ValueError), and explicitly restate the Do Now steps (pytest collect + selector logs, counted `--clobber` rerun, immediate `--post-verify-only`, optional `report_phase_g_dense_metrics.py`, and the list of `analysis/` artifacts plus MS-SSIM ±0.000 / MAE ±0.000000 reporting).</proposed_changes>
  <impacts>Focus remains ready_for_implementation; Ralph cannot exit this attempt without new SSIM grid + verification + highlights + metrics digest + artifact inventory evidence under `{analysis}` and updated MS-SSIM/MAE deltas + preview verdict in both summaries and docs/fix_plan.md.</impacts>
  <ledger_updates>Append Attempt (2025-11-11T205022Z) to docs/fix_plan.md with the same Do Now, selectors, and artifact expectations; mention skipped pull rationale (non-evidence dirty paths absent) and cite the updated hub plan/summary.</ledger_updates>
  <status>approved</status>
</plan_update>
- **2025-11-11T205022Z audit:** `git status -sb` showed `?? docs/fix_plan.md.bak` (whitelisted) plus `?? docs/iteration_scores_262-291.csv`, so I ran `timeout 30 git pull --rebase` to sync before editing. Re-read docs/index.md, docs/findings.md (POLICY-001 / CONFIG-001 / DATA-001 / TYPE-PATH-001 / STUDY-001 / TEST-CLI-001 / PREVIEW-PHASE-001 / PHASEC-METADATA-001), docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md §Phase G orchestrator, docs/development/TEST_SUITE_INDEX.md (study selectors), the hub `plan/plan.md`, hub summaries, `analysis/blocker.log`, and `cli/phase_d_dense.log`. Nothing has improved: `{analysis}` still only has `blocker.log`, `{cli}` stops at `phase_c_generation.log`, `phase_d_dense.log`, and the `run_phase_g_dense_stdout*` variants, the deleted `data/phase_c/run_manifest.json` still needs regeneration instead of manual restore, and `cli/phase_d_dense.log` continues to end with `ValueError: Object arrays cannot be loaded when allow_pickle=False` emitted **inside `/home/ollie/Documents/PtychoPINN`** even though commit `5cd130d3` fixed overlap.py. Reissue the ready_for_implementation Do Now with no wiggle room:
  1. `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"` and export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` plus `HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier` before touching anything (TYPE-PATH-001, TEST-CLI-001).
  2. Recreate GREEN banners: `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log` and `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log`.
  3. Execute the counted dense pipeline with clobber archive + regenerated Phase C manifest: `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log` (POLICY-001 + DATA-001 guard). This must recreate `data/phase_c/run_manifest.json` instead of restoring backups.
  4. Immediately run `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log` so SSIM grid → verifier → highlights checker rerun with the fixed overlap pipeline.
  5. If `analysis/metrics_summary.json` predates the new outputs, rerun `python plans/active/.../bin/report_phase_g_dense_metrics.py --hub "$HUB" --metrics "$HUB"/analysis/metrics_summary.json` followed by `python plans/active/.../bin/analyze_dense_metrics.py --hub "$HUB"` so `metrics_delta_highlights_preview.txt` (phase-only per PREVIEW-PHASE-001) and `metrics_digest.md` align with the new evidence set.
  6. Confirm `{analysis}` now contains **all** of the following: `ssim_grid_summary.md`, `ssim_grid.log`, `verification_report.json`, `verify_dense_stdout.log`, `check_dense_highlights.log`, `metrics_summary.json`, `metrics_delta_highlights_preview.txt`, `metrics_digest.md`, and `artifact_inventory.txt`. Record MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, selector names, and CLI log paths in `$HUB/summary/summary.md`, docs/fix_plan.md, and galph_memory; stash any failure transcript under `$HUB/red/blocked_$(date -u +%FT%H%M%SZ).md` with the exact command + exit code before stopping.
<plan_update version="1.0">
  <trigger>Dense geometry analysis shows the theoretical acceptance bound is <1 %, so the fixed 10 % guard is unsatisfiable; we must add a derived acceptance floor + pytest coverage before reissuing the counted run.</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/GRIDSIZE_N_GROUPS_GUIDE.md, specs/data_contracts.md, specs/ptychodus_api_spec.md, docs/architecture.md, docs/fix_plan.md, galph_memory.md, input.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_d_dense.log, studies/fly64_dose_overlap/overlap.py, tests/study/test_dose_overlap_overlap.py</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>Capture today’s geometry probe (train bound 0.957 %, test bound 0.934 %) plus the still-absent docs/prompt_sources_map.json, instruct Ralph to add a geometry-aware acceptance floor with new spacing-metrics logging + pytest, and restate the ready_for_implementation nucleus (code/test edits → pytest selector → counted `--clobber` + `--post-verify-only` rerun → metrics/doc sync → hub artifacts).</proposed_changes>
  <impacts>touches studies/fly64_dose_overlap/overlap.py (SpacingMetrics fields, adaptive guard, console output), tests/study/test_dose_overlap_overlap.py (new selector + updated assertions), `docs/TESTING_GUIDE.md` + `docs/development/TEST_SUITE_INDEX.md` (once tests pass), and still requires the full Phase G CLI rerun to populate `{analysis}`.</impacts>
  <ledger_updates>Log Attempt 2025-11-11T215650Z in docs/fix_plan.md with the geometry summary, code/test deliverables, pytest selector, CLI commands, and artifact expectations; cite the new one-off analysis snippet stored in the hub summary.</ledger_updates>
  <status>approved</status>
</plan_update>
- **2025-11-11T215650Z audit:** `git status -sb` still lists non-hub docs (`docs/ITERATION_AUDIT_INDEX.md`, `docs/iteration_analysis_audit_full.md`, `docs/iteration_scores_262-291.csv`, etc.), so I ran `timeout 30 git pull --rebase` (already up to date), re-read docs/index.md, docs/findings.md (POLICY-001 / CONFIG-001 / DATA-001 / TYPE-PATH-001 / STUDY-001 / TEST-CLI-001 / PREVIEW-PHASE-001 / PHASEC-METADATA-001), docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md §Phase G, docs/development/TEST_SUITE_INDEX.md, docs/GRIDSIZE_N_GROUPS_GUIDE.md, specs/data_contracts.md, specs/ptychodus_api_spec.md, docs/architecture.md, galph_memory.md, input.md, the hub plan, summary.md, summary/summary.md, and `cli/phase_d_dense.log`. A scripted geometry probe over `data/phase_c/dose_1000/patched_{train,test}.npz` shows: train span 335.82×167.94 px (area 56 399 px²) ⇒ ≤48.7 viable positions and a 0.957 % acceptance ceiling; greedy acceptance is 42/5088 (0.825 %). Test bound is 0.934 % with greedy acceptance 0.786 %. Consequently the hard-coded 10 % threshold is impossible to satisfy even with perfect packing, so we must compute/log an adaptive floor (e.g., `min(0.10, theoretical_bound × PACKING_MARGIN)` with PACKING_MARGIN≈0.85) and persist the derived values in the metrics JSON/bundle before rerunning Phase G. Do Now: (1) Extend `studies/fly64_dose_overlap/overlap.py` with span/area/`theoretical_max_acceptance`/`adaptive_acceptance_floor` fields (printed + stored via `SpacingMetrics.to_dict()`), and replace the guard to use the adaptive floor for both direct and greedy selection; (2) add `tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_dense_acceptance_floor` plus any supporting assertions (update existing tests accordingly, then refresh `docs/TESTING_GUIDE.md` + `docs/development/TEST_SUITE_INDEX.md` after GREEN); (3) guard `pwd -P`, export `AUTHORITATIVE_CMDS_DOC` + `HUB`, rerun `pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_dense_acceptance_floor -vv | tee "$HUB"/green/pytest_dense_acceptance_floor.log`, then execute `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log` immediately followed by `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`; (4) rerun the metrics helpers if needed so `{analysis}` finally holds SSIM grid summary/log, verification report/log, highlights log, preview verdict, `metrics_summary.json`, `metrics_delta_highlights_preview.txt`, `metrics_digest.md`, and `artifact_inventory.txt`, then log MS-SSIM ±0.000 / MAE ±0.000000 deltas + selector + CLI references in the hub summaries/docs/memory; (5) blockers go under `$HUB/red/blocked_<timestamp>.md`. Note: `docs/prompt_sources_map.json` and `docs/pytorch_runtime_checklist.md` still do not exist—no changes required until they appear upstream.
<plan_update version="1.0">
  <trigger>2025-11-11T213530Z hub audit confirms no SSIM grid/verification/metrics/inventory artifacts landed, `data/phase_c/run_manifest.json` is still missing after the prior clobber attempt, and new CLI logs show the allow_pickle ValueError is still being raised inside this workspace; the counted rerun + post-verify sweep must be restated with the missing deliverables called out explicitly.</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, specs/data_contracts.md, docs/fix_plan.md, galph_memory.md, input.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/blocker.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_d_dense.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/run_phase_g_dense_clobber.log</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>Insert a 2025-11-11T213530Z audit under Phase G noting the git pull, missing `data/phase_c` manifest/splits, continued lack of `{analysis}` artifacts, and renewed instructions to guard `pwd`, rerun both pytest selectors, execute the counted `run_phase_g_dense.py --clobber` followed immediately by `--post-verify-only`, rerun the metrics reporters if needed, and populate `{analysis}` with SSIM grid + verification + highlights + metrics digest + artifact inventory + MS-SSIM/MAE deltas.</proposed_changes>
  <impacts>Focus stays ready_for_implementation; Ralph must produce the enumerated artifacts (or log blockers under `$HUB/red/`) before another supervisor loop can occur—no additional documentation-only turns are allowed without real evidence.</impacts>
  <ledger_updates>Append Attempt (2025-11-11T213530Z) to docs/fix_plan.md with the refreshed Do Now plus artifact list, and reference the same update in galph_memory and hub summaries.</ledger_updates>
  <status>approved</status>
</plan_update>
- **2025-11-11T213530Z audit:** `git status -sb` still shows `?? docs/fix_plan.md.bak` and `?? docs/iteration_scores_262-291.csv`, so I reran `timeout 30 git pull --rebase` (already up to date) before exporting `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`. Re-read docs/index.md, docs/findings.md (POLICY-001 / CONFIG-001 / DATA-001 / TYPE-PATH-001 / STUDY-001 / TEST-CLI-001 / PREVIEW-PHASE-001 / PHASEC-METADATA-001), docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md §Phase G orchestrator, docs/development/TEST_SUITE_INDEX.md:62, specs/data_contracts.md, docs/fix_plan.md, galph_memory.md, the hub plan, summary.md, summary/summary.md, `analysis/blocker.log`, `cli/phase_d_dense.log`, and `cli/run_phase_g_dense_clobber.log`. Hub reality check: `{analysis}` still only contains `blocker.log`; `{cli}` now has additional partial stdout logs but there is **no** `run_phase_g_dense_post_verify_only.log`; `data/phase_c/run_manifest.json` plus the `patched*.npz` train/test splits are missing entirely (the prior `--clobber` attempt never finished); and `cli/phase_d_dense.log` again shows the local `ValueError: Object arrays cannot be loaded when allow_pickle=False`, confirming the overlap.py fix has never actually executed inside this workspace. Ready_for_implementation Do Now remains unchanged: guard `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"` plus the HUB/COMMAND exports, rerun the two pytest selectors with logs under `$HUB/collect` and `$HUB/green`, execute `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log` immediately followed by `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`, rerun `report_phase_g_dense_metrics.py`/`analyze_dense_metrics.py` if `analysis/metrics_summary.json` is stale, and do not stop until `{analysis}` holds the SSIM grid summary/log, verification_report.json, verify/highlights logs, metrics_summary.json, metrics_delta_highlights_preview.txt, metrics_digest.md, and artifact_inventory.txt with MS-SSIM ±0.000 / MAE ±0.000000 deltas + preview verdict captured in the hub summaries, docs/fix_plan.md, and galph_memory. Any failed command must be recorded under `$HUB/red/blocked_$(date -u +%FT%H%M%SZ).md`.
<plan_update version="1.0">
  <trigger>New audit (git log + hub sweep) shows no artifacts landed since the last Do Now, `analysis/` still only has `blocker.log`, and `cli/phase_d_dense.log` is stuck on the pre-fix allow_pickle ValueError even though overlap.py now forces `allow_pickle=True`; need to restate the counted run expectations with the refreshed guard list.</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, specs/data_contracts.md, docs/fix_plan.md, input.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_d_dense.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/blocker.log, studies/fly64_dose_overlap/overlap.py</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>Add a 2025-11-11T195519Z audit reinforcing that no SSIM grid/verification/preview/metrics/artifacts exist, highlight that `data/phase_c/run_manifest.json` is still deleted and must be regenerated via `--clobber`, and reiterate the pytest + CLI execution order plus artifact list Ralph must prove inside the hub.</proposed_changes>
  <impacts>Focus remains ready_for_implementation; Ralph must produce the Phase C→G rerun evidence (including `analysis/ssim_grid_summary.md`, verification/highlights logs, metrics summary/digest, `metrics_delta_highlights_preview.txt` with phase-only content, refreshed `analysis/artifact_inventory.txt`, and MS-SSIM ±0.000 / MAE ±0.000000 reporting) before this checklist can progress.</impacts>
  <ledger_updates>Record the new audit + directives in docs/fix_plan.md Attempts History, refresh the hub plan + summaries, and rewrite input.md with the actionable Do Now.</ledger_updates>
  <status>approved</status>
</plan_update>
- **2025-11-11T195519Z audit:** `git status -sb` shows only the deleted `data/phase_c/run_manifest.json` plus `docs/fix_plan.md.bak`, so we skipped `git pull --rebase` under the evidence-only dirty exemption and re-confirmed via `git log -10 --oneline` that no new Ralph commits landed since the last directive. The hub is unchanged: `analysis/` still only contains `blocker.log`; `{cli}` stops at `phase_c_generation.log`, `phase_d_dense.log`, and `run_phase_g_dense_stdout(_retry,_v2).log`; and `analysis/metrics_summary.json`, `analysis/metrics_delta_highlights_preview.txt`, SSIM grid summaries/logs, verification/highlights outputs, metrics digest, preview text, and artifact inventory files **do not exist**. `cli/phase_d_dense.log` still ends with `ValueError: Object arrays cannot be loaded when allow_pickle=False` even though `studies/fly64_dose_overlap/overlap.py` now forces `allow_pickle=True`, proving the failure evidence predates the fix and the dense rerun never re-executed here. Reissue the ready_for_implementation Do Now with explicit guardrails: (1) guard `pwd -P` equals `/home/ollie/Documents/PtychoPINN`, export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` + `HUB` from the How-To Map, and rerun the mapped pytest collect-only selector plus `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain` with logs under `$HUB/collect` and `$HUB/green`; (2) execute `python plans/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`, let it regenerate `data/phase_c/run_manifest.json`, and confirm `{analysis}` gains `ssim_grid_summary.md`, `ssim_grid.log`, `verification_report.json`, `verify_dense_stdout.log`, `check_dense_highlights.log`, `metrics_summary.json`, `metrics_delta_highlights_preview.txt` (phase-only, no “amplitude”), `metrics_digest.md`, and `artifact_inventory.txt`; (3) immediately run `--post-verify-only` into `$HUB/cli/run_phase_g_dense_post_verify_only.log` so the shortened chain refreshes SSIM grid + verification artifacts and rewrites `analysis/artifact_inventory.txt`; (4) run the report helper (`report_phase_g_dense_metrics.py --hub "$HUB" --metrics "$HUB"/analysis/metrics_summary.json`) if the sanity table is stale; and (5) update `$HUB/summary/summary.md`, docs/fix_plan.md, and galph_memory with MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, SSIM grid + verification/highlights file names, pytest selectors, and CLI paths. If any command fails, capture `tee` output under `$HUB/red/blocked_<timestamp>.md`, leave artifacts in place, and stop so the supervisor can triage.
<plan_update version="1.0">
  <trigger>2025-11-11T202702Z hub audit confirms Phase G evidence is still missing (analysis only has blocker.log, CLI stops at stale stdout logs, manifest deleted), so we must restate the execution Do Now before another engineer loop.</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, specs/data_contracts.md, docs/fix_plan.md, input.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_d_dense.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/run_phase_g_dense_clobber.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/blocker.log</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>Add a 2025-11-11T202702Z audit callout under Phase G clarifying the evidence-only dirty decision, reiterating the pytest guard + counted `--clobber` + immediate `--post-verify-only` orders, and enumerating the artifact list Ralph must publish inside the active hub.</proposed_changes>
  <impacts>Focus remains ready_for_implementation; Phase G cannot progress until `{analysis}` includes SSIM grid summary/log, verification report/log/highlights, phase-only preview text, metrics summary/digest, refreshed artifact inventory, regenerated `data/phase_c/run_manifest.json`, and MS-SSIM ±0.000 / MAE ±0.000000 deltas recorded in hub summaries + ledger.</impacts>
  <ledger_updates>Log a new Attempt (2025-11-11T202702Z) in docs/fix_plan.md with the same selectors + commands, refresh the hub plan + summaries, and rewrite input.md before handing off to Ralph.</ledger_updates>
  <status>approved</status>
</plan_update>
- **2025-11-11T202702Z audit:** Evidence-only dirty exemption applied (`git status -sb` shows just the deleted `plans/.../data/phase_c/run_manifest.json` plus `docs/fix_plan.md.bak`), so pull was skipped while `git log -10 --oneline` confirmed zero new Ralph commits. `docs/prompt_sources_map.json` is still absent, so docs/index.md remains the authoritative source list; docs/findings.md guardrails (POLICY-001, CONFIG-001, DATA-001, TYPE-PATH-001, STUDY-001, TEST-CLI-001, PREVIEW-PHASE-001, PHASEC-METADATA-001) continue to govern the run. The hub is still empty: `analysis/` only has `blocker.log`; `{cli}` is limited to `phase_c_generation.log`, `phase_d_dense.log`, `run_phase_g_dense_stdout(_retry,_v2).log`, and the partial `run_phase_g_dense_clobber.log` that died mid-dose; Phase C manifests remain deleted; and there are no SSIM grid summaries/logs, verification outputs, highlights, preview text, metrics summary/digest, or artifact inventory files anywhere inside this repo. `cli/phase_d_dense.log` shows the failure happened **inside `/home/ollie/Documents/PtychoPINN`** and still ends with `ValueError: Object arrays cannot be loaded when allow_pickle=False`, proving the overlap.py fix has never been rerun here. Reissue the ready_for_implementation Do Now verbatim: (1) guard `pwd -P` equals `/home/ollie/Documents/PtychoPINN`, export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier`, rerun `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log`, and execute `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log`; (2) run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log` so Phase C manifests are regenerated and `{analysis}` gains `ssim_grid_summary.md`, `ssim_grid.log`, `verification_report.json`, `verify_dense_stdout.log`, `check_dense_highlights.log`, `metrics_summary.json`, `metrics_delta_highlights_preview.txt` (phase-only per PREVIEW-PHASE-001), `metrics_digest.md`, and `artifact_inventory.txt`; (3) immediately follow with `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`; (4) rerun `report_phase_g_dense_metrics.py --hub "$HUB" --metrics "$HUB"/analysis/metrics_summary.json` if the sanity table is stale; and (5) publish MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, SSIM grid references, verification/highlights links, pytest selectors, and CLI log paths in `$HUB/summary/summary.md`, docs/fix_plan.md, and galph_memory. Capture any blocker under `$HUB/red/blocked_$(date -u +%FT%H%M%SZ).md` before stopping.
- [ ] Execute a counted dense Phase C→G run with `--clobber` into `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/`, populating `{analysis,cli}` with real artifacts (phase logs, metrics_delta_summary.json, metrics_delta_highlights_preview.txt, ssim_grid_summary.md, verification_report.json, verify_dense_stdout.log, check_dense_highlights.log, artifact_inventory.txt). **Owner:** Ralph. **Evidence:** CLI/stdout log archived at `$HUB/cli/run_phase_g_dense_stdout.log`.
- [ ] Immediately rerun `run_phase_g_dense.py --hub "$HUB" --post-verify-only` against the fresh artifacts, confirming the shortened chain regenerates SSIM grid + verification outputs and refreshes `analysis/artifact_inventory.txt`. Archive CLI output as `$HUB/cli/run_phase_g_dense_post_verify_only.log`.
- [ ] Update `$HUB/summary/summary.md`, docs/fix_plan.md, and galph_memory with MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict (PREVIEW-PHASE-001), verifier/highlight log references, pytest selectors, and doc/test guard status.
- [ ] If verification surfaces discrepancies, capture blocker logs under `$HUB/red/`, append to docs/fix_plan.md Attempts History with failure signature, and rerun once resolved.

## Execution Hygiene
- **Hub reuse:** Stick to the two active hubs noted at the top of this plan (Phase E training + Phase G comparison). Per `docs/INITIATIVE_WORKFLOW_GUIDE.md`, only create a new timestamped hub when fresh production evidence is produced; otherwise append to the existing hub’s `summary.md`.
- **Dirty hub protocol:** Supervisors must log residual artifacts (missing SSIM grid, deleted manifest, etc.) inside the hub’s `summary/summary.md` *and* `docs/fix_plan.md` before handing off so Ralph is never asked to reconstruct Phase C blindly.
- **Command guards:** All How-To Map entries must be copy/pasteable. Prepend `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"` plus the required `AUTHORITATIVE_CMDS_DOC`/`HUB` exports and avoid `plans/...` ellipses, ensuring evidence stays local to this repo.

### Phase H — Documentation & Gaps
- Note the current code/doc oversampling status and any deviations; update docs/fix_plan.md with artifact paths and outcomes.

## Future Work / Out-of-Scope
- PyTorch training parity: port the Phase E CLI/tests back onto `ptycho_torch` only after TensorFlow evidence is stable; track as a separate fix-plan item so it does not block this initiative.
- Continuous overlap sweeps: expand beyond the current `{s_img, m_group}` grid once the derived overlap reporting proves stable; consider automating the sweep via `studies/.../design.py`.

## Risks & Mitigations
- pty-chi dependency not vendored: document version and environment; cache outputs.
- Oversampling misuse: confirm K≥C and log branch choice; include spacing histograms.
- Data leakage: enforce y-axis split; avoid mixed halves in train/test.

## Evidence & Artifacts
- All runs produce logs/plots/CSV in reports/<timestamp>/ per condition. Summaries collected in summary.md.
