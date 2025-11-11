# Initiative: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001

> **Plan maintenance:** This is the single, evolving plan for the dose/overlap study. Update this file in place instead of creating new `plan/plan.md` documents. The active reports hub is `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/` for Phase E and `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/` for Phase G execution until a milestone changes—reuse them for logs/summaries unless a new milestone is declared.

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

## Backend Selection (Policy for this Study)
- PINN training/inference: **TensorFlow only.** This initiative depends on the legacy `ptycho_train` stack because it is the fully tested backend per CLAUDE.md.
- Iterative baseline (pty-chi): PyTorch under the hood is acceptable for Phase F scripts, but keep it isolated from the PINN pipeline.
- PyTorch parity belongs to future work. Log the remaining work as TODO items rather than mixing stacks mid-initiative.

## Phases

### Phase A — Design & Constraints (COMPLETE)
**Status:** Complete — constants encoded in `studies/fly64_dose_overlap/design.py::get_study_design()`

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
- `studies/fly64_dose_overlap/overlap.py:23-89` implementing spacing utilities (`compute_spacing_matrix`, `build_acceptance_mask`) and `generate_overlap_views:124-227` that materializes the overlap extremes (historically labeled dense/sparse, now defined by `s_img`/`n_groups` inputs) and emits consolidated `metrics_bundle.json` with per-split metrics paths.
- CLI entry `python -m studies.fly64_dose_overlap.overlap` (lines 230-327) batches Phase C artifacts into `{dense,sparse}_{train,test}.npz` files for continuity, but also records `image_subsampling`, `n_groups_requested`, and the measured overlap in each manifest; when `--artifact-root` is set, copies `metrics_bundle.json` into the reports hub for traceability.
- Pytest coverage in `tests/study/test_dose_overlap_overlap.py` with 10 tests validating spacing math (`test_compute_spacing_matrix_*`), orchestration (`test_generate_overlap_views_*`), metrics bundle structure (`test_generate_overlap_views_metrics_manifest`), and failure handling (RED→GREEN evidence: `reports/2025-11-04T034242Z/phase_d_overlap_filtering/{red,green,collect}/`).
- Documentation synchronized: Phase D sections updated in this file, `test_strategy.md`, and `plan.md` D4 marked `[x]`; ledger Attempt #10/11 referencing logs & metrics bundle artifacts.

**Metrics Bundle Workflow (Attempt #10):**
- `generate_overlap_views()` returns `metrics_bundle_path` pointing to aggregated JSON containing `train` and `test` keys, each with `{output_path, spacing_stats_path, image_subsampling, n_groups_requested, overlap_fraction}` entries.
- CLI main copies bundle from temp output to `<artifact_root>/metrics/<dose>/<view>.json` for archival.
- Test guard `test_generate_overlap_views_metrics_manifest` asserts bundle contains required keys and paths exist on disk.
- Execution proof: `reports/2025-11-04T045500Z/phase_d_cli_validation/` (RED→GREEN logs, CLI transcript, copied bundle).

**Key Constraints & References:**
- Spacing targets derived from StudyDesign: overlap extremes near 0.7 (≈38.4 px) and 0.2 (≈102.4 px) are now expressed as recorded metadata rather than enum labels (`docs/GRIDSIZE_N_GROUPS_GUIDE.md:154-172`).
- Oversampling guardrail: neighbor_count=7 ≥ C=4 for gridsize=2 (`docs/SAMPLING_USER_GUIDE.md:112-140`, `docs/findings.md` OVERSAMPLING-001).
- CONFIG-001 boundaries maintained: `overlap.py` loads NPZ via `np.load` only; no params.cfg mutation; validator invoked with `view` parameter.
- DATA-001 compliance ensured via Phase B validator (`validate_fly64_dose_overlap_dataset`) called from `generate_overlap_views`.

**Artifact Hubs:**
- Spacing filter implementation: `reports/2025-11-04T034242Z/phase_d_overlap_filtering/`
- Metrics bundle + CLI validation: `reports/2025-11-04T045500Z/phase_d_cli_validation/`
- Documentation sync: `reports/2025-11-04T051200Z/phase_d_doc_sync/`

**Findings Applied:** CONFIG-001 (pure NPZ loading; legacy bridge deferred to training), DATA-001 (validator enforces canonical NHW layout + dtype/key contracts), OVERSAMPLING-001 (K=7 ≥ C=4 preserved in group construction).

### Phase E — Train PtychoPINN (PAUSED — awaiting TensorFlow rework)
**Status:** Paused. We must restore the TensorFlow training pipeline before any further runs; PyTorch work is retained below as historical context but no longer authoritative for this initiative.

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
- Run scripts/reconstruction/ptychi_reconstruct_tike.py with algorithm='LSQML', num_epochs=100 per test set; capture outputs.

### Phase G — Comparison & Analysis
- Use scripts/compare_models.py three-way comparisons with --ms-ssim-sigma 1.0 and registration; produce plots, CSVs, aligned NPZs; write per-condition summaries.
- After `summarize_phase_g_outputs` completes, run `plans/active/.../bin/report_phase_g_dense_metrics.py --metrics <hub>/analysis/metrics_summary.json` (optionally with `--ms-ssim-threshold 0.80`). Treat the generated **MS-SSIM Sanity Check** table as the go/no-go gate: any row flagged `LOW (...)` indicates reconstruction quality is suspect and the Attempt must be marked blocked until re-run.
- `plans/active/.../bin/analyze_dense_metrics.py` now also embeds the same sanity table inside `metrics_digest.md`; archive this digest under the hub’s `analysis/` directory so reviewers can see absolute MS-SSIM values without hunting through CSVs.

#### Phase G — Active Checklist (2025-11-12)
- [x] Wire post-verify automation (`run_phase_g_dense.py::main`) so every dense run automatically executes SSIM grid → verifier → highlights checker with success banner references (commit 74a97db5).
- [x] Add pytest coverage for collect-only + execution chain (`tests/study/test_phase_g_dense_orchestrator.py::{test_run_phase_g_dense_collect_only_post_verify_only,test_run_phase_g_dense_post_verify_only_executes_chain}`) and archive GREEN logs under `reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/`.
- [x] Normalize success-banner path prints to hub-relative strings (`run_phase_g_dense.py::main`, both full run and `--post-verify-only`) and extend `test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain` to assert the relative `cli/` + `analysis/` lines (TYPE-PATH-001). — commit `7dcb2297`.
- [x] Extend `test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview` so the **full** execution path asserts hub-relative `CLI logs: cli`, `Analysis outputs: analysis`, and `analysis/artifact_inventory.txt` strings (TYPE-PATH-001) to guard the counted run prior to execution evidence.
- [x] Deduplicate the success banner's "Metrics digest" lines in `plans/active/.../bin/run_phase_g_dense.py::main` so only one Markdown path prints and the CLI log line stays distinct, avoiding conflicting guidance before we archive evidence.
- [x] Extend `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest` so it asserts **exactly one** `Metrics digest:` line appears in stdout (guarding against future banner duplication) and continue to check for the `Metrics digest log:` reference. — commit `4cff9e38`.
- [x] Add a follow-on assertion in `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest` to ensure `stdout.count("Metrics digest log: ") == 1` so CLI log references cannot duplicate when future banner edits land (TYPE-PATH-001, TEST-CLI-001). — commit `32b20a94`.
- [x] Extend `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest` so it also asserts the success banner references `analysis/verification_report.json`, `analysis/verify_dense_stdout.log`, `analysis/check_dense_highlights.log`, and the SSIM grid summary/log paths. This keeps verification evidence lines guarded alongside the digest banner (TEST-CLI-001, TYPE-PATH-001). — commit `6a51d47a`.
- [x] Extend `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain` so the `--post-verify-only` path also asserts the SSIM grid summary/log plus verification report/log/highlights lines and ensures stubbed CLI logs exist (TEST-CLI-001, TYPE-PATH-001). — commit `ba93f39a`.
- **2025-11-11T131617Z reality check:** After stashing/restoring the deleted `data/phase_c/run_manifest.json` plus CLI/pytest logs to satisfy `git pull --rebase`, `{analysis}` still only contains `blocker.log` while `{cli}` is limited to `phase_c_generation.log`, `phase_d_dense.log`, and `run_phase_g_dense_stdout.log`. No SSIM grid, verification, preview, metrics, or artifact-inventory outputs exist yet, so the counted dense run + immediate `--post-verify-only` sweep remain the blocking deliverables for this phase.
- **2025-11-12T210000Z retrospective:** Latest hub inspection confirms nothing changed since the prior attempt—`analysis/` still only holds `blocker.log`, `cli/` still has the short trio of logs, and `data/phase_c/run_manifest.json` remains deleted (must be regenerated by the counted run, not restored manually). `cli/phase_d_dense.log` shows the last execution ran from `/home/ollie/Documents/PtychoPINN2` and failed with `ValueError: Object arrays cannot be loaded when allow_pickle=False`, so enforcing `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"` before every command stays mandatory. A quick `git log -10 --oneline` retrospective shows no new Ralph commits landed since the previous Do Now, so the focus remains ready_for_implementation with the same pytest + CLI guardrails.
- **2025-11-12T231800Z audit:** After stash→pull→pop the repo remains dirty only under this hub; `{analysis}` still just holds `blocker.log`, `cli/` still stops at `{phase_c_generation,phase_d_dense,run_phase_g_dense_stdout}.log`, and there are zero SSIM grid, verification, preview, metrics, or artifact-inventory artifacts. `cli/phase_d_dense.log` again ends with `ValueError: Object arrays cannot be loaded when allow_pickle=False`, confirming the dense rerun never advanced past Phase D inside `/home/ollie/Documents/PtychoPINN`.
- **2025-11-13T003000Z audit:** `timeout 30 git pull --rebase` now runs cleanly (no stash cycle needed) and `git log -10 --oneline` shows the overlap `allow_pickle=True` fix (`5cd130d3`) is already on this branch, so the lingering ValueError in `cli/phase_d_dense.log` is stale output from `/home/ollie/Documents/PtychoPINN2`. The active hub still lacks any counted dense evidence: `analysis/` only holds `blocker.log`, `{cli}` contains `{phase_c_generation,phase_d_dense,run_phase_g_dense_stdout(_retry,_v2).log}`, and there are zero SSIM grid summaries, verification/previews, metrics delta files, or artifact inventory snapshots. With the fix landed locally, the only remaining action is to rerun `run_phase_g_dense.py --clobber ...` followed immediately by `--post-verify-only` **from /home/ollie/Documents/PtychoPINN** so Phase C manifests regenerate and `{analysis,cli}` fill with real artifacts.
- **2025-11-11T191109Z audit:** `timeout 30 git pull --rebase` reported “Already up to date,” and revisiting docs/index.md + docs/findings.md reconfirmed the governing findings (POLICY-001, CONFIG-001, DATA-001, TYPE-PATH-001, STUDY-001, TEST-CLI-001, PREVIEW-PHASE-001, PHASEC-METADATA-001). The hub has **no progress**: `analysis/` still only contains `blocker.log`, `{cli}` is limited to `phase_c_generation.log`, `phase_d_dense.log`, and `run_phase_g_dense_stdout(_retry,_v2).log`, and `cli/phase_d_dense.log` again ends with the local `ValueError: Object arrays cannot be loaded when allow_pickle=False` stack trace even though the fix landed. There are zero SSIM grid summaries, verification/highlights logs, preview text, metrics digests, or artifact-inventory files. The counted dense rerun + immediate `--post-verify-only` sweep therefore remains the blocking deliverable; rerun both commands from `/home/ollie/Documents/PtychoPINN` with the mapped pytest guard and tee’d CLI logs into this hub.
- **2025-11-11T180800Z audit:** Repeated the `git stash push --include-untracked` → `timeout 30 git pull --rebase` → `git stash pop` flow to stay synced while preserving the deleted `data/phase_c/run_manifest.json` and local `.bak` notes. Hub contents remain unchanged: `analysis/` only has `blocker.log` and `cli/` still holds `phase_c_generation.log`, `phase_d_dense.log`, and the `run_phase_g_dense_stdout*.log` variants—there are still zero SSIM grid, verification, preview, metrics, highlights, or artifact-inventory outputs. Importantly, `cli/phase_d_dense.log` shows the failed overlap step was executed **inside** `/home/ollie/Documents/PtychoPINN` (not the secondary clone) and still hit `ValueError: Object arrays cannot be loaded when allow_pickle=False`, which means the post-fix pipeline has never been re-run. Guard every command with `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"` and regenerate the Phase C manifest via the counted run rather than restoring it manually.
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
