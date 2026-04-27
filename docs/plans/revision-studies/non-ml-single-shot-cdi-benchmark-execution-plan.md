# Non-ML Single-Shot CDI Benchmark Implementation Plan

> **Supersession note (2026-04-13):** The solver implementation sections in this plan predate the PyNX replacement decision and should not be used to continue the old `study_local_hio_er` implementation. Use `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-pynx-replacement-plan.md` for the default PyNX solver path and `docs/plans/revision-studies/non-ml-single-shot-cdi-known-probe-hio-er-plan.md` for the explicit known-probe object-domain diagnostic/candidate path. The data identity, metric contract, support policy, ambiguity policy, and artifact contract material here remains useful context unless one of those later plans says otherwise.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create a worktree for this initiative.

**Goal:** Produce reviewer-ready evidence, or a documented pivot, for a support-constrained non-ML single-shot CDI comparator against the Table 2 `C_g=1` synthetic line-pattern condition.

**Architecture:** Keep the benchmark as a narrow revision-study entry point under `scripts/reconstruction/`, with study-local HIO/ER helpers until the result proves reusable. Reuse existing grid-lines probe preparation, data generation, stitching, invocation logging, and `eval_reconstruction` conventions; write explicit manifests before inspecting any comparator metrics.

**Tech Stack:** Python, NumPy, TensorFlow/Keras only for same-split PtychoPINN reruns, pytest, existing PtychoPINN grid-lines workflow helpers, optional external CDI solver only if discovery passes provenance/license/install checks.

---

## Initiative

- ID: `non-ml-single-shot-cdi-benchmark`
- Status: pending implementation
- Source design: `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-design-seed.md`
- Plan path: `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-execution-plan.md`
- Workflow state root: `state/revision-study-priority-stack/non-ml-single-shot-cdi-benchmark/non-ml-single-shot-cdi-benchmark/plan-phase`
- Paper checklist: `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`

## Compliance Matrix

- [ ] **Design contract:** Preserve Table 2 constants: `N=64`, `gridsize=1`, `data_source="lines"`, `size=392`, `offset=4`, `outer_offset_train=8`, `outer_offset_test=20`, `nimgs_train=2`, `nimgs_test=2`, `nphotons=1e9`, and probe source `datasets/Run1084_recon3_postPC_shrunk_3.npz`.
- [ ] **Design contract:** Resolve the Table 2 data-identity branch and metric-contract manifest before inspecting HIO/ER or same-split rerun metrics.
- [ ] **Design contract:** Use direct support-anchored stitching/evaluation for the main HIO/ER row, without ground-truth shift/twin/orientation alignment.
- [ ] **Design contract:** Primary HIO/ER row uses `support_threshold=0.05`, restart seeds `[2026041201, 2026041202, 2026041203]`, `beta=0.9`, `1000` HIO iterations, `200` ER cleanup iterations, and final Fourier-amplitude residual restart selection.
- [ ] **Finding ID:** `CONFIG-001` - call `update_legacy_dict(params.cfg, config)` via existing grid-lines helpers before touching legacy data/model paths.
- [ ] **Finding ID:** `NORMALIZATION-001` - keep physics, statistical, and display/evaluation normalization separate; HIO/ER operates on stored normalized amplitude `X`.
- [ ] **Finding ID:** `STITCH-GRIDSIZE-001` - use `ptycho.workflows.grid_lines_workflow.stitch_predictions` or an equivalent study-local gridsize-1-safe stitch path.
- [ ] **Finding ID:** `TF-REPEATED-MODEL-OOM-001` - if the same-split rerun branch trains multiple TensorFlow models, use subprocess isolation or explicit Keras/backend cleanup between runs.
- [ ] **Project policy:** No edits to `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` unless a later approved plan explicitly authorizes it.
- [ ] **Project policy:** Use PATH `python`; for long-running commands use tmux with `ptycho311` activated or PATH configured to that environment, track the exact launched PID, and do not duplicate runs writing to the same `--output-root`.

## File Structure

- Create: `scripts/reconstruction/hio_cdi_benchmark.py`
  - Study CLI and local helper functions:
    - `discover_solvers`
    - `make_probe_support`
    - `forward_amplitude`
    - `project_fourier_magnitude`
    - `hio_update`
    - `er_cleanup`
    - `run_restarts`
    - `select_restart_by_residual`
    - `build_table2_condition`
    - `write_solver_manifest`
    - `write_data_identity_manifest`
    - `write_metric_contract_manifest`
    - `write_pinn_randomness_manifest`
    - `write_benchmark_manifest`
  - Reuse from `ptycho.workflows.grid_lines_workflow`: `GridLinesConfig`, `load_probe_guess`, `load_ideal_disk_probe`, `normalize_probe_transform_pipeline`, `apply_probe_transform_pipeline`, `apply_probe_mask`, `simulate_grid_data`, `configure_legacy_params`, `stitch_predictions`, and `save_recon_artifact`.
  - Reuse `scripts.studies.invocation_logging.write_invocation_artifacts` and `capture_runtime_provenance`.

- Create: `tests/scripts/test_hio_cdi_benchmark.py`
  - Focused unit tests for support construction, normalized Fourier amplitude projection, HIO/ER update behavior, residual/restart selection, non-oracle ambiguity policy, duplicate output-root refusal, and manifest writing.

- Optional, only if `hio_cdi_benchmark.py` becomes too large during implementation: create `scripts/reconstruction/hio_cdi_lib.py` for pure helper functions. Do not promote helpers to `ptycho/` modules in this plan.

- Modify only if a narrow compatibility blocker is proven before benchmark work can proceed:
  - `ptycho/evaluation.py` or import-adjacent FRC compatibility files, with a dedicated regression test. Do not rewrite metric definitions.

## Context Priming

Re-read before implementation:

- `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-design-seed.md`
- `docs/index.md`
- `docs/findings.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/DATA_GENERATION_GUIDE.md`
- `docs/DATA_NORMALIZATION_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/development/INVOCATION_LOGGING_GUIDE.md`
- `specs/data_contracts.md`
- `ptycho/workflows/grid_lines_workflow.py`
- `ptycho/evaluation.py`
- `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`
- `/home/ollie/Documents/ptychopinnpaper2/data/README.md`
- `/home/ollie/Documents/ptychopinnpaper2/data/sim_lines_4x_metrics.json`
- `/home/ollie/Documents/ptychopinnpaper2/tables/scripts/generate_sim_lines_4x_metrics.py`
- `/home/ollie/Documents/ptychopinnpaper2/tables/sim_lines_4x_metrics.tex`

## Tranche A - Preflight, Solver Discovery, and Frozen Gates

Purpose: complete the non-metric prerequisites and decide whether an external solver is acceptable before coding against one.

- [ ] A1: Verify import preflight from the target environment.

  Run:

  ```bash
  python - <<'PY'
  from ptycho.evaluation import eval_reconstruction
  from ptycho.workflows.grid_lines_workflow import GridLinesConfig, stitch_predictions
  print("preflight imports ok")
  PY
  ```

  Expected: prints `preflight imports ok`. If this fails because optional FRC code is unavailable, stop and create the narrow compatibility test/fix before proceeding.

- [ ] A2: Search repo, PyPI, GitHub, and the web for single-frame CDI HIO/ER/ADMM implementations. Record candidates and reject ptychographic multi-frame solvers for the reviewer-facing row.

  Minimum solver manifest fields:

  ```json
  {
    "search_date": "YYYY-MM-DD",
    "searched_sources": ["repo", "PyPI", "GitHub", "web"],
    "candidates": [
      {
        "name": "...",
        "source_url": "...",
        "package_version": "...",
        "license": "...",
        "install_command": "...",
        "api_entry_point": "...",
        "accepted": false,
        "reason": "..."
      }
    ],
    "selected_solver": "study_local_hio_er"
  }
  ```

  Default result if no acceptable external solver is found: use the study-local HIO/ER implementation.

- [ ] A3: Decide and record the Table 2 data-identity branch before metric inspection:
  - Preferred first attempt: frozen-artifact branch only if exact Table 2 `C_g=1` inputs and reconstruction/metric artifacts can be located and checksummed.
  - Fallback: same-split rerun branch, using loader-compatible data generation unless the plan is updated before generation to support a study-local seeded branch.

- [ ] A4: Reconcile paper-side Table 2 metric notes before metric inspection:
  - `/home/ollie/Documents/ptychopinnpaper2/data/sim_lines_4x_metrics.json` says `align_for_evaluation` with `global_offsets` and `stitch_patch_size=20`.
  - `/home/ollie/Documents/ptychopinnpaper2/tables/scripts/generate_sim_lines_4x_metrics.py` and `/home/ollie/Documents/ptychopinnpaper2/tables/sim_lines_4x_metrics.tex` say `nsamples=1000`, `seed=7`.
  - `ptycho.workflows.grid_lines_workflow.run_tf_comparison_workflow` stitches with `stitch_predictions(...)` and calls `eval_reconstruction(...)` directly.

- [ ] A5: Resolve or explicitly annotate the `gs2_ideal_nll` versus `gs2_ideal` provenance mismatch before any paper-table regeneration. This is a gate for later reporting, not for implementing the HIO/ER script.

Verification for Tranche A:

- [ ] `eval_reconstruction` import preflight passes or a blocking compatibility issue is documented.
- [ ] Solver manifest exists under `.artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/<run_id>/solver_manifest.json`.
- [ ] Data-identity branch decision exists and explicitly says whether old Table 2 metrics can be treated as same-data comparator or only historical context.
- [ ] Metric-contract manifest draft exists with each paper-side note marked `authoritative`, `stale`, or `unresolved`.
- [ ] No HIO/ER or same-split PtychoPINN metrics have been inspected yet.

## Tranche B - Test-First Study-Local HIO/ER Core

Purpose: implement the classical baseline mechanics in a small, testable script without changing stable physics/model modules.

- [ ] B1: Add failing tests in `tests/scripts/test_hio_cdi_benchmark.py` for `make_probe_support`.
  - Empty/full support is invalid.
  - Threshold `0.05` records pixel count and support fraction.
  - Threshold grid `[0.01, 0.05, 0.10]` is preserved without metric-based selection.

  Run:

  ```bash
  pytest tests/scripts/test_hio_cdi_benchmark.py -k support -vv
  ```

  Expected before implementation: fail because the module/functions do not exist.

- [ ] B2: Implement support-mask helpers and manifest serialization in `scripts/reconstruction/hio_cdi_benchmark.py`.

- [ ] B3: Add failing tests for normalized Fourier convention:
  - `forward_amplitude(psi)` computes `abs(fftshift(fft2(psi)) / sqrt(N * N))`.
  - `project_fourier_magnitude` preserves the target normalized magnitude while retaining phase.
  - The self-consistency helper rejects shape mismatches and non-finite arrays.

  Run:

  ```bash
  pytest tests/scripts/test_hio_cdi_benchmark.py -k "fourier or projection" -vv
  ```

- [ ] B4: Implement Fourier projection and residual helpers.

- [ ] B5: Add failing tests for HIO/ER update behavior and restart selection:
  - HIO update applies inside-support Fourier projection and outside-support feedback with `beta=0.9`.
  - ER cleanup enforces support without using ground truth.
  - Residual is `norm(abs(F(psi)) - X) / max(norm(X), 1e-12)`.
  - Restart selection chooses lowest final residual and breaks ties by lower seed.
  - Per-restart residual curves are retained.

  Run:

  ```bash
  pytest tests/scripts/test_hio_cdi_benchmark.py -k "hio or er or restart or residual" -vv
  ```

- [ ] B6: Implement HIO/ER loop, restart seeding, residual recording, and object-patch recovery:
  - Initialize from stored normalized `X` plus deterministic random phase.
  - Use at least three restarts for benchmark runs.
  - Use `probe_division_epsilon = 1e-6 * max(abs(P))`.
  - Recover `O_norm = psi / P_safe` inside support and write zero outside support for the main artifact.

- [ ] B7: Add failing tests for ambiguity-policy enforcement:
  - Main row path does not call ground-truth shift/twin/orientation alignment.
  - Oracle diagnostics, if requested, must use a separate output label and cannot replace the primary row.

  Run:

  ```bash
  pytest tests/scripts/test_hio_cdi_benchmark.py -k "ambiguity or oracle" -vv
  ```

Verification for Tranche B:

- [ ] Targeted HIO/ER unit tests pass:

  ```bash
  pytest tests/scripts/test_hio_cdi_benchmark.py -vv
  ```

- [ ] No changes were made to `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- [ ] HIO/ER helpers operate on normalized amplitude `X`, not photon-scaled intensity.

## Tranche C - CLI, Manifests, and Table 2 Contract Freezing

Purpose: make the study runnable and auditable before any benchmark metrics are consumed.

- [ ] C1: Add CLI parsing to `scripts/reconstruction/hio_cdi_benchmark.py` with at least:
  - `--output-root`
  - `--run-id`
  - `--probe-npz`
  - `--probe-source {custom,ideal_disk}`
  - `--probe-scale-mode`
  - `--probe-smoothing-sigma`
  - `--support-thresholds`
  - `--primary-support-threshold`
  - `--restart-seeds`
  - `--beta`
  - `--hio-iters`
  - `--er-iters`
  - `--max-test-frames`
  - `--data-identity-branch {frozen-artifact,same-split-rerun}`
  - `--data-generation-control {loader-compatible,study-local-seeded}`
  - `--metric-contract-mode {direct-stitch,align-for-evaluation,unresolved}`
  - `--preflight-only`
  - `--smoke`
  - `--force`

- [ ] C2: Add invocation logging at the start of `main`, before expensive work, using `write_invocation_artifacts(output_dir=args.output_root, script_path="scripts/reconstruction/hio_cdi_benchmark.py", ...)`.

- [ ] C3: Refuse duplicate output roots unless `--force` is passed and the command is explicitly a destructive rerun. The normal benchmark path must fail if required artifacts already exist.

- [ ] C4: Implement `--preflight-only` to write:
  - `solver_manifest.json`
  - `data_identity_manifest.json`
  - `metric_contract_manifest.json`
  - `runtime_provenance.json`
  - `manifest.json`
  - `invocation.json`
  - `invocation.sh`

- [ ] C5: For frozen-artifact branch, locate and checksum exact Table 2 `C_g=1` artifacts if available:
  - `.artifacts/sim_lines_4x_metrics_2026-01-27/gs1_custom/metrics.json`
  - `.artifacts/sim_lines_4x_metrics_2026-01-27/gs1_ideal/metrics.json`
  - any corresponding train/test NPZs, `YY_ground_truth`, `norm_Y_I`, probe array after transform, and `YY_pred` artifacts found.

- [ ] C6: For same-split rerun branch, implement loader-compatible default data generation:
  - Use `simulate_grid_data(...)` and record that train/test object/noise seeds come from the loader path: train seed `1`, test seed `2`.
  - Disable memoization with `PTYCHO_DISABLE_MEMOIZE=1` for freshly generated bundles, or record cache mode, cache key, path, checksum, and branch label if reused.
  - Persist generated train/test NPZs and key-level checksums.

- [ ] C7: If same-split rerun is required, freeze PtychoPINN randomness before model construction:
  - Default primary training seed: `2026041211`.
  - Fresh process with `PYTHONHASHSEED=2026041211`.
  - `TF_DETERMINISTIC_OPS=1` where supported.
  - `tf.keras.utils.set_random_seed(2026041211)` plus `tf.config.experimental.enable_op_determinism()` when available.
  - If deterministic training order cannot be proven, switch to stochastic repeated-rerun mode with seeds `[2026041211, 2026041212, 2026041213]`.

Verification for Tranche C:

- [ ] CLI and manifest tests pass:

  ```bash
  pytest tests/scripts/test_hio_cdi_benchmark.py -k "cli or manifest or output_root or invocation or data_identity or metric_contract" -vv
  ```

- [ ] Preflight command succeeds without computing HIO/ER metrics:

  ```bash
  PTYCHO_DISABLE_MEMOIZE=1 python scripts/reconstruction/hio_cdi_benchmark.py \
    --output-root .artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/preflight_gs1_custom \
    --run-id preflight_gs1_custom \
    --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
    --probe-source custom \
    --probe-scale-mode pad_preserve \
    --probe-smoothing-sigma 0.5 \
    --support-thresholds 0.01 0.05 0.10 \
    --primary-support-threshold 0.05 \
    --restart-seeds 2026041201 2026041202 2026041203 \
    --data-identity-branch frozen-artifact \
    --metric-contract-mode unresolved \
    --preflight-only
  ```

- [ ] Preflight output includes all required manifests and no `metrics.json` from HIO/ER.
- [ ] If the metric contract remains unresolved, the manifest labels the result as non-Table-2-compatible exploratory only.

## Tranche D - Smoke Benchmark

Purpose: prove the fixed HIO/ER path runs on a small subset and can be stitched/evaluated without oracle alignment.

- [ ] D1: Run a support/fourier self-consistency smoke on 8 to 16 test frames for `gs1_custom`.

  Use tmux if runtime is long enough to leave unattended. Non-tmux foreground smoke command:

  ```bash
  PTYCHO_DISABLE_MEMOIZE=1 python scripts/reconstruction/hio_cdi_benchmark.py \
    --output-root .artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/smoke_gs1_custom \
    --run-id smoke_gs1_custom \
    --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
    --probe-source custom \
    --probe-scale-mode pad_preserve \
    --probe-smoothing-sigma 0.5 \
    --support-thresholds 0.05 \
    --primary-support-threshold 0.05 \
    --restart-seeds 2026041201 2026041202 2026041203 \
    --beta 0.9 \
    --hio-iters 1000 \
    --er-iters 200 \
    --max-test-frames 16 \
    --data-identity-branch frozen-artifact \
    --metric-contract-mode direct-stitch \
    --smoke
  ```

- [ ] D2: If the frozen-artifact branch cannot locate exact same-data artifacts, rerun the smoke on the same-split branch and label old Table 2 values historical only.

- [ ] D3: Confirm smoke artifacts:
  - `manifest.json`
  - `solver_manifest.json`
  - `data_identity_manifest.json`
  - `metric_contract_manifest.json`
  - `metrics_gs1_custom_support_0p05.json`
  - `residuals_gs1_custom_support_0p05.json`
  - `recons/gs1_custom_support_0p05/recon.npz`
  - `invocation.json`
  - `invocation.sh`

- [ ] D4: Confirm smoke quality gates without post-hoc parameter selection:
  - support is nonempty and not full-frame
  - residuals are finite
  - no NaNs in `O_norm`, stitched object, or metrics
  - the forward/amplitude self-consistency check passes
  - any coordinate-convention correction is justified by self-consistency, not metric improvement
  - no ground-truth shift/twin/orientation alignment was used in the main row

Verification for Tranche D:

- [ ] Smoke run exits code `0`.
- [ ] Required artifacts are freshly written after the launched process exits.
- [ ] Smoke metrics are labeled `smoke` and cannot be promoted to the reviewer-facing row.
- [ ] If smoke fails, write `.artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/smoke_gs1_custom/attempt_note.md` with exact failure mode and decide whether to fix or pivot.

## Tranche E - Same-Split PtychoPINN Comparator, Only If Needed

Purpose: provide a fair comparator when frozen Table 2 same-data artifacts cannot be proven.

- [ ] E1: Before training, write `pinn_randomness_manifest.json` with seed policy, determinism settings, TensorFlow/Keras/NumPy/Python versions, CUDA/cuDNN info when available, and metric-contract checksum.

- [ ] E2: Generate or reuse the same deterministic test bundle under the data-generation branch chosen in Tranche C.

- [ ] E3: Run the primary PtychoPINN seed `2026041211` in a fresh process and persist:
  - initial model weight checksums
  - final model/checkpoint checksums
  - training history
  - invocation artifacts
  - environment variables
  - metric output

- [ ] E4: If deterministic mode is invalid, run stochastic repeated-rerun mode for seeds `[2026041211, 2026041212, 2026041213]` and report median amplitude SSIM and PSNR with min-to-max range. Do not compare HIO/ER against the best seed only.

- [ ] E5: Compare the same-split rerun against historical `gs1_custom` Table 2 context only after metrics are produced:
  - expected historical amplitude SSIM: `0.9044216561120993`
  - expected historical amplitude PSNR: `68.8864772792175`
  - Table-2-compatible tolerance: `abs(delta amplitude SSIM) <= 0.02` and `abs(delta amplitude PSNR) <= 2.0 dB`

Verification for Tranche E:

- [ ] `pinn_randomness_manifest.json` exists before model construction/training.
- [ ] Same-split data bundle checksums match between PtychoPINN and HIO/ER.
- [ ] Same-split PtychoPINN metrics are never merged into the old Table 2 row unless tolerance and data-identity conditions are satisfied.
- [ ] For repeated TensorFlow training, subprocess isolation or explicit cleanup is used to avoid `TF-REPEATED-MODEL-OOM-001`.

## Tranche F - Full HIO/ER Benchmark Attempt

Purpose: run the reviewer-critical baseline under the fixed policy.

- [ ] F1: Run `gs1_custom` primary row first with support threshold `0.05`.

  Long-run command pattern using tmux and exact PID tracking:

  ```bash
  tmux new-session -d -s hio_cdi_gs1_custom_primary \
    'bash -lc "cd /home/ollie/Documents/PtychoPINN && conda activate ptycho311 && PTYCHO_DISABLE_MEMOIZE=1 python scripts/reconstruction/hio_cdi_benchmark.py \
      --output-root .artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/full_gs1_custom_primary \
      --run-id full_gs1_custom_primary \
      --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
      --probe-source custom \
      --probe-scale-mode pad_preserve \
      --probe-smoothing-sigma 0.5 \
      --support-thresholds 0.05 \
      --primary-support-threshold 0.05 \
      --restart-seeds 2026041201 2026041202 2026041203 \
      --beta 0.9 \
      --hio-iters 1000 \
      --er-iters 200 \
      --data-identity-branch frozen-artifact \
      --metric-contract-mode direct-stitch & pid=\$!; wait \"\$pid\""'
  ```

  If `conda activate ptycho311` is unavailable in the tmux shell, use the configured `ptycho311` PATH `python` explicitly and record that environment choice in `runtime_provenance.json`.

- [ ] F2: After the exact launched PID exits with code `0`, verify artifacts exist and are freshly written. Do not use broad `pgrep -f` polling as the completion check.

- [ ] F3: Run support sensitivity rows `0.01` and `0.10` only after the primary `0.05` row is complete. Keep `0.05` as the primary row regardless of metric outcome.

- [ ] F4: Run `gs1_ideal` only after `gs1_custom` is stable or if the failed/negative result needs a controlled idealized-probe comparison.

- [ ] F5: Run oracle diagnostics only under a separate label if ambiguity analysis is needed. Oracle diagnostics must not replace the main support-anchored row.

Verification for Tranche F:

- [ ] Each full run writes:
  - `manifest.json`
  - `solver_manifest.json`
  - `data_identity_manifest.json`
  - `metric_contract_manifest.json`
  - `metrics_*.json`
  - `residuals_*.json`
  - reconstruction NPZ with `YY_pred`
  - invocation artifacts
- [ ] Metrics JSON records all HIO/ER hyperparameters, support policy, ambiguity policy, threshold grid status, data-identity branch, metric-contract id, seeds, residuals, timing, and selected restart.
- [ ] No ptychographic multi-frame solver is labeled as single-shot CDI.
- [ ] No threshold, beta, iteration count, cleanup length, epsilon, or restart is selected by ground-truth metrics.

## Tranche G - Outcome Gate and Handoff

Purpose: decide whether the implementation produces reviewer-ready evidence or a pivot note.

- [ ] G1: If Table-2-compatible evidence exists, write an outcome summary under `.artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/<run_id>/outcome_summary.md` with:
  - branch decision
  - metric-contract decision
  - primary `gs1_custom` HIO/ER row
  - completed support sensitivity rows
  - comparison policy against PtychoPINN
  - caveats about support/probe priors and CDI ambiguities

- [ ] G2: If evidence is not comparable, write a pivot note under the same run root with:
  - attempted solver path
  - exact commands
  - failure mode
  - why the result is not a valid/reproducible comparator
  - proposed manuscript/reviewer-response claim narrowing

- [ ] G3: Update `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md` progress notes after the outcome gate. Keep `changelog.txt` synchronized only if reviewer-facing manuscript/table text changes are made.

- [ ] G4: Do not regenerate paper data JSON, Table 2, the manuscript, or changelog in this plan unless the outcome summary explicitly accepts the result as comparable and a follow-up paper-update plan is approved.

Verification for Tranche G:

- [ ] Exactly one of these exists for the primary full attempt: `outcome_summary.md` or `pivot_note.md`.
- [ ] Outcome/pivot document links to the exact run root and manifests.
- [ ] Reviewer checklist has a concise progress note if this initiative advanced beyond planning.
- [ ] Paper-side files are unchanged unless a separate approved paper-update step exists.

## Compatibility Boundaries and Migrations

- No shared API migration is planned. HIO/ER code remains study-local under `scripts/reconstruction/` until a successful result and follow-up design justify promotion.
- No core physics/model migration is planned. Do not edit `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not add a committed external dependency unless solver discovery proves it is necessary, reproducibly installable, and license-compatible. Prefer manifesting external candidates and using study-local HIO/ER.
- Data generation must use existing grid-lines contracts. Same-split loader-compatible generation records train seed `1` and test seed `2`; study-local seeded generation requires an explicit implementation change before use.
- Evaluation must either match the resolved Table 2 contract or label deviations. A different crop, subset, or alignment path cannot be called Table-2-compatible.
- HIO/ER reconstructions must remain in normalized object units until `stitch_predictions(..., norm_Y_I, part="complex")` or the approved equivalent restores object scale at the evaluation boundary.
- Oracle ambiguity diagnostics are compatibility diagnostics, not the main comparator.

## Explicit Non-Goals

- Do not build a general CDI package.
- Do not implement a full ADMM framework unless solver discovery finds a clean, bounded integration.
- Do not use Tike, PtyChi, RPIE, DM, LSQML, PIE, or any multi-position ptychographic solver as the requested single-shot CDI baseline.
- Do not add learned, supervised, trainable-probe, or joint probe/position-refinement baselines.
- Do not rewrite `eval_reconstruction` or Table 2 generation unless a narrow compatibility blocker is proven and fixed with tests.
- Do not claim broad superiority or inferiority against all classical CDI methods.
- Do not edit the manuscript, reviewer response, changelog, or paper tables until the outcome gate accepts comparable evidence or a pivot response is approved.

## Verification Commands

Targeted unit and CLI checks:

```bash
pytest tests/scripts/test_hio_cdi_benchmark.py -vv
```

Import and metric preflight:

```bash
python - <<'PY'
from ptycho.evaluation import eval_reconstruction
from ptycho.workflows.grid_lines_workflow import GridLinesConfig, stitch_predictions
print("preflight imports ok")
PY
```

Preflight-only manifest run:

```bash
PTYCHO_DISABLE_MEMOIZE=1 python scripts/reconstruction/hio_cdi_benchmark.py \
  --output-root .artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/preflight_gs1_custom \
  --run-id preflight_gs1_custom \
  --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --probe-source custom \
  --probe-scale-mode pad_preserve \
  --probe-smoothing-sigma 0.5 \
  --support-thresholds 0.01 0.05 0.10 \
  --primary-support-threshold 0.05 \
  --restart-seeds 2026041201 2026041202 2026041203 \
  --data-identity-branch frozen-artifact \
  --metric-contract-mode unresolved \
  --preflight-only
```

Smoke run:

```bash
PTYCHO_DISABLE_MEMOIZE=1 python scripts/reconstruction/hio_cdi_benchmark.py \
  --output-root .artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/smoke_gs1_custom \
  --run-id smoke_gs1_custom \
  --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --probe-source custom \
  --probe-scale-mode pad_preserve \
  --probe-smoothing-sigma 0.5 \
  --support-thresholds 0.05 \
  --primary-support-threshold 0.05 \
  --restart-seeds 2026041201 2026041202 2026041203 \
  --beta 0.9 \
  --hio-iters 1000 \
  --er-iters 200 \
  --max-test-frames 16 \
  --data-identity-branch frozen-artifact \
  --metric-contract-mode direct-stitch \
  --smoke
```

## Completion Criteria

- [ ] Plan target exists at `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-execution-plan.md`.
- [ ] `scripts/reconstruction/hio_cdi_benchmark.py` exists with invocation logging and manifest-producing CLI.
- [ ] `tests/scripts/test_hio_cdi_benchmark.py` passes.
- [ ] Solver discovery manifest exists and rejects multi-frame ptychographic solvers for the reviewer-facing row.
- [ ] Data-identity manifest exists and selects frozen-artifact or same-split rerun branch with checksums and comparator policy.
- [ ] Metric-contract manifest exists and resolves or labels paper-side alignment/subsampling notes.
- [ ] If same-split rerun is used, PtychoPINN randomness manifest exists before model construction.
- [ ] Smoke run exits code `0` and writes fresh required artifacts.
- [ ] Full `gs1_custom` primary row either produces comparable benchmark evidence or a pivot note with exact failure mode and proposed response.
- [ ] Reviewer checklist is updated at the outcome gate, and paper/changelog edits are deferred unless a follow-up paper-update step is approved.

## Artifacts Index

- Plan: `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-execution-plan.md`
- Superseding default solver plan: `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-pynx-replacement-plan.md`
- Superseding secondary known-probe diagnostic/candidate plan: `docs/plans/revision-studies/non-ml-single-shot-cdi-known-probe-hio-er-plan.md`
- Workflow pointer: `state/revision-study-priority-stack/non-ml-single-shot-cdi-benchmark/non-ml-single-shot-cdi-benchmark/plan-phase/plan_path.txt`
- Benchmark run root: `.artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/<run_id>/`
- Review report target reserved by workflow: `artifacts/review/revision-studies/non-ml-single-shot-cdi-benchmark-plan-review.json`
- Execution report target reserved by workflow: `artifacts/work/revision-studies/non-ml-single-shot-cdi-benchmark-execution-report.md`

## Documents Read While Drafting This Plan

- `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-design-seed.md`
- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/plans/templates/implementation_plan.md`
- `docs/DATA_GENERATION_GUIDE.md`
- `docs/DATA_NORMALIZATION_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/development/INVOCATION_LOGGING_GUIDE.md`
- `specs/data_contracts.md`
- `ptycho/workflows/grid_lines_workflow.py`
- `ptycho/evaluation.py`
- `scripts/studies/grid_lines_workflow.py`
- `scripts/studies/probe_mischaracterization_stress_test.py`
- `scripts/reconstruction/README.md`
- `tests/studies/test_probe_mischaracterization_stress_test.py`
- `tests/test_grid_lines_invocation_logging.py`
- `state/revision-study-priority-stack/non-ml-single-shot-cdi-benchmark/current-item-inputs.json`
- `state/revision-study-priority-stack/non-ml-single-shot-cdi-benchmark/non-ml-single-shot-cdi-benchmark/plan-phase/design_path.txt`
- `state/revision-study-priority-stack/non-ml-single-shot-cdi-benchmark/non-ml-single-shot-cdi-benchmark/plan-phase/open_findings.json`
- `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`
- `/home/ollie/Documents/ptychopinnpaper2/data/README.md`
- `/home/ollie/Documents/ptychopinnpaper2/data/sim_lines_4x_metrics.json`
- `/home/ollie/Documents/ptychopinnpaper2/tables/scripts/generate_sim_lines_4x_metrics.py`
- `/home/ollie/Documents/ptychopinnpaper2/tables/sim_lines_4x_metrics.tex`
