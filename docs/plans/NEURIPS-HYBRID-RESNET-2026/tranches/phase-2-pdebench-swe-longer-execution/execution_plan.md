# Phase 2 PDEBench SWE Longer Execution, Baselines, and Pivot Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the post-smoke Roadmap Phase 2 PDEBench SWE one-step evidence package: longer Hybrid ResNet execution, local FNO and U-Net baselines, focused spectral/local ablations only if the primary comparison remains viable, and a durable proceed/pivot/block summary.

**Architecture:** Extend the existing `scripts/studies/pdebench_swe/` smoke harness into a reusable longer-run path instead of creating a parallel benchmark stack. Keep data identity, full split/subset manifests, model profiles, metrics, run provenance, long-run launching, freshness validation, result collation, and durable summary writing as separate responsibilities so the final `pde_execution_summary.md` can be reviewed without trusting ad hoc logs. All bulky support artifacts remain under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/`.

**Tech Stack:** Python 3.11 via PATH `python`, `ptycho311` for long-running tmux launches, `h5py`, NumPy, PyTorch, `neuralop.models.FNO` when available, existing `ptycho_torch` Hybrid ResNet components, Markdown/JSON/CSV provenance artifacts.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Title: Phase 2 PDEBench SWE Longer Execution, Baselines, and Pivot Gate
- Status: pending
- Spec/Source:
  - Design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
  - Roadmap: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
  - Tranche context: `state/NEURIPS-HYBRID-RESNET-2026/items/phase-2-pdebench-swe-longer-execution/tranche-context.md`
  - Phase 1 selection: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`
  - Phase 2 smoke gate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_swe_smoke_gate.md`
  - Plan review report: `.artifacts/review/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution-plan-review.json`
- Plan path: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-swe-longer-execution/execution_plan.md`

## Compliance Matrix

- [ ] **Roadmap Phase Order:** Execute only Roadmap Phase 2 longer SWE execution. Do not start Phase 3 CDI anchor regeneration, Phase 4 `256x256` CDI scaling, or Phase 5 paper-facing artifact assembly.
- [ ] **Primary Benchmark Pin:** Use PDEBench 2D SWE with official file `/home/ollie/Documents/pdebench-data/swe/2D_rdb_NA_NA.h5`, DaRUS datafile `133021`, SHA256 `28f0c33723d70eebb420fc170e94b675c18e032fb697dcef080e114ca9645e3a`, grouped `*/data`, axis order `NTHWC`, and shape `[1000, 101, 128, 128, 1]`.
- [ ] **Smoke Decision Binding:** The approved smoke gate decision is `proceed with longer SWE execution`; do not skip to CDI polish while this Phase 2 gate is open.
- [ ] **Shared Contract Across Models:** Hybrid ResNet, FNO, U-Net, and any Hybrid ResNet ablation profiles must share the same HDF5 file, full split, run subset, one-step horizon, normalization policy, metric implementation, and evaluation budget.
- [ ] **Local Baselines:** Run local FNO and U-Net baselines under the same contract, or write `pde_execution_summary.md` as `block` or `pivot` if the local-baseline requirement cannot be met.
- [ ] **Review Follow-Ups:** Before reporting longer evidence, either reset CUDA peak memory per profile or label memory as process-level peak; clarify normalization limit semantics so samples and batches are not conflated; pin and record a fixed model/training RNG seed.
- [ ] **License/Access:** Resolve and record PDEBench repository/data license and DaRUS access terms before using longer-run results for paper-facing claims. If terms remain unresolved, the summary must visibly caveat them.
- [ ] **Published SOTA Boundary:** Published SOTA may be mentioned only as protocol-dependent context unless the same task, split, preprocessing, model code or accepted reimplementation, metric code, and evaluation horizon are locally reproduced.
- [ ] **Artifact Hygiene:** Store bulky logs, checkpoints, raw metrics, machine-readable manifests, and run outputs under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/` or another ignored/external root named in this plan's summary. Do not commit the HDF5 file or raw long-run outputs.
- [ ] **Durable Summary:** Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md` with exactly one decision: `proceed to CDI Phase 3`, `pivot to OpenFWI FlatVel-A`, or `block for human decision`.
- [ ] **Discoverability:** Update `docs/studies/index.md` and `docs/index.md` if this tranche creates a durable longer-run runbook, new summary document, or other discoverable tracked surface.
- [ ] **Stable Modules:** Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- [ ] **No Worktrees:** Use the current checkout only.
- [ ] **Interpreter Policy:** Use PATH `python` in commands and subprocesses. Do not introduce repository-specific interpreter wrappers.
- [ ] **Long-Run Guardrail:** Launch long-running commands in tmux with `ptycho311`, persist the selected `run_id`, `RUN_ROOT`, and tmux session under the raw-root logs before launch, track the exact launched child PID using `cmd ... & pid=$!; wait "$pid"`, always write the child exit code after `wait`, and do not launch a duplicate run writing to the same output root.
- [ ] **Freshness Gate:** Treat a long run as complete only when the tracked PID exits `0` and required artifacts exist, match the persisted selected `run_id` and `RUN_ROOT`, match the tracked PID/provenance where applicable, and have mtimes newer than the recorded start marker. Do not validate by sorting existing run directories.
- [ ] **Dirty Worktree Safety:** Preserve unrelated dirty files. Only touch files named by this plan unless a focused blocker requires a narrow companion file.

## Spec Alignment

- **Normative roadmap phase:** Phase 2 - Deep PDE Benchmark Execution.
- **Covered slice:** post-smoke primary benchmark execution for PDEBench 2D SWE, including longer one-step Hybrid ResNet execution, FNO and U-Net local baselines, focused spectral/local ablations if viable, and the Phase 2 proceed/pivot/block summary.
- **Required durable output:** `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md`.
- **Required ignored support root:** `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/`.
- **Required support artifacts:**
  - `preflight/disk_gpu.json`
  - `preflight/license_access.md`
  - `preflight/package_provenance.json`
  - `run_budget.json`
  - `dataset_manifest.json`
  - `hdf5_metadata.json`
  - `split_manifest_full.json`
  - `split_manifest_run.json`
  - `normalization_stats.json`
  - `runs/<profile_id>/metrics.json` or `runs/<profile_id>/blocker.json`
  - `runs/<profile_id>/provenance.json`
  - `comparison_summary.csv`
  - `comparison_summary.json`
  - root `invocation.json` and `invocation.sh`
  - raw-root `logs/selected_longer.run_id`, `logs/selected_longer.run_root`, `logs/selected_longer.tmux_session`
  - per-run `logs/longer.run_id`, `logs/longer.started_at_ns`, `logs/longer.pid`, `logs/longer.exit_code`, stdout/stderr logs
- **Explicit non-goals:** no CDI anchor regeneration, CDI baselines, CDI ablations, classical CDI/PyNX/HIO/ER comparisons, `256x256` CDI scaling, OpenFWI execution, full NS/CFD/PDEArena benchmark switch, `/home/ollie/Documents/neurips/` artifacts, manuscript prose, worktree creation, stable core physics/model edits, or same-protocol SOTA claims without matched reproduction.

## Documents Read For This Plan Draft

- User-provided AGENTS/CLAUDE instructions for `/home/ollie/Documents/PtychoPINN`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `state/NEURIPS-HYBRID-RESNET-2026/items/phase-2-pdebench-swe-longer-execution/tranche-context.md`
- `state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-2-pdebench-swe-longer-execution/plan-phase/plan_path.txt`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `docs/index.md`
- `docs/templates/` listing
- `docs/plans/templates/implementation_plan.md`
- `docs/findings.md`
- `docs/studies/index.md`
- `docs/workflows/pytorch.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/TESTING_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/development/INVOCATION_LOGGING_GUIDE.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_swe_smoke_gate.md`
- `.artifacts/review/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate-implementation-review.md`
- `scripts/studies/run_pdebench_swe_smoke.py`
- `scripts/studies/pdebench_swe/manifest.py`
- `scripts/studies/pdebench_swe/splits.py`
- `scripts/studies/pdebench_swe/data.py`
- `scripts/studies/pdebench_swe/metrics.py`
- `scripts/studies/pdebench_swe/models.py`
- `scripts/studies/pdebench_swe/smoke.py`
- `tests/studies/test_pdebench_swe_manifest.py`
- `tests/studies/test_pdebench_swe_splits_data.py`
- `tests/studies/test_pdebench_swe_metrics.py`
- `tests/studies/test_pdebench_swe_models.py`
- `tests/studies/test_pdebench_swe_smoke_cli.py`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-1-pde-benchmark-selection/execution_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-swe-primary-smoke-gate/execution_plan.md`

## Plan Review Finding Resolutions

- `PLAN-H1` is resolved in this revision by making Phase F persist the selected `RUN_ID`, `RUN_ROOT`, and tmux session before launch; making the tmux wrapper use `set +e` around `wait "$pid"` so `logs/longer.exit_code` is written for nonzero child exits; and making Phase F/H validation read the persisted selected root instead of `sorted(raw_root / "runs")`. The freshness checks now compare the selected run root, run ID, child PID, invocation/provenance metadata, exit code, start marker, and required artifact mtimes.

## Implementation Review Finding Resolutions

- `H1` is resolved by treating `logs/longer.pid` as incomplete-run evidence until `logs/longer.exit_code` exists. The Python runner rejects any PID marker without `logs/longer.exit_code` regardless of PID liveness and regardless of `--allow-existing-output-root`; the tmux wrapper records the shell-tracked child PID under the raw-root logs and leaves the per-run PID marker to the Python runner.
- `H2` is resolved for new runs by adding a required `training_seed` to the run budget, CLI, invocation metadata, profile metrics, and profile provenance. The selected pre-review run remains explicitly downgraded in `pde_execution_summary.md` because it did not record a model/training seed.
- `M1` is resolved by requiring both local baselines, `fno_base` and `unet_base`, in reporting and by rejecting budget-backed `--profiles` overrides that omit required primary profiles outside inspect-only mode.
- Current implementation-review `H1` is resolved by schema-migrating the delivered reusable `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/run_budget.json` to include `training_seed=20260420`, validating that shipped artifact with `load_run_budget()`, and clarifying in `pde_execution_summary.md` that the selected run remains historical unseeded evidence while the corrected budget is valid for reruns.
- Current second-review `H1` is resolved by making `_guard_output_root()` reject stale or live `logs/longer.pid` markers without exit-code evidence, making `validate_fresh_artifacts()` require fresh `logs/longer.run_id`, `logs/longer.started_at_ns`, `logs/longer.pid`, and `logs/longer.exit_code == "0"`, and adding regressions for stale PID markers, missing exit-code evidence, and nonzero exit-code evidence.
- Latest implementation-review `H1` is resolved by making `_guard_output_root()` reject any live `logs/longer.pid` marker even when stale `logs/longer.exit_code` evidence exists, making `_write_start_markers()` remove stale completion evidence before a rerun can be considered fresh, and making `validate_fresh_artifacts()` reject invocation metadata whose parsed `output_root` does not resolve to the validated selected root.

## Implementation Review Fix Documents Read

- Consumed design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- Current plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-swe-longer-execution/execution_plan.md`
- Consumed execution report: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/implementation-phase/execution_report.md`
- Consumed implementation review report: `.artifacts/review/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution-implementation-review.md`
- `docs/index.md`
- `docs/findings.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `state/NEURIPS-HYBRID-RESNET-2026/items/phase-2-pdebench-swe-longer-execution/tranche-context.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_swe_smoke_gate.md`
- `.artifacts/review/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate-implementation-review.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/development/INVOCATION_LOGGING_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/studies/index.md`
- `scripts/studies/pdebench_swe/longer.py`
- `scripts/studies/pdebench_swe/run_config.py`
- `tests/studies/test_pdebench_swe_longer_cli.py`
- `tests/studies/test_pdebench_swe_run_config.py`

## Plan Review Revision Documents Read

- Consumed design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- Consumed roadmap: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- Consumed tranche context: `state/NEURIPS-HYBRID-RESNET-2026/items/phase-2-pdebench-swe-longer-execution/tranche-context.md`
- Consumed plan pointer: `state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-2-pdebench-swe-longer-execution/plan-phase/plan_path.txt`
- Consumed plan review report: `.artifacts/review/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution-plan-review.json`
- Current plan target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-swe-longer-execution/execution_plan.md`
- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/workflows/pytorch.md`
- `docs/development/INVOCATION_LOGGING_GUIDE.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_swe_smoke_gate.md`
- `docs/studies/index.md`
- `docs/plans/templates/implementation_plan.md`

## Implementation Architecture

This tranche needs an Implementation Architecture section because it crosses external data/license provenance, HDF5 IO, deterministic split ownership, run-budget decisions, model profile definitions, long-running command control, metrics, ignored machine artifacts, durable tracked docs, and the roadmap pivot gate. Future CDI Phase 3 work depends on the final proceed/pivot/block decision, so the work must not be collapsed into one broad validator or one broad run script.

### Material Decisions And Missing Decisions

- The approved smoke gate already established a study-specific supervised SWE path that reuses Hybrid ResNet architectural components. Continue that path and label it as Hybrid ResNet on SWE one-step prediction. Do not route SWE through CDI physics loss modules unless a later approved design changes the data/physics contract.
- The roadmap says Hybrid ResNet must be competitive but does not define a numeric competitiveness threshold. This plan pins an operational compute gate, not a paper claim threshold: run ablations only if Hybrid ResNet has finite test `err_nRMSE` and is no worse than `10%` relative above the best local baseline on the shared test subset. If this threshold appears scientifically misleading after results are available, write `block for human decision` rather than inventing a stronger claim.
- Exact long-run budgets are constrained by a required disk/GPU preflight. The default target budget is pinned below; if the preflight fails, write `run_budget.json` with the lower fallback budget before launching any long run and explain the downgrade in `pde_execution_summary.md`.

### Unit 1: Preflight, License, Package, and Budget Lock

- **Owns:** disk/GPU preflight, active-output-root check, PDEBench/DaRUS license and access note, package/module provenance including `neuralop` import source, and the exact budget record used by long runs.
- **Proposed files:**
  - Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/preflight/disk_gpu.json`
  - Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/preflight/license_access.md`
  - Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/preflight/package_provenance.json`
  - Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/run_budget.json`
- **Stable interfaces/artifacts:**
  - Target output root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/`
  - Target budget, if preflight passes: `epochs=15`, `batch_size=16`, `learning_rate=1e-3`, `max_train_trajectories=800`, `max_val_trajectories=100`, `max_test_trajectories=100`, `max_pairs_per_trajectory=10`, `normalization_max_samples=8000`, `eval_splits=val,test`, `num_workers=2`, `device=cuda`, `training_seed=20260420`.
  - Fallback budget, only if target preflight fails: `epochs=10`, `batch_size=8`, `max_train_trajectories=400`, `max_val_trajectories=50`, `max_test_trajectories=50`, `max_pairs_per_trajectory=10`, `normalization_max_samples=4000`, `eval_splits=val,test`, `num_workers=0`, `device=cuda`, `training_seed=20260420`.
- **Must not own:** HDF5 split writing, model execution, metric formulas, or summary decision prose.
- **Dependency direction:** all other units consume the locked budget and provenance from this unit.
- **Compatibility boundary:** do not launch training if disk free space is below `15 GiB`, if the SWE file checksum does not match the smoke gate, or if another live PID/lock is already writing the selected output root.
- **Focused tests/checks:** JSON parse for `run_budget.json`, package provenance includes `torch`, `h5py`, and either a `neuralop` package version or import `__file__`/distribution note; license file exists before summary.

### Unit 2: Full Split, Run Subset, and Normalization Contract

- **Owns:** preserving the full 1000-trajectory deterministic split, generating a budget-capped run subset manifest, lazy HDF5 loading, train-only normalization, and clear sample-vs-batch semantics.
- **Proposed files:**
  - Modify: `scripts/studies/pdebench_swe/splits.py`
  - Modify: `scripts/studies/pdebench_swe/data.py`
  - Modify: `scripts/studies/pdebench_swe/metrics.py`
  - Test: `tests/studies/test_pdebench_swe_splits_data.py`
  - Test: `tests/studies/test_pdebench_swe_metrics.py`
- **Stable interfaces/artifacts:**
  - `split_manifest_full.json`: full 80/10/10 trajectory IDs for all 1000 official trajectories, with uncapped one-step pair counts.
  - `split_manifest_run.json`: exact capped trajectory IDs and pair counts used by the locked budget.
  - `normalization_stats.json`: records `source=train_split_inputs_only`, `limit_kind=samples`, `normalization_max_samples`, `num_samples`, and `num_values_per_channel`.
  - CLI fields: `--normalization-max-samples`, not `--max-train-batches`, for normalization.
- **Must not own:** model profile definitions, CUDA memory reporting, license terms, or final pivot decision.
- **Dependency direction:** consumes Unit 1 budget and Unit 3 HDF5 metadata, then feeds Units 4-7.
- **Compatibility boundary:** keep grouped `*/data` handling and lazy HDF5 access from the smoke gate. Do not materialize the full 6.2 GiB dataset into memory.
- **Focused tests:** full split has 800/100/100 IDs with seed `20260420`; run subset is a prefix/subset of the full split according to the locked budget; pair counts are correct; normalization sample cap is not named as batches; dataset remains lazy before first item access.

### Unit 3: Dataset Manifest and Source Contract Reuse

- **Owns:** reusing the smoke manifest logic for the official SWE file while adding the longer-run license/access note and freshness fields.
- **Proposed files:**
  - Modify: `scripts/studies/pdebench_swe/manifest.py`
  - Test: `tests/studies/test_pdebench_swe_manifest.py`
- **Stable interfaces/artifacts:**
  - `dataset_manifest.json`
  - `hdf5_metadata.json`
  - selected dataset remains `*/data`, `path_pattern={trajectory_id:04d}/data`, axis order `NTHWC`.
- **Must not own:** split caps, model training, metric computation, or result collation.
- **Dependency direction:** consumes Unit 1 license/access note and official data path; feeds Units 2 and 6.
- **Compatibility boundary:** selected file identity must match smoke-gate SHA256 before any longer result is promoted.
- **Focused tests:** checksum/size/mtime manifest still works; grouped layout selection remains stable; license/access note is passed through or linked from the manifest.

### Unit 4: Model Profiles and Ablation Boundaries

- **Owns:** model profile registry for primary models and focused Hybrid ResNet ablations.
- **Proposed files:**
  - Modify: `scripts/studies/pdebench_swe/models.py`
  - Create: `scripts/studies/pdebench_swe/run_config.py`
  - Test: `tests/studies/test_pdebench_swe_models.py`
  - Test: `tests/studies/test_pdebench_swe_run_config.py`
- **Stable interfaces:**
  - `profile_id` output directories under `runs/<profile_id>/`.
  - Built-in primary profiles:
    - `hybrid_resnet_base`: `base_model=hybrid_resnet`, `hidden_channels=16`, `fno_modes=8`, `fno_blocks=4`, `hybrid_downsample_steps=1`, `hybrid_resnet_blocks=2`.
    - `fno_base`: `base_model=fno`, `hidden_channels=16`, `fno_modes=8`, `fno_blocks=4`.
    - `unet_base`: `base_model=unet`, `hidden_channels=16`.
  - Built-in ablation profiles, only run after primary viability:
    - `hybrid_resnet_spectral_reduced`: same as base but `fno_modes=2`.
    - `hybrid_resnet_local_reduced`: same as base but `hybrid_resnet_blocks=1`.
  - Forward contract remains `model(x: FloatTensor[B,C,H,W]) -> FloatTensor[B,C,H,W]`.
- **Must not own:** training loop, split policy, or benchmark summary.
- **Dependency direction:** consumes Unit 2 channel/spatial metadata; feeds Unit 5 execution.
- **Compatibility boundary:** do not modify `ptycho_torch/generators/hybrid_resnet.py`, `ptycho_torch/generators/fno.py`, generator registry defaults, or CDI runner configs for this tranche.
- **Focused tests:** every built-in profile validates, builds on CPU with tiny inputs, preserves output shape, runs one backward pass, records parameter count and profile config, and produces a controlled blocker if FNO dependencies fail.

### Unit 5: Longer Runner, Locking, Memory, and Freshness

- **Owns:** the longer CLI, long-run lock/freshness guard, per-profile train/eval loop, per-profile CUDA peak reset or explicit process-level label, per-profile metrics/blockers/provenance, invocation artifacts, and PID/run-id/start-marker validation.
- **Proposed files:**
  - Create: `scripts/studies/pdebench_swe/longer.py`
  - Create: `scripts/studies/run_pdebench_swe_longer.py`
  - Modify only if needed for shared helpers: `scripts/studies/pdebench_swe/smoke.py`
  - Test: `tests/studies/test_pdebench_swe_longer_cli.py`
  - Test: `tests/studies/test_pdebench_swe_smoke_cli.py`
- **Stable interfaces/artifacts:**
  - CLI: `--data-file`, `--output-root`, `--profiles`, `--epochs`, `--batch-size`, `--learning-rate`, `--normalization-max-samples`, `--eval-splits`, `--training-seed`, `--run-id`, `--device`, `--num-workers`, `--allow-existing-output-root`.
  - `runs/<profile_id>/metrics.json`
  - `runs/<profile_id>/provenance.json`
  - `runs/<profile_id>/blocker.json`
  - raw-root launch selection markers: `logs/selected_longer.run_id`, `logs/selected_longer.run_root`, and `logs/selected_longer.tmux_session`.
  - per-run markers: `logs/longer.started_at_ns`, `logs/longer.pid`, `logs/longer.exit_code`, and `logs/longer.run_id`.
- **Must not own:** pde summary interpretation beyond raw status, docs index entries, or fallback execution.
- **Dependency direction:** consumes Units 1-4 and writes raw evidence for Unit 6.
- **Compatibility boundary:** default behavior refuses non-empty output roots unless explicitly allowed and freshness validation is enabled. Any live `logs/longer.pid` is rejected regardless of `logs/longer.exit_code`; any dead/stale `logs/longer.pid` without exit-code evidence is rejected even when `--allow-existing-output-root` is set. Starting a new run removes stale per-run completion evidence before writing fresh start markers. Long-running launch remains in shell/tmux; Python runner should not hide broad polling loops.
- **Focused tests:** parser accepts planned flags including `--training-seed`; duplicate root rejected; live PID marker roots rejected even with stale exit-code evidence; run ID propagated; CPU synthetic run writes metrics/provenance including training seed; CUDA peak reset path is isolated behind a testable helper; freshness validation rejects stale run IDs, old mtimes, PID mismatches, missing/nonzero exit-code evidence, and invocation `output_root` values that do not resolve to the selected run root.

### Unit 6: Result Collation and Pivot Gate Inputs

- **Owns:** converting per-profile raw artifacts into comparable summary tables and machine-readable gate inputs.
- **Proposed files:**
  - Create: `scripts/studies/pdebench_swe/reporting.py`
  - Test: `tests/studies/test_pdebench_swe_reporting.py`
- **Stable interfaces/artifacts:**
  - `comparison_summary.csv`
  - `comparison_summary.json`
  - Gate input fields: `primary_profiles_complete`, `baseline_profiles_complete`, `hybrid_test_err_nRMSE`, `best_baseline_test_err_nRMSE`, `relative_gap_vs_best_baseline`, `recommended_decision_input`, `ablation_profiles_complete`.
- **Must not own:** running training, modifying metrics, or writing human-facing conclusions without the executor's review.
- **Dependency direction:** consumes Unit 5 profile artifacts; feeds Unit 7 summary.
- **Compatibility boundary:** missing optional ablations can be recorded as skipped only when the primary/baseline viability gate failed or budget was explicitly exhausted. Missing FNO or U-Net baseline is not optional.
- **Focused tests:** collator rejects incomparable run IDs/splits; computes relative gap correctly; treats blocker files separately from metrics; writes deterministic CSV column order.

### Unit 7: Durable Summary and Discoverability

- **Owns:** tracked `pde_execution_summary.md`, optional docs index updates, and final structural verification.
- **Proposed files:**
  - Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md`
  - Modify: `docs/studies/index.md`
  - Modify: `docs/index.md`
- **Required summary sections:** scope, documents and artifacts used, dataset identity, license/access status, HDF5 layout, full split and run subset, normalization and metric contract, run budget, long-run commands, local baseline results, Hybrid ResNet result, ablation results or skip rationale, published-SOTA caveats if any, residual risks, raw artifact links, gate checks, and exactly one decision.
- **Must not own:** paper-facing `/home/ollie/Documents/neurips/` index, CDI planning, fallback execution, or manuscript prose.
- **Dependency direction:** consumes Units 1-6 and closes the selected tranche.
- **Focused tests:** structural check requires one allowed decision, all required artifact links, no `/home/ollie/Documents/neurips/` paths created, index entries present when docs/runbook surfaces are added.

## Compatibility, Migration, and Boundary Notes

- No migrations are planned.
- No production dependency changes are planned. If `neuralop` provenance remains unclear, record package/import source evidence rather than editing dependency metadata.
- Keep the longer SWE path under `scripts/studies/pdebench_swe/`; do not introduce a general PDEBench framework unless a later plan requires it.
- The local metric implementation remains the contract for this tranche. Do not import upstream PDEBench metric code unless license/access terms are resolved and a test proves formula equivalence.
- The SWE supervised model profiles output real state channels, not CDI complex outputs. Do not apply CDI `OUTPUT-COMPLEX-001` conversion behavior here.
- The ignored `.artifacts/` root is the raw evidence surface; tracked docs should link to it and summarize.
- The OpenFWI FlatVel-A fallback is only a decision target in this tranche. Do not execute fallback data download, smoke, or training here.
- `/home/ollie/Documents/neurips/` remains untouched until Roadmap Phase 5.

## Context Priming Before Edits

Re-read these before implementing:

- `docs/index.md`
- `docs/findings.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `state/NEURIPS-HYBRID-RESNET-2026/items/phase-2-pdebench-swe-longer-execution/tranche-context.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_swe_smoke_gate.md`
- `.artifacts/review/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate-implementation-review.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/development/INVOCATION_LOGGING_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/studies/index.md`

Required findings and policies to keep in scope:

- `POLICY-001`: PyTorch is mandatory.
- `ANTIPATTERN-001`: no import-time data loading or hidden side effects.
- `PYTHON-ENV-001`: use PATH `python`.
- `FNO-DEPTH-001` and `FNO-DEPTH-002`: keep spectral SWE profiles bounded on the RTX 3090.
- `STABLE-CRASH-DEPTH-001`: do not overclaim from one seed; this tranche is a Phase 2 gate, not a broad robustness study.
- `FORWARD-SIG-001`: FNO/Hybrid-like SWE profiles use single-input `model(x)` semantics.
- `REPORTING-ARTIFACT-BOUNDARY-001`: optional reporting artifacts must not convert a successful core run into a crash.

## Phases

### Phase A: Scope, Workspace, and Preflight

**Files:**
- Read: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Read: `state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-2-pdebench-swe-longer-execution/plan-phase/plan_path.txt`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/preflight/disk_gpu.json`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/preflight/license_access.md`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/preflight/package_provenance.json`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/run_budget.json`

- [ ] A1: Verify the current checkout and selected plan pointer.

Run:

```bash
pwd
git status --short
sed -n '1p' state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-2-pdebench-swe-longer-execution/plan-phase/plan_path.txt
```

Expected: working directory is `/home/ollie/Documents/PtychoPINN`; unrelated dirty files are noted and not reverted; pointer prints `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-swe-longer-execution/execution_plan.md`.

- [ ] A2: Confirm prior roadmap gates and smoke decision.

Run:

```bash
python - <<'PY'
from pathlib import Path
import json

ledger = json.loads(Path("state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json").read_text(encoding="utf-8"))
required = [
    "phase-0-evidence-inventory",
    "phase-1-pde-benchmark-selection",
    "phase-2-pdebench-swe-primary-smoke-gate",
]
missing = [item for item in required if item not in ledger.get("completed_tranches", [])]
if missing:
    raise SystemExit(f"missing completed prerequisite tranches: {missing}")
smoke = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_swe_smoke_gate.md").read_text(encoding="utf-8")
if "Decision: proceed with longer SWE execution" not in smoke:
    raise SystemExit("smoke gate does not authorize longer SWE execution")
print("prerequisite gates authorize this tranche")
PY
```

Expected: prints `prerequisite gates authorize this tranche`.

- [ ] A3: Create and verify the ignored Phase 2 longer-execution artifact root.

Run:

```bash
RAW_ROOT=".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution"
mkdir -p "${RAW_ROOT}/preflight" "${RAW_ROOT}/logs"
git check-ignore -v "${RAW_ROOT}/probe.json"
```

Expected: `git check-ignore` reports an ignore rule for `.artifacts/`.

- [ ] A4: Record disk/GPU preflight and package provenance before editing long-run behavior.

Run:

```bash
python - <<'PY'
from pathlib import Path
from importlib import metadata
import importlib.util
import json
import shutil
import subprocess
import sys

root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/preflight")
root.mkdir(parents=True, exist_ok=True)

disk = shutil.disk_usage("/")
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"], text=True, capture_output=True, check=False)
payload = {
    "disk_root_total_bytes": disk.total,
    "disk_root_used_bytes": disk.used,
    "disk_root_free_bytes": disk.free,
    "nvidia_smi_returncode": gpu.returncode,
    "nvidia_smi_stdout": gpu.stdout.strip(),
    "nvidia_smi_stderr": gpu.stderr.strip(),
}
(root / "disk_gpu.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

packages = {}
for name in ["torch", "h5py", "neuralop", "neuraloperator", "lightning", "numpy"]:
    try:
        packages[name] = {"version": metadata.version(name)}
    except metadata.PackageNotFoundError:
        packages[name] = {"version": None}
for module_name in ["torch", "h5py", "neuralop"]:
    spec = importlib.util.find_spec(module_name)
    packages.setdefault(module_name, {})["module_origin"] = spec.origin if spec else None
packages["python"] = {"executable": sys.executable, "version": sys.version}
(root / "package_provenance.json").write_text(json.dumps(packages, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("wrote disk/GPU and package preflight")
PY
```

Expected: command exits `0` and writes both JSON files. If free disk is below `15 GiB`, plan the fallback budget and do not launch the target budget.

- [ ] A5: Resolve and record PDEBench/DaRUS license and access status.

Write `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/preflight/license_access.md` with source URLs, access date, repository license, data terms or DaRUS terms, and any caveat. Use primary sources; if terms remain unclear, say so explicitly.

- [ ] A6: Lock the exact run budget in `run_budget.json`.

Use the target budget unless A4 fails disk/GPU feasibility. The file must contain only the chosen budget and downgrade reason if any.

Run:

```bash
python - <<'PY'
from pathlib import Path
import json

root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution")
budget = {
    "schema_version": "pdebench_swe_run_budget_v1",
    "budget_id": "target",
    "data_file": "/home/ollie/Documents/pdebench-data/swe/2D_rdb_NA_NA.h5",
    "epochs": 15,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "max_train_trajectories": 800,
    "max_val_trajectories": 100,
    "max_test_trajectories": 100,
    "max_pairs_per_trajectory": 10,
    "normalization_max_samples": 8000,
    "eval_splits": ["val", "test"],
    "num_workers": 2,
    "device": "cuda",
    "training_seed": 20260420,
    "primary_profiles": ["hybrid_resnet_base", "fno_base", "unet_base"],
    "ablation_profiles": ["hybrid_resnet_spectral_reduced", "hybrid_resnet_local_reduced"],
    "downgrade_reason": None,
}
(root / "run_budget.json").write_text(json.dumps(budget, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("run budget locked")
PY
```

Expected: prints `run budget locked`. If fallback budget is required, edit the JSON values to the fallback values listed in Unit 1 and record a non-null `downgrade_reason`.

### Phase B: Test-First Contract Updates

**Files:**
- Modify: `tests/studies/test_pdebench_swe_manifest.py`
- Modify: `tests/studies/test_pdebench_swe_splits_data.py`
- Modify: `tests/studies/test_pdebench_swe_metrics.py`
- Modify: `tests/studies/test_pdebench_swe_models.py`
- Create: `tests/studies/test_pdebench_swe_run_config.py`
- Create: `tests/studies/test_pdebench_swe_longer_cli.py`
- Create: `tests/studies/test_pdebench_swe_reporting.py`

- [ ] B1: Add failing tests for full split and run subset manifests.

Cover `split_manifest_full.json`, `split_manifest_run.json`, 800/100/100 IDs for the full split, capped IDs as subsets of full IDs, correct capped pair counts, and deterministic seed `20260420`.

- [ ] B2: Add failing tests for normalization sample semantics.

The tests must assert `normalization_stats.json` records `limit_kind="samples"` and does not call the cap `batches`.

- [ ] B3: Add failing tests for model profile definitions.

Assert all five profile IDs validate, build, run a forward pass, preserve `(B,C,H,W)`, and expose the expected config values.

- [ ] B4: Add failing tests for longer CLI and freshness validation.

Cover parser flags, duplicate output-root rejection, lock/run ID propagation, synthetic CPU run for one tiny profile, stale mtime rejection, wrong `run_id` rejection, and PID mismatch rejection.

- [ ] B5: Add failing tests for reporting and gate input collation.

Use synthetic per-profile metrics and blockers to assert deterministic `comparison_summary.csv`, correct relative gap calculation, rejection of mismatched split/run IDs, and required local-baseline completeness.

- [ ] B6: Run the new/changed tests and confirm the expected failures.

Run:

```bash
python -m pytest \
  tests/studies/test_pdebench_swe_manifest.py \
  tests/studies/test_pdebench_swe_splits_data.py \
  tests/studies/test_pdebench_swe_metrics.py \
  tests/studies/test_pdebench_swe_models.py \
  tests/studies/test_pdebench_swe_run_config.py \
  tests/studies/test_pdebench_swe_longer_cli.py \
  tests/studies/test_pdebench_swe_reporting.py \
  -v
```

Expected: failures correspond to missing longer-run implementation, not import crashes or unrelated errors.

### Phase C: Implement Split, Metrics, Profiles, and Manifests

**Files:**
- Modify: `scripts/studies/pdebench_swe/manifest.py`
- Modify: `scripts/studies/pdebench_swe/splits.py`
- Modify: `scripts/studies/pdebench_swe/data.py`
- Modify: `scripts/studies/pdebench_swe/metrics.py`
- Modify: `scripts/studies/pdebench_swe/models.py`
- Create: `scripts/studies/pdebench_swe/run_config.py`

- [ ] C1: Extend split helpers to write separate full and run subset manifests.

Keep the existing smoke behavior compatible; add new explicit functions rather than changing smoke artifacts silently.

- [ ] C2: Update normalization helpers to use sample-count terminology.

Rename or wrap the longer path around `normalization_max_samples`; keep smoke compatibility if existing smoke tests depend on `max_batches`.

- [ ] C3: Add profile registry and budget validation in `run_config.py`.

Implement built-in profiles and budget validation exactly as listed in Unit 4 and Unit 1.

- [ ] C4: Extend model building to consume profile configs.

Keep model construction side-effect free. Do not edit `ptycho_torch` generator modules.

- [ ] C5: Preserve manifest source contracts and license/access links.

The longer manifest should include or link to `preflight/license_access.md` without changing smoke summary semantics.

- [ ] C6: Run focused unit tests for C-phase components.

Run:

```bash
python -m pytest \
  tests/studies/test_pdebench_swe_manifest.py \
  tests/studies/test_pdebench_swe_splits_data.py \
  tests/studies/test_pdebench_swe_metrics.py \
  tests/studies/test_pdebench_swe_models.py \
  tests/studies/test_pdebench_swe_run_config.py \
  -v
```

Expected: all listed tests pass.

### Phase D: Implement Longer Runner and Reporting

**Files:**
- Create: `scripts/studies/pdebench_swe/longer.py`
- Create: `scripts/studies/run_pdebench_swe_longer.py`
- Create: `scripts/studies/pdebench_swe/reporting.py`
- Modify only if necessary for shared helpers: `scripts/studies/pdebench_swe/smoke.py`
- Modify: `tests/studies/test_pdebench_swe_longer_cli.py`
- Modify: `tests/studies/test_pdebench_swe_reporting.py`

- [ ] D1: Implement `run_pdebench_swe_longer.py` as a thin entrypoint.

Follow the existing smoke entrypoint pattern: add the repo root to `sys.path`, import `scripts.studies.pdebench_swe.longer.main`, and return its exit code.

- [ ] D2: Implement the longer CLI and invocation logging.

Use `write_invocation_artifacts()` before expensive work. Persist raw argv, parsed args, run ID, runtime provenance, budget path, and package provenance.

- [ ] D3: Implement output-root lock and freshness markers.

Write `logs/longer.started_at_ns`, `logs/longer.run_id`, and process PID. Refuse a live lock for the same output root. Keep `--allow-existing-output-root` explicit and guarded by freshness validation.

- [ ] D4: Implement per-profile train/eval execution.

Train primary profiles first. Evaluate on both val and test splits under the same normalization. Reset CUDA peak memory per profile with `torch.cuda.reset_peak_memory_stats(device)` when CUDA is used; otherwise label memory as unavailable.

- [ ] D5: Implement primary viability gate before ablations.

If `hybrid_resnet_base`, `fno_base`, and `unet_base` all write finite test metrics and Hybrid ResNet test `err_nRMSE <= 1.10 * min(fno_base, unet_base)`, run both ablation profiles under the same budget. Otherwise skip ablations and record the reason in `comparison_summary.json`.

- [ ] D6: Implement reporting collation.

Write `comparison_summary.csv` and `comparison_summary.json` from raw profile artifacts. Preserve blockers separately from metrics and require matching data file, split manifest, run ID, horizon, and metric units.

- [ ] D7: Run longer-run CLI/reporting tests.

Run:

```bash
python -m pytest \
  tests/studies/test_pdebench_swe_longer_cli.py \
  tests/studies/test_pdebench_swe_reporting.py \
  tests/studies/test_pdebench_swe_smoke_cli.py \
  -v
```

Expected: all listed tests pass, including existing smoke CLI compatibility.

### Phase E: Focused Verification Before Long Runs

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/focused_pytest.log`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/longer_help.log`

- [ ] E1: Run the focused SWE test suite and archive the log.

Run:

```bash
mkdir -p .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs
python -m pytest \
  tests/studies/test_pdebench_swe_manifest.py \
  tests/studies/test_pdebench_swe_splits_data.py \
  tests/studies/test_pdebench_swe_metrics.py \
  tests/studies/test_pdebench_swe_models.py \
  tests/studies/test_pdebench_swe_run_config.py \
  tests/studies/test_pdebench_swe_longer_cli.py \
  tests/studies/test_pdebench_swe_reporting.py \
  tests/studies/test_pdebench_swe_smoke_cli.py \
  -v | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/focused_pytest.log
```

Expected: all tests pass.

- [ ] E2: Verify the longer entrypoint help and parser.

Run:

```bash
python scripts/studies/run_pdebench_swe_longer.py --help \
  | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/longer_help.log
```

Expected: help includes `PDEBench SWE`, `--profiles`, `--normalization-max-samples`, `--eval-splits`, and `--run-id`.

- [ ] E3: Run an inspect-only or tiny CPU synthetic sanity check if supported.

If the longer CLI supports `--inspect-only`, run it against the official HDF5 to write manifests without training. If not, run the synthetic CPU path covered by tests and rely on Phase F for official execution.

### Phase F: Launch Longer Primary Run In tmux

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/runs/<run_id>/`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/longer_<run_id>.stdout.log`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/longer_<run_id>.stderr.log`

- [ ] F1: Choose a unique `RUN_ID`, persist the selected launch identity, and verify no process is writing the same run root.

Run:

```bash
RAW_ROOT=".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution"
RUN_ID="$(date -u +%Y%m%dT%H%M%S.%NZ)"
RUN_ROOT="${RAW_ROOT}/runs/${RUN_ID}"
SESSION="swe_longer_${RUN_ID//[^A-Za-z0-9_]/_}"
mkdir -p "${RAW_ROOT}/logs"
test ! -e "${RUN_ROOT}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "${SESSION}" >&2
  exit 1
fi
printf '%s\n' "${RUN_ID}" > "${RAW_ROOT}/logs/selected_longer.run_id"
printf '%s\n' "${RUN_ROOT}" > "${RAW_ROOT}/logs/selected_longer.run_root"
printf '%s\n' "${SESSION}" > "${RAW_ROOT}/logs/selected_longer.tmux_session"
printf 'selected run_id: %s\nselected run root: %s\nselected tmux session: %s\n' "${RUN_ID}" "${RUN_ROOT}" "${SESSION}"
```

Expected: `test ! -e` succeeds and the three `selected_longer.*` files record the exact run identity that all later launch, monitor, and freshness commands must use.

- [ ] F2: Launch the target budget in tmux with exact PID tracking.

Use the tmux skill for interactive monitoring. Launch with PATH `python` after activating `ptycho311`.

Run:

```bash
RAW_ROOT=".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution"
RUN_ID="$(sed -n '1p' "${RAW_ROOT}/logs/selected_longer.run_id")"
RUN_ROOT="$(sed -n '1p' "${RAW_ROOT}/logs/selected_longer.run_root")"
SESSION="$(sed -n '1p' "${RAW_ROOT}/logs/selected_longer.tmux_session")"
mkdir -p "${RAW_ROOT}/logs"
test "${RUN_ROOT}" = "${RAW_ROOT}/runs/${RUN_ID}"
test ! -e "${RUN_ROOT}"
tmux new-session -d -s "${SESSION}" "bash -lc '
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
START_NS=\$(python - <<\"PY\"
import time
print(time.time_ns())
PY
)
mkdir -p \"${RUN_ROOT}/logs\" \"${RAW_ROOT}/logs\"
printf \"%s\n\" \"${RUN_ID}\" > \"${RUN_ROOT}/logs/longer.run_id\"
printf \"%s\n\" \"\${START_NS}\" > \"${RUN_ROOT}/logs/longer.started_at_ns\"
python scripts/studies/run_pdebench_swe_longer.py \
  --data-file /home/ollie/Documents/pdebench-data/swe/2D_rdb_NA_NA.h5 \
  --output-root \"${RUN_ROOT}\" \
  --dataset-source PDEBench \
  --dataset-source-url https://github.com/pdebench/PDEBench \
  --dataset-darus-id 133021 \
  --license-note-file \"${RAW_ROOT}/preflight/license_access.md\" \
  --run-budget-file \"${RAW_ROOT}/run_budget.json\" \
  --split-seed 20260420 \
  --training-seed 20260420 \
  --train-fraction 0.8 \
  --val-fraction 0.1 \
  --test-fraction 0.1 \
  --profiles hybrid_resnet_base,fno_base,unet_base \
  --run-ablations-if-viable \
  --run-id \"${RUN_ID}\" \
  --device cuda \
  > \"${RAW_ROOT}/logs/longer_${RUN_ID}.stdout.log\" \
  2> \"${RAW_ROOT}/logs/longer_${RUN_ID}.stderr.log\" &
pid=\$!
printf \"%s\n\" \"\$pid\" > \"${RAW_ROOT}/logs/longer_${RUN_ID}.pid\"
set +e
wait \"\$pid\"
rc=\$?
set -e
printf \"%s\n\" \"\$rc\" > \"${RUN_ROOT}/logs/longer.exit_code\"
printf \"%s\n\" \"\$rc\" > \"${RAW_ROOT}/logs/longer_${RUN_ID}.exit_code\"
exit \"\$rc\"
'"
printf 'tmux session: %s\nrun root: %s\n' "${SESSION}" "${RUN_ROOT}"
```

Expected: tmux session starts. The selected run ID/root come only from the persisted `selected_longer.*` files, the wrapper tracks the exact child PID in the raw-root log, the Python runner owns `${RUN_ROOT}/logs/longer.pid`, and the wrapper writes `logs/longer.exit_code` even if the child Python process exits nonzero. Do not start another run writing to the same `RUN_ROOT`.

- [ ] F3: Monitor the exact tmux session and wait for completion.

Run:

```bash
RAW_ROOT=".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution"
SESSION="$(sed -n '1p' "${RAW_ROOT}/logs/selected_longer.tmux_session")"
RUN_ROOT="$(sed -n '1p' "${RAW_ROOT}/logs/selected_longer.run_root")"
tmux list-sessions | rg 'swe_longer_'
tmux capture-pane -pt "${SESSION}" -S -120
```

Expected: the selected tmux session continues or exits. When it exits, `${RUN_ROOT}/logs/longer.exit_code` must exist; it must contain `0` before the run can be treated as complete.

- [ ] F4: Validate official longer-run freshness and required artifacts.

Run:

```bash
python - <<'PY'
from pathlib import Path
import json

raw_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution")
selected_run_id = (raw_root / "logs" / "selected_longer.run_id").read_text(encoding="utf-8").strip()
selected_run_root = Path((raw_root / "logs" / "selected_longer.run_root").read_text(encoding="utf-8").strip())
selected_session = (raw_root / "logs" / "selected_longer.tmux_session").read_text(encoding="utf-8").strip()
expected_run_root = raw_root / "runs" / selected_run_id
if selected_run_root.resolve() != expected_run_root.resolve():
    raise SystemExit(f"selected run root mismatch: {selected_run_root} != {expected_run_root}")
run_root = selected_run_root
if not run_root.exists():
    raise SystemExit(f"selected run root missing: {run_root}")
run_id = (run_root / "logs" / "longer.run_id").read_text(encoding="utf-8").strip()
if run_id != selected_run_id:
    raise SystemExit(f"per-run run_id {run_id!r} does not match selected {selected_run_id!r}")
pid = (run_root / "logs" / "longer.pid").read_text(encoding="utf-8").strip()
if not pid.isdigit():
    raise SystemExit(f"tracked PID is not numeric: {pid!r}")
start_ns = int((run_root / "logs" / "longer.started_at_ns").read_text(encoding="utf-8").strip())
exit_code_path = run_root / "logs" / "longer.exit_code"
if not exit_code_path.exists():
    raise SystemExit(f"missing child exit-code evidence: {exit_code_path}")
exit_code = exit_code_path.read_text(encoding="utf-8").strip()
if exit_code != "0":
    raise SystemExit(f"longer run failed with exit code {exit_code}")
if exit_code_path.stat().st_mtime_ns < start_ns:
    raise SystemExit(f"stale exit-code artifact predates run start: {exit_code_path}")

required_root = [
    "dataset_manifest.json",
    "hdf5_metadata.json",
    "split_manifest_full.json",
    "split_manifest_run.json",
    "normalization_stats.json",
    "comparison_summary.csv",
    "comparison_summary.json",
    "invocation.json",
    "invocation.sh",
]
missing = [name for name in required_root if not (run_root / name).exists()]
if missing:
    raise SystemExit(f"missing root artifacts: {missing}")
for path in [run_root / name for name in required_root]:
    if path.stat().st_mtime_ns < start_ns:
        raise SystemExit(f"stale artifact predates run start: {path}")

for path in [
    raw_root / "logs" / f"longer_{run_id}.stdout.log",
    raw_root / "logs" / f"longer_{run_id}.stderr.log",
    raw_root / "logs" / f"longer_{run_id}.exit_code",
]:
    if not path.exists():
        raise SystemExit(f"missing raw-root launch log: {path}")

invocation = json.loads((run_root / "invocation.json").read_text(encoding="utf-8"))
if str(invocation.get("pid")) != pid:
    raise SystemExit(f"invocation PID {invocation.get('pid')!r} does not match tracked child PID {pid}")
parsed_args = invocation.get("parsed_args", {})
if str(parsed_args.get("run_id")) != run_id:
    raise SystemExit(f"invocation run_id {parsed_args.get('run_id')!r} does not match {run_id}")
if Path(str(parsed_args.get("output_root"))).resolve() != run_root.resolve():
    raise SystemExit(f"invocation output_root does not match selected run root: {parsed_args.get('output_root')!r}")

profiles = ["hybrid_resnet_base", "fno_base", "unet_base"]
for profile in profiles:
    profile_root = run_root / "runs" / profile
    provenance = profile_root / "provenance.json"
    metrics = profile_root / "metrics.json"
    blocker = profile_root / "blocker.json"
    if not provenance.exists():
        raise SystemExit(f"missing provenance for {profile}")
    if not metrics.exists() and not blocker.exists():
        raise SystemExit(f"{profile} wrote neither metrics nor blocker")
    for path in [provenance, metrics if metrics.exists() else blocker]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("run_id")) != run_id:
            raise SystemExit(f"{path} does not match run_id {run_id}")
        payload_pid = payload.get("pid", payload.get("process_pid", payload.get("launcher_pid")))
        if payload_pid is not None and str(payload_pid) != pid:
            raise SystemExit(f"{path} PID {payload_pid!r} does not match tracked child PID {pid}")
        if path.stat().st_mtime_ns < start_ns:
            raise SystemExit(f"stale profile artifact: {path}")
print(f"official longer SWE artifacts are fresh for run_id={run_id} pid={pid} session={selected_session} root={run_root}")
PY
```

Expected: prints `official longer SWE artifacts are fresh...`.

### Phase G: Write Execution Summary and Discoverability Updates

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md`
- Modify: `docs/studies/index.md`
- Modify: `docs/index.md`

- [ ] G1: Decide whether ablations were run or skipped.

Use `comparison_summary.json`. If primary profiles completed and Hybrid ResNet met the 10% operational competitiveness gate, ablations should have run. If not, record ablations as skipped with the exact reason.

- [ ] G2: Draft `pde_execution_summary.md`.

The summary must include the required sections from Unit 7 and must state exactly one decision:

- `Decision: proceed to CDI Phase 3`
- `Decision: pivot to OpenFWI FlatVel-A`
- `Decision: block for human decision`

Use `pivot to OpenFWI FlatVel-A` when SWE data/metric/baseline execution is valid but Hybrid ResNet is clearly noncompetitive under the local baseline gate. Use `block for human decision` when the SWE run cannot satisfy data, license, local-baseline, metric, or provenance requirements and fallback execution is out of this tranche scope.

- [ ] G3: Update `docs/studies/index.md`.

Add or update a `pdebench-swe-longer-execution` entry that names:

- `scripts/studies/run_pdebench_swe_longer.py`
- official SWE file identity
- one-step horizon
- output artifact root
- local model profiles
- boundary: longer Phase 2 execution only, not CDI, OpenFWI, rollout evaluation, or paper-facing artifact assembly

- [ ] G4: Update `docs/index.md`.

Add a concise entry for `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md` because it is a durable roadmap gate document.

- [ ] G5: Run structural summary and discoverability checks.

Run:

```bash
python - <<'PY'
from pathlib import Path
import re

summary = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md")
text = summary.read_text(encoding="utf-8") if summary.exists() else ""
required_terms = [
    "PDEBench",
    "2D_rdb_NA_NA.h5",
    "28f0c33723d70eebb420fc170e94b675c18e032fb697dcef080e114ca9645e3a",
    "split",
    "normalization",
    "err_nRMSE",
    "Hybrid ResNet",
    "FNO",
    "U-Net",
    "provenance",
    "published SOTA",
]
missing = [term for term in required_terms if term.lower() not in text.lower()]
if missing:
    raise SystemExit(f"pde execution summary missing required terms: {missing}")
decisions = re.findall(r"^Decision: (.+)$", text, flags=re.MULTILINE)
allowed = {"proceed to CDI Phase 3", "pivot to OpenFWI FlatVel-A", "block for human decision"}
if len(decisions) != 1 or decisions[0] not in allowed:
    raise SystemExit(f"summary must contain exactly one allowed decision, got {decisions}")

studies = Path("docs/studies/index.md").read_text(encoding="utf-8")
if "pdebench-swe-longer-execution" not in studies or "run_pdebench_swe_longer.py" not in studies:
    raise SystemExit("docs/studies/index.md missing longer SWE runbook entry")
index = Path("docs/index.md").read_text(encoding="utf-8")
if "pde_execution_summary.md" not in index:
    raise SystemExit("docs/index.md missing PDE execution summary entry")
print("PDE execution summary and discoverability checks passed")
PY
```

Expected: prints `PDE execution summary and discoverability checks passed`.

### Phase H: Final Verification and Artifact Hygiene

**Files:**
- Read: `state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-2-pdebench-swe-longer-execution/plan-phase/plan_path.txt`
- Read: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/`
- Read: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md`

- [ ] H1: Rerun all focused tests and archive the final log.

Run:

```bash
python -m pytest \
  tests/studies/test_pdebench_swe_manifest.py \
  tests/studies/test_pdebench_swe_splits_data.py \
  tests/studies/test_pdebench_swe_metrics.py \
  tests/studies/test_pdebench_swe_models.py \
  tests/studies/test_pdebench_swe_run_config.py \
  tests/studies/test_pdebench_swe_longer_cli.py \
  tests/studies/test_pdebench_swe_reporting.py \
  tests/studies/test_pdebench_swe_smoke_cli.py \
  -v | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/final_pytest.log
```

Expected: all tests pass.

- [ ] H2: Verify output-contract pointer and plan target.

Run:

```bash
python - <<'PY'
from pathlib import Path

pointer = Path("state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-2-pdebench-swe-longer-execution/plan-phase/plan_path.txt")
value = pointer.read_text(encoding="utf-8").strip()
expected = "docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-swe-longer-execution/execution_plan.md"
if value != expected:
    raise SystemExit(f"plan_path mismatch: {value!r}")
if not Path(value).exists():
    raise SystemExit(f"plan target missing: {value}")
print("plan_path pointer is valid")
PY
```

Expected: prints `plan_path pointer is valid`.

- [ ] H3: Re-verify exact launched run identity and exit-code evidence.

Run:

```bash
python - <<'PY'
from pathlib import Path
import json

raw_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution")
selected_run_id = (raw_root / "logs" / "selected_longer.run_id").read_text(encoding="utf-8").strip()
selected_run_root = Path((raw_root / "logs" / "selected_longer.run_root").read_text(encoding="utf-8").strip())
expected_run_root = raw_root / "runs" / selected_run_id
if selected_run_root.resolve() != expected_run_root.resolve():
    raise SystemExit(f"selected run root mismatch: {selected_run_root} != {expected_run_root}")
run_root = selected_run_root
run_id = (run_root / "logs" / "longer.run_id").read_text(encoding="utf-8").strip()
pid = (run_root / "logs" / "longer.pid").read_text(encoding="utf-8").strip()
start_ns = int((run_root / "logs" / "longer.started_at_ns").read_text(encoding="utf-8").strip())
exit_code_path = run_root / "logs" / "longer.exit_code"
exit_code = exit_code_path.read_text(encoding="utf-8").strip()
if run_id != selected_run_id:
    raise SystemExit(f"run_id mismatch: {run_id} != {selected_run_id}")
if exit_code != "0":
    raise SystemExit(f"selected longer run did not exit 0: {exit_code}")
if exit_code_path.stat().st_mtime_ns < start_ns:
    raise SystemExit("exit-code evidence predates selected run start marker")
invocation = json.loads((run_root / "invocation.json").read_text(encoding="utf-8"))
if str(invocation.get("pid")) != pid:
    raise SystemExit(f"invocation PID {invocation.get('pid')!r} does not match tracked PID {pid}")
if Path(str(invocation.get("parsed_args", {}).get("output_root"))).resolve() != run_root.resolve():
    raise SystemExit("invocation output root does not match selected run root")
for name in ["comparison_summary.json", "comparison_summary.csv", "dataset_manifest.json", "split_manifest_run.json"]:
    path = run_root / name
    if not path.exists():
        raise SystemExit(f"missing required final artifact: {path}")
    if path.stat().st_mtime_ns < start_ns:
        raise SystemExit(f"stale final artifact predates selected run start: {path}")
print(f"selected longer run verified: run_id={run_id} pid={pid} root={run_root}")
PY
```

Expected: prints `selected longer run verified...`. This check must not inspect `sorted(raw_root / "runs")`; it only trusts the persisted `selected_longer.*` files from Phase F.

- [ ] H4: Verify no paper-facing artifacts or out-of-scope files were created.

Run:

```bash
python - <<'PY'
from pathlib import Path

neurips_root = Path("/home/ollie/Documents/neurips")
if neurips_root.exists():
    # This tranche must not create or update this root. Human-existing files may exist;
    # inspect git/artifact notes manually if mtimes changed during this run.
    print(f"paper-facing root exists; verify no files were created by this tranche: {neurips_root}")
else:
    print("paper-facing root absent")
for forbidden in ["ptycho/model.py", "ptycho/diffsim.py", "ptycho/tf_helper.py"]:
    print(f"forbidden stable module should be unchanged by this tranche: {forbidden}")
PY
git status --short
```

Expected: no `/home/ollie/Documents/neurips/` artifacts were created by this tranche; stable core physics/model modules are not touched by this work; unrelated pre-existing dirty files remain unrelated.

- [ ] H5: Record final execution report for the workflow materialized path.

Write the implementation execution report at the workflow-provided path for this tranche, listing tests run, logs, long-run root, summary decision, and any residual risks.

## Verification Commands

Run these before claiming the tranche is complete:

```bash
python -m pytest \
  tests/studies/test_pdebench_swe_manifest.py \
  tests/studies/test_pdebench_swe_splits_data.py \
  tests/studies/test_pdebench_swe_metrics.py \
  tests/studies/test_pdebench_swe_models.py \
  tests/studies/test_pdebench_swe_run_config.py \
  tests/studies/test_pdebench_swe_longer_cli.py \
  tests/studies/test_pdebench_swe_reporting.py \
  tests/studies/test_pdebench_swe_smoke_cli.py \
  -v
```

```bash
python - <<'PY'
from pathlib import Path
import re

summary = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md")
text = summary.read_text(encoding="utf-8")
decisions = re.findall(r"^Decision: (.+)$", text, flags=re.MULTILINE)
allowed = {"proceed to CDI Phase 3", "pivot to OpenFWI FlatVel-A", "block for human decision"}
if len(decisions) != 1 or decisions[0] not in allowed:
    raise SystemExit(f"summary must contain exactly one allowed decision, got {decisions}")
print(f"summary decision: {decisions[0]}")
PY
```

```bash
python - <<'PY'
from pathlib import Path

pointer = Path("state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-2-pdebench-swe-longer-execution/plan-phase/plan_path.txt")
value = pointer.read_text(encoding="utf-8").strip()
expected = "docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-swe-longer-execution/execution_plan.md"
if value != expected:
    raise SystemExit(f"plan_path mismatch: {value!r}")
if not Path(value).exists():
    raise SystemExit(f"plan target missing: {value}")
print("plan_path pointer is valid")
PY
```

```bash
python - <<'PY'
from pathlib import Path
import json

raw_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution")
selected_run_id = (raw_root / "logs" / "selected_longer.run_id").read_text(encoding="utf-8").strip()
run_root = Path((raw_root / "logs" / "selected_longer.run_root").read_text(encoding="utf-8").strip())
expected_run_root = raw_root / "runs" / selected_run_id
if run_root.resolve() != expected_run_root.resolve():
    raise SystemExit(f"selected run root mismatch: {run_root} != {expected_run_root}")
pid = (run_root / "logs" / "longer.pid").read_text(encoding="utf-8").strip()
start_ns = int((run_root / "logs" / "longer.started_at_ns").read_text(encoding="utf-8").strip())
exit_code_path = run_root / "logs" / "longer.exit_code"
exit_code = exit_code_path.read_text(encoding="utf-8").strip()
if exit_code != "0":
    raise SystemExit(f"selected longer run did not exit 0: {exit_code}")
if exit_code_path.stat().st_mtime_ns < start_ns:
    raise SystemExit("exit-code evidence predates selected run start marker")
invocation = json.loads((run_root / "invocation.json").read_text(encoding="utf-8"))
if str(invocation.get("pid")) != pid:
    raise SystemExit(f"invocation PID {invocation.get('pid')!r} does not match tracked PID {pid}")
for name in ["comparison_summary.json", "comparison_summary.csv", "dataset_manifest.json", "split_manifest_run.json"]:
    path = run_root / name
    if not path.exists():
        raise SystemExit(f"missing required final artifact: {path}")
    if path.stat().st_mtime_ns < start_ns:
        raise SystemExit(f"stale final artifact predates selected run start: {path}")
print(f"selected longer run verified: run_id={selected_run_id} pid={pid} root={run_root}")
PY
```

## Completion Criteria

- [ ] The official SWE data identity, grouped HDF5 layout, license/access status, full split, run subset, normalization policy, and metric contract are recorded.
- [ ] Longer Hybrid ResNet, FNO, and U-Net primary profiles either write comparable finite metrics or explicit blockers that force `pivot` or `block`.
- [ ] At least two local baselines are run successfully, or `pde_execution_summary.md` records why the local-baseline requirement failed and chooses `pivot` or `block`.
- [ ] CUDA peak memory is per-profile or clearly labeled otherwise; normalization limits are in samples, not mislabeled batches.
- [ ] Focused ablations are run only after the primary viability gate, or skipped with explicit rationale.
- [ ] `comparison_summary.csv` and `comparison_summary.json` collate comparable metrics under one run/split/metric contract.
- [ ] Long-run completion evidence is bound to the persisted selected run root, with `logs/longer.exit_code=0`, tracked child PID, matching invocation/provenance run ID, and freshness checks that do not sort existing run directories.
- [ ] `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md` contains exactly one allowed decision.
- [ ] `docs/studies/index.md` and `docs/index.md` are updated if the longer runbook and durable summary are created.
- [ ] All verification commands pass and logs are archived under the ignored tranche artifact root.
- [ ] No CDI Phase 3 work, Phase 4 scaling, Phase 5 paper-facing artifact assembly, OpenFWI execution, stable module edit, or worktree creation occurs.

## Artifacts Index

- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-swe-longer-execution/execution_plan.md`
- Durable Phase 2 summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md`
- Ignored machine artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/`
- Official data file: `/home/ollie/Documents/pdebench-data/swe/2D_rdb_NA_NA.h5`
- Phase 1 selection: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`
- Smoke gate summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_swe_smoke_gate.md`
- Prior smoke implementation review: `.artifacts/review/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate-implementation-review.md`
