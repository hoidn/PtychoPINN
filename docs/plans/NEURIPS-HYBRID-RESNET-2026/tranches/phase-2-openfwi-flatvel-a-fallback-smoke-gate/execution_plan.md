# Phase 2 OpenFWI FlatVel-A Fallback Smoke Gate Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove whether the selected OpenFWI FlatVel-A fallback can proceed to a later benchmark-performance run by validating smoke-shard access, data contracts, MAE/RMSE/SSIM metric plumbing, and tiny Hybrid ResNet-compatible plus local-baseline supervised inversion execution. This smoke gate is readiness evidence only, not a model-performance assessment.

**Architecture:** Keep this as the Roadmap Phase 2 fallback smoke/data-access gate after the approved PDEBench SWE pivot, not full OpenFWI execution, not a benchmark-performance tranche, and not CDI polish. Add a narrow study harness under `scripts/studies/openfwi_flatvel_a/` that owns shard manifests, shape validation, deterministic smoke splits, normalization, MAE/RMSE/SSIM metrics, tiny model adapters, command/provenance IO, and a durable proceed/block/reject summary whose decision is limited to readiness and operational viability. Bulky data, logs, checkpoints, and machine-readable evidence stay under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/` or another ignored/external root named in the summary.

**Tech Stack:** Python 3.11 via PATH `python`, `ptycho311` for long-running tmux launches, NumPy `.npy` shards, PyTorch, existing `ptycho_torch` Hybrid ResNet components, `scikit-image` SSIM, Markdown/JSON/CSV provenance artifacts, optional external OpenFWI repository checkout for official InversionNet compatibility only.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Title: Phase 2 OpenFWI FlatVel-A Fallback Smoke Gate
- Status: pending
- Spec/Source:
  - Design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
  - Roadmap: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
  - Tranche context: `state/NEURIPS-HYBRID-RESNET-2026/items/phase-2-openfwi-flatvel-a-fallback-smoke-gate/tranche-context.md`
  - Phase 1 selection: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`
  - OpenFWI source note: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/source_notes/openfwi_2d_acoustic_fwi.md`
  - SWE pivot summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md`
- Plan path: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-openfwi-flatvel-a-fallback-smoke-gate/execution_plan.md`

## Compliance Matrix

- [ ] **Roadmap Phase Order:** Execute only the selected Roadmap Phase 2 fallback smoke/data-access gate. Do not start Phase 3 CDI anchor regeneration, Phase 4 `256x256` CDI scaling, or Phase 5 paper-facing artifact assembly.
- [ ] **Pivot Binding:** Treat `Decision: pivot to OpenFWI FlatVel-A` in `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md` as active. Do not rerun or rescue PDEBench SWE in this tranche.
- [ ] **Fallback Benchmark Pin:** Use OpenFWI FlatVel-A only. Do not switch to CurveVel, Fault, Style, Kimberlina, PDEArena, PDEBench SWE, or another PDE benchmark without a later approved selection update.
- [ ] **Smoke-Shard Scope:** Start with `data1.npy`/`model1.npy` for train smoke and `data49.npy`/`model49.npy` for validation/test smoke when shard-level access exists. Do not download or train on the full 43 GB FlatVel-A dataset unless limited to a bounded preflight that does not replace a later longer-execution tranche.
- [ ] **Shape Contract:** Validate seismic shard shape `(500, 5, 1000, 70)` and velocity shard shape `(500, 1, 70, 70)`, or document the official schema difference before any training.
- [ ] **Shared Contract Across Models:** Hybrid ResNet-compatible and baseline smoke runs must share the same shard manifest, deterministic sample caps, seed `20260420`, normalization, target preprocessing, metrics, and evaluation budget.
- [ ] **Smoke Evidence Boundary:** Treat all smoke metrics as sanity/provenance artifacts only. Do not compute or publish model-ranking summary fields, do not call Hybrid ResNet competitive/noncompetitive from this gate, and do not reject the fallback for performance from smoke metrics.
- [ ] **Baseline Requirement:** Attempt official InversionNet compatibility first if practical. If official compatibility is blocked, run at least one local baseline, preferably `unet_smoke`, under the same smoke contract and record official OpenFWI rows only as protocol-caveated published context.
- [ ] **Metrics:** Report MAE as primary and RMSE/SSIM as secondary. Metrics must be computed on denormalized velocity maps when target normalization is used.
- [ ] **External Code Boundary:** Do not vendor OpenFWI source or add unscoped production dependencies. If official code is needed, use an external checkout path supplied by CLI and record its URL, commit, license, and compatibility status.
- [ ] **Artifact Hygiene:** Store OpenFWI data outside git. Store bulky logs, raw metrics, checkpoints, and support JSON/CSV under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/` or a documented ignored/external root.
- [ ] **Durable Summary:** Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/openfwi_flatvel_a_fallback_smoke_gate.md` with exactly one decision: `proceed to OpenFWI longer execution`, `block for storage/data/human decision`, or `reject fallback as nonviable`.
- [ ] **Discoverability:** Update `docs/studies/index.md` and `docs/index.md` if this tranche creates a durable runbook, tracked adapter, or tracked summary future workers must discover.
- [ ] **Stable Modules:** Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- [ ] **No Worktrees:** Use the current checkout only.
- [ ] **Interpreter Policy:** Use PATH `python` in commands and subprocesses. Do not introduce repository-specific interpreter wrappers.
- [ ] **Long-Run Guardrail:** Launch nontrivial smoke commands in tmux with `ptycho311`, persist selected run identity before launch, track the exact child PID with `cmd ... & pid=$!; wait "$pid"`, write the child exit code, and do not launch a duplicate run writing to the same `--output-root`.
- [ ] **Freshness Gate:** Treat a smoke run as complete only when the tracked PID exits `0` and required artifacts exist, match the selected run ID/output root, and have mtimes newer than the recorded start marker.
- [ ] **Dirty Worktree Safety:** Preserve unrelated dirty files. Only touch files named by this plan unless a focused blocker requires a narrow companion file.

## Spec Alignment

- **Normative roadmap phase:** Phase 2 - Deep PDE Benchmark Execution.
- **Covered slice:** fallback smoke gate for OpenFWI FlatVel-A after the selected PDEBench SWE primary produced a pivot decision.
- **Required durable output:** `docs/plans/NEURIPS-HYBRID-RESNET-2026/openfwi_flatvel_a_fallback_smoke_gate.md`.
- **Required ignored support root:** `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/`.
- **Required support artifacts:**
  - `preflight/disk_gpu.json`
  - `preflight/package_provenance.json`
  - `preflight/openfwi_source_access.md`
  - `data_manifest.json`
  - `shard_shapes.json`
  - `split_manifest.json`
  - `normalization_stats.json`
  - `official_inversionnet_compatibility.json` or `official_inversionnet_blocker.json`
  - `runs/hybrid_resnet_smoke/metrics.json` or `runs/hybrid_resnet_smoke/blocker.json`
  - `runs/<baseline_profile>/metrics.json` or `runs/<baseline_profile>/blocker.json`
  - `runs/*/provenance.json`
  - `comparison_summary.csv`
  - `comparison_summary.json`
  - root `invocation.json` and `invocation.sh`
  - launch markers under `logs/`: `selected_smoke.run_id`, `selected_smoke.run_root`, `selected_smoke.tmux_session`
  - per-run markers under `runs/<run_id>/logs/`: `smoke.run_id`, `smoke.started_at_ns`, `smoke.pid`, `smoke.exit_code`
- **Explicit non-goals:** no CDI Phase 3 regeneration or packaging, no CDI baselines/ablations, no classical CDI/PyNX/HIO/ER work, no Phase 4 `256x256` scaling, no `/home/ollie/Documents/neurips/` evidence maps/tables/figure manifests/manuscript prose, no full OpenFWI FlatVel-A training, no OpenFWI family switch, no PDEBench SWE rerun, no stable core physics/model edits, and no worktree creation.

## Documents Read For This Plan Draft

- User-provided AGENTS/CLAUDE instructions for `/home/ollie/Documents/PtychoPINN`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `state/NEURIPS-HYBRID-RESNET-2026/items/phase-2-openfwi-flatvel-a-fallback-smoke-gate/tranche-context.md`
- `state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-2-openfwi-flatvel-a-fallback-smoke-gate/plan-phase/plan_path.txt`
- `docs/index.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/plans/templates/implementation_plan.md`
- `docs/templates/` listing
- `docs/findings.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/studies/index.md`
- `docs/workflows/pytorch.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/development/INVOCATION_LOGGING_GUIDE.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/source_notes/openfwi_2d_acoustic_fwi.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-swe-primary-smoke-gate/execution_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-swe-longer-execution/execution_plan.md`
- `scripts/studies/pdebench_swe/manifest.py`
- `scripts/studies/pdebench_swe/data.py`
- `scripts/studies/pdebench_swe/metrics.py`
- `scripts/studies/pdebench_swe/models.py`
- `scripts/studies/pdebench_swe/run_config.py`
- `scripts/studies/pdebench_swe/longer.py`
- `scripts/studies/pdebench_swe/reporting.py`
- `tests/studies/test_pdebench_swe_longer_cli.py`
- `tests/studies/test_studies_index_entries.py`
- `pyproject.toml`

## Implementation Architecture

This tranche needs an Implementation Architecture section because it crosses external data/source access, license provenance, large ignored data roots, deterministic shard/split manifests, normalization, metric semantics, model-adapter boundaries, optional external official-code compatibility, long-running command control, machine-readable artifacts, tracked documentation, and the roadmap gate that determines whether Phase 3 CDI work can begin. A single implementation unit would make stale-artifact validation, data IO, model behavior, and summary decisions too entangled to verify or review safely.

### Material Decisions And Missing Decisions

- **Pinned decision:** OpenFWI FlatVel-A is the fallback. This plan must not expand to OpenFWI CurveVel, Fault, Style, Kimberlina, PDEArena, or a SWE rerun.
- **Pinned decision:** The smoke gate proves data access, data contract, metric writing, and tiny supervised inversion execution. It does not produce full official OpenFWI evidence, benchmark-performance evidence, or a Hybrid ResNet competitiveness judgment.
- **Pinned smoke preprocessing:** Local Hybrid ResNet-compatible and local U-Net smoke profiles may use a documented deterministic resize of seismic shot-gather inputs from `(5, 1000, 70)` to `(5, 70, 70)` before the 2D image-to-image adapter. This is a smoke adapter boundary, not a claim that the full OpenFWI protocol uses that preprocessing.
- **Official baseline boundary:** Official InversionNet compatibility is preferred, but the plan cannot assume the older official code runs in the current environment. If compatibility fails, write an official-code blocker and run a local baseline substitute under the same smoke shards.
- **Missing decision to call out during implementation if needed:** The exact official or verified shard download/access path may not support individual `data1.npy`/`model1.npy` and `data49.npy`/`model49.npy` retrieval. If shard-level access is unavailable and full 43 GB staging lacks approved storage, the tranche must write `block for storage/data/human decision` rather than downloading the full dataset implicitly.

### Unit 1: Preflight, Source Access, and Data Manifest

- **Owns:** disk/GPU preflight, ignored data-root policy, official or verified source URL/access terms, FlatVel-A shard identity, file size/mtime/checksum where practical, shard-level availability blocker, package provenance, and duplicate-output-root guard.
- **Proposed files:**
  - Create: `scripts/studies/openfwi_flatvel_a/__init__.py`
  - Create: `scripts/studies/openfwi_flatvel_a/manifest.py`
  - Test: `tests/studies/test_openfwi_flatvel_a_manifest.py`
  - Create during execution: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/preflight/disk_gpu.json`
  - Create during execution: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/preflight/openfwi_source_access.md`
  - Create during execution: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/preflight/package_provenance.json`
  - Create during execution: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/data_manifest.json`
- **Stable interfaces/artifacts:**
  - CLI inputs: `--data-root`, `--output-root`, `--source-url`, `--source-access-note`, `--license-note`, `--run-id`, `--allow-existing-output-root`.
  - Required shard names: `data1.npy`, `model1.npy`, `data49.npy`, `model49.npy`.
  - `data_manifest.json` schema fields: `schema_version`, `run_id`, `dataset_family`, `dataset_variant`, `source_url`, `license_note`, `access_note`, `local_data_root`, `redistribution_policy`, `shards`, `created_at_utc`.
- **Must not own:** sample splitting, model preprocessing, training loops, metric formulas, or final decision prose.
- **Dependency direction:** feeds Units 2-7.
- **Compatibility boundary:** data root must be outside git or explicitly ignored. If a shard is missing, write `data_access_blocker.json` and stop before model execution.
- **Focused tests:** file identity with size/mtime/checksum, missing-shard blocker payload, data-root-inside-git warning/error, JSON schema round-trip, package provenance includes `torch`, `numpy`, `scikit-image`, and optional OpenFWI checkout metadata.

### Unit 2: Shape Validation, Split Manifest, and Lazy Dataset

- **Owns:** shard shape validation, train/validation/test smoke split manifest, deterministic sample caps with seed `20260420`, lazy memory-mapped `.npy` loading, target/input normalization stats, and deterministic local smoke preprocessing from raw shot gathers to model tensors.
- **Proposed files:**
  - Create: `scripts/studies/openfwi_flatvel_a/data.py`
  - Test: `tests/studies/test_openfwi_flatvel_a_data.py`
- **Stable interfaces/artifacts:**
  - `shard_shapes.json`
  - `split_manifest.json`
  - `normalization_stats.json`
  - Dataset item contract for local profiles:
    - `input_raw`: shape `(5, 1000, 70)` before preprocessing.
    - `input`: `FloatTensor[5, 70, 70]` after deterministic resize for local 2D adapters.
    - `target`: `FloatTensor[1, 70, 70]`.
    - `sample_id`, `source_shard`, `split`, and `preprocessing`.
  - Default smoke caps: `train_samples=32`, `val_samples=16`, `test_samples=16`, `seed=20260420`.
  - CLI inputs: `--train-samples`, `--val-samples`, `--test-samples`, `--split-seed`, `--input-resize-mode bilinear`.
- **Must not own:** source/license notes, metric formulas, model factories, or summary decision.
- **Dependency direction:** consumes Unit 1 shard identities and feeds Units 3-6.
- **Compatibility boundary:** any official schema difference must be recorded in `shard_shapes.json` and the durable summary before training. Do not silently transpose or squeeze unexpected axes.
- **Focused tests:** synthetic `.npy` shards with expected and unexpected shapes, deterministic split sample IDs, disjoint train/test shard roles, `np.load(..., mmap_mode="r")` path, resize output shape `(5, 70, 70)`, target shape preservation, normalization computed from train split only, no eager load of full shard during construction.

### Unit 3: MAE, RMSE, SSIM, and Result Schema

- **Owns:** metric formulas, denormalization before velocity-map metrics, per-sample and aggregate metric payloads, SSIM data-range policy, CSV/JSON serializability, and published-context caveat fields.
- **Proposed files:**
  - Create: `scripts/studies/openfwi_flatvel_a/metrics.py`
  - Test: `tests/studies/test_openfwi_flatvel_a_metrics.py`
- **Stable interfaces/artifacts:**
  - Metric payload keys: `MAE`, `RMSE`, `SSIM`, `per_sample.MAE`, `per_sample.RMSE`, `per_sample.SSIM`, `num_eval_samples`, `metric_units`, `target_normalization`, `ssim_data_range_policy`.
  - Formula: `MAE = mean(abs(prediction - target))`.
  - Formula: `RMSE = sqrt(mean((prediction - target)^2))`.
  - SSIM: use `skimage.metrics.structural_similarity` on each `(70,70)` denormalized velocity map, then average over samples. Record the data-range policy explicitly.
- **Must not own:** data loading, model forward passes, official-code compatibility, or proceed/block/reject decisions.
- **Dependency direction:** consumes Unit 2 tensors and feeds Units 5-7.
- **Compatibility boundary:** metrics from smoke shards are same-shard smoke metrics only. Published OpenFWI benchmark rows are protocol-caveated unless a later tranche reproduces the full official split and metric protocol.
- **Focused tests:** hand-computed MAE/RMSE, identity SSIM equals 1.0, SSIM handles constant targets with explicit data range, denormalized metric path, JSON serialization with finite floats, metric payload rejects mismatched prediction/target shapes.

### Unit 4: Model Profiles and Official InversionNet Compatibility

- **Owns:** small supervised model profiles, local 2D adapter factories, official InversionNet compatibility probe, profile validation, parameter counts, and controlled blocker payloads.
- **Proposed files:**
  - Create: `scripts/studies/openfwi_flatvel_a/models.py`
  - Create: `scripts/studies/openfwi_flatvel_a/run_config.py`
  - Test: `tests/studies/test_openfwi_flatvel_a_models.py`
  - Test: `tests/studies/test_openfwi_flatvel_a_run_config.py`
- **Stable interfaces:**
  - Required local profile: `hybrid_resnet_smoke`.
  - Required fallback local baseline profile when official InversionNet is blocked: `unet_smoke`.
  - Optional profile if budget allows: `fno_smoke`.
  - Optional official compatibility profile: `official_inversionnet_probe`.
  - Local profile forward contract: `model(x: FloatTensor[B,5,70,70]) -> FloatTensor[B,1,70,70]`.
  - Official probe contract: uses raw OpenFWI input shape or records why official code cannot be imported/executed.
- **Must not own:** shard manifests, split policy, metric formulas, long-run tmux launch, or durable summary writing.
- **Dependency direction:** consumes Unit 2 spatial/channel metadata and feeds Unit 5.
- **Compatibility boundary:** do not modify `ptycho_torch/generators/hybrid_resnet.py`, `ptycho_torch/generators/fno.py`, the generator registry, or CDI Lightning behavior. Local OpenFWI adapters live under `scripts/studies/openfwi_flatvel_a/` only.
- **Focused tests:** each local profile builds on CPU, preserves output shape, supports one backward pass, records parameter count/profile config, fails cleanly for unknown profile, official-checkout probe returns a blocker without an installed external repo.

### Unit 5: Smoke Runner, CLI, Long-Run Guard, and Provenance

- **Owns:** CLI parsing, invocation artifacts, exact run ID propagation, root locking/freshness guard, package/runtime/git provenance, bounded train/eval loop, per-profile metrics/blockers/provenance, runtime and CUDA peak memory capture where available, and same-contract enforcement across profiles.
- **Proposed files:**
  - Create: `scripts/studies/openfwi_flatvel_a/smoke.py`
  - Create: `scripts/studies/run_openfwi_flatvel_a_smoke.py`
  - Test: `tests/studies/test_openfwi_flatvel_a_smoke_cli.py`
- **Stable interfaces/artifacts:**
  - CLI: `--data-root`, `--output-root`, `--source-url`, `--source-access-note`, `--license-note`, `--profiles`, `--official-openfwi-repo`, `--epochs`, `--batch-size`, `--learning-rate`, `--train-samples`, `--val-samples`, `--test-samples`, `--split-seed`, `--device`, `--num-workers`, `--run-id`, `--allow-existing-output-root`, `--inspect-only`.
  - Root artifacts: `invocation.json`, `invocation.sh`, `data_manifest.json`, `shard_shapes.json`, `split_manifest.json`, `normalization_stats.json`.
  - Profile artifacts: `runs/<profile_id>/metrics.json`, `runs/<profile_id>/blocker.json`, `runs/<profile_id>/provenance.json`.
  - Logs: `logs/selected_smoke.run_id`, `logs/selected_smoke.run_root`, `logs/selected_smoke.tmux_session`, `runs/<run_id>/logs/smoke.run_id`, `runs/<run_id>/logs/smoke.started_at_ns`, `runs/<run_id>/logs/smoke.pid`, `runs/<run_id>/logs/smoke.exit_code`.
- **Must not own:** final human-facing decision prose, docs index entries, full OpenFWI training, or paper-facing artifacts.
- **Dependency direction:** consumes Units 1-4 and writes evidence for Units 6-7.
- **Compatibility boundary:** default behavior refuses non-empty output roots unless `--allow-existing-output-root` is passed and freshness validation is enabled. Any live PID marker is rejected. Any PID marker without exit-code evidence is rejected.
- **Focused tests:** CLI help mentions required flags; duplicate-root rejection; prelaunch marker allowance; live/stale PID guards; synthetic CPU smoke run writes invocation, manifest, metrics, provenance, and comparison files; run ID mismatch rejected; nonzero exit-code freshness rejected; official InversionNet blocker still allows local `unet_smoke` baseline path.

### Unit 6: Smoke Contract Collation and Gate Input

- **Owns:** deterministic smoke CSV/JSON, local-baseline completeness checks, same-contract validation across profiles, official InversionNet compatibility status, explicit smoke-only evidence-scope fields, and machine-readable readiness input for the durable summary.
- **Proposed files:**
  - Create: `scripts/studies/openfwi_flatvel_a/reporting.py`
  - Test: `tests/studies/test_openfwi_flatvel_a_reporting.py`
- **Stable interfaces/artifacts:**
  - `comparison_summary.csv`
  - `comparison_summary.json`
  - Gate fields: `evidence_scope`, `metric_interpretation`, `performance_assessment_complete`, `data_access_complete`, `shape_validation_complete`, `hybrid_profile_complete`, `local_baseline_complete`, `official_inversionnet_status`, `recommended_decision_input`, and `profiles`.
- **Must not own:** running models, altering metrics, computing model-ranking summary fields, or making human-facing performance claims without a later benchmark-performance plan.
- **Dependency direction:** consumes Unit 5 profile artifacts and feeds Unit 7.
- **Compatibility boundary:** missing official InversionNet is not fatal if a local baseline completed and the official blocker is recorded. Missing both official and local baseline evidence is fatal for this gate.
- **Focused tests:** collator rejects incomparable run IDs/split manifests/normalization stats, handles blocker files separately from metrics, writes stable CSV column order, emits explicit smoke-only evidence-scope fields, omits `hybrid_MAE`/`best_baseline_MAE`/`relative_gap_vs_best_baseline` summary fields, and emits allowed `recommended_decision_input` values only.

### Unit 7: Durable Smoke Summary and Discoverability

- **Owns:** tracked `openfwi_flatvel_a_fallback_smoke_gate.md`, final decision, concise artifact links, summary structural validation, and documentation index updates when durable runbook/tracked adapter surfaces are added.
- **Proposed files:**
  - Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/openfwi_flatvel_a_fallback_smoke_gate.md`
  - Modify: `docs/studies/index.md`
  - Modify: `docs/index.md`
  - Test: `tests/studies/test_studies_index_entries.py` if index entries are added, or a new focused structural test if this file grows too broad.
- **Required summary sections:** scope, documents and artifacts used, source/access/license status, data-root policy, shard identity and shape validation, split manifest, normalization and preprocessing, metric contract, official InversionNet compatibility, local model results/blockers, runtime/memory/provenance, published-context caveats, gate checks, residual risks, raw artifact links, explicit non-goals confirmed, and exactly one decision.
- **Must not own:** raw logs, full OpenFWI longer execution summary, `/home/ollie/Documents/neurips/` evidence index, CDI plans, or manuscript prose.
- **Dependency direction:** consumes Units 1-6 and closes the selected tranche.
- **Focused tests:** structural check requires one allowed decision, all required artifact links or blocker links, no paper-facing paths created, no CDI/Phase 4 work mentioned as completed, and index entries present if tracked runbook/summary surfaces were added.

## Compatibility, Migration, and Boundary Notes

- No migrations are planned.
- No production dependency changes are planned. `scikit-image` is already a project dependency. If a dependency is unexpectedly unavailable in the environment, record a package blocker instead of editing dependency metadata.
- Do not import OpenFWI code at module import time. Optional official-code compatibility belongs behind CLI arguments and controlled blockers.
- Do not use ptychographic physics loss, `PtychoPINN_Lightning`, or legacy `params.cfg` for this smoke adapter unless a later approved design changes the OpenFWI data/physics contract.
- Do not modify Hybrid ResNet production generator code. Local supervised wrappers can reuse generator components, but they must live under `scripts/studies/openfwi_flatvel_a/`.
- Do not present local resized-input smoke results as full official OpenFWI protocol evidence.
- Do not write OpenFWI data, checkpoints, full logs, or raw metrics into tracked repo paths.
- `/home/ollie/Documents/neurips/` remains untouched until Roadmap Phase 5.

## Context Priming Before Edits

Re-read these before executing implementation tasks:

- `docs/index.md`
- `docs/findings.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `state/NEURIPS-HYBRID-RESNET-2026/items/phase-2-openfwi-flatvel-a-fallback-smoke-gate/tranche-context.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/source_notes/openfwi_2d_acoustic_fwi.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/development/INVOCATION_LOGGING_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/studies/index.md`

Required findings and policies to carry into implementation:

- `POLICY-001`: PyTorch is mandatory.
- `ANTIPATTERN-001`: no import-time data loading or hidden side effects.
- `PYTHON-ENV-001`: use PATH `python`.
- `FORWARD-SIG-001`: FNO/Hybrid-like study models should use single-input `model(x)` semantics.
- `FNO-DEPTH-001` and `FNO-DEPTH-002`: keep spectral smoke models tiny and bounded on the RTX 3090.
- `STABLE-CRASH-DEPTH-001`: do not overinterpret one tiny seed; this tranche proves fallback executability, not final robustness.
- `REPORTING-ARTIFACT-BOUNDARY-001`: optional reporting artifacts must not reclassify a successful core smoke run as failed.

## Phases

### Phase A: Scope, Pointer, and Workspace Preflight

**Files:**
- Read: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Read: `state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-2-openfwi-flatvel-a-fallback-smoke-gate/plan-phase/plan_path.txt`
- Create during execution: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/preflight/`

- [ ] A1: Verify current checkout, dirty state, and plan pointer.

Run:

```bash
pwd
git status --short
sed -n '1p' state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-2-openfwi-flatvel-a-fallback-smoke-gate/plan-phase/plan_path.txt
```

Expected: working directory is `/home/ollie/Documents/PtychoPINN`; unrelated dirty files are noted and not reverted; pointer prints `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-openfwi-flatvel-a-fallback-smoke-gate/execution_plan.md`.

- [ ] A2: Verify prior roadmap gates and pivot state.

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
ledger = json.loads(Path("state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json").read_text())
completed = set(ledger.get("completed_tranches", []))
required = {
    "phase-0-evidence-inventory",
    "phase-1-pde-benchmark-selection",
    "phase-2-pdebench-swe-primary-smoke-gate",
    "phase-2-pdebench-swe-longer-execution",
}
missing = sorted(required - completed)
if missing:
    raise SystemExit(f"missing prerequisite completed tranches: {missing}")
summary = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md").read_text()
if "Decision: pivot to OpenFWI FlatVel-A" not in summary:
    raise SystemExit("SWE longer summary does not record the required OpenFWI pivot decision")
print("OpenFWI fallback smoke gate prerequisites are satisfied")
PY
```

Expected: prints `OpenFWI fallback smoke gate prerequisites are satisfied`.

- [ ] A3: Create and record the ignored raw artifact root, and reject concurrent writers.

Run:

```bash
RAW_ROOT=".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate"
mkdir -p "$RAW_ROOT"/preflight "$RAW_ROOT"/logs
python - <<'PY'
from pathlib import Path
raw = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate")
pid_markers = list(raw.glob("runs/*/logs/smoke.pid"))
live = []
for marker in pid_markers:
    pid = marker.read_text().strip()
    if pid.isdigit() and Path(f"/proc/{pid}").exists():
        live.append((str(marker), pid))
if live:
    raise SystemExit(f"live OpenFWI smoke output root exists: {live}")
print("raw root ready; no live smoke writer detected")
PY
```

Expected: prints `raw root ready; no live smoke writer detected`.

- [ ] A4: Record disk/GPU/package preflight without downloading data.

Run:

```bash
python - <<'PY'
import json, platform, shutil, subprocess, sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/preflight")
root.mkdir(parents=True, exist_ok=True)
usage = shutil.disk_usage(Path.cwd())
packages = {}
for name in ["numpy", "torch", "scikit-image"]:
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
try:
    import torch
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
except Exception as exc:
    cuda_available = False
    gpu_name = None
    packages["torch_import_error"] = str(exc)
payload = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "cwd": str(Path.cwd()),
    "python": sys.version,
    "platform": platform.platform(),
    "disk_bytes": {"total": usage.total, "used": usage.used, "free": usage.free},
    "cuda_available": cuda_available,
    "gpu_name": gpu_name,
    "packages": packages,
}
(root / "disk_gpu.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("wrote disk/GPU/package preflight")
PY
```

Expected: writes `preflight/disk_gpu.json`; it does not download or stage OpenFWI data.

### Phase B: Data Access Manifest and Shape Validator

**Files:**
- Create: `scripts/studies/openfwi_flatvel_a/__init__.py`
- Create: `scripts/studies/openfwi_flatvel_a/manifest.py`
- Test: `tests/studies/test_openfwi_flatvel_a_manifest.py`

- [ ] B1: Write failing manifest tests for shard identity and blockers.

Test cases:

```python
def test_file_identity_records_size_mtime_and_sha256(tmp_path):
    path = tmp_path / "data1.npy"
    path.write_bytes(b"abc")
    payload = file_identity(path)
    assert payload["filename"] == "data1.npy"
    assert payload["size_bytes"] == 3
    assert payload["sha256"]

def test_required_shards_missing_writes_blocker(tmp_path):
    with pytest.raises(OpenFWIManifestBlocker) as exc:
        resolve_required_shards(tmp_path)
    assert exc.value.reason == "missing_required_shards"
    assert "data1.npy" in exc.value.missing

def test_data_root_inside_repo_is_rejected_unless_ignored(tmp_path, monkeypatch):
    repo_root = Path.cwd()
    data_root = repo_root / "tmp" / "openfwi-data"
    data_root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(OpenFWIManifestBlocker):
        validate_data_root_policy(data_root, repo_root=repo_root)
```

- [ ] B2: Implement manifest helpers.

Required functions/classes:

```python
REQUIRED_SHARDS = ("data1.npy", "model1.npy", "data49.npy", "model49.npy")

class OpenFWIManifestBlocker(RuntimeError):
    reason: str
    def to_payload(self, *, run_id: str | None = None) -> dict: ...

def file_identity(path: Path, *, sha256: bool = True) -> dict: ...
def resolve_required_shards(data_root: Path) -> dict[str, Path]: ...
def validate_data_root_policy(data_root: Path, *, repo_root: Path) -> None: ...
def build_data_manifest(... ) -> dict: ...
def write_json(path: Path, payload: dict) -> Path: ...
```

- [ ] B3: Run manifest tests.

Run:

```bash
pytest tests/studies/test_openfwi_flatvel_a_manifest.py -v
```

Expected: tests pass.

- [ ] B4: Stage or identify FlatVel-A smoke shards outside git, or write a controlled data-access blocker.

Run only after the user or environment supplies a candidate root:

```bash
export OPENFWI_FLATVEL_A_ROOT="${OPENFWI_FLATVEL_A_ROOT:?set to external FlatVel-A shard root}"
python - <<'PY'
from pathlib import Path
from scripts.studies.openfwi_flatvel_a.manifest import (
    OpenFWIManifestBlocker,
    build_data_manifest,
    resolve_required_shards,
    validate_data_root_policy,
    write_json,
)

raw = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate")
data_root = Path(__import__("os").environ["OPENFWI_FLATVEL_A_ROOT"]).expanduser().resolve()
try:
    validate_data_root_policy(data_root, repo_root=Path.cwd())
    shards = resolve_required_shards(data_root)
    manifest = build_data_manifest(
        data_root=data_root,
        shards=shards,
        source_url="https://openfwi-lanl.github.io/docs/data.html",
        license_note="OpenFWI datasets: CC BY-NC-SA 4.0; code: BSD-3-Clause",
        access_note="Shard-level FlatVel-A smoke files referenced locally; not redistributed.",
        run_id="preflight",
    )
    write_json(raw / "data_manifest.json", manifest)
    print("OpenFWI smoke shards resolved")
except OpenFWIManifestBlocker as exc:
    write_json(raw / "data_access_blocker.json", exc.to_payload(run_id="preflight"))
    raise SystemExit(f"OpenFWI data access blocker: {exc}")
PY
```

Expected if shards are present: writes `data_manifest.json`. Expected if unavailable: writes `data_access_blocker.json` and stops before training.

### Phase C: Shape, Split, Dataset, and Normalization

**Files:**
- Create: `scripts/studies/openfwi_flatvel_a/data.py`
- Test: `tests/studies/test_openfwi_flatvel_a_data.py`

- [ ] C1: Write failing data tests for shape validation, split identity, lazy loading, and local preprocessing.

Test cases:

```python
def test_validate_expected_flatvel_shapes(tmp_path):
    write_pair(tmp_path, "data1.npy", (4, 5, 1000, 70), "model1.npy", (4, 1, 70, 70))
    payload = inspect_shard_pair(tmp_path / "data1.npy", tmp_path / "model1.npy")
    assert payload["data_shape"][1:] == [5, 1000, 70]
    assert payload["model_shape"][1:] == [1, 70, 70]

def test_split_manifest_uses_train_and_test_shards_with_seed(tmp_path):
    manifest = build_split_manifest(train_count=500, test_count=500, train_samples=32, val_samples=16, test_samples=16, seed=20260420)
    assert manifest["train"]["data_shard"] == "data1.npy"
    assert manifest["val"]["data_shard"] == "data49.npy"
    assert manifest["test"]["data_shard"] == "data49.npy"
    assert manifest["seed"] == 20260420

def test_dataset_returns_local_adapter_tensors(tmp_path):
    dataset = OpenFWIShardDataset(...)
    item = dataset[0]
    assert item["input"].shape == (5, 70, 70)
    assert item["target"].shape == (1, 70, 70)
```

- [ ] C2: Implement shape inspection, split manifest, memory-mapped dataset, and normalization helpers.

Required functions/classes:

```python
EXPECTED_DATA_SHAPE_SUFFIX = (5, 1000, 70)
EXPECTED_MODEL_SHAPE_SUFFIX = (1, 70, 70)

def inspect_shard_pair(data_path: Path, model_path: Path) -> dict: ...
def build_split_manifest(..., seed: int = 20260420) -> dict: ...
def compute_normalization_stats(train_dataset, *, max_samples: int | None = None) -> dict: ...

class OpenFWIShardDataset(torch.utils.data.Dataset):
    # Opens np.load(..., mmap_mode="r") lazily.
    # Returns input resized to (5,70,70) and target (1,70,70) for local profiles.
```

- [ ] C3: Run data tests.

Run:

```bash
pytest tests/studies/test_openfwi_flatvel_a_data.py -v
```

Expected: tests pass.

### Phase D: Metrics and Profile Config

**Files:**
- Create: `scripts/studies/openfwi_flatvel_a/metrics.py`
- Create: `scripts/studies/openfwi_flatvel_a/run_config.py`
- Test: `tests/studies/test_openfwi_flatvel_a_metrics.py`
- Test: `tests/studies/test_openfwi_flatvel_a_run_config.py`

- [ ] D1: Write failing metric tests.

Test cases:

```python
def test_mae_rmse_are_hand_computed():
    pred = torch.tensor([[[[1.0, 3.0]]]])
    target = torch.tensor([[[[2.0, 1.0]]]])
    payload = metric_payload([pred], [target], normalized=False, target_stats=None)
    assert payload["MAE"] == pytest.approx(1.5)
    assert payload["RMSE"] == pytest.approx((2.5) ** 0.5)

def test_identity_ssim_is_one():
    image = torch.ones(1, 1, 70, 70)
    payload = metric_payload([image], [image], normalized=False, target_stats=None)
    assert payload["SSIM"] == pytest.approx(1.0)
```

- [ ] D2: Implement MAE/RMSE/SSIM metric payloads.

Required behavior:

- Denormalize prediction and target before metrics when target stats are supplied.
- Compute MAE/RMSE over all evaluation pixels and samples.
- Compute SSIM per sample using `skimage.metrics.structural_similarity`, then average.
- Record `metric_units`, `target_normalization`, `ssim_data_range_policy`, `num_eval_samples`, and finite floats.

- [ ] D3: Write failing profile-config tests.

Required profile defaults:

```python
PRIMARY_PROFILE_IDS = ["hybrid_resnet_smoke", "unet_smoke"]
OPTIONAL_PROFILE_IDS = ["fno_smoke", "official_inversionnet_probe"]
DEFAULT_RUN_BUDGET = {
    "epochs": 1,
    "batch_size": 4,
    "learning_rate": 1e-3,
    "train_samples": 32,
    "val_samples": 16,
    "test_samples": 16,
    "split_seed": 20260420,
    "device": "cuda",
    "num_workers": 0,
}
```

- [ ] D4: Implement profile and budget validation.

Required functions/classes:

```python
@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    base_model: str
    hidden_channels: int
    fno_modes: int | None = None
    fno_blocks: int | None = None
    hybrid_downsample_steps: int | None = None
    hybrid_resnet_blocks: int | None = None

def get_model_profile(profile_id: str) -> ModelProfile: ...
def parse_profile_ids(value: str | list[str] | None) -> list[str]: ...
def validate_run_budget(payload: dict) -> dict: ...
```

- [ ] D5: Run metric and run-config tests.

Run:

```bash
pytest tests/studies/test_openfwi_flatvel_a_metrics.py tests/studies/test_openfwi_flatvel_a_run_config.py -v
```

Expected: tests pass.

### Phase E: Model Adapters and Official InversionNet Probe

**Files:**
- Create: `scripts/studies/openfwi_flatvel_a/models.py`
- Test: `tests/studies/test_openfwi_flatvel_a_models.py`

- [ ] E1: Write failing model adapter tests.

Test cases:

```python
def test_hybrid_resnet_smoke_builds_and_preserves_target_shape():
    model = build_model("hybrid_resnet_smoke", in_channels=5, out_channels=1, spatial_shape=(70, 70), profile_config={})
    x = torch.randn(2, 5, 70, 70)
    y = model(x)
    assert y.shape == (2, 1, 70, 70)

def test_unet_smoke_runs_backward():
    model = build_model("unet_smoke", in_channels=5, out_channels=1, spatial_shape=(70, 70), profile_config={})
    x = torch.randn(2, 5, 70, 70)
    target = torch.randn(2, 1, 70, 70)
    loss = torch.nn.functional.l1_loss(model(x), target)
    loss.backward()

def test_official_inversionnet_probe_blocks_without_repo(tmp_path):
    blocker = probe_official_inversionnet(tmp_path / "missing")
    assert blocker["status"] == "blocked"
```

- [ ] E2: Implement local model factories.

Required behavior:

- `hybrid_resnet_smoke` uses existing Hybrid ResNet components where practical and stays local to this study harness.
- `unet_smoke` is a compact local 2D U-Net baseline.
- `fno_smoke` is optional and may write a blocker if `neuralop` build/import fails.
- All local profiles accept `(B,5,70,70)` and output `(B,1,70,70)`.

- [ ] E3: Implement official InversionNet compatibility probe.

Required behavior:

- Accept `--official-openfwi-repo <path>` and never import official code unless that path is supplied.
- Record repository path, git commit if available, license path if available, import attempt result, and minimal forward-pass status if available.
- If blocked, write `official_inversionnet_blocker.json` and continue to local baseline execution.

- [ ] E4: Run model tests.

Run:

```bash
pytest tests/studies/test_openfwi_flatvel_a_models.py -v
```

Expected: tests pass.

### Phase F: Smoke Runner and CLI

**Files:**
- Create: `scripts/studies/openfwi_flatvel_a/smoke.py`
- Create: `scripts/studies/run_openfwi_flatvel_a_smoke.py`
- Test: `tests/studies/test_openfwi_flatvel_a_smoke_cli.py`

- [ ] F1: Write failing CLI and runner tests.

Required test coverage:

- `--help` includes `OpenFWI FlatVel-A`, `--data-root`, `--output-root`, `--profiles`, `--run-id`, and `--official-openfwi-repo`.
- Non-empty output root is rejected by default.
- Live PID marker is rejected even with stale exit-code evidence.
- Synthetic CPU smoke run writes invocation, manifests, metrics, provenance, and comparison outputs.
- Missing official InversionNet path writes blocker and still runs `unet_smoke`.
- Freshness validation rejects stale run ID, missing exit code, nonzero exit code, or old mtimes.

- [ ] F2: Implement CLI parser, invocation logging, output-root guard, and smoke runner.

Required behavior:

- Use `scripts.studies.invocation_logging.write_invocation_artifacts`.
- Write invocation artifacts before expensive work.
- Refuse duplicate output roots unless explicitly allowed.
- Seed Python, NumPy, and PyTorch with `split_seed`/`training_seed=20260420`.
- Run bounded train/eval for `hybrid_resnet_smoke` and at least one baseline profile.
- Write `blocker.json` instead of crashing for controlled data/model/official-code blockers.
- Write per-profile metrics in denormalized velocity units.

- [ ] F3: Run smoke CLI unit tests.

Run:

```bash
pytest tests/studies/test_openfwi_flatvel_a_smoke_cli.py -v
```

Expected: tests pass.

- [ ] F4: Run a CPU synthetic smoke test.

Run:

```bash
pytest tests/studies/test_openfwi_flatvel_a_manifest.py \
  tests/studies/test_openfwi_flatvel_a_data.py \
  tests/studies/test_openfwi_flatvel_a_metrics.py \
  tests/studies/test_openfwi_flatvel_a_models.py \
  tests/studies/test_openfwi_flatvel_a_run_config.py \
  tests/studies/test_openfwi_flatvel_a_smoke_cli.py -v
```

Expected: all OpenFWI FlatVel-A unit/synthetic smoke tests pass.

### Phase G: Real Smoke Launch, Metrics, and Comparison

**Files:**
- Create during execution: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/runs/<run_id>/`
- Create during execution: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/comparison_summary.json`
- Create during execution: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/comparison_summary.csv`

- [ ] G1: If shard access was blocked in Phase B, skip model execution and prepare a blocked summary.

Required blocker condition:

- `data_access_blocker.json` exists, or any required shard is missing.
- Do not download full 43 GB FlatVel-A implicitly.
- Phase I decision must be `block for storage/data/human decision`.

- [ ] G2: Launch the real OpenFWI smoke run in tmux with exact PID tracking.

Run when `OPENFWI_FLATVEL_A_ROOT` is set and Phase B/C validations passed:

```bash
RAW_ROOT="$PWD/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate"
RUN_ID="openfwi-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="$RAW_ROOT/runs/$RUN_ID"
TMUX_SESSION="openfwi_flatvel_a_smoke_${RUN_ID//[^A-Za-z0-9_]/_}"
START_NS="$(date +%s%N)"
mkdir -p "$RAW_ROOT/logs" "$RUN_ROOT/logs"
printf '%s\n' "$RUN_ID" > "$RAW_ROOT/logs/selected_smoke.run_id"
printf '%s\n' "$RUN_ROOT" > "$RAW_ROOT/logs/selected_smoke.run_root"
printf '%s\n' "$TMUX_SESSION" > "$RAW_ROOT/logs/selected_smoke.tmux_session"
printf '%s\n' "$RUN_ID" > "$RUN_ROOT/logs/smoke.run_id"
printf '%s\n' "$START_NS" > "$RUN_ROOT/logs/smoke.started_at_ns"
tmux new-session -d -s "$TMUX_SESSION" "bash -lc '
  set +e
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate ptycho311
  cd /home/ollie/Documents/PtychoPINN
  python scripts/studies/run_openfwi_flatvel_a_smoke.py \
    --data-root \"${OPENFWI_FLATVEL_A_ROOT:?set external FlatVel-A root}\" \
    --output-root \"$RUN_ROOT\" \
    --source-url \"https://openfwi-lanl.github.io/docs/data.html\" \
    --source-access-note \"FlatVel-A smoke shards referenced locally; data are not redistributed.\" \
    --license-note \"OpenFWI datasets: CC BY-NC-SA 4.0; code: BSD-3-Clause\" \
    --profiles hybrid_resnet_smoke,unet_smoke \
    --epochs 1 \
    --batch-size 4 \
    --learning-rate 0.001 \
    --train-samples 32 \
    --val-samples 16 \
    --test-samples 16 \
    --split-seed 20260420 \
    --device cuda \
    --num-workers 0 \
    --run-id \"$RUN_ID\" \
    > \"$RAW_ROOT/logs/${RUN_ID}.stdout.log\" \
    2> \"$RAW_ROOT/logs/${RUN_ID}.stderr.log\" &
  pid=\$!
  printf \"%s\n\" \"\$pid\" > \"$RUN_ROOT/logs/smoke.pid\"
  wait \"\$pid\"
  code=\$?
  printf \"%s\n\" \"\$code\" > \"$RUN_ROOT/logs/smoke.exit_code\"
  exit \"\$code\"
'"
printf 'launched %s for run %s\n' "$TMUX_SESSION" "$RUN_ID"
```

Expected: tmux session starts; selected run markers are written before launch.

- [ ] G3: Wait for the exact launched run to finish and verify exit code.

Run:

```bash
RAW_ROOT="$PWD/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate"
RUN_ID="$(cat "$RAW_ROOT/logs/selected_smoke.run_id")"
RUN_ROOT="$(cat "$RAW_ROOT/logs/selected_smoke.run_root")"
TMUX_SESSION="$(cat "$RAW_ROOT/logs/selected_smoke.tmux_session")"
tmux wait-for -L "wait_${TMUX_SESSION}" 2>/dev/null || true
while tmux has-session -t "$TMUX_SESSION" 2>/dev/null; do sleep 30; done
test -f "$RUN_ROOT/logs/smoke.exit_code"
test "$(cat "$RUN_ROOT/logs/smoke.exit_code")" = "0"
echo "OpenFWI smoke run exited 0: $RUN_ID"
```

Expected: prints `OpenFWI smoke run exited 0: <run_id>`. If nonzero, preserve logs and write the durable summary as blocked or rejected according to the failure.

- [ ] G4: Validate freshness and required artifacts for the selected run.

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
from scripts.studies.openfwi_flatvel_a.smoke import validate_fresh_artifacts

raw = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate")
run_id = (raw / "logs/selected_smoke.run_id").read_text().strip()
run_root = Path((raw / "logs/selected_smoke.run_root").read_text().strip())
marker_run_id = (run_root / "logs/smoke.run_id").read_text().strip()
if marker_run_id != run_id:
    raise SystemExit(f"smoke run marker does not match selected run: {marker_run_id} != {run_id}")
start_ns = int((run_root / "logs/smoke.started_at_ns").read_text().strip())
errors = validate_fresh_artifacts(output_root=run_root, run_id=run_id, start_ns=start_ns)
profile_metrics = run_root / "runs/hybrid_resnet_smoke/metrics.json"
if not profile_metrics.exists():
    errors.append(f"missing hybrid metrics: {profile_metrics}")
if errors:
    raise SystemExit("OpenFWI smoke freshness validation failed:\n" + "\n".join(errors))
summary = json.loads((run_root / "comparison_summary.json").read_text())
if not summary.get("local_baseline_complete"):
    raise SystemExit("local baseline is incomplete")
print("OpenFWI smoke artifacts are fresh and structurally complete")
PY
```

Expected: prints `OpenFWI smoke artifacts are fresh and structurally complete`.

### Phase H: Reporting, Discoverability, and Summary

**Files:**
- Create: `scripts/studies/openfwi_flatvel_a/reporting.py`
- Test: `tests/studies/test_openfwi_flatvel_a_reporting.py`
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/openfwi_flatvel_a_fallback_smoke_gate.md`
- Modify: `docs/studies/index.md`
- Modify: `docs/index.md`

- [ ] H1: Write reporting tests.

Required test coverage:

- Collator handles metrics and blocker files.
- Collator rejects mismatched run IDs or split manifests.
- Collator emits `evidence_scope: smoke_feasibility_only`, `metric_interpretation: sanity_only_not_benchmark_performance`, and `performance_assessment_complete: false`.
- Collator does not compute or publish `hybrid_MAE`, `best_baseline_MAE`, or `relative_gap_vs_best_baseline` in the JSON summary.
- Collator emits one of `smoke_contract_complete`, `block_data_access`, `block_baseline_incomplete`, or `block_metrics_incomplete` as `recommended_decision_input`.

- [ ] H2: Implement reporting collation.

Required outputs:

- `comparison_summary.json`
- `comparison_summary.csv`
- deterministic CSV columns: `profile_id,status,test_MAE,test_RMSE,test_SSIM,val_MAE,val_RMSE,val_SSIM,runtime_sec,parameter_count,blocker_reason`.

- [ ] H3: Run reporting tests.

Run:

```bash
pytest tests/studies/test_openfwi_flatvel_a_reporting.py -v
```

Expected: tests pass.

- [ ] H4: Update durable study discoverability if tracked runbook/summary surfaces were added.

Update `docs/studies/index.md` with a `openfwi-flatvel-a-fallback-smoke-gate` entry containing:

- purpose
- script `scripts/studies/run_openfwi_flatvel_a_smoke.py`
- required shards
- output root
- decision boundary
- explicit non-goals

Update `docs/index.md` with an entry for `docs/plans/NEURIPS-HYBRID-RESNET-2026/openfwi_flatvel_a_fallback_smoke_gate.md` after the durable summary exists.

- [ ] H5: Write the durable smoke-gate summary.

The summary must include:

- Scope and non-goals.
- Documents and artifacts used.
- Data access status and license/access terms.
- Local data root policy and whether data are referenced only or redistributed.
- Shard identity, size/mtime/checksum where practical.
- Shape validation results or blocker.
- Split manifest and sample caps.
- Normalization and deterministic input-resize preprocessing.
- Metric contract: MAE, RMSE, SSIM and SSIM data-range policy.
- Official InversionNet compatibility status or blocker.
- Hybrid ResNet-compatible smoke result or blocker.
- Local baseline result or blocker.
- Runtime, memory, commands, git state, package versions, seed, output paths.
- Published OpenFWI rows caveat.
- Raw artifact links.
- Gate checks.
- Exactly one final decision: `proceed to OpenFWI longer execution`, `block for storage/data/human decision`, or `reject fallback as nonviable`.

- [ ] H6: Run structural summary checks.

Run:

```bash
python - <<'PY'
from pathlib import Path
summary = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/openfwi_flatvel_a_fallback_smoke_gate.md")
if not summary.exists():
    raise SystemExit("missing OpenFWI smoke summary")
text = summary.read_text()
required_terms = [
    "OpenFWI FlatVel-A",
    "data1.npy",
    "model1.npy",
    "data49.npy",
    "model49.npy",
    "MAE",
    "RMSE",
    "SSIM",
    "seed",
    "normalization",
    "Hybrid ResNet",
    "baseline",
]
missing = [term for term in required_terms if term.lower() not in text.lower()]
if missing:
    raise SystemExit(f"summary missing required terms: {missing}")
decisions = [
    "Decision: proceed to OpenFWI longer execution",
    "Decision: block for storage/data/human decision",
    "Decision: reject fallback as nonviable",
]
hits = [decision for decision in decisions if decision in text]
if len(hits) != 1:
    raise SystemExit(f"expected exactly one allowed decision, found {hits}")
required_boundary_terms = ["smoke", "not benchmark-performance", "performance_assessment_complete: false"]
missing_boundary = [term for term in required_boundary_terms if term.lower() not in text.lower()]
if missing_boundary:
    raise SystemExit(f"summary missing smoke evidence boundary terms: {missing_boundary}")
for forbidden in ["/home/ollie/Documents/neurips/index.md", "CDI anchor", "256x256 scaling"]:
    if forbidden in text and "non-goal" not in text.lower():
        raise SystemExit(f"summary appears to promote forbidden later-phase work: {forbidden}")
print("OpenFWI smoke summary is structurally valid")
PY
```

Expected: prints `OpenFWI smoke summary is structurally valid`.

### Phase I: Final Verification and Handoff

**Files:**
- Read: `state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-2-openfwi-flatvel-a-fallback-smoke-gate/plan-phase/plan_path.txt`
- Verify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-openfwi-flatvel-a-fallback-smoke-gate/execution_plan.md`

- [ ] I1: Run focused OpenFWI tests.

Run:

```bash
pytest tests/studies/test_openfwi_flatvel_a_manifest.py \
  tests/studies/test_openfwi_flatvel_a_data.py \
  tests/studies/test_openfwi_flatvel_a_metrics.py \
  tests/studies/test_openfwi_flatvel_a_models.py \
  tests/studies/test_openfwi_flatvel_a_run_config.py \
  tests/studies/test_openfwi_flatvel_a_smoke_cli.py \
  tests/studies/test_openfwi_flatvel_a_reporting.py -v
```

Expected: all focused OpenFWI tests pass.

- [ ] I2: Run discoverability tests if `docs/studies/index.md` or `docs/index.md` changed.

Run:

```bash
pytest tests/studies/test_studies_index_entries.py -v
```

Expected: tests pass or are extended narrowly for the new OpenFWI entries.

- [ ] I3: Verify output contract pointer still contains only this plan path.

Run:

```bash
python - <<'PY'
from pathlib import Path
pointer = Path("state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-2-openfwi-flatvel-a-fallback-smoke-gate/plan-phase/plan_path.txt")
text = pointer.read_text()
expected = "docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-openfwi-flatvel-a-fallback-smoke-gate/execution_plan.md\n"
if text != expected:
    raise SystemExit(f"unexpected plan_path content: {text!r}")
target = Path(text.strip())
if not target.exists():
    raise SystemExit(f"plan target does not exist: {target}")
print("plan pointer is valid")
PY
```

Expected: prints `plan pointer is valid`.

- [ ] I4: Verify no forbidden later-phase artifacts were created.

Run:

```bash
python - <<'PY'
from pathlib import Path
paper_root = Path("/home/ollie/Documents/neurips")
for relative in ["index.md", "evidence_checklist.md"]:
    path = paper_root / relative
    if path.exists():
        print(f"paper-facing artifact exists from other work, not created by this tranche: {path}")
for forbidden in [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_n256_scaling_summary.md"),
]:
    if forbidden.exists():
        raise SystemExit(f"forbidden later-phase summary exists in this tranche context: {forbidden}")
print("no forbidden later-phase tracked summaries found")
PY
```

Expected: prints `no forbidden later-phase tracked summaries found` unless pre-existing unrelated paper artifacts are merely reported.

## Workflow Compatibility Contract

When this plan is executed by the orchestration workflow:

- Backlog/workflow item must keep `plan_path` pointing to `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-openfwi-flatvel-a-fallback-smoke-gate/execution_plan.md`.
- Execution unit is this selected tranche only.
- Completion requires the durable smoke summary and focused verification commands.
- If shard-level data access is unavailable, completion may be a blocked result with `data_access_blocker.json` and a durable decision of `block for storage/data/human decision`.
- If both official InversionNet and local baseline paths fail, completion is blocked or rejected, not silently downgraded to a Hybrid-only result.
- The next roadmap phase remains Phase 2 longer OpenFWI execution only if the final decision is `proceed to OpenFWI longer execution`. Phase 3 CDI remains gated until this fallback path is advanced or blocked with a recorded reason.

## Verification Commands

Use the focused commands from the phases above. The minimum final verification set is:

```bash
pytest tests/studies/test_openfwi_flatvel_a_manifest.py \
  tests/studies/test_openfwi_flatvel_a_data.py \
  tests/studies/test_openfwi_flatvel_a_metrics.py \
  tests/studies/test_openfwi_flatvel_a_models.py \
  tests/studies/test_openfwi_flatvel_a_run_config.py \
  tests/studies/test_openfwi_flatvel_a_smoke_cli.py \
  tests/studies/test_openfwi_flatvel_a_reporting.py -v

pytest tests/studies/test_studies_index_entries.py -v

python - <<'PY'
from pathlib import Path
summary = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/openfwi_flatvel_a_fallback_smoke_gate.md")
assert summary.exists()
text = summary.read_text()
decisions = [
    "Decision: proceed to OpenFWI longer execution",
    "Decision: block for storage/data/human decision",
    "Decision: reject fallback as nonviable",
]
assert sum(decision in text for decision in decisions) == 1
for term in ["not benchmark-performance", "performance_assessment_complete: false"]:
    assert term in text
print("OpenFWI fallback smoke gate summary decision is valid")
PY
```

If real smoke data are available, also run the tmux launch and freshness checks in Phase G.

## Completion Criteria

- [ ] Required consumed artifacts and OpenFWI handoff context were read before implementation.
- [ ] Exact FlatVel-A smoke shard access is either validated with source/license/local-path/size/mtime/checksum manifest or blocked with a durable data-access reason.
- [ ] Expected shard shapes are validated, or an official schema difference is documented before model execution.
- [ ] Deterministic train/validation/test smoke split uses `data1`/`model1`, `data49`/`model49`, sample caps, and seed `20260420`.
- [ ] MAE, RMSE, and SSIM metrics are implemented, tested, and reported on denormalized velocity maps when normalization is used.
- [ ] Hybrid ResNet-compatible smoke run completes or writes a controlled blocker.
- [ ] Official InversionNet compatibility is proved or explicitly blocked, and at least one local baseline completes or the tranche blocks/rejects.
- [ ] Raw artifacts stay under `.artifacts/` or another ignored/external root; OpenFWI data remain outside git.
- [ ] Long-running commands use tmux, `ptycho311`, exact PID tracking, and selected-run freshness checks.
- [ ] Durable summary exists at `docs/plans/NEURIPS-HYBRID-RESNET-2026/openfwi_flatvel_a_fallback_smoke_gate.md` and ends with exactly one allowed decision.
- [ ] Discoverability docs are updated if durable runbook/summary/tracked adapter surfaces were created.
- [ ] No CDI Phase 3, Phase 4 scaling, Phase 5 paper-facing artifact assembly, stable core-module edit, or worktree creation occurs.

## Artifacts Index

- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-openfwi-flatvel-a-fallback-smoke-gate/execution_plan.md`
- Durable smoke summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/openfwi_flatvel_a_fallback_smoke_gate.md`
- Raw support root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/`
- Source note: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/source_notes/openfwi_2d_acoustic_fwi.md`
- Phase 1 selection: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`
- SWE pivot summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md`
- Future execution report: workflow-materialized path for `phase-2-openfwi-flatvel-a-fallback-smoke-gate`
- Future implementation review: workflow-materialized path for `phase-2-openfwi-flatvel-a-fallback-smoke-gate`
