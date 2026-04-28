# PDEBench CNS Hybrid-Spectral 2048-Cap Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one fresh capped `2048 / 256 / 256`, `40`-epoch PDEBench `2d_cfd_cns` finalist compare for `spectral_resnet_bottleneck_base` and `spectral_resnet_bottleneck_shared_blocks10`, then publish a machine-checkable `512 -> 1024 -> 2048` scaling trend without widening the claim boundary beyond capped decision-support evidence.

**Architecture:** Treat this backlog item as a narrow follow-up to the completed fixed-shell CNS hybrid-spectral architecture ablation. Reuse the frozen `512 / 64 / 64` and `1024 / 128 / 128` finalist artifacts as immutable references, verify that the existing reporting and runner surfaces already support the `2048` scaling lane, repair only concrete workflow blockers, then launch exactly one fresh `pilot` run under the unchanged CNS contract and summarize the resulting trend.

**Tech Stack:** PATH `python`; tmux + `ptycho311` for long runs; PyTorch/Lightning; `scripts/studies/pdebench_image128/`; pytest; compileall; Markdown/JSON/CSV artifacts under `.artifacts/`

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap`
- Selection mode: `RECOVERED_IN_PROGRESS`
- Date: `2026-04-28`
- Selected-item authority: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap/selected-item-context.md`
- Plan-path authority: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap/plan-phase/plan_path.txt`
- Previous plan background only: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap/execution_plan.md`
- Durable summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_scaling_2048cap_summary.md`
- CNS summary sync target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/`

This document supersedes any earlier seed or recovered draft for this backlog item and is the execution authority for implementation.

## Inputs Read

- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/studies/index.md`
- `docs/steering.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap/selected-item-context.md`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap/plan-phase/plan_path.txt`
- `scripts/studies/run_pdebench_image128_suite.py`
- `scripts/studies/pdebench_image128/cfd_cns.py`
- `scripts/studies/pdebench_image128/reporting.py`
- `scripts/studies/pdebench_image128/run_config.py`
- `tests/studies/test_pdebench_image128_models.py`
- `tests/studies/test_pdebench_image128_runner.py`

## Selected Objective

- Extend the completed CNS hybrid-spectral architecture ablation with one fresh capped `2048 / 256 / 256`, `40`-epoch run for:
  - `spectral_resnet_bottleneck_base`
  - `spectral_resnet_bottleneck_shared_blocks10`
- Compare that new run against the frozen `512 / 64 / 64` and `1024 / 128 / 128` rows to answer one bounded question:
  - does either finalist keep improving faster as the capped training set grows from `512` to `1024` to `2048`?
- Publish:
  - absolute metrics at all three caps
  - `512 -> 1024` deltas
  - `1024 -> 2048` deltas
  - improvement per added training trajectory
  - runtime change

## Scope

- Task remains `2d_cfd_cns` only.
- Dataset remains `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`.
- Resolution remains native `128x128`.
- Fixed fairness contract for all frozen and fresh rows:
  - `history_len=2`
  - `max_windows_per_trajectory=8`
  - training loss `mse`
  - batch size `4`
  - `40` epochs
  - metric family `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- Split-cap ladder:
  - frozen pilot lane `512 / 64 / 64`
  - frozen confirmation lane `1024 / 128 / 128`
  - fresh trend lane `2048 / 256 / 256`
- Fresh execution is limited to the two finalist profiles named above.

## Explicit Non-Goals

- Do not widen this into a full-training benchmark tranche, PDEBench suite expansion, or paper-facing competitiveness claim.
- Do not rerun the `512 / 64 / 64` or `1024 / 128 / 128` rows.
- Do not add new architecture axes, mode changes, history-length changes, FFNO rows, GNOT rows, CDI rows, or physics-regularization rows.
- Do not change the fixed CNS contract to make either finalist look better.
- Do not promote either finalist into a default profile or paper-ready claim from this item alone.
- Do not create `/home/ollie/Documents/neurips/` artifacts or manuscript prose.
- Do not touch `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not create worktrees.

## Steering, Roadmap, And Execution Constraints

- Steering is binding:
  - keep equal-footing comparisons explicit
  - do not silently relax fairness constraints
  - preserve the metric, data-split, and protocol boundaries already approved
- The approved design and roadmap are binding on claim boundaries:
  - this item stays capped decision-support evidence only
  - it does not satisfy the roadmap’s full available training-split benchmark gate
  - it does not revise the roadmap’s required PDE and external-baseline priorities
- The backlog item’s reviewer notes are binding:
  - this is a trend check, not a benchmark claim
  - the question is scaling separation between the two finalists, not whether either row is paper-ready
  - if the trend is ambiguous, preserve the bounded conclusion instead of forcing a promotion
- `REPORTING-ARTIFACT-BOUNDARY-001` applies:
  - optional galleries may fail without redefining a metrics-complete run as failed
  - required metrics, manifests, and provenance artifacts still must exist
- Follow `PYTHON-ENV-001`:
  - invoke Python as plain PATH `python`
- Follow the long-run guardrail:
  - use tmux for the `2048` run
  - activate `ptycho311`
  - track the exact launched PID and `wait` on that PID
  - do not use broad `pgrep -f` polling as the main completion check
  - do not start a duplicate run against the same `--output-root`
  - do not call the run complete until the tracked PID exits `0` and required output artifacts are freshly written
- Normal verification failures are not automatic `BLOCKED` outcomes:
  - diagnose, make the narrowest fix in scope, rerun checks, then continue
  - reserve `BLOCKED` for missing hardware/resources, unavailable external dependency, roadmap conflict, user decision required, or an unrecoverable failure after a documented narrow fix attempt

## Prerequisite Status

### Satisfied

- `2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation` is complete in `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`.
- The prerequisite conclusion is already fixed:
  - `spectral_resnet_bottleneck_shared_blocks10` won the shared-depth `512 / 64 / 64` lane
  - `spectral_resnet_bottleneck_base` recovered the aggregate lead on the stronger `1024 / 128 / 128` confirmation lane
  - `shared_blocks10` retained only a narrower `fRMSE_mid/high` edge and higher runtime cost
- The official CNS file, split/normalization contract, denormalized metric family, and supervised runner are already implemented and verified.
- `scripts/studies/pdebench_image128/reporting.py::write_split_cap_scaling_trend` and its runner tests already exist, so this item should only need code changes if a concrete `2048` workflow blocker appears.

### Frozen Reference Roots

- `512 / 64 / 64`, `40`-epoch shared-base root:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-40ep-20260428T035154Z`
- `512 / 64 / 64`, `40`-epoch `shared_blocks10` root:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-depth-shared-40ep-20260428T043715Z`
- `1024 / 128 / 128`, `40`-epoch finalist root:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z`
- Existing larger-cap finalist delta payload:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/finalist_delta_1024cap.json`

## Implementation Architecture

- **Reference Audit Unit:** freeze the exact `512` and `1024` finalist references into explicit manifests, confirm that only `split_counts` differ across caps, and prove that the runner/reporting contract is ready for the fresh lane.
- **Workflow Repair Unit:** patch reporting or runner code only if the current split-cap scaling helper, manifest contract, or artifact plumbing fails against the required `2048` workflow.
- **Execution And Publication Unit:** run the fresh `2048 / 256 / 256` finalist compare, generate the `512 -> 1024 -> 2048` trend payload, then update the durable summary and initiative state.

## Concrete File And Artifact Targets

### Repo Files To Inspect First

- `scripts/studies/pdebench_image128/reporting.py`
- `tests/studies/test_pdebench_image128_runner.py`
- `scripts/studies/run_pdebench_image128_suite.py`
- `scripts/studies/pdebench_image128/cfd_cns.py`
- `scripts/studies/pdebench_image128/run_config.py`
- `tests/studies/test_pdebench_image128_models.py`

### Files Likely To Change Only If A Real Blocker Appears

- `scripts/studies/pdebench_image128/reporting.py`
- `tests/studies/test_pdebench_image128_runner.py`
- `scripts/studies/run_pdebench_image128_suite.py`
- `scripts/studies/pdebench_image128/cfd_cns.py`

### Durable Docs And State To Update

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_scaling_2048cap_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- `docs/studies/index.md`
- `docs/index.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `docs/findings.md` only if this item yields a reusable engineering/reporting rule rather than a one-off result

### Required Generated Artifacts

- study root:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/`
- frozen reference manifests:
  - `reference_runs_512cap_40ep.json`
  - `reference_runs_1024cap_40ep.json`
- preflight inspect root:
  - `inspect-2048cap-<timestamp>/`
- fresh run root:
  - `cns-hybrid-spectral-finalists-2048cap-40ep-<timestamp>/`
- fresh run tracker sidecar:
  - `cns-hybrid-spectral-finalists-2048cap-40ep-<timestamp>.launch/`
- scaling payloads:
  - `finalist_scaling_trend_512_1024_2048.json`
  - `finalist_scaling_trend_512_1024_2048.csv`
- verification archive:
  - `verification/preflight_pytest.log`
  - `verification/preflight_compileall.log`
  - `verification/workflow_fix_pytest.log` if repo code changes
  - `verification/workflow_fix_compileall.log` if repo code changes
  - `verification/workflow_fix_integration.log` if broader production workflow code changes
  - `verification/final_artifact_validation.log`
  - `verification/final_pytest.log`
  - `verification/final_compileall.log`
  - `verification/final_integration.log` if broader production workflow code changes

## Mandatory Deterministic Checks

These backlog-item checks are required before claiming completion:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

Use stronger checks when the edit surface demands them:

- If `reporting.py`, `cfd_cns.py`, or `run_pdebench_image128_suite.py` changes, run a focused selector that covers the repaired behavior before rerunning the full required pytest command.
- If implementation changes broader production workflow plumbing rather than only study-local reporting/test code, rerun `pytest -v -m integration` before the expensive `2048` run and again before completion, per `docs/TESTING_GUIDE.md`.
- The expensive `2048` run must wait for green required checks and any triggered stronger replacement checks.

### Task 1: Freeze Reference Manifests And Preflight The 2048 Workflow

**Files:**
- Inspect: `scripts/studies/pdebench_image128/reporting.py`
- Inspect: `tests/studies/test_pdebench_image128_runner.py`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/reference_runs_512cap_40ep.json`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/reference_runs_1024cap_40ep.json`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/inspect-2048cap-<timestamp>/`

- [ ] Audit the frozen `512` and `1024` roots and write two explicit reference manifests containing only the two finalist rows plus the fixed contract fields that the scaling helper validates.
- [ ] Confirm the manifests prove contract equality on every required invariant field except `split_counts`.
- [ ] Run a fresh `inspect`-mode preflight for the target `2048 / 256 / 256` cap so the split manifest, dataset manifest, and HDF5 metadata for the exact fresh lane exist before any expensive training starts.
- [ ] If any missing artifact is due to a normal harness, manifest, or path defect, diagnose and fix it in scope rather than marking the item blocked.

**Verification**

- [ ] Run the required pytest command and archive output to `verification/preflight_pytest.log`.
- [ ] Run the required compileall command and archive output to `verification/preflight_compileall.log`.
- [ ] Run this preflight inspect command and archive or reference its root:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode inspect \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/inspect-2048cap-<timestamp> \
  --history-len 2 \
  --max-train-trajectories 2048 \
  --max-val-trajectories 256 \
  --max-test-trajectories 256 \
  --max-windows-per-trajectory 8
```

- [ ] Confirm the inspect root contains `invocation.json`, `dataset_manifest.json`, `split_manifest.json`, and `hdf5_metadata.json`.

### Task 2: Repair Workflow Surfaces Only If The Audit Finds A Real Blocker

**Files:**
- Modify only if needed: `scripts/studies/pdebench_image128/reporting.py`
- Modify only if needed: `tests/studies/test_pdebench_image128_runner.py`
- Modify only if needed: `scripts/studies/run_pdebench_image128_suite.py`
- Modify only if needed: `scripts/studies/pdebench_image128/cfd_cns.py`

- [ ] If the audit shows the existing scaling helper, runner plumbing, or manifest contract is insufficient for the `2048` workflow, write the smallest fix that preserves the current fairness contract and evidence boundary.
- [ ] Keep repairs limited to artifact plumbing, manifest validation, or workflow reliability. Do not introduce new scientific axes, new profiles, or new interpretation rules.
- [ ] Add or adjust focused tests only for the repaired behavior.

**Verification**

- [ ] Run the focused selector that covers the repaired behavior and archive it to `verification/workflow_fix_pytest.log`.
- [ ] Rerun the required pytest command and archive it to `verification/workflow_fix_pytest.log` or a second clearly labeled file in the same directory.
- [ ] Rerun the required compileall command and archive it to `verification/workflow_fix_compileall.log`.
- [ ] If broader production workflow code changed, run `pytest -v -m integration` and archive it to `verification/workflow_fix_integration.log`.

### Task 3: Launch Exactly One Fresh 2048-Cap Finalist Run

**Files / Artifacts:**
- Use: `scripts/studies/run_pdebench_image128_suite.py`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-<timestamp>/`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-<timestamp>.launch/`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/finalist_scaling_trend_512_1024_2048.json`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/finalist_scaling_trend_512_1024_2048.csv`

- [ ] Do not start this task until Task 1 and any Task 2 repairs are green.
- [ ] Launch exactly one fresh `pilot` run in tmux for `spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_shared_blocks10` under the unchanged `2048 / 256 / 256`, `40`-epoch CNS contract.
- [ ] Use a unique fresh output root and track the exact PID in a sibling `.launch/` directory that records the command, PID, exit code, and completion timestamps.
- [ ] Refuse to start a duplicate run if another process is already writing to the same output root.
- [ ] After the run exits, verify the fresh root is complete, then generate the `512 -> 1024 -> 2048` scaling payload from the two frozen manifests plus the fresh `2048` root.
- [ ] Ensure the trend payload reports absolute metrics, cap-to-cap deltas, improvement per added trajectory, runtime change, and the explicit evidence boundary `capped_decision_support_only`.
- [ ] If the run or trend generation fails because of a normal repo or harness defect, diagnose, apply the narrowest in-scope fix, rerun checks, and relaunch. Reserve `BLOCKED` for the narrow conditions listed above.

**Execution Command**

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-<timestamp> \
  --profiles spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_shared_blocks10 \
  --history-len 2 \
  --epochs 40 \
  --batch-size 4 \
  --max-train-trajectories 2048 \
  --max-val-trajectories 256 \
  --max-test-trajectories 256 \
  --max-windows-per-trajectory 8 \
  --device cuda
```

**Verification**

- [ ] Validate the tracked run exited `0`.
- [ ] Validate the fresh run root contains `invocation.json`, `dataset_manifest.json`, `split_manifest.json`, `comparison_summary.json`, `metrics_spectral_resnet_bottleneck_base.json`, `metrics_spectral_resnet_bottleneck_shared_blocks10.json`, and matching `model_profile_*.json` files.
- [ ] Validate the scaling JSON/CSV exist and that their contract check treats `split_counts` as the only allowed cap-varying field.
- [ ] Archive the artifact-validation output to `verification/final_artifact_validation.log`.

### Task 4: Publish The Result And Sync Initiative State

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_scaling_2048cap_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/studies/index.md`
- Modify: `docs/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Modify conditionally: `docs/findings.md`

- [ ] Write a durable summary that names the frozen roots, the fresh `2048` root, the scaling payload paths, the absolute metrics, the cap-to-cap deltas, the runtime deltas, and the bounded interpretation.
- [ ] Update the CNS summary with a short pointer to this new capped scaling lane while preserving that it is not a full-training benchmark result.
- [ ] Update `docs/studies/index.md` and `docs/index.md` so the new summary is discoverable.
- [ ] Add a progress-ledger completion entry for this backlog item with decision, evidence scope, metric interpretation, artifact root, fresh run root, summary path, and any follow-on recommendation implied by the trend.
- [ ] Update `docs/findings.md` only if this item yields a reusable engineering/reporting lesson rather than a one-off experimental conclusion.

**Verification**

- [ ] Rerun the required pytest command and archive to `verification/final_pytest.log`.
- [ ] Rerun the required compileall command and archive to `verification/final_compileall.log`.
- [ ] If broader production workflow code changed during the item, rerun `pytest -v -m integration` and archive to `verification/final_integration.log`.
- [ ] Confirm every summary, index, and ledger path listed above exists and points at the fresh `2048` scaling artifacts rather than the older `1024`-cap-only tranche.

## Completion Standard

This backlog item is complete only when all of the following are true:

- the two frozen reference manifests exist and prove the fixed contract for the `512` and `1024` lanes
- the fresh `2048 / 256 / 256` finalist run completed successfully under the unchanged CNS contract
- the `finalist_scaling_trend_512_1024_2048.json` and `.csv` payloads exist and encode the bounded scaling question
- the required deterministic checks are green, plus any stronger checks triggered by workflow edits
- the durable summary, CNS summary sync, discoverability updates, and progress-ledger completion entry are all written
- the final interpretation stays within the approved boundary: capped scaling evidence may guide the next backlog item, but it does not by itself justify benchmark or default-profile promotion
