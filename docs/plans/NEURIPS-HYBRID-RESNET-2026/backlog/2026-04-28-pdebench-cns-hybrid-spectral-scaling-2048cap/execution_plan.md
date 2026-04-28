# PDEBench CNS Hybrid-Spectral 2048-Cap Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one fresh capped `2048 / 256 / 256`, `40`-epoch PDEBench `2d_cfd_cns` finalist compare for `spectral_resnet_bottleneck_base` and `spectral_resnet_bottleneck_shared_blocks10`, then publish an auditable `512 -> 1024 -> 2048` scaling trend without widening the claim boundary beyond capped decision-support evidence.

**Architecture:** Treat this item as a narrow continuation of the completed CNS hybrid-spectral architecture ablation. Freeze the existing `512 / 64 / 64` and `1024 / 128 / 128` finalist artifacts into explicit reference manifests, prove the fixed contract only changes in `split_counts`, confirm the current runner/reporting path is ready for the fresh `2048` lane, repair only concrete study-local blockers if preflight exposes them, then run exactly one fresh capped compare and summarize the scaling deltas. The expensive `2048` run must wait for green deterministic checks and any stronger targeted checks triggered by code changes.

**Tech Stack:** PATH `python`; tmux + `ptycho311` for the long run; PyTorch/Lightning; `scripts/studies/pdebench_image128/`; pytest; compileall; Markdown/JSON/CSV artifacts under `.artifacts/`

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap`
- Selection mode: `RECOVERED_IN_PROGRESS`
- Date: `2026-04-28`
- Consumed selected-item authority: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/4/items/2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap/selected-item-context.md`
- Plan-path authority: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/4/items/2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap/plan-phase/plan_path.txt`
- Previous plan note: the recovered draft at this same plan path is background context only and is superseded by this document.
- Durable summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_scaling_2048cap_summary.md`
- CNS summary sync target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/`

This document supersedes any recovered draft for this backlog item and is the new execution authority.

## Inputs Read

- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/studies/index.md`
- `docs/steering.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/4/items/2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap/selected-item-context.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- `scripts/studies/pdebench_image128/reporting.py`
- `scripts/studies/pdebench_image128/cfd_cns.py`
- `scripts/studies/run_pdebench_image128_suite.py`
- `tests/studies/test_pdebench_image128_models.py`
- `tests/studies/test_pdebench_image128_runner.py`

## Selected Objective

- Extend the completed CNS hybrid-spectral ablation with one fresh capped `2048 / 256 / 256`, `40`-epoch run for:
  - `spectral_resnet_bottleneck_base`
  - `spectral_resnet_bottleneck_shared_blocks10`
- Compare that fresh lane against the frozen `512 / 64 / 64` and `1024 / 128 / 128` finalist rows to answer one bounded question:
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
- Fixed fairness contract for frozen and fresh rows:
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

## Steering, Roadmap, And Claim Constraints

- Steering is binding:
  - keep equal-footing comparisons explicit
  - do not silently relax fairness constraints
  - preserve the approved metric, split, and protocol boundaries
- The approved design and roadmap are binding on claim scope:
  - this item stays capped decision-support evidence only
  - it does not satisfy the roadmap’s full-available-training-split benchmark gate
  - it does not reopen Phase 2 ordering or external-baseline decisions
- The selected-item reviewer notes are binding:
  - this is a trend check, not a benchmark claim
  - the central question is scaling separation between the two finalists
  - if the trend is ambiguous, keep the bounded conclusion instead of forcing a promotion
- `REPORTING-ARTIFACT-BOUNDARY-001` applies:
  - required metrics, manifests, provenance, and scaling payloads must exist
  - optional gallery or rerender failures may be recorded as warnings instead of turning a metrics-complete run into a failed study
- `PYTHON-ENV-001` applies:
  - invoke Python as plain PATH `python`
- Long-run guardrail for the `2048` compare:
  - use tmux
  - activate `ptycho311`
  - launch exactly one run against a fresh `--output-root`
  - track the exact launched PID and wait on that PID
  - do not use broad `pgrep -f` polling as the completion check
  - accept completion only after exit code `0` and freshly written required artifacts
- Normal check, import, path, harness, or manifest failures are not automatic `BLOCKED` outcomes:
  - diagnose
  - apply the narrowest in-scope fix
  - rerun the affected checks
  - reserve `BLOCKED` for missing hardware/resources, unavailable external dependency, roadmap conflict, user decision required, or unrecoverable failure after a documented narrow fix attempt

## Prerequisite Status

### Satisfied

- The CNS hybrid-spectral architecture ablation is complete and already narrowed the finalists:
  - `spectral_resnet_bottleneck_shared_blocks10` won the shared-depth `512 / 64 / 64` lane
  - `spectral_resnet_bottleneck_base` recovered the aggregate lead on the `1024 / 128 / 128` confirmation lane
- The official CNS file, split/normalization contract, denormalized metric family, and supervised runner already exist and are verified in the current repo state.
- `scripts/studies/pdebench_image128.reporting.write_split_cap_scaling_trend` and dedicated regression tests already exist, so repo code changes are justified only if the audit or `2048` preflight exposes a real workflow blocker.

### Progress-Ledger Context That Matters

- The ledger already records the completed CNS hybrid-spectral architecture ablation and the exact artifact roots needed for the frozen `512` and `1024` references.
- The ledger and current CNS summary already record that the official `2d_cfd_cns` HDF5 is staged and verified, so this item is not data-blocked.
- No roadmap entry authorizes broadening this item into full-training or paper-claim work; the current ledger/design state continues to treat capped CNS rows as decision-support only.

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

- **Reference Audit Unit:** freeze the exact `512` and `1024` finalist references into explicit manifests, prove that the fixed contract differs only on `split_counts`, and confirm the current runner/reporting path is ready for the fresh `2048` lane.
- **Workflow Repair Unit:** patch reporting, runner, or test surfaces only if the audit or preflight exposes a real `2048` workflow blocker.
- **Execution And Publication Unit:** run the fresh `2048 / 256 / 256` finalist compare, generate the `512 -> 1024 -> 2048` trend payload, update durable summaries, and sync initiative state.

## Concrete File And Artifact Targets

### Repo Files To Inspect First

- `scripts/studies/pdebench_image128/reporting.py`
- `scripts/studies/pdebench_image128/cfd_cns.py`
- `scripts/studies/run_pdebench_image128_suite.py`
- `tests/studies/test_pdebench_image128_runner.py`
- `tests/studies/test_pdebench_image128_models.py`

### Files Likely To Change Only If A Real Blocker Appears

- `scripts/studies/pdebench_image128/reporting.py`
- `scripts/studies/pdebench_image128/cfd_cns.py`
- `scripts/studies/run_pdebench_image128_suite.py`
- `tests/studies/test_pdebench_image128_runner.py`
- `tests/studies/test_pdebench_image128_models.py`

### Durable Docs And State To Update

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_scaling_2048cap_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- `docs/studies/index.md`
- `docs/index.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `docs/findings.md` only if this item exposes a reusable workflow/reporting rule rather than a one-off scientific result

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
  - `compare_scaling_512_1024_2048_sample0.png`
  - `compare_scaling_512_1024_2048_sample0_error.png`
- verification archive:
  - `verification/preflight_pytest.log`
  - `verification/preflight_compileall.log`
  - `verification/workflow_fix_focused.log` if repo code changes
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

Use stronger targeted checks when the edit surface demands them:

- If `reporting.py` or the scaling-payload path changes, run these targeted regressions before rerunning the full required selector:

```bash
pytest -q \
  tests/studies/test_pdebench_image128_runner.py::test_scaling_trend_split_cap_delta_writes_outputs_and_deltas \
  tests/studies/test_pdebench_image128_runner.py::test_split_cap_delta_rejects_contract_drift_outside_split_counts \
  tests/studies/test_pdebench_image128_runner.py::test_scaling_trend_records_nonfatal_gallery_blocker
```

- If `cfd_cns.py` or inspect-mode artifact writing changes, run this targeted regression before rerunning the full required selector:

```bash
pytest -q tests/studies/test_pdebench_image128_runner.py::test_cfd_cns_inspect_runner_writes_split_manifest
```

- If broader production workflow plumbing changes rather than only study-local code, rerun `pytest -v -m integration` before the expensive `2048` run and again before completion, per `docs/TESTING_GUIDE.md`.
- The expensive `2048` run must wait for green required checks and any triggered stronger checks.

## Task 1: Freeze Reference Lanes And Preflight The 2048 Workflow

**Files:**
- Inspect: `scripts/studies/pdebench_image128/reporting.py`
- Inspect: `scripts/studies/pdebench_image128/cfd_cns.py`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/reference_runs_512cap_40ep.json`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/reference_runs_1024cap_40ep.json`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/inspect-2048cap-<timestamp>/`

- [ ] Audit the frozen `512` and `1024` roots and write explicit reference manifests containing only the two finalist rows plus the fixed contract fields that `write_split_cap_scaling_trend` validates.
- [ ] Confirm the two reference manifests prove equality on every scaling-invariant contract field and differ only on `split_counts`.
- [ ] Run the backlog item’s required pytest and compileall checks before any expensive execution, archiving logs under `verification/`.
- [ ] Run a fresh `inspect`-mode preflight for the target `2048 / 256 / 256` cap so the split manifest, dataset manifest, and HDF5 metadata for the exact fresh lane exist before training starts.
- [ ] Diagnose and fix any normal harness, manifest, import, or path defect in scope instead of marking the item blocked.

**Verification**

- [ ] Archive the required pytest log:

```bash
set -o pipefail
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py \
  2>&1 | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/preflight_pytest.log
```

- [ ] Archive the required compileall log:

```bash
set -o pipefail
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py \
  2>&1 | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/preflight_compileall.log
```

- [ ] Run and retain the inspect root:

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

## Task 2: Repair Only Concrete 2048 Workflow Blockers

**Files:**
- Modify only if needed: `scripts/studies/pdebench_image128/reporting.py`
- Modify only if needed: `scripts/studies/pdebench_image128/cfd_cns.py`
- Modify only if needed: `scripts/studies/run_pdebench_image128_suite.py`
- Modify only if needed: `tests/studies/test_pdebench_image128_runner.py`
- Modify only if needed: `tests/studies/test_pdebench_image128_models.py`

- [ ] If Task 1 exposes a real blocker, write or tighten the narrowest failing test that reproduces it before changing code.
- [ ] Patch only the minimal study-local surface needed to restore the `2048` workflow; do not reopen architecture behavior, contract policy, or unrelated PDEBench tasks.
- [ ] Rerun the focused selector that covers the repaired behavior, then rerun the backlog item’s full required pytest and compileall checks.
- [ ] If the repair touches broader production workflow plumbing, rerun `pytest -v -m integration` and archive the log before the expensive run.
- [ ] Do not proceed to Task 3 until every required and triggered stronger check is green.

**Verification**

- [ ] Archive focused regression evidence under `verification/workflow_fix_focused.log` or a similarly specific log name for the repaired surface.
- [ ] Archive the rerun required checks under:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/workflow_fix_pytest.log`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/workflow_fix_compileall.log`
- [ ] If integration ran, archive:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/workflow_fix_integration.log`

## Task 3: Run The Fresh 2048-Cap Finalist Compare

**Files:**
- Generate only: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-<timestamp>/`
- Generate only: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-<timestamp>.launch/`

- [ ] Create a fresh output root and a sibling `.launch/` tracker directory; do not reuse an older run root and do not start a duplicate run against the same output root.
- [ ] Launch the `2048 / 256 / 256`, `40`-epoch compare in tmux from the `ptycho311` environment, track the exact PID, wait on that PID, and record the exit code.
- [ ] Run exactly these two profiles and no others:
  - `spectral_resnet_bottleneck_base`
  - `spectral_resnet_bottleneck_shared_blocks10`
- [ ] Require fresh completion artifacts at minimum:
  - `invocation.json`
  - `dataset_manifest.json`
  - `split_manifest.json`
  - `comparison_summary.json`
  - `metrics_spectral_resnet_bottleneck_base.json`
  - `metrics_spectral_resnet_bottleneck_shared_blocks10.json`
  - `model_profile_spectral_resnet_bottleneck_base.json`
  - `model_profile_spectral_resnet_bottleneck_shared_blocks10.json`

**Run Command**

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

**Long-Run Guardrail Sketch**

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
mkdir -p <launch_dir>
python scripts/studies/run_pdebench_image128_suite.py ... \
  > <launch_dir>/stdout.log 2>&1 &
pid=$!
printf '%s\n' "$pid" > <launch_dir>/pid.txt
wait "$pid"
code=$?
printf '%s\n' "$code" > <launch_dir>/exit_code.txt
exit "$code"
```

## Task 4: Emit And Validate The 512 -> 1024 -> 2048 Scaling Payload

**Files:**
- Use: `scripts/studies/pdebench_image128/reporting.py`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/finalist_scaling_trend_512_1024_2048.json`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/finalist_scaling_trend_512_1024_2048.csv`

- [ ] Reuse `write_split_cap_scaling_trend` with the two reference manifests and the fresh `2048` run root; do not hand-build the scaling payload.
- [ ] Ensure the emitted payload keeps `evidence_scope="capped_decision_support_only"` and `metric_interpretation="decision_support_not_benchmark_performance"`.
- [ ] Require the payload to report:
  - absolute metrics by cap
  - `1024_minus_512` deltas
  - `2048_minus_1024` deltas
  - improvement per added training trajectory
  - runtime deltas
- [ ] Treat a missing cross-run gallery as a warning only if the payload, CSV, and blocker reason are still emitted; required metrics and manifests remain mandatory.

**Emission Snippet**

```bash
python - <<'PY'
from pathlib import Path
from scripts.studies.pdebench_image128.reporting import write_split_cap_scaling_trend

root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap")
profile_ids = [
    "spectral_resnet_bottleneck_base",
    "spectral_resnet_bottleneck_shared_blocks10",
]
write_split_cap_scaling_trend(
    output_root=root,
    profile_ids=profile_ids,
    reference_manifest_paths=[
        root / "reference_runs_512cap_40ep.json",
        root / "reference_runs_1024cap_40ep.json",
    ],
    fresh_run_root=root / "cns-hybrid-spectral-finalists-2048cap-40ep-<timestamp>",
    fresh_profile_ids=profile_ids,
    fresh_source_document="docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_scaling_2048cap_summary.md",
)
PY
```

**Validation**

- [ ] Archive a final artifact-validation log that asserts:
  - both reference manifests exist
  - the fresh run root exists
  - `finalist_scaling_trend_512_1024_2048.json` exists
  - `finalist_scaling_trend_512_1024_2048.csv` exists
  - `cap_sequence == ["512cap", "1024cap", "2048cap"]`
  - both finalist profile IDs are present
  - `allowed_contract_delta.delta_kind == "split_counts_only"`
  - evidence scope stays capped decision-support only

## Task 5: Publish The Durable Study Summary And Sync Project State

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_scaling_2048cap_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/studies/index.md`
- Modify: `docs/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Modify only if warranted: `docs/findings.md`

- [ ] Write the durable summary with:
  - fixed fairness contract
  - frozen `512` and `1024` reference roots
  - fresh `2048` run root
  - absolute metrics at all three caps
  - `512 -> 1024` and `1024 -> 2048` deltas
  - per-trajectory improvement values
  - runtime deltas
  - the bounded interpretation of whether one finalist scales more favorably
  - an explicit claim boundary stating this remains capped decision-support evidence only
- [ ] Update `pdebench_2d_cfd_cns_summary.md` with a short discoverable note linking this bounded scaling follow-up.
- [ ] Update `docs/studies/index.md` and `docs/index.md` so the new summary is discoverable.
- [ ] Update `progress_ledger.json` with the completed capped scaling lane and its durable artifact paths.
- [ ] Update `docs/findings.md` only if the work produced a reusable engineering/reporting rule rather than a one-off scientific observation.

## Final Verification And Completion Conditions

- [ ] Re-run the backlog item’s required pytest and compileall checks after any code changes and archive:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/final_pytest.log`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/final_compileall.log`
- [ ] If broader production workflow code changed, rerun and archive:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/final_integration.log`
- [ ] Confirm the final summary, CNS summary sync, and progress-ledger update all point at the same fresh `2048` run root and scaling payload paths.
- [ ] Completion is satisfied only if:
  - the fresh `2048` compare exited `0`
  - required run artifacts are freshly written
  - both reference manifests exist
  - the scaling JSON and CSV exist
  - the durable summary is written
  - required deterministic checks are green

If the `2048` run cannot proceed because the GPU host, dataset, or another external prerequisite is unavailable, record the blocker with the narrowest truthful reason and stop there. Otherwise, diagnose and repair normal verification or harness failures before considering the item blocked.
