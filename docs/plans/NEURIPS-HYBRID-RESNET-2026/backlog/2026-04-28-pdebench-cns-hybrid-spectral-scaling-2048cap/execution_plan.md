# PDEBench CNS Hybrid-Spectral 2048-Cap Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one fresh capped `2048 / 256 / 256`, `40`-epoch PDEBench `2d_cfd_cns` compare for `spectral_resnet_bottleneck_base` and `spectral_resnet_bottleneck_shared_blocks10`, then publish an auditable `512 -> 1024 -> 2048` scaling trend without widening this item beyond capped decision-support evidence.

**Architecture:** Treat this item as a bounded follow-up to the completed CNS hybrid-spectral architecture ablation. Reuse the frozen `512 / 64 / 64` and `1024 / 128 / 128` finalist roots through machine-readable reference manifests, prove the runner still stages the `2048 / 256 / 256` contract in `inspect` mode, then launch exactly one fresh two-profile `2048` run and write a deterministic scaling payload plus narrow durable summaries. Default expectation is zero production-code changes; edit only the smallest study-local surface if a real preflight blocker appears.

**Tech Stack:** PATH `python`; tmux with `ptycho311` active for long runs; PyTorch/Lightning; `scripts/studies/pdebench_image128/`; pytest; compileall; Markdown/JSON/CSV artifacts under `.artifacts/`.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap`
- Selection mode: `RECOVERED_IN_PROGRESS`
- Date: `2026-04-28`
- Authoritative selected-item context:
  `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/12/items/2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap/selected-item-context.md`
- Previous plan path used as background only:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap/execution_plan.md`
- Durable summary target:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_scaling_2048cap_summary.md`
- CNS summary sync target:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/`

This document is the execution authority for this backlog item. It supersedes prior plan content at this path.

## Inputs Read

- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/model_baselines.md`
- `docs/workflows/pytorch.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/steering.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/12/items/2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap/selected-item-context.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`

## Selected Objective

- Extend the completed CNS hybrid-spectral architecture ablation with one fresh capped `2048 / 256 / 256`, `40`-epoch run for:
  - `spectral_resnet_bottleneck_base`
  - `spectral_resnet_bottleneck_shared_blocks10`
- Compare that fresh lane against the frozen `512 / 64 / 64` and `1024 / 128 / 128` finalist rows to answer one bounded question:
  - does either finalist continue improving faster as the capped training set grows from `512` to `1024` to `2048`?
- Publish:
  - absolute metrics at all three caps
  - `512 -> 1024` deltas
  - `1024 -> 2048` deltas
  - improvement per added training trajectory
  - runtime change

## Scope

- Keep the same PDEBench `2d_cfd_cns` contract used by the completed hybrid-spectral architecture ablation:
  - official `128x128` CNS file:
    `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
  - `history_len=2`
  - `max_windows_per_trajectory=8`
  - training loss `mse`
  - batch size `4`
  - `40` epochs
  - metric family `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- Run only the new `2048 / 256 / 256` cap for the prior larger-cap finalists:
  - `spectral_resnet_bottleneck_base`
  - `spectral_resnet_bottleneck_shared_blocks10`
- Treat the completed `512 / 64 / 64` and `1024 / 128 / 128` artifacts as frozen references. Do not rerun them.
- Preserve the fixed hybrid-spectral shell for both finalists:
  - `base_model="spectral_resnet_bottleneck_net"`
  - `hidden_channels=32`
  - `fno_modes=12`
  - `fno_blocks=4`
  - `hybrid_downsample_steps=2`
  - `hybrid_resnet_blocks=6`
  - `hybrid_skip_connections=True`
  - `hybrid_skip_style="add"`
  - `hybrid_upsampler="pixelshuffle"`
  - `spectral_bottleneck_modes=12`
  - `spectral_bottleneck_gate_init=0.1`
  - `spectral_bottleneck_gate_mode="shared"`
- Vary only the finalist-defining spectral bottleneck depth:
  - `spectral_resnet_bottleneck_base`: `spectral_bottleneck_blocks=6`
  - `spectral_resnet_bottleneck_shared_blocks10`: `spectral_bottleneck_blocks=10`
- Default expectation is zero production-code changes. Code or test edits are authorized only if preflight exposes a concrete study-local blocker in the current runner, reporting, manifest, or inspect path.

## Explicit Non-Goals

- Do not widen this into a full-training benchmark tranche, suite-wide benchmark claim, or paper-facing competitiveness result.
- Do not rerun the `512 / 64 / 64` or `1024 / 128 / 128` rows.
- Do not add new architecture axes, mode changes, history-length changes, FFNO rows, GNOT rows, CDI rows, or physics-regularization rows.
- Do not revisit finalist selection; the completed architecture ablation already fixed the two rows this item must scale-check.
- Do not change the fixed CNS contract to make either finalist look better.
- Do not promote either finalist into a default profile or paper-ready claim from this item alone.
- Do not create `/home/ollie/Documents/neurips/` artifacts or manuscript prose.
- Do not touch `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not create worktrees.

## Steering, Roadmap, And Prerequisite Constraints

- Steering is binding:
  - keep equal-footing comparisons explicit
  - do not silently relax fairness constraints
  - preserve the approved metric, data-split, and protocol boundaries
  - do not let this item displace or reopen the external-baseline queue ordering; this item is a local hybrid-spectral follow-up, not an FFNO or GNOT baseline tranche
- The approved design and roadmap are binding on scope:
  - this item remains Phase 2 PDEBench CNS work
  - this item stays capped decision-support evidence only
  - this item does not satisfy the roadmap full-available-training-split benchmark gate
  - this item does not reopen Darcy work, broader image-suite work, or Phase 3 CDI work
  - the roadmap requirement that meaningful benchmark rows use the full available training split remains in force, which is why every result from this item must stay labeled as capped decision support
- The selected backlog item’s review notes are binding:
  - this is a trend check, not a benchmark claim
  - the central question is scaling separation between the two finalists
  - if the trend is mixed or ambiguous, keep the bounded conclusion instead of forcing a promotion
- Prerequisite status from the progress ledger:
  - `2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation` is complete
  - that prerequisite already froze the finalist profile set and the exact `512` and `1024` reference roots
  - that prerequisite remains `performance_assessment_complete=false`
  - that prerequisite remains `evidence_scope=capped_decision_support_only`
  - that prerequisite recorded the comparison standard:
    lower `relative_l2`, then lower `err_nRMSE`, then lower `fRMSE_high`, then lower parameter count
  - the current CNS task-local contract already fixed `mse` as the correct loss for this PDEBench CNS lane; do not drift back to generic MAE defaults from unrelated supervised adapters
- Fairness context to preserve from the completed ablation:
  - at `512 / 64 / 64`, `spectral_resnet_bottleneck_shared_blocks10` beat the shared base row on aggregate metrics
  - at `1024 / 128 / 128`, `spectral_resnet_bottleneck_base` recovered the aggregate lead while `shared_blocks10` kept only a narrow `fRMSE_mid/high` edge and higher runtime
  - this item exists only to test whether `2048 / 256 / 256` materially changes that trend
- `REPORTING-ARTIFACT-BOUNDARY-001` applies:
  - required metrics, manifests, provenance, and scaling payloads must exist
  - optional gallery or rerender failures may be recorded as warnings instead of turning a metrics-complete run into a failed study
- `PYTHON-ENV-001` applies:
  - invoke Python as plain PATH `python`
- Long-run execution guardrails are mandatory:
  - use tmux for the long training run
  - activate `ptycho311`
  - launch exactly one run against a fresh `--output-root`
  - track the exact launched PID and wait on that PID
  - do not use broad `pgrep -f` polling as the completion check
  - accept completion only after exit code `0` and fresh required output artifacts exist
- Failure handling rule:
  - do not mark the item `BLOCKED` because a normal pytest, compileall, import, inspect-mode, or runner-harness check fails; diagnose, patch narrowly, and rerun first
  - reserve `BLOCKED` for missing dataset, unavailable hardware or host, output-root conflict that cannot be safely resolved, roadmap conflict outside current authority, user decision required, or another unrecoverable failure after a documented narrow fix attempt

## Frozen Reference Inputs

The scaling payload must be built from these already-completed roots and no others unless preflight proves one is unusable:

- `512 / 64 / 64`, `40` epochs:
  - `spectral_resnet_bottleneck_base`:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-40ep-20260428T035154Z`
  - `spectral_resnet_bottleneck_shared_blocks10`:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-depth-shared-40ep-20260428T043715Z`
- `1024 / 128 / 128`, `40` epochs:
  - both finalists:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z`
- Source summary for the frozen references:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`
- Existing reporting helpers to use instead of ad hoc payload assembly:
  - `scripts/studies/pdebench_image128/reporting.py::build_reference_run_manifest`
  - `scripts/studies/pdebench_image128/reporting.py::write_reference_run_manifest`
  - `scripts/studies/pdebench_image128/reporting.py::write_split_cap_scaling_trend`

## Implementation Architecture

- Reference-manifest unit:
  - build item-local `512cap` and `1024cap` reference manifests through the existing reporting helpers so the scaling compare validates against machine-readable contracts rather than prose alone
- Preflight and repair unit:
  - run the required deterministic checks, stage a fresh inspect snapshot for `2048 / 256 / 256`, and patch only the smallest study-local surface if that workflow is broken
- Fresh execution unit:
  - launch exactly one new two-profile `2048 / 256 / 256`, `40`-epoch run under the fixed CNS contract after preflight is green
- Reporting and state-sync unit:
  - emit the scaling JSON/CSV payload, write the durable summary, sync the broader CNS summary, and update ledger and discoverability surfaces only when durable knowledge changed

## Concrete File And Artifact Targets

- Study code and tests that may change only if preflight exposes a blocker:
  - `scripts/studies/pdebench_image128/reporting.py`
  - `scripts/studies/pdebench_image128/cfd_cns.py`
  - `scripts/studies/run_pdebench_image128_suite.py`
  - `tests/studies/test_pdebench_image128_models.py`
  - `tests/studies/test_pdebench_image128_runner.py`
- Item-local artifacts to create or refresh:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/reference_runs_512cap_40ep.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/reference_runs_1024cap_40ep.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/inspect-2048cap-<timestamp>/`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-<timestamp>/`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-<timestamp>.launch/`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/finalist_scaling_trend_512_1024_2048.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/finalist_scaling_trend_512_1024_2048.csv`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/*.log`
- Durable docs and state to create or update after successful execution:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_scaling_2048cap_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
  - `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
  - `docs/index.md` because the new durable summary becomes part of the docs hub
  - `docs/findings.md` only if execution uncovers a reusable engineering or reporting rule beyond this one backlog item
  - `docs/studies/index.md` only if a durable runbook/discoverability contract changes

### Task 1: Freeze Reference Manifests And Run Deterministic Preflight

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/reference_runs_512cap_40ep.json`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/reference_runs_1024cap_40ep.json`
- Use: `scripts/studies/pdebench_image128/reporting.py`
- Use: `scripts/studies/run_pdebench_image128_suite.py`

- [ ] Create the item artifact root and its `verification/` subdirectory.
- [ ] Materialize the two frozen reference manifests through the existing reporting helpers, not by hand-editing JSON. Use a repo-root `python - <<'PY'` snippet equivalent to:

```bash
python - <<'PY'
from pathlib import Path
from scripts.studies.pdebench_image128.reporting import (
    build_reference_run_manifest,
    write_reference_run_manifest,
)

artifact_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap")
metric_family = ["err_RMSE", "err_nRMSE", "relative_l2", "fRMSE_low", "fRMSE_mid", "fRMSE_high"]
dataset_file = "/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
source_document = "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md"

manifest_512 = build_reference_run_manifest(
    task_id="2d_cfd_cns",
    dataset_file=dataset_file,
    split_counts={"train": 512, "val": 64, "test": 64},
    max_windows_per_trajectory=8,
    history_len=2,
    training_loss="mse",
    batch_size=4,
    metric_family=metric_family,
    required_rows={
        "40ep": [
            {
                "run_root": ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-40ep-20260428T035154Z",
                "profile_id": "spectral_resnet_bottleneck_base",
                "epochs": 40,
                "source_document": source_document,
            },
            {
                "run_root": ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-depth-shared-40ep-20260428T043715Z",
                "profile_id": "spectral_resnet_bottleneck_shared_blocks10",
                "epochs": 40,
                "source_document": source_document,
            },
        ]
    },
)
write_reference_run_manifest(manifest_512, artifact_root / "reference_runs_512cap_40ep.json")

manifest_1024 = build_reference_run_manifest(
    task_id="2d_cfd_cns",
    dataset_file=dataset_file,
    split_counts={"train": 1024, "val": 128, "test": 128},
    max_windows_per_trajectory=8,
    history_len=2,
    training_loss="mse",
    batch_size=4,
    metric_family=metric_family,
    required_rows={
        "40ep": [
            {
                "run_root": ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z",
                "profile_id": "spectral_resnet_bottleneck_base",
                "epochs": 40,
                "source_document": source_document,
            },
            {
                "run_root": ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z",
                "profile_id": "spectral_resnet_bottleneck_shared_blocks10",
                "epochs": 40,
                "source_document": source_document,
            },
        ]
    },
)
write_reference_run_manifest(manifest_1024, artifact_root / "reference_runs_1024cap_40ep.json")
PY
```

- [ ] Ensure both manifests record the fixed contract explicitly:
  - dataset file
  - split counts
  - `max_windows_per_trajectory=8`
  - `history_len=2`
  - `training_loss="mse"`
  - `batch_size=4`
  - `epochs=40`
  - metric family `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
  - `source_document` pointing at `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`
- [ ] Run the backlog item’s required deterministic checks before any inspect or training run. These are mandatory, and the expensive `2048` run must wait for both to be green:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

- [ ] Archive those exact checks under the verification directory, for example:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py \
  | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/preflight_pytest.log

python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py \
  2>&1 | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/preflight_compileall.log
```

- [ ] Verify the selected CNS dataset file exists at the fixed path before continuing.
- [ ] Do not continue to Task 2 or Task 3 until both required checks are green.

**Verification for Task 1**

- [ ] The required pytest selector passed and its log is archived.
- [ ] The required compileall command exited `0` and its log is archived.
- [ ] Both reference manifest JSON files exist, parse, and enumerate exactly the expected two rows.
- [ ] Both manifests preserve the same fixed contract except for the intentionally different split counts.

### Task 2: Prove The `2048 / 256 / 256` Contract In Inspect Mode And Repair Only Narrow Blockers

**Files:**
- Use: `scripts/studies/run_pdebench_image128_suite.py`
- Modify only if needed: `scripts/studies/pdebench_image128/reporting.py`
- Modify only if needed: `scripts/studies/pdebench_image128/cfd_cns.py`
- Modify only if needed: `scripts/studies/run_pdebench_image128_suite.py`
- Modify only if needed: `tests/studies/test_pdebench_image128_models.py`
- Modify only if needed: `tests/studies/test_pdebench_image128_runner.py`

- [ ] Stage one fresh inspect root such as:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/inspect-2048cap-<timestamp>/`
- [ ] Run `inspect` mode with the exact fixed `2048 / 256 / 256` contract:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode inspect \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/inspect-2048cap-<timestamp> \
  --profiles spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_shared_blocks10 \
  --history-len 2 \
  --epochs 40 \
  --batch-size 4 \
  --max-train-trajectories 2048 \
  --max-val-trajectories 256 \
  --max-test-trajectories 256 \
  --max-windows-per-trajectory 8 \
  --device cuda \
  --num-workers 0
```

- [ ] Confirm the inspect root records the staging contract through:
  - `invocation.json`
  - `dataset_manifest.json`
  - `split_manifest.json`
- [ ] Verify `split_manifest.json` reports `train=2048`, `val=256`, `test=256`, `history_len=2`, and `max_windows_per_trajectory=8`.
- [ ] If inspect mode fails, diagnose and patch only the smallest study-local surface needed to restore the contract.
- [ ] After any narrow repair, rerun the required deterministic checks from Task 1 before rerunning inspect mode.
- [ ] Do not proceed to the expensive training run until inspect mode is green and the contract is proven from generated artifacts.

**Verification for Task 2**

- [ ] The inspect command exits `0`.
- [ ] The inspect root contains `invocation.json`, `dataset_manifest.json`, and `split_manifest.json`.
- [ ] The inspect root proves the intended `2048 / 256 / 256`, `history_len=2`, `max_windows_per_trajectory=8`, `batch_size=4`, `epochs=40` contract.
- [ ] Any code/test repair stays confined to the authorized study-local surfaces and has green rerun evidence.

### Task 3: Launch Exactly One Fresh `2048` Capped Run After Preflight Is Green

**Files:**
- Use: `scripts/studies/run_pdebench_image128_suite.py`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-<timestamp>/`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-<timestamp>.launch/`

- [ ] Choose one fresh timestamped run root and one matching launch-directory root. Do not reuse an output root that another process might still be writing.
- [ ] Launch the run in tmux with `ptycho311` active and explicit PID tracking. The shell wrapper must capture the exact child PID, wait on that PID, and record the exit code. Use a pattern equivalent to:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311

RUN_ROOT=.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-<timestamp>
LAUNCH_ROOT=.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-<timestamp>.launch

mkdir -p "$LAUNCH_ROOT"

python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root "$RUN_ROOT" \
  --profiles spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_shared_blocks10 \
  --history-len 2 \
  --epochs 40 \
  --batch-size 4 \
  --max-train-trajectories 2048 \
  --max-val-trajectories 256 \
  --max-test-trajectories 256 \
  --max-windows-per-trajectory 8 \
  --device cuda \
  --num-workers 0 \
  > "$LAUNCH_ROOT/train.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$LAUNCH_ROOT/pid.txt"
wait "$pid"
rc=$?
printf '%s\n' "$rc" > "$LAUNCH_ROOT/exit_code.txt"
exit "$rc"
```

- [ ] The expensive training run must wait for a green Task 1 and Task 2. Do not launch it earlier.
- [ ] Accept the run as complete only when:
  - the tracked PID exits with code `0`
  - the run root is freshly written
  - required artifacts exist for both profiles:
    - `invocation.json`
    - `dataset_manifest.json`
    - `split_manifest.json`
    - `comparison_summary.json`
    - `comparison_summary.csv`
    - `metrics_spectral_resnet_bottleneck_base.json`
    - `metrics_spectral_resnet_bottleneck_shared_blocks10.json`
    - `model_profile_spectral_resnet_bottleneck_base.json`
    - `model_profile_spectral_resnet_bottleneck_shared_blocks10.json`
- [ ] If a normal harness/import/runtime bug appears, diagnose and patch narrowly, rerun the required checks, and relaunch once with a new output root.
- [ ] If the run cannot proceed because the dataset is unavailable, the GPU/host is unavailable, or another unrecoverable external blocker remains after a narrow fix attempt, record `BLOCKED` with the exact blocker.

**Verification for Task 3**

- [ ] The launch metadata captures `pid.txt`, `exit_code.txt`, and `train.log`.
- [ ] `comparison_summary.json` reports `evidence_scope="capped_decision_support_only"` and `metric_interpretation="decision_support_not_benchmark_performance"`.
- [ ] Both finalist profiles completed under the fixed `2048 / 256 / 256`, `40`-epoch contract.
- [ ] The fresh run root contains every artifact required by `scripts/studies/pdebench_image128/reporting.py::_load_run_record`.

### Task 4: Write The Scaling Payload, Durable Summary, And State Sync

**Files:**
- Use: `scripts/studies/pdebench_image128/reporting.py`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/finalist_scaling_trend_512_1024_2048.json`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/finalist_scaling_trend_512_1024_2048.csv`
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_scaling_2048cap_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Modify: `docs/index.md`
- Modify only if needed: `docs/findings.md`
- Modify only if needed: `docs/studies/index.md`

- [ ] Generate the scaling payload via `write_split_cap_scaling_trend`, not by hand-editing JSON/CSV. Use a repo-root `python - <<'PY'` snippet equivalent to:

```bash
python - <<'PY'
from pathlib import Path
from scripts.studies.pdebench_image128.reporting import write_split_cap_scaling_trend

artifact_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap")
write_split_cap_scaling_trend(
    output_root=artifact_root,
    profile_ids=[
        "spectral_resnet_bottleneck_base",
        "spectral_resnet_bottleneck_shared_blocks10",
    ],
    reference_manifest_paths=[
        artifact_root / "reference_runs_512cap_40ep.json",
        artifact_root / "reference_runs_1024cap_40ep.json",
    ],
    fresh_run_root=artifact_root / "cns-hybrid-spectral-finalists-2048cap-40ep-<timestamp>",
    fresh_profile_ids=[
        "spectral_resnet_bottleneck_base",
        "spectral_resnet_bottleneck_shared_blocks10",
    ],
    fresh_source_document="docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_scaling_2048cap_summary.md",
)
PY
```

- [ ] Confirm the generated scaling payload includes:
  - `cap_sequence=["512cap","1024cap","2048cap"]`
  - absolute metrics for both profiles at all three caps
  - `delta_1024_minus_512`
  - `delta_2048_minus_1024`
  - runtime deltas for both transitions
  - improvement per added training trajectory
  - fixed-contract proof and `allowed_contract_delta.delta_kind="split_counts_only"`
- [ ] Write the durable summary at `pdebench_cns_hybrid_spectral_scaling_2048cap_summary.md` with:
  - item metadata and scope
  - the fixed fairness contract
  - the frozen reference roots and fresh `2048` run root
  - reference manifest paths
  - scaling JSON/CSV artifact paths
  - per-profile metrics across `512`, `1024`, and `2048`
  - `512 -> 1024` and `1024 -> 2048` deltas
  - runtime changes
  - a bounded interpretation of whether either finalist scales faster
  - the explicit claim boundary that this is capped decision-support evidence only and does not justify default-profile promotion by itself
  - verification log paths
- [ ] Sync `pdebench_2d_cfd_cns_summary.md` with a short subsection or note that links this new scaling tranche, records the fresh `2048` artifact root, and preserves the same capped-evidence boundary.
- [ ] Update `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` with a new entry for this backlog item that records:
  - `updated_at_utc`
  - `update_reason`
  - bounded decision text
  - `decision_scope`
  - `plan_path`
  - `artifact_root`
  - `reference_manifest_paths`
  - `fresh_run_root`
  - `summary_path`
  - `cns_summary_path`
  - `performance_assessment_complete=false`
  - `evidence_scope="capped_decision_support_only"`
  - `metric_interpretation="decision_support_not_benchmark_performance"`
  - the preserved comparison standard
  - scaling-outcome fields for both profiles
- [ ] Update `docs/index.md` so the new durable summary is discoverable from the docs hub.
- [ ] Update `docs/findings.md` only if the work uncovered a reusable rule that should constrain future PDEBench CNS work. Do not manufacture a finding from ordinary study results.
- [ ] Update `docs/studies/index.md` only if implementation changed a durable study/runbook surface that future workers must discover.

**Verification for Task 4**

- [ ] `finalist_scaling_trend_512_1024_2048.json` and `.csv` exist and parse.
- [ ] The scaling payload proves contract invariants and only permits split-count drift.
- [ ] The durable summary, CNS summary sync, progress-ledger entry, and docs index entry all point to the same artifact roots and interpretation boundary.
- [ ] If no reusable project-wide rule emerged, `docs/findings.md` remains unchanged.

## Final Completion Gate

- [ ] The exact required check commands from the selected backlog item passed and are archived:
  - `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py`
  - `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
- [ ] Any narrow blocker repair reran those required checks before the expensive run.
- [ ] One and only one fresh `2048 / 256 / 256` two-profile run was launched for this execution attempt.
- [ ] The scaling payload compares the frozen `512` and `1024` references against the fresh `2048` run without rerunning older caps.
- [ ] The final written interpretation answers only the selected backlog question:
  - whether one finalist keeps improving faster as the capped training set grows
  - without claiming benchmark completeness, default-profile promotion, or paper readiness
