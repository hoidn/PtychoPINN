# Shared-Blocks10 1024-Cap Longer-Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one fresh longer-budget `spectral_resnet_bottleneck_shared_blocks10` PDEBench CNS pilot at the fixed `1024 / 128 / 128` cap so the tranche can answer whether the prior `40`-epoch `1024cap` row was under-converged, without widening this item beyond capped decision-support evidence.

**Architecture:** Reuse the frozen `1024cap`, `40`-epoch finalist row from the completed CNS hybrid-spectral architecture ablation, freeze its full shell contract from the saved `model_profile_*.json`, then launch exactly one fresh `80`-epoch `pilot` rerun under the same dataset, split, history, loss, batch, and shell settings. Publish two reporting surfaces from that fixed contract: a convergence audit for the fresh run and a same-profile `40ep -> 80ep` delta payload that permits only the epoch budget to change, then sync the bounded interpretation into the durable CNS summaries and progress ledger.

**Tech Stack:** PATH `python`; tmux with `ptycho311` active for the long run; PyTorch; `scripts/studies/pdebench_image128/`; pytest; compileall; Markdown/JSON/CSV artifacts under `.artifacts/`.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-29-pdebench-cns-shared-blocks10-1024cap-longer-convergence`
- Selection mode: `RECOVERED_IN_PROGRESS`
- Date: `2026-04-28`
- Authoritative selected-item context:
  `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/22/items/2026-04-29-pdebench-cns-shared-blocks10-1024cap-longer-convergence/selected-item-context.md`
- Previous plan path used only as background:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-pdebench-cns-shared-blocks10-1024cap-longer-convergence/execution_plan.md`
- Durable summary target:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_shared_blocks10_1024cap_longer_convergence_summary.md`
- CNS summary sync target:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/`

This file is the new execution authority for the selected backlog item. It supersedes prior plan text at this path.

## Inputs Read

- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/steering.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `docs/backlog/in_progress/2026-04-29-pdebench-cns-shared-blocks10-1024cap-longer-convergence.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_scaling_2048cap_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/22/items/2026-04-29-pdebench-cns-shared-blocks10-1024cap-longer-convergence/selected-item-context.md`
- `.artifacts/review/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-pdebench-cns-shared-blocks10-1024cap-longer-convergence-plan-review.json`

## Review Findings Addressed In This Revision

- `PLAN-M1`: if this item requires changes to production workflow/config-plumbing files, closeout must rerun `pytest -v -m integration` and archive the resulting log under the item artifact root, in addition to the backlog item's required check commands.
- `PLAN-M2`: the item-local payload gate and the final closeout gate now use an explicit inline artifact-validation command that checks the real reporting payload schema, shell-contract match, durable-doc presence, and launch exit-code sidecar.

## Selected Objective

- Rerun `spectral_resnet_bottleneck_shared_blocks10` on PDEBench `2d_cfd_cns` with the fixed `1024 / 128 / 128` cap and a bounded longer `80`-epoch budget.
- Answer one narrow question:
  - does the more converged `shared_blocks10` row materially change the existing bounded interpretation that the shared base row remains the better aggregate local reference while `shared_blocks10` keeps only narrower `fRMSE_mid/high` advantages?
- Publish:
  - the frozen `40`-epoch reference manifest for the authoritative `1024cap` row
  - the frozen shell contract for `spectral_resnet_bottleneck_shared_blocks10`
  - one fresh `80`-epoch run root with launch proof
  - one fresh convergence audit based on the `80`-epoch train-loss trajectory
  - one shell-validated same-profile `40ep -> 80ep` delta payload
  - a durable summary and CNS summary sync that keep the claim explicitly capped and decision-support-only

## Scope

- Run only `spectral_resnet_bottleneck_shared_blocks10`.
- Keep the completed `1024cap` finalist contract fixed:
  - task: `2d_cfd_cns`
  - dataset:
    `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
  - split cap: `1024 / 128 / 128`
  - history contract: `history_len=2`, `concat u[t-2:t] -> u[t]`
  - `max_windows_per_trajectory=8`
  - training loss: `mse`
  - batch size: `4`
  - metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
  - run mode: `pilot`
  - fixed shell:
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
    - `spectral_bottleneck_share_weights=True`
    - `spectral_bottleneck_blocks=10`
- Freeze the shell from the authoritative saved model-profile artifact and require both the live repo profile and the fresh run's model-profile artifact to match it exactly.
- Lock the longer budget at `80` epochs for this item. Do not extend further within this tranche.
- Preserve capped decision-support scope. This item must not be presented as full-training benchmark evidence.

## Explicit Non-Goals

- Do not rerun `spectral_resnet_bottleneck_base` or any other profile.
- Do not rerun `512cap` or `2048cap` lanes.
- Do not change cap size, split family, history length, loss, batch size, or shell fields to make the run easier.
- Do not reinterpret this item as a same-budget architecture winner test; the base row is not rerun at `80` epochs here.
- Do not auto-extend past `80` epochs even if the loss curve is still dropping; report that bounded outcome.
- Do not expand to Darcy, the full PDEBench suite, external baselines, CDI work, or `/home/ollie/Documents/neurips/` artifact assembly.
- Do not touch `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not create worktrees.

## Steering, Roadmap, And Prerequisite Constraints

- Steering is binding:
  - keep equal-footing boundaries explicit
  - do not silently relax fairness constraints
  - if the bounded result remains mixed, record the mixed outcome instead of forcing a promotion
- The approved design and roadmap are binding:
  - this item remains Phase 2 PDEBench CNS work
  - it stays explicitly capped and decision-support-only
  - it does not satisfy the roadmap’s full available training-split benchmark gate
  - it must not expand to later roadmap phases or unrelated backlog items
- The roadmap’s current routing state is binding:
  - the immediate active CNS lane is this shared-blocks10 longer-convergence follow-up at `1024 / 128 / 128`
  - the outcome must be read against the completed `40`-epoch finalist row and the completed `2048cap` scaling summary
- Prerequisite status from the progress ledger and durable summaries:
  - `2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation` is complete and froze the authoritative `1024cap`, `40`-epoch finalist root:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z`
  - `2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap` is complete and established the bounded prior interpretation that the shared base row remains the stronger aggregate local reference beyond `1024cap`
- `REPORTING-ARTIFACT-BOUNDARY-001` applies:
  - required metrics, manifests, delta payloads, convergence payloads, and summary updates decide completion
  - optional gallery warnings may be recorded without invalidating the tranche if required artifacts are present
- `PYTHON-ENV-001` applies: invoke Python as plain PATH `python`
- Long-run guardrails are mandatory:
  - use tmux with `ptycho311` active
  - do not launch against an already-used `--output-root`
  - track the exact launched PID and wait on that PID
  - declare the run complete only when the tracked PID exits `0` and the required fresh artifacts exist
- Failure handling:
  - if pytest, compileall, inspect mode, import, or reporting-helper checks fail, diagnose/fix/rerun before considering any block
  - reserve `BLOCKED` for missing dataset access, unavailable hardware, roadmap conflict outside current authority, user decision required, or unrecoverable failure after a documented narrow fix attempt

## Frozen Reference Inputs

- Frozen `1024cap`, `40`-epoch finalist root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z`
- Frozen source summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`
- Completed `2048cap` interpretation source:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_scaling_2048cap_summary.md`
- Frozen model-shell artifact:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z/model_profile_spectral_resnet_bottleneck_shared_blocks10.json`
- Live profile-definition source that must still match the frozen shell:
  `scripts/studies/pdebench_image128/run_config.py::get_model_profile("spectral_resnet_bottleneck_shared_blocks10")`
- Existing reporting helpers that are part of the default execution path:
  - `scripts/studies/pdebench_image128.reporting::build_reference_run_manifest`
  - `scripts/studies/pdebench_image128.reporting::write_reference_run_manifest`
  - `scripts/studies/pdebench_image128.reporting::write_cfd_cns_convergence_audit`
  - `scripts/studies/pdebench_image128.reporting::write_same_profile_epoch_budget_delta`

## Implementation Architecture

- Contract-freeze unit:
  - create the item-local reference manifest and shell-contract artifact from the frozen `40`-epoch row and verify the current repo profile still matches before any expensive work
- Inspect-and-launch unit:
  - run the required deterministic checks, prove the `1024 / 128 / 128`, `history_len=2`, `mse`, batch-`4`, `80`-epoch pilot contract through `inspect` mode, then launch exactly one fresh run under tmux
- Reporting-and-sync unit:
  - generate the convergence audit and shell-validated `40ep -> 80ep` delta payload, then update the durable summary, CNS summary, progress ledger, and docs index

## Concrete File And Artifact Targets

- Study code and tests that may change only if preflight exposes a real blocker:
  - `scripts/studies/pdebench_image128/reporting.py`
  - `scripts/studies/pdebench_image128/run_config.py`
  - `scripts/studies/pdebench_image128/cfd_cns.py`
  - `tests/studies/test_pdebench_image128_models.py`
  - `tests/studies/test_pdebench_image128_runner.py`
- Item-local artifacts to create:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/reference_runs_1024cap_40ep.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/reference_shell_contract_shared_blocks10_1024cap.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/inspect-1024cap-80ep-<timestamp>/`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/cns-shared-blocks10-1024cap-80ep-<timestamp>/`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/cns-shared-blocks10-1024cap-80ep-<timestamp>.launch/`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/convergence_audit.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/convergence_audit.csv`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/shared_blocks10_1024cap_40ep_vs_80ep.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/shared_blocks10_1024cap_40ep_vs_80ep.csv`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/verification/*.log`
- Durable docs and state to create or update:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_shared_blocks10_1024cap_longer_convergence_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
  - `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
  - `docs/index.md`
  - `docs/findings.md` only if a reusable engineering rule is discovered during execution

### Task 1: Freeze The Reference Contract And Clear The Deterministic Gates

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/reference_runs_1024cap_40ep.json`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/reference_shell_contract_shared_blocks10_1024cap.json`
- Modify only if a preflight failure exposes real drift: `scripts/studies/pdebench_image128/reporting.py`
- Modify only if a preflight failure exposes real drift: `scripts/studies/pdebench_image128/run_config.py`
- Test: `tests/studies/test_pdebench_image128_models.py`
- Test: `tests/studies/test_pdebench_image128_runner.py`

- [ ] Create the item artifact root and `verification/` subdirectory.
- [ ] Run the backlog item’s required deterministic checks and archive both logs under the item artifact root before inspect or training:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py \
  2>&1 | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/verification/preflight_pytest.log

python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py \
  2>&1 | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/verification/preflight_compileall.log
```

- [ ] If either required check fails, diagnose and fix the narrow cause first, then rerun both commands to green before any expensive run. Do not label the item `BLOCKED` for an ordinary local verification failure.
- [ ] Write the item-local frozen reference manifest for the authoritative `40`-epoch row using the existing reporting helper, not hand-edited JSON:

```bash
python - <<'PY'
from pathlib import Path
from scripts.studies.pdebench_image128.reporting import (
    build_reference_run_manifest,
    write_reference_run_manifest,
)

artifact_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence")
payload = build_reference_run_manifest(
    task_id="2d_cfd_cns",
    dataset_file="/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5",
    split_counts={"train": 1024, "val": 128, "test": 128},
    max_windows_per_trajectory=8,
    history_len=2,
    training_loss="mse",
    batch_size=4,
    metric_family=["err_RMSE", "err_nRMSE", "relative_l2", "fRMSE_low", "fRMSE_mid", "fRMSE_high"],
    required_rows={
        "40ep": [
            {
                "run_root": ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z",
                "profile_id": "spectral_resnet_bottleneck_shared_blocks10",
                "epochs": 40,
                "source_document": "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md",
            }
        ]
    },
)
write_reference_run_manifest(payload, artifact_root / "reference_runs_1024cap_40ep.json")
PY
```

- [ ] Write the item-local frozen shell contract from the frozen `model_profile_spectral_resnet_bottleneck_shared_blocks10.json`, then compare it against the live repo profile before any inspect or training launch:

```bash
python - <<'PY' \
  2>&1 | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/verification/frozen_shell_contract.log
from pathlib import Path
import json
from scripts.studies.pdebench_image128.run_config import get_model_profile

artifact_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence")
frozen_model_profile_path = Path(
    ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/"
    "cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z/"
    "model_profile_spectral_resnet_bottleneck_shared_blocks10.json"
)
frozen = json.loads(frozen_model_profile_path.read_text())
shell_contract = {
    "schema_version": "pdebench_image128_shell_contract_v1",
    "profile_id": frozen["profile_id"],
    "base_model": frozen["base_model"],
    "profile_config": frozen["profile_config"],
    "parameter_count": frozen["parameter_count"],
    "source_run_root": (
        ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/"
        "cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z"
    ),
    "source_document": "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md",
}
live = get_model_profile("spectral_resnet_bottleneck_shared_blocks10")
if live.base_model != shell_contract["base_model"]:
    raise SystemExit(f"live base_model drifted: {live.base_model!r} != {shell_contract['base_model']!r}")
if live.to_model_config() != shell_contract["profile_config"]:
    raise SystemExit("live profile_config drifted from the frozen 1024cap shell contract")
path = artifact_root / "reference_shell_contract_shared_blocks10_1024cap.json"
path.write_text(json.dumps(shell_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("frozen shell contract recorded and live profile matches")
PY
```

- [ ] Verify the two frozen artifacts exist and that the frozen reference row still records `epochs=40`, `split_counts=1024/128/128`, `history_len=2`, `training_loss=mse`, and `profile_id=spectral_resnet_bottleneck_shared_blocks10`.

### Task 2: Prove The Fresh 80-Epoch Contract In Inspect Mode Before Training

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/inspect-1024cap-80ep-<timestamp>/`
- Modify only if inspect exposes a real runner bug: `scripts/studies/pdebench_image128/cfd_cns.py`
- Test only if code changes: `tests/studies/test_pdebench_image128_runner.py`

- [ ] Generate a unique inspect root and run the exact capped contract through `inspect` mode. Training must not launch until this inspect step is green:

```bash
python -m scripts.studies.pdebench_image128.cfd_cns \
  --task 2d_cfd_cns \
  --mode inspect \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/inspect-1024cap-80ep-<timestamp> \
  --profiles spectral_resnet_bottleneck_shared_blocks10 \
  --history-len 2 \
  --epochs 80 \
  --batch-size 4 \
  --max-train-trajectories 1024 \
  --max-val-trajectories 128 \
  --max-test-trajectories 128 \
  --max-windows-per-trajectory 8 \
  --device cuda \
  2>&1 | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/verification/inspect.log
```

- [ ] Confirm the inspect root contains `invocation.json`, `dataset_manifest.json`, `split_manifest.json`, `hdf5_metadata.json`, and `invocation.sh`.
- [ ] Validate from those inspect artifacts that the fresh run contract still fixes:
  - `mode="inspect"`
  - `task_id="2d_cfd_cns"`
  - `profiles=["spectral_resnet_bottleneck_shared_blocks10"]`
  - `epochs=80`
  - `batch_size=4`
  - `history_len=2`
  - `max_train_trajectories=1024`
  - `max_val_trajectories=128`
  - `max_test_trajectories=128`
  - `max_windows_per_trajectory=8`
- [ ] If inspect exposes a runner or contract bug, fix it narrowly, rerun the required deterministic checks from Task 1, and rerun inspect to green before training. Do not bypass inspect.

### Task 3: Launch Exactly One Fresh 80-Epoch Pilot Run With PID-Tracked Completion Proof

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/cns-shared-blocks10-1024cap-80ep-<timestamp>/`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/cns-shared-blocks10-1024cap-80ep-<timestamp>.launch/`

- [ ] Generate a unique fresh run root and matching `.launch/` directory. Refuse to launch if either path already exists or another process is already writing to that `--output-root`.
- [ ] Start the long run in tmux with `ptycho311` active and PID tracking that writes an exit-code sidecar. Use `pilot` mode so the artifacts are labeled `capped_decision_support_only` rather than readiness-only:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
python -m scripts.studies.pdebench_image128.cfd_cns \
  --task 2d_cfd_cns \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/cns-shared-blocks10-1024cap-80ep-<timestamp> \
  --profiles spectral_resnet_bottleneck_shared_blocks10 \
  --history-len 2 \
  --epochs 80 \
  --batch-size 4 \
  --max-train-trajectories 1024 \
  --max-val-trajectories 128 \
  --max-test-trajectories 128 \
  --max-windows-per-trajectory 8 \
  --device cuda
```

- [ ] In the tmux pane, wrap the launch so the exact PID is captured and awaited, then persist the exit code to:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/cns-shared-blocks10-1024cap-80ep-<timestamp>.launch/exit_code.txt`
- [ ] Consider the run complete only when:
  - the tracked PID exits with code `0`
  - the fresh run root contains `comparison_summary.json`
  - the fresh run root contains `metrics_spectral_resnet_bottleneck_shared_blocks10.json`
  - the fresh run root contains `model_profile_spectral_resnet_bottleneck_shared_blocks10.json`
  - the fresh metrics payload records exactly `80` `train_epoch_losses`
- [ ] If the process exits nonzero or required artifacts are missing/stale, diagnose and retry within scope before considering a block.

### Task 4: Generate The Reporting Payloads And Sync Durable State

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/convergence_audit.json`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/convergence_audit.csv`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/shared_blocks10_1024cap_40ep_vs_80ep.json`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/shared_blocks10_1024cap_40ep_vs_80ep.csv`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_shared_blocks10_1024cap_longer_convergence_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Modify: `docs/index.md`
- Modify only if a reusable engineering rule emerged: `docs/findings.md`

- [ ] Generate the convergence audit from the fresh run and require the full `80`-epoch loss history:

```bash
python - <<'PY' \
  2>&1 | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/verification/convergence_audit.log
from pathlib import Path
from scripts.studies.pdebench_image128.reporting import write_cfd_cns_convergence_audit

artifact_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence")
fresh_run_root = artifact_root / "cns-shared-blocks10-1024cap-80ep-<timestamp>"
write_cfd_cns_convergence_audit(
    output_root=artifact_root,
    run_root=fresh_run_root,
    profile_id="spectral_resnet_bottleneck_shared_blocks10",
    expected_loss_count=80,
)
PY
```

- [ ] Generate the same-profile epoch-budget delta payload using the existing shell-validating helper:

```bash
python - <<'PY' \
  2>&1 | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/verification/epoch_budget_delta.log
from pathlib import Path
from scripts.studies.pdebench_image128.reporting import write_same_profile_epoch_budget_delta

artifact_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence")
write_same_profile_epoch_budget_delta(
    output_root=artifact_root,
    reference_manifest_path=artifact_root / "reference_runs_1024cap_40ep.json",
    fresh_run_root=artifact_root / "cns-shared-blocks10-1024cap-80ep-<timestamp>",
    profile_id="spectral_resnet_bottleneck_shared_blocks10",
    fresh_source_document="docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_shared_blocks10_1024cap_longer_convergence_summary.md",
    shell_contract_path=artifact_root / "reference_shell_contract_shared_blocks10_1024cap.json",
)
PY
```

- [ ] Validate the payloads with an item-local artifact check that confirms:
  - convergence audit JSON and CSV exist
  - delta JSON and CSV exist
  - the fresh run profile still matches the frozen shell contract
  - the delta payload records `allowed_contract_delta.delta_kind="epochs_only"`
  - the delta payload records `reference_epochs=40` and `fresh_epochs=80`
  - the delta payload keeps `evidence_scope="capped_decision_support_only"` and `metric_interpretation="decision_support_not_benchmark_performance"`
  - the convergence audit reflects an `80`-entry `train_epoch_losses` series

```bash
python - <<'PY' \
  2>&1 | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/verification/payload_validation.log
from pathlib import Path
import json

artifact_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence")
fresh_run_root = artifact_root / "cns-shared-blocks10-1024cap-80ep-<timestamp>"
required_paths = [
    artifact_root / "convergence_audit.json",
    artifact_root / "convergence_audit.csv",
    artifact_root / "shared_blocks10_1024cap_40ep_vs_80ep.json",
    artifact_root / "shared_blocks10_1024cap_40ep_vs_80ep.csv",
    artifact_root / "reference_shell_contract_shared_blocks10_1024cap.json",
    artifact_root / "reference_runs_1024cap_40ep.json",
    artifact_root / "cns-shared-blocks10-1024cap-80ep-<timestamp>.launch" / "exit_code.txt",
    fresh_run_root / "model_profile_spectral_resnet_bottleneck_shared_blocks10.json",
    fresh_run_root / "metrics_spectral_resnet_bottleneck_shared_blocks10.json",
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_shared_blocks10_1024cap_longer_convergence_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json"),
    Path("docs/index.md"),
]
missing = [str(path) for path in required_paths if not path.exists()]
if missing:
    raise SystemExit(f"missing required artifact(s): {missing}")

exit_code = (artifact_root / "cns-shared-blocks10-1024cap-80ep-<timestamp>.launch" / "exit_code.txt").read_text(encoding="utf-8").strip()
if exit_code != "0":
    raise SystemExit(f"expected launch exit code 0, found {exit_code!r}")

shell_contract = json.loads((artifact_root / "reference_shell_contract_shared_blocks10_1024cap.json").read_text(encoding="utf-8"))
fresh_model_profile = json.loads((fresh_run_root / "model_profile_spectral_resnet_bottleneck_shared_blocks10.json").read_text(encoding="utf-8"))
for field in ["profile_id", "base_model", "profile_config", "parameter_count"]:
    if fresh_model_profile.get(field) != shell_contract.get(field):
        raise SystemExit(f"fresh model profile field {field!r} drifted from frozen shell contract")

delta_payload = json.loads((artifact_root / "shared_blocks10_1024cap_40ep_vs_80ep.json").read_text(encoding="utf-8"))
allowed_delta = delta_payload.get("allowed_contract_delta", {})
if allowed_delta.get("delta_kind") != "epochs_only":
    raise SystemExit(f"expected epochs_only delta, found {allowed_delta.get('delta_kind')!r}")
if int(allowed_delta.get("reference_epochs", -1)) != 40:
    raise SystemExit(f"expected reference_epochs=40, found {allowed_delta.get('reference_epochs')!r}")
if int(allowed_delta.get("fresh_epochs", -1)) != 80:
    raise SystemExit(f"expected fresh_epochs=80, found {allowed_delta.get('fresh_epochs')!r}")
if delta_payload.get("evidence_scope") != "capped_decision_support_only":
    raise SystemExit(f"unexpected evidence_scope: {delta_payload.get('evidence_scope')!r}")
if delta_payload.get("metric_interpretation") != "decision_support_not_benchmark_performance":
    raise SystemExit(
        f"unexpected metric_interpretation: {delta_payload.get('metric_interpretation')!r}"
    )

convergence_payload = json.loads((artifact_root / "convergence_audit.json").read_text(encoding="utf-8"))
profiles = convergence_payload.get("profiles", [])
if convergence_payload.get("profile_ids") != ["spectral_resnet_bottleneck_shared_blocks10"]:
    raise SystemExit(f"unexpected convergence profile_ids: {convergence_payload.get('profile_ids')!r}")
if len(profiles) != 1:
    raise SystemExit(f"expected exactly one convergence profile row, found {len(profiles)}")
if int(profiles[0].get("loss_count", -1)) != 80:
    raise SystemExit(f"expected loss_count=80, found {profiles[0].get('loss_count')!r}")
if convergence_payload.get("evidence_scope") != "capped_decision_support_only":
    raise SystemExit(f"unexpected convergence evidence_scope: {convergence_payload.get('evidence_scope')!r}")
if convergence_payload.get("metric_interpretation") != "decision_support_not_benchmark_performance":
    raise SystemExit(
        f"unexpected convergence metric_interpretation: {convergence_payload.get('metric_interpretation')!r}"
    )

print("payload validation passed")
PY
```

- [ ] Write the durable summary with:
  - the fixed fairness contract
  - frozen references and fresh run root
  - the fresh `80`-epoch train-loss trajectory
  - the fresh held-out eval metrics
  - the shell-validated `40ep -> 80ep` metric deltas
  - the bounded interpretation against the frozen `40`-epoch row and the completed `2048cap` scaling summary
  - explicit claim boundaries that this remains capped decision-support evidence only
- [ ] Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md` with a concise note that this capped longer-budget follow-up was completed and how it changed or preserved the existing local interpretation.
- [ ] Update `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` with the item completion record, including:
  - plan path
  - artifact root
  - frozen reference manifest path
  - shell-contract path
  - inspect proof root
  - fresh run root
  - convergence-audit paths
  - same-profile epoch-budget delta paths
  - durable summary path
  - verification log paths
  - final bounded decision statement
- [ ] Update `docs/index.md` so the new durable summary is discoverable.
- [ ] Update `docs/findings.md` only if the work uncovered a reusable engineering rule rather than a one-off study outcome.

## Final Verification

- [ ] If any production code changed during Tasks 1-4, rerun the backlog item’s required deterministic checks and archive fresh final logs:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py \
  2>&1 | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/verification/final_pytest.log

python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py \
  2>&1 | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/verification/final_compileall.log
```

- [ ] If any production workflow/config-plumbing file changed during Tasks 1-4 (`scripts/studies/pdebench_image128/reporting.py`, `scripts/studies/pdebench_image128/run_config.py`, `scripts/studies/pdebench_image128/cfd_cns.py`, or `scripts/studies/run_pdebench_image128_suite.py`), rerun the required integration marker and archive the log per `docs/TESTING_GUIDE.md`:

```bash
pytest -v -m integration \
  2>&1 | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/verification/final_integration.log
```

- [ ] Run the explicit final artifact-validation command and archive its log at
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/verification/final_artifact_validation.log`:

```bash
python - <<'PY' \
  2>&1 | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/verification/final_artifact_validation.log
from pathlib import Path
import json

artifact_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence")
fresh_run_root = artifact_root / "cns-shared-blocks10-1024cap-80ep-<timestamp>"
required_paths = [
    artifact_root / "convergence_audit.json",
    artifact_root / "convergence_audit.csv",
    artifact_root / "shared_blocks10_1024cap_40ep_vs_80ep.json",
    artifact_root / "shared_blocks10_1024cap_40ep_vs_80ep.csv",
    artifact_root / "reference_shell_contract_shared_blocks10_1024cap.json",
    artifact_root / "reference_runs_1024cap_40ep.json",
    artifact_root / "cns-shared-blocks10-1024cap-80ep-<timestamp>.launch" / "exit_code.txt",
    fresh_run_root / "model_profile_spectral_resnet_bottleneck_shared_blocks10.json",
    fresh_run_root / "metrics_spectral_resnet_bottleneck_shared_blocks10.json",
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_shared_blocks10_1024cap_longer_convergence_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json"),
    Path("docs/index.md"),
]
missing = [str(path) for path in required_paths if not path.exists()]
if missing:
    raise SystemExit(f"missing required artifact(s): {missing}")

exit_code = (artifact_root / "cns-shared-blocks10-1024cap-80ep-<timestamp>.launch" / "exit_code.txt").read_text(encoding="utf-8").strip()
if exit_code != "0":
    raise SystemExit(f"expected launch exit code 0, found {exit_code!r}")

shell_contract = json.loads((artifact_root / "reference_shell_contract_shared_blocks10_1024cap.json").read_text(encoding="utf-8"))
fresh_model_profile = json.loads((fresh_run_root / "model_profile_spectral_resnet_bottleneck_shared_blocks10.json").read_text(encoding="utf-8"))
for field in ["profile_id", "base_model", "profile_config", "parameter_count"]:
    if fresh_model_profile.get(field) != shell_contract.get(field):
        raise SystemExit(f"fresh model profile field {field!r} drifted from frozen shell contract")

delta_payload = json.loads((artifact_root / "shared_blocks10_1024cap_40ep_vs_80ep.json").read_text(encoding="utf-8"))
allowed_delta = delta_payload.get("allowed_contract_delta", {})
if allowed_delta.get("delta_kind") != "epochs_only":
    raise SystemExit(f"expected epochs_only delta, found {allowed_delta.get('delta_kind')!r}")
if int(allowed_delta.get("reference_epochs", -1)) != 40:
    raise SystemExit(f"expected reference_epochs=40, found {allowed_delta.get('reference_epochs')!r}")
if int(allowed_delta.get("fresh_epochs", -1)) != 80:
    raise SystemExit(f"expected fresh_epochs=80, found {allowed_delta.get('fresh_epochs')!r}")
if delta_payload.get("evidence_scope") != "capped_decision_support_only":
    raise SystemExit(f"unexpected evidence_scope: {delta_payload.get('evidence_scope')!r}")
if delta_payload.get("metric_interpretation") != "decision_support_not_benchmark_performance":
    raise SystemExit(
        f"unexpected metric_interpretation: {delta_payload.get('metric_interpretation')!r}"
    )

convergence_payload = json.loads((artifact_root / "convergence_audit.json").read_text(encoding="utf-8"))
profiles = convergence_payload.get("profiles", [])
if convergence_payload.get("profile_ids") != ["spectral_resnet_bottleneck_shared_blocks10"]:
    raise SystemExit(f"unexpected convergence profile_ids: {convergence_payload.get('profile_ids')!r}")
if len(profiles) != 1:
    raise SystemExit(f"expected exactly one convergence profile row, found {len(profiles)}")
if int(profiles[0].get("loss_count", -1)) != 80:
    raise SystemExit(f"expected loss_count=80, found {profiles[0].get('loss_count')!r}")
if convergence_payload.get("evidence_scope") != "capped_decision_support_only":
    raise SystemExit(f"unexpected convergence evidence_scope: {convergence_payload.get('evidence_scope')!r}")
if convergence_payload.get("metric_interpretation") != "decision_support_not_benchmark_performance":
    raise SystemExit(
        f"unexpected convergence metric_interpretation: {convergence_payload.get('metric_interpretation')!r}"
    )

print("final artifact validation passed")
PY
```
- [ ] Confirm the launch sidecar `exit_code.txt` records `0` and the final artifact-validation log confirms all required item-local outputs and durable docs exist.

## Completion Criteria

- [ ] The fresh `80`-epoch run exists and is completion-proven by the tracked PID exit code plus fresh required artifacts.
- [ ] The item-local reference manifest and shell contract were generated from the frozen `40`-epoch row and the live repo profile matched before launch.
- [ ] The convergence audit and same-profile `40ep -> 80ep` delta payload both exist and validate the shell contract.
- [ ] The durable summary, CNS summary sync, progress ledger, and docs index are updated.
- [ ] The final interpretation stays explicitly capped and decision-support-only and states whether the longer budget changed the prior aggregate-vs-frequency read.
