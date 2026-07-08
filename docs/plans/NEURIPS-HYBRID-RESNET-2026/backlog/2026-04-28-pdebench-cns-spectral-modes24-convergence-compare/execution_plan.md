# PDEBench CNS Spectral Modes-24 Convergence Compare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a fresh same-contract PDEBench `2d_cfd_cns` comparison between shared spectral `12/12` and `24/24` mode rows at a convergence-oriented capped budget, so the result separates spectral-mode value from obvious under-convergence.

**Architecture:** Treat this as a bounded Roadmap Phase 2 CNS follow-up with three units: first prove the exact capped contract and repo surfaces are green, then run one fresh paired long-budget `pilot` job for `spectral_resnet_bottleneck_base` and `spectral_resnet_bottleneck_modes24` at one shared resolved batch size, then publish a convergence audit plus a durable capped-evidence summary without widening scope. Reuse the existing CNS runner, the already-landed manual `spectral_resnet_bottleneck_modes24` profile, and `write_cfd_cns_convergence_audit(...)`; only patch code if a deterministic execution or reporting gap blocks the planned run.

**Tech Stack:** PATH `python`, `pytest`, `compileall`, PyTorch CNS runner under the documented `ptycho311` environment for long CUDA runs, tmux with exact PID tracking for long jobs, repo-local Markdown/JSON/CSV artifacts, and `.artifacts/NEURIPS-HYBRID-RESNET-2026/` evidence storage.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-28-pdebench-cns-spectral-modes24-convergence-compare`
- Selection mode: `RECOVERED_IN_PROGRESS`
- Date: `2026-04-28`
- Roadmap lane: bounded Phase 2 PDEBench CNS comparison follow-up
- Authoritative plan path: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-28-pdebench-cns-spectral-modes24-convergence-compare/execution_plan.md`
- Durable summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes24_convergence_summary.md`
- Item artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/`
- This document supersedes any earlier plan for this backlog item and is the execution authority for the implementation phase.

## Selected Backlog Objective

- Revisit the shared spectral CNS mode count under a convergence-oriented budget by comparing exactly two rows:
  - baseline: `spectral_resnet_bottleneck_base` with `fno_modes=12`, `spectral_bottleneck_modes=12`
  - candidate: `spectral_resnet_bottleneck_modes24` with `fno_modes=24`, `spectral_bottleneck_modes=24`
- Keep the local PDEBench `2d_cfd_cns` contract fixed on the official `128x128` file and the capped `1024 / 128 / 128`, `history_len=2`, `max_windows_per_trajectory=8`, `mse` slice.
- Use batch size `16` for both rows if feasible. If `16` fails for a concrete runtime-capacity reason, fall back to the largest smaller identical batch size that allows both rows to complete and record the exact failure and fallback evidence.
- Rerun both rows fresh under one selected long budget. Do not reuse the earlier `40`-epoch shared `12/12` row as the final baseline for this item.
- Report the train-loss trajectory, late-window convergence diagnostics, final eval metrics, and whether either row is still materially improving at stop time.

## Scope

### In Scope

- verify that the already-landed `spectral_resnet_bottleneck_modes24` profile and the CNS convergence-audit path still match the intended contract
- prove the exact `1024 / 128 / 128`, `history_len=2`, `max_windows_per_trajectory=8`, `mse` contract in inspect mode before any expensive run
- determine the largest passing identical batch size, starting at `16`
- run one fresh paired long-budget `pilot` job containing only `spectral_resnet_bottleneck_base` and `spectral_resnet_bottleneck_modes24`
- emit an item-local convergence audit from the final run's `train_epoch_losses`
- write a durable capped-evidence summary and sync only the discoverability docs needed to make the result findable with the correct claim boundary

### Explicit Non-Goals

- Do not widen this into a full mode sweep.
- Do not add `16`, `32`, decoupled encoder-only, or bottleneck-only mode settings.
- Do not mix in `spectral_resnet_bottleneck_noshare`, deeper shared bottlenecks, FFNO, GNOT, CDI/ptycho work, history-length changes, `2048` training-size scaling, or physics regularization.
- Do not relax the fixed task, split, metric, or training-loss contract for convenience.
- Do not convert this capped item into a full-training benchmark claim.
- Do not create `/home/ollie/Documents/neurips/` artifacts or manuscript prose.
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not create worktrees.

## Binding Constraints

### Steering And Roadmap Constraints

- Preserve the approved NeurIPS roadmap gate: this is still Roadmap Phase 2 capped CNS decision-support work, not Phase 3 CDI or any benchmark-complete PDE claim.
- Keep equal-footing comparisons explicit and preserve the fixed metric, split, and protocol boundaries from the design, roadmap, steering document, and backlog item.
- Do not silently relax fairness constraints to make the item easier. If the compare cannot be kept on equal footing, record the incompatibly constrained outcome instead of drifting the protocol.
- The roadmap's full-training-split and baseline-recipe rules apply to benchmark-performance rows. This item is intentionally capped and uses the local CNS `mse` contract, so implementation must preserve that bounded follow-up scope rather than silently converting it into a benchmark-style rerun.
- Because the steering document prefers work that strengthens core comparison evidence while required roadmap evidence is still active, this recovered in-progress item is allowed only as a bounded compare-quality follow-up and must not expand into optional side work.

### Fixed Local Compare Contract

- task: `2d_cfd_cns`
- dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- split: `1024 / 128 / 128` trajectories
- `history_len=2`
- `max_windows_per_trajectory=8`
- training loss: `mse`
- batch size target: `16`, otherwise largest smaller identical fallback for both rows
- metrics: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- fixed shell:
  - `base_model="spectral_resnet_bottleneck_net"`
  - `hidden_channels=32`
  - `fno_blocks=4`
  - `hybrid_downsample_steps=2`
  - `hybrid_resnet_blocks=6`
  - `hybrid_skip_connections=True`
  - `hybrid_skip_style="add"`
  - `hybrid_upsampler="pixelshuffle"`
  - `spectral_bottleneck_blocks=6`
  - `spectral_bottleneck_share_weights=True`
  - `spectral_bottleneck_gate_init=0.1`
  - `spectral_bottleneck_gate_mode="shared"`
- the only allowed config change is the coupled mode increase:
  - baseline: `fno_modes=12`, `spectral_bottleneck_modes=12`
  - candidate: `fno_modes=24`, `spectral_bottleneck_modes=24`

### Progress-Ledger Prerequisite Status

- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` shows Phase 0 and Phase 1 complete and no blocked tranches, so this bounded Phase 2 item can proceed without a roadmap pivot.
- The prerequisite backlog item `2026-04-21-pdebench-cns-spectral-modes32-compare` is complete and provides the motivation for a longer shared budget instead of a broader new search.
- The repo already contains the key surfaces this item expects:
  - `spectral_resnet_bottleneck_modes24` is defined in `scripts/studies/pdebench_image128/run_config.py`
  - the `2d_cfd_cns` runner already writes `dataset_manifest.json`, `split_manifest.json`, per-profile metrics JSON, and `comparison_summary.json/.csv`
  - `write_cfd_cns_convergence_audit(...)` exists in `scripts/studies/pdebench_image128/reporting.py` and currently expects `80` loss values by default
- Because those pieces are already landed, implementation should prefer execution and narrow gap-fixing over speculative refactors.

### Execution Rules

- The expensive paired long run must wait until all deterministic checks, the exact-contract inspect gate, and the batch-size feasibility gate are green.
- If a normal verification, import, path, environment, or harness failure occurs, implementation must diagnose, apply the narrowest credible fix, and rerun the same check before considering the item blocked.
- Reserve `BLOCKED` for missing data, unavailable GPU/runtime resources, roadmap conflict, user decision required, external dependency outside current authority, or a failure that remains unrecoverable after a documented narrow fix attempt.
- For long runs, use tmux, activate `ptycho311`, track the exact launched PID, and do not launch a duplicate run writing to the same `--output-root`.

## Convergence Standard

Define this rule before launching the long run and do not change it mid-item unless this plan is explicitly revised.

### Motivation

- The earlier `40`-epoch `12/12` and `32/32` spectral trajectories were still dropping materially, so that stop point was too early for a fair spectral-mode judgment.
- This item exists to test whether a less aggressive `24/24` increase remains worthwhile once both compared rows have a longer shared budget.

### Fixed Stop Budget

- Long paired run budget: `80` epochs for both rows in one fresh `pilot` run.
- This remains capped decision-support evidence, not a benchmark row, but it is long enough to make the existing convergence-audit helper meaningful without changing its default `expected_loss_count`.

### Per-Row Audit Rule

For each profile, compute from the emitted `train_epoch_losses`:

- `late_window_mean_prev = mean(losses[60:70])`
- `late_window_mean_final = mean(losses[70:80])`
- `late_window_ratio = late_window_mean_final / late_window_mean_prev`
- `last5_delta = losses[79] - losses[74]`
- `final_eval_metrics` from the final metrics JSON

Mark a row as **still materially improving** at the stop point if either condition is true:

- `late_window_ratio < 0.95`
- `last5_delta <= -0.001`

Interpretation rule:

- if both rows are still materially improving at `80` epochs, the conclusion must be **inconclusive**
- if only one row is still materially improving, record that asymmetry explicitly and do not present the flatter row as a clean win over the improving row
- if neither row is still materially improving, compare final eval metrics directly and describe any aggregate-versus-frequency tradeoff without inventing a broader promotion claim

## Implementation Architecture

- **Contract And Readiness Unit:** verify the repo surfaces, deterministic checks, exact capped CNS split contract, and batch-size feasibility before any long run. Patch only the narrow files needed if an execution or reporting gap blocks those gates.
- **Execution Unit:** run one fresh paired `80`-epoch `pilot` job for the two spectral rows on the same exact contract and one shared resolved batch size, in tmux, with exact PID tracking and no duplicate output-root launches.
- **Reporting Unit:** emit convergence-audit JSON/CSV plus the durable summary, then sync only the summary/index surfaces needed to keep the capped result discoverable and correctly bounded.

## Concrete File And Artifact Targets

### Code And Tests That May Change Only If A Real Gap Is Found

- `scripts/studies/pdebench_image128/run_config.py`
- `scripts/studies/pdebench_image128/reporting.py`
- `scripts/studies/pdebench_image128/cfd_cns.py`
- `scripts/studies/run_pdebench_image128_suite.py`
- `tests/studies/test_pdebench_image128_models.py`
- `tests/studies/test_pdebench_image128_runner.py`

### Expected Artifact Outputs

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/verification/pytest_preflight.log`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/verification/compileall_preflight.log`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/inspect-b16-<timestamp>/`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/batchsize-probe-b<candidate>-<timestamp>/`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/resolved_batch_size.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/cns-spectral-modes24-80ep-<timestamp>/`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/convergence_audit.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/convergence_audit.csv`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/verification/final_artifact_validation.log`
- if code changes are made, add `pytest_postfix.log`, `compileall_postfix.log`, and `integration_postfix.log` under the same verification root

### Durable Docs And Discoverability Surfaces

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes24_convergence_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- `docs/studies/index.md`
- `docs/index.md`

## Task Checklist

### Task 1: Prove The Contract And Deterministic Checks Are Green

**Files:**

- Read/verify: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- Read/verify: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- Read/verify: `docs/steering.md`
- Read/verify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Verify/patch only if needed: `scripts/studies/pdebench_image128/run_config.py`
- Verify/patch only if needed: `scripts/studies/pdebench_image128/reporting.py`
- Test: `tests/studies/test_pdebench_image128_models.py`
- Test: `tests/studies/test_pdebench_image128_runner.py`

- [ ] Reconfirm that `spectral_resnet_bottleneck_modes24` changes only the two mode knobs and remains manual-only.
- [ ] Run the backlog item's required deterministic checks and archive the logs:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/verification/pytest_preflight.log
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py 2>&1 | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/verification/compileall_preflight.log
```

- [ ] If either required check fails, make the narrowest credible fix, extend the targeted tests only as needed, and rerun the same required checks before proceeding.
- [ ] Run an exact-contract inspect gate for the intended long-run surface:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode inspect \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/inspect-b16-<timestamp> \
  --profiles spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_modes24 \
  --history-len 2 \
  --epochs 80 \
  --batch-size 16 \
  --max-train-trajectories 1024 \
  --max-val-trajectories 128 \
  --max-test-trajectories 128 \
  --max-windows-per-trajectory 8 \
  --device cuda \
  --num-workers 0
```

- [ ] Verify the inspect output contains the correct dataset manifest, split manifest, `history_len=2`, and capped `1024 / 128 / 128` trajectory counts.
- [ ] Do not launch any expensive run until the required checks and the inspect gate are green.

### Task 2: Resolve One Shared Batch Size For Both Rows

**Files:**

- Verify/patch only if needed: `scripts/studies/pdebench_image128/cfd_cns.py`
- Verify/patch only if needed: `scripts/studies/run_pdebench_image128_suite.py`
- Verify/patch only if needed: `tests/studies/test_pdebench_image128_runner.py`

- [ ] Probe batch size `16` first using the exact fixed contract, a `pilot` run, and one epoch for both rows together.
- [ ] If batch `16` fails for a concrete runtime-capacity reason, retry descending sizes one at a time until the first passing identical batch size is found; record each failure briefly and stop at the first passing value because that is the largest passing fallback.
- [ ] Keep the probe on the same fixed contract except for the minimal one-epoch feasibility budget.
- [ ] Write `resolved_batch_size.json` with the chosen batch size, the failed candidates if any, and the reason batch `16` could not be kept.

Suggested probe command template:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/batchsize-probe-b<CANDIDATE>-<timestamp> \
  --profiles spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_modes24 \
  --history-len 2 \
  --epochs 1 \
  --batch-size <CANDIDATE> \
  --max-train-trajectories 1024 \
  --max-val-trajectories 128 \
  --max-test-trajectories 128 \
  --max-windows-per-trajectory 8 \
  --device cuda \
  --num-workers 0
```

- [ ] Do not start the `80`-epoch paired run until one shared batch size has passed.

### Task 3: Run The Fresh Paired 80-Epoch Compare

**Files:**

- Execution entrypoint: `scripts/studies/run_pdebench_image128_suite.py`
- Underlying task runner: `scripts/studies/pdebench_image128/cfd_cns.py`

- [ ] Launch the final paired run in tmux under `ptycho311`, with exact PID tracking and a unique output root.
- [ ] Use mode `pilot`, the resolved identical batch size, `80` epochs, and only the two approved spectral profiles.
- [ ] Do not launch a duplicate run if another process is already writing to the same output root.
- [ ] Consider the run complete only when the tracked PID exits with code `0` and the run root contains the required summary and per-profile metrics artifacts.

Required long-run command template:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/cns-spectral-modes24-80ep-<timestamp> \
  --profiles spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_modes24 \
  --history-len 2 \
  --epochs 80 \
  --batch-size <RESOLVED_BATCH_SIZE> \
  --max-train-trajectories 1024 \
  --max-val-trajectories 128 \
  --max-test-trajectories 128 \
  --max-windows-per-trajectory 8 \
  --device cuda \
  --num-workers 0
```

- [ ] Verify the finished run root contains at least:
  - `comparison_summary.json`
  - `comparison_summary.csv`
  - `metrics_spectral_resnet_bottleneck_base.json`
  - `metrics_spectral_resnet_bottleneck_modes24.json`
  - `model_profile_spectral_resnet_bottleneck_base.json`
  - `model_profile_spectral_resnet_bottleneck_modes24.json`

### Task 4: Emit The Convergence Audit And Interpret The Result

**Files:**

- Primary: `scripts/studies/pdebench_image128/reporting.py`
- Test if patched: `tests/studies/test_pdebench_image128_runner.py`

- [ ] Run the existing convergence-audit helper against the fresh `80`-epoch run root with the two approved profile IDs.
- [ ] If the helper fails because the run did not emit the expected `80` losses per profile, diagnose the narrow cause and fix only the responsible reporting or runner surface before rerunning the long-run result interpretation.
- [ ] Validate that the audit payload and CSV report the fixed contract, `expected_loss_count=80`, late-window diagnostics, and the final eval metrics for both rows.
- [ ] Apply the plan's interpretation rule exactly:
  - both rows still improving -> mark the result inconclusive
  - only one row still improving -> record asymmetry and avoid a clean win claim
  - neither row still improving -> compare final metrics directly and describe any aggregate versus frequency tradeoff

Helper invocation:

```bash
python - <<'PY'
from pathlib import Path
from scripts.studies.pdebench_image128.reporting import write_cfd_cns_convergence_audit

artifact_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare")
run_root = artifact_root / "cns-spectral-modes24-80ep-<timestamp>"
write_cfd_cns_convergence_audit(
    output_root=artifact_root,
    run_root=run_root,
    profile_ids=[
        "spectral_resnet_bottleneck_base",
        "spectral_resnet_bottleneck_modes24",
    ],
    expected_loss_count=80,
)
PY
```

### Task 5: Publish The Durable Summary And Discoverability Updates

**Files:**

- Create/update: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes24_convergence_summary.md`
- Update: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Update: `docs/studies/index.md`
- Update: `docs/index.md`

- [ ] Write the durable summary with the fixed contract, resolved batch size, convergence rule, final metrics, convergence-audit interpretation, runtime/parameter context, and explicit capped-evidence claim boundary.
- [ ] Update the broader CNS summary with a short note and link to the new modes-24 summary so the CNS study history remains discoverable.
- [ ] Update `docs/studies/index.md` and `docs/index.md` so the new summary is discoverable from the repo documentation map.
- [ ] Keep all wording explicit that this item is capped decision-support evidence only and does not justify a default-profile promotion or a benchmark-complete CNS claim.

### Task 6: Final Verification And Artifact Validation

**Files:**

- Verify outputs under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/`

- [ ] Re-run the backlog item's required deterministic checks if any production code changed after the initial preflight.
- [ ] If any production workflow code changed, also run `pytest -m integration -v` and archive the log because `docs/TESTING_GUIDE.md` requires the integration marker for workflow-touching changes.
- [ ] Run a final artifact validation check that confirms the durable summary, convergence audit, resolved batch size record, and long-run summary artifacts all exist.

Suggested final artifact validation:

```bash
python - <<'PY' | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/verification/final_artifact_validation.log
from pathlib import Path

artifact_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare")
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes24_convergence_summary.md"),
    artifact_root / "resolved_batch_size.json",
    artifact_root / "convergence_audit.json",
    artifact_root / "convergence_audit.csv",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing required outputs: {missing}")
print("modes24 convergence summary and audit outputs present")
PY
```

## Required Deterministic Checks

These backlog-item checks are mandatory and are not replaced by narrower checks:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

If a code patch extends the runner or reporting contract, add the narrowest targeted test selector needed for the change, but keep the two required checks above.

## Completion Criteria

- A fresh paired `80`-epoch `pilot` run exists for `spectral_resnet_bottleneck_base` and `spectral_resnet_bottleneck_modes24` on the fixed `1024 / 128 / 128`, `history_len=2`, `max_windows_per_trajectory=8`, `mse` CNS contract.
- Both rows used the same resolved batch size, with any fallback from `16` recorded concretely.
- `convergence_audit.json` and `convergence_audit.csv` exist and match the plan's fixed convergence rule.
- The durable summary states whether the result is converged or inconclusive and preserves the capped decision-support claim boundary.
- Discoverability docs point to the new summary.
