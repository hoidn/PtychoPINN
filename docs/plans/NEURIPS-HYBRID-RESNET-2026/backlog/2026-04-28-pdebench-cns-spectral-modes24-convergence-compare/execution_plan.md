# PDEBench CNS Spectral Modes-24 Convergence Compare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a fresh same-contract `12/12` versus `24/24` PDEBench `2d_cfd_cns` comparison on the capped `1024 / 128 / 128` slice, using a fixed convergence-oriented budget and an explicit stop rule so the result separates spectral-mode value from obvious under-convergence.

**Architecture:** Treat this as a bounded Roadmap Phase 2 CNS follow-up with three units: first prove the current repo and exact capped CNS contract are green, then run one fresh paired long run for `spectral_resnet_bottleneck_base` and `spectral_resnet_bottleneck_modes24` at the same resolved batch size, then publish an item-local convergence audit plus a durable summary with the correct capped-evidence claim boundary. The existing `24/24` model profile, CNS runner, and convergence-audit writer already exist; implementation should audit and reuse them rather than re-adding them blindly, and only patch code if a deterministic gap blocks the required execution or audit.

**Tech Stack:** PATH `python`, `pytest`, `compileall`, PyTorch CNS runner under the documented `ptycho311` environment for long CUDA runs, tmux with exact PID tracking for long jobs, repo-local Markdown/JSON/CSV artifacts, and updates under `docs/plans/`, `docs/index.md`, `docs/studies/index.md`, `.artifacts/`, and `state/NEURIPS-HYBRID-RESNET-2026/`.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-28-pdebench-cns-spectral-modes24-convergence-compare`
- Date: `2026-04-28`
- Roadmap lane: Phase 2 bounded capped CNS follow-up compare
- Authoritative plan path: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-28-pdebench-cns-spectral-modes24-convergence-compare/execution_plan.md`
- Durable summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes24_convergence_summary.md`
- Item artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/`

## Selected Backlog Objective

- Revisit the shared spectral CNS mode count under a convergence-oriented budget by comparing exactly two rows:
  - baseline: `spectral_resnet_bottleneck_base` with `fno_modes=12`, `spectral_bottleneck_modes=12`
  - candidate: `spectral_resnet_bottleneck_modes24` with `fno_modes=24`, `spectral_bottleneck_modes=24`
- Keep the local PDEBench `2d_cfd_cns` contract fixed on the official `128x128` file and the capped `1024 / 128 / 128`, `history_len=2`, `max_windows_per_trajectory=8`, `mse` slice.
- Use batch size `16` for both rows if feasible; otherwise fall back to the largest smaller identical batch size that allows both rows to run, and record the fallback reason.
- Rerun both rows fresh under one selected long budget. Do not reuse the older `40`-epoch shared `12/12` row as the final baseline for this item.
- Report train loss trajectory, late-window convergence diagnostics, final eval metrics, and whether either row is still materially improving at stop time.

## Scope

- In scope:
  - verify the already-landed manual-only `24/24` profile and existing convergence-audit support
  - prove the exact `1024 / 128 / 128`, `history_len=2`, `mse` contract in inspect mode before any expensive run, and rerun inspect once at the resolved shared fallback batch size if `16` does not hold
  - determine the largest passing identical batch size, starting from `16`
  - run one fresh paired long run containing `spectral_resnet_bottleneck_base` and `spectral_resnet_bottleneck_modes24` together
  - emit an item-local convergence audit from the final run’s `train_epoch_losses`
  - write a durable summary and update discoverability/state docs that need the new result
- Context only, not final evidence:
  - the completed capped `modes32` summary
  - prior batch-`4` capped rows
  - the `1024cap` finalist confirmation row from the architecture-ablation item
  - the `2048cap` scaling follow-up

## Explicit Non-Goals

- Do not widen this into a full mode sweep.
- Do not add `16`, `32`, decoupled encoder-only, or bottleneck-only mode settings.
- Do not mix in `spectral_resnet_bottleneck_noshare`, deeper shared bottlenecks, FFNO, GNOT, CDI/ptycho work, history-length changes, training-set-size scaling to `2048`, or physics regularization.
- Do not relax the fixed task, split, metric, or training-loss contract for convenience.
- Do not convert this capped item into a full-training benchmark claim.
- Do not create `/home/ollie/Documents/neurips/` artifacts or manuscript prose.
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not create worktrees.

## Binding Constraints

### Steering And Roadmap

- Keep equal-footing comparisons explicit.
- Preserve the metric, split, and protocol boundaries from the approved design and roadmap.
- Do not silently relax fairness constraints to make the item easier.
- Keep this item in the current Phase 2 capped CNS window; do not expand into later roadmap phases or unrelated backlog items.
- Treat every artifact here as capped decision-support evidence only, never as full-training benchmark evidence or paper-grade competitiveness evidence.

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
- only the two coupled mode knobs may change:
  - baseline: `fno_modes=12`, `spectral_bottleneck_modes=12`
  - candidate: `fno_modes=24`, `spectral_bottleneck_modes=24`

### Progress-Ledger Prerequisite Status

- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` shows Phase 0 and Phase 1 complete and no blocked tranches, so this bounded Phase 2 item can proceed without a roadmap pivot.
- The backlog item’s declared prerequisite `2026-04-21-pdebench-cns-spectral-modes32-compare` is already complete and supplies the motivation for a longer convergence budget.
- Current CNS infrastructure is already present in-repo:
  - `spectral_resnet_bottleneck_modes24` is defined in `scripts/studies/pdebench_image128/run_config.py`
  - the `2d_cfd_cns` runner already writes `split_manifest.json`, metrics payloads, and `train_epoch_losses`
  - `write_cfd_cns_convergence_audit(...)` already exists in `scripts/studies/pdebench_image128/reporting.py`
- Because those pieces already exist, implementation should prefer execution and narrow gap-fixing over speculative refactors.

## Convergence Standard

Define this rule before launching the long run and do not change it mid-item unless the plan itself is explicitly updated.

### Motivation

- The prior `40`-epoch `12/12` and `32/32` spectral trajectories were still dropping materially, so the earlier stop point was too early for a fair spectral-mode judgment.
- This item exists to test whether a less aggressive `24/24` increase remains worthwhile once both compared rows have a longer shared budget.

### Fixed Stop Budget

- Long paired run budget: `80` epochs for both rows in the same fresh `pilot` run.
- This budget stays capped and decision-support-only, but it is long enough to make the convergence audit meaningful.

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

- **Contract And Preflight Unit:** verify the existing repo surfaces, the exact capped CNS split contract, and the batch-size feasibility gate before any long run. Start with `batch_size=16` as the target contract, but if Task 2 proves a smaller identical fallback is required, rerun inspect once at that resolved size before Task 3 so the final inspect manifest and long-run contract stay aligned. Only patch code if a deterministic gap prevents those checks or the convergence audit from running repeatably.
- **Execution Unit:** run one fresh paired `80`-epoch `pilot` job for the two spectral rows on the same exact contract and resolved common batch size, in tmux, with exact PID tracking and no duplicate output-root launches.
- **Reporting Unit:** emit convergence-audit JSON/CSV plus the durable summary, then sync only the discovery/state docs needed to make the result findable with the right claim boundary.

## Concrete File And Artifact Targets

### Code And Tests That May Change If A Real Gap Is Found

- `scripts/studies/pdebench_image128/run_config.py`
- `scripts/studies/pdebench_image128/reporting.py`
- `scripts/studies/pdebench_image128/cfd_cns.py`
- `scripts/studies/run_pdebench_image128_suite.py`
- `tests/studies/test_pdebench_image128_models.py`
- `tests/studies/test_pdebench_image128_runner.py`

Only touch these files if the required checks or the convergence-audit publication path actually fail. Do not rework unrelated PDEBench infrastructure.

### Expected Artifact Outputs

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/inspect-*/`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/batchsize-probe-*/`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/cns-spectral-modes24-80ep-*/`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/convergence_audit.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/convergence_audit.csv`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/resolved_batch_size.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/verification/pytest_preflight.log`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/verification/compileall_preflight.log`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/verification/pytest_postfix.log` when code changes after preflight
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/verification/compileall_postfix.log` when code changes after preflight
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/verification/pytest_integration_postfix.log` when any production workflow file changes after preflight
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/verification/final_artifact_validation.log`
- optional narrow helper logs or validation notes under the same item artifact root

### Durable Docs And State Surfaces

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes24_convergence_summary.md`
- `docs/studies/index.md`
- `docs/index.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`

## Execution Tasks

### Task 1: Preflight The Existing Surface And Exact CNS Contract

**Files:**

- Read/verify: `scripts/studies/pdebench_image128/run_config.py`
- Read/verify: `scripts/studies/pdebench_image128/reporting.py`
- Read/verify: `scripts/studies/pdebench_image128/cfd_cns.py`
- Test: `tests/studies/test_pdebench_image128_models.py`
- Test: `tests/studies/test_pdebench_image128_runner.py`

- [ ] Confirm `spectral_resnet_bottleneck_modes24` still exists, remains manual-only, and differs from `spectral_resnet_bottleneck_base` only in the two mode knobs.
- [ ] Confirm the current runner and reporting surfaces can already support the item:
  - `split_manifest.json` and dataset metadata are written in `inspect` mode
  - the long-run metrics payload will carry `train_epoch_losses`
  - `write_cfd_cns_convergence_audit(...)` matches the fixed `80`-epoch rule above
- [ ] Create `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/verification/` before running deterministic checks so the required evidence logs have fixed paths.
- [ ] Run the backlog item’s required deterministic checks before any expensive run and archive their passing output:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py \
  2>&1 | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/verification/pytest_preflight.log
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py \
  2>&1 | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/verification/compileall_preflight.log
```

- [ ] If either required check fails, diagnose, apply the smallest narrow fix, and rerun the same checks before proceeding. Do not mark the item `BLOCKED` for a normal test/import/path/harness failure unless a documented narrow fix attempt still leaves it unrecoverable.
- [ ] If any narrow fix changes a production workflow file (`scripts/studies/pdebench_image128/*.py` or `scripts/studies/run_pdebench_image128_suite.py`), rerun `pytest -v -m integration` and archive the passing output to `verification/pytest_integration_postfix.log` before resuming expensive execution.
- [ ] Run an exact-contract `inspect` gate into a fresh item-local inspect root using `batch_size=16` as the initial target contract:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode inspect \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/inspect-<timestamp> \
  --profiles spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_modes24 \
  --history-len 2 \
  --epochs 80 \
  --batch-size 16 \
  --max-train-trajectories 1024 \
  --max-val-trajectories 128 \
  --max-test-trajectories 128 \
  --max-windows-per-trajectory 8 \
  --device cuda
```

- [ ] Verify the inspect root contains `invocation.json`, `dataset_manifest.json`, `split_manifest.json`, and `hdf5_metadata.json`, and that the split counts, history length, loss contract, and profile IDs match this plan exactly.
- [ ] Treat this first inspect root as the contract proof for dataset/split/loss/profile identity. If Task 2 later proves that `16` is not the largest passing identical batch size, rerun inspect once with the resolved smaller shared batch size into a fresh inspect root before Task 3. That rerun is required so the final inspect manifest and long-run invocation match exactly.

**Verification for Task 1:**

- `verification/pytest_preflight.log` and `verification/compileall_preflight.log` exist and show passing checks.
- A green inspect root exists for the exact `1024 / 128 / 128`, `history_len=2`, `max_windows_per_trajectory=8`, `mse`, and two-profile contract, with `16` recorded as the initial batch-size target.
- No long run starts until Task 2 resolves the shared batch size and there is a green inspect root that matches that resolved size.

### Task 2: Resolve The Largest Passing Identical Batch Size

**Files:**

- Reuse: `scripts/studies/pdebench_image128/cfd_cns.py`
- Reuse: `scripts/studies/run_pdebench_image128_suite.py`
- Modify only if required by a real failure: same code/test files from Task 1

- [ ] Start from `batch_size=16` and run a short paired feasibility probe on the exact same capped contract in a fresh probe root.
- [ ] If `16` fails for a concrete runtime reason such as OOM, step down to the next smaller identical size for both rows and retry until one size passes.
- [ ] Record the resolved batch size and any fallback reason in `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/resolved_batch_size.json`. If `16` passes, record that explicitly.
- [ ] If the resolved batch size is smaller than `16`, rerun Task 1 inspect once with that resolved shared size and archive the new inspect root before launching the long run.
- [ ] Do not reinterpret the probe metrics as model-ranking evidence; this step exists only to prove the long run can launch under a fair common batch size.
- [ ] If batch-size probing fails because of a normal harness or code defect, diagnose/fix/rerun first. Reserve `BLOCKED` for missing resources, unavailable hardware, or an unrecoverable failure after a documented narrow fix attempt.
- [ ] If any narrow fix in this task changes a production workflow file, rerun `pytest -v -m integration` and archive the passing output to `verification/pytest_integration_postfix.log` before the next probe or long-run launch.

Suggested probe command pattern:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/batchsize-probe-<timestamp> \
  --profiles spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_modes24 \
  --history-len 2 \
  --epochs 1 \
  --batch-size <candidate_batch_size> \
  --max-train-trajectories 1024 \
  --max-val-trajectories 128 \
  --max-test-trajectories 128 \
  --max-windows-per-trajectory 8 \
  --device cuda
```

**Verification for Task 2:**

- A fresh probe root proves the largest passing identical batch size.
- `resolved_batch_size.json` exists and records either `16` as confirmed or the smaller shared fallback plus its reason.
- The final inspect root now matches the resolved shared batch size, whether that stays at `16` or falls back.
- The long paired run command is frozen only after this task completes.

### Task 3: Run The Fresh Paired 80-Epoch Compare

**Files:**

- Reuse: `scripts/studies/pdebench_image128/cfd_cns.py`
- Reuse: `scripts/studies/run_pdebench_image128_suite.py`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/`

- [ ] Launch one fresh paired `80`-epoch `pilot` run for `spectral_resnet_bottleneck_base` and `spectral_resnet_bottleneck_modes24` on the fixed contract and the resolved common batch size.
- [ ] Run the long job in tmux, activate `ptycho311`, and follow the project long-run guardrail:
  - track the exact launched PID
  - wait on that PID, not a broad `pgrep`
  - do not launch a duplicate command into the same `--output-root`
- [ ] Keep the output root unique for the long run so partial probes and the final paired run cannot be confused.
- [ ] Treat the run as complete only when the tracked PID exits `0` and the required fresh artifacts exist.
- [ ] If the long run fails for a recoverable code/path/harness reason, attempt the smallest narrow fix, rerun the required checks, and relaunch with a fresh output root. Do not downgrade directly to `BLOCKED`.
- [ ] If a recoverable fix here changes a production workflow file, rerun `pytest -v -m integration` and archive the passing output to `verification/pytest_integration_postfix.log` before relaunching.

Canonical long-run command pattern:

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
  --batch-size <resolved_batch_size> \
  --max-train-trajectories 1024 \
  --max-val-trajectories 128 \
  --max-test-trajectories 128 \
  --max-windows-per-trajectory 8 \
  --device cuda
```

- [ ] Verify the fresh run root contains, at minimum:
  - `invocation.json`
  - `dataset_manifest.json`
  - `split_manifest.json`
  - `comparison_summary.json`
  - `comparison_summary.csv`
  - `metrics_spectral_resnet_bottleneck_base.json`
  - `metrics_spectral_resnet_bottleneck_modes24.json`
  - `model_profile_spectral_resnet_bottleneck_base.json`
  - `model_profile_spectral_resnet_bottleneck_modes24.json`
- [ ] Confirm the fresh run still shows `evidence_scope="capped_decision_support_only"` and does not masquerade as benchmark performance.

**Verification for Task 3:**

- The tracked long-run PID exits `0`.
- The fresh run root contains the required artifacts and fresh timestamps.
- The fixed contract matches the final inspect root and the resolved batch-size record.

### Task 4: Publish The Convergence Audit And Durable Summary

**Files:**

- Reuse or minimally extend: `scripts/studies/pdebench_image128/reporting.py`
- Reuse or minimally extend: `scripts/studies/run_pdebench_image128_suite.py`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes24_convergence_summary.md`
- Modify: `docs/studies/index.md`
- Modify: `docs/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Test if code changed: `tests/studies/test_pdebench_image128_runner.py`

- [ ] Use `write_cfd_cns_convergence_audit(...)` against the fresh `80`-epoch run root to emit:
  - `convergence_audit.json`
  - `convergence_audit.csv`
- [ ] If there is no stable checked-in entrypoint for the audit and a one-off call would be too fragile, add the smallest reusable wrapper and cover it with tests. Otherwise prefer reusing the existing function without widening the code surface.
- [ ] Validate that the audit payload records:
  - the fixed contract
  - the two profile IDs
  - `expected_loss_count=80`
  - the late-window ratio and `last5_delta` for both rows
  - the final eval metrics copied from the fresh run
- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes24_convergence_summary.md` with:
  - status, scope, governing plan, and artifact root
  - the exact fixed fairness contract
  - the resolved batch-size decision and any fallback reason
  - the fresh run root and required output proofs
  - the convergence audit results for both rows
  - the final interpretation rule:
    - inconclusive if both rows still materially improve
    - asymmetry note if only one row still materially improves
    - direct metric comparison only if both are no longer materially improving
  - a strict claim boundary stating the result is capped CNS decision-support evidence only
- [ ] Update `docs/studies/index.md` and `docs/index.md` so the new summary is discoverable.
- [ ] Update `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` only as needed to reflect the completed bounded follow-up and keep routing/state accurate without rewriting the roadmap.

**Verification for Task 4:**

- If any code changed, rerun the required deterministic checks:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py \
  2>&1 | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/verification/pytest_postfix.log
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py \
  2>&1 | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/verification/compileall_postfix.log
```

- If any production workflow file changed, rerun the required integration gate and archive it:

```bash
pytest -v -m integration \
  2>&1 | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/verification/pytest_integration_postfix.log
```

- Run a final artifact validation that confirms:
  - the durable summary exists
  - `convergence_audit.json` and `convergence_audit.csv` exist
  - the fresh run root exists
  - the summary’s claim boundary matches the audit outcome
  - any docs index entries point to the new summary

Suggested final validation snippet:

```bash
python - <<'PY' 2>&1 | tee .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/verification/final_artifact_validation.log
from pathlib import Path

required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes24_convergence_summary.md"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/convergence_audit.json"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/convergence_audit.csv"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing required outputs: {missing}")
print("modes24 convergence summary and audit outputs present")
PY
```

## Required Deterministic Checks

These are mandatory for this backlog item unless a future approved plan revision explicitly replaces them with a stronger gate:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

Archive the first passing run to `verification/pytest_preflight.log` and `verification/compileall_preflight.log`. If implementation changes code after that point, rerun and archive the new passing output to `verification/pytest_postfix.log` and `verification/compileall_postfix.log` before closing the item. If any production workflow file changes, also archive a passing `pytest -v -m integration` rerun to `verification/pytest_integration_postfix.log` before the next expensive step or final closeout.

## Completion Standard

- The repo proves the exact `24/24` versus `12/12` same-contract lane without fairness drift.
- The expensive long run waits for green deterministic checks, archived verification logs, and a green inspect gate that matches the resolved shared batch size.
- The final result comes from one fresh paired `80`-epoch run under the resolved common batch size.
- The convergence audit is present and governs interpretation.
- The durable summary, index updates, and progress-ledger state all reflect the same bounded claim.
