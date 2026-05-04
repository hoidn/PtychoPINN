# CNS History-Len-5 Comparator Gap Fill Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining all-`history_len=5` CNS comparator gap by running fresh capped `40`-epoch `fno_base` and `unet_strong` rows under the same local contract already used for the completed authored-FFNO and spectral `history_len=5` authorities, then publish durable comparison and discoverability updates without reopening the locked `history_len=2` paper lane.

**Architecture:** Reuse the existing `scripts/studies/pdebench_image128/` CNS runner and reporting helpers. First freeze the exact reference rows and validate the `history_len=5` contract with deterministic checks plus inspect-mode proof, then launch one fresh capped pilot run root for `fno_base` and `unet_strong` in `tmux` with tracked PID ownership, then emit two comparison surfaces: one history-delta compare against the frozen `history_len=2` same-cap anchors for the same model IDs, and one same-history cross-run compare against the completed `history_len=5` authored-FFNO and spectral authorities. Finish by writing the new summary and synchronizing the durable evidence indexes while leaving the current `history_len=2`, `2048 / 256 / 256` CNS authority unchanged unless this item is fully and honestly complete.

**Tech Stack:** PATH `python`, PyTorch, h5py, NumPy, tmux with the `ptycho311` environment for long runs, `scripts/studies/run_pdebench_image128_suite.py`, `scripts/studies/pdebench_image128/reporting.py`, `.artifacts/NEURIPS-HYBRID-RESNET-2026/` evidence roots.

---

## Selected Objective

- Run the missing PDEBench CNS `history_len=5`, `40`-epoch comparator rows for `fno_base` and `unet_strong` so the completed authored-FFNO and spectral `history_len=5` rows no longer sit beside missing same-history FNO and U-Net comparators.

## Scope

- Reuse the official `2d_cfd_cns` dataset, existing task-local CNS runner, existing capped split policy, and existing comparison helpers.
- Match the completed `history_len=5` authorities on:
  - `history_len=5`
  - `epochs=40`
  - capped split `512 / 64 / 64`
  - `max_windows_per_trajectory=8`
  - emitted windows `4096 / 512 / 512`
  - train-only normalization
  - training loss `mse`
  - metric family `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- Produce:
  - a history-delta compare against the frozen same-profile `history_len=2` capped anchors for `fno_base` and `unet_strong`
  - a same-history cross-run compare against the completed `history_len=5` authored-FFNO and spectral authorities
- Write the new durable summary and synchronize the required evidence/discoverability surfaces.

## Explicit Non-Goals

- No rerun of `author_ffno_cns_base` or `spectral_resnet_bottleneck_base`.
- No reopening of the locked CNS paper contract, locked `history_len=2` headline table, or active `2048 / 256 / 256` capped authority bundle.
- No new full-training CNS benchmark claim, no Phase 3 CDI work, and no candidate-lane work.
- No silent promotion of `history_len=5` rows into the manuscript headline table during this item.
- No `/home/ollie/Documents/neurips/` artifact or manuscript-prose work.

## Binding Constraints And Source Of Truth

- Steering is binding:
  - keep equal-footing comparisons explicit
  - preserve the metric, split, and protocol boundaries already approved
  - strengthen core CNS comparison evidence rather than spending budget on optional follow-ups
- The roadmap is binding:
  - this item is an allowed bounded Phase 2 CNS follow-up
  - it must consume the completed `history_len=5` authored-FFNO and spectral authorities
  - it must rerun only `fno_base` and `unet_strong`
  - it must not treat the result as permission to reopen the locked `history_len=2` paper contract
- The design is binding:
  - capped or pilot CNS rows are decision-support or adjacent capped context only
  - smoke and inspect outputs never count as benchmark-performance evidence
- The current CNS paper contract remains binding:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_2048cap_extension_summary.md`
  define the active `history_len=2` authority and must remain the current paper-facing CNS authority unless a later reviewed decision changes that
- Long-running command ownership is binding:
  - use `tmux` plus `ptycho311`
  - track the exact launched PID
  - do not reuse an output root that already has an active writer
  - do not mark the item `BLOCKED` for ordinary import/test/path/environment failures before a narrow local fix attempt is made
- Evidence-index maintenance is binding:
  - update every applicable durable evidence surface or explicitly record why a surface does not change

## Prerequisite Status

- Completed and required as frozen inputs:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_spectral_history_len4plus_compare_summary.md`
    with the completed spectral `history_len=5`, `40`-epoch root:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/history5-pilot-40ep-20260501T101147Z`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_history_length_summary.md`
    with the completed authored-FFNO `history_len=5`, `40`-epoch root:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/history5-pilot-40ep-20260502T074500Z`
  - frozen same-cap `history_len=2` reference rows for the missing profiles, from the locked capped CNS paper lane:
    - `fno_base`:
      `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`
    - `unet_strong`:
      `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`
- Already in place and should be reused, not redesigned:
  - `scripts/studies/pdebench_image128/cfd_cns.py`
  - `scripts/studies/pdebench_image128/reporting.py`
  - `scripts/studies/run_pdebench_image128_suite.py`
  - dataset contract entry `cns_history5_cap512_40ep` in `model_variant_index.json`
- Still missing:
  - fresh `history_len=5`, `40`-epoch rows for `fno_base` and `unet_strong`
  - a new summary authority for this backlog item
  - durable index entries that close or replace the current pending-gap notes

## Implementation Architecture

- **Reference And Contract Unit:** freeze the exact h2 and completed h5 authority rows, prove the `history_len=5` contract via inspect mode, and fix only narrow runner/reporting issues if the current path cannot honestly produce the missing rows.
- **Execution And Comparison Unit:** launch the missing `fno_base` and `unet_strong` rows under one fixed capped contract, then emit the history-delta and same-history compare sidecars with existing reporting helpers.
- **Durable Closeout Unit:** write the new summary, update the evidence indexes and progress surfaces, and keep the paper-facing CNS authority and claim boundary unchanged unless both rows complete cleanly.

## File And Artifact Targets

Mandatory contract inputs to read and honor:

- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_2048cap_extension_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_spectral_history_len4plus_compare_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_history_length_summary.md`

Mandatory repo surfaces expected to change:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history5_comparator_gap_fill_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cns-history5-comparator-gap-fill/execution_report.md`
- `artifacts/checks/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cns-history5-comparator-gap-fill-checks.json`
- `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cns-history5-comparator-gap-fill-summary.json`

Conditional discoverability surfaces only if they would otherwise remain factually stale after both new rows complete:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- `scripts/studies/paper_results_refresh.py`
- `tests/studies/test_paper_results_refresh.py`

Mandatory item-local artifact root to create and use:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history5-comparator-gap-fill/`

Mandatory generated artifacts for a completed item:

- `history5-inspect-<timestamp>/`
- `history2_history5_reference_runs.json`
- `history5-gap-fill-40ep-<timestamp>/`
- `launch-history5-gap-fill-40ep-<timestamp>/exit_code`
- `compare_40ep_history5_against_history2.json`
- `compare_40ep_history5_against_history2.csv`
- `compare_40ep_against_existing.json`
- `compare_40ep_against_existing.csv`
- verification logs copied or linked under the item artifact root

Preferred packaging that should be preserved when available but must not decide success by itself:

- `compare_40ep_history5_against_history2_sample0.png`
- `compare_40ep_history5_against_history2_sample0_error.png`
- `compare_40ep_sample0.png`
- `compare_40ep_sample0_error.png`

Conditional source-edit surfaces only if a narrow blocking bug is proven:

- `scripts/studies/pdebench_image128/reporting.py`
- `scripts/studies/pdebench_image128/cfd_cns.py`
- `scripts/studies/run_pdebench_image128_suite.py`
- `tests/studies/test_pdebench_image128_runner.py`
- `tests/studies/test_pdebench_image128_models.py`
- `tests/studies/test_pdebench_cfd_cns_data.py`
- `tests/studies/test_pdebench_cfd_cns_metrics.py`

## Task Checklist

### Task 1: Freeze The Exact Contract And Prove The Missing Rows Are Launchable

**Purpose:** confirm the fixed capped contract, freeze the required references, and prove the `history_len=5` run path before any expensive launch.

- [ ] Verify the required frozen sources exist before touching code or launching training. This is blocking.

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_2048cap_extension_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_spectral_history_len4plus_compare_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_history_length_summary.md"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/history5-pilot-40ep-20260501T101147Z"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/history5-pilot-40ep-20260502T074500Z"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing required CNS history5 gap-fill inputs: {missing}")
print("cns history5 gap-fill inputs present")
PY
```

- [ ] Run the backlog item's required deterministic checks before any source edit or long run. Both are blocking.

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

- [ ] Materialize `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history5-comparator-gap-fill/history2_history5_reference_runs.json` with `build_reference_run_manifest(...)` and `write_reference_run_manifest(...)`; do not hand-author the JSON. Record:
  - required reference rows for the same-profile history-delta compare:
    `fno_base` and `unet_strong` from the frozen same-cap `history_len=2`, `40`-epoch root
  - optional reference rows for the same-history cross-run compare:
    `author_ffno_cns_base` and `spectral_resnet_bottleneck_base` from the completed `history_len=5`, `40`-epoch roots
- [ ] Run an inspect-mode proof for the missing rows under the exact target contract. This is blocking before the 40-epoch launch.

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode inspect \
  --profiles fno_base,unet_strong \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history5-comparator-gap-fill/history5-inspect-<timestamp> \
  --history-len 5 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8
```

- [ ] Confirm the inspect root proves only the allowed history delta:
  - `history_len=5`
  - `input_channels=20`
  - sample contract `concat u[t-5:t] -> u[t]`
  - same official dataset file
  - same split caps `512 / 64 / 64`
  - same emitted window counts `4096 / 512 / 512`
  - same batch size `4`
  - same loss `mse`
  - same metric family
- [ ] If a deterministic check or inspect proof fails because of repo-local code drift, fix the narrowest credible surface and rerun the same blocking checks before moving on. Do not mark the item `BLOCKED` yet.

Supporting verification for Task 1:

```bash
pytest tests/studies/test_pdebench_image128_runner.py -k 'reference_run_manifest or cross_run_compare or history' -v
```

### Task 2: Launch The Missing `history_len=5`, `40`-Epoch Comparator Rows

**Purpose:** produce the missing FNO and U-Net rows under one fixed capped contract, with implementation retaining ownership until the run really finishes.

- [ ] Choose a fresh run root under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history5-comparator-gap-fill/`, for example `history5-gap-fill-40ep-<timestamp>`. Do not reuse any existing root and do not launch if another process is already writing there.
- [ ] Launch the fresh run in `tmux` using the `ptycho311` environment. This is blocking work, not fire-and-forget.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode pilot \
  --profiles fno_base,unet_strong \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history5-comparator-gap-fill/history5-gap-fill-40ep-<timestamp> \
  --history-len 5 \
  --epochs 40 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8 \
  --device cuda \
  --num-workers 0
```

- [ ] Track the exact launched PID and wait for that PID. Only treat the run as complete when:
  - the tracked process exits `0`
  - the run root contains fresh `invocation.json`, `dataset_manifest.json`, `split_manifest.json`, `normalization_stats_state.json`, `comparison_summary.json`, `model_profile_fno_base.json`, `model_profile_unet_strong.json`, `metrics_fno_base.json`, and `metrics_unet_strong.json`
  - the launch proof directory records the exit code for this exact run
- [ ] If only one profile fails, diagnose the smallest honest fix, rerun only the missing profile under a fresh root that preserves the fixed contract, and keep the completed row untouched.
- [ ] Treat missing PNG/NPZ comparison galleries as supporting-only packaging debt unless the core metrics/manifests also failed.
- [ ] If the missing row still cannot complete after a documented narrow fix attempt, convert that row to an explicit row-level blocker in the summary and closeout surfaces rather than marking the entire backlog item `BLOCKED` immediately.

Blocking verification for Task 2:

```bash
python - <<'PY'
from pathlib import Path
root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history5-comparator-gap-fill")
candidates = sorted([p for p in root.iterdir() if p.is_dir() and "history5-gap-fill-40ep-" in p.name])
if not candidates:
    raise SystemExit("missing fresh history5 gap-fill run root")
run_root = candidates[-1]
required = [
    run_root / "invocation.json",
    run_root / "dataset_manifest.json",
    run_root / "split_manifest.json",
    run_root / "normalization_stats_state.json",
    run_root / "comparison_summary.json",
    run_root / "model_profile_fno_base.json",
    run_root / "model_profile_unet_strong.json",
    run_root / "metrics_fno_base.json",
    run_root / "metrics_unet_strong.json",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing required history5 run artifacts: {missing}")
print(run_root)
PY
```

### Task 3: Emit The Two Required Comparison Surfaces

**Purpose:** make the new rows interpretable against both their same-profile frozen h2 anchors and the already-completed h5 authorities, without inventing new reporting formats.

- [ ] Use `write_history_delta_compare(...)` to emit `compare_40ep_history5_against_history2.json` and `.csv` from the fresh run root, with `fresh_profile_ids=["fno_base", "unet_strong"]` and the frozen same-profile `history_len=2` capped reference rows.
- [ ] The history-delta compare must prove that only `history_len` changed relative to those h2 references; it must carry:
  - `delta_kind = history_len_only`
  - `reference_history_len = 2`
  - `fresh_history_len = 5`
  - the sample-contract and input-channel delta
  - absolute metrics and metric deltas
  - runtime, parameter count, and raw/emitted window counts
- [ ] Use `write_cross_run_compare(...)` to emit `compare_40ep_against_existing.json` and `.csv` from the same fresh run root, with:
  - fresh profiles `fno_base` and `unet_strong`
  - required same-history reference rows `author_ffno_cns_base` and `spectral_resnet_bottleneck_base`
  - no h2 rows in this same-history compare, because `write_cross_run_compare(...)` requires a strict fixed-contract match including `history_len`
- [ ] Preserve manuscript-label mapping where relevant:
  - repo row `spectral_resnet_bottleneck_base` remains manuscript label `SRU-Net*`
  - the new rows remain `FNO` and `U-Net`
- [ ] If a reporting helper bug blocks either compare surface, patch only the narrow reporting path, add or extend targeted tests, rerun the required backlog-item checks, and regenerate the compare outputs.

Blocking verification for Task 3:

- [ ] `compare_40ep_history5_against_history2.json` and `.csv` exist and reference only the same-profile frozen h2 rows for `fno_base` and `unet_strong`.
- [ ] `compare_40ep_against_existing.json` and `.csv` exist and include the completed h5 authored-FFNO and spectral authorities plus the fresh FNO/U-Net rows.

Supporting verification for Task 3:

```bash
pytest tests/studies/test_pdebench_image128_runner.py -k 'reference_run_manifest or cross_run_compare or history' -v
```

### Task 4: Write The Durable Summary And Synchronize Discoverability

**Purpose:** publish the outcome in the repo’s durable evidence surfaces without overstating what changed.

- [ ] Create `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history5_comparator_gap_fill_summary.md`.
- [ ] The summary must explicitly state:
  - this item is `adjacent_capped_context_only`
  - the fixed capped contract
  - the frozen h2 reference roots used for same-profile history-delta compares
  - the completed h5 authored-FFNO and spectral authority roots used for the same-history cross-run compare
  - fresh metrics for `fno_base` and `unet_strong`
  - whether each missing row completed or ended as a row-level blocker
  - that the current active CNS authority remains the locked `history_len=2`, `2048 / 256 / 256` bundle unless a later reviewed decision says otherwise
- [ ] Update `paper_evidence_index.md`:
  - add a new `decision_support` row for this backlog item when at least one fresh row or explicit blocker outcome is durable
  - replace the current pending-gap note only if both missing rows are now completed or explicitly blocked with summary-owned justification
- [ ] Update `evidence_matrix.md`:
  - remove or rewrite the current pending-gap text so it reflects the actual post-run state
  - add the new backlog item to the completed-output coverage section
  - if the new h5 FNO or U-Net rows become the best observed capped family row, update the “best observed capped CNS rows by model family” section without changing the current paper authority section
- [ ] Update `model_variant_index.json`:
  - keep the existing dataset contract `cns_history5_cap512_40ep`
  - add model variants for any successfully completed new rows:
    - `cns_history5_cap512_40ep__fno_base__supervised`
    - `cns_history5_cap512_40ep__unet_strong__supervised`
  - if both rows complete or blockers are fully recorded elsewhere, remove or replace the stale placeholder pointer `primary_authorities.cns_history5_gap_item`
  - keep the current `cns_history2_cap2048_40ep` authority entries intact
- [ ] Update `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` so future selection and planning can discover the outcome without rereading raw artifacts.
- [ ] If this item also closes the repo-local paper-results audit gap, update `scripts/studies/paper_results_refresh.py` and its targeted test only after the summary and evidence indexes already reflect the completed state. Do not let this optional consistency sync expand into manuscript-table edits.

Blocking verification for Task 4:

- [ ] The required backlog-item deterministic checks must be rerun after any code edit made in Tasks 1 through 3.
- [ ] `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history5_comparator_gap_fill_summary.md`, `paper_evidence_index.md`, `evidence_matrix.md`, and `model_variant_index.json` all point to the same new backlog-item summary and artifact root.

Supporting verification for Task 4:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history5_comparator_gap_fill_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing durable closeout surfaces: {missing}")
print("durable closeout surfaces present")
PY
```

## Required Deterministic Checks

These are the backlog item's required deterministic checks and remain mandatory unless an implementation review explicitly approves a stronger replacement. They are blocking before long-running execution and blocking again after any code edit.

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

## Completion Conditions

- Both missing rows completed under the exact capped `history_len=5` contract, or any incomplete row is documented as a row-level blocker after a narrow fix attempt.
- The history-delta compare against the frozen h2 same-profile anchors exists.
- The same-history compare against the completed h5 authored-FFNO and spectral authorities exists.
- The new summary and durable evidence surfaces are synchronized and do not falsely imply that the locked `history_len=2` CNS paper authority changed.
- Long-run ownership evidence exists for the fresh launch.
