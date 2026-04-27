# PDEBench CNS Markov History-1 Compare Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce capped decision-support `history_len=1` Markov-style CNS comparisons at `10` and `40` epochs for `spectral_resnet_bottleneck_base`, `hybrid_resnet_cns`, `fno_base`, and `unet_strong`, then document how those fresh rows differ from the frozen `history_len=2` anchors without drifting the fixed local contract.

**Architecture:** Keep `scripts/studies/pdebench_image128/cfd_cns.py` authoritative for the CNS data, split, normalization, metrics, and artifact-writing surface, and use `pilot` mode for the fresh ranked `history_len=1` runs. Freeze the existing `history_len=2` anchor roots in a manifest, backfill only the missing shell-locked `40`-epoch `hybrid_resnet_cns` history-2 anchor if no exact audited root exists, and extend reporting so the compare sidecar allows only the approved contract delta (`history_len` and its derived sample-contract/input-channel change) while still enforcing dataset, split, loss, batch-size, and metric-family equality. No worktree is used because repo policy explicitly forbids worktrees here.

**Tech Stack:** PATH `python`, conda env `ptycho311`, PyTorch/Lightning, existing PDEBench image-suite runner under `scripts/studies/pdebench_image128/`, pytest, compileall, tmux for long runs, Markdown/JSON/CSV/PNG artifacts under `.artifacts/`

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-21-pdebench-cns-markov-history1-compare`
- Status: pending
- Date: 2026-04-23
- Scope owner: Roadmap Phase 2 selector-authorized capped CNS follow-up lane
- Selected-item context: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-21-pdebench-cns-markov-history1-compare/selected-item-context.md`
- Previous background design: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_markov_history1_compare_design.md`
- Durable summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_markov_history1_compare_summary.md`
- CNS summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/`

This document supersedes the previous background design above and is the execution authority for this backlog item.

## Inputs Read

- `AGENTS.md`
- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/model_baselines.md`
- `docs/studies/index.md`
- `docs/workflows/pytorch.md`
- `docs/steering.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_equal_footing_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_cns_compare_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_markov_history1_compare_design.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-21-pdebench-cns-markov-history1-compare/selected-item-context.md`
- `scripts/studies/pdebench_image128/{cfd_cns.py,reporting.py,run_config.py}`
- `tests/studies/{test_pdebench_image128_runner.py,test_pdebench_cfd_cns_data.py,test_pdebench_cfd_cns_metrics.py}`

## Objective

- Determine whether switching the local capped CNS contract from `history_len=2` to Markov-style `history_len=1` changes the ranking among `spectral_resnet_bottleneck_base`, `hybrid_resnet_cns`, `fno_base`, and `unet_strong`.
- Keep the fixed capped CNS surface intact: same dataset, same split caps, same `max_windows_per_trajectory`, same batch size, same `mse` training loss, same metric family, and same epoch budgets.
- Publish a durable summary that answers two questions explicitly for both `10` and `40` epochs:
  - does `spectral_resnet_bottleneck_base` improve under `history_len=1`?
  - does the row ranking change relative to the frozen `history_len=2` anchor family?

## Scope

- Run fresh `history_len=1` four-row CNS compares at `10` and `40` epochs under the fixed capped slice.
- Freeze and audit the exact `history_len=2` anchor roots needed for those two budgets.
- Backfill only the missing shell-locked `40`-epoch `hybrid_resnet_cns` history-2 anchor if no exact audited root exists.
- Extend the existing reporting/collation path so the fresh history-1 runs can be compared against history-2 anchors without silently treating a contract mismatch as equal footing.
- Update durable summary, CNS summary, discoverability docs, and progress ledger.

## Explicit Non-Goals

- Do not widen this work into rollout or autoregressive evaluation.
- Do not widen this work into author-FFNO, GNOT, Hybrid-spectral architecture, spectral-modes-32, physics-regularization, or any other CNS backlog lane.
- Do not change optimizer, scheduler family, training loss, split counts, `max_windows_per_trajectory`, batch size, metric family, or dataset path for the fresh history-1 rows.
- Do not treat `hybrid_resnet_base` `40`-epoch artifacts as a proxy history-2 anchor for `hybrid_resnet_cns`; the shell differs and that would confound the question.
- Do not promote any result from this item into benchmark-complete CNS evidence or full-suite PDE evidence.
- Do not create `/home/ollie/Documents/neurips/` artifacts or manuscript prose.
- Do not touch `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not create worktrees.

## Steering And Roadmap Constraints

- Steering requires explicit equal-footing comparisons and forbids silently relaxing fairness to make a backlog item easier.
- For this backlog item, the only intended scientific variable between the frozen history-2 anchors and the fresh history-1 rows is the temporal-context contract:
  - old: `history_len=2`, `concat u[t-2:t] -> u[t]`, `input=(8,128,128)`, `target=(4,128,128)`
  - new: `history_len=1`, `concat u[t-1:t] -> u[t]`, `input=(4,128,128)`, `target=(4,128,128)`
- Everything else must remain fixed unless the run is explicitly documented as a missing-anchor backfill:
  - dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
  - split: `512 / 64 / 64` trajectories
  - `max_windows_per_trajectory=8`
  - batch size: `4`
  - training loss: `mse`
  - metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
  - row set for the fresh history-1 compare: `spectral_resnet_bottleneck_base`, `hybrid_resnet_cns`, `fno_base`, `unet_strong`
- The roadmap and current runner/reporting surface now distinguish `pilot` from `readiness`. Because this item must rank capped rows within a decision-support lane, fresh ranked Markov runs must use `pilot` mode, not `readiness`.
- Existing history-2 readiness roots may still be reused as frozen legacy anchors when they match the fixed contract, but the durable summary must state that they predate pilot-mode labeling and remain capped decision-support evidence only.
- The roadmap still treats full-training benchmark rows as a later gate. This item must stay explicitly capped and benchmark-incomplete.

## Roadmap Gate Position

- This backlog item is authorized as a bounded capped CNS follow-up lane within Phase 2, not as the generic full-training suite-ablation gate.
- The item is allowed to answer a narrower decision-support question now because the CNS adapter/data/metric path and the capped history-2 anchor family already exist.
- The plan must preserve two truths at once:
  - the fresh history-1 rows may be ranked internally because they will run in `pilot` mode on a fixed capped contract
  - the resulting interpretation remains capped decision-support evidence only and cannot satisfy the roadmap’s benchmark gate

## Prerequisite Status

- Satisfied from the current durable summaries and progress ledger:
  - the official `2d_cfd_cns` file is staged and checksum-verified
  - the CNS loader supports both `history_len=2` and `history_len=1`
  - the canonical local CNS Hybrid row is `hybrid_resnet_cns` with skip-add and `pixelshuffle`
  - exact capped history-2 anchor roots already exist for:
    - `10` epochs: `spectral_resnet_bottleneck_base`, `hybrid_resnet_cns`, `fno_base`, `unet_strong`
    - `40` epochs: `spectral_resnet_bottleneck_base`, `fno_base`, `unet_strong`
  - the repo already has pilot-mode semantics for capped ranked CNS runs
- Missing and required before the final `40`-epoch history-1 interpretation:
  - no exact shell-locked `40`-epoch `history_len=2` `hybrid_resnet_cns` anchor was found in the current artifact tree
  - a bounded backfill of that single anchor is therefore required unless an exact audited root is found before execution
- Background-only artifact state:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history1-10ep-20260422T193524Z` already exists and matches the scientific `history_len=1` contract, but it was run in `mode=readiness` and is not reflected in the progress ledger or a durable summary
  - treat that root as smoke/provenance background only; it does **not** satisfy the final `10`-epoch ranked tranche for this plan

## Fixed History-2 Reference Anchors

These are the authoritative existing history-2 comparison roots unless an explicit contract audit proves otherwise.

### Required `10`-Epoch History-2 Anchors

- `spectral_resnet_bottleneck_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
- `hybrid_resnet_cns`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
- `fno_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`
- `unet_strong`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`

### Required `40`-Epoch History-2 Anchors

- `spectral_resnet_bottleneck_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`
- `fno_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`
- `unet_strong`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`
- `hybrid_resnet_cns`:
  no exact shell-locked root found yet; backfill this single anchor under the fixed history-2 contract before interpreting the fresh `40`-epoch Markov run

### Forbidden Proxy Anchor

- Do **not** use the earlier `hybrid_resnet_base` `40`-epoch history-2 row as the history-2 comparator for `hybrid_resnet_cns`; the shell differs, so that would mix the Markov change with a skip/upsampler shell change.

## Implementation Architecture

- **Reference and Compare Unit:** `scripts/studies/pdebench_image128/reporting.py` plus `tests/studies/test_pdebench_image128_runner.py` own the frozen reference manifest, the controlled history-delta compare sidecar, and the rule that only `history_len` / derived sample-contract deltas are allowed across the history-1 vs history-2 comparison.
- **Execution Unit:** existing `scripts/studies/pdebench_image128/cfd_cns.py` pilot-mode runs own the fresh `history_len=1` `10`/`40`-epoch four-row execution and the contingency backfill of the missing `40`-epoch `history_len=2` `hybrid_resnet_cns` anchor.
- **Interpretation Unit:** the durable Markov summary, CNS summary update, discoverability docs, and progress ledger own the final interpretation and must preserve the capped decision-support boundary explicitly.

## Concrete File And Artifact Targets

### Expected Code And Test Changes

- Modify: `scripts/studies/pdebench_image128/reporting.py`
- Modify: `tests/studies/test_pdebench_image128_runner.py`
- Modify only if the compare sidecar needs extra emitted metadata that is not already present in run artifacts: `scripts/studies/pdebench_image128/cfd_cns.py`

### Expected Durable Documentation And State Changes

- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_markov_history1_compare_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/index.md`
- Modify: `docs/studies/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Modify only if the result yields a stable reusable takeaway at the same level as the existing CNS findings: `docs/findings.md`

### Fresh Artifacts Required

- Create study root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2_reference_runs.json`
- Create only if needed: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-<timestamp>/`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history1-pilot-10ep-<timestamp>/`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history1-pilot-40ep-<timestamp>/`
- Create top-level compare sidecars:
  - `compare_10ep_against_history2.json`
  - `compare_10ep_against_history2.csv`
  - `compare_10ep_against_history2_sample0.png` if targets align
  - `compare_10ep_against_history2_sample0_error.png` if targets align
  - `compare_40ep_against_history2.json`
  - `compare_40ep_against_history2.csv`
  - `compare_40ep_against_history2_sample0.png` if targets align
  - `compare_40ep_against_history2_sample0_error.png` if targets align
- Require each fresh run root to contain the normal CNS run artifacts:
  - `invocation.json`
  - `invocation.sh`
  - `dataset_manifest.json`
  - `split_manifest.json`
  - `comparison_summary.json`
  - `comparison_summary.csv`
  - `metrics_<profile>.json`
  - `model_profile_<profile>.json`
  - `comparison_<profile>_sample0.npz`
  - `comparison_<profile>_sample0.png`

## Required Deterministic Checks

These are mandatory gates from the selected-item context and remain required even if stronger focused checks are added.

```bash
pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

Recommended stronger focused check before long runs:

```bash
pytest tests/studies/test_pdebench_image128_runner.py -k 'history1 or reference_run_manifest or cross_run_compare' -v
```

## Task 1: Lock The Cross-History Compare Contract In Tests

**Files:**
- Modify: `tests/studies/test_pdebench_image128_runner.py`
- Modify: `scripts/studies/pdebench_image128/reporting.py`

- [ ] **Step 1: Add failing tests for the approved history-only delta**

Add tests that prove the compare sidecar:

- accepts a fresh history-1 run against history-2 reference rows only when these fields match exactly:
  - `dataset_file`
  - `split_counts`
  - `max_windows_per_trajectory`
  - `epochs`
  - `batch_size`
  - `training_loss`
  - `metric_family`
- records row-local contract fields for both the fresh history-1 rows and the history-2 anchors
- records the allowed delta explicitly (`history_len` and the derived sample-contract/input-channel change)
- rejects dataset, split, batch-size, training-loss, metric-family, or epoch mismatches
- rejects `hybrid_resnet_base` as a proxy anchor for the `40`-epoch `hybrid_resnet_cns` comparison

- [ ] **Step 2: Run the focused red slice**

Run:

```bash
pytest tests/studies/test_pdebench_image128_runner.py -k 'history1 or reference_run_manifest or cross_run_compare' -v
```

Expected: FAIL until the compare helper supports the approved cross-history contract.

## Task 2: Implement The Frozen Reference Manifest And Controlled Compare Sidecar

**Files:**
- Modify: `scripts/studies/pdebench_image128/reporting.py`
- Modify: `tests/studies/test_pdebench_image128_runner.py`

- [ ] **Step 1: Freeze the history-2 anchor manifest**

Use the existing row-local manifest machinery as the starting point and write:

` .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2_reference_runs.json`

The manifest must record:

- exact run roots
- `profile_id`
- `epochs`
- `dataset_file`
- `split_counts`
- `max_windows_per_trajectory`
- `history_len`
- `training_loss`
- `batch_size`
- `metric_family`
- `source_document`

- [ ] **Step 2: Extend reporting with a history-delta compare helper**

Implement a compare path that:

- loads the fresh history-1 run root and the frozen history-2 reference rows
- enforces exact contract parity on everything except the approved history delta
- writes top-level `compare_<budget>_against_history2.json/csv`
- renders galleries only when saved targets align
- preserves the capped decision-support boundary in its JSON payload

- [ ] **Step 3: Green the focused tests and then run the mandatory backlog checks**

Run:

```bash
pytest tests/studies/test_pdebench_image128_runner.py -k 'history1 or reference_run_manifest or cross_run_compare' -v
pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

**Verification for Task 2**

- The new focused tests pass.
- The two backlog-mandated deterministic checks pass.
- The reporting payload makes the history-only delta explicit instead of pretending the fresh and frozen runs were same-contract rows.

## Task 3: Audit Existing Anchors And Backfill The Missing `40`-Epoch `hybrid_resnet_cns` History-2 Row If Needed

**Files / artifacts:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2_reference_runs.json`
- Create only if needed: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-<timestamp>/`

- [ ] **Step 1: Audit the frozen history-2 roots against the manifest**

Verify the required artifact set exists for the reused `10`-epoch and `40`-epoch history-2 roots and that each reused row matches its recorded contract.

- [ ] **Step 2: Treat the existing history-1 readiness root as background only**

Record that:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history1-10ep-20260422T193524Z`
  is useful as smoke/provenance background
- it does **not** satisfy the final ranked `10`-epoch tranche because it ran in `mode=readiness`

- [ ] **Step 3: If no exact `40`-epoch `hybrid_resnet_cns` history-2 anchor exists, run the bounded backfill**

Launch in `tmux`, activate `ptycho311`, track the exact launched PID, and wait on that PID. Use a fresh output root.

Command shape:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-<timestamp> \
  --profiles hybrid_resnet_cns \
  --history-len 2 \
  --epochs 40 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8 \
  --device cuda \
  --num-workers 0
```

- [ ] **Step 4: Refresh the frozen reference manifest**

Record the reused or freshly backfilled `40`-epoch `hybrid_resnet_cns` root in `history2_reference_runs.json`.

**Verification for Task 3**

- Every reused history-2 row has the required run artifacts and the expected contract.
- The missing `40`-epoch `hybrid_resnet_cns` anchor is either found as an exact audited root or freshly backfilled.
- No spectral, FNO, or U-Net history-2 rows are rerun unnecessarily.

## Task 4: Run The Fresh `10`-Epoch Markov Four-Row Pilot

**Artifacts:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history1-pilot-10ep-<timestamp>/`
- Create: top-level `compare_10ep_against_history2.*`

- [ ] **Step 1: Launch the fresh history-1 pilot in `ptycho311`**

Launch in `tmux`, activate `ptycho311`, track the exact PID, and wait on that PID.

Command shape:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history1-pilot-10ep-<timestamp> \
  --profiles spectral_resnet_bottleneck_base,hybrid_resnet_cns,fno_base,unet_strong \
  --history-len 1 \
  --epochs 10 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8 \
  --device cuda \
  --num-workers 0
```

- [ ] **Step 2: Verify the fresh `10`-epoch run root**

Check that:

- `comparison_summary.json` reports `mode="pilot"`
- `evidence_scope="capped_decision_support_only"`
- `history_len=1`
- all four required `metrics_*.json` and `model_profile_*.json` files exist

- [ ] **Step 3: Write the `10`-epoch history1-vs-history2 compare sidecar**

Use the frozen `10`-epoch history-2 anchors and the new compare helper to write:

- `compare_10ep_against_history2.json`
- `compare_10ep_against_history2.csv`
- gallery PNGs if targets align, otherwise an explicit gallery blocker in the JSON payload

**Verification for Task 4**

- The fresh `10`-epoch pilot exits `0`.
- The run root contains all required artifacts.
- The compare sidecar makes the history-only delta explicit and includes all four fresh Markov rows plus the frozen `10`-epoch history-2 anchors.

## Task 5: Run The Fresh `40`-Epoch Markov Four-Row Pilot

**Artifacts:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history1-pilot-40ep-<timestamp>/`
- Create: top-level `compare_40ep_against_history2.*`

- [ ] **Step 1: Launch the fresh `40`-epoch history-1 pilot**

Launch in `tmux`, activate `ptycho311`, track the exact PID, and wait on that PID.

Command shape:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history1-pilot-40ep-<timestamp> \
  --profiles spectral_resnet_bottleneck_base,hybrid_resnet_cns,fno_base,unet_strong \
  --history-len 1 \
  --epochs 40 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8 \
  --device cuda \
  --num-workers 0
```

- [ ] **Step 2: Verify the fresh `40`-epoch run root**

Check the same artifact and pilot-mode requirements as Task 4.

- [ ] **Step 3: Write the `40`-epoch history1-vs-history2 compare sidecar**

Use the frozen `40`-epoch history-2 anchors, including the shell-locked `hybrid_resnet_cns` backfill from Task 3 if needed.

**Verification for Task 5**

- The fresh `40`-epoch pilot exits `0`.
- The run root contains all required artifacts.
- The compare sidecar includes the exact history-2 anchors required for a fair `40`-epoch interpretation.

## Task 6: Publish The Durable Markov Interpretation And Update State

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_markov_history1_compare_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/index.md`
- Modify: `docs/studies/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Modify only if warranted: `docs/findings.md`

- [ ] **Step 1: Write the durable Markov summary**

The summary must record:

- the fresh `10`-epoch and `40`-epoch history-1 pilot roots
- the frozen history-2 anchor roots
- whether the `40`-epoch `hybrid_resnet_cns` history-2 anchor had to be backfilled
- per-budget answers to:
  - whether `spectral_resnet_bottleneck_base` improved under `history_len=1`
  - whether the four-row ranking changed
- the explicit boundary that all evidence remains capped decision-support only

- [ ] **Step 2: Update the CNS summary**

Add a Markov-history subsection that points to the durable summary and clarifies how the history-1 lane relates to the earlier history-2 capped anchors.

- [ ] **Step 3: Update discoverability docs**

Update `docs/index.md` and `docs/studies/index.md` so the Markov compare is discoverable alongside the other CNS follow-up lanes.

- [ ] **Step 4: Update the progress ledger**

Record:

- this plan path
- the durable summary path
- the fresh artifact roots
- the history-2 anchor roots reused or backfilled
- the required deterministic checks and long-run commands that passed
- the capped decision-support interpretation boundary

- [ ] **Step 5: Add a finding only if the result is stable enough to be reusable outside this summary**

If the outcome is too context-specific or too dependent on capped pilot conditions, keep it in the summary and ledger only.

**Verification for Task 6**

- The durable summary, CNS summary, docs index, studies index, and progress ledger all cite the same plan path and artifact roots.
- The durable summary answers the backlog item’s ranking questions directly.
- The final write-up never presents the result as benchmark-complete CNS evidence.

## Completion Gate

- [ ] The required deterministic checks pass on the final code state.
- [ ] A fresh `10`-epoch `history_len=1` pilot root exists for the four required rows.
- [ ] A fresh `40`-epoch `history_len=1` pilot root exists for the four required rows.
- [ ] A fair `40`-epoch `history_len=2` `hybrid_resnet_cns` anchor exists, either reused or freshly backfilled.
- [ ] `compare_10ep_against_history2.*` and `compare_40ep_against_history2.*` exist with explicit history-delta metadata.
- [ ] `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_markov_history1_compare_summary.md` exists and answers whether spectral improved and whether the ranking changed.
