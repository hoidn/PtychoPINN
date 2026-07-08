# PDEBench CNS Markov History-1 Compare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce capped decision-support `history_len=1` CNS comparisons at `10` and `40` epochs for `spectral_resnet_bottleneck_base`, `hybrid_resnet_cns`, `fno_base`, and `unet_strong`, then document how those rows differ from the frozen `history_len=2` anchors without changing the local CNS contract beyond the history window.

**Architecture:** Keep `scripts/studies/pdebench_image128/cfd_cns.py` authoritative for CNS data loading, split logic, metrics, and artifact writing; use `pilot` mode for the fresh ranked `history_len=1` runs because the roadmap now distinguishes ranked capped evidence from smoke/readiness. Reporting must explicitly permit only the approved contract delta (`history_len`, derived sample contract, and derived input-channel count) across fresh-vs-anchor comparisons while enforcing equality for dataset file, split counts, capped window budget, epochs, batch size, training loss, and metric family.

**Tech Stack:** PATH `python`, `ptycho311` for long CNS runs, PyTorch/Lightning, `scripts/studies/pdebench_image128/`, pytest, compileall, tmux for long-running commands, Markdown/JSON/CSV/PNG artifacts under `.artifacts/`.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-21-pdebench-cns-markov-history1-compare`
- Plan authority date: `2026-04-27`
- Scope owner: roadmap Phase 2 capped CNS follow-up lane
- Selected-item context: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-21-pdebench-cns-markov-history1-compare/selected-item-context.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/`
- Durable summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_markov_history1_compare_summary.md`
- CNS summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`

This document supersedes prior background material for this backlog item and is the execution authority for implementation.

## Objective

- Determine whether switching the capped local CNS one-step contract from `history_len=2` to Markov-style `history_len=1` changes the ranking among `spectral_resnet_bottleneck_base`, `hybrid_resnet_cns`, `fno_base`, and `unet_strong`.
- Answer the narrower scientific question explicitly at both `10` and `40` epochs:
  - does `spectral_resnet_bottleneck_base` improve under `history_len=1`?
  - does the four-row ranking change relative to the frozen `history_len=2` anchor family?

## Scope

- Keep the current capped CNS slice fixed.
- Change only the history contract from `history_len=2` to `history_len=1` for the fresh compare.
- Rerun the full four-row set at `10` and `40` epochs under `pilot` mode:
  - `spectral_resnet_bottleneck_base`
  - `hybrid_resnet_cns`
  - `fno_base`
  - `unet_strong`
- Freeze and audit the exact `history_len=2` anchor roots required for those budgets.
- Backfill only the missing shell-locked `40`-epoch `history_len=2` `hybrid_resnet_cns` anchor if no exact audited root is available.
- Write durable interpretation against the existing `history_len=2` anchors.

## Explicit Non-Goals

- Do not widen this work into rollout or autoregressive evaluation.
- Do not widen this work into author-FFNO, GNOT, hybrid-spectral architecture, modes-32, physics regularization, Darcy, SWE, or any other backlog lane.
- Do not change optimizer family, scheduler family, training loss, dataset path, split counts, `max_windows_per_trajectory`, batch size, or metric family for the fresh `history_len=1` rows.
- Do not treat `hybrid_resnet_base` `40`-epoch artifacts as a proxy for `hybrid_resnet_cns`; the shell differs and would confound the Markov comparison.
- Do not promote this item’s outputs into benchmark-complete CNS evidence, full-suite PDE evidence, or manuscript artifacts under `/home/ollie/Documents/neurips/`.
- Do not touch `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not create worktrees.

## Steering, Roadmap, And Fairness Constraints

- Steering requires explicit equal-footing comparisons and forbids silently relaxing fairness constraints.
- The roadmap allows this as a bounded Phase 2 capped CNS follow-up lane, not as a full-training benchmark tranche.
- The fresh ranked rows must use `pilot` mode, not `readiness`, because they are intended to answer an internal ranking question on a fixed capped contract.
- Existing `history_len=2` readiness roots may be reused as frozen legacy anchors when they match the fixed contract, but every summary must label them as capped decision-support evidence only.
- The only scientific variable across the frozen history-2 anchors and the fresh history-1 rows is the temporal-context contract:
  - history-2: `concat u[t-2:t] -> u[t]`, `input=(8,128,128)`, `target=(4,128,128)`
  - history-1: `concat u[t-1:t] -> u[t]`, `input=(4,128,128)`, `target=(4,128,128)`
- Every other field must stay fixed unless the run is the single allowed missing-anchor backfill:
  - dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
  - split: `512 / 64 / 64` trajectories
  - `max_windows_per_trajectory=8`
  - batch size: `4`
  - training loss: `mse`
  - metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`

## Prerequisite Status

- Already satisfied:
  - the official `2d_cfd_cns` file is staged and checksum-verified
  - the CNS loader supports both `history_len=2` and `history_len=1`
  - the canonical local CNS Hybrid row is `hybrid_resnet_cns` with skip-add and `pixelshuffle`
  - the repo already distinguishes `pilot` vs `readiness` evidence for capped CNS runs
  - exact capped history-2 anchors exist for all four rows at `10` epochs
  - exact capped history-2 anchors exist for `spectral_resnet_bottleneck_base`, `fno_base`, and `unet_strong` at `40` epochs
- Still missing:
  - no exact shell-locked `40`-epoch `history_len=2` `hybrid_resnet_cns` anchor has been confirmed in the artifact tree
  - this single anchor must be found or backfilled before the final `40`-epoch history-1 interpretation is accepted
- Background only, not sufficient:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history1-10ep-20260422T193524Z` matches the scientific `history_len=1` contract but ran in `mode=readiness`; treat it only as provenance/smoke background

## Frozen History-2 Anchors

### Required `10`-Epoch Anchor Roots

- `spectral_resnet_bottleneck_base` and `hybrid_resnet_cns`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
- `fno_base` and `unet_strong`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`

### Required `40`-Epoch Anchor Roots

- `spectral_resnet_bottleneck_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`
- `fno_base` and `unet_strong`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`
- `hybrid_resnet_cns`:
  not yet confirmed; either find an exact audited root or backfill only this one row under the fixed history-2 contract

### Forbidden Proxy

- Never use the older `hybrid_resnet_base` `40`-epoch history-2 row as the comparator for `hybrid_resnet_cns`.

## Implementation Architecture

- **Reference/compare unit:** `scripts/studies/pdebench_image128/reporting.py` and `tests/studies/test_pdebench_image128_runner.py` own the reference-run manifest, the cross-history compare payload, and the rule that only the approved history-contract delta is allowed.
- **Execution unit:** `scripts/studies/pdebench_image128/cfd_cns.py` and `scripts/studies/run_pdebench_image128_suite.py` own the fresh `pilot` runs and the contingency backfill of the missing `40`-epoch history-2 hybrid anchor.
- **Interpretation unit:** the new Markov summary, CNS summary update, discoverability docs, and progress ledger own the durable result and must preserve the capped decision-support boundary.

## Concrete File And Artifact Targets

### Code And Test Surfaces

- Modify: `scripts/studies/pdebench_image128/reporting.py`
- Modify: `tests/studies/test_pdebench_image128_runner.py`
- Modify only if run artifacts still lack metadata needed by the compare helper: `scripts/studies/pdebench_image128/cfd_cns.py`

### Durable Docs And State

- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_markov_history1_compare_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/index.md`
- Modify: `docs/studies/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Modify only if the outcome rises to a durable finding-level rule: `docs/findings.md`

### Required New Artifacts

- Create study root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/`
- Create manifest: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2_reference_runs.json`
- Create if needed: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-<timestamp>/`
- Create fresh run roots:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history1-pilot-10ep-<timestamp>/`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history1-pilot-40ep-<timestamp>/`
- Create top-level compare sidecars:
  - `compare_10ep_against_history2.json`
  - `compare_10ep_against_history2.csv`
  - `compare_10ep_against_history2_sample0.png` if targets align
  - `compare_10ep_against_history2_sample0_error.png` if targets align
  - `compare_40ep_against_history2.json`
  - `compare_40ep_against_history2.csv`
  - `compare_40ep_against_history2_sample0.png` if targets align
  - `compare_40ep_against_history2_sample0_error.png` if targets align

## Required Deterministic Checks

The selected backlog item makes these mandatory unless a stronger replacement is explicitly justified. This plan keeps them unchanged and adds narrower checks ahead of long runs.

```bash
pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

Focused pre-run check:

```bash
pytest tests/studies/test_pdebench_image128_runner.py -k 'history1 or reference_run_manifest or cross_run_compare' -v
```

Testing evidence requirement:

- Archive the pytest output used to claim success under the active plan hub or linked artifact location, per `docs/TESTING_GUIDE.md`.

## Task 1: Lock The Cross-History Compare Contract

**Files:**
- Modify: `tests/studies/test_pdebench_image128_runner.py`
- Modify: `scripts/studies/pdebench_image128/reporting.py`

- [ ] Add or tighten tests that prove the cross-history compare accepts only the approved delta:
  - fixed-equality fields must match exactly: `dataset_file`, `split_counts`, `max_windows_per_trajectory`, `epochs`, `batch_size`, `training_loss`, `metric_family`
  - allowed differences are limited to `history_len`, derived `sample_contract`, and derived `input_channels`
- [ ] Keep the explicit guard that rejects `hybrid_resnet_base` as a proxy reference for the `hybrid_resnet_cns` row.
- [ ] Ensure the reference-run manifest records every row-local contract field needed to audit reused anchors.
- [ ] Ensure the compare payload writes merged JSON/CSV outputs and only renders galleries when targets align exactly across the compared runs.
- [ ] Run the focused runner selector and confirm it fails before the implementation if new assertions were added first.
- [ ] Implement the minimal reporting changes to satisfy the new tests.
- [ ] Re-run the focused runner selector until it passes.

**Verification for Task 1**

```bash
pytest tests/studies/test_pdebench_image128_runner.py -k 'history1 or reference_run_manifest or cross_run_compare' -v
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

## Task 2: Freeze And Audit The History-2 Anchor Manifest

**Files:**
- Reuse emitted artifacts from Task 1 helper behavior
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2_reference_runs.json`

- [ ] Audit each required history-2 anchor root for the required artifacts:
  - `invocation.json`
  - `dataset_manifest.json`
  - `split_manifest.json`
  - `comparison_summary.json`
  - `metrics_<profile>.json`
  - `model_profile_<profile>.json`
- [ ] Confirm every reused anchor matches the fixed contract and the expected epoch budget.
- [ ] Record those anchors in `history2_reference_runs.json` using the reporting helper format so later compare steps consume a frozen manifest instead of ad hoc paths.
- [ ] If an exact `40`-epoch `hybrid_resnet_cns` history-2 root is found during audit, add it to the manifest and skip the backfill task below.
- [ ] If not found, leave the manifest and plan state explicitly showing that only the `40`-epoch hybrid anchor remains missing.

**Verification for Task 2**

- [ ] Manually inspect the emitted `history2_reference_runs.json` and confirm it contains:
  - both budget buckets: `10ep` and `40ep`
  - the four expected rows at `10ep`
  - the three confirmed rows plus either a found hybrid row or an explicit missing note at `40ep`

## Task 3: Backfill The Missing `40`-Epoch History-2 Hybrid Anchor If Required

**Files:**
- Reuse: `scripts/studies/run_pdebench_image128_suite.py`
- Reuse: `scripts/studies/pdebench_image128/cfd_cns.py`
- Artifact only if needed: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-<timestamp>/`

- [ ] Skip this task entirely if Task 2 finds an exact audited anchor.
- [ ] If backfill is required, launch exactly one `hybrid_resnet_cns` `history_len=2` `40`-epoch run under the fixed capped contract and record it as capped decision-support evidence.
- [ ] Use tmux for the long run, activate `ptycho311`, and obey the long-run guardrail:
  - track the exact PID
  - do not duplicate a run into the same output root
  - treat completion as `exit 0` plus fresh required artifacts present
- [ ] Keep the shell identical to the canonical CNS row: skip-add plus `pixelshuffle`.
- [ ] Add the resulting run root to `history2_reference_runs.json`.

**Verification for Task 3**

- [ ] Confirm the backfill root contains the standard CNS run artifacts plus per-profile metrics and comparison sample outputs.
- [ ] Confirm the resulting `comparison_summary.json` is `mode: pilot` or, if existing runner semantics force another mode, document why and keep the final summary explicit about capped-only evidence.

## Task 4: Run The Fresh `history_len=1` Pilot Compares

**Files:**
- Reuse: `scripts/studies/run_pdebench_image128_suite.py`
- Reuse: `scripts/studies/pdebench_image128/cfd_cns.py`
- Fresh artifacts:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history1-pilot-10ep-<timestamp>/`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history1-pilot-40ep-<timestamp>/`

- [ ] Run the full four-row shell at `10` epochs with `history_len=1` under `pilot` mode.
- [ ] Run the full four-row shell at `40` epochs with `history_len=1` under `pilot` mode.
- [ ] Do not rerun only spectral; the backlog question requires the full four-row ranking.
- [ ] Keep dataset, split counts, capped window count, batch size, loss, and metric family identical to the history-2 anchors.
- [ ] Capture invocation and runtime provenance in the normal run-root artifacts.
- [ ] Use tmux and the PID-based long-run guardrail for both runs.

**Suggested commands**

Use the PDEBench data root parent, not the HDF5 file path itself, because the
`2d_cfd_cns` runner resolves
`<data-root>/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
internally.

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --profiles spectral_resnet_bottleneck_base,hybrid_resnet_cns,fno_base,unet_strong \
  --history-len 1 \
  --epochs 10 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8 \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history1-pilot-10ep-<timestamp>
```

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --profiles spectral_resnet_bottleneck_base,hybrid_resnet_cns,fno_base,unet_strong \
  --history-len 1 \
  --epochs 40 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8 \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history1-pilot-40ep-<timestamp>
```

**Verification for Task 4**

- [ ] Confirm both fresh run roots contain `invocation.json`, `invocation.sh`, `dataset_manifest.json`, `split_manifest.json`, `comparison_summary.json`, `comparison_summary.csv`, per-profile metrics, and sample outputs.
- [ ] Confirm both runs record `history_len=1`, the correct sample contract, and `mode=pilot`.

## Task 5: Write Cross-History Compare Sidecars

**Files:**
- Reuse: `scripts/studies/pdebench_image128/reporting.py`
- Create top-level artifacts in the Markov study root

- [ ] Use the frozen reference manifest plus the fresh history-1 run roots to emit:
  - `compare_10ep_against_history2.json`
  - `compare_10ep_against_history2.csv`
  - `compare_40ep_against_history2.json`
  - `compare_40ep_against_history2.csv`
- [ ] Render `sample0` prediction/error galleries only if targets align exactly; otherwise preserve the blocker payload without treating it as run failure.
- [ ] Ensure the compare payload states that the fresh rows are `fresh_history1` and the reused anchors are `reference_history2`.
- [ ] Ensure the compare payload preserves the explicit allowed-delta explanation and the capped decision-support evidence boundary.

**Verification for Task 5**

- [ ] Open each compare JSON and confirm the merged rows include all four fresh history-1 rows and the correct reference family.
- [ ] Confirm the payload records `cross_run_gallery_blocked: null` only when targets align; otherwise confirm the blocker reason is explicit and non-fatal.

## Task 6: Publish Durable Interpretation And Queue State

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_markov_history1_compare_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/index.md`
- Modify: `docs/studies/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Modify only if warranted: `docs/findings.md`

- [ ] Write the new durable summary with these required statements:
  - whether `spectral_resnet_bottleneck_base` improved at `10` and `40` epochs under `history_len=1`
  - whether row ranking changed relative to history-2 at each budget
  - that the compare remains capped decision-support evidence only
  - whether the missing `40`-epoch hybrid history-2 anchor was found or backfilled
- [ ] Update the CNS summary so future selectors can see the Markov compare result without rereading raw artifacts.
- [ ] Update `docs/index.md` and `docs/studies/index.md` so the new summary is discoverable.
- [ ] Update `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` with the selector-relevant decision and artifact paths.
- [ ] Add a finding to `docs/findings.md` only if the Markov history change yields a stable reusable rule at the same level as other CNS findings; otherwise keep the conclusion summary-local.

**Verification for Task 6**

- [ ] Re-open the summary and confirm it names the exact fresh run roots, reference manifest path, and compare sidecars.
- [ ] Re-open the progress ledger entry and confirm it points to the new durable summary and records capped decision-support scope, not benchmark completion.

## Final Verification Gate

- [ ] Run the mandatory deterministic checks:

```bash
pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

- [ ] Verify all fresh or updated artifact paths referenced by the durable summary exist.
- [ ] Capture the passing pytest log in the active evidence location before claiming completion.

## Completion Criteria

- [ ] The history-only fairness boundary is enforced by tests and reporting helpers.
- [ ] A frozen audited history-2 reference manifest exists.
- [ ] An exact history-2 hybrid `40`-epoch anchor is either found or backfilled under the fixed shell.
- [ ] Fresh `history_len=1` four-row pilot compares exist for `10` and `40` epochs.
- [ ] Cross-history compare sidecars document the fresh-vs-anchor result for both budgets.
- [ ] Durable summary, CNS summary, discoverability docs, and progress ledger are updated.
- [ ] Every conclusion is explicitly labeled as capped decision-support evidence only.
