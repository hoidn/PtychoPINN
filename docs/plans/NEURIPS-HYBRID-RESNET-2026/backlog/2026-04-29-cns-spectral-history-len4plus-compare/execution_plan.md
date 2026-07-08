# CNS Spectral History Length 4+ Compare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the capped PDEBench CNS spectral row continues improving beyond `history_len=3`, quantify any gain or saturation against the frozen `history_len=2` and `history_len=3` anchors, and keep the result strictly as adjacent capped context rather than a headline-paper contract change.

**Architecture:** Treat this as a bounded Phase 2 CNS follow-up with three implementation units: first freeze and verify the exact capped compare contract plus existing reference anchors; second run fresh `history_len=4` spectral-only pilots and generate anchored compare sidecars against both frozen history lanes; third open a tightly defined `history_len=5` branch only if the recorded `history_len=4` result clears a predeclared improvement gate, then publish one durable summary and evidence-index updates without touching the locked CNS paper lane. The earlier `history_len=3` summary's closed `history_len=4` gate is background context only; this selected backlog item is the new authority that intentionally reopens the question for the spectral row.

**Tech Stack:** PATH `python`, PyTorch CNS runner under documented `ptycho311` for long CUDA runs, tmux with exact PID tracking, pytest, compileall, Markdown/JSON/CSV artifact outputs under `.artifacts/NEURIPS-HYBRID-RESNET-2026/`.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-29-cns-spectral-history-len4plus-compare`
- Selection mode: `ACTIVE_SELECTION`
- Date: `2026-05-01`
- Roadmap lane: bounded Roadmap Phase 2 PDEBench CNS follow-up
- Authoritative plan path: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-spectral-history-len4plus-compare/execution_plan.md`
- Durable summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_spectral_history_len4plus_compare_summary.md`
- Item artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/`
- This document supersedes any earlier plan for this backlog item and is the execution authority for the implementation phase.

## Selected Backlog Objective

- Run the PDEBench `2d_cfd_cns` `spectral_resnet_bottleneck_base` row with longer temporal context than the already completed `history_len=3` capped study.
- Start with `history_len=4`.
- Compare the fresh `history_len=4` row against the frozen audited `history_len=2` and `history_len=3` spectral anchors at matching reduced-budget settings.
- Report absolute metrics and deltas for `err_nRMSE`, `err_RMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`, runtime, train/eval sample counts, emitted-window counts, and valid raw windows per trajectory.
- Preserve the manuscript/repo mapping:
  manuscript label `SRU-Net*` == repo row `spectral_resnet_bottleneck_base`.
- Open `history_len=5` only if the predeclared gate says `history_len=4` still improves aggregate error without an unacceptable high-band regression, or if implementation records a concrete scientific reason before any `history_len=5` launch.

## Scope

### In Scope

- the official CNS file only:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- the spectral row only:
  `spectral_resnet_bottleneck_base`
- the same reduced capped contract family already used by the completed CNS history-length study:
  `512 / 64 / 64` trajectories, `max_windows_per_trajectory=8`, `mse`, batch size `4`, same metric family, same normalization policy, same optimizer/scheduler family
- fresh `history_len=4` pilots at the same epoch budgets used by the completed history-length study: `10` and `40`
- anchored compare payloads against frozen `history_len=2` and `history_len=3` spectral references
- optional `history_len=5` pilots at the same `10` and `40` budgets only if the gate opens
- one durable summary plus the required evidence-index and discoverability updates for new capped-ablation evidence

### Explicit Non-Goals

- Do not reopen the CNS paper headline lane. The locked paper contract remains `history_len=2`.
- Do not rerun `fno_base`, `unet_strong`, `author_ffno_cns_base`, or `hybrid_resnet_cns`.
- Do not mix history lengths in the headline CNS ranking table or claim that this work changes the locked table.
- Do not widen this item into autoregressive rollouts, full-training benchmarks, physics-regularization tuning, mode-count sweeps, shell sweeps, GNOT, authored FFNO, or broader architecture search.
- Do not change the official file, split counts, normalization rules, batch size, loss family, optimizer/scheduler family, or metric family unless a checked-in plan amendment also reruns all affected comparison anchors.
- Do not create `/home/ollie/Documents/neurips/` artifacts or manuscript prose.
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not create worktrees.

## Binding Constraints And Prerequisites

### Steering And Roadmap Constraints

- `docs/steering.md` requires equal-footing comparisons, forbids silently relaxing fairness constraints, and prefers work that strengthens current paper evidence without rewriting roadmap scope.
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md` still treats CNS as bounded capped decision-support evidence unless a later checked-in contract decision says otherwise.
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md` and `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_row_lock_summary.md` lock the headline CNS paper lane to `history_len=2`; this item must stay adjacent context only.
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history_len3plus_compare_summary.md` is the authoritative prerequisite for the prior longer-history pass. Its closed `history_len=4` gate is not a blocker here because the new selected backlog item explicitly authorizes the next follow-up.

### Fixed Local Compare Contract

Hold these surfaces fixed unless a checked-in plan amendment explicitly says otherwise and reruns all compared anchors:

- task: `2d_cfd_cns`
- dataset file:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- split: `512 / 64 / 64` trajectories
- `max_windows_per_trajectory=8`
- emitted windows for every compared lane:
  `4096 / 512 / 512`
- batch size: `4`
- training loss: `mse`
- optimizer: `Adam`, `lr=2e-4`
- scheduler:
  `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`
- metric family:
  `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- profile shell:
  `spectral_resnet_bottleneck_base` exactly as already defined for the CNS runner
- allowed delta:
  `history_len` and its derived input-channel/raw-window counts only

Derived history contracts that must be recorded explicitly:

- `history_len=2`:
  `input_channels=8`, raw `windows_per_trajectory=19`, raw `available_windows=190000`
- `history_len=3`:
  `input_channels=12`, raw `windows_per_trajectory=18`, raw `available_windows=180000`
- `history_len=4`:
  `input_channels=16`, raw `windows_per_trajectory=17`, raw `available_windows=170000`
- `history_len=5`:
  `input_channels=20`, raw `windows_per_trajectory=16`, raw `available_windows=160000`

Because the emitted capped windows stay fixed at `8` per trajectory for all four lanes, direct delta claims remain allowed on the emitted capped contract as long as inspect outputs confirm those counts.

### Prerequisite Status

Progress-ledger and checked-in prerequisite state already complete:

- Phase 0 evidence inventory
- Phase 1 PDE benchmark selection
- CNS adapter, readiness, and capped-comparison groundwork
- CNS paper-contract decision
- CNS paper row lock

Backlog-summary prerequisites already complete:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_markov_history1_compare_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history_len3plus_compare_summary.md`

Reference-anchor consequence for this item:

- frozen `history_len=2` and `history_len=3` spectral roots already exist for both `10` and `40` epochs
- the previous item's mixed `10`/`40` spectral signal is the reason this item must report both budgets again instead of pretending a single-budget answer is enough

### Execution Rules

- The backlog item's required deterministic checks are blocking before any expensive launch.
- If a normal import, path, pytest, compile, or harness failure occurs, diagnose it, apply the narrowest credible fix, and rerun the same check before considering the item blocked.
- Reserve `BLOCKED` for missing external artifacts, unavailable GPU/runtime resources, roadmap conflict, external dependency outside current authority, required user decision, or a failure still unrecoverable after a documented narrow fix attempt.
- Long runs remain implementation-owned until terminal success or documented recoverable failure handling is complete. Use tmux, activate `ptycho311`, track the exact launched PID, and do not launch duplicate jobs writing to the same `--output-root`.

## Concrete File And Artifact Targets

### Code And Tests To Verify First, Modify Only If A Real Gap Appears

- `scripts/studies/pdebench_image128/cfd_cns.py`
- `scripts/studies/pdebench_image128/reporting.py`
- `scripts/studies/run_pdebench_image128_suite.py`
- `tests/studies/test_pdebench_image128_runner.py`
- `tests/studies/test_pdebench_cfd_cns_data.py`
- `tests/studies/test_pdebench_cfd_cns_metrics.py`

### Mandatory Contract Outputs

- create:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_spectral_history_len4plus_compare_summary.md`
- update:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- update:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- update:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- update:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- item artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/`
- required item-local artifacts:
  - frozen reference manifest covering history-2 and history-3 spectral anchors at `10` and `40` epochs
  - `history4-inspect-<timestamp>/`
  - `history4-pilot-10ep-<timestamp>/`
  - `history4-pilot-40ep-<timestamp>/`
  - `compare_10ep_history4_against_history2_history3.json` and `.csv`
  - `compare_40ep_history4_against_history2_history3.json` and `.csv`
  - `history5_gate_decision.json`
  - if the gate opens:
    `history5-inspect-<timestamp>/`,
    `history5-pilot-10ep-<timestamp>/`,
    `history5-pilot-40ep-<timestamp>/`,
    `compare_10ep_history5_against_history2_history3_history4.json` and `.csv`,
    `compare_40ep_history5_against_history2_history3_history4.json` and `.csv`

### Preferred Packaging And Discoverability Updates

- update:
  `docs/studies/index.md`
- update:
  `docs/index.md`
- update `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md` only if this result needs an explicit paper-evidence-audit discoverability entry; otherwise record in the execution report why the capped adjacent-context item does not require a paper-evidence-index change

## Implementation Architecture

- **Contract And Anchor Unit:** owns deterministic checks, inspect gates, frozen reference manifests, and any narrow runner/reporting fixes required to support `history_len>=4` without contract drift.
- **Execution And Gate Unit:** owns fresh `history_len=4` pilots, the explicit `history_len=5` gate decision, and any conditional `history_len=5` pilots under the same fixed contract.
- **Reporting And Index Unit:** owns cross-history compare sidecars, manuscript-label mapping, durable summary text, and evidence-index/discoverability updates while preserving the locked `history_len=2` paper lane.

## Task Checklist

### Task 1: Freeze The Exact Contract And Verify Readiness Before Any Long Run

**Files:**

- verify first:
  `scripts/studies/pdebench_image128/cfd_cns.py`
- verify first:
  `scripts/studies/pdebench_image128/reporting.py`
- verify first:
  `scripts/studies/run_pdebench_image128_suite.py`
- test:
  `tests/studies/test_pdebench_image128_runner.py`
- test:
  `tests/studies/test_pdebench_cfd_cns_data.py`
- test:
  `tests/studies/test_pdebench_cfd_cns_metrics.py`

- [ ] Reconfirm from the runner/config surfaces that `spectral_resnet_bottleneck_base` is still the intended CNS spectral row and that no shell/config drift beyond `history_len` will be introduced.
- [ ] Materialize an item-local frozen reference manifest for the spectral anchors at `history_len=2` and `history_len=3`, covering both `10` and `40` epochs and recording source run roots, metrics paths, split manifests, and label mapping `SRU-Net* -> spectral_resnet_bottleneck_base`.
- [ ] Run the backlog item's required deterministic checks and archive the logs before any expensive launch:

```bash
pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

- [ ] If a required check fails, make the narrowest credible fix, extend only the targeted tests needed to prove the fix, and rerun the same required commands before moving on.
- [ ] Run an exact-contract inspect for `history_len=4` and verify:
  official file, split counts `512 / 64 / 64`, batch size `4`, `mse`, emitted windows `4096 / 512 / 512`, `input_channels=16`, and raw `windows_per_trajectory=17`.
- [ ] Run a pre-gate inspect for `history_len=5` only to prove the future contract shape if the gate opens; do not treat inspect as authorization to launch the pilot.
- [ ] If inspect reveals a real gap for `history_len>=4` handling or compare sidecar generation, patch only the narrow runner/reporting surface that blocks execution.

**Verification after Task 1**

- [ ] **Blocking:** the two required backlog-item commands above must pass before any fresh long run starts.
- [ ] **Supporting:** validate the inspect artifacts show `history_len=4` as `input_channels=16`, raw `17` windows per trajectory, and fixed emitted capped counts; validate the `history_len=5` inspect shows `input_channels=20` and raw `16` windows per trajectory.

### Task 2: Prepare Or Repair The Cross-History Reporting Surface

**Files:**

- modify only if a gap is proven:
  `scripts/studies/pdebench_image128/reporting.py`
- modify only if a gap is proven:
  `scripts/studies/pdebench_image128/cfd_cns.py`
- modify only if a gap is proven:
  `scripts/studies/run_pdebench_image128_suite.py`
- test if code changes are needed:
  `tests/studies/test_pdebench_image128_runner.py`
- test if code changes are needed:
  `tests/studies/test_pdebench_cfd_cns_metrics.py`

- [ ] Reuse the existing history-compare sidecar pattern from the `history_len=3` summary instead of inventing a new reporting format.
- [ ] Ensure the compare payload for `history_len=4` can report absolute metrics plus deltas against both frozen anchors:
  `history_len=2` and `history_len=3`.
- [ ] Require the compare payload to carry:
  `history_len`, repo row id, manuscript label, runtime, train/val/test trajectories, emitted windows, raw windows per trajectory, metric values, metric deltas, and a claim-scope field such as `adjacent_capped_context_only`.
- [ ] If the code path currently assumes only one reference lane, extend it narrowly so `history_len=4` and optional `history_len=5` can compare against multiple frozen anchors without changing the underlying CNS metric contract.
- [ ] If code changes are made, rerun the required backlog-item checks before any fresh pilot launch.

**Verification after Task 2**

- [ ] **Blocking if code changed:** rerun
  `pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py`
  and
  `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`.
- [ ] **Supporting:** add or extend tests so multi-anchor compare payloads preserve metric keys, label mapping, runtime/sample-count fields, and claim-scope labeling.

### Task 3: Run `history_len=4` At The Fixed `10`- And `40`-Epoch Budgets

**Files:**

- execution entrypoint:
  `scripts/studies/run_pdebench_image128_suite.py`
- supporting runner surface:
  `scripts/studies/pdebench_image128/cfd_cns.py`
- item artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/`

- [ ] Launch one fresh `history_len=4`, `10`-epoch spectral-only capped pilot under the exact fixed contract:
  official file, `512 / 64 / 64`, `max_windows_per_trajectory=8`, batch size `4`, `mse`, `Adam lr=2e-4`, plateau scheduler, and the existing spectral shell.
- [ ] Launch one fresh `history_len=4`, `40`-epoch spectral-only capped pilot under the same exact contract.
- [ ] Use tmux and the `ptycho311` environment for both runs, track the exact launched PID, and treat the run as complete only when exit code is `0` and the expected metrics/manifests are freshly written.
- [ ] Generate anchored compare sidecars against the frozen spectral anchors for the same epoch budgets:
  `compare_10ep_history4_against_history2_history3.*`
  and
  `compare_40ep_history4_against_history2_history3.*`
- [ ] Record the first regressed metric if `history_len=4` does not improve, not just the best-looking metric.
- [ ] Keep the result row-local. Do not let these runs trigger new FNO/U-Net/authored-FFNO launches.

**`history_len=5` gate definition**

- [ ] Record the gate in `history5_gate_decision.json` before any `history_len=5` launch.
- [ ] Open the gate only if the fresh `40`-epoch `history_len=4` spectral row improves both `err_nRMSE` and `err_RMSE` versus the fresh or frozen `40`-epoch `history_len=3` spectral anchor and does not worsen `fRMSE_high`.
- [ ] Treat the `10`-epoch `history_len=4` direction as supporting context only. It must be reported, but it does not by itself close the gate if the `40`-epoch row is cleanly better.
- [ ] If implementation wants to open the gate for a different scientific reason, write that reason into the gate JSON before launching `history_len=5`.
- [ ] If the gate stays closed, stop after the `history_len=4` compare and write the saturation or mixed-signal conclusion explicitly.

**Verification after Task 3**

- [ ] **Supporting:** validate the two fresh `history_len=4` roots each have `invocation.json`, `dataset_manifest.json`, `split_manifest.json`, `model_profile_*.json`, `metrics_*.json`, and tracked launcher exit proof.
- [ ] **Supporting:** validate the `10`- and `40`-epoch compare sidecars exist and each include deltas against both `history_len=2` and `history_len=3`.
- [ ] **Supporting:** validate `history5_gate_decision.json` exists even when the gate is closed.

### Task 4: If And Only If The Gate Opens, Run `history_len=5` Under The Same Contract

**Files:**

- execution entrypoint:
  `scripts/studies/run_pdebench_image128_suite.py`
- item artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/`

- [ ] If the gate opened, run a `history_len=5` inspect artifact first and confirm the same fixed capped contract with derived `input_channels=20` and raw `windows_per_trajectory=16`.
- [ ] Run fresh spectral-only `history_len=5` pilots at both `10` and `40` epochs under the same capped contract family used for histories `2`, `3`, and `4`.
- [ ] Generate anchored compare sidecars against the frozen or fresh spectral anchors:
  `history_len=2`, `history_len=3`, and `history_len=4`.
- [ ] State explicitly whether `history_len=5` continues the improvement trend, saturates, or regresses, and name the first regressed metric if it regresses.
- [ ] If `history_len=5` fails for a recoverable harness/runtime reason, fix narrowly and rerun; do not mark the whole item blocked unless the failure remains unrecoverable after the narrow fix attempt.

**Verification after Task 4**

- [ ] **Supporting:** validate the fresh `history_len=5` roots and `compare_*_history5_against_history2_history3_history4.*` files exist and carry the same metric/sample/runtime fields as the `history_len=4` payloads.
- [ ] **Supporting:** validate the summary logic can distinguish `continues_improving`, `mixed_signal`, and `saturated_or_regressed`.

### Task 5: Publish The Durable Summary And Evidence-Index Updates

**Files:**

- create:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_spectral_history_len4plus_compare_summary.md`
- update:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- update:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- update:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- update:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- preferred discoverability updates:
  `docs/studies/index.md`,
  `docs/index.md`,
  and optionally `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`

- [ ] Write the durable summary as the scientific authority for this item.
- [ ] The summary must include:
  - exact fixed contract
  - frozen anchor roots used
  - fresh run roots and exit-proof paths
  - per-budget absolute metrics and deltas
  - raw valid-window shrinkage by history length
  - repo-row/manuscript-label mapping
  - gate decision and rationale
  - explicit statement that the result is `adjacent capped context only` and does not change the locked `history_len=2` paper lane
- [ ] Update `pdebench_2d_cfd_cns_summary.md` so it points to the new history-4-plus summary and preserves the existing paper-contract language unchanged.
- [ ] Update `evidence_matrix.md` so the CNS history-length family reflects the new `history_len=4` outcome and, if applicable, the conditional `history_len=5` outcome.
- [ ] Update `ablation_index.json` for the CNS history-length family and `model_variant_index.json` for any new evaluated history-specific spectral rows.
- [ ] Update `docs/studies/index.md` and `docs/index.md` if the new summary should be discoverable from the CNS study index and top-level docs hub.
- [ ] Update `paper_evidence_index.md` only if the result meaningfully changes manuscript-facing evidence navigation; otherwise record why no paper-evidence-index update was needed.

**Verification after Task 5**

- [ ] **Blocking if code changed late in the item:** rerun the backlog-item required commands once more before closeout:

```bash
pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

- [ ] **Supporting:** run a final sync check that the durable summary, CNS summary, evidence matrix, and machine-readable indexes all agree on:
  item id,
  artifact root,
  history lanes actually executed,
  claim scope,
  and whether the `history_len=5` branch opened or stayed closed.
