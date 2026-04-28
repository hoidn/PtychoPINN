# PDEBench CNS Spectral Modes-32 Compare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the selected capped PDEBench `2d_cfd_cns` modes-32 follow-up by reusing the landed modes-32 profile and recovered artifacts, resolving one authoritative `40`-epoch modes-32 row under the unchanged capped contract, and publishing the anchored compare plus durable summary/state updates without widening beyond this decision-support lane.

**Architecture:** Treat this item as a reuse-first resume, not a fresh feature build. First re-audit the already-landed code/test surface and the current artifact tree under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/`, including the recovered `10`-epoch row, the emitted anchored `10`-epoch compare, the tracked `40`-epoch root, and the roadmap routing state that selected this bounded CNS lane. The roadmap still keeps this work inside Phase 2 capped decision-support scope only, so the plan must not drift into benchmark-complete or full-suite claims; within that boundary, the selected item authorizes finishing the bounded `10`/`40`-epoch modes-32 compare. If the tracked `40`-epoch root is unusable, launch at most one replacement `40`-epoch run under the same capped contract; if a concrete execution or contract blocker prevents that, stop blocked and write it to the workflow-owned progress/state surfaces named below.

**Tech Stack:** PATH `python`, `pytest`, `compileall`, PyTorch in the documented `ptycho311` environment for long CNS runs, tmux with exact PID tracking, JSON/CSV/PNG artifacts, and repo-local `docs/plans/`, `.artifacts/`, and `state/` updates.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-21-pdebench-cns-spectral-modes32-compare`
- Status: pending resume from recovered in-progress state; bounded capped follow-up execution is authorized by the selected-item context and roadmap routing state
- Date: `2026-04-27`
- Roadmap lane: Phase 2 capped CNS decision-support compare
- Authoritative plan path: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-pdebench-cns-spectral-modes32-compare/execution_plan.md`
- Durable summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes32_compare_summary.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/`

## Inputs Read

- `AGENTS.md`
- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/development/TEST_SUITE_INDEX.md`
- `docs/model_baselines.md`
- `docs/studies/index.md`
- `docs/workflows/pytorch.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/steering.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_design.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_weight_sharing_summary.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-21-pdebench-cns-spectral-modes32-compare/selected-item-context.md`
- `.artifacts/review/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-pdebench-cns-spectral-modes32-compare-plan-review.json`
- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-pdebench-cns-spectral-modes32-compare/progress_report.md`
- `scripts/studies/pdebench_image128/{run_config.py,reporting.py,cfd_cns.py,models.py}`
- `scripts/studies/run_pdebench_image128_suite.py`
- `tests/studies/{test_pdebench_image128_models.py,test_pdebench_image128_runner.py}`

## Selected Backlog Objective

- Increase both encoder and bottleneck spectral modes from `12` to `32` for the shared spectral CNS variant.
- Keep the current capped CNS slice and training contract fixed.
- Compare the higher-mode row against only the current shared `12/12` spectral row plus the existing `fno_base` and `unet_strong` anchors.
- Produce a durable modes-32 summary and completion-state updates after the `10`-epoch and `40`-epoch anchored compares are both available.

## Scope

- Reuse the already-implemented `2d_cfd_cns` loader, split, normalization, MSE loss contract, denormalized metric family, and cross-run comparison/reporting machinery where audit confirms they remain correct.
- Reuse the recovered fresh `10`-epoch modes-32 run and the already-emitted anchored `10`-epoch compare if they satisfy the fixed contract.
- Resolve the remaining `40`-epoch modes-32 row by reusing the current tracked run root if it proves authoritative under the fixed contract; if that root is unusable, launch at most one replacement run under the same capped contract and stop blocked only if a concrete execution or contract blocker prevents an authoritative result.
- Write the dedicated modes-32 summary, update the CNS summary and discoverability surfaces, and record completion in initiative state.

## Explicit Non-Goals

- Do not widen this into a full spectral-mode sweep.
- Do not change only one of `fno_modes` or `spectral_bottleneck_modes`.
- Do not mix this item with `spectral_resnet_bottleneck_noshare`, deeper bottlenecks, `history_len=1`, physics regularization, authored FFNO, or GNOT work.
- Do not rerun the frozen shared `12/12`, `fno_base`, or `unet_strong` reference rows unless a documented contract or artifact failure makes reuse impossible.
- Do not promote any capped row to benchmark-complete evidence.
- Do not create `/home/ollie/Documents/neurips/` artifacts or manuscript prose.
- Do not touch `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not create worktrees.

## Steering, Roadmap, And Fairness Constraints

- Steering requires equal-footing comparisons and forbids silently relaxing fairness constraints to make the item easier.
- The roadmap and PDEBench image-suite plan keep this work in Phase 2 capped decision-support scope only; no later evidence-bundle, manuscript, or full-training benchmark work belongs here.
- The PDEBench image-suite plan's Stage F keeps focused ablations downstream of full-training primary profiles as the general suite-level rule, but the more specific roadmap routing state and selected-item context already authorize this bounded capped CNS follow-up lane inside Phase 2.
- The selected backlog item is binding: both spectral mode knobs move together and all non-mode spectral settings remain fixed so the result stays attributable.
- This item stays outside the steering queue’s external-baseline priority rule: FFNO and GNOT are separate backlog items and must not be pulled into this plan.
- The roadmap routing state is explicit on this point:
  - `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md` allows the next selected PDE scope to be either the Darcy full-training benchmark tranche or a bounded CNS follow-up compare/ablation that stays capped and decision-support-only
  - the same roadmap revision identifies `modes32` as one of the separate capped CNS ablation lanes that reuse the already-available `history_len=2` anchors and should not be serialized behind the other CNS lanes
  - this plan therefore does not require any extra roadmap/progress-ledger exception to finish the already selected bounded modes-32 lane
- The fixed local compare contract is:
  - task: `2d_cfd_cns`
  - dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
  - split: `512 / 64 / 64` trajectories
  - `max_windows_per_trajectory=8`
  - `history_len=2`
  - training loss: `mse`
  - batch size: `4`
  - epochs: `10` and `40`
  - metrics: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- The fixed modes-32 shell is:
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
  - only the two mode knobs change: `fno_modes=32`, `spectral_bottleneck_modes=32`
- Existing findings and summaries constrain interpretation:
  - `PDEBENCH-CNS-SPECTRAL-40EP-001`: shared `12/12` spectral is the strongest recorded local capped `40`-epoch row against the older `hybrid_resnet_base`, `fno_base`, and `unet_strong` anchors.
  - `PDEBENCH-CNS-SPECTRAL-SHARE-001`: non-shared `12/12` improved over shared `12/12` at `10` epochs, but that remains a separate manual lane and must not be mixed into this item.

## Prerequisite Status

- Progress-ledger prerequisites are satisfied:
  - Phase 0 evidence inventory is complete.
  - Phase 1 PDE benchmark selection is complete.
  - No currently recorded blocked tranche prevents audit/reuse of this recovered item.
- Roadmap authorization status:
  - `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md` routing state explicitly authorizes either the Darcy full-training benchmark tranche or a bounded CNS follow-up compare/ablation as the next Phase 2 scope.
  - the same routing state names `modes32` as one of the separate capped CNS ablation lanes that reuse the verified `history_len=2` anchors and do not need to be serialized behind the other CNS lanes.
  - this authorization is narrow: it permits finishing this capped decision-support compare only. It does not satisfy the broader full-training benchmark-completeness gate and does not permit suite-level or benchmark-complete claims.
- CNS task prerequisites are already in place:
  - the official CNS HDF5 is staged and verified
  - the supervised real-channel CNS adapter, split manifests, normalization, and denormalized metric family are implemented
  - the capped MSE contract already exists for the shared `12/12`, `fno_base`, and `unet_strong` anchor rows
  - cross-run comparison/reporting machinery already exists in `scripts/studies/pdebench_image128/reporting.py`
- This item does not depend on authored FFNO or GNOT execution, and it does not need a new roadmap exception so long as it stays inside the already selected capped `modes32` lane.

## Current Recovered Execution State

The plan must respect the live recovered state instead of replaying completed setup work.

- Recovered code state:
  - `scripts/studies/pdebench_image128/run_config.py` already defines `spectral_resnet_bottleneck_modes32`.
  - `tests/studies/test_pdebench_image128_models.py` already contains modes-32 profile/fairness assertions.
  - `tests/studies/test_pdebench_image128_runner.py` already contains modes-32 emitted-profile and cross-run compare coverage.
- Recovered artifact state already present under the item artifact root:
  - `reference_runs_10ep.json`
  - `reference_runs_40ep.json`
  - fresh `10`-epoch run root:
    - `cns-spectral-modes32-10ep-20260428T010825Z`
  - anchored `10`-epoch compare sidecars and galleries:
    - `compare_10ep_against_existing.json`
    - `compare_10ep_against_existing.csv`
    - `compare_10ep_sample0.png`
    - `compare_10ep_sample0_error.png`
- Recovered `40`-epoch state observed while drafting this plan:
  - `cns-spectral-modes32-40ep-20260428T014226Z`: empty abandoned root
  - `cns-spectral-modes32-40ep-20260428T014306Z`: failed launch attempt containing only tracker/log files after the runner rejected a non-empty output root
  - `cns-spectral-modes32-40ep-20260428T014353Z`: provenance files present (`invocation.json`, manifests, profile, normalization, HDF5 metadata) but completion artifacts must be re-audited before reuse
  - `cns-spectral-modes32-40ep-20260428T014353Z.launch`: sibling tracker directory used to avoid the non-empty-root guard; re-read this tracker before deciding whether to wait, reuse, or relaunch
- `progress_report.md` recorded the live resume state at revision time:
  - the focused selectors and `compileall` checks were green after the recovered `10`-epoch reuse
  - the `014353Z` run was active under tracked tmux shell PID `215347` and Python PID `215359`
  - the tracker log had reached epoch `3`
  - implementers must re-read the tracker rather than trusting that those PIDs are still live
- Timestamp note:
  - the `20260428T...Z` artifact names are UTC timestamps and correspond to the evening of `2026-04-27` in `America/Los_Angeles`; do not treat them as evidence that the plan is referencing a future run

## Fixed Reference Rows

These roots are the authoritative reused anchors for this item unless implementation finds a documented contract or artifact mismatch.

### Required `10`-Epoch Reference Rows

- `spectral_resnet_bottleneck_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
- `fno_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`
- `unet_strong`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`

### Required `40`-Epoch Reference Rows

- `spectral_resnet_bottleneck_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`
- `fno_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`
- `unet_strong`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`

## Implementation Architecture

- **Audit And Reuse Unit:** verify the landed modes-32 profile/tests and classify the recovered artifact tree so already-complete work is reused, not replayed.
- **Single-Run Resolution Unit:** classify the tracked `40`-epoch modes-32 root under the unchanged capped contract, reuse it if valid, and otherwise use the single replacement-run path without widening beyond the selected capped lane.
- **Comparison And Publication Unit:** emit the anchored `40`-epoch compare from the frozen reference manifest, then publish the durable summary, discoverability updates, and completion-state changes.

## Concrete File And Artifact Targets

### Code And Tests

Reuse-first targets; modify only if the audit finds a real contract mismatch:

- `scripts/studies/pdebench_image128/run_config.py`
- `scripts/studies/pdebench_image128/reporting.py`
- `scripts/studies/pdebench_image128/cfd_cns.py`
- `scripts/studies/pdebench_image128/models.py`
- `scripts/studies/run_pdebench_image128_suite.py`
- `tests/studies/test_pdebench_image128_models.py`
- `tests/studies/test_pdebench_image128_runner.py`

### Durable Docs And State

- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes32_compare_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/studies/index.md`
- Modify: `docs/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Modify conditionally: `docs/findings.md`

### Required Artifacts

- Reuse: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/reference_runs_10ep.json`
- Reuse: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/reference_runs_40ep.json`
- Reuse: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_10ep_against_existing.json`
- Reuse: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_10ep_against_existing.csv`
- Reuse if valid: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/cns-spectral-modes32-10ep-20260428T010825Z/`
- Reuse if complete and valid: the tracked `40`-epoch run root under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/`
- Create only if the tracked root is unusable: one new authoritative `40`-epoch run root under the same artifact directory
- Create if absent: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_40ep_against_existing.json`
- Create if absent: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_40ep_against_existing.csv`

### Optional Artifacts

- Reuse or create only if sample alignment is valid:
  - `compare_40ep_sample0.png`
  - `compare_40ep_sample0_error.png`
- Leave failed or abandoned `40`-epoch roots in place for provenance; do not delete them as part of this backlog item.

### Required Check Commands

These are the backlog item’s minimum deterministic checks and must be rerun before completion even if the audit shows that no code edits were needed:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

## Task 1: Re-Audit The Modes-32 Surface, Roadmap Authorization, And Current Artifact State

**Files:**
- Verify: `docs/steering.md`
- Verify: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- Verify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- Verify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Verify: `scripts/studies/pdebench_image128/run_config.py`
- Verify: `tests/studies/test_pdebench_image128_models.py`
- Verify: `tests/studies/test_pdebench_image128_runner.py`
- Verify: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/reference_runs_{10ep,40ep}.json`
- Verify: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/cns-spectral-modes32-10ep-20260428T010825Z/`
- Verify: `.artifacts/review/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-pdebench-cns-spectral-modes32-compare-plan-review.json`
- Verify: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-pdebench-cns-spectral-modes32-compare/progress_report.md`
- Modify only if audit fails: `scripts/studies/pdebench_image128/{run_config.py,reporting.py,cfd_cns.py,models.py}`, `tests/studies/{test_pdebench_image128_models.py,test_pdebench_image128_runner.py}`

- [ ] **Step 1: Reconfirm that this backlog item is the roadmap-authorized bounded CNS lane**

Confirm from the roadmap, suite plan, steering, and progress ledger that:

- this backlog item is the selected capped CNS follow-up compare/ablation authorized by the roadmap routing state and selected-item context
- the authorization is narrow: the work remains capped and decision-support-only, and it must not be reframed as full-training benchmark evidence
- `modes32` remains a separate capped CNS lane that reuses the `history_len=2` anchors instead of widening into the history-contract, external-baseline, or hybrid-spectral architecture lanes

Record one of two explicit states before touching Task 2:

- `selected_capped_lane_authorized`
- `authorization_conflict_requires_human_review`

The expected state from the consumed artifacts is `selected_capped_lane_authorized`. If a newly discovered contradiction inside the same authoritative surfaces would require changing the selected backlog scope or roadmap order, stop blocked and record that contradiction instead of improvising a new scope.

- [ ] **Step 2: Verify the manual profile still matches the fixed fairness contract**

Confirm that `spectral_resnet_bottleneck_modes32`:

- exists
- remains outside `PRIMARY_CFD_CNS_PROFILE_IDS`, `READINESS_CFD_CNS_PROFILE_IDS`, and `PRIMARY_DARCY_PROFILE_IDS`
- keeps the shared spectral shell unchanged
- changes only `fno_modes: 12 -> 32` and `spectral_bottleneck_modes: 12 -> 32`
- preserves manual-only / decision-support evidence labeling

- [ ] **Step 3: Re-classify the recovered artifact tree before any launch decision**

Confirm that:

- the recovered `10`-epoch run root still records task `2d_cfd_cns`, `history_len=2`, `training_loss="mse"`, `epochs=10`, split `512 / 64 / 64`, `max_windows_per_trajectory=8`, and profile `spectral_resnet_bottleneck_modes32`
- `compare_10ep_against_existing.{json,csv}` already compare the fresh modes-32 row only against the shared `12/12`, `fno_base`, and `unet_strong` anchors
- `progress_report.md` and the live artifact tree agree that the `014353Z` root is the only candidate authoritative `40`-epoch root
- the `.launch` tracker paired with `cns-spectral-modes32-40ep-20260428T014353Z` is the first resume target to inspect
- the empty `014226Z` root and the failed `014306Z` tracker-only attempt are treated as provenance, not as resumable completion roots

- [ ] **Step 4: Run the deterministic checks before deciding whether any code patching is needed**

Run:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

If these fail, patch the smallest possible surface and rerun them before continuing.

**Verification:**

- Required check commands above pass.
- The plan records either `selected_capped_lane_authorized` or `authorization_conflict_requires_human_review`.
- The `10`-epoch run root and `compare_10ep_against_existing.json` both encode the fixed capped CNS contract.

## Task 2: Resolve The Existing Modes-32 `40`-Epoch Root Under The Authorized Capped Lane

**Files:**
- Reuse first: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/cns-spectral-modes32-40ep-20260428T014353Z/`
- Inspect tracker first: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/cns-spectral-modes32-40ep-20260428T014353Z.launch/`
- Preserve as failed provenance: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/cns-spectral-modes32-40ep-20260428T014306Z/`
- Create only if needed: one fresh replacement `40`-epoch run root under the same artifact directory

- [ ] **Step 1: Perform a deterministic contract audit on the tracked `014353Z` root**

Before deciding whether to wait, reuse, block, or relaunch, verify from the tracked root and tracker that:

- `.launch/python_pid.txt` and `.launch/tmux_shell_pid.txt` exist
- `invocation.json` records:
  - task `2d_cfd_cns`
  - mode `readiness`
  - profile `spectral_resnet_bottleneck_modes32`
  - epochs `40`
  - history length `2`
  - batch size `4`
  - `max_train_trajectories=512`
  - `max_val_trajectories=64`
  - `max_test_trajectories=64`
  - `max_windows_per_trajectory=8`
- `dataset_manifest.json` points to the official CNS HDF5 contract
- `split_manifest.json` records the fixed `512 / 64 / 64` trajectory split
- `model_profile_spectral_resnet_bottleneck_modes32.json` keeps the shared spectral shell unchanged and changes only `fno_modes=32` and `spectral_bottleneck_modes=32`

Do not treat the root as authoritative until both the contract audit and the completion audit below pass.

- [ ] **Step 2: Reuse, wait, block, or relaunch according to the audited state**

Branch explicitly:

- if the exact tracked Python PID is still active, wait on that run rather than launching a duplicate
- if the PID exited and `exit_code.txt` is `0`, only accept the run as complete when the output root also contains:
  - `metrics_spectral_resnet_bottleneck_modes32.json`
  - `comparison_summary.json`
  - `comparison_summary.csv`
  - `comparison_spectral_resnet_bottleneck_modes32_sample0.npz`
  - `comparison_spectral_resnet_bottleneck_modes32_sample0.png`
- if the tracked run is incomplete or failed and no concrete blocker remains, proceed to the replacement-run branch below
- if the tracked run is incomplete or failed and a concrete blocker prevents safe replacement under the same capped contract, stop here as blocked and write that blocker plus its return condition to:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-pdebench-cns-spectral-modes32-compare/progress_report.md`
  - `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-21-pdebench-cns-spectral-modes32-compare/implementation-phase/implementation_state.json`
  - `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-21-pdebench-cns-spectral-modes32-compare/implementation-phase/implementation_state.txt`

- [ ] **Step 3: Launch exactly one replacement run when the tracked root is unusable**

Guardrails:

- use tmux and the `ptycho311` environment for the long run
- keep tracker files in a sibling `.launch` directory, not inside the output root
- do not launch into an output root that already contains provenance or results
- do not run a broad `pgrep -f` polling loop; wait on the exact launched PID

Canonical run command:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode readiness \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/<fresh-40ep-root> \
  --profiles spectral_resnet_bottleneck_modes32 \
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

- [ ] **Step 4: Record the authoritative `40`-epoch run root or the explicit blocker**

Whichever `40`-epoch root passes both the contract audit and the completion gate becomes the only authoritative fresh modes-32 `40`-epoch row for the rest of this item. Any failed or abandoned roots remain referenced only as provenance notes. If no authoritative root exists after auditing the tracked run and, when needed, attempting the single replacement run, record this backlog item as blocked with the exact concrete return condition in the workflow-owned progress/state files named above instead of widening the study.

**Verification:**

- Exactly one of the following is true:
  - one authoritative `40`-epoch modes-32 run root is identified
  - the item is explicitly blocked because a concrete contract or execution blocker prevented an authoritative `40`-epoch root
- Any accepted authoritative root matches the fixed capped CNS contract and has exit status `0` plus the required completion artifacts.

## Task 3: Emit The Anchored `40`-Epoch Compare And Keep The Claim Boundary Tight

**Files:**
- Reuse: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/reference_runs_40ep.json`
- Reuse: authoritative fresh `40`-epoch modes-32 run root from Task 2
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_40ep_against_existing.json`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_40ep_against_existing.csv`
- Optional: `compare_40ep_sample0.png`, `compare_40ep_sample0_error.png`

Precondition: Execute this task only if Task 1 recorded `selected_capped_lane_authorized` and Task 2 identified one authoritative `40`-epoch modes-32 root. If Task 2 ended blocked because no authoritative root exists, do not emit the anchored `40`-epoch sidecar or advance the backlog item to completion.

- [ ] **Step 1: Generate the anchored `40`-epoch cross-run compare from the frozen reference manifest**

The compare must include only:

- fresh `spectral_resnet_bottleneck_modes32`
- shared `spectral_resnet_bottleneck_base`
- `fno_base`
- `unet_strong`

Do not inject `hybrid_resnet_cns`, `spectral_resnet_bottleneck_noshare`, FFNO, or GNOT rows into this sidecar.

- [ ] **Step 2: Render top-level galleries only when sample alignment is valid**

If cross-run gallery rendering succeeds, keep the emitted galleries beside the `compare_40ep_against_existing` JSON/CSV. If rendering fails but the core compare JSON/CSV is correct, record the gallery failure as optional-reporting-only and continue.

- [ ] **Step 3: Capture the bounded interpretation**

Summarize only the local capped decision-support outcome:

- whether modes-32 improved or regressed versus the shared `12/12` spectral row
- whether it stayed ahead of or behind the reused `fno_base` and `unet_strong` anchors
- whether any improvement is concentrated in `fRMSE_high` or only in aggregate error

Keep the wording explicitly capped-readiness / decision-support only.

**Verification:**

- `compare_40ep_against_existing.json` and `.csv` exist and reference the fixed contract plus the four intended rows.
- Optional gallery failure does not block completion if the core compare files are correct.

## Task 4: Publish Durable Summary, Discoverability, And Completion State

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes32_compare_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/studies/index.md`
- Modify: `docs/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Modify on completion or blockage: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-pdebench-cns-spectral-modes32-compare/progress_report.md`
- Modify on completion or blockage: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-21-pdebench-cns-spectral-modes32-compare/implementation-phase/implementation_state.json`
- Modify on completion or blockage: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-21-pdebench-cns-spectral-modes32-compare/implementation-phase/implementation_state.txt`
- Modify conditionally: `docs/findings.md`

Precondition: Execute this task only after Task 3 completes. If Task 2 ended blocked because no authoritative `40`-epoch root exists, update the concrete workflow-owned blocker surfaces named above, do not write completion-state summaries, and do not move the item to `COMPLETED`.

- [ ] **Step 1: Write the dedicated modes-32 summary**

The summary must include:

- objective and fairness boundary
- exact fixed contract
- recovered `10`-epoch reuse facts
- authoritative `10`-epoch and `40`-epoch run roots
- anchored `10`-epoch and `40`-epoch compare artifact paths
- key metric deltas versus shared `12/12`, `fno_base`, and `unet_strong`
- explicit decision-support-only claim boundary

- [ ] **Step 2: Update the CNS summary and discoverability docs**

Update:

- `pdebench_2d_cfd_cns_summary.md` so the modes-32 lane is discoverable from the CNS task summary
- `docs/studies/index.md` so the item appears in the PDEBench CNS study lineage
- `docs/index.md` so the new summary is reachable from the documentation hub

- [ ] **Step 3: Update initiative state**

Record completion in `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`, including the summary path and the fact that this backlog item completed as a capped CNS decision-support compare. Also update:

- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-pdebench-cns-spectral-modes32-compare/progress_report.md`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-21-pdebench-cns-spectral-modes32-compare/implementation-phase/implementation_state.json`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-21-pdebench-cns-spectral-modes32-compare/implementation-phase/implementation_state.txt`

so the item moves from `RUNNING` to `COMPLETED` consistently across the workflow surfaces.

- [ ] **Step 4: Update `docs/findings.md` only if the result changes reusable guidance**

Do this only if the modes-32 result materially changes what future workers should prefer or avoid. A one-off manual result that stays capped decision-support evidence does not automatically justify a new finding.

**Verification:**

Run the required check commands again after any code/doc/state edits, then confirm the durable outputs exist:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes32_compare_summary.md"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_10ep_against_existing.json"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_40ep_against_existing.json"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing required outputs: {missing}")
print("modes32 summary and compare outputs present")
PY
```

## Completion Criteria

- The landed modes-32 profile/tests still satisfy the fixed fairness contract.
- The recovered `10`-epoch row and anchored `10`-epoch compare are validated and reused.
- The roadmap routing state and selected-item context are re-read and confirmed to authorize this bounded capped CNS lane without widening scope or changing roadmap order.
- Exactly one authoritative fresh `40`-epoch modes-32 run root completes under the unchanged capped CNS contract.
- `compare_40ep_against_existing.{json,csv}` exists and compares only the intended four rows.
- The dedicated summary, CNS summary update, discoverability updates, and initiative state updates are all written.
- The required deterministic checks pass at the end of the item.

## Blocked Outcome

If neither the tracked `014353Z` root nor the single allowed replacement run can be accepted as authoritative because of a concrete contract or execution blocker, stop with this backlog item blocked. Write the blocker and its exact return condition to:

- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-pdebench-cns-spectral-modes32-compare/progress_report.md`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-21-pdebench-cns-spectral-modes32-compare/implementation-phase/implementation_state.json`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-21-pdebench-cns-spectral-modes32-compare/implementation-phase/implementation_state.txt`

Use the real encountered blocker as the return condition, for example:

- the required pytest or `compileall` checks must pass after a minimal repair
- the official CNS HDF5 path or split/provenance contract must be restaged or corrected
- the single replacement run must exit `0` and produce the required completion artifacts

Do not widen the study, change the fixed compare contract, or mark the item complete in this blocked branch.
