# Lines128 Minimum Paper Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the minimum paper-grade `lines128` CDI subset under one frozen contract for Hybrid ResNet, paired CDI `cnn` local baselines, and the fixed `fno_vanilla` comparator, using honest same-root recovery when possible and a fresh rerun only when recovery is not paper-grade-safe.

**Architecture:** Treat the checked-in minimum-subset execution authority as the launch-controlling contract, the harness preflight note as readiness-only background, and the recovered-root audit as the current state surface. Prefer completing the best existing same-root candidate (`minimum_subset_20260429T213028Z`) because all four required rows already finished there and the known collation crash has already been fixed in `scripts/studies/metrics_tables.py`; fall back to a fresh four-row rerun only if that root cannot be completed honestly under the full provenance contract. Any long-running rerun stays under implementation ownership until the tracked tmux/PID path exits `0` and the required artifacts are freshly written.

**Tech Stack:** PATH `python`, tmux with `ptycho311` for long-running commands, TensorFlow grid-lines workflow for `baseline` and `pinn`, PyTorch/Lightning for `pinn_hybrid_resnet` and `pinn_fno_vanilla`, `scripts/studies/lines128_paper_benchmark.py`, `scripts/studies/grid_lines_compare_wrapper.py`, `scripts/studies/grid_lines_torch_runner.py`, `scripts/studies/metrics_tables.py`, pytest, `compileall`, Markdown/JSON/CSV/TeX artifacts.

---

## Selected Backlog Objective

- Implement backlog item `2026-04-29-cdi-lines128-minimum-paper-table`.
- Produce the roadmap Phase `3.3b` minimum CDI claim subset:
  - `baseline`: CDI `cnn` + supervised
  - `pinn`: CDI `cnn` + PINN
  - `pinn_hybrid_resnet`: Hybrid ResNet + PINN
  - `pinn_fno_vanilla`: FNO Vanilla + PINN
- Emit merged JSON/CSV/TeX tables, metric schema, source reconstruction arrays, fixed-sample amplitude/phase panels, error panels, FRC curves, row-local provenance, and a durable summary under `docs/plans/NEURIPS-HYBRID-RESNET-2026/`.
- Mark the subset `paper_grade` only if every required row in the chosen root has complete invocation, config, git, environment, dataset, split, metric, and visual provenance.

## Scope And Explicit Non-Goals

In scope:

- Reuse the frozen contract surfaces from:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_preflight.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json`
- Prefer honest same-root completion of `runs/minimum_subset_20260429T213028Z` because it already contains all four required rows and failed only at final bundle collation.
- If same-root completion is not honest or not possible, run a fresh four-row same-contract rerun in a brand-new root after deterministic gates are green and disk space is available.
- Keep the CDI `cnn` local-baseline family aligned with the PDEBench CNS `unet_strong` local-baseline role in labels and summary text, while explicitly stating that the implementations are task-local and not identical.
- Preserve and update the current recovered-root state under:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/recovered_root_audit.md`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/progress_report.md`

Explicit non-goals:

- Do not wait for `pinn_spectral_resnet_bottleneck_net` or `pinn_ffno`.
- Do not change the selected FNO comparator from `fno_vanilla`.
- Do not change the fixed `seed=3` policy after seeing metrics.
- Do not mix rows from multiple output roots into a paper-grade bundle.
- Do not promote incomplete historical roots to paper-grade evidence.
- Do not silently relax the locked CDI contract to make a row easier to run.
- Do not treat SRU-Net as interchangeable with the CDI `cnn` row family or PDEBench `unet_strong`.
- Do not create `/home/ollie/Documents/neurips/` artifacts in this item.
- Do not broaden into the complete `lines128` benchmark, later roadmap phases, or unrelated backlog items.
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not create worktrees.

## Binding Constraints And Prerequisites

Strategic and roadmap constraints:

- `docs/steering.md` requires equal-footing comparisons, preserves the current Phase 2 plus Phase 3 selection window, and forbids silently relaxing fairness constraints.
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md` authorizes this as Phase `3.3b` work and requires the output to remain labeled as the minimum CDI subset, not the complete `lines128` benchmark.
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md` keeps CDI anchored at `128x128`, preserves the provenance bar for paper-grade evidence, and forbids `/home/ollie/Documents/neurips/` artifact assembly before the later evidence-bundle phase.
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md` requires the minimum CDI subset to include Hybrid ResNet, paired CDI `cnn` supervised and PINN rows, and the selected FNO comparator.
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md` requires one fixed benchmark contract, a pre-run comparator freeze, and shared metrics and visual policy across every row.

Prerequisite and current-state status:

- `phase-0-evidence-inventory` and `phase-1-pde-benchmark-selection` are complete in `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`.
- `blocked_tranches` is currently empty in the progress ledger, so no roadmap-level blocker prevents this CDI work.
- The prerequisite harness item `2026-04-29-cdi-lines128-paper-benchmark-harness` is complete; its durable summary is `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_summary.md`.
- This backlog item is already a recovered in-progress item with audited roots:
  - `minimum_subset_20260429T204000Z`: `stale_do_not_reuse`
  - `minimum_subset_20260429T204642Z`: `failed_recoverable`
  - `minimum_subset_20260429T213028Z`: `failed_recoverable`, but all four rows completed and the known bundle-collation crash was a nested NumPy scalar JSON-serialization failure
  - `minimum_subset_20260429T224103Z`: `failed_recoverable` due `OSError: [Errno 28] No space left on device`
- No active writer remains for this item. A new writer may launch only into a brand-new root.

Authority split for this item:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_preflight.md` and its `benchmark_decisions.json` remain readiness-only authority for the prerequisite harness item.
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md` and `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json` are the launch-controlling surfaces for this item.
- The minimum subset uses runtime row ids `baseline`, `pinn`, `pinn_hybrid_resnet`, and `pinn_fno_vanilla`; the summary must preserve their paper-facing labels exactly.

Locked contract that must not drift unless a checked-in design amendment changes all rows together:

- `N=128`, `gridsize=1`, synthetic grid-lines, `set_phi=True`
- custom Run1084 probe at `datasets/Run1084_recon3_postPC_shrunk_3.npz`
- `probe_scale_mode=pad_extrapolate`, `probe_smoothing_sigma=0.5`, `probe_mask=off`
- `nimgs_train=2`, `nimgs_test=2`, `nphotons=1e9`
- fixed `seed=3`
- `torch_epochs=40`, `torch_learning_rate=2e-4`
- `torch_scheduler=ReduceLROnPlateau`, `torch_plateau_factor=0.5`, `torch_plateau_patience=2`, `torch_plateau_min_lr=1e-4`, `torch_plateau_threshold=0.0`
- `torch_loss_mode=mae`, `torch_mae_pred_l2_match_target=off`, `torch_output_mode=real_imag`
- `fno_modes=12`, `fno_width=32`, `fno_blocks=4`, `fno_cnn_blocks=2`
- fixed sample ids `0`, `1`
- shared visual scales derived from stitched numeric arrays for amplitude, phase, amplitude absolute error, and phase absolute error

Findings and workflow rules that must stay enforced:

- `POLICY-001`
- `CONFIG-001`
- `GRIDLINES-OBJECT-BIG-001`
- `GRIDLINES-PROBE-BIG-001`
- `GRIDLINES-PROBE-PIPELINE-001`
- `FORWARD-SIG-001`
- `OUTPUT-COMPLEX-001`
- `REPORTING-ARTIFACT-BOUNDARY-001`
- `FORWARD-SIG-001`: keep the single-input Torch forward signature `model(X)` for `pinn_hybrid_resnet` and `pinn_fno_vanilla`.
- `OUTPUT-COMPLEX-001`: convert Torch `real_imag` outputs through the existing complex-output helper before metrics, stitching, or bundle collation.
- Long-running commands must run in tmux, activate `ptycho311`, track the exact launched PID, and never launch a duplicate writer into the same `--output-root`.
- Ordinary test, import, path, environment, or harness failures must be diagnosed, fixed narrowly, and rerun before considering `BLOCKED`.
- Reserve `BLOCKED` only for missing resources, unavailable hardware, roadmap conflict, required user decision, or an unrecoverable contract mismatch after one documented narrow fix attempt.

## Implementation Architecture

- Authority and recovery unit:
  - Confirm the checked-in execution authority and derived JSON still match.
  - Decide whether `minimum_subset_20260429T213028Z` can be completed honestly as the paper-grade same-root bundle.
- Bundle and provenance unit:
  - `scripts/studies/lines128_paper_benchmark.py`, `scripts/studies/grid_lines_compare_wrapper.py`, `scripts/studies/grid_lines_torch_runner.py`, and `scripts/studies/metrics_tables.py` own the fixed-contract orchestration, per-row provenance, merged metrics, schema, and visuals.
- Execution and resource unit:
  - If same-root completion is not possible, reclaim or provision enough disk, then run a fresh four-row rerun in a new root with tmux/PID tracking and no duplicate writers.
- Documentation unit:
  - Update the item audit/progress notes and write `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_summary.md`.
  - Update `docs/index.md`, `docs/studies/index.md`, or `docs/findings.md` only if the resulting summary or a newly discovered pitfall becomes durable project knowledge.

## Concrete File And Artifact Targets

Primary code surfaces to verify or modify only if the audit exposes a remaining gap:

- `scripts/studies/lines128_paper_benchmark.py`
- `scripts/studies/grid_lines_compare_wrapper.py`
- `scripts/studies/grid_lines_torch_runner.py`
- `scripts/studies/metrics_tables.py`

Primary test surfaces:

- `tests/studies/test_lines128_paper_benchmark.py`
- `tests/studies/test_metrics_tables.py`
- `tests/test_grid_lines_compare_wrapper.py`
- `tests/torch/test_grid_lines_torch_runner.py`
- `tests/torch/test_grid_lines_hybrid_resnet_integration.py`

Primary artifact surfaces:

- Audit and status:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/recovered_root_audit.md`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/progress_report.md`
- Launch authority:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json`
- Preferred same-root recovery candidate:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T213028Z/`
- Fresh-rerun fallback root:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_<timestamp>/`
- Verification logs:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/`
- Durable summary:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_summary.md`

## Task 1: Reconfirm Authority And Choose The Honest Recovery Path

**Files / artifacts:**

- Read and compare:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_preflight.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json`
- Audit:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/recovered_root_audit.md`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/progress_report.md`

- [ ] Confirm the execution-authority note and derived JSON still match exactly on state, comparator, fixed contract, seed policy, fixed sample ids, shared visual-scale policy, and the four-row minimum roster.
- [ ] Confirm the harness preflight note and prerequisite decision artifact still match the execution authority on only the fields they are supposed to freeze.
- [ ] Recheck that `minimum_subset_20260429T213028Z` contains all four required row-local outputs and is the preferred same-root completion candidate.
- [ ] Recheck that `minimum_subset_20260429T224103Z` is not reusable for launch because it failed during dataset writing under disk exhaustion.
- [ ] Decide one of two execution paths and record it explicitly in the audit note:
  - `same_root_recovery` using `minimum_subset_20260429T213028Z`
  - `fresh_rerun_required` with the exact provenance or completeness gap that disqualifies same-root recovery
- [ ] If any authority drift exists, fix the checked-in launch authority first, then regenerate the derived execution JSON so the note remains the human-reviewable source of truth.
- [ ] Do not launch any new writer while this decision is unresolved.

**Verification after Task 1:**

- [ ] The audit note names the chosen execution path and chosen root.
- [ ] The required runtime roster remains exactly `baseline`, `pinn`, `pinn_hybrid_resnet`, and `pinn_fno_vanilla`.
- [ ] The audit note explicitly preserves the readiness-only versus launch-authority boundary.

## Task 2: Close Only The Remaining Code Or Provenance Gap

**Files:**

- Modify only if Task 1 exposes a concrete remaining gap:
  - `scripts/studies/lines128_paper_benchmark.py`
  - `scripts/studies/grid_lines_compare_wrapper.py`
  - `scripts/studies/grid_lines_torch_runner.py`
  - `scripts/studies/metrics_tables.py`
- Update tests only where behavior changed:
  - `tests/studies/test_lines128_paper_benchmark.py`
  - `tests/studies/test_metrics_tables.py`
  - `tests/test_grid_lines_compare_wrapper.py`
  - `tests/torch/test_grid_lines_torch_runner.py`

- [ ] If same-root recovery is chosen, verify the existing post-serialization-fix code is sufficient to finish collation from row-local artifacts without retraining rows.
- [ ] If the audit exposes another concrete bundle, schema, manifest, or provenance gap, patch only that gap.
- [ ] Preserve the fixed row labels, architecture ids, training procedures, metric schema, fixed sample ids, shared visual scales, and `claim_boundary=minimum_draftable_cdi_subset`.
- [ ] Keep `FORWARD-SIG-001` and `OUTPUT-COMPLEX-001` intact for the Torch rows.
- [ ] Do not reopen spectral/FFNO routing or later complete-table scope.

**Verification after Task 2:**

- [ ] No known schema, provenance, or collation gap remains for the four required rows.
- [ ] Focused selectors covering touched surfaces are green before the mandatory backlog checks run.
- [ ] If no code change was needed, record that the existing fix set was reused unchanged.

## Task 3: Run Deterministic Gates Before Any Write-Side Recovery Or Fresh Benchmark Step

**Archive logs under:**

- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/`

- [ ] If Task 2 changed code, run focused selectors first:

```bash
pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py
```

- [ ] Run the backlog item’s required deterministic checks exactly as written. These are mandatory for this item and are not replaced:

```bash
pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py
python -m compileall -q ptycho_torch scripts/studies
```

- [ ] Archive every command output under the verification root.
- [ ] Do not start same-root write-side collation, a fresh benchmark launch, or any disk-recovery mutation that changes item outputs until these required gates are green on the current workspace state.
- [ ] If a deterministic gate fails, diagnose, patch narrowly, and rerun before considering `BLOCKED`.

**Verification after Task 3:**

- [ ] Both backlog-required commands are green and archived.
- [ ] Any stronger focused selectors used during iteration are green and archived.
- [ ] No expensive or state-mutating execution step begins on a red gate.

## Task 4: Preferred Path - Complete The Existing Same-Root Bundle

**Use this task only if Task 1 selected `same_root_recovery`.**

**Root:**

- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T213028Z/`

- [ ] Re-run only the missing final bundle/collation path needed to finish the paper bundle from the already completed row-local artifacts in this root.
- [ ] Do not retrain or reroute any of the four rows if the existing row-local artifacts remain contract-complete.
- [ ] Emit or refresh all missing required bundle outputs in the same root:
  - `metrics.json`
  - `metric_schema.json`
  - `model_manifest.json`
  - `metrics_table.csv`
  - `metrics_table.tex`
  - `metrics_table_best.tex`
  - fixed-sample visuals
  - FRC curves
  - source arrays sufficient to regenerate figures
- [ ] Validate that wrapper-level provenance, row-local provenance, dataset/split identity, and visual/sample policy are complete enough for `paper_grade`.
- [ ] If same-root recovery still fails or a previously hidden provenance gap makes paper-grade promotion dishonest, stop using this path, update the audit note with the exact reason, and continue to Task 5 instead.

**Verification after Task 4:**

- [ ] The chosen root contains a complete four-row same-root bundle with no mixed-root artifacts.
- [ ] The merged outputs state `claim_boundary=minimum_draftable_cdi_subset`.
- [ ] The bundle is either honestly `paper_complete` with row-level `paper_grade` status or explicitly downgraded with the precise remaining reason.

## Task 5: Fallback Path - Resolve The Resource Gate And Launch A Fresh Four-Row Rerun

**Use this task only if Task 1 selected `fresh_rerun_required` or Task 4 proved same-root recovery dishonest/impossible.**

**Files / artifacts:**

- Reuse:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json`
- Create or update:
  - a new root under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_<timestamp>/`
  - root-local `invocation.sh`, `invocation.json`, manifests, logs, metrics, source arrays, visuals, and bundle outputs

- [ ] First handle the current missing-resource blocker explicitly. If `/` still lacks enough free space, make one narrow recovery attempt before calling the item blocked:
  - preserve the current audit note and any unique failure/provenance evidence
  - reclaim or relocate only superseded, item-local bulky artifacts that are not the chosen same-root evidence path
  - do not delete the preferred recovery candidate root unless the audit has already ruled it unusable
- [ ] If adequate free space still cannot be obtained after that narrow attempt, record `BLOCKED` as `missing_resource` with exact filesystem evidence and stop.
- [ ] If space is available, launch the benchmark in tmux under `ptycho311`, track the exact shell PID, and wait for that PID rather than polling unrelated processes.
- [ ] Never launch into a previously used output root.
- [ ] Require both conditions before calling the rerun complete:
  - tracked PID exits `0`
  - required output artifacts exist and are freshly written in the new root
- [ ] If the run fails for an ordinary harness, import, or path reason, diagnose, patch narrowly, and relaunch in a brand-new root rather than marking the item blocked.

**Verification after Task 5:**

- [ ] The new root contains all four rows and the complete merged bundle.
- [ ] The launcher evidence includes tmux session identity, tracked PID, exit status, and root-local invocation artifacts.
- [ ] Any blocker state is reserved for the narrow approved cases only, with missing-resource evidence if disk remains the limiter.

## Task 6: Write The Durable Summary And Update Discoverability Only If Warranted

**Files / artifacts:**

- Create or update:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_summary.md`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/progress_report.md`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/recovered_root_audit.md`
- Update only if durable knowledge changed:
  - `docs/index.md`
  - `docs/studies/index.md`
  - `docs/findings.md`

- [ ] Summarize the final execution path used: same-root recovery or fresh rerun.
- [ ] Record the chosen root, per-row status, benchmark status, provenance completeness, fixed comparator/seed policy, and the CDI `cnn` versus CNS `unet_strong` labeling note.
- [ ] State clearly that `pinn_spectral_resnet_bottleneck_net` and `pinn_ffno` remain later complete-table work and were intentionally out of scope here.
- [ ] If the item ended blocked on disk or another allowed blocker, record the exact blocker class, the narrow recovery attempt already made, and the concrete resume condition.
- [ ] Update `docs/index.md` only if the new summary becomes a durable authority readers should discover from the hub.
- [ ] Update `docs/studies/index.md` only if the final minimum-subset result becomes a reusable study-discovery surface rather than item-local evidence.
- [ ] Update `docs/findings.md` only if this pass exposed a reusable pitfall that future grid-lines or paper-bundle work is likely to hit again.

**Verification after Task 6:**

- [ ] The durable summary is self-contained enough for downstream paper-assembly work to understand the minimum CDI subset without reopening backlog notes.
- [ ] Discoverability docs were updated only when justified by durable knowledge changes.
- [ ] The final state is unambiguous: `paper_complete`, downgraded-but-usable with explicit reasons, or `BLOCKED` with an allowed blocker class.
