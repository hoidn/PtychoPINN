# CDI FFNO Generator On Lines128 Best Contract Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete one auditable equal-footing CDI compare between `pinn_hybrid_resnet` and `pinn_ffno` on the fixed Lines128 legacy-best contract, then publish durable metrics, visuals, provenance, and status without broadening into the full four-row paper benchmark.

**Architecture:** Keep `scripts/studies/grid_lines_compare_wrapper.py` as the shared dataset/provenance/collation owner and `scripts/studies/grid_lines_torch_runner.py` as the per-row execution owner. Start from the checked-in preflight and summary as the current durable state, but re-audit the stable output root at execution time before deciding whether to wait, repair, relaunch, or simply finalize. If FFNO fails on the CDI path, patch only the minimal generator/runner/wrapper/test surfaces needed to keep FFNO on the same real/imag Torch CDI contract as `hybrid_resnet`.

**Tech Stack:** Python 3.11 via PATH `python`, long-running repo commands in tmux with `conda activate ptycho311`, PyTorch Lightning, `ptycho_torch` generator registry, `scripts/studies/grid_lines_compare_wrapper.py`, `scripts/studies/grid_lines_torch_runner.py`, pytest, `compileall`, Markdown/JSON/CSV/NPZ artifacts.

---

## Selected Objective

- Complete backlog item `2026-04-27-cdi-ffno-generator-lines-best-config`.
- Keep FFNO on the existing Torch CDI generator path only if it satisfies the same output, stitching, probe, split, optimizer, loss, and metric contract already used by `hybrid_resnet`.
- Produce one durable `pinn_hybrid_resnet` versus `pinn_ffno` row pair under the study-indexed `grid-lines-n128-hybrid-resnet-legacy-best-e40-seed3` contract that later `lines128` benchmark work can reuse.
- Publish quantitative metrics, amplitude/phase comparison figures, and enough invocation/runtime provenance for later paper-facing packaging.

## Scope And Explicit Non-Goals

In scope:

- re-audit the current stable output root and classify it as `active_writer`, `completed_candidate`, or `stale_or_failed_partial`
- preserve the recovered fixed Lines128 contract already documented in the checked-in preflight
- repair only proven FFNO CDI-path or wrapper-collation failures
- complete or validate one stable-root pairwise compare under the fixed contract
- finalize durable result status and discoverability for this prerequisite evidence slice

Explicit non-goals:

- no full four-row `lines128` paper benchmark table
- no FNO comparator choice for the later four-row benchmark
- no FFNO family sweep, no broader CDI architecture search, and no borrowing PDEBench CNS FFNO conclusions as CDI evidence
- no silent pivot to the separate current-anchor `50`-epoch / `torch_mae_pred_l2_match_target=on` contract
- no `/home/ollie/Documents/neurips/` Phase 5 evidence-bundle work
- no edits to `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` unless a new narrower approved plan authorizes that scope

## Binding Constraints

- Steering boundary:
  - this item is valid work inside the current Phase 2 plus Phase 3 selection window
  - it strengthens the CDI comparison story but does not satisfy remaining Phase 2 PDEBench requirements
- Roadmap boundary:
  - stay within Phase 3 CDI-preparation scope
  - preserve equal-footing comparison rules and keep this output labeled as prerequisite evidence only, not the complete paper benchmark
- Backlog-item boundary:
  - use `docs/studies/index.md` to preserve the best-lines contract
  - if FFNO cannot satisfy the generator output contract, record a blocker instead of changing the CDI workflow silently
  - keep this work separate from CNS-only FFNO and Hybrid-spectral ablations
- Lines128 benchmark boundary from `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md`:
  - the paper-benchmark target is the study-indexed `N=128`, `gridsize=1`, `seed=3` legacy-best contract unless a later checked-in pre-run decision changes every row together
  - the durable contract-reconstruction preflight remains mandatory before any later multi-row paper-grade launch
  - this item must not silently broaden into the full benchmark harness or change the benchmark contract to fit FFNO
- Fixed compare contract to preserve for both rows unless a later checked-in override changes every row together:
  - `N=128`, `gridsize=1`, synthetic grid-lines, `set_phi=True`
  - custom Run1084 probe
  - `probe_scale_mode=pad_extrapolate`, `probe_smoothing_sigma=0.5`
  - `nimgs_train=2`, `nimgs_test=2`, `nphotons=1e9`
  - `seed=3`
  - `torch_epochs=40`, `torch_learning_rate=2e-4`
  - `torch_scheduler=ReduceLROnPlateau`, `torch_plateau_factor=0.5`, `torch_plateau_patience=2`, `torch_plateau_min_lr=1e-4`, `torch_plateau_threshold=0.0`
  - `torch_loss_mode=mae`, `torch_mae_pred_l2_match_target=off`
  - `torch_output_mode=real_imag`, `probe_mask=off`
  - `fno_modes=12`, `fno_width=32`, `fno_blocks=4`, `fno_cnn_blocks=2`
- Baseline authority note:
  - `docs/model_baselines.md` remains the general baseline authority
  - this item intentionally uses the recovered study-owned legacy-best override above, including `torch_mae_pred_l2_match_target=off`
- Findings and workflow constraints to preserve:
  - obey `GRIDLINES-OBJECT-BIG-001` and `GRIDLINES-PROBE-BIG-001` when touching the Torch runner config bridge for grid-lines parity
  - treat probe preprocessing as dataset-contract provenance per `GRIDLINES-PROBE-PIPELINE-001`; do not change probe normalization/preprocessing silently
  - treat missing optional galleries as warnings, not primary run-failure evidence, per `REPORTING-ARTIFACT-BOUNDARY-001`; the row pair still needs the required comparison visuals defined by this item
  - if runner or wrapper config plumbing changes, keep `update_legacy_dict(params.cfg, config)` in place before data loading or legacy-module use
  - keep new or edited Python subprocess examples and orchestration commands on PATH `python` per `PYTHON-ENV-001`
- Invocation and long-run guardrails:
  - keep `invocation.json` and `invocation.sh` artifact logging intact for any modified `scripts/studies/*` surface
  - do not launch a duplicate process into the same stable root
  - for long-running work, track the exact launched PID and wait on that PID
  - do not call the run complete until the tracked PID exits `0` and the required wrapper artifacts are freshly written
- Failure policy:
  - do not mark the item `BLOCKED` for normal import, path, environment, config, or test-harness failures
  - diagnose, patch narrowly, and rerun first
  - reserve `BLOCKED` for unresolved FFNO CDI-contract incompatibility after one narrow documented fix cycle, missing external resources, roadmap conflict, unavailable hardware, or user decision required

## Prerequisite Status And Current Durable State

- Progress-ledger status that matters here:
  - `phase-0-evidence-inventory` is complete
  - `phase-1-pde-benchmark-selection` is complete
  - the roadmap explicitly allows current Phase 3 CDI-preparation selection in parallel with remaining Phase 2 work
  - no incomplete Phase 2 lane blocks this item
- Checked-in durable context that implementation should treat as the starting state:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_summary.md`
- Those documents record, as of `2026-04-29`, a recovered fixed contract, the stable compare root below, and an active-writer/no-duplicate-launch decision. Implementation must re-audit the live root and process state before trusting those exact PIDs or deciding to wait, relaunch, or finalize.
- This row pair is prerequisite CDI evidence for later `lines128` benchmark packaging and does not authorize the later `spectral_resnet_bottleneck_net` or selected-FNO rows.

## Implementation Architecture

- Runtime-state audit unit:
  - reclassify the stable root at execution time and keep one authoritative completion decision
- Narrow repair unit:
  - if needed, patch only the FFNO generator, runner, wrapper, or result-collation surfaces implicated by the audit
- Completion-and-publication unit:
  - finish the stable root, verify deterministic gates, and update summary/index surfaces so later paper-benchmark work can consume the result safely

## Concrete File And Artifact Targets

- Validate and update as needed:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_summary.md`
  - `docs/studies/index.md`
  - `docs/findings.md` only if execution exposes a durable FFNO/CDI pitfall future workers need
  - `docs/index.md` only if discoverability changes materially
- Stable runtime root:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`
- Verification log root:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/verification/`
- Conditional code/test surfaces only if a real failure is confirmed:
  - `ptycho_torch/generators/ffno.py`
  - `ptycho_torch/generators/registry.py`
  - `ptycho_torch/config_params.py`
  - `scripts/studies/grid_lines_torch_runner.py`
  - `scripts/studies/grid_lines_compare_wrapper.py`
  - `tests/torch/test_generator_registry.py`
  - `tests/torch/test_grid_lines_torch_runner.py`
  - `tests/test_grid_lines_compare_wrapper.py`
- Required stable-root completion artifacts:
  - wrapper `invocation.json` and `invocation.sh`
  - wrapper `metrics.json`, `metrics_table.csv`, and any emitted table/TeX outputs
  - wrapper comparison visuals for amplitude and phase
  - `runs/pinn_hybrid_resnet/` and `runs/pinn_ffno/` with `metrics.json`, `history.json`, invocation artifacts, and reconstruction outputs
  - one shared dataset subroot under the stable root

### Task 1: Re-Audit The Stable Root And Preserve The Single-Root Decision

**Files / artifacts:**
- Read/update: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md`
- Read/update: `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_summary.md`
- Audit: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`

- [ ] Re-audit the stable compare root before changing code or launching anything. Classify the current state as one of: `active_writer`, `completed_candidate`, or `stale_or_failed_partial`.
- [ ] Verify the observed root against the actual completion contract: both row directories, wrapper-level merged metrics/tables/visuals, per-row metrics/history/invocation/reconstruction outputs, and shared dataset provenance.
- [ ] If a writer is still active, identify the exact tracked PID or tmux-owned shell PID responsible for the root and preserve the no-duplicate-run rule.
- [ ] If the root is stale or failed, preserve the failed-attempt artifacts that matter for diagnosis, especially wrapper invocation, logs, and any partial row outputs, without letting stale files masquerade as completion.
- [ ] Update the preflight note and summary only where runtime state has genuinely changed; keep them aligned on the same root, decision status, and claim boundary.

**Verification before moving on:**
- [ ] One runtime-state classification is recorded and justified.
- [ ] The preflight note and summary agree on the stable root and current execution decision.
- [ ] No duplicate launch decision is taken while an active writer still owns the root.

### Task 2: Repair Only Proven FFNO CDI-Path Failures

**Files:**
- Conditional modify: `ptycho_torch/generators/ffno.py`, `ptycho_torch/generators/registry.py`, `ptycho_torch/config_params.py`, `scripts/studies/grid_lines_torch_runner.py`, `scripts/studies/grid_lines_compare_wrapper.py`
- Conditional test updates: `tests/torch/test_generator_registry.py`, `tests/torch/test_grid_lines_torch_runner.py`, `tests/test_grid_lines_compare_wrapper.py`

- [ ] If Task 1 finds completion, skip straight to Task 4 and leave code unchanged.
- [ ] If Task 1 finds a stale or failed run, inspect the exact failure mode before editing: FFNO output-contract mismatch, architecture acceptance, wrapper result collation, visualization assumptions, missing completion markers, or provenance gaps.
- [ ] Patch only the minimal surface required to keep FFNO on the same CDI Torch path as `hybrid_resnet`; do not relax the dataset, probe, split, scheduler, loss, output, or metric contract.
- [ ] Preserve invocation logging and any required `update_legacy_dict(params.cfg, config)` bridge behavior if runner or wrapper plumbing changes.
- [ ] Add or update focused regression coverage for the exact failure that was fixed.

**Verification before moving on:**
- [ ] Any code change has a matching focused regression test.
- [ ] Focused fast checks for the touched FFNO path are green before broader gates, for example:
  - `pytest -q tests/torch/test_generator_registry.py`
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k ffno`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k ffno`

### Task 3: Run The Mandatory Deterministic Gates Before Any Relaunch

**Files / artifacts:**
- Archive logs under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/verification/`

- [ ] Use the backlog item’s required deterministic checks exactly as written below for any relaunch path and after any code change.
- [ ] If Task 1 proves the launched run already completed cleanly without further edits, verify that the pre-launch green evidence for these same commands is durably archived; if it is missing or ambiguous, rerun the commands now before closing the item.
- [ ] Archive the outputs with timestamps so the later summary can point to durable gate evidence.
- [ ] If either command fails, diagnose, patch narrowly, and rerun; do not mark the item `BLOCKED` for ordinary verification failures.
- [ ] No expensive relaunch may begin until these gates are green.

**Required deterministic checks:**

```bash
pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py
python -m compileall -q ptycho_torch scripts/studies
```

**Verification before moving on:**
- [ ] Both required commands pass, or already-used matching green logs are verified as durable evidence for a no-edit completed run.
- [ ] Archived logs identify the command, timestamp, and stable output root they gated.

### Task 4: Wait For Completion Or Relaunch Exactly One Fixed-Contract Compare

**Files / artifacts:**
- Stable root: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`

- [ ] If Task 1 found an active writer, wait on the tracked PID, then validate the stable root after exit. Do not start another run.
- [ ] If Task 1 found a completed candidate and Task 3 did not require a relaunch, validate the full completion artifact set and proceed to Task 5.
- [ ] If Task 1 found a stale or failed partial root, and Task 3 is green, relaunch exactly one clean compare under the same stable root in tmux with `ptycho311` activated. Track the launched PID directly.
- [ ] Use this fixed compare command unless the preflight note is explicitly updated to reflect one narrow audited fix:

```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 128 \
  --gridsize 1 \
  --output-dir .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet \
  --architectures hybrid_resnet,ffno \
  --nimgs-train 2 \
  --nimgs-test 2 \
  --nphotons 1e9 \
  --batch-size 16 \
  --seed 3 \
  --probe-source custom \
  --probe-scale-mode pad_extrapolate \
  --probe-smoothing-sigma 0.5 \
  --set-phi \
  --torch-epochs 40 \
  --torch-batch-size 16 \
  --torch-learning-rate 2e-4 \
  --torch-scheduler ReduceLROnPlateau \
  --torch-plateau-factor 0.5 \
  --torch-plateau-patience 2 \
  --torch-plateau-min-lr 1e-4 \
  --torch-plateau-threshold 0.0 \
  --torch-loss-mode mae \
  --torch-no-mae-pred-l2-match-target \
  --torch-output-mode real_imag \
  --fno-modes 12 \
  --fno-width 32 \
  --fno-blocks 4 \
  --fno-cnn-blocks 2
```

- [ ] Keep dataset identity, probe path, seed, epochs, scheduler, output mode, and row list identical across `pinn_hybrid_resnet` and `pinn_ffno`.
- [ ] If the first relaunch exposes one concrete FFNO CDI-contract bug, do one narrow documented fix cycle and rerun once. Mark `BLOCKED` only if FFNO still cannot satisfy the existing CDI contract after that narrow fix attempt.

**Verification before moving on:**
- [ ] The tracked PID exits `0`, or a completed-candidate path is validated without relaunch.
- [ ] The stable root contains complete `runs/pinn_hybrid_resnet/` and `runs/pinn_ffno/` trees.
- [ ] Wrapper-level merged metrics, tables, and amplitude/phase visuals exist and are freshly written.
- [ ] Row-level metrics, history, reconstructions, and invocation artifacts exist for both rows.

### Task 5: Finalize Durable Result Notes And Discoverability

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_summary.md`
- Modify: `docs/studies/index.md`
- Modify if needed: `docs/findings.md`
- Modify if needed: `docs/index.md`

- [ ] Finalize the summary with final status, exact stable root, key metrics for both rows, figure paths, and the execution decision that produced the final result.
- [ ] State explicitly that this row pair is prerequisite CDI evidence for later `lines128` paper benchmarking and does not itself produce the complete four-row benchmark.
- [ ] Update `docs/studies/index.md` so the relevant Lines128 FFNO/Hybrid entry points to the final root and final state instead of in-progress wording.
- [ ] Add a concise evidence-backed finding to `docs/findings.md` only if execution exposed a durable FFNO/CDI integration pitfall that future workers should follow.
- [ ] Touch `docs/index.md` only if the discoverability surface changed materially.

**Verification before closing the item:**
- [ ] The summary, studies-index entry, and any optional index updates all point to the same stable root.
- [ ] Any blocked outcome is described as a contract/resource blocker, not as silent protocol drift.
- [ ] The final documentation preserves the prerequisite-only claim boundary.

## Completion Criteria

- The checked-in preflight note remains accurate for the final execution decision and stable root.
- `ffno` either completes through the existing CDI Torch path under the same output/stitching contract as `hybrid_resnet`, or the item ends with an explicit documented CDI-contract blocker after one narrow fix cycle.
- One fresh auditable `pinn_hybrid_resnet` versus `pinn_ffno` compare exists under the stable root with shared dataset, probe, seed, loss, scheduler, epoch budget, and output-mode settings.
- The backlog item’s deterministic checks are accounted for with durable green evidence, and any relaunch waited for those checks to pass first.
- Metrics, visuals, and provenance are captured for both rows and summarized durably.
- Documentation clearly states that this work unlocks later `lines128` benchmark packaging but is not the complete paper benchmark.
