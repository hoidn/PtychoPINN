# CDI FFNO Generator On Lines128 Best Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover or complete a fresh auditable `ffno` versus `hybrid_resnet` CDI row pair on the fixed Lines128 `N=128`, `seed=3`, `40`-epoch best documented contract without drifting the existing ptycho/CDI workflow.

**Architecture:** Treat the existing compare root `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet` as recoverable in-progress state, not as proof of completion. Keep `scripts/studies/grid_lines_compare_wrapper.py` as the shared dataset/provenance authority and `scripts/studies/grid_lines_torch_runner.py` as the per-row training/inference authority; patch `ffno` generator, runner, or wrapper code only if recovery audit or targeted checks expose a real CDI-contract or collation bug. This item produces one prerequisite `pinn_hybrid_resnet` versus `pinn_ffno` row pair for later `lines128` benchmark packaging; it does not replace the later four-row paper benchmark harness.

**Tech Stack:** Python 3.11 via PATH `python`, `ptycho311` for long-running repo workflows, PyTorch Lightning, `ptycho_torch` generator registry, `scripts/studies/grid_lines_torch_runner.py`, `scripts/studies/grid_lines_compare_wrapper.py`, pytest, `compileall`, Markdown/JSON/NPZ artifacts, tmux for long runs.

---

## Selected Objective

- Recover or finish the selected backlog item `2026-04-27-cdi-ffno-generator-lines-best-config`.
- Use the existing CDI `ffno` generator path only if it satisfies the same real/imag output, stitching, probe, and metric contract already used by `hybrid_resnet`.
- Produce one fresh, equal-footing `pinn_hybrid_resnet` versus `pinn_ffno` comparison under the study-indexed best Lines128 contract needed by the downstream `lines128` paper benchmark design.
- Publish durable metrics, standard amplitude/phase comparison figures, and provenance that later benchmark work can trust.

## Scope Boundaries

In scope:

- recover the current in-progress compare state and decide resume versus relaunch
- write a durable contract-reconstruction preflight note before any expensive rerun
- verify the existing `ffno` CDI path and patch only the minimal code needed for contract compatibility or wrapper collation
- complete one fixed-contract `pinn_hybrid_resnet` versus `pinn_ffno` compare under one stable output root
- update the NeurIPS summary and study index so the row pair is discoverable

Explicit non-goals:

- no full four-row `lines128` paper benchmark table
- no FNO-comparator selection for the later four-row benchmark
- no FFNO parameter sweep or broader CDI architecture search
- no reuse of PDEBench CNS FFNO evidence as CDI evidence
- no silent pivot to the separate current-anchor `50`-epoch / `torch_mae_pred_l2_match_target=on` contract
- no `/home/ollie/Documents/neurips/` Phase 5 evidence-bundle work
- no edits to `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` unless a narrower approved change becomes unavoidable

## Binding Constraints

- Steering authority:
  - this is allowed Phase 3 CDI-preparation work inside the current Phase 2 plus Phase 3 selection window
  - it strengthens core comparison evidence but does not satisfy remaining Phase 2 PDEBench gates
- Roadmap authority:
  - stay within Phase 3 CDI-preparation scope and preserve the current claim boundary that this row pair is prerequisite evidence only
  - preserve equal-footing metric, split, and protocol boundaries unless a reviewed roadmap/design update says otherwise
- Lines128 benchmark authority:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md` requires a durable contract-reconstruction validation artifact before any paper-grade multi-row benchmark launch
  - this item must not silently broaden into that full benchmark
- Study-contract authority:
  - the downstream target remains the recovered study-indexed `grid-lines-n128-hybrid-resnet-legacy-best-e40-seed3` contract from `docs/studies/index.md`
  - preserve the recovered contract exactly unless a checked-in pre-run override changes every row together
- Fixed contract to preserve:
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
- Baseline override note:
  - `docs/model_baselines.md` is the general starting point, but this item intentionally follows the study-indexed legacy-best contract above, including the explicit `torch_mae_pred_l2_match_target=off` override
- Long-run execution guardrail:
  - do not launch a duplicate run while another process is writing to the same `--output-dir`
  - for any long-running compare, track the exact launched PID, wait on that PID, and call the run complete only when the PID exits `0` and the required wrapper artifacts are freshly written
  - use tmux and activate `ptycho311` for the long run
- Failure policy:
  - do not mark the item `BLOCKED` for normal import, path, environment, config, or test-harness failures
  - diagnose, patch narrowly, and rerun first
  - reserve `BLOCKED` for unresolved FFNO CDI-contract incompatibility after one narrow documented fix cycle, missing required external resources, roadmap conflict, or user decision required

## Prerequisite Status

- Progress ledger status that matters here:
  - `phase-0-evidence-inventory` is complete
  - `phase-1-pde-benchmark-selection` is complete
  - the roadmap explicitly allows current Phase 3 CDI-preparation selection in parallel with remaining Phase 2 work
  - no active backlog item must land before this one
- Current recovered state:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_summary.md` already names the stable compare root and marks the state as `run_launched_pending_completion`
  - the stable compare root currently exists but does not yet prove wrapper-complete success; implementation must audit it before any relaunch
- Downstream dependency note:
  - this row pair is prerequisite evidence for later `lines128` paper benchmarking, but it is not itself the full benchmark authority

## Implementation Architecture

- Recovery and preflight unit:
  - audit the existing compare root, process state, and missing-versus-complete artifact set
  - write the durable contract-reconstruction preflight note and the resume/relaunch decision
- Code-path unit:
  - keep `ffno`, runner, wrapper, and test changes tightly scoped to actual CDI-contract or collation issues exposed by the audit or targeted checks
- Execution and evidence unit:
  - complete one clean pairwise compare under one stable root
  - require wrapper-level metrics/tables/visuals plus row-level provenance before calling the item complete

## Concrete File And Artifact Targets

- Create:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/verification/` for archived check logs
- Update:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_summary.md`
  - `docs/studies/index.md`
  - `docs/index.md` only if the new preflight or summary becomes a durable discoverable documentation surface
- Conditional code changes only if the audit/checks require them:
  - `ptycho_torch/generators/ffno.py`
  - `ptycho_torch/generators/registry.py`
  - `ptycho_torch/generators/README.md`
  - `ptycho_torch/config_params.py`
  - `scripts/studies/grid_lines_torch_runner.py`
  - `scripts/studies/grid_lines_compare_wrapper.py`
  - `tests/torch/test_generator_registry.py`
  - `tests/test_generator_registry.py`
  - `tests/torch/test_grid_lines_torch_runner.py`
  - `tests/test_grid_lines_compare_wrapper.py`
- Stable runtime root:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`
- Required completion artifacts under the stable root:
  - wrapper `invocation.json` / `invocation.sh`
  - wrapper `metrics.json` plus table artifacts (`metrics_table.csv` and any emitted TeX table files)
  - wrapper visual artifacts for the row pair
  - per-row `runs/pinn_hybrid_resnet/` and `runs/pinn_ffno/` with `metrics.json`, `history.json`, invocation artifacts, and reconstruction outputs
  - dataset pair under one shared dataset subroot

### Task 1: Recover Current State And Freeze The Contract

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md`
- Read/update: `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_summary.md`
- Audit runtime root: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`

- [ ] Audit the existing stable compare root before changing code or relaunching anything. Record whether there is an active writer, a stale partial attempt, or a wrapper-complete result.
- [ ] Verify the current root against the actual completion contract for this item: two row directories, wrapper-level merged metrics/tables/visuals, and provenance for both rows. Treat missing wrapper outputs as incomplete even if checkpoints or partial training logs exist.
- [ ] Write `lines128_paper_benchmark_preflight.md` with:
  - the recovered contract fields and where each field came from
  - the stable output root
  - the exact two-row model list (`pinn_hybrid_resnet`, `pinn_ffno`)
  - the fixed-seed policy (`seed=3`)
  - the exact compare command to use on resume/relaunch
  - a go/no-go decision for reusing, resuming, or relaunching the current root
- [ ] If the current root is partial and no active writer remains, document how to preserve that partial attempt without letting stale files masquerade as completion.

**Verification before moving on:**
- [ ] `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md` exists and explicitly lists the fixed contract, row list, output root, and go/no-go decision.
- [ ] The summary and preflight note agree on the same stable output root and recovered-state description.

### Task 2: Verify The Existing FFNO CDI Path And Patch Only What Is Broken

**Files:**
- Conditional modify: `ptycho_torch/generators/ffno.py`, `ptycho_torch/generators/registry.py`, `ptycho_torch/generators/README.md`, `ptycho_torch/config_params.py`, `scripts/studies/grid_lines_torch_runner.py`, `scripts/studies/grid_lines_compare_wrapper.py`
- Conditional test updates: `tests/torch/test_generator_registry.py`, `tests/test_generator_registry.py`, `tests/torch/test_grid_lines_torch_runner.py`, `tests/test_grid_lines_compare_wrapper.py`

- [ ] Confirm that `ffno` already resolves through the current generator registry and runner/wrapper surfaces as a CDI generator using the same real/imag output contract and stitching path as the other Torch rows.
- [ ] If the recovery audit or a focused smoke path exposes an FFNO-specific bug, patch only the minimal surface needed:
  - generator output contract mismatch
  - runner architecture acceptance or artifact naming
  - wrapper row mapping / result collation / visualization
  - missing provenance or completion markers needed by the recovery logic
- [ ] Do not change the CDI data contract, probe contract, scheduler, loss mode, or stable physics code to make FFNO easier to run.
- [ ] Add or update focused regression coverage for the exact failure mode that was fixed.

**Verification before moving on:**
- [ ] If code changed, run focused fast checks for the touched FFNO path first, for example:
  - `pytest -q tests/torch/test_generator_registry.py`
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k ffno`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k ffno`
- [ ] Do not move to the expensive compare until the targeted failure mode is covered and green.

### Task 3: Run The Mandatory Deterministic Gates

**Files / artifacts:**
- Archive logs under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/verification/`

- [ ] Run the backlog item’s required deterministic checks exactly as written below after any code changes and before any expensive compare resume/relaunch.
- [ ] Archive the pytest and compileall outputs under the verification directory so the item has durable evidence of passing checks.
- [ ] If either command fails, diagnose, patch narrowly, and rerun; do not mark the item `BLOCKED` for ordinary verification failures.
- [ ] The expensive pairwise compare must wait for these checks to be green.

**Required deterministic checks:**

```bash
pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py
python -m compileall -q ptycho_torch scripts/studies
```

**Verification before moving on:**
- [ ] Both required commands pass.
- [ ] Archived verification logs identify the command, timestamp, and output root they gated.

### Task 4: Resume Or Relaunch The Equal-Footing Row Pair

**Files / artifacts:**
- Stable runtime root: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`
- Per-row roots under `runs/pinn_hybrid_resnet/` and `runs/pinn_ffno/`

- [ ] If Task 1 found an active writer, do not launch a duplicate run; wait on the tracked PID and then validate the required artifacts.
- [ ] If Task 1 found a stale or incomplete attempt, preserve or quarantine the partial outputs, then relaunch one clean compare under the stable root so the final root contains one coherent result set.
- [ ] Launch long-running work in tmux after activating `ptycho311`, and track the exact PID from the launched wrapper command.
- [ ] Use the fixed contract exactly and keep dataset, probe, seed, loss, scheduler, epochs, output mode, and row list identical across `pinn_hybrid_resnet` and `pinn_ffno`.
- [ ] If the first resumed/relaunched run reveals one concrete FFNO CDI-contract bug, do one narrow fix cycle and rerun once. Mark `BLOCKED` only if FFNO still cannot satisfy the existing CDI contract after that documented narrow fix attempt.

**Verification during/after run:**
- [ ] The tracked PID exits `0`.
- [ ] The stable root contains both `runs/pinn_hybrid_resnet/` and `runs/pinn_ffno/`.
- [ ] Wrapper-level merged artifacts exist and are freshly written after the successful run.
- [ ] Row-level metrics, history, reconstructions, and invocation artifacts exist for both rows.
- [ ] Standard amplitude/phase comparison visuals exist for the row pair.

### Task 5: Publish Durable Result Notes And Discoverability Updates

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_summary.md`
- Modify: `docs/studies/index.md`
- Modify if needed: `docs/index.md`
- Modify if needed: `docs/findings.md`

- [ ] Finalize the summary with:
  - final status (`completed` or explicit blocker state)
  - exact stable output root
  - key metrics for both rows
  - figure paths
  - the recovery/resume/relaunch decision that produced the final result
  - the explicit claim boundary that this is a prerequisite row pair, not the full `lines128` paper benchmark
- [ ] Update `docs/studies/index.md` so the `grid-lines-n128-ffno-vs-hybrid-resnet-best-contract` entry points to the final root and final status rather than the in-progress placeholder state.
- [ ] Update `docs/index.md` only if the new preflight or final summary should be surfaced from the documentation hub.
- [ ] If execution exposed a durable FFNO/CDI pitfall that future workers need to avoid, add a concise evidence-backed finding to `docs/findings.md`.

**Verification before closing the item:**
- [ ] The summary, study-index entry, and any index updates all point to the same stable output root.
- [ ] Any blocked outcome is described as a contract/resource blocker, not as silent protocol drift.
- [ ] The summary explicitly states that this item unlocks later `lines128` work but does not itself produce the full paper benchmark.

## Completion Criteria

- A durable preflight note exists for the recovered Lines128 contract and clearly records the resume/relaunch decision.
- `ffno` runs through the existing CDI Torch path under the same output/stitching contract as `hybrid_resnet`, or the item ends with an explicit documented contract blocker after a narrow fix attempt.
- A fresh equal-footing `pinn_hybrid_resnet` versus `pinn_ffno` compare exists under one stable output root with shared dataset, probe, loss, scheduler, epoch budget, and seed settings.
- The required deterministic checks pass and their logs are archived.
- Metrics, figures, and provenance are captured for both rows and summarized durably.
- Docs clearly state that this work is prerequisite CDI evidence for later `lines128` packaging, not the complete four-row paper table.
