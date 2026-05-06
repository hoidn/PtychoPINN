# Lines128 SRU-Net ConvNeXt Bottleneck Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement this plan task-by-task. Keep this file as the execution authority for the selected backlog item.

**Goal:** Add one bounded same-contract `lines128` CDI SRU-Net variant that replaces the current ResNet bottleneck with a ConvNeXt-style bottleneck, launch only that missing PINN row against the locked `pinn_hybrid_resnet` anchor by lineage, and publish append-only decision-support evidence without reopening the completed six-row CDI bundle.

**Architecture:** Reuse the authoritative `lines128` complete-table bundle as the fixed contract and baseline-row provenance source. Implement a new explicit architecture surface `hybrid_resnet_convnext_bottleneck` that keeps the SRU-Net encoder, downsampling path, decoder, skip policy, bottleneck width, block count, training shell, and output contract unchanged while swapping only the constant-resolution bottleneck body for a ConvNeXt-style stack. Route the new row through the existing Torch runner and compare-wrapper flow, then collate a new append-only ablation root that reuses `pinn_hybrid_resnet` by lineage and launches only `pinn_hybrid_resnet_convnext_bottleneck`.

**Tech Stack:** PATH `python`, `ptycho311` for long-running launches, PyTorch/Lightning, `ptycho_torch/generators/hybrid_resnet.py`, `ptycho_torch/generators/resnet_components.py`, `scripts/studies/grid_lines_torch_runner.py`, `scripts/studies/grid_lines_compare_wrapper.py`, a narrow `lines128` ablation helper under `scripts/studies/`, Markdown/JSON evidence indexes.

---

## Selected Backlog Objective

- Add and run a ConvNeXt-style bottleneck replacement for SRU-Net on the frozen `lines128` CDI contract.
- Reuse the completed `pinn_hybrid_resnet` row by lineage from the authoritative complete-table bundle; do not rerun it.
- Launch exactly one fresh same-contract row for this item:
  - `pinn_hybrid_resnet_convnext_bottleneck`
- Publish a concise append-only durable summary at:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_srunet_convnext_bottleneck_ablation_summary.md`

## Scope Boundaries

### In Scope

- A narrow architecture variant such as `hybrid_resnet_convnext_bottleneck` with an explicit row id and label.
- A ConvNeXt-style constant-resolution bottleneck that replaces only the current `ResnetBottleneck` body while preserving:
  - encoder branch structure
  - downsampling topology
  - decoder family
  - skip policy
  - bottleneck width
  - bottleneck block count
  - dataset, split, probe preprocessing, seed, epoch budget, scheduler, output mode, loss, metric schema, fixed visual samples, and shared visual scales
- One append-only ablation run root plus durable evidence/index updates.

### Explicit Non-Goals

- Do not rerun completed CNN, FNO, FFNO, U-NO, spectral-bottleneck, baseline SRU-Net, skip/residual-ablation, or branch/objective-ablation rows just to assemble this comparison.
- Do not change encoder branch gates, encoder branch selection, decoder skip style, residual-scale policy, loss, probe, schedule, dataset contract, visual scales, or training procedure in the ConvNeXt row.
- Do not combine this item with the SRU-Net branch/objective ablation or with broader Hybrid ResNet mechanism sweeps.
- Do not run the canonical tiny ConvNeXt LayerScale initialization in this item. The first row must use the current SRU-Net LayerScale convention so block family is the only intended axis.
- Do not rewrite or supersede the authoritative six-row CDI complete-table bundle or the U-NO append-only extension.

## Steering, Roadmap, And Policy Constraints

- Steering keeps roadmap gates and fairness constraints binding. This item must remain an equal-footing same-contract compare and must not silently relax the protocol to make the ConvNeXt row easier.
- The roadmap explicitly allows this SRU-Net ConvNeXt bottleneck ablation only as an optional Phase 3 append-only mechanism item after the six-row `lines128` CDI bundle is complete. It is allowed because it changes only the bottleneck block family and reuses completed rows by lineage.
- The roadmap also states these optional Phase 3 extensions are lower priority than eligible Phase 2 PDEBench and candidate-lane work unless steering reprioritizes them. The selector has already chosen this item, so implementation must keep it tightly bounded and append-only.
- Long-running execution stays under implementation ownership until terminal success or recoverable failure handling is complete:
  - launch in `tmux`
  - activate `ptycho311`
  - keep PATH `python`
  - track the exact launched PID
  - accept completion only when the tracked PID exits `0` and required fresh artifacts exist
- Do not mark the item `BLOCKED` for normal verification failures, import/path issues, or first-pass harness regressions. Diagnose, apply a narrow fix, and rerun first. Reserve `BLOCKED` for missing external resources, unavailable hardware, roadmap conflict, user decision required, or a failure that remains unrecoverable after a documented narrow fix attempt.
- Keep core physics/model stability rules intact:
  - do not modify `ptycho/model.py`
  - do not modify `ptycho/diffsim.py`
  - do not modify `ptycho/tf_helper.py`

## Prerequisite Status

- Binding baseline authority is already satisfied by:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - authoritative root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- Relevant same-contract context already exists and should be referenced, not rerun:
  - skip/residual ablation summary:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_skip_residual_ablation_summary.md`
  - branch/objective ablation summary:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_srunet_branch_objective_ablation_summary.md`
  - U-NO extension summary:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_summary.md`
- Progress-ledger note:
  - `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` records early approved tranches and does not enumerate the later Phase 3 CDI completions this item depends on.
  - For this item, treat the selected backlog context plus the completed CDI summary authorities above as the effective satisfied prerequisite authority.

## Fixed Contract To Preserve

- Dataset contract id: `cdi_lines128_seed3`
- Baseline reused row: `pinn_hybrid_resnet`
- Fixed headline source root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- Frozen fields that must not drift:
  - `N=128`
  - `gridsize=1`
  - synthetic grid-lines with `set_phi=True`
  - custom Run1084 probe with `probe_scale_mode=pad_extrapolate`
  - `nimgs_train=2`
  - `nimgs_test=2`
  - `nphotons=1e9`
  - `seed=3`
  - `torch_epochs=40`
  - `torch_learning_rate=2e-4`
  - `torch_scheduler=ReduceLROnPlateau`
  - `torch_plateau_factor=0.5`
  - `torch_plateau_patience=2`
  - `torch_plateau_min_lr=1e-4`
  - `torch_plateau_threshold=0.0`
  - `torch_loss_mode=mae`
  - `torch_mae_pred_l2_match_target=off`
  - `torch_output_mode=real_imag`
  - `probe_mask=off`
  - `fno_modes=12`
  - `fno_width=32`
  - `fno_blocks=4`
  - `fno_cnn_blocks=2`
  - fixed visual sample ids `0` and `1`
- The remaining row-local launch fields not listed above must be copied from the authoritative baseline artifacts before launching the ConvNeXt row, especially:
  - `runs/pinn_hybrid_resnet/config.json`
  - `model_manifest.json`
  - `metric_schema.json`
  - any wrapper-level benchmark manifest that records the accepted row contract

## Implementation Architecture

- **Unit 1: ConvNeXt bottleneck module and architecture registration**
  - Add a channels-first ConvNeXt-style bottleneck implementation and expose it through a new explicit architecture id, keeping the rest of the SRU-Net shell unchanged.
- **Unit 2: Same-contract Lines128 launch and collation path**
  - Route the new row id through the existing grid-lines runner/wrapper surfaces, reuse the baseline row by lineage, and launch only the missing ConvNeXt row under a new ablation root.
- **Unit 3: Durable evidence surfacing**
  - Write one summary authority and refresh the NeurIPS evidence/index surfaces so the new row is discoverable and clearly labeled as append-only decision-support evidence.

## Concrete File And Artifact Targets

### Mandatory code and test surfaces

- Modify: `ptycho_torch/generators/resnet_components.py`
- Modify: `ptycho_torch/generators/hybrid_resnet.py`
- Modify: `ptycho/config/config.py`
- Modify: `ptycho_torch/config_params.py`
- Modify if architecture validation or execution payload routing requires it:
  - `ptycho_torch/workflows/components.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Add preferred narrow ablation helper:
  - `scripts/studies/lines128_srunet_convnext_bottleneck_ablation.py`
  - If a new helper is not needed, touch the smallest safe surface in `scripts/studies/lines128_paper_benchmark.py` instead.
- Modify or add tests:
  - `tests/torch/test_fno_generators.py`
  - `tests/torch/test_grid_lines_torch_runner.py`
  - `tests/test_grid_lines_compare_wrapper.py`
  - add focused study coverage such as:
    `tests/studies/test_lines128_srunet_convnext_bottleneck_ablation.py`

### Mandatory contract outputs

- New backlog artifact root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-srunet-convnext-bottleneck-ablation/`
- Under that root, create at minimum:
  - `execution_manifest.json`
  - `row_contract_audit.json`
  - `comparison_summary.json`
  - `model_manifest.json`
  - `metrics.json`
  - `verification/`
  - one fresh `runs/<timestamp-or-row-root>/` tree containing row-local invocation, config, history, metrics, reconstruction, visual, and completion-proof artifacts for `pinn_hybrid_resnet_convnext_bottleneck`
- Durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_srunet_convnext_bottleneck_ablation_summary.md`
- Required discoverability/index updates:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/studies/index.md`

### Preferred packaging only after core completion

- A small `bundle/` directory under the new artifact root that promotes the reused baseline row by lineage and collates only the fresh ConvNeXt row plus comparison outputs.
- TeX/CSV row fragments or figure refreshes aligned to existing `lines128` naming conventions, but only if they fall out naturally from the shared collation helpers.
- Any paper-local table refresh remains non-blocking for this backlog item.

## Execution Checklist

### Tranche 1: Freeze The Baseline Authority And Scaffold The Append-Only Root

- [ ] Create the durable summary file `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_srunet_convnext_bottleneck_ablation_summary.md` immediately as an execution scaffold with:
  - selected backlog item id
  - this plan path
  - authoritative baseline root
  - intended fresh row id
  - claim boundary placeholder `decision_support_append_only`
  - explicit statement that the six-row CDI bundle remains unchanged
- [ ] Create `execution_manifest.json` under the new artifact root with:
  - authoritative baseline source root
  - reused row `pinn_hybrid_resnet`
  - fresh row `pinn_hybrid_resnet_convnext_bottleneck`
  - paper label `Hybrid ResNet (ConvNeXt bottleneck) + PINN`
  - architecture id `hybrid_resnet_convnext_bottleneck`
  - fixed sample ids and contract id
  - referenced related-summary paths
- [ ] Audit the exact accepted baseline-row contract and write `row_contract_audit.json` before touching the runner/wrapper launch logic.
- [ ] Decide the implementation route explicitly:
  - use a new architecture id, not an overloaded hidden flag on `hybrid_resnet`
  - use the current SRU-Net LayerScale convention for the first ConvNeXt row
  - defer canonical tiny ConvNeXt LayerScale initialization

Verification for Tranche 1:

- [ ] **Blocking:** the summary scaffold, `execution_manifest.json`, and `row_contract_audit.json` all exist and agree on the same baseline root, row ids, and claim boundary before expensive work starts.
- [ ] **Blocking:** if baseline provenance fields are missing or inconsistent, repair the audit surface first instead of launching a drifted row.

### Tranche 2: Implement The ConvNeXt Bottleneck Variant

- [ ] Add a narrow ConvNeXt-style bottleneck stack in `ptycho_torch/generators/resnet_components.py` with:
  - depthwise spatial convolution
  - normalization suitable for channels-first tensors
  - pointwise expansion and projection
  - GELU
  - residual LayerScale initialized to the current SRU-Net convention for this first row
- [ ] Keep the bottleneck constant-resolution and parameterized by the same bottleneck width and block count used by the baseline SRU-Net row.
- [ ] Register a new explicit architecture path in `ptycho_torch/generators/hybrid_resnet.py` that reuses the existing lifter, encoder, downsample, adapter, decoder, skip-fusion, and output code and swaps only the bottleneck implementation.
- [ ] Extend architecture-validation literals and config choices in `ptycho/config/config.py`, `ptycho_torch/config_params.py`, and any execution-payload or CLI validation surface that enumerates valid architectures.
- [ ] Ensure the new architecture emits clear row-local provenance so later summaries can prove it differs only by bottleneck family.

Verification for Tranche 2:

- [ ] **Blocking:** generator-focused tests prove the new architecture builds, preserves tensor shapes, keeps the same output contract, and uses the expected LayerScale initialization.
- [ ] **Blocking:** invalid architecture or invalid ConvNeXt-specific parameter combinations fail loudly and deterministically.
- [ ] **Supporting:** `python -m compileall -q ptycho_torch scripts/studies` after code edits to catch syntax/import drift before runner integration.

### Tranche 3: Wire The New Row Through The Lines128 Study Surfaces

- [ ] Extend `scripts/studies/grid_lines_torch_runner.py` so the new architecture id is accepted in CLI parsing, config validation, command reconstruction, deterministic-mode handling, and provenance emission.
- [ ] Extend `scripts/studies/grid_lines_compare_wrapper.py` with a canonical row spec for:
  - `pinn_hybrid_resnet_convnext_bottleneck`
  - architecture `hybrid_resnet_convnext_bottleneck`
  - training mode `pinn`
  - explicit human label `Hybrid ResNet (ConvNeXt bottleneck) + PINN`
- [ ] Keep the wrapper/runner route same-contract:
  - baseline reused by lineage
  - no auxiliary comparator reruns
  - no training-procedure changes
- [ ] Add a narrow ablation helper or minimal harness extension that can:
  - promote the baseline `pinn_hybrid_resnet` row by lineage
  - launch only the missing ConvNeXt row
  - collate merged metrics and visuals under the new append-only root
  - fail closed if the reused baseline lineage or the fresh row completion proof is missing

Verification for Tranche 3:

- [ ] **Blocking:** runner tests prove the new architecture is accepted and reconstructed correctly in commands/config artifacts.
- [ ] **Blocking:** compare-wrapper tests prove the new row spec, label, and architecture mapping are stable and do not alter existing rows.
- [ ] **Blocking:** if a new ablation helper is added, a focused study-level test proves no-rerun baseline promotion and correct fresh-row planning.
- [ ] **Supporting:** a dry-run or preflight that prints the resolved row roster, source lineage, and output root before the full launch.

### Tranche 4: Run The Required Deterministic Gates Before Expensive Training

- [ ] Run the backlog item’s required deterministic checks exactly as written once the code and minimal harness path are in place:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md"),
    Path("docs/backlog/done/2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
    Path("scripts/studies/grid_lines_compare_wrapper.py"),
    Path("scripts/studies/grid_lines_torch_runner.py"),
    Path("ptycho_torch/generators/hybrid_resnet.py"),
    Path("ptycho_torch/generators/resnet_components.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing SRU-Net ConvNeXt bottleneck ablation inputs: {missing}")
print("SRU-Net ConvNeXt bottleneck ablation inputs present")
PY
pytest -q tests/torch/test_fno_generators.py -k "hybrid_resnet or convnext"
pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or convnext"
pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or convnext"
python -m compileall -q scripts/studies ptycho_torch
```

- [ ] Add one narrower supplemental blocking selector if the new ablation helper has coverage outside those existing selectors.
- [ ] Archive all verification logs under the new item root’s `verification/` directory.

Verification for Tranche 4:

- [ ] **Blocking:** every check command above passes before any expensive training launch.
- [ ] **Blocking:** if a required check fails for a normal harness/environment reason, diagnose, fix, and rerun first rather than declaring `BLOCKED`.
- [ ] **Supporting:** any additional focused study-selector logs that cover the new helper or new architecture registry path.

### Tranche 5: Launch Only The Missing ConvNeXt Row And Keep Ownership Until Completion

- [ ] Create a fresh run root under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-srunet-convnext-bottleneck-ablation/runs/`
- [ ] Launch only `pinn_hybrid_resnet_convnext_bottleneck` in `tmux`, using PATH `python` inside `ptycho311`, with the exact fixed contract from `row_contract_audit.json`.
- [ ] Reuse `pinn_hybrid_resnet` by lineage from the authoritative complete-table root; do not relaunch it.
- [ ] Keep the changed factor isolated to the bottleneck family:
  - baseline shell unchanged
  - ConvNeXt bottleneck active
  - current SRU-Net LayerScale convention active
  - all other knobs identical to the baseline row
- [ ] If the first launch fails, diagnose the narrow cause, apply one bounded fix, and relaunch or resume. Do not broaden scope or add follow-up rows opportunistically.

Verification for Tranche 5:

- [ ] **Blocking:** the tracked PID exits `0`.
- [ ] **Blocking:** the fresh row root contains invocation, config, metrics, history, reconstruction, visuals, and completion-proof artifacts with fresh timestamps.
- [ ] **Blocking:** the append-only collation output includes both the reused baseline lineage and the fresh ConvNeXt row with merged metrics/comparison outputs.
- [ ] **Supporting:** tmux/live logs, training curves, and row-local notes explaining any narrow recovery step that was required.

### Tranche 6: Summarize The Outcome And Refresh Discoverability

- [ ] Finalize `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_srunet_convnext_bottleneck_ablation_summary.md` with:
  - fixed contract and authoritative baseline source root
  - exact fresh row id and architecture id
  - direct metric deltas versus `pinn_hybrid_resnet`
  - visual/qualitative interpretation
  - explicit statement that this is append-only decision-support evidence
  - explicit non-promotion statement: the completed six-row CDI bundle remains the headline authority
  - note that canonical tiny ConvNeXt LayerScale initialization is deferred
- [ ] Update `model_variant_index.json` with a new `cdi_lines128_seed3` variant entry for `pinn_hybrid_resnet_convnext_bottleneck`, including:
  - `nearest_anchor_row: pinn_hybrid_resnet`
  - `training_mode: pinn`
  - `architecture_id: hybrid_resnet_convnext_bottleneck`
  - `changed_factor` text that isolates bottleneck family only
- [ ] Add a new `ablation_index.json` family entry for the SRU-Net ConvNeXt bottleneck ablation.
- [ ] Update `evidence_matrix.md`, `paper_evidence_index.md`, and `docs/studies/index.md` so the row and summary are discoverable from the normal NeurIPS CDI evidence surfaces.
- [ ] Keep any paper-local table refresh or TeX fragment generation non-blocking unless it is required to make the ablation bundle itself auditable.

Verification for Tranche 6:

- [ ] **Blocking:** the durable summary exists and points to the correct artifact root and authoritative baseline root.
- [ ] **Blocking:** evidence/index surfaces agree on row id, architecture id, artifact root, and claim boundary.
- [ ] **Blocking:** the backlog item’s deterministic checks still pass after documentation/index updates.
- [ ] **Supporting:** optional table-fragment refresh diff review if a non-blocking packaging surface was updated.

## Required Deterministic Checks

Treat these as required deterministic gates for this backlog item. They are blocking before the expensive launch, and they should be rerun as final closeout checks after the summary/index updates unless a strictly stronger checked-in replacement is justified in the execution report.

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md"),
    Path("docs/backlog/done/2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
    Path("scripts/studies/grid_lines_compare_wrapper.py"),
    Path("scripts/studies/grid_lines_torch_runner.py"),
    Path("ptycho_torch/generators/hybrid_resnet.py"),
    Path("ptycho_torch/generators/resnet_components.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing SRU-Net ConvNeXt bottleneck ablation inputs: {missing}")
print("SRU-Net ConvNeXt bottleneck ablation inputs present")
PY
pytest -q tests/torch/test_fno_generators.py -k "hybrid_resnet or convnext"
pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or convnext"
pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or convnext"
python -m compileall -q scripts/studies ptycho_torch
```

## Completion Criteria

- The only fresh scientific row launched for this item is `pinn_hybrid_resnet_convnext_bottleneck`.
- The ConvNeXt row is auditable as same-contract against `pinn_hybrid_resnet`, with no hidden drift in training shell, data, visuals, or metric schema.
- The durable summary and all required evidence surfaces are updated consistently.
- The result is clearly labeled as append-only `lines128` CDI mechanism evidence and does not rewrite or overclaim beyond the completed six-row CDI authority.
