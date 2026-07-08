# Lines128 SRU-Net Branch And Objective Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a bounded, same-contract `lines128` CDI SRU-Net follow-up that isolates encoder-branch necessity and adds the missing supervised SRU-Net objective-control row without rewriting the locked six-row CDI authority or broadening into a new sweep.

**Architecture:** Reuse the completed `lines128` CDI paper bundle as the fixed contract and promote the existing `pinn_hybrid_resnet` row by lineage. Add one narrow Torch-only encoder-branch ablation control that can disable the spectral branch or the local spatial branch without changing the residual path, bottleneck, decoder, loss, schedule, probe, or dataset contract; then route the new row ids through the existing grid-lines compare/runner stack and collate an append-only ablation bundle under the new backlog artifact root. Keep the result decision-support and append-only: compare against completed FFNO/FNO/U-NO/CNN/spectral rows by reference, not by rerun.

**Tech Stack:** Python via PATH `python`, `ptycho311` for long-running launches, PyTorch/Lightning, `scripts/studies/grid_lines_compare_wrapper.py`, `scripts/studies/grid_lines_torch_runner.py`, `scripts/studies/lines128_paper_benchmark.py`, `ptycho_torch/generators/hybrid_resnet.py`, Markdown/JSON evidence indexes.

---

## Selected Objective

- Add exactly these missing same-contract rows unless a documented audit proves one already exists and is fully usable:
  - `pinn_hybrid_resnet_encoder_conv_only`
  - `pinn_hybrid_resnet_encoder_spectral_only`
  - `supervised_hybrid_resnet` or an equivalently explicit row id labeled `SRU-Net + supervised`
- Reuse the existing `pinn_hybrid_resnet` row by lineage as the fixed SRU-Net + PINN anchor.
- Interpret the new results as:
  - branch-necessity evidence for local spatial features versus global spectral coupling
  - an objective-control comparison for `SRU-Net + PINN` versus `SRU-Net + supervised`

## Scope Boundaries

### In Scope

- Narrow model/config plumbing needed to disable exactly one SRU-Net encoder branch at a time.
- Compare-wrapper and runner support for the two branch-ablation row ids plus the supervised SRU-Net row.
- An append-only ablation bundle under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-srunet-branch-objective-ablation/` with baseline lineage plus fresh row-local artifacts.
- A durable summary and discoverability updates.

### Explicit Non-Goals

- Do not rerun completed FFNO, FNO, U-NO, CNN/U-Net-class, spectral-bottleneck, or baseline SRU-Net PINN rows unless a deterministic audit proves a reused prerequisite row is unusable.
- Do not turn this into a broad SRU-Net hyperparameter sweep, fusion-mode redo, skip-style study, bottleneck study, `256x256` study, CNS work, BRDT work, or WaveBench work.
- Do not conflate this item with the completed branch-gating / LayerScale ablation. That item tested learned fusion weighting; this item tests branch necessity by removal.
- Do not change dataset, split, probe preprocessing, seed, epoch budget, scheduler, loss, output mode, bottleneck, decoder, skip setting, metric schema, fixed sample ids, or shared visual scales.
- Do not overwrite the six-row complete-table authority or the eight-row U-NO extension authority.

## Steering, Roadmap, And Policy Constraints

- Steering keeps the current Phase 2 plus Phase 3 selection window open and allows one bounded CDI-strengthening follow-up; this item must stay bounded and append-only.
- The roadmap’s Phase 3 contract remains binding: `lines128` CDI is fixed at synthetic grid-lines `N=128`, `seed=3`, `40` epochs, fixed sample ids `0` and `1`, shared visual scales, and existing paper metric schema.
- Fairness is mandatory: if a branch-disable row cannot preserve the locked contract, record the incompatibility precisely instead of drifting the protocol.
- Long-running execution stays under implementation ownership until terminal success or recoverable failure handling is complete:
  - launch in `tmux`
  - activate `ptycho311`
  - keep PATH `python`
  - track the exact launched PID
  - accept a run only when the tracked PID exits `0` and required artifacts are freshly written
- Do not mark the item `BLOCKED` for ordinary test failures, import/path issues, or first-pass harness regressions. Diagnose, apply a narrow fix, and rerun first. Reserve `BLOCKED` for missing hardware/resources, roadmap conflict, unavailable prerequisite evidence, external dependency outside authority, required user decision, or a failure that remains unrecoverable after a documented narrow fix attempt.

## Prerequisite Status

- Satisfied by completed backlog outputs and summaries:
  - complete six-row CDI authority:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - supervised FFNO objective-control extension:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_supervised_equivalent_rows_summary.md`
  - U-NO append-only extension:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_summary.md`
  - completed encoder-fusion ablation used as contrast-only context:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_encoder_fusion_variants_summary.md`
- Important ledger note:
  - `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` only records early Phases 0-2 and does not enumerate the later Phase 3 CDI completions that this backlog item depends on.
  - For this item, treat the selected backlog context plus the completed backlog summaries above as the effective satisfied prerequisite authority.

## Implementation Architecture

- **Unit 1: Deterministic branch-disable contract**
  - Add a new explicit SRU-Net encoder branch-ablation control separate from the existing `hybrid_encoder_fusion_mode`.
  - The control must disable exactly one encoder branch before fusion and leave the identity residual, bottleneck, decoder, and downstream training shell untouched.
  - Do not fake branch disablement with learned gates initialized near zero; the ablation must be deterministic and auditable.

- **Unit 2: Lines128 launch/collation path**
  - Route the new row ids through the existing Torch compare-wrapper and runner surfaces.
  - Reuse `pinn_hybrid_resnet` by lineage and freshly launch only the missing rows.
  - Collate a small append-only ablation bundle with row-local artifacts and explicit baseline lineage rather than rebuilding the full paper table.

- **Unit 3: Durable evidence surfacing**
  - Write one summary authority and update the evidence indexes so the new branch/objective-control family is discoverable from the normal NeurIPS entry points.
  - Keep any paper-local table refresh optional and downstream of the verified ablation bundle.

## File And Artifact Targets

### Mandatory code / test surfaces

- Modify: `ptycho_torch/generators/hybrid_resnet.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: `scripts/studies/lines128_paper_benchmark.py` or add one narrowly-scoped helper under `scripts/studies/` if that is cleaner for append-only ablation collation
- Modify: `tests/torch/test_fno_generators.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Modify: `tests/test_grid_lines_compare_wrapper.py`
- Modify or add focused study-level tests:
  - `tests/studies/test_lines128_paper_benchmark.py`
  - or a new `tests/studies/test_lines128_srunet_branch_objective_ablation.py`

### Mandatory contract outputs

- New item artifact root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-srunet-branch-objective-ablation/`
- Fresh row-local run roots for each launched row with invocation/config/history/metrics/checkpoint/recon/visual/completion-proof artifacts
- A collated append-only ablation bundle with:
  - reused `pinn_hybrid_resnet` baseline lineage
  - fresh branch/objective row manifests
  - merged metric payload for this ablation family
  - explicit claim boundary such as `decision_support_append_only`
- Durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_srunet_branch_objective_ablation_summary.md`
- Discoverability/index updates:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/studies/index.md`

### Preferred packaging only after core completion

- Refresh existing paper-local objective-control tables only if the append-only bundle is complete and the refresh can preserve lineage to the immutable authorities:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_objective_comparison.*`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.*`
- Do not make manuscript-table refresh a blocker for completing this backlog item.

## Execution Checklist

### Task 1: Freeze The Fixed Contract And Baseline Lineage

- [ ] Confirm the locked `lines128` contract from the complete-table summary and current indexes:
  - dataset contract id `cdi_lines128_seed3`
  - `seed=3`
  - `40` epochs
  - fixed sample ids `0`, `1`
  - existing probe preprocessing and metric schema
- [ ] Audit the baseline row source for `pinn_hybrid_resnet` and record the exact reused root in the new item’s execution manifest.
- [ ] Decide the ablation bundle scope up front:
  - mandatory rows: baseline lineage + `pinn_hybrid_resnet_encoder_conv_only` + `pinn_hybrid_resnet_encoder_spectral_only` + `supervised_hybrid_resnet`
  - optional row: `neither/linearized` only if it is cleanly implemented without changing the locked shell
- [ ] If a fresh collation mode or helper is needed, keep it append-only and small; do not repurpose it into a new full-table builder.

Verification for Task 1:

- Blocking:
  - required input presence command from the backlog item
- Supporting:
  - any focused manifest/contract dry-run added for the new collation path

### Task 2: Implement Deterministic Encoder Branch Disablement

- [ ] Add a distinct branch-ablation control in `ptycho_torch/generators/hybrid_resnet.py` that can express at least:
  - baseline / both branches enabled
  - conv-only
  - spectral-only
  - optional neither only if clean and low-risk
- [ ] Keep the ablation control orthogonal to `hybrid_encoder_fusion_mode`; existing `layerscale` / `branch_gated` paths must keep working unchanged.
- [ ] Implement the forward-path behavior so only the selected branch output is removed or preserved before fusion; do not change decoder, bottleneck, residual addition, or output mode.
- [ ] Plumb the new control through `scripts/studies/grid_lines_torch_runner.py` config, CLI, validation, provenance emission, and runner command reconstruction.
- [ ] Add focused generator and runner tests that prove:
  - branch-disable modes validate correctly
  - baseline behavior stays unchanged
  - conv-only truly disables the spectral path
  - spectral-only truly disables the local path
  - invalid branch-mode combinations fail loudly

Verification for Task 2:

- Blocking before any long run:
  - focused generator tests
  - focused runner tests
- Supporting:
  - compile check on the touched modules if done incrementally

### Task 3: Add New Row Routing And Objective-Control Support

- [ ] Extend `scripts/studies/grid_lines_compare_wrapper.py` with canonical row specs, labels, architecture ids, and training-procedure overrides for:
  - `pinn_hybrid_resnet_encoder_conv_only`
  - `pinn_hybrid_resnet_encoder_spectral_only`
  - `supervised_hybrid_resnet`
- [ ] Keep labels explicit about architecture plus training procedure so tables do not collapse `SRU-Net + PINN` and `SRU-Net + supervised`.
- [ ] Route the supervised SRU-Net row through the exact same SRU-Net body as `pinn_hybrid_resnet`, changing only the training procedure and the new row id metadata.
- [ ] Update or add compare-wrapper tests for row normalization, route selection, labels, and row metadata.

Verification for Task 3:

- Blocking before any long run:
  - focused compare-wrapper tests
  - rerun focused runner tests if row-spec plumbing changed command construction
- Supporting:
  - a dry-run/preflight that prints the selected row plan and resolved overrides

### Task 4: Build The Append-Only Ablation Bundle Path

- [ ] Extend `scripts/studies/lines128_paper_benchmark.py` or add a narrow sibling helper so this backlog item can:
  - promote the existing `pinn_hybrid_resnet` baseline row by lineage
  - launch only the missing rows
  - collate an ablation-family metrics/manifests bundle under the new item root
- [ ] Record explicit source-lineage metadata pointing back to the immutable six-row authority and any referenced extension roots without overwriting them.
- [ ] Ensure the bundle can surface row-local completion proofs, merged metrics, and fixed-contract metadata for the new family.
- [ ] Add study-level tests for:
  - row roster validation
  - no-rerun baseline promotion
  - append-only output-root behavior
  - failure when expected lineage or row-local completion artifacts are missing

Verification for Task 4:

- Blocking before launch:
  - focused study-level tests for the new bundle path
- Supporting:
  - local dry-run/preflight showing row order, output root, and source lineage

### Task 5: Launch Fresh Rows And Keep Ownership Until Completion

- [ ] Launch the fresh rows in `tmux` from the repo root after code/tests/preflight pass.
- [ ] Activate `ptycho311`, use PATH `python`, and track the exact launcher PID.
- [ ] Reuse the baseline SRU-Net row by lineage; do not relaunch it.
- [ ] Accept the launch only when:
  - the tracked PID exits `0`
  - each fresh row has row-local invocation/config/history/metrics/recon/completion-proof artifacts
  - the append-only ablation bundle has merged metrics/manifests and fresh timestamps
- [ ] If a long run fails, diagnose the narrow cause, fix it, and relaunch or resume once. Do not declare `BLOCKED` until the narrow fix path has been attempted and documented.

Verification for Task 5:

- Blocking:
  - tracked PID exit `0`
  - required fresh artifacts present and freshly written
  - row-local completion proof present for each launched row
- Supporting:
  - live/tmux logs
  - loss curves and intermediate histories

### Task 6: Summarize Results And Refresh Discoverability

- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_srunet_branch_objective_ablation_summary.md` with:
  - fixed contract
  - reused baseline lineage
  - fresh row roster
  - key amplitude/phase metric deltas versus `pinn_hybrid_resnet`
  - the objective-control interpretation for `SRU-Net + supervised`
  - explicit boundary that this family is append-only decision-support evidence and does not replace the six-row CDI authority
- [ ] Update `model_variant_index.json` with one entry per fresh row under `dataset_contract_id: cdi_lines128_seed3`.
- [ ] Add a new `ablation_index.json` family for SRU-Net branch/objective ablations.
- [ ] Update `evidence_matrix.md`, `paper_evidence_index.md`, and `docs/studies/index.md` so the new result is discoverable from both CDI evidence surfaces and the study map.
- [ ] Refresh paper-local objective-control tables only if the append-only bundle is complete and the refresh does not rewrite prior authorities.

Verification for Task 6:

- Blocking:
  - summary exists and points to the correct artifact root
  - required evidence indexes are updated consistently
  - backlog item deterministic checks all pass
- Supporting:
  - optional paper-table refresh diff review

## Required Deterministic Checks

Treat these as final blocking gates for the backlog item unless a strictly stronger replacement is checked in and justified in the execution report:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md"),
    Path("docs/backlog/done/2026-04-29-cdi-lines128-supervised-equivalent-rows.md"),
    Path("docs/backlog/done/2026-04-30-cdi-lines128-uno-table-extension.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_encoder_fusion_variants_summary.md"),
    Path("scripts/studies/grid_lines_compare_wrapper.py"),
    Path("scripts/studies/grid_lines_torch_runner.py"),
    Path("ptycho_torch/generators/hybrid_resnet.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing SRU-Net branch/objective ablation inputs: {missing}")
print("SRU-Net branch/objective ablation inputs present")
PY
pytest -q tests/torch/test_fno_generators.py -k "hybrid_resnet or hybrid_encoder"
pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or hybrid_encoder or supervised"
pytest -q tests/test_grid_lines_compare_wrapper.py
python -m compileall -q scripts/studies ptycho_torch
```

## Completion Standard

- The item is complete only when the new SRU-Net ablation family is reproducible, append-only, and discoverable:
  - the locked `lines128` contract is preserved
  - the baseline SRU-Net row is reused by lineage rather than rerun
  - the two required branch-disable rows and the supervised SRU-Net row are present or carry a precise row-level blocker
  - the summary and evidence indexes reflect the new family consistently
  - the six-row complete CDI authority and the eight-row U-NO extension remain immutable upstream authorities
