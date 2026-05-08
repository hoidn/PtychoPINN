# CDI Lines128 Supervised FFNO Depth-24 No-Refiner Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create worktrees.

**Goal:** Add one supervised CDI `lines128` FFNO row with `fno_blocks=24` and `fno_cnn_blocks=0`, compare it against the corrected four-block no-refiner `supervised_ffno` row by lineage, and publish it as the supervised depth-24 companion required before any later final FFNO depth-24 paper refresh.

**Architecture:** Reuse the corrected four-block no-refiner supervised row and the completed PINN depth-24 ablation as locked lineage inputs. Add only the smallest wrapper/runner row-spec support needed for a distinct `supervised_ffno_depth24` row, launch exactly one new supervised rerun under an item-local artifact root, then write a depth-only contract audit plus summary/index updates without promoting the result into paper tables. Long-running execution remains under implementation ownership until the tracked PID exits `0` and the required row-local artifacts are freshly written.

**Tech Stack:** PATH `python`, `ptycho311` for long-running launches, PyTorch/Lightning, `scripts/studies/grid_lines_compare_wrapper.py`, `scripts/studies/grid_lines_torch_runner.py`, `ptycho_torch/generators/ffno.py`, Markdown/JSON evidence surfaces under `docs/plans/NEURIPS-HYBRID-RESNET-2026/`.

---

## Selected Backlog Objective

- Add one new supervised CDI `lines128` FFNO row id, preferably `supervised_ffno_depth24`, whose only intentional architectural change versus the corrected four-block no-refiner `supervised_ffno` row is `fno_blocks: 4 -> 24`.
- Compare the fresh supervised depth-24 row only against the corrected four-block no-refiner `supervised_ffno` root from `2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun`.
- Produce the supervised depth-24 companion needed before `2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh` can decide whether the depth-24 FFNO pair should replace the current four-block no-refiner interim FFNO rows.

## Scope

### In Scope

- Audit the current supervised FFNO launch path and determine whether it can already emit a distinct supervised depth-24 row with truthful row-local provenance and completion artifacts.
- If needed, add the narrowest possible row-spec / label / runner support for `supervised_ffno_depth24` while preserving `supervised_ffno` as the corrected four-block no-refiner authority.
- Launch exactly one fresh supervised FFNO depth-24 no-refiner run under the fixed `lines128` contract.
- Write a contract audit and summary comparing the fresh depth-24 row against the corrected four-block no-refiner supervised baseline by lineage.
- Update the required NeurIPS discoverability surfaces:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/studies/index.md`

### Fixed Contract

- Dataset contract id: `cdi_lines128_seed3`
- `N=128`, `gridsize=1`, synthetic grid-lines, Run1084 fixed probe
- `probe_scale_mode=pad_extrapolate`
- `probe_smoothing_sigma=0.5`
- `set_phi=True`
- `seed=3`
- `nimgs_train=2`, `nimgs_test=2`
- fixed sample ids `0`, `1`
- `40` epochs
- batch size `16`
- Adam `2e-4`
- `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-4)`
- `torch_loss_mode=mae`
- `torch_output_mode=real_imag`
- `fno_modes=12`
- `fno_width=32`
- `fno_cnn_blocks=0`

### Allowed Delta

- corrected baseline `supervised_ffno`: `fno_blocks=4`
- fresh row `supervised_ffno_depth24`: `fno_blocks=24`

## Explicit Non-Goals

- Do not rerun the corrected four-block no-refiner `supervised_ffno` row.
- Do not rerun `pinn_ffno_depth24`; reuse the completed PINN depth-24 summary/root only as prerequisite lineage context for the later final refresh.
- Do not compare against or relabel the historical `fno_cnn_blocks=2` supervised FFNO proxy row as canonical evidence.
- Do not rerun non-FFNO `lines128` rows, PDEBench work, BRDT work, WaveBench work, multi-seed CDI work, natural-patch work, or any `256x256` CDI work.
- Do not refresh manuscript-facing CDI tables, efficiency tables, appendix tables, or figure bundles here. Those promotions belong to `2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh`.
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not mark the item `BLOCKED` for ordinary import/test/path/harness failures. Diagnose, patch narrowly, and rerun first.

## Steering And Roadmap Constraints

- Steering keeps current work inside required paper-evidence lanes and forbids silent fairness drift. This item is valid because it finishes the supervised half of the already-approved FFNO depth-24 CDI follow-up.
- Roadmap authority remains Phase 3 CDI packaging only. This plan must not expand into later roadmap phases or rewrite the roadmap.
- Equal-footing comparison is binding:
  - same dataset identity
  - same probe and probe preprocessing
  - same seed, sample ids, epochs, optimizer, scheduler, loss mode, output mode, and metric schema
  - only `fno_blocks` may differ between the corrected supervised baseline and the fresh supervised depth-24 row
- Historical `fno_cnn_blocks=2` FFNO rows remain proxy lineage only.
- Long-running commands stay under implementation ownership until tracked terminal success or recoverable failure handling is complete:
  - launch in `tmux`
  - activate `ptycho311`
  - keep PATH `python`
  - track the exact launched PID
  - do not launch a duplicate writer into the same `--output-root`
  - accept completion only when the tracked PID exits `0` and required artifacts are freshly written

## Prerequisite Status

- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` confirms the initiative is already active and its early setup tranches completed, but it does not enumerate the later Phase 3 CDI backlog items this work depends on. For this item, the completed Phase 3 summaries below are the effective prerequisite authority, including the cheaper four-block refresh wave that must already have landed before any depth-24 FFNO execution starts.
- Required completed four-block refresh wave:
  - summary:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_no_refiner_ffno_table_refresh_summary.md`
  - authoritative root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh/`
- Required corrected supervised depth-4 baseline:
  - summary:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_supervised_ffno_no_refiner_rerun_summary.md`
  - authoritative root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/runs/supervised_ffno_no_refiner_20260506T232535Z`
- Required completed PINN depth-24 companion context:
  - summary:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_ablation_summary.md`
  - authoritative root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/runs/ffno_depth24_20260507T052301Z`
- Fixed contract lineage still comes from the completed `lines128` CDI benchmark authorities and must be reused, not rerun.

## Implementation Architecture

### 1. Distinct Supervised Depth-24 Row Surface

- Preserve `supervised_ffno` as the corrected four-block no-refiner row.
- Add or expose a separate `supervised_ffno_depth24` surface with:
  - architecture `ffno`
  - training procedure `supervised`
  - label such as `FFNO-24 + supervised`
  - overrides `fno_blocks=24`, `fno_cnn_blocks=0`
- Keep the change centered in the compare-wrapper row-spec registry unless the runner also needs a label, preferred visual order, or artifact-routing update.

### 2. One Controlled Long Run Plus Depth-Only Audit

- Launch exactly one new supervised depth-24 row into an item-local root.
- Reuse the corrected supervised depth-4 row by lineage and audit that only `fno_blocks` changed.
- If the run fails, diagnose the narrow cause, fix it, and rerun once before considering the failure unrecoverable.

### 3. Durable Discoverability Without Promotion

- Publish a summary that states the claim boundary, lineage roots, locked contract, depth-only comparison, and residual caveats.
- Update evidence/index surfaces so downstream paper-refresh work can discover both supervised depth-4 and supervised depth-24 no-refiner rows without mistaking this item for the final paper-promotion authority.

## File And Artifact Targets

### Mandatory Contract Outputs

- Fresh execution authority:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/execution_plan.md`
- Durable summary:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_supervised_ffno_depth24_no_refiner_summary.md`
- Required discoverability updates:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/studies/index.md`
- Item-local artifact root:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/`
- Required wrapper invocation directory:
  - `runs/supervised_ffno_depth24_<timestamp>/invocation.json`
- Required stable row-local contents under the item-local artifact root:
  - `runs/supervised_ffno_depth24/{invocation.json,config.json,history.json,metrics.json,model.pt,exit_code_proof.json,launcher_completion.json,stdout.log,stderr.log}`
  - `recons/supervised_ffno_depth24/recon.npz`
  - `visuals/amp_phase_supervised_ffno_depth24.png`
  - `visuals/amp_phase_error_supervised_ffno_depth24.png`
- Required audit artifacts:
  - `verification/contract_audit_supervised_depth24_vs_depth4.md` or equivalent contract diff vs corrected supervised depth-4 lineage
  - `verification/comparison_supervised_depth24_vs_depth4.json`
  - `verification/comparison_supervised_depth24_vs_depth4.csv`
  - optional note or manifest linking to the completed PINN depth-24 companion for later final-refresh discoverability

### Likely Code And Test Targets

- Likely primary edit surface:
  - `scripts/studies/grid_lines_compare_wrapper.py`
- Likely regression coverage:
  - `tests/test_grid_lines_compare_wrapper.py`
- Conditional only if row label routing, visual ordering, or saved paper-row payloads need an update:
  - `scripts/studies/grid_lines_torch_runner.py`
  - `tests/torch/test_grid_lines_torch_runner.py`
- Conditional only if fresh row completion/provenance reveals a new shared helper gap:
  - `scripts/studies/paper_provenance.py`
  - `tests/studies/test_paper_provenance.py`
- Check surfaces, not planned edit surfaces:
  - `ptycho_torch/generators/ffno.py`
  - `scripts/studies/grid_lines_torch_runner.py`
  - `scripts/studies/grid_lines_compare_wrapper.py`

### Preferred Packaging

- Preferred wrapper invocation directory:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/runs/supervised_ffno_depth24_<timestamp>/`
- Preferred row-local directory naming:
  - `runs/supervised_ffno_depth24/`
  - `recons/supervised_ffno_depth24/`
- Preferred row-local visual names:
  - `visuals/amp_phase_supervised_ffno_depth24.png`
  - `visuals/amp_phase_error_supervised_ffno_depth24.png`
- Preferred verification directory:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/verification/`
- Preferred derived comparison payload names:
  - `comparison_supervised_depth24_vs_depth4.json`
  - `comparison_supervised_depth24_vs_depth4.csv`

## Task Checklist

### Task 1: Freeze Authorities And Run The Input Presence Gate

**Files:**
- Read/confirm:
  - `docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md`
  - `docs/backlog/done/2026-05-06-cdi-lines128-ffno-depth24-ablation.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_ablation_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_no_refiner_ffno_table_refresh_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_supervised_ffno_no_refiner_rerun_summary.md`

- [ ] Run the required presence gate from the selected backlog item before editing or launching anything expensive:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md"),
    Path("docs/backlog/done/2026-05-06-cdi-lines128-ffno-depth24-ablation.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_ablation_summary.md"),
    Path("ptycho_torch/generators/ffno.py"),
    Path("scripts/studies/grid_lines_compare_wrapper.py"),
    Path("scripts/studies/grid_lines_torch_runner.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing supervised FFNO depth-24 inputs: {missing}")
print("supervised FFNO depth-24 inputs present")
PY
```

- [ ] Record the completed four-block refresh root, the corrected supervised depth-4 lineage root, and the completed PINN depth-24 lineage root in item-local notes or the summary draft before code changes.
- [ ] Audit whether the current wrapper already supports a distinct supervised depth-24 model id with truthful row-local provenance. If yes, keep that evidence and skip unnecessary production edits. If no, proceed to Task 2 with the smallest possible change set.

**Verification:**
- Blocking:
  - the presence gate above passes
  - the active four-block refresh root is confirmed as already landed paper-local prerequisite evidence for this later depth-24 wave
  - the corrected supervised depth-4 root is confirmed as the only allowed comparison baseline for this item
  - the completed PINN depth-24 summary/root is confirmed as prerequisite companion context, not a rerun target
- Supporting:
  - inspect the corrected supervised depth-4 `invocation.json` or `config.json` and note `fno_blocks=4`, `fno_cnn_blocks=0` for later contract diff use

### Task 2: Add Minimal `supervised_ffno_depth24` Row Plumbing

**Files:**
- Likely modify:
  - `scripts/studies/grid_lines_compare_wrapper.py`
  - `tests/test_grid_lines_compare_wrapper.py`
- Modify only if needed:
  - `scripts/studies/grid_lines_torch_runner.py`
  - `tests/torch/test_grid_lines_torch_runner.py`
  - `scripts/studies/paper_provenance.py`
  - `tests/studies/test_paper_provenance.py`

- [ ] Add or expose a distinct row spec for `supervised_ffno_depth24` that preserves the locked supervised FFNO contract and changes only `fno_blocks=24`.
- [ ] Preserve `supervised_ffno` as the corrected four-block no-refiner row. Do not mutate its default meaning, row id, or label.
- [ ] Ensure invocation/config artifacts for the new row persist:
  - `model_id = supervised_ffno_depth24`
  - `training_procedure = supervised`
  - `fno_blocks = 24`
  - `fno_cnn_blocks = 0`
- [ ] If the runner or payload builder needs help to keep row-local naming or human-readable labels correct, add only the narrow support needed for `supervised_ffno_depth24`.
- [ ] Add focused regressions proving:
  - default `supervised_ffno` remains four blocks
  - `supervised_ffno_depth24` emits the depth-24 override with no refiner
  - row-local artifact naming and completion-finalization work under `supervised_ffno_depth24`

**Verification:**
- Blocking before any long run:
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "supervised_ffno or ffno"`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "supervised_ffno or ffno"`
  - `python -m compileall -q ptycho_torch scripts/studies`
- Supporting:
  - a focused dry-run, preflight-only, or manifest inspection showing `supervised_ffno_depth24` resolves to `fno_blocks=24` and `fno_cnn_blocks=0`

### Task 3: Launch Exactly One Fresh Supervised Depth-24 No-Refiner Run

**Files / outputs:**
- Fresh item root:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/`
- Wrapper invocation directory:
  - `runs/supervised_ffno_depth24_<timestamp>/`
- Stable row-local output directories:
  - `runs/supervised_ffno_depth24/`
  - `recons/supervised_ffno_depth24/`
  - `visuals/`

- [ ] Launch exactly one supervised depth-24 row under the item-local root using the existing compare-wrapper/runner path.
- [ ] Reuse the corrected supervised depth-4 no-refiner row by lineage; do not rerun it.
- [ ] Run in `tmux`, activate `ptycho311`, track the exact launched PID, and wait on that PID instead of polling broad process names.
- [ ] Keep implementation ownership until the launcher exits `0` and all required row-local outputs are freshly written.
- [ ] If the first run fails, diagnose the narrow cause, fix it, and rerun once before treating the item as unrecoverable.

**Verification:**
- Blocking:
  - tracked PID exits `0`
  - the wrapper invocation directory exists and the item-local root contains the required stable row-local artifacts listed in this plan
  - the fresh row records `fno_blocks=24` and `fno_cnn_blocks=0` in invocation/config artifacts
- Supporting:
  - tmux live log
  - training history sanity and loss-curve inspection

### Task 4: Audit Same-Contract Fairness And Depth-Only Delta

**Files / outputs:**
- Fresh audit artifacts under `verification/`
- Comparison payloads under `verification/`

- [ ] Write a contract audit or contract diff comparing the fresh supervised depth-24 row against the corrected supervised depth-4 lineage and show that the only allowed contract change is `fno_blocks: 4 -> 24`.
- [ ] Compare the fresh supervised depth-24 metrics only against the corrected supervised depth-4 no-refiner row for the canonical result.
- [ ] Keep any mention of the completed PINN depth-24 row explicitly secondary and only for later final-refresh discoverability.
- [ ] If any non-allowed field drift appears, treat it as a correctness failure and fix/rerun rather than accepting an unequal-footing comparison.

**Verification:**
- Blocking:
  - contract audit shows no non-allowed drift
  - machine-readable comparison payload exists for supervised depth-24 vs corrected supervised depth-4
  - historical `fno_cnn_blocks=2` proxy rows are not used as the canonical comparator
- Supporting:
  - parameter-count delta
  - runtime delta
  - optional context note tying this item to the completed PINN depth-24 companion

### Task 5: Write The Durable Summary And Refresh Discoverability

**Files:**
- Create:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_supervised_ffno_depth24_no_refiner_summary.md`
- Modify:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/studies/index.md`

- [ ] Write the summary with:
  - fixed contract
  - corrected supervised depth-4 lineage root
  - fresh supervised depth-24 root
  - completed PINN depth-24 prerequisite context
  - contract-audit outcome
  - key metric, parameter-count, and runtime deltas
  - explicit claim boundary that this is supervised depth-24 companion evidence only
  - explicit note that final paper-table promotion remains owned by `2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh`
- [ ] Update `evidence_matrix.md` so the CDI depth-ablation / objective-control surfaces can discover the new supervised depth-24 no-refiner row without implying it already replaced paper-local tables.
- [ ] Update `model_variant_index.json` with a new `supervised_ffno_depth24` row entry under `cdi_lines128_seed3`.
- [ ] Update `ablation_index.json` so the depth-24 family records both the existing PINN depth-24 evidence and this supervised companion without confusing either with the final paper refresh.
- [ ] Update `docs/studies/index.md` with the new supervised depth-24 study or extension entry, including plan path, summary authority, item root, and claim boundary.
- [ ] Do not expand this task into `docs/index.md` or `paper_evidence_index.md` updates unless implementation introduces a genuinely new top-level reusable study surface that those indexes must discover.

**Verification:**
- Blocking:
  - summary exists at the required path and names the corrected supervised depth-4 lineage, fresh supervised depth-24 root, and deferred final-refresh owner
  - `python -m compileall -q ptycho_torch scripts/studies` has already passed from Task 2
  - `python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json > /dev/null`
  - `python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json > /dev/null`
- Supporting:
  - a concise manual diff review of `docs/studies/index.md` and `evidence_matrix.md` for wording/lineage correctness

## Required Deterministic Checks

Run these as required gates for this item unless a stronger replacement is explicitly recorded in the implementation report. The listed pytest selectors and compile gate are authoritative and should normally be kept exactly as written.

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md"),
    Path("docs/backlog/done/2026-05-06-cdi-lines128-ffno-depth24-ablation.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_ablation_summary.md"),
    Path("ptycho_torch/generators/ffno.py"),
    Path("scripts/studies/grid_lines_compare_wrapper.py"),
    Path("scripts/studies/grid_lines_torch_runner.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing supervised FFNO depth-24 inputs: {missing}")
print("supervised FFNO depth-24 inputs present")
PY
pytest -q tests/torch/test_grid_lines_torch_runner.py -k "supervised_ffno or ffno"
pytest -q tests/test_grid_lines_compare_wrapper.py -k "supervised_ffno or ffno"
python -m compileall -q ptycho_torch scripts/studies
```

Blocking interpretation:
- the presence gate must pass before expensive execution
- both pytest selectors and the compile gate must pass before the long run

Supporting checks allowed in addition to the required gates:
- preflight-only wrapper invocation or manifest inspection for `supervised_ffno_depth24`
- manual inspection of saved `invocation.json`, `config.json`, and `launcher_completion.json`
- any narrow helper test added only if a shared provenance helper changes

## Completion Criteria

- A fresh `supervised_ffno_depth24` row completes under the locked `lines128` no-refiner supervised contract with `fno_blocks=24` and `fno_cnn_blocks=0`.
- The canonical comparison for this item is supervised depth-24 vs corrected supervised depth-4 by lineage, with no historical local-refiner proxy used as canonical evidence.
- The durable summary and required discoverability surfaces are updated and point downstream paper-refresh work at this new supervised companion row without prematurely promoting it into paper tables.
