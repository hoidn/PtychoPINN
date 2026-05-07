# CDI Lines128 Supervised FFNO No-Refiner Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create worktrees.

**Goal:** Produce the corrected `lines128` CDI `supervised_ffno` row with `fno_cnn_blocks=0`, then refresh the two-row FFNO objective-control evidence so both training procedures use the same pure-FFNO no-refiner contract.

**Architecture:** Reuse the locked `lines128` CDI contract, the completed supervised-FFNO extension lineage, and the completed corrected `pinn_ffno` no-refiner rerun. Launch only the missing `supervised_ffno` no-refiner row under the existing grid-lines Torch compare/runner stack, prove by audit that only `fno_cnn_blocks` changed versus the historical supervised proxy row, and collate a narrow objective-control bundle plus durable summary and index refreshes without rewriting the immutable six-row CDI authority.

**Tech Stack:** PATH `python`, `ptycho311` for long-running launches, PyTorch/Lightning, `scripts/studies/grid_lines_compare_wrapper.py`, `scripts/studies/grid_lines_torch_runner.py`, optional narrow `lines128` study collation helpers, Markdown/JSON/CSV/TeX evidence surfaces.

---

## Selected Objective

- Rerun `supervised_ffno` on the fixed `lines128` CDI contract with `fno_cnn_blocks=0`.
- Compare it only against the completed corrected no-refiner `pinn_ffno` row from `2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun`.
- Preserve the completed supervised extension and the historical FFNO-local-refiner proxy rows as provenance-bearing historical context only.

## Scope Boundaries

### In Scope

- A narrow audit of the existing supervised FFNO launch path to confirm whether it already supports `fno_cnn_blocks=0` with truthful provenance and row-local completion artifacts.
- Any minimal code/test changes needed to make that existing path emit a correct no-refiner supervised row and objective-control collation without changing the frozen `lines128` contract.
- One fresh supervised FFNO no-refiner run under:
  - the locked `lines128` contract
  - the historical supervised-extension training procedure
  - the corrected no-refiner FFNO architecture setting
- A contract/provenance audit proving the new row differs from the historical supervised extension only in `fno_cnn_blocks`.
- A refreshed two-row FFNO objective-control bundle and durable discoverability updates.

### Explicit Non-Goals

- Do not rerun `pinn_ffno`; reuse the completed corrected no-refiner root by lineage.
- Do not rerun CNN, Hybrid ResNet, spectral-bottleneck, FNO vanilla, U-NO, SRU-Net, natural-patch, PDEBench, BRDT, WaveBench, or `256x256` work.
- Do not rewrite the authoritative six-row `lines128` CDI root or silently promote the corrected no-refiner FFNO rows into that canonical table in this item.
- Do not change dataset identity, split, probe source, probe preprocessing, `seed=3`, `40` epochs, scheduler, loss mode, output mode, metric schema, fixed sample ids, or shared visual-scale policy.
- Do not treat normal import/test/path/harness failures as automatic `BLOCKED`. Diagnose, apply a narrow fix, and rerun first. Reserve `BLOCKED` for missing prerequisite artifacts, unavailable hardware/resources, roadmap conflict, external dependency outside current authority, required user decision, or a failure that remains unrecoverable after a documented narrow fix attempt.

## Steering, Roadmap, And Policy Constraints

- Steering keeps the active window on core paper evidence and forbids optional expansion while required evidence remains incomplete. This item is justified because it repairs a manuscript-facing objective-control row already referenced by current CDI packaging.
- Roadmap authority remains Phase 3 CDI packaging only:
  - `lines128` stays the CDI headline contract.
  - pure-FFNO claims must use no-refiner rows.
  - historical `fno_cnn_blocks=2` FFNO rows remain caveated proxy evidence only.
- Fairness is binding:
  - both compared FFNO rows must use the same `lines128` dataset/split/probe/seed/epoch/loss/scheduler/metric contract
  - the only intended contract change relative to the historical supervised extension is `fno_cnn_blocks: 2 -> 0`
- Long-running execution stays under implementation ownership until terminal success or recoverable failure handling is complete:
  - launch in `tmux`
  - activate `ptycho311`
  - keep PATH `python`
  - track the exact launched PID
  - accept completion only when the tracked PID exits `0` and required artifacts are freshly written
- PyTorch policy and legacy bridge rules still apply if code changes are needed: use PATH `python`, and if any legacy-config path is touched ensure `update_legacy_dict(params.cfg, config)` still happens before legacy-dependent module usage.

## Prerequisite Status

- Satisfied prerequisite:
  - `2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun`
  - authoritative summary:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_no_refiner_row_rerun_summary.md`
  - corrected row root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/runs/ffno_no_refiner_20260506T223454Z`
- Reused fixed-contract authorities:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_supervised_equivalent_rows_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_preflight.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md`
- Important ledger note:
  - `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` records early Phases 0-2 only and does not enumerate the completed later Phase 3 CDI items this backlog task depends on.
  - For this item, the selected backlog context plus the completed Phase 3 summaries above are the effective prerequisite authority.

## Implementation Architecture

- **Unit 1: Fixed-contract supervised rerun path**
  - Confirm the existing supervised FFNO route can express `fno_cnn_blocks=0` while preserving the rest of the historical supervised-extension contract.
  - If it cannot, apply the smallest possible runner/wrapper/collator fix and cover it with focused tests before any long run.

- **Unit 2: Row-level contract and provenance audit**
  - Prove the fresh supervised row uses zero local refiners in both instantiated config and saved execution artifacts.
  - Audit against the historical supervised extension so only `fno_cnn_blocks` changes while dataset/probe/training/metric identity stays fixed.

- **Unit 3: Objective-control and discoverability refresh**
  - Build a narrow two-row FFNO objective-control output using corrected no-refiner `pinn_ffno` plus fresh no-refiner `supervised_ffno`.
  - Update summary and evidence indexes so downstream manuscript work discovers the corrected pair without mistaking the historical proxy rows for canonical pure-FFNO evidence.

## File And Artifact Targets

### Mandatory contract outputs

- Fresh item artifact root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/`
- Fresh supervised run root under that item, containing at minimum:
  - row-local `invocation.json`
  - `config.json`
  - `history.json`
  - `metrics.json`
  - `model.pt`
  - `exit_code_proof.json`
  - `launcher_completion.json`
  - `stdout.log`
  - `stderr.log`
  - `recons/supervised_ffno/recon.npz`
  - row-local visuals
- Row-level audits under the item root:
  - contract diff versus the historical supervised extension
  - no-refiner inspection proving zero executed refiner keys
  - objective-control comparison audit or equivalent merged manifest
- Durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_supervised_ffno_no_refiner_rerun_summary.md`
- Required discoverability refreshes:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`

### Likely code / test surfaces if the existing path needs a narrow fix

- `scripts/studies/grid_lines_compare_wrapper.py`
- `scripts/studies/grid_lines_torch_runner.py`
- `scripts/studies/lines128_paper_benchmark.py` or a narrow sibling helper under `scripts/studies/`
- `tests/torch/test_grid_lines_torch_runner.py`
- `tests/test_grid_lines_compare_wrapper.py`
- optionally a focused study-level selector such as `tests/studies/test_lines128_paper_benchmark.py` if collation or lineage handling changes

### Preferred packaging after core completion

- Refresh the current manuscript-facing CDI objective-control/table surfaces if and only if the new corrected pair is complete and the refresh can preserve clear lineage:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_objective_comparison.tex`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.csv`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.tex`
- Keep broader manuscript/config-appendix refreshes out of scope unless they are purely mechanical fallout of the corrected objective-control pair.

## Execution Checklist

### Task 1: Freeze Authorities And Audit The Existing Launch Path

- [ ] Confirm the fixed `lines128` contract from the complete-table and supervised-extension authorities:
  - contract id `cdi_lines128_seed3`
  - `seed=3`
  - `40` epochs
  - fixed sample ids `0`, `1`
  - current probe source and preprocessing
  - current metric schema
- [ ] Confirm the corrected no-refiner `pinn_ffno` prerequisite root exists and is the only allowed comparator for this item.
- [ ] Inspect the existing supervised FFNO route and determine whether it already supports a truthful `fno_cnn_blocks=0` rerun with row-local completion/provenance artifacts.
- [ ] If the current route is already sufficient, record that no production code edit is required and proceed directly to the deterministic preflight gates.
- [ ] If the current route is insufficient, write down the smallest required fix surface before editing anything.

Verification for Task 1:

- Blocking before any long run:
  - verify the prerequisite corrected `pinn_ffno` summary/root and the historical supervised-extension root are both present and readable
- Supporting:
  - any local dry-run or manifest inspection showing the planned `supervised_ffno` row spec resolves to `fno_cnn_blocks=0`

### Task 2: Apply Only The Minimal Runner/Wrapper/Test Fixes Needed

- [ ] Only if Task 1 found a real gap, implement the smallest viable change so `supervised_ffno` can rerun with `fno_cnn_blocks=0` and still emit truthful row-local provenance and completion artifacts.
- [ ] Keep the fix narrowly scoped to the supervised FFNO no-refiner path or generic provenance/collation correctness it depends on.
- [ ] Do not change the row id, dataset generation logic, paper metric schema, or canonical six-row authority during this fix.
- [ ] Add or update focused tests that prove:
  - `supervised_ffno` routes the intended no-refiner FFNO configuration
  - existing FFNO/PINN behavior is not regressed
  - row-local recovery/collation still fails closed if required completion artifacts are missing

Verification for Task 2:

- Blocking before any long run:
  - `python - <<'PY'\nfrom ptycho_torch.generators.ffno import FfnoGeneratorModule\nmodel = FfnoGeneratorModule(cnn_blocks=0)\nassert len(model.refiners) == 0\nprint(\"CDI supervised FFNO no-refiner generator instantiates\")\nPY`
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "supervised_ffno or ffno"`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "supervised_ffno or ffno"`
  - `python -m compileall -q ptycho_torch scripts/studies`
- Supporting:
  - any focused study-level collation/lineage selector added by the narrow fix

### Task 3: Launch The Fresh Supervised FFNO No-Refiner Row

- [ ] Launch exactly one fresh supervised rerun under the item-local artifact root using the existing compare-wrapper/runner path and `fno_cnn_blocks=0`.
- [ ] Reuse the corrected no-refiner `pinn_ffno` comparator by lineage; do not rerun it.
- [ ] Run in `tmux`, activate `ptycho311`, track the exact launched PID, and wait for that PID rather than polling broad process names.
- [ ] Keep implementation ownership until the launcher exits `0` and all required row-local outputs are freshly written.
- [ ] If the first long run fails, diagnose the narrow cause, fix it, and rerun or resume once before considering the item unrecoverable.

Verification for Task 3:

- Blocking:
  - tracked PID exits `0`
  - item-local run root contains the required row-local artifacts listed in this plan
  - the fresh supervised row’s config/invocation records `fno_cnn_blocks=0`
- Supporting:
  - tmux live log
  - training history/loss curve sanity

### Task 4: Audit No-Refiner Purity And Same-Contract Fairness

- [ ] Write a contract-diff artifact comparing the fresh supervised no-refiner row against the historical supervised-extension root and show that the only allowed contract change is `fno_cnn_blocks: 2 -> 0`.
- [ ] Write a no-refiner inspection artifact proving the executed supervised row has zero local refiner keys in saved execution artifacts.
- [ ] Compare the fresh supervised row against the corrected no-refiner `pinn_ffno` row only; do not mix in the historical proxy row except as caveated context.
- [ ] If any non-allowed field drift appears, treat it as a correctness failure and fix/rerun rather than accepting an unequal-footing comparison.

Verification for Task 4:

- Blocking:
  - contract diff shows no non-allowed drift
  - no-refiner inspection proves zero executed refiner keys
  - objective-control audit confirms both compared FFNO rows use `fno_cnn_blocks=0`
- Supporting:
  - parameter-count delta and metric delta summaries versus the historical proxy for context only

### Task 5: Refresh Objective-Control Outputs Without Rewriting The Base Table

- [ ] Build a narrow corrected objective-control output rooted in:
  - corrected no-refiner `pinn_ffno`
  - fresh no-refiner `supervised_ffno`
- [ ] Preserve the historical local-refiner FFNO objective-control extension as lineage/proxy context only.
- [ ] Refresh current manuscript-facing objective-control table assets if they can be updated mechanically from the corrected pair while preserving explicit lineage and caveat labels.
- [ ] Do not relabel the immutable six-row CDI authority as pure-FFNO corrected evidence in this item; that belongs to the separate table-refresh scope.

Verification for Task 5:

- Blocking:
  - corrected two-row objective-control output exists and uses only no-refiner FFNO rows
  - any refreshed table asset clearly points to the corrected pair rather than the historical local-refiner proxy
- Supporting:
  - regenerated TeX/CSV/JSON objective-control payload diffs

### Task 6: Write The Durable Summary And Update Discoverability

- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_supervised_ffno_no_refiner_rerun_summary.md` with:
  - fixed contract
  - prerequisite corrected `pinn_ffno` lineage
  - fresh supervised no-refiner root
  - contract/no-refiner audit outcome
  - key metric and parameter deltas
  - explicit claim boundary and residual caveats
- [ ] Update `evidence_matrix.md` so the CDI lines128 matrix no longer implies the active supervised FFNO row is still the local-refiner proxy.
- [ ] Update `model_variant_index.json`, `ablation_index.json`, and `paper_evidence_index.md` so downstream planning/manuscript tasks discover the corrected pair and its claim boundary.
- [ ] Only mention historical `fno_cnn_blocks=2` FFNO rows as preserved proxy lineage, not canonical pure-FFNO evidence.

Verification for Task 6:

- Blocking:
  - summary exists and names the corrected supervised no-refiner root plus the corrected `pinn_ffno` comparator root
  - evidence indexes reference the new summary and distinguish corrected no-refiner evidence from preserved proxy lineage
- Supporting:
  - targeted grep or structured-json spot checks over the touched index/table files

## Deterministic Verification Gate

Run these as the required deterministic checks for this backlog item unless an explicitly stronger replacement is documented in the execution report. Because the item includes a fresh training rerun, these are blocking before the expensive launch:

- `python - <<'PY'
from ptycho_torch.generators.ffno import FfnoGeneratorModule
model = FfnoGeneratorModule(cnn_blocks=0)
assert len(model.refiners) == 0
print("CDI supervised FFNO no-refiner generator instantiates")
PY`
- `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "supervised_ffno or ffno"`
- `pytest -q tests/test_grid_lines_compare_wrapper.py -k "supervised_ffno or ffno"`
- `python -m compileall -q ptycho_torch scripts/studies`

Recommended supporting checks if code changes touch study-level collation or table refresh logic:

- `pytest -q tests/studies/test_lines128_paper_benchmark.py -k "ffno or objective"`
- a narrow table/manifest validation command that proves the corrected objective-control outputs reference only no-refiner FFNO rows

## Completion Standard

This backlog item is complete only when all of the following are true:

- the fresh `supervised_ffno` run records `fno_cnn_blocks=0`
- the compared `pinn_ffno` and `supervised_ffno` objective-control rows are both no-refiner FFNO rows
- the historical local-refiner FFNO rows remain preserved but explicitly caveated as proxy lineage only
- the durable summary and evidence indexes reflect the corrected no-refiner objective-control state
- any long-running launch used for the item reached tracked-PID exit `0` and left the required fresh artifacts in place
