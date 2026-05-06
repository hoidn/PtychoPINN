# CDI Lines128 FFNO No-Refiner Row Rerun Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` (recommended) or `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create worktrees.

**Goal:** Produce a corrected `lines128` CDI `pinn_ffno` row under the locked paper contract with `fno_cnn_blocks=0`, while preserving the historical `fno_cnn_blocks=2` row only as FFNO-local-proxy evidence.

**Architecture:** Reuse the existing Torch/grid-lines compare-wrapper path as the preferred launch surface because it already owns the shared dataset/probe contract, row-local invocation/config/history/metrics outputs, wrapper provenance, split manifests, and completion evidence. Treat the authoritative six-row `lines128` bundle as the fixed contract source and rerun exactly one fresh row into a new append-only artifact root; do not rebuild or relabel the canonical table in this item.

**Tech Stack:** `scripts/studies/grid_lines_compare_wrapper.py`, `scripts/studies/grid_lines_torch_runner.py`, `ptycho_torch/generators/ffno.py`, PyTorch/Lightning in `ptycho311`, tmux for the long run, Markdown/JSON evidence indexes.

---

## Selected Backlog Objective

- Correct the CDI `pinn_ffno` row so the generator is pure FFNO with no `_LocalResidualRefiner` modules.
- Keep every locked `lines128` paper-contract field fixed except `fno_cnn_blocks`, which must change from `2` to `0`.
- Produce a durable rerun summary and lineage/audit artifacts that downstream table-refresh work can consume.

## Scope

- In scope:
  - one fresh `pinn_ffno` rerun only
  - contract reconstruction from the authoritative completed `lines128` bundle
  - no-refiner model instantiation proof
  - row-local audit proving only the local-refiner count changed
  - item-local artifacts, summary, and required evidence-index updates for the new row
- Out of scope:
  - rerunning non-FFNO `lines128` rows
  - changing the locked FNO comparator, seed policy, sample IDs, probe, loss, scheduler, or metric schema
  - promoting the corrected row into the canonical six-row table
  - rewriting historical artifact roots or relabeling them as pure FFNO
  - the supervised FFNO rerun and the later no-refiner table refresh

## Steering And Roadmap Constraints

- This item is Phase 3 CDI correction work and must not expand into later roadmap phases or unrelated Phase 2 PDE work.
- Keep equal-footing comparison boundaries explicit. The corrected row must remain on the same `lines128` contract as the authoritative bundle except for `fno_cnn_blocks`.
- Preserve the roadmap claim boundary: this item creates corrected row-local evidence, not a refreshed manuscript-facing table.
- Preserve fairness constraints. If any contract field other than the allowed FFNO local-refiner count drifts, treat that as an implementation defect and fix or rerun rather than silently accepting a weaker comparison.
- Keep the historical `fno_cnn_blocks=2` row discoverable only as `FFNO-local proxy` lineage. Do not overwrite or mutate the old roots.

## Prerequisite Status

- Progress-ledger status relevant here:
  - `blocked_tranches` is empty at the initiative level.
  - Earlier roadmap phases are already complete enough that this item is not waiting on PDE or benchmark-selection work.
- Completed `lines128` authority already exists:
  - durable summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - authoritative six-row root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- The authoritative bundle already records the blocker that justifies this rerun:
  - `pinn_ffno` there used `fno_cnn_blocks=2`
  - pure-FFNO CDI claims are deferred to this rerun and the later table-refresh item
- Downstream dependency:
  - `2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh` must consume this rerun by lineage and should not be folded into this item.

## Implementation Architecture

- **Unit 1: Contract and launch-surface preflight**
  - Source the exact frozen contract from the authoritative complete-table bundle and its recorded manifests.
  - Confirm the existing compare-wrapper path can launch only `pinn_ffno` while preserving wrapper and row-local provenance.
- **Unit 2: Single-row corrected execution**
  - Launch one append-only wrapper run that emits only `pinn_ffno` with `fno_cnn_blocks=0` into a new item-local root.
  - Keep long-running execution under implementation ownership until the tracked process exits cleanly and the required artifacts are freshly written.
- **Unit 3: Row audit and durable publication**
  - Compare the fresh row against the historical proxy root and the completed-table contract.
  - Publish a durable summary plus the required evidence-index updates without changing the canonical table authority.

## File And Artifact Targets

Mandatory contract outputs:

- Fresh plan authority:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/execution_plan.md`
- Item-local execution root:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/`
- Fresh wrapper root under the item-local execution root, including at minimum:
  - `invocation.json`
  - `invocation.sh`
  - `dataset_identity_manifest.json`
  - `split_manifest.json`
  - `metrics.json`
  - `metrics_by_model.json`
  - `model_manifest.json`
  - `paper_benchmark_manifest.json` if emitted by the chosen wrapper path
  - `runs/pinn_ffno/{invocation.json,config.json,history.json,metrics.json,model.pt,exit_code_proof.json,launcher_completion.json,stdout.log,stderr.log}`
  - `recons/pinn_ffno/recon.npz`
  - `visuals/amp_phase_pinn_ffno.png`
  - `visuals/amp_phase_error_pinn_ffno.png`
- Row-audit artifacts under the item-local execution root:
  - a machine-readable contract diff proving only `fno_cnn_blocks` changed
  - a no-refiner inspection artifact proving zero local-refiner modules
  - verification logs for deterministic checks
- Durable summary:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_no_refiner_row_rerun_summary.md`

Mandatory durable index updates before completion:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`

Claim-boundary authority to reuse for every manifest/index surface touched by this item:

- `lines128_ffno_vs_hybrid_prerequisite_pair`
  - Reason: this rerun is still prerequisite CDI decision-support evidence for pure-FFNO manuscript/table use and for the downstream table-refresh item; it must not reuse `complete_lines128_cdi_benchmark`, and implementation must not invent an unregistered claim-boundary token during execution of this item.

Conditionally editable code/test surfaces if preflight proves current tooling does not preserve `fno_cnn_blocks=0` correctly:

- `ptycho_torch/generators/ffno.py`
- `scripts/studies/grid_lines_torch_runner.py`
- `scripts/studies/grid_lines_compare_wrapper.py`
- `tests/torch/test_grid_lines_torch_runner.py`
- `tests/test_grid_lines_compare_wrapper.py`

No `docs/index.md` update is required unless implementation adds a new reusable runbook or authority document beyond the summary above.

## Execution Checklist

### Tranche 1: Freeze Contract And Preflight The Existing Launch Path

- [ ] Read the authoritative contract from:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/{invocation.json,metrics.json,model_manifest.json,paper_benchmark_manifest.json,runs/pinn_ffno/invocation.json,runs/pinn_ffno/config.json}`
- [ ] Record the exact fixed contract fields that must remain unchanged:
  - `N=128`, `gridsize=1`, synthetic grid-lines, `set_phi=True`
  - Run1084 fixed probe and the same probe preprocessing path
  - `seed=3`
  - fixed sample IDs `0` and `1`
  - `nimgs_train=2`, `nimgs_test=2`, `nphotons=1e9`
  - `40` epochs, batch `16`, Adam `2e-4`
  - `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-4)`
  - `torch_loss_mode=mae`, `torch_output_mode=real_imag`
  - `fno_modes=12`, `fno_width=32`, `fno_blocks=4`
- [ ] Record the single allowed contract change:
  - `fno_cnn_blocks: 2 -> 0`
- [ ] Choose the preferred launch surface:
  - use `scripts/studies/grid_lines_compare_wrapper.py`
  - launch only `--models pinn_ffno`
  - keep `--architectures ffno` and pass the fixed contract flags explicitly
  - record the exact claim boundary `lines128_ffno_vs_hybrid_prerequisite_pair` on the item-local manifest/summary/index surfaces this item writes; do not invent a new token and do not relabel the rerun as `complete_lines128_cdi_benchmark`
- [ ] Reject `scripts/studies/lines128_paper_benchmark.py` as the primary execution surface for this item because its modes assemble or promote bundles rather than perform a single isolated rerun.
- [ ] Preflight output-root hygiene:
  - choose a new append-only item-local wrapper root
  - verify no active writer is already targeting that root
  - if an old partial root exists, inspect whether it is resumable before creating another root

### Tranche 2: Verify Or Repair No-Refiner Plumbing Before The Expensive Run

- [ ] Run the required model-instantiation proof with `cnn_blocks=0` and capture the result into the item-local verification area.
- [ ] Run the required targeted selectors and compile gate before any expensive rerun.
- [ ] Inspect whether the current runner/wrapper path already records `fno_cnn_blocks=0` into both wrapper-level and row-level invocation/config artifacts.
- [ ] If the preflight checks pass and the launch path preserves zero refiners correctly, skip production code edits.
- [ ] If the preflight checks fail because `fno_cnn_blocks=0` is dropped, rewritten, or still produces `_LocalResidualRefiner` modules, patch only the minimal owner files listed in `Conditionally editable code/test surfaces`, add or tighten targeted tests, rerun the same required selectors, and only then proceed to the long run.
- [ ] Do not mark the item `BLOCKED` for an ordinary test/import/path/harness failure. Diagnose, narrow-fix, and rerun first.

### Tranche 3: Execute The Single Corrected FFNO Row

- [ ] Launch exactly one compare-wrapper run for `pinn_ffno` into the new item-local root with the frozen `lines128` contract and `--fno-cnn-blocks 0`.
- [ ] Run the long command in tmux from the repo root after activating `ptycho311`.
- [ ] Follow the repo long-run guardrail exactly:
  - track the exact launched PID from the shell that starts the Python command
  - wait on that PID directly instead of polling broad process patterns
  - treat the run as complete only if the tracked PID exits with code `0` and the required wrapper/row artifacts are freshly written
  - do not launch a duplicate run against the same output root
- [ ] Preserve the historical proxy roots read-only:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`
  - the `pinn_ffno` row inside the authoritative six-row bundle
- [ ] If the long run fails after launch, attempt one documented narrow recovery on the same authority surface before considering the item unrecoverable.

### Tranche 4: Audit The Fresh Row Against The Frozen Contract

- [ ] Produce a machine-readable audit that compares:
  - the fresh row root
  - the historical proxy root
  - the authoritative complete-table contract
- [ ] The audit must prove:
  - `fno_cnn_blocks=0` is recorded in the fresh wrapper invocation, row invocation, and row config
  - the instantiated model has `len(model.refiners) == 0`
  - no `_LocalResidualRefiner` modules remain in the executed model path
  - the fresh row emitted the required checkpoint artifact at `runs/pinn_ffno/model.pt`
  - all other locked contract fields match the authoritative `lines128` contract
  - the old row remains labeled historical proxy evidence and is not overwritten
- [ ] Record the fresh metric values, parameter count, runtime/provenance facts, and a direct comparison against the historical `fno_cnn_blocks=2` proxy row.
- [ ] Treat any unexpected drift in non-allowed contract fields as a blocking defect for completion and rerun after fixing the cause.

### Tranche 5: Publish The Durable Summary And Update Discovery Surfaces

- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_no_refiner_row_rerun_summary.md`.
- [ ] The summary must state:
  - the item objective and why the rerun was required
  - the authoritative historical proxy roots that remain preserved
  - the fresh corrected row root
  - proof that the new row is pure FFNO with `fno_cnn_blocks=0`
  - a comparison of corrected metrics versus the historical `fno_cnn_blocks=2` proxy row
  - the preserved claim boundary: `lines128_ffno_vs_hybrid_prerequisite_pair`; this item is row-level corrected prerequisite evidence only, and canonical table promotion remains deferred to `2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh`
- [ ] Update `evidence_matrix.md`, `model_variant_index.json`, `ablation_index.json`, and `paper_evidence_index.md` so later planning and manuscript tasks can discover the corrected row by lineage.
- [ ] If implementation determines any one of those discovery surfaces is genuinely non-applicable despite the roadmap policy, record the exact non-applicability reason in the execution report and summary instead of silently skipping the update.
- [ ] Do not change the authoritative six-row bundle status or relabel the old FFNO row as canonical pure FFNO in this item.

## Verification

Blocking checks before any expensive rerun:

- [ ] Required deterministic model-construction proof:

```bash
python - <<'PY'
from ptycho_torch.generators.ffno import FfnoGeneratorModule
model = FfnoGeneratorModule(cnn_blocks=0)
assert len(model.refiners) == 0
print("CDI FFNO no-refiner generator instantiates")
PY
```

- [ ] Required deterministic runner selector:

```bash
pytest -q tests/torch/test_grid_lines_torch_runner.py -k "ffno"
```

- [ ] Required deterministic wrapper selector:

```bash
pytest -q tests/test_grid_lines_compare_wrapper.py -k "ffno"
```

- [ ] Required deterministic syntax gate:

```bash
python -m compileall -q ptycho_torch scripts/studies
```

Blocking checks after any code edits but before the long rerun:

- [ ] Rerun every blocking deterministic check above.
- [ ] If targeted tests were added or tightened, run those exact selectors too and archive their logs in the item-local verification directory.

Blocking checks after the long rerun:

- [ ] Confirm the tracked PID exited `0`.
- [ ] Confirm the fresh wrapper root contains the mandatory wrapper-level artifacts listed in `File And Artifact Targets`.
- [ ] Confirm the fresh `runs/pinn_ffno/` row root contains:
  - `invocation.json`
  - `config.json`
  - `history.json`
  - `metrics.json`
  - `model.pt`
  - `exit_code_proof.json`
  - `launcher_completion.json`
  - `stdout.log`
  - `stderr.log`
- [ ] Confirm `recons/pinn_ffno/recon.npz` and the required row visuals are freshly written.
- [ ] Run the contract-audit check and fail completion if:
  - `fno_cnn_blocks` is missing or nonzero anywhere it must be recorded
  - any `_LocalResidualRefiner` modules remain
  - any non-allowed locked contract field changed

Supporting checks:

- [ ] JSON parse smoke for any edited durable JSON index files.
- [ ] Summary/index grep that confirms the historical row still carries proxy wording and the fresh row is the only pure-FFNO lineage.
- [ ] If a paper-facing comparison artifact is emitted early, verify it does not claim the full six-row table was refreshed in this item.

## Failure Handling

- Treat normal verification failures as implementation work, not automatic blockage.
- Reserve `BLOCKED` for:
  - missing required hardware or environment that cannot be recovered locally
  - unavailable prerequisite source artifacts needed to reconstruct the contract
  - roadmap or authority conflict outside this item’s scope
  - an unrecoverable failure that remains after one documented narrow fix attempt
- If the run cannot be completed, leave a durable note describing:
  - the exact command and output root attempted
  - tracked PID and exit code if available
  - which required artifacts were or were not produced
  - what narrow recovery was attempted
  - why the remaining blocker exceeds this item’s authority
