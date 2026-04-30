# Lines128 Supervised FFNO Control Row Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Keep this file as the execution authority for the selected backlog item.

**Goal:** Add the required same-contract `FFNO + supervised` control row for the locked `lines128` CDI benchmark, then publish an adjacent extension bundle and durable summary that make training-procedure differences explicit without rewriting the primary six-row CDI claim.

**Architecture:** Treat the `2026-04-30` `paper_complete` six-row `lines128` bundle as frozen prerequisite evidence, reuse its dataset/split/probe/sample/metric/visual contract unchanged, and create one new supervised-extension root for the `FFNO + supervised` work plus derived comparison artifacts. Keep `scripts/studies/grid_lines_torch_runner.py` as the row-local execution owner, keep shared dataset/provenance/collation in `scripts/studies/grid_lines_compare_wrapper.py` or a thin adjacent `lines128` harness mode, and record a precise `not_protocol_compatible` outcome instead of weakening the contract if supervised FFNO still cannot run after one narrow compatibility-fix cycle.

**Tech Stack:** PATH `python`, tmux with `ptycho311` for long-running commands, PyTorch/Lightning, `scripts/studies/lines128_paper_benchmark.py`, `scripts/studies/grid_lines_compare_wrapper.py`, `scripts/studies/grid_lines_torch_runner.py`, optional narrow `ptycho_torch/workflows/components.py` supervised-path plumbing, pytest, `compileall`, Markdown/JSON/CSV/TeX/NPZ artifacts.

---

## Selected Backlog Objective

- Implement backlog item `2026-04-29-cdi-lines128-supervised-equivalent-rows`.
- Add one required `FFNO + supervised` row under the same frozen `lines128` contract already used by:
  - the authoritative minimum-subset `CDI CNN + supervised` row
  - the authoritative complete-table `FFNO + PINN` row
- Emit table-ready machine-readable fragments, source reconstruction arrays, fixed-sample amplitude/phase and error panels, provenance manifests, and a durable summary under `docs/plans/NEURIPS-HYBRID-RESNET-2026/`.
- Keep the resulting supervised-equivalent package separate from the current primary six-row CDI claim unless every row included in the extension is same-contract and provenance-complete.

## Scope And Explicit Non-Goals

In scope:

- Reuse the locked `lines128` contract, fixed sample ids, and shared visual-scale policy from the already accepted `2026-04-30` complete-table bundle.
- Reuse the accepted `CDI CNN + supervised` row from the minimum-subset root as existing same-contract supervised reference evidence. Do not retrain it here.
- Reuse the accepted `FFNO + PINN` row from the complete-table root as the same-architecture training-procedure comparator for the new supervised row.
- Add only the code and artifact plumbing needed to:
  - execute FFNO in supervised mode on the locked grid-lines split, or
  - close honestly as `not_protocol_compatible`
- Keep model labels explicit by architecture and training procedure, for example `FFNO + supervised` versus `FFNO + PINN`.

Explicit non-goals:

- Do not rewrite or relaunch the authoritative six-row `lines128` paper-complete bundle.
- Do not rerun the CDI `cnn` supervised row unless a checked-in contract amendment requires rerunning every affected comparator.
- Do not substitute supervised `fno` or any other architecture for the required FFNO supervised row.
- Do not change the selected `fno_vanilla` comparator, fixed `seed=3`, probe preprocessing, split identity, shared visual scales, or any other locked `lines128` contract field after seeing supervised-row metrics.
- Do not broaden into multi-seed sweeps, broader supervised-equivalent sweeps, PDEBench, `256x256` CDI scaling, manuscript prose, or `/home/ollie/Documents/neurips/` Phase 5 artifact assembly.
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not create worktrees.

## Binding Constraints And Prerequisite Status

Strategic and roadmap constraints:

- `docs/steering.md` requires equal-footing comparisons and forbids silently relaxing fairness constraints.
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md` places this work in Phase `3.3e`: the supervised FFNO control row runs only after the minimum local supervised/PINN pair and the complete PINN-trained `lines128` table are locked.
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md` keeps CDI anchored at `128x128`, preserves the paper-grade provenance bar, and forbids Phase 5 evidence-bundle work here.
- The backlog item requires this extension to remain separate from the primary CDI claim unless every included supervised row is same-contract and passes the same metric/provenance checks.

Prerequisites and current accepted evidence:

- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` shows `phase-0-evidence-inventory` and `phase-1-pde-benchmark-selection` complete, with no ledger-level blocked tranches.
- The backlog prerequisite `2026-04-29-cdi-lines128-paper-benchmark-execution` is complete as of `2026-04-30`.
  - Authoritative root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
  - Accepted six-row roster:
    `baseline`, `pinn`, `pinn_hybrid_resnet`, `pinn_fno_vanilla`, `pinn_spectral_resnet_bottleneck_net`, `pinn_ffno`
- The authoritative minimum-subset root already owns the accepted `CDI CNN + supervised` row:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T084339Z`
- The earlier FFNO-versus-Hybrid prerequisite root remains preserved prerequisite evidence only:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`

Locked contract that must not drift:

- `N=128`, `gridsize=1`, synthetic grid-lines, `set_phi=True`
- custom Run1084 probe at `datasets/Run1084_recon3_postPC_shrunk_3.npz`
- `probe_source=custom`, `probe_scale_mode=pad_extrapolate`, `probe_smoothing_sigma=0.5`, `probe_mask=off`
- `nimgs_train=2`, `nimgs_test=2`, `nphotons=1e9`
- fixed `seed=3`
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
- `fno_modes=12`, `fno_width=32`, `fno_blocks=4`, `fno_cnn_blocks=2`
- fixed sample ids: `0`, `1`
- shared visual scales derived from stitched numeric arrays

Findings and workflow rules that must stay enforced:

- `POLICY-001`
- `CONFIG-001`
- `GRIDLINES-OBJECT-BIG-001`
- `GRIDLINES-PROBE-BIG-001`
- `GRIDLINES-PROBE-PIPELINE-001`
- `OUTPUT-COMPLEX-001`
- `DATA-SUP-001`
- `REPORTING-ARTIFACT-BOUNDARY-001`
- Long-running commands must run in tmux, activate `ptycho311`, track the exact launched PID, and never launch a duplicate writer into the same output root.
- Normal import, path, environment, or test-harness failures must be diagnosed, fixed narrowly, and rerun before considering `BLOCKED`.
- If supervised FFNO still cannot satisfy the locked contract after one documented narrow compatibility-fix attempt, close the row as `not_protocol_compatible`; do not invent a broader supervised sweep as a substitute.

## Implementation Architecture

- **Authority and contract unit**
  - Freeze one supervised-extension execution authority and machine-readable manifest before launch.
  - That manifest must point to the preserved minimum-subset and complete-table roots, restate the frozen contract, and define the new row id/label pair for the supervised FFNO row.
- **Execution and compatibility unit**
  - Add only the narrow runner/wrapper/harness plumbing needed to execute FFNO in supervised mode on the locked split with full row-level provenance and explicit training-procedure labeling.
  - If the current codebase cannot support supervised FFNO without contract drift, emit `not_protocol_compatible` explicitly rather than overstating support.
- **Publication unit**
  - Produce an adjacent extension bundle and durable summary that compare `FFNO + supervised` against `FFNO + PINN`, reuse the accepted `CDI CNN + supervised` evidence by reference, and keep the six-row primary CDI benchmark summary unchanged as the headline claim authority.

## Concrete File And Artifact Targets

Likely code targets:

- `scripts/studies/lines128_paper_benchmark.py`
- `scripts/studies/grid_lines_compare_wrapper.py`
- `scripts/studies/grid_lines_torch_runner.py`
- `scripts/studies/metrics_tables.py`
- `scripts/studies/paper_provenance.py`
- `ptycho_torch/workflows/components.py` only if supervised FFNO requires a narrow existing-supervised-path compatibility fix

Likely test targets:

- `tests/studies/test_lines128_paper_benchmark.py`
- `tests/studies/test_metrics_tables.py`
- `tests/test_grid_lines_compare_wrapper.py`
- `tests/torch/test_grid_lines_torch_runner.py`
- `tests/torch/test_grid_lines_hybrid_resnet_integration.py`

Durable doc targets:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_supervised_equivalent_rows_execution_authority.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_supervised_equivalent_rows_summary.md`
- `docs/studies/index.md`
- `docs/index.md` only if discoverability materially changes
- `docs/findings.md` only if the work exposes a durable supervised-FFNO integration rule future workers need

Primary artifact targets:

- Item artifact root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/`
- Execution authority / audit / verification:
  - `.../execution/supervised_equivalent_execution_manifest.json`
  - `.../execution/protocol_compatibility_audit.md`
  - `.../verification/`
- Fresh extension run or bundle root:
  - `.../runs/supervised_ffno_extension_<timestamp>/`

## Execution Checklist

### Tranche 1: Freeze Extension Authority And Audit Protocol Compatibility

- [ ] Reconfirm that the authoritative six-row complete-table root and the authoritative minimum-subset root are both present, immutable inputs for this item.
- [ ] Create `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_supervised_equivalent_rows_execution_authority.md` plus `.../execution/supervised_equivalent_execution_manifest.json` that freeze:
  - the locked `lines128` contract listed above
  - the preserved prerequisite roots
  - the explicit extension claim boundary, for example `lines128_supervised_ffno_extension`
  - the reused comparator row `pinn_ffno`
  - the new supervised row id and label; use one explicit row id consistently, preferably `supervised_ffno` with paper label `FFNO + supervised`
- [ ] Audit the current code path for whether FFNO can run in supervised mode on the locked split without contract drift:
  - runner/config support for an explicit training procedure or model type
  - labeled-data compatibility under `DATA-SUP-001`
  - output-mode compatibility with the existing complex reconstruction path
  - row-local invocation/config/randomness/provenance completeness
  - bundle/table support for an adjacent supervised-extension result
- [ ] Record one of these pre-launch statuses in the audit note:
  - `supported_for_execution`
  - `requires_narrow_fix`
  - `not_protocol_compatible_pending_fix_attempt`
- [ ] Do not launch any run while this authority and audit are unresolved.

Verification before moving on:

- [ ] The checked-in authority note and machine-readable manifest agree on contract, row ids, preserved prerequisite roots, and claim boundary.
- [ ] The audit note states clearly whether a supervised FFNO launch is authorized, needs a narrow fix first, or appears likely `not_protocol_compatible`.

### Tranche 2: Close Only The Minimal Supervised FFNO Execution Gap

- [ ] Modify only the surfaces required by the Tranche 1 audit.
- [ ] If supervised FFNO needs explicit runner support, add the minimal plumbing necessary for:
  - an explicit supervised training procedure/model-type selection
  - explicit row labeling and provenance for the supervised FFNO row
  - reuse of the locked train/test split and fixed sample ids
  - full row-local outputs under the same artifact contract used by other paper-grade rows
- [ ] If the locked NPZ split lacks the exact supervised-loader shape or label bridge needed by the existing PyTorch supervised path, implement only a deterministic adapter from the locked split to the existing supervised labels contract. Do not introduce a different dataset, different sample ids, or a changed train/test split.
- [ ] If the adjacent extension bundle needs a new harness mode or manifest surface, add it narrowly without regressing existing `preflight`, `minimum_subset`, or `complete_table` behavior.
- [ ] If the current bundle schema cannot represent a truthful `not_protocol_compatible` supervised row, extend that schema and its tests rather than collapsing to a misleading status.

Verification before moving on:

- [ ] Run focused selectors covering touched surfaces, for example:

```bash
pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py
```

- [ ] Focused tests prove that the supervised FFNO row is either executable under the frozen contract or represented honestly as `not_protocol_compatible`.
- [ ] Existing minimum-subset and complete-table behavior remains intact.

### Tranche 3: Run Mandatory Deterministic Gates Before Any Expensive Training

- [ ] Archive logs under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/`
- [ ] Run the backlog item’s required deterministic checks exactly as written. These remain mandatory and are not replaced:

```bash
pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py
python -m compileall -q ptycho_torch scripts/studies
```

- [ ] Do not start supervised FFNO training, extension-bundle regeneration that mutates the new run root, or any other expensive execution until both commands are green and archived.
- [ ] If a check fails, diagnose, patch narrowly, and rerun before considering `BLOCKED`.

Verification before moving on:

- [ ] Both required commands are green and archived with timestamps.
- [ ] No expensive training starts on a red deterministic gate.

### Tranche 4: Launch Supervised FFNO Or Close Honestly As `not_protocol_compatible`

- [ ] Use this tranche only after Tranche 3 is green.
- [ ] If the Tranche 1/2 audit still shows protocol incompatibility after one narrow fix attempt, stop launches and emit a precise `not_protocol_compatible` row outcome plus bundle/summary evidence. This is an acceptable truthful closeout for the backlog item.
- [ ] If supervised FFNO is supported, create one new unique extension root under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/`
- [ ] Launch any long-running supervised FFNO training in tmux from repo root with `ptycho311` activated, exact PID tracking, and no duplicate writer into the same extension root.
- [ ] Preserve the locked contract exactly:
  - same dataset/split identity
  - same probe preprocessing and seed
  - same optimizer/scheduler/loss/output-mode budget
  - same sample ids and shared visual scales
- [ ] Do not rerun the accepted `CDI CNN + supervised` or `FFNO + PINN` rows. Reuse them by reference or deterministic promotion into the extension bundle as needed.
- [ ] Accept the supervised FFNO row only when the tracked PID exits `0` and the row emits the required artifacts, including:
  - `invocation.json` and `invocation.sh`
  - `config.json`
  - `history.json`
  - `metrics.json`
  - `randomness_contract.json`
  - row-local completion proof
  - `recons/<row>/recon.npz`
  - fixed-sample visuals and error panels required by the extension bundle

Verification before moving on:

- [ ] Either:
  - the supervised FFNO row exists with complete same-contract artifacts in the new extension root, or
  - the extension root/manifest records a precise `not_protocol_compatible` outcome with the missing interface or contract detail
- [ ] The preserved six-row primary benchmark root remains untouched.

### Tranche 5: Build The Adjacent Supervised-Equivalent Bundle And Durable Summary

- [ ] Build extension artifacts from the new supervised FFNO result plus preserved same-contract prerequisite rows.
- [ ] At minimum, publish an explicit training-procedure comparison for the same architecture:
  - `FFNO + supervised`
  - `FFNO + PINN`
- [ ] Reuse the accepted `CDI CNN + supervised` row by reference in the extension manifest and summary so the repo still exposes both supervised row families without retraining the CNN row.
- [ ] Emit table-ready JSON/CSV/TeX fragments, machine-readable manifests, source reconstruction arrays, fixed-sample amplitude/phase panels, and error panels under the new extension root.
- [ ] Keep the extension labeled as adjacent evidence unless every included supervised row is same-contract and provenance-complete.
- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_supervised_equivalent_rows_summary.md` with:
  - authoritative extension root
  - final row roster and labels
  - whether supervised FFNO executed or closed as `not_protocol_compatible`
  - explicit relation to the preserved six-row primary CDI benchmark
  - verification-log paths
  - any remaining caveats
- [ ] Update `docs/studies/index.md` so the new extension is discoverable without replacing the complete-table summary as the main `lines128` claim authority.
- [ ] Update `docs/index.md` or `docs/findings.md` only if a new durable discoverability or supervised-integration rule emerged.

Verification before moving on:

- [ ] The extension summary, manifest, and study-index entry all point to the same extension root and same claim boundary.
- [ ] Documentation explicitly distinguishes:
  - the preserved six-row primary CDI benchmark
  - the minimum-subset supervised CNN evidence
  - the new supervised FFNO extension result

### Tranche 6: Final Deterministic Closeout

- [ ] Rerun the backlog-required deterministic checks and archive the fresh logs under this item’s verification directory:

```bash
pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py
python -m compileall -q ptycho_torch scripts/studies
```

- [ ] Archive any focused selectors used to validate supervised FFNO execution or `not_protocol_compatible` schema handling.
- [ ] Record the final extension root, final row outcome, and final verification-log paths in the durable summary.

## Completion Criteria

- The backlog item ends with one of two truthful outcomes:
  - a same-contract supervised FFNO extension root with complete artifacts, or
  - a precise `not_protocol_compatible` result after one documented narrow fix attempt
- The authoritative six-row `lines128` benchmark root from `2026-04-30` remains the preserved primary CDI claim authority unless a later checked-in plan broadens that claim explicitly.
- The supervised FFNO extension does not drift from the frozen `lines128` contract.
- Any new or extended bundle/table labels architecture and training procedure explicitly.
- The durable summary and `docs/studies/index.md` point to the same extension root and same claim boundary.
- The backlog item’s required deterministic checks pass and their logs are archived.

## Required Deterministic Checks

Implementation may use stronger focused tests during development, but these backlog-item checks remain mandatory for completion and must be green before any expensive supervised FFNO training launch:

```bash
pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py
python -m compileall -q ptycho_torch scripts/studies
```
