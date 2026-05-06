# SRU-Net FFNO-To-PtychoBlock Encoder CDI/CNS Small-Cap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement exactly one encoder-only SRU-Net mechanism variant that swaps the baseline SRU-Net encoder for a fixed shared-weight 24-block FFNO stack plus two `PtychoBlock` encoder stages, then evaluate exactly one fresh `lines128` CDI row and one fresh capped PDEBench CNS row for `20` epochs without rerunning completed baselines.

**Architecture:** The work has four implementation units: a reusable FFNO stack/helper plus Hybrid ResNet encoder variant, CDI grid-lines registration for one new `pinn_*` row, PDEBench CNS manual-only registration for one new capped profile, and append-only evidence collation/index updates. The SRU-Net shell after the encoder must remain fixed, CDI and CNS conclusions must remain separate, and long-running launches stay under implementation ownership until the tracked run exits `0` and the required fresh artifacts exist.

**Tech Stack:** PATH `python`, long runs in `ptycho311`, PyTorch/Lightning, `ptycho_torch/generators/*`, `ptycho_torch/model.py`, `ptycho/config/config.py`, `ptycho_torch/workflows/components.py`, `scripts/studies/grid_lines_*`, `scripts/studies/pdebench_image128/*`, Markdown/JSON evidence indexes, `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/...`.

---

## Paused Status

Paused on 2026-05-06 at operator request because the corrected 24-layer
FFNO-encoder mechanism probe was too slow for the current backlog-drain pass.
The plan is retained for later reactivation, but the active workflow should
move on to other backlog items.

## Selected Objective

- Add one fresh CDI row on the locked `lines128` contract:
  `pinn_hybrid_resnet_ffno_ptychoblock_encoder` unless a clearer explicit name is justified and used consistently everywhere.
- Add one fresh capped CNS row on the matched-condition headline lane:
  `hybrid_resnet_ffno_ptychoblock_encoder_cns` unless a clearer explicit profile id is justified and used consistently everywhere.
- Answer one causal question only: does replacing the SRU-Net encoder with
  `shared-weight 24x FactorizedFfnoBlock stack -> 2x(PtychoBlock + downsample)`
  help or hurt the same downstream SRU-Net body on CDI and on capped CNS?

## Scope And Explicit Non-Goals

### In Scope

- Implement one explicit encoder-profile family shared across the CDI and PDEBench image128 entry points.
- Keep the SRU-Net shell outside the encoder fixed:
  same lifter/input policy, same two-step downsampling schedule, same bottleneck family and width/depth, same decoder family, same skip wiring, same residual scaling, same output mode, same loss family per benchmark, same scheduler policy, same seed policy, same visual sample policy, and same metric schema.
- Reuse completed baseline rows by lineage only:
  - CDI anchor bundle authority:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - CDI SRU-Net mechanism authority:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_srunet_branch_objective_ablation_summary.md`
  - CNS matched-condition authority:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md`
- Emit one durable append-only summary and the required discoverability/index updates.

### Explicit Non-Goals

- Do not rerun completed baselines just to assemble the comparison.
- Do not tune FFNO encoder depth, modes, sharing, gate init, normalization, MLP ratio, or local-conv policy after seeing the first metrics. Hyperparameter search is outside this item.
- Do not change decoder skip wiring, bottleneck family, residual scaling, probe/data contract, training objective, split policy, the approved `20`-epoch mechanism-probe budget, or metric definitions while claiming this is an encoder-only ablation.
- Do not average CDI and CNS into one scalar ranking or promote this item into a new default SRU-Net family.
- Do not reopen full-training CNS claims, rewrite the roadmap, overwrite the completed `lines128` paper bundle, or replace the matched-condition CNS headline table.
- Do not create worktrees.

## Binding Steering, Roadmap, And Policy Constraints

- This item is valid only because steering allows work inside the current Roadmap Phase 2 plus Phase 3 CDI-preparation window when it strengthens core comparison evidence. Do not expand the work into later roadmap phases or unrelated backlog items.
- Equal-footing and fairness constraints are binding. If the new row cannot stay on the locked CDI contract or the matched CNS contract, record the incompatibility instead of silently relaxing the protocol.
- Keep CDI and CNS conclusions separate. The CNS row remains `bounded_capped_decision_support_only` even if it performs well.
- Preserve the current headline authorities:
  - CDI headline bundle stays the six-row complete `lines128` authority.
  - CNS headline table stays the matched `history_len=5`, `512 / 64 / 64`, `40`-epoch capped lane.
- Scope update accepted on 2026-05-06: this item's corrected FFNO-encoder rows
  are `20`-epoch mechanism-probe rows. They answer whether the encoder swap is
  promising under the same data/split/loss shell; they do not replace the
  existing `40`-epoch CDI or CNS headline authorities.
- Long-run ownership rule:
  - use `tmux` for training/evaluation launches;
  - activate `ptycho311`, then invoke plain PATH `python`;
  - track the exact launched PID/session until exit;
  - do not start a duplicate run writing to the same output root;
  - treat a run as complete only when the tracked launch exits `0` and the required fresh row artifacts exist and are freshly written.
- Failure policy:
  - diagnose, fix, and rerun normal import, path, config, shape, environment, or harness failures;
  - do not mark the item `BLOCKED` for ordinary verification/test issues;
  - reserve `BLOCKED` for missing data/hardware, unavailable external dependency outside current authority, roadmap conflict, required user decision, or an unrecoverable failure that remains after a documented narrow fix attempt.
- Project-policy constraints that remain in force:
  - PATH `python` only (`PYTHON-ENV-001`);
  - preserve current grid-lines Torch contracts from `docs/findings.md`, especially the current probe-mask semantics and the `object_big=False` / `probe_big=False` parity expectations on the grid-lines path;
  - if a touched path still relies on legacy globals, preserve `update_legacy_dict(params.cfg, config)` ordering before touching legacy modules.

## Prerequisite Status

- Progress-ledger status relevant to this item:
  - completed tranches include `phase-0-evidence-inventory` and `phase-1-pde-benchmark-selection`;
  - no global blocked tranches are recorded in the consumed ledger;
  - this item depends on already-completed backlog authorities, not unresolved roadmap blockers.
- Backlog authorities that must already exist before implementation proceeds:
  - `docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md`
  - `docs/backlog/done/2026-05-04-cdi-lines128-srunet-branch-objective-ablation.md`
  - `docs/backlog/done/2026-05-04-cns-matched-condition-table-refresh.md`
- Fixed CDI contract to preserve:
  - dataset contract id `cdi_lines128_seed3`
  - `N=128`, `gridsize=1`, `seed=3`
  - `20` epochs, `batch_size=16`, `lr=2e-4`
  - `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-4, threshold=0.0)`
  - `torch_loss_mode=mae`, `torch_output_mode=real_imag`
  - fixed sample ids `0`, `1`
  - baseline/comparator lineage rows to reuse by reference:
    `pinn_hybrid_resnet`, `pinn_hybrid_resnet_encoder_spectral_only`,
    historical proxy `pinn_ffno`, `pinn_hybrid_resnet_ffno_bottleneck`.
    Label `pinn_ffno` as `FFNO-local proxy` unless the corrected no-refiner
    row is explicitly substituted by lineage.
- Fixed CNS contract to preserve:
  - selected lane `h5_512_64_64_40ep` as the split/history/loss authority
  - `history_len=5`
  - split counts `512 / 64 / 64`
  - `20` epochs, `batch_size=4`, Adam `2e-4`
  - training loss `mse`
  - metric family `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
  - comparator lineage rows to reuse by reference:
    `author_ffno_cns_base`, `spectral_resnet_bottleneck_base`, `fno_base`, `unet_strong`

## Implementation Architecture

- **Shared FFNO encoder stack:** expose one reusable FFNO stack helper built from `FactorizedFfnoBlock` / `FactorizedSpectralConv2d` so the end-to-end FFNO generator and the encoder-ablation variant do not maintain divergent block-stack logic.
- **Hybrid encoder variant:** implement one explicit Hybrid ResNet family that keeps the SRU-Net shell fixed but swaps the encoder for the fixed FFNO-first recipe followed by exactly two `PtychoBlock` stages paired with the existing downsample steps.
- **CDI integration surface:** register one append-only `pinn_*` row through the grid-lines runner and compare-wrapper path so config reconstruction, manifests, labels, and parameter reporting remain deterministic.
- **CNS integration surface:** register one manual-only PDEBench image128 profile for the same encoder recipe under the locked `history_len=5`, `512 / 64 / 64`, `20`-epoch mechanism-probe CNS shell and keep it out of default profile bundles.

## File And Artifact Targets

### Mandatory Code Targets

- Modify: `ptycho_torch/generators/hybrid_resnet.py`
- Modify: `ptycho_torch/generators/ffno.py`
- Modify only if helper extraction or compatibility requires it:
  `ptycho_torch/generators/ffno_bottleneck.py`
- Modify if architecture/config round-trip requires it:
  `ptycho_torch/model.py`
- Modify if architecture enums/validation require it:
  `ptycho/config/config.py`
- Modify if override round-trip requires it:
  `ptycho_torch/workflows/components.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: `scripts/studies/pdebench_image128/models.py`
- Modify: `scripts/studies/pdebench_image128/run_config.py`

### Mandatory Test Targets

- Modify or add focused generator/shape/metadata coverage:
  - `tests/torch/test_fno_generators.py`
  - `tests/torch/test_generator_registry.py`
  - `tests/torch/test_lightning_checkpoint.py`
- Modify or add grid-lines runner/wrapper coverage:
  - `tests/torch/test_grid_lines_torch_runner.py`
  - `tests/test_grid_lines_compare_wrapper.py`
- Modify or add PDEBench model/profile coverage:
  - `tests/studies/test_pdebench_image128_models.py`
  - `tests/studies/test_pdebench_image128_runner.py` if manual-only profile routing or row collation changes

### Mandatory Contract Outputs

- Item artifact root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/`
- Fresh CDI run root under the item artifact root, containing:
  row-local invocation/config/history/metrics/reconstruction/visuals plus `exit_code_proof.json`
- Fresh CNS run root under the item artifact root, containing:
  row-local invocation/config/history/metrics/field-visual artifacts plus `exit_code_proof.json`
- Durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/srunet_ffno_ptychoblock_encoder_cdi_cns_smallcap_summary.md`

### Preferred Packaging And Discoverability Updates

- Update:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/studies/index.md`
- Update `docs/index.md` only if implementation creates a reusable new study entry point that should be discoverable from the hub.
- Update `docs/findings.md` only if implementation surfaces a durable, reusable project-wide rule rather than an item-local result.

## Required Deterministic Checks

These are required gates for this item. They remain mandatory unless a later approved plan revision explicitly replaces them with a stronger equivalent.

### Prerequisite Presence Gate

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md"),
    Path("docs/backlog/done/2026-05-04-cdi-lines128-srunet-branch-objective-ablation.md"),
    Path("docs/backlog/done/2026-05-04-cns-matched-condition-table-refresh.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_srunet_branch_objective_ablation_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md"),
    Path("ptycho_torch/generators/ffno.py"),
    Path("ptycho_torch/generators/fno.py"),
    Path("ptycho_torch/generators/ffno_bottleneck.py"),
    Path("ptycho_torch/generators/hybrid_resnet.py"),
    Path("scripts/studies/grid_lines_torch_runner.py"),
    Path("scripts/studies/run_pdebench_image128_suite.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing SRU-Net FFNO->PtychoBlock encoder inputs: {missing}")
print("SRU-Net FFNO->PtychoBlock encoder inputs present")
PY
```

### Required Pytest And Compile Gates

```bash
pytest -q tests/torch/test_fno_generators.py -k "PtychoBlock or hybrid_resnet or ffno"
pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno"
pytest -q tests/studies/test_pdebench_image128_models.py -k "hybrid_resnet or ffno"
python -m compileall -q ptycho_torch scripts/studies
```

## Execution Checklist

### Task 1: Freeze Authorities And Baseline Lineage

**Files / Artifacts**

- Read-only authority anchors:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_srunet_branch_objective_ablation_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md`
- Create item-local execution notes under the item artifact root if useful for lineage bookkeeping.

- [ ] Run the prerequisite presence gate exactly as written before editing code.
- [ ] Record the exact CDI comparison contract, CNS comparison contract, and the reused lineage row ids in item-local notes so later collation does not infer them ad hoc.
- [ ] Confirm that this item remains append-only mechanism evidence and does not replace the authoritative CDI or CNS headline bundles.

**Verification**

- Blocking: prerequisite presence gate passes.
- Supporting: item-local lineage note captures the reused row ids and authority docs before fresh launches begin.

### Task 2: Implement The Shared FFNO Encoder Variant

**Files**

- Modify: `ptycho_torch/generators/ffno.py`
- Modify: `ptycho_torch/generators/hybrid_resnet.py`
- Modify if needed for helper reuse only:
  `ptycho_torch/generators/ffno_bottleneck.py`
- Modify if configuration/checkpoint round-trip requires it:
  `ptycho_torch/model.py`, `ptycho/config/config.py`, `ptycho_torch/workflows/components.py`
- Test:
  `tests/torch/test_fno_generators.py`
  `tests/torch/test_generator_registry.py`
  `tests/torch/test_lightning_checkpoint.py`

- [ ] Add one explicit architecture id, preferably `hybrid_resnet_ffno_ptychoblock_encoder`.
- [ ] Factor the reusable FFNO block stack so both the end-to-end FFNO generator and the encoder-ablation variant share the same local FFNO-stack implementation.
- [ ] Implement the encoder recipe exactly once and keep it fixed for this item:
  - `encoder_variant = "ffno_ptychoblock_encoder"`
  - `ffno_encoder_blocks = 24`
  - `ffno_encoder_modes = 12` unless a fixed-shape CNS constraint forces a documented change
  - `ffno_encoder_share_weights = true`
  - `ffno_encoder_gate_init = 0.1`
  - `ffno_encoder_norm = "instance"`
  - `ffno_encoder_mlp_ratio = 2.0`
  - no FFNO local residual refiners in this encoder ablation
  - `ptychoblock_stage_count = 2`
  - `downsample_steps = 2`
  - `downsample_op` inherited from the existing SRU-Net shell
- [ ] Preserve baseline `hybrid_resnet` behavior exactly.
- [ ] Ensure the required manifest/config fields round-trip cleanly:
  `encoder_variant`, `ptychoblock_stage_count`, `downsample_steps`,
  `downsample_op`, `ffno_encoder_blocks`, `ffno_encoder_modes`,
  `ffno_encoder_share_weights`, `ffno_encoder_gate_init`,
  `ffno_encoder_norm`, `ffno_encoder_mlp_ratio`.

**Verification**

- Blocking: `pytest -q tests/torch/test_fno_generators.py -k "PtychoBlock or hybrid_resnet or ffno"`
- Blocking: `python -m compileall -q ptycho_torch`
- Supporting: `pytest -q tests/torch/test_generator_registry.py`
- Supporting: `pytest -q tests/torch/test_lightning_checkpoint.py -k "hybrid_resnet or ffno"`

### Task 3: Wire The CDI Grid-Lines Row

**Files**

- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify if config bridge requires it:
  `ptycho_torch/model.py`, `ptycho_torch/workflows/components.py`, `ptycho/config/config.py`
- Test:
  `tests/torch/test_grid_lines_torch_runner.py`
  `tests/test_grid_lines_compare_wrapper.py`

- [ ] Register the new row id, label, architecture mapping, and append-only claim/status behavior for the CDI compare-wrapper path.
- [ ] Thread the fixed encoder recipe into runner config reconstruction, manifest output, and row-local metadata.
- [ ] Preserve the existing locked `lines128` contract, including seed/sample policy, probe lineage, loss mode, scheduler, and current grid-lines Torch parity defaults.
- [ ] Make sure parameter counts, row manifests, and run-level manifests expose the fixed encoder recipe fields and do not auto-promote this row to headline paper evidence.

**Verification**

- Blocking: `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno"`
- Blocking: `pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or ffno"`
- Supporting: `python -m compileall -q scripts/studies`

### Task 4: Wire The Manual-Only CNS Profile

**Files**

- Modify: `scripts/studies/pdebench_image128/models.py`
- Modify: `scripts/studies/pdebench_image128/run_config.py`
- Test:
  `tests/studies/test_pdebench_image128_models.py`
  `tests/studies/test_pdebench_image128_runner.py` if manual-only profile routing or bundle selection changes

- [ ] Register one manual-only profile id, preferably `hybrid_resnet_ffno_ptychoblock_encoder_cns`.
- [ ] Build the profile under the canonical CNS shell:
  `history_len=5`, `512 / 64 / 64`, `20` epochs, `batch_size=4`, Adam `2e-4`, MSE, same metric family.
- [ ] Preserve the existing supervised real-channel adapter semantics, downsample schedule, skip structure, decoder family, and current CNS row-manifest/reporting behavior outside the encoder.
- [ ] Keep the new profile out of default `required_primary_profiles_for_task()` bundles so it launches only when explicitly requested.
- [ ] Ensure profile description/manifests surface the fixed encoder recipe fields and `encoder_variant`.

**Verification**

- Blocking: `pytest -q tests/studies/test_pdebench_image128_models.py -k "hybrid_resnet or ffno"`
- Blocking: `python -m compileall -q scripts/studies`
- Supporting: `pytest -q tests/studies/test_pdebench_image128_runner.py -k "hybrid_resnet or ffno"`

### Task 5: Re-Run Deterministic Code Gates Before Expensive Launches

**Files / Artifacts**

- Verification logs under the item artifact root.

- [ ] Re-run the prerequisite presence gate.
- [ ] Re-run every required deterministic check from this plan after code changes.
- [ ] Add stronger narrow checks for touched routing/round-trip surfaces because the required backlog gates do not fully cover them.
- [ ] Do not start CDI or CNS training until all blocking code gates are passing.

**Verification**

- Blocking:
  - prerequisite presence gate
  - `pytest -q tests/torch/test_fno_generators.py -k "PtychoBlock or hybrid_resnet or ffno"`
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno"`
  - `pytest -q tests/studies/test_pdebench_image128_models.py -k "hybrid_resnet or ffno"`
  - `python -m compileall -q ptycho_torch scripts/studies`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or ffno"`
- Supporting:
  - `pytest -q tests/torch/test_generator_registry.py`
  - `pytest -q tests/torch/test_lightning_checkpoint.py -k "hybrid_resnet or ffno"`
  - `pytest -q tests/studies/test_pdebench_image128_runner.py -k "hybrid_resnet or ffno"`
  - `pytest -v -m integration` if the active environment supports the repo-wide integration marker; if it does not, record the reason and keep the narrower workflow evidence.

### Task 6: Launch The Fresh CDI Row

**Files / Artifacts**

- Launch path:
  `scripts/studies/grid_lines_compare_wrapper.py`
- Fresh row artifacts under the item artifact root.

- [ ] Use the existing grid-lines compare-wrapper path to launch only the new CDI row on the locked `lines128` contract.
- [ ] Run in `tmux`, activate `ptycho311`, and keep the launch under implementation ownership until exit `0` or a documented recoverable failure handling path completes.
- [ ] If a cheap dry-run or manifest preflight exists for the exact row wiring, it may run first as a supporting check only. It does not satisfy the CDI evidence requirement.
- [ ] On completion, verify fresh row-local invocation/config/history/metrics, reconstructions, fixed-sample visuals, and `exit_code_proof.json`.
- [ ] Reuse baseline rows only by lineage; do not rerun them.

**Verification**

- Blocking: tracked CDI launch exits `0`.
- Blocking: the fresh CDI row artifacts exist under the item root and are newer than the launch start.
- Supporting: item-local audit note records which authoritative CDI baseline rows were compared by lineage.

### Task 7: Launch The Fresh CNS Row

**Files / Artifacts**

- Launch path:
  `scripts/studies/run_pdebench_image128_suite.py`
  plus the `scripts/studies/pdebench_image128/*` stack
- Fresh CNS row artifacts under the item artifact root.

- [ ] Launch only the new manual CNS profile on official `2d_cfd_cns` data under the matched `h5_512_64_64_40ep` split/history/loss authority with `history_len=5`, but set the mechanism-probe epoch budget to `20`.
- [ ] Keep the row on the existing CNS MSE recipe and matched metric family. Do not mix caps or history lengths. If comparing against `40`-epoch endpoint rows, label the epoch-budget difference explicitly and prefer same-epoch lineage comparisons where histories exist.
- [ ] A shorter smoke/pilot launch is allowed only to prove implementation viability or recover from a normal failure. Label it readiness-only and keep it separate from the required `20`-epoch item result.
- [ ] On completion, verify fresh row-local metrics, profile/config metadata, field-visual artifacts, invocation/history, and `exit_code_proof.json`.

**Verification**

- Blocking: tracked `20`-epoch CNS launch exits `0`, or a documented unrecoverable external blocker is recorded after a narrow fix attempt.
- Blocking: fresh CNS row artifacts exist and the eventual summary labels the result `bounded_capped_decision_support_only`.
- Supporting: if a smoke/pilot run was needed, archive it separately and label it readiness-only.

### Task 8: Collate Append-Only Evidence And Update Discoverability

**Files**

- Create or update:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/srunet_ffno_ptychoblock_encoder_cdi_cns_smallcap_summary.md`
- Update:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  `docs/studies/index.md`

- [ ] Build an item-local comparison bundle containing only the two fresh rows plus lineage references to the reused CDI and CNS baselines.
- [ ] Write the durable summary with separate CDI and CNS sections, explicit claim-boundary language, the fixed encoder recipe fields, row lineage, and domain-dependent interpretation if the variant helps one benchmark and hurts the other.
- [ ] Update the evidence/index surfaces required by the roadmap evidence-index policy.
- [ ] Keep paper-facing updates bounded: this item is mechanism evidence only and must not replace the current CDI or CNS headline authorities.
- [ ] If implementation surfaced a reusable repo-wide rule or a new durable study entry point, update `docs/findings.md` and/or `docs/index.md` in the same pass. Otherwise leave them unchanged and say so in the execution report.

**Verification**

- Blocking: the durable summary exists and names the fresh row ids, reused lineage rows, fixed CDI contract, fixed CNS contract, and claim boundary.
- Blocking: every new durable output is reflected in the appropriate evidence/index surface, or the summary explicitly states why a given surface was intentionally unchanged.
- Supporting: a final pass confirms the summary does not overclaim beyond append-only mechanism evidence or capped CNS decision-support evidence.

## Completion Criteria

- The codebase exposes one explicit `hybrid_resnet_ffno_ptychoblock_encoder` family with the fixed encoder recipe and manifest/config round-trip fields.
- The grid-lines path can launch exactly one fresh append-only CDI row for that family under the locked `lines128` contract.
- The PDEBench image128 path can launch exactly one fresh manual-only capped CNS row for that family under the matched `history_len=5`, `512 / 64 / 64`, `20`-epoch mechanism-probe contract.
- All blocking deterministic checks in this plan pass before the expensive launches.
- Both long runs either complete with exit `0` plus fresh required artifacts or conclude with a documented narrow-fix attempt and a legitimate blocker under the item’s blocker policy.
- The durable summary and evidence indexes are updated without rewriting the authoritative CDI or CNS headline bundles.
