# SRU-Net FFNO-To-PtychoBlock Encoder CDI/CNS Small-Cap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate and publish the corrected `24`-block SRU-Net FFNO-to-PtychoBlock encoder mechanism probe by producing one fresh `lines128` CDI row and one fresh capped PDEBench CNS row, then rebuilding the item-local evidence bundle and index surfaces from those corrected outputs only.

**Architecture:** Treat the generator/profile wiring as already present but not yet fully discharged as evidence. The execution work is an audit-plus-rerun pipeline: confirm the approved fixed encoder recipe is encoded consistently across generator, CDI runner, and CNS profile surfaces; run the required deterministic gates; launch fresh `20`-epoch CDI and CNS reruns under tracked ownership; and then rewrite the summary/index surfaces so the corrected `24`-block roots are current authority while the historical `2`-block roots remain superseded lineage only. CDI and CNS stay separate, and neither corrected row may replace the existing `40`-epoch headline authorities.

**Tech Stack:** PATH `python`, long runs in `ptycho311`, tmux-backed launch ownership, PyTorch/Lightning, `ptycho_torch/generators/*`, `ptycho_torch/model.py`, `scripts/studies/grid_lines_*`, `scripts/studies/pdebench_image128/*`, Markdown/JSON evidence indexes, `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/...`.

---

## Selected Objective

- Publish one corrected CDI mechanism row on the locked `lines128` contract:
  `pinn_hybrid_resnet_ffno_ptychoblock_encoder`.
- Publish one corrected capped CNS mechanism row on the matched-condition capped
  lane:
  `hybrid_resnet_ffno_ptychoblock_encoder_cns`.
- Answer one causal question only: does replacing the current SRU-Net encoder
  with `24x shared-weight FFNO -> 2x(PtychoBlock + downsample)` help or hurt
  the same downstream SRU-Net shell on CDI and on capped CNS?

## Scope And Explicit Non-Goals

### In Scope

- Audit the already-landed architecture/profile registration and repair it only
  if a remaining recipe mismatch or run-blocking defect is found.
- Reuse completed baseline rows by lineage only:
  - CDI headline authority:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - CDI SRU-Net mechanism authority:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_srunet_branch_objective_ablation_summary.md`
  - CNS matched-condition authority:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md`
- Produce two fresh corrected `20`-epoch reruns, one on CDI and one on CNS,
  under the fixed encoder recipe and the locked per-benchmark shells.
- Rebuild the item-local comparison bundle, durable summary, and evidence/index
  surfaces from the corrected run roots.

### Explicit Non-Goals

- Do not redesign the architecture family. This item is not a new brainstorming
  pass and not a new default SRU-Net proposal.
- Do not tune `ffno_encoder_blocks`, modes, sharing, gate init, norm, MLP ratio,
  or any other encoder hyperparameter after seeing the first metrics.
- Do not change decoder skip wiring, bottleneck family, residual scaling,
  dataset/probe contract, training objective, split policy, metric schema, or
  visual sample policy while claiming this is an encoder-only ablation.
- Do not rerun completed baseline rows just to assemble comparisons.
- Do not average CDI and CNS into one scalar ranking.
- Do not replace the six-row CDI authority or the matched-condition CNS
  headline table.
- Do not reopen full-training CNS claims, later roadmap phases, or unrelated
  backlog items.
- Do not create worktrees.

## Binding Steering, Roadmap, And Policy Constraints

- Steering allows this item only because it strengthens the current Phase 2
  plus Phase 3 evidence window. Do not expand beyond that window.
- Equal-footing and fairness constraints are binding. If the new row cannot stay
  on the locked CDI or CNS contract, record the incompatibility instead of
  silently relaxing the protocol.
- Keep CDI and CNS conclusions separate. The CNS row remains
  `bounded_capped_decision_support_only` even if it performs well.
- Preserve the current headline authorities:
  - CDI headline bundle remains the six-row `lines128` authority.
  - CNS headline bundle remains the matched `history_len=5`, `512 / 64 / 64`,
    `40`-epoch capped lane.
- The corrected mechanism rows are `20`-epoch probes. They do not replace the
  existing `40`-epoch CDI or CNS headline authorities.
- Long-running commands remain under implementation ownership until completion:
  use tmux, activate `ptycho311`, track the exact launched PID/session, do not
  duplicate a writer to the same output root, and consider a run complete only
  when the tracked launch exits `0` and the required fresh artifacts exist.
- Failure handling:
  - diagnose, fix, and rerun ordinary import, path, config, shape, environment,
    or harness failures;
  - do not mark the item `BLOCKED` for normal verification failures;
  - reserve `BLOCKED` for missing data/hardware, unavailable external
    dependency outside current authority, roadmap conflict, required user
    decision, or a failure that remains unrecoverable after a documented narrow
    fix attempt.
- PyTorch and legacy-config guardrails remain active:
  - PATH `python` only;
  - preserve `docs/findings.md` contracts such as probe-mask semantics and
    grid-lines parity expectations;
  - if a touched path still depends on legacy globals, keep
    `update_legacy_dict(params.cfg, config)` ordering intact before touching
    legacy modules.

## Prerequisite Status And Current Starting State

- Progress-ledger status relevant to this item:
  - completed tranches include `phase-0-evidence-inventory` and
    `phase-1-pde-benchmark-selection`;
  - no global blocked tranches are recorded in the consumed ledger.
- Required completed backlog authorities already exist:
  - `docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md`
  - `docs/backlog/done/2026-05-04-cdi-lines128-srunet-branch-objective-ablation.md`
  - `docs/backlog/done/2026-05-04-cns-matched-condition-table-refresh.md`
- Current repo state matters:
  - the codebase already exposes
    `hybrid_resnet_ffno_ptychoblock_encoder`,
    `pinn_hybrid_resnet_ffno_ptychoblock_encoder`, and
    `hybrid_resnet_ffno_ptychoblock_encoder_cns`;
  - the durable summary currently says
    `implementation_corrected_rerun_pending`;
  - the historical CDI/CNS roots under this item still reflect the superseded
    `ffno_encoder_blocks=2` recipe and must not remain current authority.

### Fixed CDI Contract To Preserve

- dataset contract id: `cdi_lines128_seed3`
- `N=128`, `gridsize=1`, `seed=3`
- `20` epochs, `batch_size=16`, `lr=2e-4`
- `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-4, threshold=0.0)`
- `torch_loss_mode=mae`, `torch_output_mode=real_imag`
- fixed sample ids `0`, `1`
- reuse these comparison rows by lineage only:
  `pinn_hybrid_resnet`, `pinn_hybrid_resnet_encoder_spectral_only`,
  `pinn_ffno` as `FFNO-local proxy` unless a corrected no-refiner row is
  explicitly substituted by lineage, and
  `pinn_hybrid_resnet_ffno_bottleneck`

### Fixed CNS Contract To Preserve

- authority lane: `h5_512_64_64_40ep` for split/history/loss lineage
- `history_len=5`
- split counts `512 / 64 / 64`
- `20` epochs, `batch_size=4`, Adam `2e-4`
- training loss `mse`
- metric family:
  `err_RMSE`, `err_nRMSE`, `relative_l2`,
  `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- reuse these comparison rows by lineage only:
  `spectral_resnet_bottleneck_base`, `author_ffno_cns_base`,
  `fno_base`, `unet_strong`

### Approved Fixed Encoder Recipe

- architecture id: `hybrid_resnet_ffno_ptychoblock_encoder`
- `encoder_variant=ffno_ptychoblock_encoder`
- `ffno_encoder_blocks=24`
- `ffno_encoder_modes=12`
- `ffno_encoder_share_weights=true`
- `ffno_encoder_gate_init=0.1`
- `ffno_encoder_norm=instance`
- `ffno_encoder_mlp_ratio=2.0`
- `ptychoblock_stage_count=2`
- `downsample_steps=2`
- `downsample_op=stride_conv`

## Implementation Architecture

- **Contract audit unit:** verify the fixed recipe and manifest fields across
  generator, registry, CDI runner, and CNS profile surfaces before any rerun.
- **CDI rerun unit:** produce one fresh corrected `lines128` row under tracked
  ownership and validate its full artifact set.
- **CNS rerun unit:** produce one fresh corrected capped CNS row under tracked
  ownership and validate its full artifact set without promoting it to the
  headline `40`-epoch table.
- **Evidence-repair unit:** rebuild the item-local comparison bundle and update
  summary/index surfaces so only corrected `24`-block roots are current
  authority.

## File And Artifact Targets

### Mandatory Contract Outputs

- Item artifact root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/`
- Fresh CDI run root under the item artifact root, with row-local invocation,
  config, history, metrics, visuals, and `exit_code_proof.json`
- Fresh CNS run root under the item artifact root, with row-local invocation,
  config, history, metrics, field visuals, and `exit_code_proof.json`
- Rebuilt machine-readable compare bundle:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/comparison_bundle.json`
- Updated durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/srunet_ffno_ptychoblock_encoder_cdi_cns_smallcap_summary.md`

### Conditional Code Targets

Modify these only if the audit finds a remaining fixed-recipe mismatch, a
manifest gap, or a rerun-blocking defect:

- `ptycho_torch/generators/hybrid_resnet.py`
- `ptycho_torch/model.py`
- `ptycho_torch/generators/registry.py`
- `scripts/studies/grid_lines_torch_runner.py`
- `scripts/studies/grid_lines_compare_wrapper.py`
- `scripts/studies/pdebench_image128/models.py`
- `scripts/studies/pdebench_image128/run_config.py`

### Conditional Test Targets

Touch or extend these only if the audit or a rerun fix changes the owned code
surface:

- `tests/torch/test_fno_generators.py`
- `tests/torch/test_generator_registry.py`
- `tests/torch/test_lightning_checkpoint.py`
- `tests/torch/test_grid_lines_torch_runner.py`
- `tests/test_grid_lines_compare_wrapper.py`
- `tests/studies/test_pdebench_image128_models.py`
- `tests/studies/test_pdebench_image128_runner.py`

### Preferred Packaging And Discoverability Updates

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- `docs/studies/index.md`
- Update `docs/index.md` only if implementation creates or materially changes a
  reusable study entry point that the hub should advertise.
- Update `docs/findings.md` only if the rerun uncovers a durable,
  project-wide rule rather than an item-local result.

## Required Deterministic Checks

These are required gates for this item. They remain mandatory unless a later
approved plan revision replaces them with a stronger equivalent.

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

### Stronger Conditional Checks

Use these when the audit or a rerun fix changes the corresponding surface:

```bash
pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or ffno"
pytest -q tests/torch/test_generator_registry.py
pytest -q tests/torch/test_lightning_checkpoint.py -k "hybrid_resnet or ffno"
pytest -v -m integration
```

- `pytest -v -m integration` is blocking if any production workflow code is
  edited during this item. If the implementation stays docs-and-reruns only,
  it is supporting evidence rather than a required gate.

## Execution Checklist

### Task 1: Reconfirm Authority, Audit The Fixed Recipe, And Prepare Clean Reruns

**Files / Artifacts**

- Read-only authority docs:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_srunet_branch_objective_ablation_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/srunet_ffno_ptychoblock_encoder_cdi_cns_smallcap_summary.md`
- Audit-first code surfaces:
  - `ptycho_torch/generators/hybrid_resnet.py`
  - `ptycho_torch/model.py`
  - `ptycho_torch/generators/registry.py`
  - `scripts/studies/grid_lines_torch_runner.py`
  - `scripts/studies/grid_lines_compare_wrapper.py`
  - `scripts/studies/pdebench_image128/models.py`
  - `scripts/studies/pdebench_image128/run_config.py`
- Item artifact root for fresh corrected reruns:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/`

- [ ] Run the prerequisite presence gate exactly as written before editing or launching anything.
- [ ] Audit the current recipe across generator, registry, grid-lines runner, compare-wrapper row spec, PDEBench model/profile surfaces, and current summary state.
- [ ] Confirm the required manifest fields are preserved:
  `encoder_variant`, `ptychoblock_stage_count`, `downsample_steps`,
  `downsample_op`, `ffno_encoder_blocks`, `ffno_encoder_modes`,
  `ffno_encoder_share_weights`, `ffno_encoder_gate_init`,
  `ffno_encoder_norm`, `ffno_encoder_mlp_ratio`.
- [ ] If the audit finds a remaining mismatch or run-blocking defect, make the narrowest possible fix and rerun the affected deterministic checks before any expensive launch.
- [ ] Reserve fresh timestamped run roots for CDI and CNS under the item artifact root and keep the historical `2`-block roots read-only as superseded context.

**Verification**

- Blocking:
  - prerequisite presence gate passes;
  - `pytest -q tests/torch/test_fno_generators.py -k "PtychoBlock or hybrid_resnet or ffno"`
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno"`
  - `pytest -q tests/studies/test_pdebench_image128_models.py -k "hybrid_resnet or ffno"`
  - `python -m compileall -q ptycho_torch scripts/studies`
- Supporting:
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or ffno"`
  - `pytest -q tests/torch/test_generator_registry.py`
  - `pytest -q tests/torch/test_lightning_checkpoint.py -k "hybrid_resnet or ffno"`
  - `pytest -v -m integration` if no production code changed

### Task 2: Run The Corrected `lines128` CDI Mechanism Row

**Files / Artifacts**

- Launch surfaces:
  - `scripts/studies/grid_lines_torch_runner.py`
  - `scripts/studies/grid_lines_compare_wrapper.py`
- Fresh run root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/runs/cdi_ffno_ptychoblock_encoder_<timestampZ>/`

- [ ] Launch only the new corrected row `pinn_hybrid_resnet_ffno_ptychoblock_encoder` on the locked `cdi_lines128_seed3` contract.
- [ ] Keep the inherited CDI shell fixed:
  `20` epochs, `batch_size=16`, Adam `2e-4`, MAE loss,
  `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-4, threshold=0.0)`,
  fixed sample ids `0` and `1`, and the standard `real_imag` output mode.
- [ ] Launch via tmux in `ptycho311`, track the exact started PID/session, and wait for terminal completion rather than polling broad process matches.
- [ ] If the run fails for an ordinary harness or code reason, diagnose, fix narrowly, rerun the affected tests, and relaunch under a fresh output root.
- [ ] Mark the row complete only after the tracked launch exits `0` and the fresh run root contains invocation/config/history/metrics/visuals plus `exit_code_proof.json`.
- [ ] Preserve the historical `2`-block CDI root as superseded debugging lineage only; do not merge it into the corrected evidence root.

**Verification**

- Blocking:
  - tracked CDI launch exits `0`;
  - fresh CDI artifact root exists and is freshly written;
  - emitted metadata records the approved fixed encoder recipe;
  - emitted metrics/visuals correspond to the corrected `24`-block row only.
- Supporting:
  - compare the corrected row against reused CDI lineage rows in item-local notes or during bundle rebuild;
  - rerun `pytest -v -m integration` as a blocking gate before relaunch if a production workflow fix was required.

### Task 3: Run The Corrected Capped CNS Mechanism Row

**Files / Artifacts**

- Launch surfaces:
  - `scripts/studies/run_pdebench_image128_suite.py`
  - `scripts/studies/pdebench_image128/models.py`
  - `scripts/studies/pdebench_image128/run_config.py`
- Fresh run root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/runs/cns_ffno_ptychoblock_encoder_<timestampZ>/`

- [ ] Launch only the manual-only profile `hybrid_resnet_ffno_ptychoblock_encoder_cns`.
- [ ] Keep the inherited CNS shell fixed:
  `history_len=5`, split `512 / 64 / 64`, `20` epochs, `batch_size=4`,
  Adam `2e-4`, MSE loss, same metric family, same `pixelshuffle` upsampler,
  same skip-add SRU-Net shell outside the encoder.
- [ ] Keep the run explicitly labeled as a `20`-epoch mechanism probe on the
  matched-condition capped lane. Do not inject it into the `40`-epoch headline
  table.
- [ ] Launch via tmux in `ptycho311`, track the exact started PID/session, and wait for terminal completion.
- [ ] If the run fails for an ordinary harness or code reason, diagnose, fix narrowly, rerun the affected tests, and relaunch under a fresh output root.
- [ ] Mark the row complete only after the tracked launch exits `0` and the fresh run root contains invocation/config/history/metrics/field visuals plus `exit_code_proof.json`.
- [ ] When interpreting the row, compare it to same-contract lineage where available and label any comparison to `40`-epoch endpoints explicitly.

**Verification**

- Blocking:
  - tracked CNS launch exits `0`;
  - fresh CNS artifact root exists and is freshly written;
  - emitted metadata records the approved fixed encoder recipe and the capped
    `history_len=5`, `512 / 64 / 64`, `20`-epoch contract;
  - summary inputs keep the CNS row in bounded mechanism-evidence scope only.
- Supporting:
  - if runner/bundle code changed, rerun the affected stronger checks, including
    `tests/studies/test_pdebench_image128_runner.py` selectors that cover the
    touched surface;
  - keep a short note mapping the corrected row to the reused CNS lineage rows
    and explicitly separating `20`-epoch probe evidence from `40`-epoch
    headline context.

### Task 4: Rebuild The Comparison Bundle And Repair Evidence Surfaces

**Files / Artifacts**

- Rebuilt compare bundle:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/comparison_bundle.json`
- Required durable docs/indexes:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/srunet_ffno_ptychoblock_encoder_cdi_cns_smallcap_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/studies/index.md`

- [ ] Rebuild the item-local `comparison_bundle.json` from the corrected CDI and CNS run roots only.
- [ ] Rewrite the durable summary from `implementation_corrected_rerun_pending` to the actual completion outcome, with the exact corrected run roots, verification evidence, and claim boundary.
- [ ] Update `evidence_matrix.md` and `paper_evidence_index.md` so the corrected `24`-block roots are current authority and the historical `2`-block roots remain explicitly superseded context only.
- [ ] Update `model_variant_index.json` and `ablation_index.json` so the corrected row entries point at the fresh corrected roots while preserving historical lineage notes.
- [ ] Update `docs/studies/index.md` so the cross-pillar probe entry no longer describes the corrected reruns as pending once both corrected roots exist.
- [ ] Update `docs/index.md` only if a reusable new study surface was added, and update `docs/findings.md` only if the rerun exposed a durable new project rule.

**Verification**

- Blocking:
  - summary and index surfaces all point to the corrected run roots or state an
    explicit blocked outcome grounded in those reruns;
  - no current-authority evidence surface still treats the historical `2`-block
    roots as the approved result;
  - CDI is labeled append-only mechanism evidence and CNS is labeled bounded
    capped mechanism evidence.
- Supporting:
  - search/grep validation that the historical roots remain labeled
    `superseded`, `historical`, or equivalent rather than current authority;
  - if only docs/JSON surfaces changed in this task, no additional compile step
    is required.

## Completion Gate

This backlog item is complete only when all of the following are true:

- the fixed `24`-block encoder recipe is confirmed across the owned code paths;
- the required deterministic checks pass on the final code state;
- the corrected CDI rerun exits `0` and emits the full required artifact set;
- the corrected CNS rerun exits `0` and emits the full required artifact set;
- the rebuilt compare bundle, durable summary, and evidence/index surfaces all
  promote the corrected roots and demote the historical `2`-block roots to
  superseded lineage only;
- the final summary keeps CDI and CNS claim boundaries separate and does not
  promote either corrected `20`-epoch mechanism row into a headline authority.
