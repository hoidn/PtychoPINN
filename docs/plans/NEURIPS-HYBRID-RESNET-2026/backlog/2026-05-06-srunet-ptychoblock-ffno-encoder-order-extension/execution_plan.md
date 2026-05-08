# SRU-Net PtychoBlock-To-FFNO Encoder-Order Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the reversed-order SRU-Net encoder row `2x(PtychoBlock + downsample) -> shared-weight 24-layer FFNO stack -> unchanged SRU-Net bottleneck/decoder`, run it once on the locked `lines128` CDI contract and once on the matched-condition capped CNS contract, then publish the three-row encoder-order comparison against regular SRU-Net and the completed corrected `FFNO -> PtychoBlock` companion row.

**Architecture:** Reuse the existing corrected `FFNO -> PtychoBlock` implementation as the template, but keep the only causal change to encoder ordering. Extract or reuse one shared no-refiner FFNO stack helper so the end-to-end CDI FFNO generator, the existing companion row, and the new reversed-order row all consume the same `FactorizedFfnoBlock`/`FactorizedSpectralConv2d` stack contract. Wire the new architecture through the CDI runner, compare wrapper, supervised CNS model/profile surfaces, then launch exactly two fresh `20`-epoch rows under tracked tmux ownership and rebuild the summary/index surfaces from those fresh roots only.

**Tech Stack:** PATH `python`, long runs in `ptycho311`, tmux-backed launch ownership, PyTorch/Lightning, `ptycho_torch/generators/*`, `ptycho_torch/model.py`, `scripts/studies/grid_lines_*`, `scripts/studies/pdebench_image128/*`, Markdown/JSON evidence indexes, `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/...`.

---

## Selected Objective

- Implement one explicit reversed-order architecture family:
  - CDI architecture id: `hybrid_resnet_ptychoblock_ffno_encoder`
  - CDI model id: `pinn_hybrid_resnet_ptychoblock_ffno_encoder`
  - CNS profile id: `hybrid_resnet_ptychoblock_ffno_encoder_cns`
- Produce one fresh CDI row on `cdi_lines128_seed3`.
- Produce one fresh capped CNS row on the matched `history_len=5`, `512 / 64 / 64`, `20`-epoch mechanism lane.
- Publish benchmark-separated encoder-order conclusions for:
  - regular SRU-Net;
  - corrected `FFNO -> 2x(PtychoBlock + downsample)` companion row;
  - new `2x(PtychoBlock + downsample) -> FFNO` row.

## Scope And Explicit Non-Goals

### In Scope

- Add the reversed-order SRU-Net encoder path while preserving the existing SRU-Net shell outside encoder ordering.
- Reuse the same local FFNO stack contract as the end-to-end CDI `FfnoGeneratorModule`, with no output-side local residual refiners in the encoder-order row.
- Register the new row through CDI and CNS study entrypoints, manifests, and checkpoint reconstruction.
- Run exactly two new rows:
  - `pinn_hybrid_resnet_ptychoblock_ffno_encoder` on locked CDI `lines128`;
  - `hybrid_resnet_ptychoblock_ffno_encoder_cns` on locked capped CNS.
- Update the durable summary and evidence/discoverability indexes for this new mechanism-evidence row family.

### Explicit Non-Goals

- Do not redesign SRU-Net, FFNO, or the broader Hybrid ResNet family.
- Do not replace the SRU-Net bottleneck with FFNO. The FFNO stack belongs after the two encoder/downsample stages and before the unchanged bottleneck.
- Do not change skip wiring, skip fusion mode, bottleneck family, decoder family, residual-scaling policy, loss, scheduler, seed policy, probe/data contract, CNS history length, CNS split cap, or metric definitions while claiming encoder-order isolation.
- Do not add the end-to-end FFNO generator's local residual refiners to the new encoder-order row.
- Do not tune hyperparameters after seeing metrics.
- Do not rerun completed baseline rows or the completed corrected companion row merely to assemble the comparison.
- Do not promote this work into the paper-grade CDI authority or the `40`-epoch CNS headline lane.
- Do not reopen later roadmap phases, optional candidate benchmarks, or unrelated backlog items.
- Do not create worktrees.

## Binding Steering, Roadmap, And Policy Constraints

- Steering keeps this item inside the current Phase 2 plus Phase 3 selection window because it strengthens core paper comparison evidence; do not expand scope beyond that window.
- Equal-footing constraints are binding. If the reversed-order row cannot stay on the locked CDI or CNS contract, record the incompatibility instead of silently relaxing the protocol.
- Keep CDI and CNS conclusions separate. Do not average or otherwise collapse them into one overall ranking.
- The current headline authorities remain unchanged:
  - CDI headline: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - CNS headline: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md`
- The new CDI row is `decision_support_append_only`.
- The new CNS row is `bounded_capped_decision_support_only`.
- Long-running commands remain under implementation ownership until completion:
  - launch in tmux;
  - activate `ptycho311`;
  - track the exact launched PID/session;
  - do not launch a duplicate writer to the same output root;
  - only treat a run as complete when the tracked launch exits `0` and the required fresh artifacts exist.
- Do not mark the item `BLOCKED` for ordinary import, path, config, environment, test-harness, or shape failures. Diagnose, fix narrowly, rerun the affected checks, and relaunch first. Reserve `BLOCKED` for missing data/hardware, roadmap conflict, external dependency outside current authority, required user decision, or unrecoverable failure after a documented narrow fix attempt.
- Keep PyTorch and config-bridge guardrails active:
  - PATH `python` only;
  - preserve `update_legacy_dict(params.cfg, config)` ordering wherever legacy modules still depend on it;
  - preserve active findings and parity rules from `docs/findings.md`.

## Prerequisite Status And Current Starting State

- Progress-ledger status:
  - `phase-0-evidence-inventory` completed;
  - `phase-1-pde-benchmark-selection` completed;
  - no global blocked tranches are recorded in the consumed ledger.
- The corrected companion row is already available and must now be consumed as comparison authority:
  - summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/srunet_ffno_ptychoblock_encoder_cdi_cns_smallcap_summary.md`
  - corrected CDI root: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/runs/cdi_ffno_ptychoblock_encoder_20260507T073814Z/`
  - corrected CNS root: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/runs/cns_ffno_ptychoblock_encoder_20260507T082701Z/`
- Regular SRU-Net comparison authorities already exist by lineage:
  - CDI baseline row authority: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - CDI mechanistic lineage: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_srunet_branch_objective_ablation_summary.md`
  - CNS matched-condition authority: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md`
- The reversed-order architecture is not yet wired in current code. The existing companion `hybrid_resnet_ffno_ptychoblock_encoder` path is present across generator, runner, compare-wrapper, PDEBench model/profile, registry, and checkpoint surfaces; use those seams as the extension template.

### Fixed CDI Contract To Preserve

- dataset contract id: `cdi_lines128_seed3`
- `N=128`, `gridsize=1`, `seed=3`
- `20` epochs, `batch_size=16`, Adam `2e-4`
- `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-4, threshold=0.0)`
- `torch_loss_mode=mae`, `torch_output_mode=real_imag`
- fixed sample ids `0`, `1`
- comparison rows reused by lineage:
  - `pinn_hybrid_resnet`
  - `pinn_hybrid_resnet_ffno_ptychoblock_encoder`
  - `pinn_hybrid_resnet_encoder_spectral_only`
  - `pinn_ffno`
  - `pinn_hybrid_resnet_ffno_bottleneck`

### Fixed CNS Contract To Preserve

- authority lane: matched-condition `history_len=5`, `512 / 64 / 64`, `40`-epoch headline bundle for lineage reuse
- new row run budget remains the bounded `20`-epoch mechanism lane
- `history_len=5`
- split counts `512 / 64 / 64`
- `20` epochs, `batch_size=4`, Adam `2e-4`
- `max_windows_per_trajectory=8`
- training loss `mse`
- metric family:
  `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- comparison rows reused by lineage:
  - `spectral_resnet_bottleneck_base`
  - `author_ffno_cns_base`
  - `fno_base`
  - `unet_strong`
  - `hybrid_resnet_ffno_ptychoblock_encoder_cns`

### Fixed Encoder Recipe To Preserve

- architecture id: `hybrid_resnet_ptychoblock_ffno_encoder`
- `encoder_variant=ptychoblock_ffno_encoder`
- `encoder_order=ptychoblock_then_ffno`
- `ptychoblock_stage_count=2`
- `downsample_steps=2`
- `downsample_op=stride_conv`
- `ffno_encoder_blocks=24`
- `ffno_encoder_modes=12`
- `ffno_encoder_share_weights=true`
- `ffno_encoder_gate_init=0.1`
- `ffno_encoder_norm=instance`
- `ffno_encoder_mlp_ratio=2.0`
- `local_conv_kernel_size=None`
- Keep `ffno_encoder_modes=12` unless the implementation documents a strict post-downsample shape-compatibility reason for a smaller value. Any such adjustment must be recorded in manifests and the durable summary as compatibility-only, not tuning.

## Implementation Architecture

- **Shared FFNO stack unit:** expose one reusable no-refiner FFNO stack helper shared by the CDI FFNO generator and both encoder-order SRU-Net variants so the local FFNO stack contract is defined once.
- **Reversed-order SRU-Net unit:** add one new generator/module path that preserves the SRU-Net shell but moves the FFNO stack from pre-encoder to post-downsample encoder output.
- **Study routing unit:** register the new CDI row, CNS profile, manifest fields, paper labels, and checkpoint rebuild path without disturbing existing authorities.
- **Evidence unit:** run exactly two fresh rows, then publish one new durable summary plus index updates that compare regular, `FFNO -> PtychoBlock`, and `PtychoBlock -> FFNO` while keeping CDI and CNS boundaries separate.

## File And Artifact Targets

### Mandatory Contract Outputs

- Plan-owned artifact root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-srunet-ptychoblock-ffno-encoder-order-extension/`
- Fresh CDI run root under that item root, with invocation, config, history, metrics, visuals, `exit_code_proof.json`, and freshness evidence.
- Fresh CNS run root under that item root, with invocation, config, history, metrics, field visuals, `exit_code_proof.json`, and freshness evidence.
- Machine-readable comparison bundle:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-srunet-ptychoblock-ffno-encoder-order-extension/comparison_bundle.json`
- Archived verification logs under the item root, for example:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-srunet-ptychoblock-ffno-encoder-order-extension/verification/`
- Durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/srunet_encoder_order_ffno_vs_ptychoblock_summary.md`

### Likely Code Targets

- `ptycho_torch/generators/ffno_bottleneck.py`
- `ptycho_torch/generators/ffno.py`
- `ptycho_torch/generators/hybrid_resnet.py`
- `ptycho_torch/generators/registry.py`
- `ptycho_torch/model.py`
- `scripts/studies/grid_lines_torch_runner.py`
- `scripts/studies/grid_lines_compare_wrapper.py`
- `scripts/studies/pdebench_image128/models.py`
- `scripts/studies/pdebench_image128/run_config.py`

### Likely Test Targets

- `tests/torch/test_fno_generators.py`
- `tests/torch/test_generator_registry.py`
- `tests/torch/test_lightning_checkpoint.py`
- `tests/torch/test_grid_lines_torch_runner.py`
- `tests/studies/test_pdebench_image128_models.py`
- `tests/test_grid_lines_compare_wrapper.py`

### Required Documentation And Index Targets

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- `docs/studies/index.md`
- Update `docs/index.md` only if a materially new reusable study surface is added.
- Update `docs/findings.md` only if the work uncovers a durable project-wide rule rather than an item-local result.

## Required Deterministic Checks

These commands are mandatory for this item. Keep them unless a later approved plan revision replaces them with a stronger equivalent.

### Required Presence Gate

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/execution_plan.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/srunet_ffno_ptychoblock_encoder_cdi_cns_smallcap_summary.md"),
    Path("ptycho_torch/generators/ffno.py"),
    Path("ptycho_torch/generators/ffno_bottleneck.py"),
    Path("ptycho_torch/generators/hybrid_resnet.py"),
    Path("scripts/studies/grid_lines_torch_runner.py"),
    Path("scripts/studies/run_pdebench_image128_suite.py"),
    Path("scripts/studies/pdebench_image128/models.py"),
    Path("scripts/studies/pdebench_image128/run_config.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing SRU-Net PtychoBlock->FFNO encoder-order inputs: {missing}")
print("SRU-Net PtychoBlock->FFNO encoder-order inputs present")
PY
```

### Required Pytest And Compile Gates

```bash
pytest -q tests/torch/test_fno_generators.py -k "PtychoBlock or hybrid_resnet or ffno"
pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno"
pytest -q tests/studies/test_pdebench_image128_models.py -k "hybrid_resnet or ffno"
pytest -q tests/test_grid_lines_compare_wrapper.py -k "ffno or hybrid_resnet"
python -m compileall -q ptycho_torch scripts/studies
```

### Stronger Additional Gates

Use these as blocking checks when the corresponding production surfaces are touched, which is expected for this item:

```bash
pytest -q tests/torch/test_generator_registry.py
pytest -q tests/torch/test_lightning_checkpoint.py -k "hybrid_resnet or ffno"
pytest -v -m integration
```

## Execution Checklist

### Task 1: Implement The Shared FFNO Stack And Reversed-Order SRU-Net Architecture

**Files**

- `ptycho_torch/generators/ffno_bottleneck.py`
- `ptycho_torch/generators/ffno.py`
- `ptycho_torch/generators/hybrid_resnet.py`
- `ptycho_torch/generators/registry.py`
- `ptycho_torch/model.py`
- `tests/torch/test_fno_generators.py`
- `tests/torch/test_generator_registry.py`
- `tests/torch/test_lightning_checkpoint.py`

- [ ] Extract or name one shared no-refiner FFNO stack helper that is built from `FactorizedFfnoBlock` and `FactorizedSpectralConv2d`, and route `FfnoGeneratorModule` through it while keeping `_LocalResidualRefiner` outside the stack.
- [ ] Add the reversed-order SRU-Net module and generator wrapper:
  - regular lifter unchanged;
  - two ordinary `PtychoBlock` encoder stages with the existing downsampling schedule first;
  - shared-weight `24`-block FFNO stack after the second downsample and before the unchanged bottleneck;
  - unchanged decoder, skip taps, skip fusion, residual scaling, and output mode.
- [ ] Record the new architecture metadata on the module/wrapper:
  `encoder_variant=ptychoblock_ffno_encoder`,
  `encoder_order=ptychoblock_then_ffno`,
  `ptychoblock_stage_count=2`,
  `downsample_steps=2`,
  `downsample_op=stride_conv`,
  plus the fixed FFNO fields.
- [ ] Add the new architecture to generator registry and checkpoint reconstruction so direct load/rebuild works without raw implementation knowledge.

**Verification**

- Blocking:
  - required presence gate passes;
  - `pytest -q tests/torch/test_fno_generators.py -k "PtychoBlock or hybrid_resnet or ffno"`
  - `pytest -q tests/torch/test_generator_registry.py`
  - `pytest -q tests/torch/test_lightning_checkpoint.py -k "hybrid_resnet or ffno"`
  - `python -m compileall -q ptycho_torch scripts/studies`
- Supporting:
  - capture any architecture/shape rationale for mode-count adjustment if one is required;
  - if no adjustment is needed, record that the default `ffno_encoder_modes=12` remained valid.

### Task 2: Wire CDI And CNS Study Entry Points For The New Row

**Files**

- `scripts/studies/grid_lines_torch_runner.py`
- `scripts/studies/grid_lines_compare_wrapper.py`
- `scripts/studies/pdebench_image128/models.py`
- `scripts/studies/pdebench_image128/run_config.py`
- `tests/torch/test_grid_lines_torch_runner.py`
- `tests/studies/test_pdebench_image128_models.py`
- `tests/test_grid_lines_compare_wrapper.py`

- [ ] Add the new paper label and architecture route for `pinn_hybrid_resnet_ptychoblock_ffno_encoder`.
- [ ] Add runner-side fixed-recipe manifest emission for the new row, including:
  `encoder_variant`,
  `encoder_order`,
  `ptychoblock_stage_count`,
  `downsample_steps`,
  `downsample_op`,
  `ffno_encoder_blocks`,
  `ffno_encoder_modes`,
  `ffno_encoder_share_weights`,
  `ffno_encoder_gate_init`,
  `ffno_encoder_norm`,
  `ffno_encoder_mlp_ratio`.
- [ ] Add the new supervised PDEBench image model path and manual-only profile `hybrid_resnet_ptychoblock_ffno_encoder_cns`, preserving the locked CNS shell outside encoder ordering.
- [ ] Keep the new row append-only/manual-only in discovery surfaces; it must not enter any default primary bundle automatically.

**Verification**

- Blocking:
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno"`
  - `pytest -q tests/studies/test_pdebench_image128_models.py -k "hybrid_resnet or ffno"`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "ffno or hybrid_resnet"`
  - `pytest -v -m integration`
- Supporting:
  - if runner-specific code paths beyond the listed selectors are touched, run the narrowest additional selector that directly exercises that surface and archive the log beside the mandatory ones.

### Task 3: Run The New CDI Row On The Locked `lines128` Contract

**Files / Artifacts**

- launch surfaces:
  - `scripts/studies/grid_lines_compare_wrapper.py`
  - `scripts/studies/grid_lines_torch_runner.py`
- fresh run root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-srunet-ptychoblock-ffno-encoder-order-extension/runs/cdi_ptychoblock_ffno_encoder_<timestampZ>/`

- [ ] Launch exactly one new CDI row: `pinn_hybrid_resnet_ptychoblock_ffno_encoder`.
- [ ] Keep the inherited CDI shell fixed:
  `20` epochs, `batch_size=16`, Adam `2e-4`, MAE loss, `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-4, threshold=0.0)`, fixed sample ids `0` and `1`, standard `real_imag` output mode, fixed-probe `lines128` lineage unchanged.
- [ ] Launch in tmux under `ptycho311`, track the exact launched PID/session, and wait for terminal completion. Do not start a second writer to the same output root.
- [ ] If the run fails for an ordinary code or harness reason, diagnose, fix narrowly, rerun the affected blocking gates, and relaunch under a fresh timestamped root.
- [ ] Mark the CDI row complete only when the tracked launch exits `0` and the fresh root contains invocation/config/history/metrics/visuals plus `exit_code_proof.json` and freshness evidence.

**Verification**

- Blocking:
  - tracked CDI launch exits `0`;
  - fresh CDI root is newly written under the current item artifact root;
  - emitted metadata records `encoder_order=ptychoblock_then_ffno` and the rest of the fixed recipe;
  - artifacts are sufficient to compare against regular SRU-Net and the completed companion row by lineage.
- Supporting:
  - record short item-local notes on whether the reversed-order row helps or hurts relative to `pinn_hybrid_resnet` and the companion `pinn_hybrid_resnet_ffno_ptychoblock_encoder`;
  - keep that interpretation out of the final headline authorities.

### Task 4: Run The New Capped CNS Row On The Matched-Condition Mechanism Lane

**Files / Artifacts**

- launch surfaces:
  - `scripts/studies/run_pdebench_image128_suite.py`
  - `scripts/studies/pdebench_image128/models.py`
  - `scripts/studies/pdebench_image128/run_config.py`
- fresh run root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-srunet-ptychoblock-ffno-encoder-order-extension/runs/cns_ptychoblock_ffno_encoder_<timestampZ>/`

- [ ] Launch exactly one new CNS row: `hybrid_resnet_ptychoblock_ffno_encoder_cns`.
- [ ] Keep the inherited CNS shell fixed:
  `history_len=5`, split `512 / 64 / 64`, `20` epochs, `batch_size=4`, Adam `2e-4`, `max_windows_per_trajectory=8`, `mse` training loss, `pixelshuffle` upsampler, skip-add SRU-Net shell, and the same metric family.
- [ ] Label the run explicitly as a bounded `20`-epoch mechanism probe on the matched-condition capped lane. It is not a new headline row.
- [ ] Launch in tmux under `ptycho311`, track the exact launched PID/session, and wait for terminal completion. Do not start a second writer to the same output root.
- [ ] If the run fails for an ordinary code or harness reason, diagnose, fix narrowly, rerun the affected blocking gates, and relaunch under a fresh timestamped root.
- [ ] Mark the CNS row complete only when the tracked launch exits `0` and the fresh root contains invocation/config/history/metrics/field visuals plus `exit_code_proof.json` and freshness evidence.

**Verification**

- Blocking:
  - tracked CNS launch exits `0`;
  - fresh CNS root is newly written under the current item artifact root;
  - emitted metadata records the fixed encoder recipe and the bounded capped CNS contract;
  - row output stays segregated from the `40`-epoch matched-condition headline bundle.
- Supporting:
  - compare against same-contract lineage rows in the summary, but label any comparison to the `40`-epoch headline lane as directional context only.

### Task 5: Publish The Encoder-Order Summary And Repair Discoverability Surfaces

**Files / Artifacts**

- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-srunet-ptychoblock-ffno-encoder-order-extension/comparison_bundle.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/srunet_encoder_order_ffno_vs_ptychoblock_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- `docs/studies/index.md`

- [ ] Build `comparison_bundle.json` from:
  - reused regular SRU-Net lineage rows;
  - the completed corrected companion row;
  - the two new reversed-order roots from this item.
- [ ] Write the durable summary so it contains two benchmark-separated three-row encoder-order tables whenever the reused companion and regular rows remain readable under the same contract.
- [ ] Because the corrected companion row is already available, treat a missing three-row comparison as exceptional. If an upstream lineage artifact becomes unreadable or incompatible during execution, record a precise row-level blocker in the summary instead of blocking the whole item.
- [ ] Update `evidence_matrix.md`, `paper_evidence_index.md`, `model_variant_index.json`, and `ablation_index.json` so the new reversed-order rows are discoverable with the correct claim boundary and so the companion/current headline authorities remain unchanged.
- [ ] Update `docs/studies/index.md` so the SRU-Net encoder-order probe is discoverable as a paired cross-pillar mechanism study rather than a pending extension.
- [ ] Update `docs/index.md` only if implementation created a materially new reusable study surface, and update `docs/findings.md` only if the work exposed a durable new project rule.

**Verification**

- Blocking:
  - the summary cites the fresh CDI and CNS roots from this item;
  - the summary preserves separate CDI and CNS interpretations;
  - the summary identifies the new row as encoder-order ablation evidence, not as a new default architecture family;
  - all updated discovery surfaces point to the correct summary and claim boundaries.
- Supporting:
  - if the new work reveals no durable study-surface change, state explicitly in the execution report that `docs/index.md` did not need an update;
  - if no project-wide rule emerged, state explicitly that `docs/findings.md` remained unchanged.

## Completion Gate

- The new row differs from the completed corrected companion row only by encoder ordering and any documented compatibility adapter needed to place the FFNO stack after downsampling.
- The durable summary publishes the three-row encoder-order comparison for CDI and CNS when the regular and companion same-contract rows are available, which they are at plan approval time.
- Any missing or incompatible comparison input is represented as a precise row-level blocker, not as a rerun of unrelated baselines.
- The CDI and CNS rows remain append-only mechanism evidence and do not overwrite current paper-grade or headline authorities.
