# SRU-Net FFNO-To-PtychoBlock Encoder CDI/CNS Small-Cap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement one encoder-only SRU-Net mechanism variant, `FFNO -> 2x(PtychoBlock + downsample)`, then evaluate exactly one fresh CDI row and one fresh capped CNS row without rerunning completed baselines.

**Architecture:** Add one explicit architecture/profile family, preferably `hybrid_resnet_ffno_ptychoblock_encoder`, that keeps the existing SRU-Net shell after the encoder unchanged. The encoder should be a fixed small FFNO-first stack at lifted resolution followed by two shape-preserving `PtychoBlock` stages paired with the existing two downsample stages; all downstream bottleneck, decoder, skip wiring, losses, schedules, seeds, and metric contracts remain fixed. Implementation must append evidence and lineage to existing authorities rather than mutating the completed Lines128 or CNS headline bundles.

**Tech Stack:** PATH `python`, long runs in `ptycho311`, PyTorch/Lightning, `ptycho_torch/generators/*`, `ptycho_torch/model.py`, `ptycho/config/config.py`, `ptycho_torch/workflows/components.py`, `scripts/studies/grid_lines_*`, `scripts/studies/pdebench_image128/*`, Markdown/JSON evidence indexes.

---

## Selected Objective

- Add one fresh CDI row on the locked `lines128` contract:
  `pinn_hybrid_resnet_ffno_ptychoblock_encoder` unless a clearer explicit name is justified and used consistently.
- Add one fresh capped CNS row on the matched-condition headline lane:
  `hybrid_resnet_ffno_ptychoblock_encoder_cns` unless a clearer explicit profile id is justified and used consistently.
- Answer one causal question only: does replacing the current SRU-Net encoder with `FFNO -> 2x(PtychoBlock + downsample)` help or hurt the same downstream SRU-Net body on CDI and on capped CNS?

## Scope And Explicit Non-Goals

### In Scope

- Implement one encoder-profile family shared across CDI and PDEBench image128 entry points.
- Keep the SRU-Net shell outside the encoder fixed:
  same input/lifter policy, two downsample stages, bottleneck family, bottleneck width/depth, decoder family, skip structure, residual scaling, output mode, loss, scheduler, seed policy, visual sample policy, and metric schema.
- Reuse completed baselines by lineage only:
  - CDI: `pinn_hybrid_resnet`, `pinn_hybrid_resnet_encoder_spectral_only`, `pinn_ffno`, `pinn_hybrid_resnet_ffno_bottleneck`.
  - CNS: `author_ffno_cns_base`, `spectral_resnet_bottleneck_base`, `fno_base`, `unet_strong` under the matched `h5_512_64_64_40ep` lane.
- Emit a durable append-only summary plus discoverability/index updates.

### Non-Goals

- Do not rerun completed baselines just to assemble comparisons.
- Do not tune FFNO encoder depth, modes, sharing, gates, normalization, or MLP ratio after seeing the first metrics.
- Do not change decoder skip wiring, bottleneck family, residual scaling, losses, probe/data preprocessing, split policy, epoch budgets, or metric definitions while claiming this is an encoder-only ablation.
- Do not average CDI and CNS into one ranking or promote this item into a new default SRU-Net family.
- Do not rewrite the roadmap, reopen full-training CNS claims, or overwrite the completed Lines128 paper bundle.

## Steering, Roadmap, And Prerequisite Constraints

- Steering keeps the current selection window within Roadmap Phase 2 plus Phase 3 CDI-preparation work. This item is allowed because it strengthens core comparison evidence without opening later phases.
- Equal-footing and fairness constraints are binding. If the encoder variant cannot stay on the locked CDI or matched CNS contracts, record the incompatibility instead of silently relaxing the protocol.
- Keep CDI and CNS conclusions separate. The CNS result remains `bounded_capped_decision_support_only` even if the row is strong.
- Progress-ledger status relevant to this item:
  - completed tranches include Phase 0 evidence inventory and Phase 1 PDE benchmark selection;
  - no blocked tranches are recorded globally;
  - this item depends on completed backlog authorities rather than unresolved roadmap blockers.
- Prerequisite authorities that must remain the comparison anchors:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_srunet_branch_objective_ablation_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md`
- Long-run ownership rule:
  - use `tmux` for training/evaluation launches;
  - activate `ptycho311`;
  - track the launched PID/session until exit code `0`;
  - do not start a duplicate writer against the same output root;
  - treat a run as complete only when exit `0` and the required fresh row artifacts both exist.
- Failure policy:
  - diagnose/fix/rerun for normal import, shape, config, path, harness, or test failures;
  - reserve `BLOCKED` for missing data/hardware, unavailable external dependency outside current authority, roadmap conflict, or an unrecoverable failure after a documented narrow fix attempt.

## Implementation Architecture

- **Shared encoder core:** add a dedicated Hybrid ResNet variant that preserves the existing SRU-Net shell but swaps the encoder for a fixed `ffno_encoder_blocks=2`, `ffno_encoder_modes=12`, `ffno_encoder_share_weights=true`, `ffno_encoder_gate_init=0.1`, `ffno_encoder_norm="instance"`, `ffno_encoder_mlp_ratio=2.0` stack followed by exactly two `PtychoBlock` stages and the existing two downsample steps. Do not enable any extra FFNO local-conv branch for this item.
- **CDI integration unit:** register the new architecture end-to-end for grid-lines compare-wrapper execution, checkpoint/config reconstruction, and row-manifest metadata so the fresh row behaves like other `pinn_*` architecture additions.
- **CNS integration unit:** add one manual-only PDEBench image128 profile that builds the same encoder variant under the canonical CNS shell (`history_len=5`, `512 / 64 / 64`, `40` epochs, MSE, batch size `4`, Adam `2e-4`) and keeps it out of default profile bundles.

## File And Artifact Targets

### Mandatory Code Targets

- Modify: `ptycho_torch/generators/hybrid_resnet.py`
- Modify: `ptycho_torch/generators/registry.py`
- Modify: `ptycho_torch/model.py`
- Modify if architecture enums/validation need extension: `ptycho/config/config.py`
- Modify if override round-trip needs extension: `ptycho_torch/workflows/components.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: `scripts/studies/pdebench_image128/models.py`
- Modify: `scripts/studies/pdebench_image128/run_config.py`

### Mandatory Test Targets

- Modify or add focused generator/shape tests:
  - `tests/torch/test_fno_generators.py`
  - `tests/torch/test_generator_registry.py`
  - `tests/torch/test_lightning_checkpoint.py`
- Modify runner/wrapper tests:
  - `tests/torch/test_grid_lines_torch_runner.py`
  - `tests/test_grid_lines_compare_wrapper.py`
- Modify PDEBench profile/model tests:
  - `tests/studies/test_pdebench_image128_models.py`
- Prefer adding focused PDEBench runner coverage if profile selection/row collation changes:
  - `tests/studies/test_pdebench_image128_runner.py`

### Mandatory Contract Outputs

- Item artifact root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/`
- Fresh CDI row root under the item run root with:
  invocation/config/history/metrics/reconstruction/visuals and `exit_code_proof.json`
- Fresh CNS row root under the item run root with:
  invocation/config/history/metrics/field visuals and `exit_code_proof.json`
- Durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/srunet_ffno_ptychoblock_encoder_cdi_cns_smallcap_summary.md`

### Preferred Packaging And Discoverability Updates

- Update:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/studies/index.md`
- Only update `docs/index.md` or `docs/findings.md` if implementation creates a new durable study entry point or exposes a reusable project-wide rule rather than an item-local result.

## Execution Checklist

### Task 1: Freeze Authorities, Contracts, And Lineage Inputs

- [ ] Run the backlog-item prerequisite presence check exactly as written before changing code.
- [ ] Confirm the CDI comparison contract from `lines128_paper_benchmark_summary.md`: `N=128`, `gridsize=1`, `seed=3`, `40` epochs, fixed sample ids `0` and `1`, `pad_extrapolate`, Run1084 probe lineage, MAE, `ReduceLROnPlateau`.
- [ ] Confirm the CNS comparison contract from `pdebench_cns_matched_condition_table_refresh_summary.md`: `h5_512_64_64_40ep`, `history_len=5`, `512 / 64 / 64`, `40` epochs, batch size `4`, MSE, matched-condition capped boundary.
- [ ] Record the exact lineage summary/artifact roots for the reused baseline rows in the item execution notes so later collation does not infer them ad hoc.

Verification:

- Blocking: run the backlog-item prerequisite command exactly as provided in `selected-item-context.md`.
- Supporting: a short execution note under the item artifact root listing the reused CDI and CNS authority roots.

### Task 2: Implement The Shared Encoder Variant And Metadata Contract

- [ ] Add one explicit architecture id, preferably `hybrid_resnet_ffno_ptychoblock_encoder`.
- [ ] Implement the FFNO-first encoder inside the Hybrid shell without changing baseline `hybrid_resnet` behavior.
- [ ] Keep the encoder recipe fixed for this item:
  `ffno_encoder_blocks=2`, `ffno_encoder_modes=12`, `ffno_encoder_share_weights=true`, `ffno_encoder_gate_init=0.1`, `ffno_encoder_norm="instance"`, `ffno_encoder_mlp_ratio=2.0`, `ptychoblock_stage_count=2`, `downsample_steps=2`, `downsample_op` inherited from the existing shell.
- [ ] Ensure the required manifest/config fields are emitted and reconstructable:
  `encoder_variant`, `ptychoblock_stage_count`, `downsample_steps`, `downsample_op`, `ffno_encoder_blocks`, `ffno_encoder_modes`, `ffno_encoder_share_weights`, `ffno_encoder_gate_init`, `ffno_encoder_norm`, `ffno_encoder_mlp_ratio`.
- [ ] Extend registry, checkpoint rebuild, and config validation surfaces only as far as needed for the new architecture id and metadata round-trip.

Verification:

- Blocking: `pytest -q tests/torch/test_fno_generators.py -k "PtychoBlock or hybrid_resnet or ffno"`
- Blocking: `python -m compileall -q ptycho_torch`
- Supporting: `pytest -q tests/torch/test_generator_registry.py`
- Supporting: `pytest -q tests/torch/test_lightning_checkpoint.py -k "hybrid_resnet or ffno"`

### Task 3: Wire The Fresh CDI Row

- [ ] Register `pinn_hybrid_resnet_ffno_ptychoblock_encoder` in the grid-lines compare-wrapper row spec table with append-only row status and explicit architecture mapping.
- [ ] Thread the new architecture through `grid_lines_torch_runner.py` argument validation, config serialization, invocation reconstruction, labels, and parameter-count reporting.
- [ ] Preserve the fixed Lines128 contract and baseline provenance behavior; only the new row is fresh.
- [ ] Make sure the row-local manifest and any run-level manifest include the encoder recipe fields and the correct append-only claim boundary.

Verification:

- Blocking: `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno"`
- Blocking: `pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or ffno"`
- Supporting: `python -m compileall -q scripts/studies`

### Task 4: Wire The Fresh CNS Profile

- [ ] Add one manual-only profile id, preferably `hybrid_resnet_ffno_ptychoblock_encoder_cns`, to `scripts/studies/pdebench_image128/run_config.py`.
- [ ] Build the profile under the canonical CNS shell in `scripts/studies/pdebench_image128/models.py`, preserving two downsample steps, skip-add, pixelshuffle upsampler, and the existing supervised real-channel adapter semantics.
- [ ] Keep the new profile out of default `required_primary_profiles_for_task()` bundles so it only runs when explicitly requested by profile id.
- [ ] Ensure `describe_model()` / profile manifests expose the fixed encoder recipe fields.

Verification:

- Blocking: `pytest -q tests/studies/test_pdebench_image128_models.py -k "hybrid_resnet or ffno"`
- Blocking: `python -m compileall -q scripts/studies`
- Supporting: add and run a focused selector in `tests/studies/test_pdebench_image128_runner.py` if the new profile changes manual-only collation or reference-row selection.

### Task 5: Run Deterministic Code Gates Before Any Expensive Launch

- [ ] Re-run all backlog-item deterministic checks after code changes; these remain required, not optional:
  - prerequisite presence script
  - `pytest -q tests/torch/test_fno_generators.py -k "PtychoBlock or hybrid_resnet or ffno"`
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno"`
  - `pytest -q tests/studies/test_pdebench_image128_models.py -k "hybrid_resnet or ffno"`
  - `python -m compileall -q ptycho_torch scripts/studies`
- [ ] Run the stronger narrow additions below because this item is expected to touch compare-wrapper routing and checkpoint/profile round-trip, and the backlog list does not cover those surfaces directly.

Verification:

- Blocking: every backlog-item `check_command` above must pass before launching CDI or CNS training.
- Blocking: `pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or ffno"`
- Supporting: `pytest -q tests/torch/test_generator_registry.py`
- Supporting: `pytest -q tests/torch/test_lightning_checkpoint.py -k "hybrid_resnet or ffno"`

### Task 6: Launch The Fresh CDI Row

- [ ] Use the existing grid-lines compare-wrapper path to launch only the new CDI row on the fixed `lines128` contract.
- [ ] Run in `tmux` with `ptycho311`, keep the launch under implementation ownership, and do not overlap it with another GPU-heavy run on the single RTX 3090.
- [ ] If the harness supports a cheap preflight or dry-run for the exact row wiring, it may run first as a supporting check only; the item still requires the full locked `40`-epoch row.
- [ ] On completion, confirm fresh row-local `metrics.json`, reconstruction NPZ, fixed-sample visuals, invocation/config/history, and `exit_code_proof.json`.

Verification:

- Blocking: tracked row launch exits `0`.
- Blocking: fresh CDI row artifacts exist under the item root and are newer than the launch start.
- Supporting: a row-local audit note capturing the exact reused baseline lineage roots referenced for comparison.

### Task 7: Launch The Fresh CNS Row

- [ ] Launch only the new manual CNS profile on `2d_cfd_cns` using the matched `h5_512_64_64_40ep` contract with `history_len=5`.
- [ ] Keep the row on the existing CNS MSE recipe and matched metric family; do not mix caps, history lengths, or epoch budgets.
- [ ] A shorter smoke/pilot launch is allowed only to prove implementation viability before the full run or when debugging a recoverable failure. It must be labeled readiness-only and cannot satisfy the item's CNS impact result.
- [ ] On completion, confirm fresh row-local `metrics_<profile>.json`, model profile JSON, field-visual artifacts, invocation/config/history, and `exit_code_proof.json`.

Verification:

- Blocking: tracked `40`-epoch CNS row launch exits `0`, or a documented unrecoverable external blocker is recorded after a narrow fix attempt.
- Blocking: fresh CNS row artifacts exist under the item root and the summary explicitly labels the result `bounded_capped_decision_support_only`.
- Supporting: if a smoke/pilot run was needed, archive it separately and label it readiness-only.

### Task 8: Collate Append-Only Evidence And Update Discoverability

- [ ] Build an item-local comparison bundle that contains only the two fresh rows plus lineage references to the reused CDI and CNS baselines.
- [ ] Write `srunet_ffno_ptychoblock_encoder_cdi_cns_smallcap_summary.md` with separate CDI and CNS sections, explicit claim boundary language, row lineage, encoder recipe fields, and domain-dependent interpretation if the variant helps one benchmark and hurts the other.
- [ ] Update `evidence_matrix.md`, `paper_evidence_index.md`, `ablation_index.json`, `model_variant_index.json`, and `docs/studies/index.md` so the new rows and summary are discoverable.
- [ ] Keep the paper-evidence update bounded: this item is mechanism evidence only and must not replace the current Lines128 or CNS headline authorities.

Verification:

- Blocking: summary exists and names the fresh row ids, reused lineage rows, fixed CDI contract, fixed CNS contract, and claim boundary.
- Blocking: evidence/index updates exist for every new durable output or the summary explicitly states why a given index surface was intentionally unchanged.
- Supporting: if implementation discovers a reusable repo-wide lesson, add the minimal `docs/findings.md` or `docs/index.md` update in the same pass.

