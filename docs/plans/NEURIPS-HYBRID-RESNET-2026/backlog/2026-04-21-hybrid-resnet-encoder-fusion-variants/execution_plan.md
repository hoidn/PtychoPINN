# Lines128 Hybrid ResNet Encoder Fusion Variants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement this plan task-by-task. Keep this file as the execution authority for the selected backlog item.

**Goal:** Quantify whether tighter encoder-branch update control improves `hybrid_resnet` on the fixed `lines128` CDI contract by running the minimum same-contract ablation set for encoder LayerScale, branch-gated fusion, and their combination, then publish an append-only ablation bundle and durable summary without rewriting the completed six-row CDI benchmark.

**Architecture:** Reuse the authoritative `lines128` complete-table bundle as the immutable baseline contract and baseline-row provenance source, keep the current identity residual path `x + update(x)` unchanged, and add Torch-only encoder-fusion controls that are isolated to the Hybrid ResNet encoder block plus runner/workflow plumbing. The first scored pass uses **per-block** learned scalars for all new encoder controls so the item is not satisfied by shared-only scalar placement; shared-scope controls and normalized fusion remain explicit follow-up lanes only if the primary rows justify reopening them.

**Tech Stack:** Python 3.11, `ptycho311`, tmux-managed long runs, PyTorch/Lightning, `scripts/studies/grid_lines_torch_runner.py`, shared Lines128 paper-metric/visual collation helpers under `scripts/studies/`, Markdown/JSON/CSV/TeX evidence artifacts.

---

## Selected Backlog Objective

- Measure whether encoder-update LayerScale and encoder branch-gated fusion improve the fixed `lines128` CDI Hybrid ResNet contract relative to the current paper-grade `pinn_hybrid_resnet` row.
- Keep the encoder residual path as identity:
  - baseline and all variants must preserve `x_next = x + (...)`
  - no projected skip `W x`
- Publish a new append-only ablation root plus the required durable summary:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_encoder_fusion_variants_summary.md`

## Scope

- Fixed baseline authority:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - authoritative root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- Fixed contract authority:
  - same dataset identity, split, probe source/preprocessing, seed, optimizer, scheduler, loss, output mode, sample IDs, metrics, and visual schema as the accepted `pinn_hybrid_resnet` row in that root
  - core frozen fields that must not drift:
    - `N=128`
    - `gridsize=1`
    - synthetic grid-lines with `set_phi=True`
    - custom Run1084 probe with `probe_scale_mode=pad_extrapolate`
    - `nimgs_train=2`, `nimgs_test=2`
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
    - `probe_mask=off`
    - `fno_modes=12`
    - `fno_width=32`
    - `fno_blocks=4`
    - `fno_cnn_blocks=2`
    - `hybrid_encoder_conv_hidden_scale=2.0`
    - `hybrid_encoder_spectral_hidden_scale=1.0`
    - `hybrid_resnet_blocks=6`
- Remaining launch fields must be reconstructed from the authoritative baseline artifacts by field class, not copied wholesale from one recovered row config:
  - **Frozen semantic/model fields**:
    copy only the non-path model/training/runtime contract fields from:
    - `.../runs/pinn_hybrid_resnet/config.json`
    - `.../paper_benchmark_manifest.json`
    - `.../model_manifest.json`
  - **Canonical dataset-input paths**:
    source dataset paths from the authoritative complete-table bundle dataset records, not from the recovered minimum-subset row config:
    - `.../paper_benchmark_manifest.json` `dataset.train_npz`
    - `.../paper_benchmark_manifest.json` `dataset.test_npz`
    - `.../paper_benchmark_manifest.json` `dataset.gt_recon`
    - `.../dataset_identity_manifest.json` if additional dataset identity fields are needed
  - **Generated row-local output paths**:
    regenerate all row-local artifact destinations under the new append-only ablation root for every fresh row, including:
    - `output_dir`
    - `recon_npz`
    - row-local logs, checkpoints, manifests, and visual output directories
  - **Do not copy forward** these historical path-bearing keys from the recovered baseline-row config because they point at the older minimum-subset lineage:
    - `train_npz`
    - `test_npz`
    - `recon_npz`
    - `output_dir`
    - any wrapper-invocation or row-local artifact path that embeds the old run root
- Mandatory scored row set for this item:
  - reused baseline only: `pinn_hybrid_resnet`
  - fresh per-block LayerScale row: `pinn_hybrid_resnet_encoder_layerscale`
  - fresh per-block branch-gated row: `pinn_hybrid_resnet_encoder_branch_gated`
  - fresh per-block combined row: `pinn_hybrid_resnet_encoder_branch_gated_layerscale`
- Explicit scalar-scope decision for this first pass:
  - all fresh scored rows use **per-block** learned scalars
  - shared-across-encoder-block scalars are treated as a separate architecture axis and are not part of the mandatory first pass
- Optional follow-up only if the three mandatory fresh rows are complete and one primary row shows a real stitched-metric improvement:
  - `pinn_hybrid_resnet_encoder_fusion_norm`
  - optional shared-scope control rows only under a deliberate extension, not as an implicit substitution for the per-block rows

## Explicit Non-Goals

- Do not rewrite, rerun, or relabel the completed six-row `lines128` paper bundle.
- Do not broaden into PDEBench CNS, Darcy, SWE, OpenFWI, U-NO, FFNO-family work, bottleneck replacements, decoder skip-fusion changes, or `256x256` CDI scaling.
- Do not change the residual path from identity to a learned projection.
- Do not mix probe, loss-contract, scheduler, or dataset changes into this ablation.
- Do not satisfy this item with shared-only scalar gates or shared-only encoder LayerScale.
- Do not promote any fresh row into paper-grade headline evidence in this item.
- Do not mark the item `BLOCKED` for ordinary import, test, path, harness, or environment failures before a narrow diagnose/fix/rerun attempt is documented.

## Binding Constraints And Prerequisite Status

- Steering and roadmap boundary:
  - this is bounded Roadmap Phase 3 CDI evidence strengthening only
  - equal-footing comparison standards remain explicit and fixed
  - the output is append-only decision-support evidence and does not replace the completed CDI headline authority
- Defer-until-later clearance for the selected backlog item:
  - the backlog item's own stop clause is satisfied through its first resume condition: the campaign now needs one bounded `N=128` architecture follow-on after the CDI anchor replay work
  - treat that condition as cleared because repo-authoritative evidence now shows the anchor lane is already complete and draftable:
    - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_audit_summary.md` records the complete six-row `lines128` bundle as the current `paper_grade` CDI headline authority and says the CDI pillar is draftable now
    - `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/run_state.json` lists `2026-04-29-cdi-lines128-paper-benchmark-execution` and `2026-04-29-paper-evidence-package-audit` in `completed_items` before selecting this backlog item on `2026-05-02T09:56:54Z`
    - `docs/steering.md` preserves the current Phase 2 plus Phase 3 selection window, so one bounded CDI-strengthening ablation is allowed without reopening roadmap order or displacing the remaining PDE authority
  - implementation must carry this selection rationale into the manifest and durable summary so execution does not have to rediscover why the backlog item's original defer condition no longer applies
- Paper-package boundary:
  - no `/home/ollie/Documents/neurips/` outputs from this item
  - result publication stays in repo-local evidence indexes and the durable summary
- Long-run ownership rule:
  - once a training/compare launch starts, implementation owns the command until tracked success or documented recoverable failure handling is complete
  - use tmux, activate `ptycho311`, track the exact PID, and require both `exit_code=0` and fresh required artifacts before treating a launch as complete
- Prerequisite status:
  - selected-item prerequisite: `2026-04-29-cdi-lines128-paper-benchmark-execution`
  - the consumed `progress_ledger.json` only records early roadmap tranches and therefore is not the authority for later CDI backlog completions
  - current repo evidence surfaces do show the prerequisite satisfied:
    - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
    - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
    - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - implementation should therefore treat the completed Lines128 benchmark bundle as the active baseline authority for this ablation
- Encoder-control implementation rule:
  - keep new encoder-fusion knobs Torch-only study plumbing
  - do not expand canonical config-bridge or legacy `params.cfg` surfaces for these ablation knobs
  - if checkpoint rebuild or run-collation requires persistence of the new knobs, confine that persistence to the PyTorch/generator path and study manifests

## Implementation Architecture

- **Encoder-fusion control surface**
  - add a minimal, explicit encoder-control API for Hybrid ResNet blocks:
    - baseline fusion
    - per-block outer LayerScale
    - per-block spectral/local branch gates
    - combined per-block branch gates plus outer LayerScale
  - keep normalization follow-up out of the core implementation path unless the mandatory rows finish first
- **Torch-only plumbing and row orchestration**
  - thread the new controls through runner config, `PyTorchExecutionConfig`, payload/workflow overrides, and Hybrid generator construction without mutating TensorFlow-facing config contracts
  - prefer one narrow Lines128 ablation helper that reuses existing compare/collation utilities while keeping the completed paper bundle immutable
- **Evidence publication**
  - collate the reused baseline plus fresh rows into one append-only ablation root
  - publish one durable summary and synchronize the NeurIPS evidence indexes so future manuscript/planning tasks can discover the outcome without rereading backlog prose

## Concrete File And Artifact Targets

- Mandatory code surfaces:
  - `ptycho_torch/generators/hybrid_resnet.py`
  - `ptycho/config/config.py`
  - `ptycho_torch/config_params.py`
  - `ptycho_torch/model.py`
  - `ptycho_torch/workflows/components.py`
  - `scripts/studies/grid_lines_torch_runner.py`
- Preferred new or changed study-orchestration surface:
  - new narrow helper such as `scripts/studies/lines128_hybrid_resnet_encoder_fusion_variants.py`
  - only touch existing Lines128 paper-benchmark helpers if extraction is materially smaller and does not risk the accepted complete-table path
- Likely test surfaces:
  - `tests/torch/test_fno_generators.py`
  - `tests/torch/test_grid_lines_torch_runner.py`
  - `tests/studies/test_lines128_hybrid_resnet_encoder_fusion_variants.py` if a new helper is created
- Mandatory durable outputs:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_encoder_fusion_variants_summary.md`
  - append-only artifact root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/`
  - under that root:
    - `execution_manifest.json`
    - `row_contract_audit.json`
    - `comparison_summary.json`
    - `metrics.json`
    - `model_manifest.json`
    - `metrics_table.csv`
    - `metrics_table.tex`
    - `visuals/`
    - `verification/`
- Mandatory index/discoverability updates for a result-producing ablation item:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/studies/index.md`
- Preferred packaging only if it falls out naturally from shared helpers:
  - fixed-sample visual panels aligned to the completed Lines128 bundle
  - explicit row-status labels: reused baseline vs fresh variant

## Execution Checklist

### Tranche 1: Freeze The Baseline Authority And The Mandatory Variant Matrix

- [ ] Create the durable summary scaffold `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_encoder_fusion_variants_summary.md` with:
  - selected backlog item id and plan path
  - explicit resume-condition clearance note explaining why this item is selectable now
  - authoritative baseline root
  - intended reused/fresh row roster
  - explicit statement that the first scored pass uses per-block scalars
  - placeholder sections for results, claim boundary, deferred work, and index updates
- [ ] Create `execution_manifest.json` under the ablation artifact root with:
  - the same resume-condition clearance statement and evidence surfaces that justify executing this item now
  - authoritative baseline source root
  - reused baseline row id
  - mandatory fresh rows
  - optional normalized-fusion gate
  - scalar-scope decision
  - claim boundary text: same-contract CDI ablation, append-only, decision-support only
- [ ] Extract the baseline-row launch/config provenance and write `row_contract_audit.json` with separate sections for:
  - frozen semantic/model fields copied from the accepted baseline row
  - canonical dataset-input paths sourced from the complete-table bundle dataset records
  - generated row-local output/artifact paths that must be rebuilt under the new ablation root
  - an explicit denylist of historical path-bearing keys that must not be copied from the recovered baseline-row config
- [ ] Record the row interpretations explicitly:
  - `pinn_hybrid_resnet_encoder_layerscale`: per-block outer update scale only
  - `pinn_hybrid_resnet_encoder_branch_gated`: per-block spectral/local gates only
  - `pinn_hybrid_resnet_encoder_branch_gated_layerscale`: per-block gates plus per-block outer update scale
  - optional `pinn_hybrid_resnet_encoder_fusion_norm`: normalization follow-up only after the primary matrix completes
- [ ] Explicitly defer shared-scope scalar controls unless the mandatory per-block rows complete and a later extension chooses to compare placement.

Verification:

- [ ] **Blocking:** confirm the summary scaffold, `execution_manifest.json`, and `row_contract_audit.json` all exist and agree on the same baseline source root, row roster, scalar-scope decision, and claim boundary before any code change or training launch.
- [ ] **Blocking:** confirm the summary scaffold and `execution_manifest.json` both record the same defer-condition clearance rationale: completed `lines128` CDI headline authority, completed paper-evidence audit, and steering's current Phase 2 plus Phase 3 selection window.
- [ ] **Blocking:** confirm `row_contract_audit.json` separates frozen semantic fields, canonical dataset-input paths, and regenerated row-local output paths, and explicitly denylists historical `train_npz` / `test_npz` / `recon_npz` / `output_dir` reuse from the recovered baseline-row config.
- [ ] **Blocking:** if the authoritative baseline root or baseline-row provenance files are missing or inconsistent, repair the audit surface first; do not launch a drifted variant set.

### Tranche 2: Add Minimal Torch-Only Encoder-Fusion Controls

- [ ] Extend `HybridResnetEncoderBlock` so the baseline path remains unchanged by default and new encoder-fusion controls are explicit and isolated:
  - fusion mode selection
  - per-block branch gates for spectral/local contributions
  - per-block outer LayerScale on the full fused update
  - optional normalization hook only if the follow-up lane is later enabled
- [ ] Preserve the current identity residual path in every mode.
- [ ] Thread the new knobs through Torch-only plumbing:
  - `scripts/studies/grid_lines_torch_runner.py`
  - `PyTorchExecutionConfig` in `ptycho/config/config.py`
  - any required PyTorch config payload surface for checkpoint rebuilds
  - `ptycho_torch/workflows/components.py`
  - `ptycho_torch/model.py`
  - `ptycho_torch/generators/hybrid_resnet.py`
- [ ] Keep the new controls out of canonical bridged `ModelConfig` / legacy-config expectations unless a narrow persistence need is documented and kept PyTorch-local.
- [ ] Add or update a narrow Lines128 helper so it can:
  - ingest the reused baseline row from the completed bundle
  - launch only the fresh encoder-fusion rows
  - collate the append-only outputs under one new root
  - record reused-vs-fresh status per row

Verification:

- [ ] **Blocking:** add or update generator tests that prove:
  - baseline, per-block LayerScale, per-block branch-gated, and combined modes all build and preserve shape
  - branch-gated and LayerScale paths initialize to small positive nonzero scalars
  - identity-residual structure is preserved
  - invalid mode/value combinations are rejected
- [ ] **Blocking:** add or update runner/workflow tests that prove:
  - the new encoder-fusion knobs stay out of canonical bridged config surfaces
  - CLI/config validation rejects bad values
  - workflow overrides forward the new knobs into the factory/generator path
- [ ] **Supporting:** if a new study helper is added, add a focused study test that proves it can reuse the baseline row, plan the fresh rows, and emit its manifest surfaces without launching real training.

### Tranche 3: Run Deterministic Gates Before Any Expensive Launch

- [ ] After the summary scaffold from Tranche 1 exists and before any long run, execute the backlog item’s required deterministic checks exactly as written:
  - `pytest -q tests/torch/test_fno_generators.py -k "hybrid_resnet_encoder or hybrid_encoder"`
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_encoder"`
- [ ] Run any narrow supplemental selector needed to cover the new helper or new encoder-fusion mode tests that are not guaranteed by the two required `-k` filters.
- [ ] Because this item touches production Torch workflow code, also run:
  - `pytest -v -m integration`

Verification:

- [ ] **Blocking:** the two backlog-required `pytest` commands must pass before any expensive ablation launch.
- [ ] **Blocking:** any supplemental selector that covers the new encoder-fusion helper/plumbing must also pass before training.
- [ ] **Blocking:** if the integration marker fails because of a normal harness, path, or environment issue, diagnose/fix/rerun first; do not declare `BLOCKED` without a documented narrow recovery attempt.
- [ ] **Supporting:** `python -m compileall -q ptycho_torch scripts/studies` after code changes to catch syntax/import drift before training.

### Tranche 4: Execute The Same-Contract Encoder-Fusion Rows

- [ ] Create a unique ablation run root under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/runs/`
- [ ] Promote or reference the baseline `pinn_hybrid_resnet` row from the authoritative complete-table root without rewriting that source root.
- [ ] Launch the mandatory fresh rows with the exact fixed contract from `row_contract_audit.json`:
  - frozen semantic/model fields copied from the accepted baseline row artifacts
  - dataset-input paths resolved from the authoritative complete-table dataset records
  - row-local output and recon destinations regenerated under the new ablation root
- [ ] Keep the changed factor isolated per row:
  - `pinn_hybrid_resnet_encoder_layerscale`: per-block outer LayerScale only
  - `pinn_hybrid_resnet_encoder_branch_gated`: per-block spectral/local branch gates only
  - `pinn_hybrid_resnet_encoder_branch_gated_layerscale`: combine only those two encoder-fusion changes
- [ ] Do not execute shared-only scalar rows as substitutes for the mandatory per-block rows.
- [ ] Launch optional normalized fusion only if:
  - the mandatory rows are complete
  - at least one mandatory fresh row shows a real stitched-metric improvement worth follow-up
  - the same fixed contract still holds
  - the extra runtime fits the bounded budget
- [ ] If a mandatory fresh row fails because of an ordinary implementation, test-harness, import, path, or environment issue:
  - diagnose the narrow failure
  - apply the minimal fix
  - rerun the deterministic gates that cover the fix
  - relaunch the row
  - do not collapse the item into a "partial complete" closeout on that basis
- [ ] For every long-running launch:
  - start in tmux from repo root
  - activate `ptycho311`
  - use a unique output root
  - track the exact PID
  - wait on that PID
  - do not relaunch if another writer is already using the same output root
  - accept completion only on `exit_code=0` plus fresh required artifacts

Verification:

- [ ] **Blocking:** after each launched row or combined compare pass, confirm the row root contains fresh `invocation.json`, `config.json`, `metrics.json`, `history.json`, reconstruction outputs, and expected launcher-completion evidence.
- [ ] **Blocking:** confirm the run manifest explicitly marks which row is reused baseline and which rows are fresh.
- [ ] **Blocking:** if any mandatory fresh row is still unresolved after a narrow recovery attempt, do not move to final completion; either finish the row or escalate a true blocker that fits the backlog-drain blocked standard.
- [ ] **Supporting:** if the optional normalized-fusion row is omitted, write the reason into the manifest and durable summary instead of silently dropping it.

### Tranche 5: Collate Metrics, Visuals, And The Encoder-Fusion Read

- [ ] Collate the reused baseline and fresh encoder-fusion rows into one same-contract comparison bundle that reuses the completed Lines128 table/figure schema wherever practical:
  - merged metrics JSON
  - CSV/TeX table outputs
  - per-row recon visual panels
  - shared compare visual panels using the same fixed sample IDs and scale policy
- [ ] Write `comparison_summary.json` with:
  - baseline-vs-row deltas
  - reused/fresh status per row
  - scalar-scope statement
  - exact changed-factor statement for each row
  - explicit note that final stitched metrics, not training loss alone, decide whether a variant helped
- [ ] Ensure the durable interpretation can answer:
  - whether per-block LayerScale improved or harmed the CDI contract
  - whether per-block branch gating improved or harmed the CDI contract
  - whether the combined row shows constructive interaction or not

Verification:

- [ ] **Blocking:** the final bundle must contain `metrics.json`, `model_manifest.json`, `comparison_summary.json`, at least one machine-readable table (`metrics_table.csv`), at least one paper-facing table artifact (`metrics_table.tex`), and the expected visual bundle.
- [ ] **Supporting:** if shared helpers cannot emit the exact completed-bundle visual schema, document the emitted equivalent schema and why it is still comparable.

### Tranche 6: Publish The Durable Summary And Evidence Index Updates

- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_encoder_fusion_variants_summary.md` with:
  - authoritative ablation root
  - fixed baseline source root
  - the same explicit resume-condition clearance note carried from Tranche 1
  - exact reused/fresh row roster
  - scalar-scope decision for the first scored pass
  - row-level changed factor for each fresh row
  - main CDI findings based on final stitched metrics
  - explicit note that shared-scope placement remains a distinct future architecture axis
  - optional normalized-fusion outcome or omission reason
  - claim boundary: append-only same-contract CDI ablation, decision-support only
- [ ] Update:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/studies/index.md`
- [ ] Do not publish the durable summary or update evidence indexes as a completed outcome unless the reused baseline plus all three mandatory fresh rows are present, auditable, and interpreted from stitched metrics; unresolved mandatory rows keep the item in-progress unless a true blocker is recorded.
- [ ] Update `docs/index.md` only if the new summary needs first-tier discoverability beyond the study/evidence indexes.

Verification:

- [ ] **Blocking:** every updated evidence/index surface must point to the same summary path, artifact root, baseline authority, and decision-support claim boundary.
- [ ] **Blocking:** the durable summary must distinguish:
  - reused baseline vs fresh encoder-fusion rows
  - per-block first-pass decision vs deferred shared-scope controls
  - stitched-metric read vs training-loss-only impressions
  - append-only ablation evidence vs the completed CDI headline authority

### Tranche 7: Final Deterministic Closeout

- [ ] Rerun the backlog-required deterministic checks and archive logs under the ablation root’s `verification/` directory:
  - `pytest -q tests/torch/test_fno_generators.py -k "hybrid_resnet_encoder or hybrid_encoder"`
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_encoder"`
- [ ] Archive any supplemental selector used for the new encoder-fusion helper/plumbing.
- [ ] Archive the integration-marker log because workflow code changed.
- [ ] Record verification log paths in the durable summary.

Verification:

- [ ] **Blocking:** the item closes only with archived passing evidence for the two backlog-required `pytest` commands.
- [ ] **Blocking:** any supplemental selector and the integration marker must also have archived passing evidence if their covered surfaces changed.
- [ ] **Supporting:** archive `python -m compileall -q ptycho_torch scripts/studies` when code changed.

## Completion Criteria

- The fixed `lines128` CDI contract is preserved and auditable from the completed baseline root through the new encoder-fusion ablation root.
- The completed six-row CDI bundle remains unchanged and the new evidence is append-only.
- The plan, manifest, and durable summary all record why the backlog item's original defer-until-later clause is now cleared: the complete `lines128` CDI anchor is already the current paper-grade headline authority, the paper-evidence audit is complete, and steering allows this bounded Phase 3 follow-on within the current selection window.
- The first scored pass explicitly uses per-block scalars for the new encoder controls and does not collapse into shared-only scalar placement.
- The reused baseline plus all three mandatory fresh rows are executed under the fixed contract; this item does not close as complete with a partial matrix.
- If any mandatory row hits an ordinary implementation, harness, import, path, or environment failure, implementation must diagnose/fix/rerun and cannot resolve the item by downgrading it to a completed partial outcome.
- If a mandatory row still cannot be produced after a documented narrow recovery attempt, the plan must record a true blocker that matches the backlog-drain blocked standard instead of treating the missing row as acceptable completion.
- The durable summary states whether per-block encoder LayerScale, per-block branch gating, or their combination improved, harmed, or traded off on final stitched CDI metrics.
- Evidence indexes and study discovery surfaces are updated consistently.
- The backlog item’s required deterministic checks pass and their logs are archived under the new artifact root.
