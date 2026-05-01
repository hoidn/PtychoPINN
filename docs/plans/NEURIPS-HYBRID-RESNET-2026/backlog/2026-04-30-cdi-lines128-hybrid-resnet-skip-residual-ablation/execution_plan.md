# Lines128 Hybrid ResNet Skip/Residual Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement this plan task-by-task. Keep this file as the execution authority for the selected backlog item.

**Goal:** Quantify how decoder skip connections and bottleneck residual scaling affect `hybrid_resnet` on the fixed `lines128` CDI paper contract, then publish an append-only ablation bundle and durable summary that cross-reference existing related evidence without rewriting the completed six-row CDI benchmark.

**Architecture:** Reuse the authoritative `lines128` complete-table bundle as the baseline contract and baseline-row provenance source, run only the minimum fresh Hybrid ResNet variants needed to isolate skip-on/off and learned-vs-fixed bottleneck residual scaling, and collate the new rows into a separate ablation root. Keep the completed six-row bundle immutable, keep all fairness knobs fixed to the accepted `pinn_hybrid_resnet` row, and surface any new residual-control implementation as Torch-only study plumbing rather than as a canonical config-bridge expansion.

**Tech Stack:** Python 3.11, `ptycho311`, tmux-managed long runs, PyTorch/Lightning, `scripts/studies/grid_lines_torch_runner.py`, `scripts/studies/grid_lines_compare_wrapper.py`, shared paper-metrics/visual collation helpers under `scripts/studies/`, Markdown/JSON/CSV/TeX evidence artifacts.

---

## Selected Backlog Objective

- Measure the effect of enabling Hybrid ResNet decoder skip fusion and of removing the learned bottleneck residual gate from the fixed `lines128` CDI contract.
- Treat the completed `pinn_hybrid_resnet` row from the authoritative six-row CDI bundle as the baseline reference; do not rerun or mutate that bundle in place.
- Publish a new append-only ablation artifact root plus the required durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_skip_residual_ablation_summary.md`.

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
- The exact remaining Hybrid-row launch fields not enumerated above must be copied from the authoritative baseline-row artifacts before any fresh launch, using:
  - `.../runs/pinn_hybrid_resnet/config.json`
  - `.../paper_benchmark_manifest.json`
  - `.../model_manifest.json`
- Mandatory fresh row set for this item:
  - baseline reference only, reused: `pinn_hybrid_resnet`
  - fresh skip row: `pinn_hybrid_resnet_skip_add`
  - fresh residual-control row: `pinn_hybrid_resnet_residual_fixed`
  - fresh interaction row: `pinn_hybrid_resnet_skip_add_residual_fixed`
- Optional row only if the primary three fresh rows are complete and the run budget stays bounded:
  - `pinn_hybrid_resnet_skip_gated_add`
- Deferred from this item unless a later approved plan reopens them:
  - `concat` skip style fresh row
  - encoder layerscale / encoder branch-gate work
  - bottleneck-family replacements or broader Hybrid sweeps

## Explicit Non-Goals

- Do not rewrite, rerun, or relabel the completed six-row `lines128` paper bundle.
- Do not broaden into encoder-fusion, encoder branch gating, encoder LayerScale, FFNO/U-NO follow-up, PDEBench CNS, or `256x256` CDI scaling.
- Do not treat prior CNS skip-add improvement as CDI evidence; it is context only.
- Do not silently relax fairness constraints, sample policy, or metric schema to make the ablation easier.
- Do not mark the item `BLOCKED` for ordinary import, test, path, harness, or environment failures before a narrow diagnose/fix/rerun attempt is documented.

## Binding Constraints And Prerequisite Status

- Steering and roadmap boundary:
  - This is bounded Roadmap Phase 3 CDI evidence strengthening only.
  - Equal-footing comparison standards remain explicit and fixed.
  - The output is decision-support and paper-context append-only evidence; it does not replace the completed CDI headline authority.
- Append-only evidence rule:
  - The ablation must live under a new dated artifact root and summary.
  - Existing evidence surfaces are cross-referenced, not mutated for content:
    - completed Lines128 summary/root
    - legacy `hybrid-resnet-mode-skip-sweep` study index entry and any discoverable scored outputs
    - PDEBench CNS skip-add evidence, clearly labeled non-CDI context
    - active encoder-fusion backlog item, clearly labeled separate future work
- Long-run ownership rule:
  - Once a training/compare launch starts, implementation owns the command until tracked success or documented recoverable failure handling is complete.
  - Use tmux, activate `ptycho311`, track the exact PID, and confirm both `exit_code=0` and fresh required output artifacts before treating the run as complete.
- Prerequisite status:
  - `2026-04-29-cdi-lines128-paper-benchmark-execution`: completed and authoritative.
  - Selected backlog prerequisite is therefore satisfied.
  - The current progress ledger records Phase 0, Phase 1, and specific approved Phase 2 tranche completions; it does not itself claim Phase 3 completion authority for this item.
  - Roadmap authority explicitly allows active NeurIPS evidence work across remaining Phase 2 PDEBench items and Phase 3 CDI items in parallel, so this selected Phase 3 CDI ablation is admissible backlog work without waiting for all later Phase 2 follow-ups to finish.
- Residual-control implementation rule:
  - The approved implementation route for this item is a Torch-only bottleneck residual-scale control, not ad hoc model surgery.
  - Introduce an execution-only knob pair with explicit provenance, for example:
    - `hybrid_resnet_bottleneck_layerscale_mode = learned | fixed`
    - `hybrid_resnet_bottleneck_layerscale_value = 1.0` for the fixed control
  - Baseline path stays `learned` with the current shared scalar gate initialization.
  - Control path fixes the shared bottleneck residual multiplier to `1.0` and makes it non-trainable, which removes learned residual scaling while preserving the classic residual branch.
  - If this route cannot be implemented cleanly without risky shell surgery, stop after the narrow failed attempt, keep the skip-only tranche, and record residual scaling as deferred in the summary instead of inventing an untracked mutation.

## Implementation Architecture

- **Authority and contract surface**
  - Write one machine-readable execution manifest that freezes the baseline source root, required fresh rows, optional rows, row labels, cross-reference sources, and the exact fixed-contract provenance files copied from the completed `pinn_hybrid_resnet` row.
- **Torch-only ablation plumbing**
  - Keep new residual-control knobs in the Torch execution path only: runner config, `PyTorchExecutionConfig`, Torch config payload, workflow overrides, and the generator/bottleneck implementation.
  - Do not add new canonical `ModelConfig` or config-bridge fields for this item.
- **Ablation harness and collation**
  - Prefer a dedicated narrow study helper under `scripts/studies/` that reuses existing compare/collation utilities while treating the completed paper bundle as an immutable promoted-source input.
  - Only touch the existing paper-benchmark harness if helper extraction is materially smaller and does not risk the accepted complete-table path.

## Concrete File And Artifact Targets

- Mandatory code surfaces if residual-control plumbing is required:
  - `ptycho_torch/generators/resnet_components.py`
  - `ptycho_torch/generators/hybrid_resnet.py`
  - `ptycho/config/config.py`
  - `ptycho_torch/config_params.py`
  - `ptycho_torch/workflows/components.py`
  - `scripts/studies/grid_lines_torch_runner.py`
- Preferred new or changed study-orchestration surface:
  - new narrow helper such as `scripts/studies/lines128_hybrid_resnet_skip_residual_ablation.py`
  - alternatively, the smallest safe extraction/update in `scripts/studies/lines128_paper_benchmark.py`
- Likely shared evidence/collation helpers if needed:
  - `scripts/studies/grid_lines_compare_wrapper.py`
  - `scripts/studies/metrics_tables.py`
  - `scripts/studies/paper_provenance.py`
- Mandatory tests:
  - `tests/torch/test_fno_generators.py`
  - `tests/torch/test_grid_lines_torch_runner.py`
  - a dedicated study-level test module for the new ablation helper if one is created, for example:
    `tests/studies/test_lines128_hybrid_resnet_skip_residual_ablation.py`
- Mandatory durable outputs:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_skip_residual_ablation_summary.md`
  - new append-only artifact root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation/`
  - under that root:
    - `execution_manifest.json`
    - `row_contract_audit.json`
    - `cross_reference_manifest.json`
    - `comparison_summary.json`
    - `metrics.json`
    - `model_manifest.json`
    - `metrics_table.csv`
    - `metrics_table.tex`
    - `visuals/`
    - `verification/`
- Mandatory registry/index updates for a result-producing backlog item:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/studies/index.md`
- Preferred packaging only if it falls out naturally from the shared helpers:
  - TeX/CSV row labels aligned to the completed `lines128` bundle
  - fixed-sample visual panels using the same sample IDs and scale policy

## Execution Checklist

### Tranche 1: Freeze The Baseline Authority And Audit The Exact Ablation Matrix

- [ ] Create the durable summary file `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_skip_residual_ablation_summary.md` immediately as a scaffold that records:
  - selected backlog item id and plan path
  - authoritative complete-table source root
  - intended fresh row roster versus reused baseline row
  - explicit placeholder sections for results, cross-references, claim boundary, and deferred-risk notes
  - status text that the file is an in-progress execution summary and will be finalized only after the fresh rows and collation complete
- [ ] Create `execution_manifest.json` under the selected item artifact root with:
  - authoritative complete-table root
  - baseline row id `pinn_hybrid_resnet`
  - mandatory fresh rows
  - optional row policy
  - exact cross-reference docs/roots
  - fixed sample IDs inherited from the completed bundle
  - claim boundary text: same-contract CDI ablation, append-only, decision-support plus paper-context only
- [ ] Extract the full baseline-row launch/config provenance from the authoritative root and write `row_contract_audit.json` that names every field that must remain fixed.
- [ ] Write `cross_reference_manifest.json` naming the legacy skip/mode study, CNS skip-add summary, and encoder-fusion backlog item with explicit context labels.
- [ ] Decide the fresh row roster as:
  - reused baseline only
  - skip-add
  - residual-fixed
  - skip-add plus residual-fixed
  - optional gated-add only after the mandatory rows are secured
- [ ] Explicitly defer `concat` unless a later approved plan reopens it.

Verification:

- [ ] **Blocking:** confirm the summary scaffold, `execution_manifest.json`, `row_contract_audit.json`, and `cross_reference_manifest.json` all exist and agree on the same baseline source root, row roster, and fixed claim boundary before any code change or training launch.
- [ ] **Blocking:** if the authoritative complete-table root or baseline-row provenance files are missing or inconsistent, diagnose the root-cause and repair the audit surface first; do not launch a drifted row set.

### Tranche 2: Add Minimal Torch-Only Residual-Control Plumbing And Harness Coverage

- [ ] Add generator/bottleneck support for a fixed shared bottleneck residual multiplier alongside the current learned shared scalar gate.
- [ ] Keep the new controls Torch-only by plumbing them through:
  - `scripts/studies/grid_lines_torch_runner.py`
  - `PyTorchExecutionConfig` in `ptycho/config/config.py`
  - `ptycho_torch/config_params.py`
  - `ptycho_torch/workflows/components.py`
  - `ptycho_torch/generators/hybrid_resnet.py`
  - `ptycho_torch/generators/resnet_components.py`
- [ ] Add or update the narrow ablation helper so it can:
  - ingest the baseline row from the completed bundle
  - launch only the fresh ablation rows
  - collate the append-only metrics/visual outputs under one new root
  - record whether each row is reused or fresh
- [ ] Do not modify the completed paper bundle in place and do not regress the existing `lines128_paper_benchmark.py` complete-table path if it is touched.

Verification:

- [ ] **Blocking:** add/adjust generator tests that prove:
  - learned and fixed bottleneck residual modes both build and preserve shape
  - the fixed mode is non-trainable and leaves the residual branch active
  - skip-add plus fixed residual builds on the same shell
  - invalid mode/value combinations are rejected
- [ ] **Blocking:** add/adjust runner/workflow tests that prove:
  - new residual-control knobs stay out of canonical bridged config surfaces
  - CLI/config validation rejects bad values
  - workflow overrides forward the residual-control knobs into the factory/generator path
- [ ] **Supporting:** if a new study helper is added, add a focused study test that proves it can reuse the baseline row, plan the fresh rows, and emit its manifest surfaces without launching real training.

### Tranche 3: Run The Required Deterministic Gates Before Expensive Training

- [ ] After the summary scaffold from Tranche 1 exists and before any expensive launch, run the backlog item’s required deterministic checks exactly as written:
  - `pytest -q tests/torch/test_fno_generators.py -k "hybrid_resnet or resnet_decoder_block or skip_style"`
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_skip or hybrid_resnet_blocks or resnet_width"`
  - `python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_skip_residual_ablation_summary.md"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing skip/residual ablation summary: {missing}")
print("skip/residual ablation summary present")
PY`
- [ ] Because this item is expected to touch production Torch workflow code, also run the workflow-policy integration gate before any expensive launch:
  - `pytest -v -m integration`
- [ ] If the new residual-control tests live outside the required `-k` selectors, run the narrow supplemental selector that covers them and archive its log alongside the required checks.

Verification:

- [ ] **Blocking:** all required check commands above must pass before any long-running ablation launch.
- [ ] **Blocking:** if the integration marker fails because of a normal environment or harness issue, diagnose/fix/rerun first; do not declare `BLOCKED` without a documented narrow recovery attempt.
- [ ] **Supporting:** `python -m compileall -q ptycho_torch scripts/studies` after code changes to catch syntax/import drift before training.

### Tranche 4: Execute The Same-Contract Ablation Rows Under One Append-Only Root

- [ ] Create a new unique ablation run root under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation/runs/`
- [ ] Promote or reference the baseline `pinn_hybrid_resnet` row from the authoritative complete-table root without rewriting that source root.
- [ ] Launch the mandatory fresh rows with the exact fixed contract copied from `row_contract_audit.json`.
- [ ] Keep the changed factor isolated per row:
  - `pinn_hybrid_resnet_skip_add`: `hybrid_skip_connections=true`, `hybrid_skip_style=add`; residual gate behavior unchanged from baseline
  - `pinn_hybrid_resnet_residual_fixed`: same shell as baseline, but bottleneck residual mode fixed at `1.0`
  - `pinn_hybrid_resnet_skip_add_residual_fixed`: combine only those two changes
- [ ] Launch optional `pinn_hybrid_resnet_skip_gated_add` only if:
  - the mandatory rows are complete
  - the same fixed contract still holds
  - the extra runtime fits the bounded budget
- [ ] Keep `concat` out of the fresh execution roster.
- [ ] For every long-running launch:
  - start in tmux from repo root
  - activate `ptycho311`
  - use a unique output root
  - track the exact PID
  - wait on that PID
  - do not relaunch if another writer is already using the same output root
  - accept completion only on `exit_code=0` plus fresh required artifacts

Verification:

- [ ] **Blocking:** after each launched row or combined compare pass, confirm the row root contains fresh `invocation.json`, `config.json`, `metrics.json`, `history.json`, reconstruction NPZs, and any launcher-completion evidence expected by the chosen harness.
- [ ] **Blocking:** confirm the promoted baseline row is explicitly marked reused and fresh ablation rows are explicitly marked fresh in the run manifest.
- [ ] **Supporting:** if the optional gated-add row is omitted, write the reason into the ablation manifest and summary instead of silently dropping it.

### Tranche 5: Collate Metrics, Visuals, And The Append-Only Interpretation

- [ ] Collate the baseline and fresh ablation rows into one same-contract comparison bundle that reuses the completed Lines128 table/figure schema wherever practical:
  - merged metrics JSON
  - CSV/TeX table outputs
  - per-row recon visual panels
  - shared compare visual panels using the same fixed sample IDs and scale policy
- [ ] Write `comparison_summary.json` with:
  - baseline vs fresh-row deltas
  - reused/fresh status per row
  - whether the interaction row was executed
  - explicit distinction between same-contract CDI evidence and cross-linked non-CDI context
- [ ] Ensure the summary can say clearly whether skip-add helps or hurts this CDI contract and whether fixed residual scaling helps or hurts relative to the learned gate.
- [ ] If residual-control implementation proved unsafe and was deferred, keep the bundle skip-only and record that the residual comparison remains unresolved.

Verification:

- [ ] **Blocking:** the final bundle must contain `metrics.json`, `model_manifest.json`, `comparison_summary.json`, at least one machine-readable table (`metrics_table.csv`), at least one paper-facing table artifact (`metrics_table.tex`), and the expected visual bundle.
- [ ] **Supporting:** if shared helpers can emit the same `compare_amp_phase` and FRC-style visuals as the completed bundle, use them; otherwise document exactly which equivalent visual schema was emitted and why.

### Tranche 6: Publish The Durable Summary And Evidence Index Updates

- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_skip_residual_ablation_summary.md` with:
  - authoritative ablation root
  - fixed baseline source root
  - exact fresh row roster and any omitted optional row
  - row-level changed factor for each fresh row
  - main CDI findings
  - explicit cross-reference section for:
    - legacy skip/mode study context
    - CNS skip-add context, labeled non-CDI
    - encoder-fusion backlog item, labeled future separate work
  - paper-facing implication versus decision-support-only interpretation
  - residual risks or deferred residual-control note if applicable
- [ ] Update:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/studies/index.md`
- [ ] Update `docs/index.md` only if the new summary needs first-tier discoverability beyond the study/evidence indexes.

Verification:

- [ ] **Blocking:** every updated evidence/index surface must point to the same ablation summary path, artifact root, and outcome boundary.
- [ ] **Blocking:** the durable summary must distinguish:
  - reused baseline row vs fresh ablation rows
  - same-contract CDI findings vs non-CDI context
  - append-only ablation evidence vs the completed CDI headline authority
- [ ] **Blocking:** replace the Tranche 1 scaffold status with the final outcome summary before closing the item; the required summary-existence check is satisfied early by the scaffold, but completion requires the finalized interpretation and cross-reference content above.

### Tranche 7: Final Deterministic Closeout

- [ ] Rerun the backlog-required deterministic checks and archive logs under the ablation root’s `verification/` directory.
- [ ] Archive the focused generator/runner/harness selectors used for the new residual-control coverage.
- [ ] Archive the integration-marker log if workflow code changed.
- [ ] Record verification log paths in the durable summary.

Verification:

- [ ] **Blocking:** the item closes only with archived passing evidence for the required check commands.
- [ ] **Supporting:** `python -m compileall -q ptycho_torch scripts/studies` log archived when code changed.

## Completion Criteria

- The fixed `lines128` CDI contract is preserved and auditable from the completed baseline root through the new ablation root.
- The completed six-row CDI bundle remains unchanged and the new evidence is append-only.
- At least the three mandatory fresh rows are executed, or a documented narrow residual-control failure leaves a clearly labeled skip-only outcome after a real implementation attempt.
- The durable summary states whether decoder skip-add and fixed residual scaling improved, hurt, or produced a trade-off on the fixed CDI contract.
- Evidence indexes and study discovery surfaces are updated consistently.
- Required deterministic checks pass and their logs are archived under the new ablation artifact root.
