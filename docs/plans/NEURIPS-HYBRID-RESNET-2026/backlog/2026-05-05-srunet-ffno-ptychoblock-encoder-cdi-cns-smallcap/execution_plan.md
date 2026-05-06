# SRU-Net FFNO-To-PtychoBlock Encoder CDI/CNS Small-Cap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evaluate one SRU-Net encoder mechanism variant, `FFNO -> 2x(PtychoBlock + downsample)`, on the fixed Lines128 CDI benchmark and a matched small-cap PDEBench CNS benchmark without rerunning completed baseline rows.

**Architecture:** Add a new explicit model/profile family, preferably `hybrid_resnet_ffno_ptychoblock_encoder`. The variant keeps the baseline SRU-Net shell after the encoder: same lifter/input transform, two downsample stages, bottleneck family/depth/width, decoder, skip wiring, residual-scaling policy, output mode, losses, scheduler, seeds, metrics, and visual policy. The only intended change is the encoder: a small FFNO-style factorized spectral stack at lifted/input resolution, followed by two shape-preserving `PtychoBlock` stages, each paired with the existing SRU-Net downsample layer.

**Tech Stack:** Python via PATH `python`, `ptycho311` for long-running runs, PyTorch/Lightning, `ptycho_torch/generators/hybrid_resnet.py`, `ptycho_torch/generators/fno.py`, `ptycho_torch/generators/ffno_bottleneck.py`, `ptycho_torch/model.py`, `scripts/studies/grid_lines_torch_runner.py`, `scripts/studies/grid_lines_compare_wrapper.py`, `scripts/studies/pdebench_image128/*`, Markdown/JSON study indexes.

---

## Selected Objective

- Add and evaluate exactly one fresh CDI row:
  - `pinn_hybrid_resnet_ffno_ptychoblock_encoder`
- Add and evaluate exactly one fresh CNS small-cap row/profile:
  - `hybrid_resnet_ffno_ptychoblock_encoder_cns` or an equivalent explicit profile id
- Reuse completed comparison rows by lineage:
  - Lines128 CDI: `pinn_hybrid_resnet`, `pinn_hybrid_resnet_encoder_spectral_only`, `pinn_ffno`, and `pinn_hybrid_resnet_ffno_bottleneck` where the row exists with compatible provenance.
  - CNS: `spectral_resnet_bottleneck_base` or the best matched SRU-Net-family row, `author_ffno_cns_base`, `fno_base`, and `unet_strong` from the matched-condition CNS authority.
- Interpret the result as an encoder mechanism ablation. Do not promote it as a new default architecture unless a separate roadmap/backlog item asks for that.

## Scope Boundaries

### In Scope

- Add model code for the FFNO-first, two-PtychoBlock encoder variant.
- Add CDI runner/compare-wrapper support for the new row id.
- Add PDEBench model-profile/factory support for the matching CNS small-cap profile.
- Launch only the new CDI and CNS rows.
- Collate an append-only summary that cross-references the completed baselines and records the fresh row artifacts.
- Update normal discoverability surfaces so the result can be found from the study index and NeurIPS evidence entry points.

### Explicit Non-Goals

- Do not rerun completed CDI or CNS baselines just to assemble the comparison.
- Do not tune FFNO encoder depth, modes, sharing, gates, normalization, or MLP ratio after seeing results.
- Do not change the SRU-Net bottleneck, decoder, skip structure, residual scaling, losses, seeds, dataset splits, probe/data preprocessing, epoch budgets, metric schemas, or visual sample policy while claiming this is an encoder-only ablation.
- Do not combine CDI and CNS metrics into a single ranking. Report them separately.
- Do not replace the completed Lines128 benchmark table, CNS matched-condition table, or paper evidence package. This item appends mechanism evidence.

## Prerequisite Status

- Required completed inputs:
  - `docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md`
  - `docs/backlog/done/2026-05-04-cdi-lines128-srunet-branch-objective-ablation.md`
  - `docs/backlog/done/2026-05-04-cns-matched-condition-table-refresh.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_srunet_branch_objective_ablation_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md`
- Required code inputs:
  - `ptycho_torch/generators/fno.py`
  - `ptycho_torch/generators/ffno_bottleneck.py`
  - `ptycho_torch/generators/hybrid_resnet.py`
  - `scripts/studies/grid_lines_torch_runner.py`
  - `scripts/studies/run_pdebench_image128_suite.py`
- The selected backlog item was previously invalid only because this execution plan was missing. This plan is the missing `plan_path` target.

## Implementation Architecture

### Model Design

- Implement a focused encoder module that can be shared by the CDI generator and PDEBench image model factory.
- Preferred factoring:
  - Keep the existing `HybridResnetGeneratorModule` unchanged for baseline architecture ids.
  - Add a new generator module/class for `hybrid_resnet_ffno_ptychoblock_encoder`, either in `ptycho_torch/generators/hybrid_resnet.py` or a narrow sibling module if imports stay simple.
  - Reuse existing `FactorizedFfnoBlock` / `SharedFactorizedFfnoBottleneck` pieces where practical, but place them before downsampling rather than in the bottleneck.
  - Reuse `PtychoBlock` semantics from `ptycho_torch/generators/fno.py` as the two shape-preserving encoder stages.
  - Reuse the existing SRU-Net downsample builders and decoder/skip/bottleneck path.
- Required encoder order:
  - `SpatialLifter`
  - FFNO encoder stack at lifted resolution
  - `PtychoBlock` stage 1
  - existing downsample layer 1
  - `PtychoBlock` stage 2
  - existing downsample layer 2
  - unchanged adapter/bottleneck/decoder/output shell
- Required manifest/config fields:
  - `encoder_variant`
  - `ptychoblock_stage_count`
  - `downsample_steps`
  - `downsample_op`
  - `ffno_encoder_blocks`
  - `ffno_encoder_modes`
  - `ffno_encoder_share_weights`
  - `ffno_encoder_gate_init`
  - `ffno_encoder_norm`
  - `ffno_encoder_mlp_ratio`

### CDI Integration

- Add the new architecture id to:
  - `ptycho_torch/model.py`
  - `scripts/studies/grid_lines_torch_runner.py`
  - `scripts/studies/grid_lines_compare_wrapper.py`
  - any model label/order helpers used by Lines128 summaries or visuals
- Preserve the fixed Lines128 contract:
  - `N=128`
  - `seed=3`
  - `40` epochs
  - fixed sample ids `0` and `1`
  - same probe/data preprocessing and metric schema as the completed Lines128 benchmark
- Fresh CDI row output must include row-local invocation/config/history/metrics/reconstruction/visual/completion-proof artifacts.

### CNS Integration

- Add a manual-only PDEBench profile for the new architecture in `scripts/studies/pdebench_image128/run_config.py`.
- Extend `scripts/studies/pdebench_image128/models.py` so the profile builds the same FFNO-first encoder variant under the CNS image-suite shell.
- Preserve the matched CNS small-cap contract:
  - task: official `2d_cfd_cns`
  - selected lane: `h5_512_64_64_40ep`
  - `history_len=5`
  - train/val/test caps: `512 / 64 / 64`
  - `40` epochs
  - batch size `4`
  - Adam at `2e-4`
  - existing CNS MSE training recipe and metric family
- Fresh CNS row output must include invocation/config/history/metrics/field-visual/completion-proof artifacts.

## File And Artifact Targets

### Mandatory Code Surfaces

- Modify: `ptycho_torch/generators/hybrid_resnet.py`
- Modify: `ptycho_torch/model.py`
- Modify as needed only if imports/exports require it: `ptycho_torch/generators/__init__.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify if needed for config/provenance round-trip: `ptycho/config/config.py`
- Modify if needed for workflow generator override round-trip: `ptycho_torch/workflows/components.py`
- Modify: `scripts/studies/pdebench_image128/run_config.py`
- Modify: `scripts/studies/pdebench_image128/models.py`
- Add or modify a narrow collation helper under `scripts/studies/` for this item if existing bundle helpers cannot append the fresh rows cleanly.

### Mandatory Tests

- Modify or add focused generator tests:
  - `tests/torch/test_fno_generators.py`
  - `tests/torch/test_ffno_bottleneck.py`
  - or a new focused `tests/torch/test_hybrid_resnet_ffno_ptychoblock_encoder.py`
- Modify runner/wrapper tests:
  - `tests/torch/test_grid_lines_torch_runner.py`
  - `tests/test_grid_lines_compare_wrapper.py`
- Modify PDEBench model-profile tests:
  - `tests/studies/test_pdebench_image128_models.py`
- Test requirements:
  - new CDI architecture id resolves and builds
  - new PDEBench profile resolves and builds for CNS-shaped tensors
  - FFNO encoder metadata appears in config/provenance
  - checkpoint/config reconstruction preserves the FFNO encoder fields
  - two PtychoBlock stages and two downsample stages are present
  - skip topology remains the same as the matched SRU-Net shell
  - baseline architecture ids still build unchanged

### Mandatory Artifact Outputs

- Item artifact root:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/`
- CDI fresh row root:
  - row id `pinn_hybrid_resnet_ffno_ptychoblock_encoder`
- CNS fresh row root:
  - profile id `hybrid_resnet_ffno_ptychoblock_encoder_cns` or documented equivalent
- Durable summary:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/srunet_ffno_ptychoblock_encoder_cdi_cns_smallcap_summary.md`
- Discoverability updates:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/studies/index.md`
  - any existing model-variant or ablation index that already covers SRU-Net mechanism studies

## Execution Checklist

### Task 1: Freeze Contracts And Existing Baselines

- [ ] Run the backlog item `check_commands` before code changes and capture the output in the execution notes.
- [ ] Confirm the fixed Lines128 contract from the completed Lines128 summary.
- [ ] Confirm the current matched CNS authority is `h5_512_64_64_40ep` from `pdebench_cns_matched_condition_table_refresh_summary.md`.
- [ ] Locate compatible existing baseline rows and record exact summary/artifact paths for lineage reuse.
- [ ] Create the item artifact root without copying or mutating completed baseline artifacts.

Verification for Task 1:

- Blocking:
  - backlog item required input check
- Supporting:
  - short JSON/Markdown execution note listing selected baseline lineage paths

### Task 2: Implement The Shared Encoder Variant

- [ ] Add the FFNO-first encoder module with explicit constructor arguments for all required manifest fields.
- [ ] Use a small, fixed default FFNO encoder recipe. Start with the existing Lines128/SRU-Net mode convention unless CNS shape forces a documented cap.
- [ ] Build exactly two shape-preserving `PtychoBlock` stages and pair each with the existing SRU-Net downsample layer.
- [ ] Preserve the downstream SRU-Net adapter/bottleneck/decoder/output shell.
- [ ] Expose the variant through `ptycho_torch/model.py` for CDI checkpoint rebuilds.
- [ ] Add generator tests proving shape, stage count, metadata, and baseline non-regression.

Verification for Task 2:

- Blocking:
  - `pytest -q tests/torch/test_fno_generators.py -k "PtychoBlock or hybrid_resnet or ffno"`
  - any new focused generator test selector
- Supporting:
  - `python -m compileall -q ptycho_torch`

### Task 3: Wire The CDI Row

- [ ] Add `hybrid_resnet_ffno_ptychoblock_encoder` to `grid_lines_torch_runner.py` architecture choices, config serialization, paper labels, visual ordering, invocation reconstruction, and validation.
- [ ] Add compare-wrapper routing for `pinn_hybrid_resnet_ffno_ptychoblock_encoder`.
- [ ] Ensure row provenance records all required encoder fields.
- [ ] Ensure config/checkpoint reconstruction preserves the required encoder fields through `ptycho/config/config.py`, `ptycho_torch/workflows/components.py`, and `ptycho_torch/model.py` where those surfaces are involved.
- [ ] Add runner/wrapper tests for row id routing and provenance.
- [ ] Run a cheap smoke/dry run only if the existing runner supports it without weakening the fixed final run.

Verification for Task 3:

- Blocking:
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno"`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or ffno"`
- Supporting:
  - compile check for touched study scripts

### Task 4: Wire The CNS Profile

- [ ] Add `ModelProfile` fields for FFNO encoder settings if the existing dataclass cannot represent them.
- [ ] Add manual-only profile `hybrid_resnet_ffno_ptychoblock_encoder_cns`.
- [ ] Extend `build_model_from_profile` to build the new profile with CNS input/output channel counts.
- [ ] Make `describe_model` or profile description include the required encoder fields.
- [ ] Keep the profile out of default primary/readiness bundles unless the backlog item explicitly runs it by profile id.
- [ ] Add tests that verify the profile builds and is not added to default bundles.

Verification for Task 4:

- Blocking:
  - `pytest -q tests/studies/test_pdebench_image128_models.py -k "hybrid_resnet or ffno"`
- Supporting:
  - `python -m compileall -q scripts/studies/pdebench_image128`

### Task 5: Launch Fresh Rows

- [ ] Launch the CDI row under `ptycho311` in `tmux` and keep the row under implementation ownership until the tracked command exits.
- [ ] Verify the CDI row wrote fresh metrics, visuals, invocation/config/history, and completion proof.
- [ ] Launch the CNS small-cap row under `ptycho311` in `tmux` and keep the row under implementation ownership until the tracked command exits.
- [ ] Verify the CNS row wrote fresh metrics, field visuals, invocation/config/history, and completion proof.
- [ ] If a run fails because of code, import, shape, test, or harness issues, fix narrowly and rerun. A partial first implementation is not a blocker.
- [ ] Use `BLOCKED` only for a genuine unrecoverable boundary such as missing data/hardware, unavailable required dependency outside local authority, or a roadmap/user-decision conflict.

Verification for Task 5:

- Blocking:
  - row-local completion proof with exit code `0`
  - required fresh metrics files
  - required visual artifacts
- Supporting:
  - captured launch command and environment provenance

### Task 6: Collate Append-Only Evidence

- [ ] Create an append-only comparison bundle under the item artifact root.
- [ ] Include fresh CDI and CNS row payloads plus lineage references to the completed baseline rows.
- [ ] Do not copy stale baseline metrics into a new authority without their source paths and provenance.
- [ ] Report CDI metrics separately from CNS metrics.
- [ ] Record whether the variant improves, matches, or worsens each benchmark relative to the selected lineage baselines.
- [ ] Include the required encoder manifest fields in the bundle.

Verification for Task 6:

- Blocking:
  - bundle JSON/Markdown can be loaded/read
  - every row has either fresh artifacts or explicit lineage source
  - no mixed-cap CNS ranking table
- Supporting:
  - optional lightweight schema/assertion test for the bundle builder

### Task 7: Update Discoverability

- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/srunet_ffno_ptychoblock_encoder_cdi_cns_smallcap_summary.md`.
- [ ] Update `evidence_matrix.md` with bounded mechanism-evidence status.
- [ ] Update `paper_evidence_index.md` without implying the result replaces locked headline tables.
- [ ] Update `docs/studies/index.md`.
- [ ] Update any existing model-variant or ablation index that covers SRU-Net mechanism rows.

Verification for Task 7:

- Blocking:
  - links in the new summary point to existing artifacts/docs
  - summary names the exact CDI and CNS contracts used
- Supporting:
  - `rg -n "ffno_ptychoblock|FFNO-To-PtychoBlock|pinn_hybrid_resnet_ffno_ptychoblock_encoder" docs/plans/NEURIPS-HYBRID-RESNET-2026 docs/studies`

### Task 8: Final Checks

- [ ] Run the backlog item check commands again.
- [ ] Run the focused tests added or modified by this implementation.
- [ ] Run `python -m compileall -q ptycho_torch scripts/studies`.
- [ ] Inspect `git diff` for accidental baseline reruns, table replacement, unrelated docs churn, or changed default profile bundles.
- [ ] Record commands, results, artifact paths, and any residual limitations in the summary.

Minimum final verification command set:

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
pytest -q tests/torch/test_fno_generators.py -k "PtychoBlock or hybrid_resnet or ffno"
pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno"
pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or ffno"
pytest -q tests/studies/test_pdebench_image128_models.py -k "hybrid_resnet or ffno"
python -m compileall -q ptycho_torch scripts/studies
```

## Completion Criteria

- The missing `plan_path` exists and the backlog item is eligible for selection.
- The new CDI row has fresh metrics and visual artifacts under the fixed Lines128 contract.
- The new CNS row has fresh metrics and field visuals under the selected small-cap matched CNS contract.
- Baseline comparisons are by explicit lineage, not rerun.
- The durable summary and indexes make the result discoverable.
- The final interpretation states whether the FFNO-first encoder helped or hurt CDI and CNS separately, with exact metric values and artifact paths.
