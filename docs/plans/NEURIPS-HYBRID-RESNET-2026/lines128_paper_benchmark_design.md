# NeurIPS Lines128 Paper-Quality CDI Benchmark Design

## Context And Authority

- Status: draft design
- Date: 2026-04-29
- Initiative: NeurIPS Hybrid ResNet 2026
- Consumed brief: design a NeurIPS paper-quality benchmark for Hybrid ResNet,
  Hybrid-spectral, FNO/FNO-vanilla, and FFNO on the `N=128` grid-lines best
  configuration, with quantitative metrics tables and visual reconstruction
  comparisons.
- Relevant docs:
  - `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
  - `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_plan.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/spectral_resnet_bottleneck_n128_integration_summary.md`
  - `docs/studies/index.md`
  - `docs/model_baselines.md`
  - `docs/workflows/pytorch.md`
  - `docs/COMMANDS_REFERENCE.md`
- Governing implementation conventions:
  - `scripts/studies/grid_lines_torch_runner.py` owns per-model Torch
    training, inference, stitching, per-model metrics, and reconstruction
    artifacts.
  - `scripts/studies/grid_lines_compare_wrapper.py` owns shared grid-lines
    dataset generation, wrapper-level provenance, multi-model routing, merged
    metrics, and combined visuals.
  - Paper-grade evidence must be regenerated when historical roots lack
    complete invocation, config, git/environment, dataset, metric, and visual
    provenance.

## Problem And Scope

The project needs a paper-quality CDI/ptycho benchmark on the study-indexed
`N=128` grid-lines best configuration. The benchmark must compare:

- `hybrid_resnet`
- `spectral_resnet_bottleneck_net`
- an FNO comparator, selected before launch as either `fno` or `fno_vanilla`
- FFNO as a CDI/grid-lines generator

The result must be suitable for paper tables and visual reconstruction figures,
not just local decision support. In scope:

- regenerate all benchmark rows under one fixed dataset/split/training/metric
  contract
- expose or add missing runner/wrapper support needed for
  `spectral_resnet_bottleneck_net` and FFNO
- produce metrics tables, merged machine-readable metrics, and visual
  amplitude/phase reconstruction comparisons
- preserve enough provenance to make the result externally auditable

Out of scope:

- using PDEBench CNS FFNO evidence as CDI generator evidence
- changing the CDI grid-lines task contract to fit FFNO
- promoting historical incomplete roots into paper-grade claims
- broad Hybrid-spectral/FFNO parameter sweeps beyond the named benchmark rows
- manuscript prose beyond producing paper-facing artifacts and summaries

## Decision Summary

Use a shared benchmark wrapper/harness that calls `grid_lines_torch_runner.py`
for every model row.

The preferred implementation is to extend `grid_lines_compare_wrapper.py` after
FFNO has a valid CDI/grid-lines Torch generator path. That keeps the paper run on
the existing shared dataset, stitching, and combined-visual tooling while
preserving the runner as the per-model authority. If FFNO support makes the
existing wrapper too awkward, add a thin `lines128` paper benchmark harness that
reuses the wrapper's dataset/provenance/collation helpers and still launches
`grid_lines_torch_runner.py` for each row. A direct collection of independent
runner commands is a fallback only; it is too easy to drift on dataset identity,
split, wrapper provenance, and combined figures.

Semantically material choices:

- Primary benchmark contract: freeze the study-indexed legacy-best `N=128`
  grid-lines contract because the requested target is the "128 lines best
  configuration setup." This contract is still reconstructed from
  non-paper-grade historical evidence, so the benchmark must produce a pre-run
  contract-reconstruction validation artifact before launching all rows.
- Current-baseline alternative: the current Hybrid ResNet anchor plan uses a
  `50` epoch baseline and `torch_mae_pred_l2_match_target=on`. That is a valid
  separate Phase 3 anchor decision, but this paper-benchmark design should not
  silently mix it with the legacy-best contract. If the project chooses the
  current baseline instead, record that as a pre-run design override and rerun
  every model row under the same changed contract.
- FNO comparator: choose either `fno` or `fno_vanilla` before the benchmark
  launches. Use `fno_vanilla` if the paper needs the least Hybrid-adjacent FNO
  baseline; use `fno` if continuity with prior local grid-lines wrappers is more
  important. The choice and rationale must be written to a durable pre-run
  decision note before execution; do not choose after seeing metrics.
- FFNO row: FFNO is mandatory for the requested benchmark, but it is blocked
  until it satisfies the CDI/grid-lines generator output contract through the
  Torch runner path.
- Seed policy: the primary benchmark is a fixed-contract `seed=3` comparison,
  not a robust statistical comparison. Any multi-seed paper claim requires a
  separate pre-run design amendment or backlog item that pins the seed list,
  aggregation rules, and extra runtime budget before execution.

## Fixed Benchmark Contract

The benchmark's primary contract is the recovered study-indexed
`grid-lines-n128-hybrid-resnet-legacy-best-e40-seed3` contract:

- `N=128`
- `gridsize=1`
- synthetic grid-lines data
- `set_phi=True`
- custom Run1084 probe
- `probe_scale_mode=pad_extrapolate`
- `probe_smoothing_sigma=0.5`
- `nimgs_train=2`
- `nimgs_test=2`
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
- `torch_output_mode=real_imag`
- `probe_mask=off`
- `fno_modes=12`
- `fno_width=32`
- `fno_blocks=4`
- `fno_cnn_blocks=2`

Historical decision-support metrics from this contract may be used only as
sanity context, not as the benchmark result. The paper benchmark must regenerate
fresh rows for all included models under one output root.

Before the full benchmark launches, the implementation must write a durable
contract-reconstruction validation artifact. Suggested path:
`docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md`
or an equivalent checked-in preflight note linked from the execution plan. That
artifact must include:

- the historical roots and docs used to reconstruct each contract field
- the confidence/source for every field above
- the fields that could not be recovered from child invocation/config artifacts
- the exact command-line flags the new wrapper/harness will use to recreate the
  contract
- the selected FNO comparator and rationale
- an explicit statement that the benchmark is fixed-seed `seed=3`, or a
  separately approved multi-seed extension
- a go/no-go result; no full paper benchmark may launch while this artifact is
  missing or unresolved

## Architecture And Modularity

### Components

1. FFNO CDI/grid-lines generator support
   - Adds or exposes an FFNO generator profile that can consume the same
     grid-lines train/test NPZs as the other Torch rows.
   - Must emit the same complex object representation and reconstructed outputs
     expected by the existing inference/stitching path.
   - Must fail closed if the generator output contract cannot be satisfied.

2. Per-model runner
   - `scripts/studies/grid_lines_torch_runner.py`
   - Owns model construction, training loop, checkpointing, inference,
     stitching, per-model metrics, reconstruction arrays, and per-model
     invocation/config artifacts.
   - Required architecture support for the benchmark:
     `hybrid_resnet`, `spectral_resnet_bottleneck_net`, selected FNO comparator,
     and FFNO.

3. Shared benchmark wrapper/harness
   - Preferred owner: `scripts/studies/grid_lines_compare_wrapper.py`.
   - Responsibilities: create or reference one shared train/test split, launch
     each child runner with normalized flags, prevent output-root collisions,
     capture wrapper-level invocation/provenance, and collate results.
   - If implemented as a separate paper harness, it must be thin and reuse the
     wrapper's dataset/collation primitives rather than replacing the runner.

4. Metrics and figure collator
   - Reads per-model `metrics.json`, training history, reconstruction arrays,
     and visual outputs.
   - Produces merged JSON/CSV/TeX tables and paper-facing comparison figures.
   - Records model display labels separately from runner architecture keys.

5. Durable summary
   - A post-run summary under
     `docs/plans/NEURIPS-HYBRID-RESNET-2026/`.
   - Later paper-facing links may be mirrored into
     `/home/ollie/Documents/neurips/` during evidence-bundle assembly.

### Source-Of-Truth Boundaries

- Runner outputs are authoritative for row-local metrics and reconstructions.
- The wrapper/harness manifest is authoritative for shared dataset identity,
  model list, child invocations, git/environment state, and table/figure
  collation.
- `docs/studies/index.md` and the durable NeurIPS summary are authoritative for
  the interpreted result after validation.

## Contracts And Data Flow

1. Preflight
   - validate model list, selected FNO comparator, output root freshness, probe
     path, and FFNO availability
   - write and review the contract-reconstruction validation artifact before
     launching the full benchmark
   - run a short parse/smoke path before any full paper run

2. Shared data
   - generate or pin one `N=128`, `gridsize=1`, `set_phi=True` train/test pair
     under the benchmark root
   - persist dataset paths, sample counts, seed, probe source, probe
     preprocessing flags, split metadata, and checksums when available

3. Child runs
   - create one child directory per row, for example:
     - `runs/hybrid_resnet/`
     - `runs/spectral_resnet_bottleneck_net/`
     - `runs/fno_vanilla/` or `runs/fno/`
     - `runs/ffno/`
   - launch every row through `grid_lines_torch_runner.py` with the same
     dataset, seed contract, loss, scheduler, epoch budget, output mode, and
     metric family
   - persist child `invocation.sh`, `invocation.json`, config, logs, metrics,
     training history, checkpoints as needed, and reconstruction arrays

4. Collation
   - merge child metrics into:
     - `metrics.json`
     - `metrics_table.csv`
     - `metrics_table.tex`
     - `metrics_table_best.tex`
     - `model_manifest.json`
   - create figure payloads under `visuals/`
   - write a `paper_benchmark_manifest.json` with all row roots, artifact mtimes,
     git SHA/dirty note, Python/Torch/CUDA/GPU provenance, and benchmark status
   - write `benchmark_decisions.json` or an equivalent manifest section with the
     frozen contract, FNO comparator decision, seed policy, and any approved
     deviations from this design

5. Documentation
   - update `docs/studies/index.md` with a result entry after successful
     execution
   - write a durable summary in the NeurIPS initiative docs
   - mirror only final paper-facing artifacts into `/home/ollie/Documents/neurips/`
     when the roadmap evidence-bundle phase requires it

## Metrics And Visual Outputs

Paper-quality status requires a complete metric schema. The merged benchmark
must set `benchmark_status=paper_complete` only if all required fields below are
present for every row, or if a field has an explicitly accepted not-applicable
status. If any required paper metric is absent, schema-incompatible, or
row-specific, the merged result must be labeled `benchmark_incomplete` and the
durable summary must state exactly which fields are missing.

Required metric fields:

- model key and display label
- parameter count
- epoch budget and final completed epoch
- final train loss
- final validation loss, if the training contract emits a real validation
  series; otherwise record `validation_loss.status=no_validation_series` and do
  not present a validation-loss table column as a measured quantity
- amplitude MAE, MSE, PSNR, SSIM, MS-SSIM, and FRC50
- phase MAE, MSE, PSNR, SSIM, MS-SSIM, and FRC50
- runtime and hardware summary
- caveat fields for missing metrics or schema differences

Required table artifacts:

- `metrics.json`
- `metrics_table.csv`
- `metrics_table.tex`
- `metrics_table_best.tex`
- `metric_schema.json`, defining required columns, units, nullability, and
  downgrade rules

Visual outputs must include:

- one fixed set of test sample IDs used for all model rows
- ground-truth amplitude and phase panels
- reconstructed amplitude and phase panels for each model
- amplitude and phase absolute-error panels for each model
- shared color scales per quantity across models
- source NPZ or stitched amplitude/phase arrays sufficient to regenerate the
  figure
- a combined `compare_amp_phase` figure plus model-specific detail figures

The visual comparison should not rely on cropped, darkened, or display-only
images as the source of truth. The numeric arrays must be preserved.

## Invariants And Failure Modes

Invariants:

- all rows use the same generated train/test split and probe preprocessing
- no row-specific change to `N`, `gridsize`, `set_phi`, `nphotons`, loss,
  scheduler, epoch budget, output mode, or metric family
- FFNO must adapt to the CDI generator contract; the CDI task contract must not
  silently change to make FFNO easier
- historical artifacts remain decision-support only unless the regenerated run
  reproduces the paper-grade artifact contract
- model labels must not obscure architecture identity

Expected failure modes:

- FFNO cannot emit the required complex object/output shape
- wrapper supports a runner architecture but does not pass all required flags
- one child run succeeds while another fails, leaving partial paper artifacts
- visual collation accidentally mixes sample IDs or color scales
- metrics schema differs by model or runner version
- a run root is reused or written by multiple processes

Detection and response:

- fail preflight if FFNO is unavailable or the selected FNO comparator is not
  explicit in the durable decision artifact
- fail preflight if the contract-reconstruction validation artifact is missing,
  unresolved, or contradicts the launch flags
- fail preflight if the output root already exists and is non-empty
- validate every child manifest before collation
- mark the full benchmark incomplete if any required row fails
- mark the full benchmark incomplete if required paper metrics are missing or
  schema-incompatible for any row
- preserve partial child artifacts for debugging, but do not publish merged
  paper tables as final
- when a long run is launched, track the exact PID and accept completion only
  after PID exit `0` plus fresh required artifacts

## Backlog Decomposition

This design should become three execution items:

1. FFNO CDI/grid-lines generator support
   - Backlog item:
     `docs/backlog/active/2026-04-27-cdi-ffno-generator-lines-best-config.md`
   - Implement or expose FFNO in the Torch CDI/grid-lines runner path.
   - Verify it can run the same `N=128` train/test NPZ contract and emit the
     standard reconstruction/metric artifacts.

2. Lines128 paper benchmark harness
   - Backlog item:
     `docs/backlog/active/2026-04-29-cdi-lines128-paper-benchmark-harness.md`
   - Extend `grid_lines_compare_wrapper.py` for
     `spectral_resnet_bottleneck_net` and FFNO, or add a thin paper harness that
     reuses wrapper helpers and calls `grid_lines_torch_runner.py`.
   - Add collation for JSON/CSV/TeX metrics and fixed-sample reconstruction
     figures.
   - Produce the pre-run contract-reconstruction validation artifact and durable
     FNO/seed decision note before enabling full-run launch.

3. Lines128 paper benchmark execution and summary
   - Backlog item:
     `docs/backlog/active/2026-04-29-cdi-lines128-paper-benchmark-execution.md`
   - Run all selected rows under the frozen contract.
   - Validate metrics, visuals, provenance, and freshness.
   - Publish the durable summary and study-index entry.

An optional fourth item may mirror final paper-facing assets into
`/home/ollie/Documents/neurips/` if the roadmap evidence-bundle phase wants that
separate from benchmark execution.

## Documentation And Discoverability

Implementation plans should update:

- `docs/index.md` for durable design/summary discovery
- `docs/studies/index.md` after successful benchmark execution
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_plan.md`
  only if the FFNO generator scope changes
- `docs/model_baselines.md` only if the project intentionally promotes a new
  CDI/grid-lines baseline; do not edit it merely to report a benchmark result
- `/home/ollie/Documents/neurips/index.md` only during paper evidence-bundle
  assembly or when the roadmap explicitly requests manuscript-facing indexing

## Verification Strategy

Before full execution:

- run targeted unit/collection checks for any changed runner/wrapper tests
- run `pytest --collect-only` on new or renamed test modules
- produce and review the contract-reconstruction validation artifact
- produce the durable FNO comparator and seed-policy decision note
- run a wrapper/harness parse or dry-run check
- run a one-row tiny-epoch smoke for each newly supported architecture path,
  especially FFNO and `spectral_resnet_bottleneck_net`
- inspect child invocations to confirm the fixed contract is actually passed

For full execution:

- launch from the repo root in the required `ptycho311` environment
- use tmux for the long run
- use a unique output root
- track the exact launched PID
- reject duplicate writers to the same output root

Post-run acceptance criteria:

- tracked PID exits `0`
- all required child row directories exist
- every required row has fresh invocation/config/log/metrics/reconstruction
  artifacts
- merged JSON/CSV/TeX metrics tables exist and include all rows
- `metric_schema.json` exists and the merged result is either
  `paper_complete` with every required metric present, or
  `benchmark_incomplete` with explicit missing-field reasons
- visual comparison figures exist and use the same sample IDs and shared scales
- manifest records git/environment/GPU/dataset/model/provenance
- manifest records the pre-run contract reconstruction, selected FNO comparator,
  and fixed-seed or approved multi-seed policy
- durable summary and study-index entry point to the final root

## Rollback And Handoff

Rollback is straightforward for code changes: remove the wrapper/harness and
FFNO-runner patches from the implementation commit if verification fails. Do not
delete failed run roots until their logs and manifests have been inspected.

Downstream phases may rely on:

- the frozen benchmark contract in this design
- the pre-run contract-reconstruction validation artifact
- the durable FNO comparator and seed-policy decision note
- the post-run paper benchmark manifest
- merged metrics tables
- source reconstruction arrays and comparison figures
- the durable NeurIPS summary and `docs/studies/index.md` result entry

Open risks:

- FFNO may require a nontrivial adapter before it can behave as a CDI generator.
- The legacy-best `40` epoch contract and the current `50` epoch Hybrid ResNet
  anchor plan differ; the project must choose one before launch and keep it
  fixed for every row.
- The selected FNO comparator is still a pre-run policy choice.
