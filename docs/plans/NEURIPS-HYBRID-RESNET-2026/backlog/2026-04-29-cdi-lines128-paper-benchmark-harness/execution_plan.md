# Lines128 Paper Benchmark Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the execution-ready `lines128` paper benchmark harness and preflight layer so later benchmark execution can run Hybrid ResNet, a CNN/PINN-style local baseline, a preselected FNO comparator, and route or explicitly block `spectral_resnet_bottleneck_net` and FFNO under one fixed CDI contract without launching the full paper benchmark in this item.

**Architecture:** Keep `scripts/studies/grid_lines_compare_wrapper.py` as the shared dataset/provenance/collation owner and `scripts/studies/grid_lines_torch_runner.py` as the per-row execution owner. Extend wrapper and collation helpers for paper-grade decision manifests, metric-schema enforcement, and fixed-sample visual collation; introduce a thin dedicated harness helper only if those additions would otherwise tangle the default compare wrapper. This item ends at green preflight, routing, schema, and bounded validation artifacts; the later full multi-row paper benchmark remains a separate backlog item.

**Tech Stack:** PATH `python`, PyTorch Lightning, `ptycho_torch` generator registry, `scripts/studies/grid_lines_compare_wrapper.py`, `scripts/studies/grid_lines_torch_runner.py`, `scripts/studies/metrics_tables.py`, pytest, `compileall`, Markdown/JSON/CSV artifacts, repo-local `.artifacts/` validation roots.

---

## Selected Objective

- Implement backlog item `2026-04-29-cdi-lines128-paper-benchmark-harness`.
- Extend the CDI/grid-lines benchmark path so one paper-quality wrapper or a thin harness can own shared contract reconstruction, routing, provenance, metrics, and visuals for the later `lines128` benchmark.
- Support the minimum draftable subset now: `pinn_hybrid_resnet`, one CNN/PINN-style local baseline, and the selected FNO comparator.
- Support `spectral_resnet_bottleneck_net` and FFNO through the same harness contract when possible, or emit explicit row-level blockers that preserve the core harness and claim boundaries.

## Scope And Explicit Non-Goals

In scope:

- Preserve the recovered study-indexed `N=128` legacy-best CDI contract and make it executable through a paper-benchmark harness/preflight path.
- Produce a durable checked-in contract-reconstruction validation note and a machine-readable benchmark decision manifest before the later full benchmark can launch.
- Add routing and collation support for the minimum paper subset plus `spectral_resnet_bottleneck_net` and FFNO row-state reporting.
- Enforce the paper metric schema so missing required fields downgrade merged results to `benchmark_incomplete` instead of silently passing.
- Add or update focused tests for routing, schema enforcement, visual-collation invariants, and preflight failure modes.
- Run only bounded validation work, such as parse checks, preflight-only flows, or tiny smoke proofs for newly supported paths when unit tests alone do not prove the route.

Explicit non-goals:

- Do not launch the full multi-row `lines128` paper benchmark from this item.
- Do not write `/home/ollie/Documents/neurips/` evidence-bundle outputs.
- Do not silently choose `fno` versus `fno_vanilla` in code without the durable decision artifact.
- Do not block the minimum Hybrid/CNN-PINN/FNO harness on `spectral_resnet_bottleneck_net` or FFNO availability.
- Do not change the CDI dataset, probe, split, scheduler, loss, output, or metric contract to make FFNO easier.
- Do not promote historical incomplete roots to paper-grade results.
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.

## Binding Constraints

- Steering constraints:
  - This item is valid because the current selection window allows Phase 2 plus Phase 3 CDI-preparation work.
  - Preserve approved roadmap gates and equal-footing comparison standards.
  - Do not spend scope on optional later evidence-bundle work.
- Roadmap constraints:
  - This is Roadmap Phase 3 CDI-preparation work under item `3.3a`.
  - The output must support the later minimum draftable subset under `3.3b` and the complete `lines128` table under `3.3c`, but this item itself stops at harness/preflight readiness.
  - The complete CDI benchmark later requires `hybrid_resnet`, `spectral_resnet_bottleneck_net`, the selected FNO comparator, and FFNO, or explicit row-level blockers / a checked-in design amendment.
- Backlog-item constraints:
  - Prefer extending `scripts/studies/grid_lines_compare_wrapper.py`; add a thin dedicated harness only if the existing wrapper cannot cleanly own paper-benchmark collation.
  - Keep `scripts/studies/grid_lines_torch_runner.py` as the authority for model construction, training, inference, stitching, per-model metrics, and reconstruction arrays.
  - Missing spectral/FFNO support must not block the core harness unless the shared contract itself cannot be represented.
  - The row-level blocker path is acceptable for spectral/FFNO only when the contract cannot be satisfied yet; it is not acceptable for the minimum Hybrid/CNN-PINN/FNO subset.
- Fixed `lines128` contract to preserve unless a later checked-in design override changes every row together:
  - `N=128`, `gridsize=1`, synthetic grid-lines, `set_phi=True`
  - custom Run1084 probe
  - `probe_scale_mode=pad_extrapolate`, `probe_smoothing_sigma=0.5`
  - `nimgs_train=2`, `nimgs_test=2`, `nphotons=1e9`
  - `seed=3`
  - `torch_epochs=40`, `torch_learning_rate=2e-4`
  - `torch_scheduler=ReduceLROnPlateau`, `torch_plateau_factor=0.5`, `torch_plateau_patience=2`, `torch_plateau_min_lr=1e-4`, `torch_plateau_threshold=0.0`
  - `torch_loss_mode=mae`, `torch_mae_pred_l2_match_target=off`
  - `torch_output_mode=real_imag`, `probe_mask=off`
  - `fno_modes=12`, `fno_width=32`, `fno_blocks=4`, `fno_cnn_blocks=2`
- Metric and figure constraints from the approved `lines128` design:
  - The harness must be able to emit `metrics.json`, `metrics_table.csv`, `metrics_table.tex`, `metrics_table_best.tex`, and `metric_schema.json`.
  - Required per-row fields are: model key and display label, parameter count, epoch budget, final completed epoch, final train loss, validation-loss status/value, amplitude and phase `mae/mse/psnr/ssim/ms_ssim/frc50`, runtime/hardware summary, and caveat fields.
  - Paper-grade status is `paper_complete` only if every required field is present or explicitly accepted as not applicable.
  - Otherwise the merged result must be labeled `benchmark_incomplete` with explicit missing-field reasons.
  - Visual outputs must preserve one fixed set of sample IDs, shared color scales, ground truth, reconstructions, absolute-error panels, source numeric arrays, and a combined `compare_amp_phase` figure plus model-specific figures.
- Findings and workflow constraints:
  - Preserve `GRIDLINES-OBJECT-BIG-001` and `GRIDLINES-PROBE-BIG-001` when touching grid-lines Torch configuration.
  - Preserve probe-processing provenance per `GRIDLINES-PROBE-PIPELINE-001`.
  - Treat optional reporting helpers as non-fatal per `REPORTING-ARTIFACT-BOUNDARY-001`, but keep the required paper visuals and schema artifacts mandatory for harness acceptance.
  - If runner or wrapper config plumbing changes, keep the required `update_legacy_dict(params.cfg, config)` bridge behavior intact before legacy-backed loading/execution.
  - Use PATH `python` in commands and examples.
- Failure policy:
  - Do not mark this item `BLOCKED` for ordinary import, environment, test, path, or harness failures.
  - Diagnose, patch narrowly, and rerun first.
  - Reserve `BLOCKED` for missing external resources, unavailable hardware, roadmap conflict, user decision required, or unresolved CDI-contract incompatibility after one narrow documented fix cycle.

## Prerequisite Status And Starting Point

- Progress-ledger state that matters:
  - `phase-0-evidence-inventory` is complete.
  - `phase-1-pde-benchmark-selection` is complete.
  - The current roadmap/steering window permits CDI-preparation work in parallel with remaining Phase 2 PDE work.
- Existing durable CDI prerequisites already available:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md` defines the fixed benchmark contract, metric schema, visual requirements, and decomposition.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md` documents the completed fixed-contract `pinn_ffno` versus `pinn_hybrid_resnet` pair as prerequisite evidence, not as the full benchmark harness.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_summary.md` records the stable two-row FFNO slice that later harness work can consume.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/spectral_resnet_bottleneck_n128_integration_summary.md` proves `spectral_resnet_bottleneck_net` can traverse the Torch/grid-lines path in smoke form, but not yet as paper-grade harness evidence.
- Starting implication:
  - FFNO is no longer a blank-slate route; the harness should reuse that prerequisite evidence or its contract decisions where possible.
  - `spectral_resnet_bottleneck_net` routing should be added to the harness or explicitly blocked at the row level, but it should not hold the minimum Hybrid/CNN-PINN/FNO harness hostage.

## Implementation Architecture

- Contract and decision unit:
  - One checked-in preflight/contract note plus one machine-readable decision manifest that freeze the row roster, selected FNO comparator, seed policy, reconstructed contract fields, confidence sources, approved deviations, and go/no-go state.
- Routing and orchestration unit:
  - The compare wrapper remains the default shared owner for dataset identity, child-run normalization, provenance capture, and merged artifact collation.
  - If paper-only behavior would overcomplicate the default wrapper, add a thin harness script that delegates actual row execution to the existing wrapper/runner helpers instead of duplicating them.
- Schema and visual-collation unit:
  - Table/schema helpers own required metric columns, downgrade rules, model labels, and emitted `metric_schema.json`.
  - Wrapper/harness collation owns fixed-sample selection, visual ordering, shared color-scale metadata, row-level completion/blocker status, and later benchmark-readiness reporting.

## Concrete File And Artifact Targets

- Likely code targets:
  - `scripts/studies/grid_lines_compare_wrapper.py`
  - `scripts/studies/metrics_tables.py`
  - `scripts/studies/grid_lines_torch_runner.py`
  - optional only if wrapper boundaries become messy: `scripts/studies/lines128_paper_benchmark.py`
- Likely test targets:
  - `tests/test_grid_lines_compare_wrapper.py`
  - `tests/studies/test_metrics_tables.py`
  - `tests/torch/test_grid_lines_torch_runner.py`
  - optional only if a new harness helper is added: `tests/studies/test_lines128_paper_benchmark.py`
- Durable docs and machine-readable artifacts:
  - Keep as prerequisite evidence: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md`
  - Create or update the harness contract authority: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_preflight.md`
  - Create or update a harness completion summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_summary.md`
  - Harness artifact root: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/`
  - Decision manifest under the harness root: `preflight/benchmark_decisions.json`
  - If preflight/schema validation emits a manifest, keep it under the harness root, not `/home/ollie/Documents/neurips/`
- Discoverability surfaces if the durable docs above are added or materially repurposed:
  - `docs/index.md`
  - `docs/studies/index.md`
- Update `docs/findings.md` only if this work exposes a durable future-facing contract pitfall that would otherwise be rediscovered.

### Task 1: Freeze The Harness Contract And Decision Surfaces

**Files / artifacts:**
- Read and preserve as prerequisite context: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md`
- Create or update: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_preflight.md`
- Emit: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json`

- [ ] Reconstruct the benchmark contract from the approved design plus the completed FFNO-versus-Hybrid prerequisite slice, but keep the two-row preflight as historical prerequisite evidence rather than overwriting it.
- [ ] Name the minimum harness row roster explicitly: `pinn_hybrid_resnet`, one CNN/PINN-style local baseline, and the selected FNO comparator.
- [ ] Record the status of `spectral_resnet_bottleneck_net` and FFNO as either `supported_for_harness` or a row-level blocker with the exact reason.
- [ ] Decide and record the FNO comparator before code paths depend on it; the decision artifact must state whether the benchmark uses `fno` or `fno_vanilla` and why.
- [ ] Record the fixed `seed=3` policy unless a reviewed multi-seed extension is explicitly approved.
- [ ] Record any approved deviation from the fixed contract; absence of such a record means implementation must preserve the contract exactly.
- [ ] Add a go/no-go section that states this item authorizes only harness/preflight and bounded validation, not the later full benchmark launch.

**Verification before moving on:**
- [ ] The checked-in harness preflight names the historical sources, confidence for reconstructed fields, selected FNO comparator, seed policy, and go/no-go state.
- [ ] `benchmark_decisions.json` agrees with the checked-in preflight on row roster, comparator choice, seed policy, and deviations.
- [ ] The prerequisite FFNO-versus-Hybrid preflight remains preserved as a separate evidence slice.

### Task 2: Extend Wrapper Routing Without Breaking Runner Authority

**Files:**
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Conditional modify only if required for harness metadata or row-state exposure: `scripts/studies/grid_lines_torch_runner.py`
- Optional add only if wrapper extension becomes too tangled: `scripts/studies/lines128_paper_benchmark.py`

- [ ] Extend the wrapper path so the harness can target the minimum paper subset and can route `spectral_resnet_bottleneck_net` through the existing Torch runner when supported.
- [ ] Keep `grid_lines_torch_runner.py` as the owner of row-local training, inference, stitching, metrics, and reconstruction outputs; do not reimplement row execution in the harness.
- [ ] Add a paper-harness mode or thin wrapper that can consume the checked-in preflight decision surface, emit the machine-readable decision manifest, and preserve row-level status for supported and blocked rows.
- [ ] Require explicit FNO comparator selection from the decision artifact or CLI/harness config; fail closed when the comparator is missing or ambiguous.
- [ ] Require explicit row-level blocker emission for unsupported spectral/FFNO paths instead of silently dropping rows or pretending the benchmark is complete.
- [ ] Preserve invocation logging and repo/runtime provenance for any new or changed wrapper/harness entry point.
- [ ] Avoid breaking existing non-paper compare-wrapper flows; paper-specific enforcement should be opt-in rather than silently changing default behavior.

**Verification before moving on:**
- [ ] Focused routing tests cover the selected FNO comparator path, the spectral path, and row-level blocker emission.
- [ ] The wrapper/harness can parse a full `lines128` contract configuration and emit the preflight/decision artifacts without launching the full benchmark.
- [ ] Existing compare-wrapper behavior remains green for non-paper usage.

### Task 3: Enforce Paper Metric Schema And Fixed-Sample Visual Rules

**Files:**
- Modify: `scripts/studies/metrics_tables.py`
- Modify if visual-collation metadata lives here: `scripts/studies/grid_lines_compare_wrapper.py`
- Update tests: `tests/studies/test_metrics_tables.py`, `tests/test_grid_lines_compare_wrapper.py`

- [ ] Extend the table/helper layer to define the paper metric schema in one place and emit `metric_schema.json` with required fields, units, nullability, and downgrade rules.
- [ ] Ensure merged results downgrade to `benchmark_incomplete` when any required metric field is missing, schema-incompatible, or row-specific.
- [ ] Preserve model labels separately from architecture keys so table output stays reviewer-readable without obscuring architecture identity.
- [ ] Add completion metadata for parameter count, epoch budget, final completed epoch, train loss, validation-loss status, runtime/hardware summary, and caveat fields.
- [ ] Make fixed-sample visual collation deterministic. The harness must preserve one shared sample-ID set and shared visual-scale metadata for every row it collates.
- [ ] Ensure the harness never publishes a merged result that looks complete if one required row failed or one required metric family is missing.

**Verification before moving on:**
- [ ] Table/schema tests prove `paper_complete` versus `benchmark_incomplete` downgrade behavior.
- [ ] Visual-collation tests prove fixed sample IDs and shared-scale metadata are preserved across rows.
- [ ] A bounded harness-validation root can emit `metrics.json`, `metrics_table.csv`, `metrics_table.tex`, `metrics_table_best.tex`, and `metric_schema.json` without pretending a full benchmark completed.

### Task 4: Add Targeted Regression Coverage First, Then Run Mandatory Gates

**Files / artifacts:**
- Tests: `tests/test_grid_lines_compare_wrapper.py`, `tests/studies/test_metrics_tables.py`, `tests/torch/test_grid_lines_torch_runner.py`
- Archive logs under: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/verification/`

- [ ] Add or update focused tests for:
  - selected FNO comparator enforcement
  - `spectral_resnet_bottleneck_net` routing or row-level blocker emission
  - FFNO blocker preservation when the CDI generator contract cannot be satisfied
  - `benchmark_incomplete` downgrade when required metrics are absent
  - fixed-sample visual collation metadata
  - preflight failure on missing decision artifact or unresolved contract note
- [ ] Run focused selectors while iterating on the code.
- [ ] Before any smoke or preflight command that exercises model code, run the backlog item’s required deterministic checks exactly as written below and archive their logs.
- [ ] If any required check fails, diagnose, patch narrowly, and rerun. Do not mark the item `BLOCKED` for ordinary verification failures.

**Required deterministic checks:**

```bash
pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py
python -m compileall -q ptycho_torch scripts/studies
```

**Verification before moving on:**
- [ ] The focused test additions are green.
- [ ] Both required deterministic commands are green and their outputs are archived under the harness verification root.
- [ ] No smoke or later benchmark-launch work proceeds on a red gate.

### Task 5: Produce Bounded Harness Validation Artifacts, Not The Full Benchmark

**Files / artifacts:**
- Harness root: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/`
- Durable summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_summary.md`

- [ ] Run the lightest validation that proves the harness contract is real:
  - parse/preflight-only flow for the full planned row roster
  - cheap collator/schema emission on synthetic or mocked row payloads if available
  - tiny one-row or one-epoch smoke only for newly supported row paths that are not already proven by tests or prerequisite evidence
- [ ] Prefer reusing the completed FFNO-versus-Hybrid prerequisite root as evidence for FFNO contract compatibility when no new FFNO runner behavior changed.
- [ ] Use bounded smoke for `spectral_resnet_bottleneck_net` or the selected FNO comparator only if unit tests and existing durable artifacts do not sufficiently prove the route.
- [ ] If a smoke is needed, keep it outside the later paper benchmark root and label it `readiness_only_not_benchmark_performance`.
- [ ] Emit the harness summary with:
  - supported minimum subset
  - selected FNO comparator
  - seed policy
  - spectral/FFNO support or blocker state
  - location of the decision manifest and schema artifacts
  - explicit statement that the full multi-row benchmark remains unlaunched
- [ ] Update discoverability surfaces if the new harness preflight/summary become durable project knowledge.

**Verification before closing the item:**
- [ ] The harness summary, checked-in harness preflight, and `benchmark_decisions.json` agree on comparator choice, seed policy, supported rows, and blocker rows.
- [ ] If a bounded smoke ran, it is clearly labeled as readiness-only and stored under the harness root.
- [ ] The item ends with a reusable harness/preflight path and does not accidentally claim that the full paper benchmark already ran.

## Documentation And Discoverability Updates

- If `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_preflight.md` or `lines128_paper_benchmark_harness_summary.md` is created, add or update entries in `docs/index.md`.
- Update `docs/studies/index.md` only enough to make the harness/preflight and later execution boundary discoverable; do not log a full benchmark result here yet unless the later execution item runs.
- Update `docs/findings.md` only if this work uncovers a durable CDI/grid-lines contract pitfall that future workers are likely to trip over again.
- Do not update `/home/ollie/Documents/neurips/index.md` from this item.

## Execution Boundary And Handoff

- This item is complete when:
  - the harness contract is frozen in checked-in docs plus `benchmark_decisions.json`
  - the wrapper/harness can represent the minimum paper subset and row-level blocker states
  - schema enforcement and fixed-sample visual rules are tested
  - the required deterministic checks are green
  - any bounded validation artifacts are preserved under the harness root
- This item is not complete merely because the old FFNO-versus-Hybrid pair exists.
- The later `lines128` benchmark execution item may then consume:
  - the checked-in harness preflight
  - the decision manifest
  - schema rules and downgrade behavior
  - routing support for the selected row roster
  - any explicit row-level blockers that still remain
