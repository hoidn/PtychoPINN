# CDI Hybrid-Spectral To FFNO Parameter-Space Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a bounded CDI-only architecture-space study on the fixed `lines128` contract that explains which encoder/downsampling, decoder, and bottleneck changes matter when moving from the current Hybrid-spectral family toward FFNO-like variants.

**Architecture:** Reuse the already-completed `lines128` CDI authorities as fixed anchors, add only the missing attributable intermediate-row plumbing, and run a thin same-contract study harness that compares reused anchor rows plus a small number of fresh bridge rows. Keep the study CDI-only, equal-footing, and decision-support-focused unless a later checked-in authority explicitly promotes any fresh row into paper-facing CDI evidence.

**Tech Stack:** Python 3.11, PyTorch/Lightning, `scripts/studies/grid_lines_compare_wrapper.py`, `scripts/studies/grid_lines_torch_runner.py`, `ptycho_torch/generators/*`, Markdown/JSON/CSV artifacts, tmux-backed long runs in `ptycho311`.

---

## Inputs Read

- `docs/index.md`
- `docs/findings.md`
- `docs/steering.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/selected-item-context.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_preflight.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/spectral_resnet_bottleneck_n128_integration_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`
- existing plan at this path
- `.artifacts/review/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi-plan-review.json`

## Objective

- Run the CDI/ptycho half of the Hybrid-spectral to FFNO architecture-space study on the best study-indexed `lines128` configuration, now that the prerequisite CDI FFNO generator baseline exists.
- Compare a bounded set of intermediate architecture points between the current Hybrid-spectral family and FFNO-like variants on CDI only.
- Keep each fresh row attributable to exactly one axis change:
  - encoder/downsampling
  - decoder
  - bottleneck

## Scope

- Reuse the fixed `N=128`, `gridsize=1`, `set_phi=True`, `seed=3`, `40`-epoch `lines128` CDI contract already captured in:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_preflight.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
- Reuse existing same-contract anchor rows instead of rerunning them unless contract or provenance validation fails:
  - `pinn_hybrid_resnet`
  - `pinn_spectral_resnet_bottleneck_net`
  - `pinn_ffno`
- Reuse the authoritative six-row `lines128` CDI bundle as the default anchor source root:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- Add exactly three fresh bridge rows, no more and no fewer:
  - `pinn_spectral_resnet_bottleneck_ds1`
  - `pinn_spectral_resnet_bottleneck_linear_decoder`
  - `pinn_hybrid_resnet_ffno_bottleneck`
- Collate reused and fresh rows into one same-contract CDI study summary that explains per-axis effects without collapsing CDI and PDE evidence into one ranking.

## Explicit Non-Goals

- Do not run PDEBench CNS, SWE, Darcy, OpenFWI, or any other non-CDI workloads from this item.
- Do not use PDEBench metrics, CNS summaries, or PDE equal-footing results as CDI evidence.
- Do not reopen the `lines128` six-row paper benchmark contract, comparator choice, seed policy, or fixed sample IDs.
- Do not silently expand this into a Cartesian sweep, a multi-seed study, a broader paper-bundle rewrite, or a new roadmap phase.
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.

## Binding Constraints And Prerequisites

- Prerequisite status:
  - backlog prerequisite `2026-04-27-cdi-ffno-generator-lines-best-config` is complete as of `2026-04-29`
  - authoritative complete `lines128` CDI paper bundle is complete as of `2026-04-30`
  - roadmap Phase 0 and Phase 1 are complete in `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Steering constraints:
  - preserve equal-footing comparisons
  - preserve metric, split, and protocol boundaries from the approved design and roadmap
  - do not relax fairness constraints to make the study easier
  - if a compare cannot be kept equal-footing, record it as incompatible instead of drifting the contract
- Roadmap constraints:
  - current roadmap authorization explicitly admits active NeurIPS evidence work across Phase 2 PDEBench, Phase 3 CDI, and `candidate-*` preflights; this plan is executable under that opened gate as of `2026-04-30`
  - within that opened gate, remaining Phase 2 PDEBench evidence stays preferred by priority and steering value, while this Phase 3 CDI item is authorized as useful parallel work because the CDI FFNO prerequisite and authoritative `lines128` bundle already exist
  - this item is Phase 3 CDI comparison work only
  - it must preserve separate phase accounting: this study may run now, but it must not count as satisfying, replacing, or reprioritizing remaining Phase 2 PDEBench evidence
  - it must not consume or reinterpret Phase 2 PDE readiness-only evidence
  - result-producing work must update the NeurIPS evidence indexes before completion
- Grid-lines findings that remain mandatory:
  - preserve `object_big=False` and `probe_big=False` parity for the Torch runner
  - preserve probe preprocessing provenance as an explicit pipeline/contract
  - optional reporting artifacts must not decide scored-run failure on their own
- Long-run execution policy:
  - before any expensive launch, first get a green targeted test pass, then run the required deterministic backlog checks
  - long runs must use tmux, activate `ptycho311`, track the launched PID, and wait on that exact PID
  - do not launch a duplicate run into an output root that is already active
  - do not mark the item `BLOCKED` for routine import/test/path/harness failures; diagnose, fix, and rerun first

## Implementation Architecture

- **Contract and matrix authority:** a checked-in CDI study preflight note plus machine-readable row matrix and reference-run manifests freeze the reused anchors, fresh-row semantics, promotion boundaries, and output roots before any full training starts.
- **Runner and model surface:** the Torch runner and generator registry own architecture construction and row-local provenance; add only the minimum new row-manifest or generator support needed to express attributable CDI bridge rows without broad config-surface churn.
- **Study harness and collation:** a thin CDI parameter-space harness reuses `grid_lines_compare_wrapper.py` dataset/provenance helpers, launches fresh rows, ingests reused rows, writes anchored comparison artifacts, and emits the durable summary plus evidence-index updates.

## Concrete File And Artifact Targets

- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify only if required for new bridge rows:
  - `ptycho_torch/generators/registry.py`
  - `ptycho_torch/generators/README.md`
  - `ptycho_torch/generators/ffno_bottleneck.py`
  - `ptycho_torch/generators/hybrid_resnet.py`
  - `ptycho_torch/generators/spectral_resnet_bottleneck.py`
- Create if the fresh bridge rows cannot be expressed as pure runner overrides:
  - `ptycho_torch/generators/spectral_resnet_bottleneck_linear_decoder.py`
  - `ptycho_torch/generators/hybrid_resnet_ffno_bottleneck.py`
- Create or modify a thin study-owned runbook/harness under `scripts/studies/runbooks/` for this bounded CDI matrix. Prefer a dedicated file over overloading the PDEBench or broad sweep runbooks.
- Add or extend tests in:
  - `tests/test_grid_lines_compare_wrapper.py`
  - `tests/torch/test_grid_lines_torch_runner.py`
  - `tests/torch/test_generator_registry.py`
  - `tests/torch/test_ffno_bottleneck.py`
  - `tests/torch/test_spectral_resnet_bottleneck.py`
  - `tests/torch/test_hybrid_resnet.py`
  - one new focused CDI study-harness test module under `tests/studies/` if the new harness owns matrix or collation logic
- Create durable study artifacts under:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/`
- Create checked-in summaries under:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_hybrid_spectral_ffno_parameter_space_preflight.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_hybrid_spectral_ffno_parameter_space_summary.md`
- Update discoverability/index surfaces as warranted:
  - `docs/studies/index.md`
  - `docs/index.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - update `paper_evidence_index.md` only if a fresh row is explicitly promoted into paper-facing claim territory

## Required Row Contract

- Shared fixed CDI contract for every reused and fresh row:
  - `N=128`
  - `gridsize=1`
  - synthetic grid-lines with `set_phi=True`
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
- Use distinct `model_id` values for fresh bridge rows even when they share a base architecture class, so outputs, summaries, and manifests remain attributable.

## Frozen Fresh Bridge Row Roster

Preflight is not allowed to invent or swap rows. It may only validate, mark a
named row incompatible, or require a narrow documented implementation fix for
the named row below.

### Reused Anchor Rows

- `pinn_hybrid_resnet`
  - reuse from the authoritative complete-table root above
  - nearest-anchor role: Hybrid shell baseline for bottleneck-only comparison
- `pinn_spectral_resnet_bottleneck_net`
  - reuse from the authoritative complete-table root above
  - nearest-anchor role: Hybrid-spectral shell baseline for encoder/downsampling
    and decoder-only comparisons
- `pinn_ffno`
  - reuse from the authoritative complete-table root above
  - role: FFNO endpoint reference only; do not treat it as the nearest-anchor
    control for any fresh row

### Fresh Row 1: Encoder/Downsampling Bridge

- `model_id`: `pinn_spectral_resnet_bottleneck_ds1`
- nearest reused anchor: `pinn_spectral_resnet_bottleneck_net`
- expression path: runner-only override on existing
  `architecture=spectral_resnet_bottleneck_net`
- exact override payload:
  - `hybrid_downsample_steps=1`
  - keep `hybrid_downsample_op=stride_conv`
  - keep the spectral bottleneck settings identical to the anchor:
    `spectral_bottleneck_blocks=6`,
    `spectral_bottleneck_modes=12`,
    `spectral_bottleneck_share_weights=True`,
    `spectral_bottleneck_gate_mode=shared`
- attribution rule:
  - the intended study change is reduced encoder/downsampling depth
  - the mirrored one-stage upsample path is treated as the forced shell
    consequence of that encoder-depth change, not as a second decoder-family
    experiment
- boundary:
  - do not attempt `hybrid_downsample_steps=0` in this item
  - a fully constant-resolution Hybrid-spectral shell is out of scope unless a
    later checked-in design amendment explicitly authorizes it

### Fresh Row 2: Decoder Bridge

- `model_id`: `pinn_spectral_resnet_bottleneck_linear_decoder`
- nearest reused anchor: `pinn_spectral_resnet_bottleneck_net`
- expression path: new generator registry entry
  `architecture=spectral_resnet_bottleneck_linear_decoder`
- required shell semantics:
  - preserve the anchor's encoder, two-step downsampling schedule, channel
    widths, spectral bottleneck family, output mode, and fixed CDI contract
  - replace only the CycleGAN transpose-convolution decoder with a lighter
    FFNO-adjacent decoder:
    - bilinear upsample by `2x`
    - `1x1` channel projection after each upsample stage
    - final projection back to the existing real/imag output contract
- attribution rule:
  - no encoder/downsampling, bottleneck, dataset, or metric change is allowed
    for this row
- fallback rule:
  - if this exact lighter decoder cannot preserve the fixed CDI output contract
    after a narrow implementation attempt, record a row-level blocker for
    `pinn_spectral_resnet_bottleneck_linear_decoder`; do not substitute a
    different decoder experiment

### Fresh Row 3: Bottleneck Bridge

- `model_id`: `pinn_hybrid_resnet_ffno_bottleneck`
- nearest reused anchor: `pinn_hybrid_resnet`
- expression path: new generator registry entry
  `architecture=hybrid_resnet_ffno_bottleneck`
- required shell semantics:
  - preserve the anchor's encoder, two-step downsampling schedule, decoder
    family, output head, and fixed CDI contract
  - replace only the ResNet bottleneck with
    `SharedFactorizedFfnoBottleneck` from
    `ptycho_torch/generators/ffno_bottleneck.py`
- exact bottleneck settings:
  - `n_blocks=6`
  - `modes=12`
  - `share_spectral_weights=True`
  - `mlp_ratio=2.0`
  - `gate_init=0.1`
  - `norm=instance`
  - `local_conv_kernel_size=None`
- attribution rule:
  - no encoder/downsampling, decoder, dataset, or metric change is allowed for
    this row

### Frozen Matrix Summary

- reused rows:
  - `pinn_hybrid_resnet`
  - `pinn_spectral_resnet_bottleneck_net`
  - `pinn_ffno`
- fresh rows:
  - `pinn_spectral_resnet_bottleneck_ds1`
  - `pinn_spectral_resnet_bottleneck_linear_decoder`
  - `pinn_hybrid_resnet_ffno_bottleneck`
- no other fresh architecture points are authorized in this item

## Task 1: Freeze The CDI Study Matrix Before Any Full Launch

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_hybrid_spectral_ffno_parameter_space_preflight.md`
- Create: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/preflight/study_matrix.json`
- Create: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/preflight/reference_runs.json`
- Update if needed: `docs/studies/index.md`

- [ ] Reconstruct the fixed CDI contract from the completed `lines128` authorities and write the checked-in preflight note.
- [ ] Carry forward the roadmap execution authorization into the preflight note: this study is allowed under the current opened Phase 2/Phase 3 parallel gate, but it remains Phase 3 CDI work and does not satisfy any remaining Phase 2 PDEBench requirement.
- [ ] Record the exact reused anchor roots from the authoritative complete-table root and map them to row IDs in `reference_runs.json`.
- [ ] Freeze the exact fresh-row roster from `Frozen Fresh Bridge Row Roster` above. The preflight note must repeat each row's `model_id`, nearest anchor, architecture key, and row-local override or generator-expression path.
- [ ] Record the output-root layout and display labels for the six-row study matrix.
- [ ] State clearly that the study is CDI-only decision-support evidence unless a later authority explicitly promotes a row.
- [ ] Fail closed in preflight if any reused root is contract-incompatible or lacks recoverable provenance. Reuse the row only after the preflight note records why it is acceptable.
- [ ] Do not let preflight rename, add, remove, or substitute bridge rows. Any inability to execute one named row must be recorded as that row's blocker, not as license to widen or redirect the matrix.

**Verification before moving on:**
- [ ] Add a small validation command or test that parses `study_matrix.json` and `reference_runs.json` and asserts the matrix contains exactly these six row IDs:
  - `pinn_hybrid_resnet`
  - `pinn_spectral_resnet_bottleneck_net`
  - `pinn_ffno`
  - `pinn_spectral_resnet_bottleneck_ds1`
  - `pinn_spectral_resnet_bottleneck_linear_decoder`
  - `pinn_hybrid_resnet_ffno_bottleneck`
- [ ] Run targeted tests for any new preflight/matrix helper.
- [ ] No full training may start until this preflight note and both machine-readable manifests exist and validate.

## Task 2: Add The Minimum Row-Plumbing Needed For Attributable CDI Bridge Rows

**Files:**
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify if needed: `ptycho_torch/generators/registry.py`
- Modify if needed: `ptycho_torch/generators/ffno_bottleneck.py`
- Modify if needed: `ptycho_torch/generators/hybrid_resnet.py`
- Modify if needed: `ptycho_torch/generators/spectral_resnet_bottleneck.py`
- Create if needed: `ptycho_torch/generators/spectral_resnet_bottleneck_linear_decoder.py`
- Create if needed: `ptycho_torch/generators/hybrid_resnet_ffno_bottleneck.py`
- Modify/add tests:
  - `tests/test_grid_lines_compare_wrapper.py`
  - `tests/torch/test_grid_lines_torch_runner.py`
  - `tests/torch/test_generator_registry.py`
  - `tests/torch/test_ffno_bottleneck.py`
  - `tests/torch/test_spectral_resnet_bottleneck.py`
  - `tests/torch/test_hybrid_resnet.py`

- [ ] Extend the compare/wrapper surface so one study can reference both reused row roots and fresh row definitions without pretending every row is a brand-new architecture launch.
- [ ] Add study-owned row-manifest support so multiple same-base rows can have distinct `model_id`, label, and override payloads.
- [ ] Implement `pinn_spectral_resnet_bottleneck_ds1` as a runner-only override on `spectral_resnet_bottleneck_net`; do not introduce a new architecture key for this row.
- [ ] Implement `pinn_spectral_resnet_bottleneck_linear_decoder` as the smallest new generator/registry extension that preserves the fixed output contract and changes only the decoder family described above.
- [ ] Implement `pinn_hybrid_resnet_ffno_bottleneck` as the smallest new generator/registry extension that preserves the fixed output contract and changes only the bottleneck family described above.
- [ ] Keep all new knobs runner-owned or study-owned unless they are broadly reusable. Do not widen the canonical config bridge for one-off study metadata.
- [ ] Preserve existing complete-table and compare-wrapper behavior for already-supported rows.

**Verification before moving on:**
- [ ] Add or update unit tests covering:
  - row-manifest parsing and validation
  - routing of same-base rows with distinct `model_id`
  - any new generator registration
  - output-contract shape preservation for any new decoder/bottleneck bridge row
- [ ] Run those focused selectors first.
- [ ] Then run the required deterministic backlog checks before any expensive study run:
  - `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  - `python -m compileall -q ptycho_torch scripts/studies`
- [ ] If these checks fail, fix the issue and rerun them before continuing.

## Task 3: Add A Thin CDI Parameter-Space Harness And Prove Preflight-Only Execution

**Files:**
- Create or modify one dedicated runbook/harness under `scripts/studies/runbooks/`
- Add one focused harness test under `tests/studies/` if matrix expansion or anchored collation logic is non-trivial

- [ ] Build a thin harness that:
  - loads the frozen study matrix
  - validates reused-root provenance
  - launches only the declared fresh rows
  - writes invocation artifacts and launcher logs
  - collates reused plus fresh rows into one anchored comparison bundle
- [ ] Support a `preflight_only` or equivalent dry-run mode that validates row routing and output-root ownership without launching training.
- [ ] Make the harness refuse duplicate active output roots and write explicit completion metadata.
- [ ] Make the harness emit per-row anchor metadata so the final summary can report every fresh row against its nearest reused anchor and against the reused `pinn_ffno` endpoint.

**Verification before moving on:**
- [ ] Run the harness in preflight-only mode and archive the resulting validation log under the backlog artifact root.
- [ ] Validate that the preflight bundle includes the frozen row roster, expected root paths, and no accidental extra rows.
- [ ] Re-run the required backlog checks if the harness touched the runner or wrapper surfaces again after Task 2.

## Task 4: Launch The Fresh CDI Bridge Rows Under The Frozen Contract

**Files/Artifacts:**
- Create fresh run roots under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/runs/`
- Create compare artifacts under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/analysis/`

- [ ] Launch only the fresh bridge rows defined in the frozen matrix. Do not rerun reused anchors unless the preflight explicitly required it.
- [ ] Launch order is fixed unless a narrower dependency reason requires reordering:
  - `pinn_spectral_resnet_bottleneck_ds1`
  - `pinn_spectral_resnet_bottleneck_linear_decoder`
  - `pinn_hybrid_resnet_ffno_bottleneck`
- [ ] Use tmux plus `ptycho311` for each long-running command and track the exact PID until exit `0`.
- [ ] Keep the study same-contract and fixed-seed. Do not mix in one-epoch smoke metrics, alternate probes, changed datasets, or altered schedulers.
- [ ] If a new row fails due to routine implementation issues, repair the narrow issue and relaunch that row. Reserve `BLOCKED` for unrecoverable external blockers only.
- [ ] After training completes, collate reused and fresh rows into anchored JSON/CSV/TeX metrics plus shared visuals and model manifests.

**Verification before closeout:**
- [ ] Validate the final study bundle with a deterministic script that checks:
  - every declared row exists
  - reused rows point to the frozen authoritative roots
  - every fresh row has metrics, history, reconstructions, invocation artifacts, and completion proof
  - the merged comparison outputs parse and include the expected row IDs
- [ ] If any required artifact is missing, fix the harness or rerun the affected row before writing the summary.

## Task 5: Write The Durable Summary And Update Evidence Indexes

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_hybrid_spectral_ffno_parameter_space_summary.md`
- Update: `docs/studies/index.md`
- Update: `docs/index.md`
- Update as applicable:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/findings.md` only if a reusable new rule or failure mode was discovered

- [ ] Summarize the per-axis CDI result:
  - which bridge rows were launched
  - which rows were reused
  - what changed on encoder/downsampling, decoder, and bottleneck axes
  - each fresh row's nearest reused anchor and its relation to the reused
    `pinn_ffno` endpoint
  - whether any fresh row is worth later promotion into default CDI comparisons or paper-facing follow-up
- [ ] Keep CDI outcomes separate from PDEBench outcomes.
- [ ] State the claim boundary explicitly. If the study remains decision-support-only, say so and do not imply paper promotion.
- [ ] Update the evidence indexes required by the roadmap:
  - `model_variant_index.json` for any new evaluated model rows
  - `ablation_index.json` for the architecture-space study itself
  - `evidence_matrix.md` for durable outputs and comparison bundles
  - `paper_evidence_index.md` only if the claim boundary changes

**Final verification and archived proof:**
- [ ] Re-run the backlog item’s required deterministic checks and archive their logs:
  - `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  - `python -m compileall -q ptycho_torch scripts/studies`
- [ ] Supporting check for production workflow-surface risk; do not treat as a blocking deterministic gate when required `check_commands` and equivalent focused evidence cover the changed paths:
  - `pytest -v -m integration`
- [ ] Archive the focused unit-test logs, preflight log, long-run launcher log, final artifact-validation log, and closeout verification logs under the backlog artifact root.

## Required Commands

Use these as the non-negotiable deterministic checks for the item:

```bash
pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py
python -m compileall -q ptycho_torch scripts/studies
```

Recommended supporting checks for this item because it changes workflow surfaces:

```bash
pytest -v -m integration
```

## Long-Run Command Policy

- Launch long runs in tmux and activate `ptycho311` inside the tmux pane.
- Track the exact PID for each launched command and wait on that PID.
- Treat a run as complete only when:
  - the tracked PID exits `0`
  - required row artifacts exist
  - the merged study outputs validate successfully
- Do not launch a second run into the same fresh output root while one is already active.

## Completion Criteria

- The preflight note, machine-readable row matrix, and reference-run manifest exist and match the fixed CDI contract.
- The study launches only the bounded fresh CDI bridge rows and reuses anchor rows without contract drift.
- The final summary explains encoder/downsampling, decoder, and bottleneck effects on CDI/ptycho separately and truthfully.
- Required deterministic checks pass and logs are archived.
- NeurIPS evidence indexes are updated or explicitly left unchanged with a reason.
