# PDEBench 2D Diffusion-Reaction Temporal-Context Design

> Superseded, 2026-04-20: the user replaced the active third PDEBench image-suite member with 2D Compressible Navier-Stokes and deleted the abandoned diffusion-reaction dataset download. This design is retained only as deferred background. Active third-task planning now lives in `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_design.md`.

## Design Metadata

- ID: `NEURIPS-HYBRID-RESNET-2026-pdebench-2d-reacdiff`
- Title: PDEBench 2D Diffusion-Reaction Temporal-Context Adapter
- Status: deferred; superseded by the 2D CNS pivot
- Date: 2026-04-20
- Source brief / issue: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- Related plan: pending
- Experiment root: `/home/ollie/Documents/PtychoPINN/`
- Data root: `/home/ollie/Documents/pdebench-data/`
- Manuscript artifact root: `/home/ollie/Documents/neurips/` (future Phase 5 root; this design must not create it)

## Consumed Inputs and Authority

- Docs index: `docs/index.md` read first.
- Project guidance: `AGENTS.md`, `CLAUDE.md`.
- Primary campaign design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`.
- Primary roadmap: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`.
- Suite plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`.
- Current preflight: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_preflight.md`.
- Baseline authority: `docs/model_baselines.md`.
- Local task spec: `scripts/studies/pdebench_image128/task_specs.py`.
- Existing dynamic-task adapter source: `scripts/studies/pdebench_swe/`.
- Official PDEBench source context:
  - PDEBench paper: `https://papers.nips.cc/paper/2022/file/0a9747136d411fb83f0cf81820d44afb-Paper-Datasets_and_Benchmarks.pdf`.
  - PDEBench repository: `https://github.com/pdebench/PDEBench`.
  - PDEBench config reference: `https://raw.githubusercontent.com/pdebench/PDEBench/main/pdebench/models/config/args/config_diff-react.yaml`.

Authority order for this design:

1. User constraint in this discussion: avoid `history_len=10` as the primary path.
2. The NeurIPS Hybrid ResNet submission design and roadmap.
3. The PDEBench `128x128` image-suite plan and current preflight.
4. Official PDEBench data/model conventions, used as protocol context rather than a binding local implementation.

## Problem and Scope

Problem: The PDEBench 2D diffusion-reaction member of the image suite is time-dependent and two-channel. The existing suite plan says it is a dynamic one-step task, but it does not fully specify how multiple successive time points should be grouped into model inputs or how that grouping should be kept fair across Hybrid ResNet, FNO, and U-Net baselines.

User / reviewer / system need: Define a simple, auditable data-grouping contract that lets the supervised real-channel Hybrid ResNet adapter and local baselines consume the same examples without turning the first implementation into a full video or autoregressive training project.

In scope:

- The supervised example shape for PDEBench 2D diffusion-reaction.
- The primary temporal history length and ablation scope.
- Split, normalization, metric, and artifact contracts needed before implementation planning.
- Claim boundaries for one-step, capped, smoke, pilot, and full-training rows.

Out of scope:

- Downloading or staging `2D_diff-react_NA_NA.h5`.
- Implementing loaders, models, metrics, or runners.
- Full autoregressive or pushforward training.
- Manuscript prose or `/home/ollie/Documents/neurips/` artifacts.
- Replacing the current three-task PDEBench image-suite plan.

Non-goals:

- Exact reproduction of PDEBench's official `initial_step=10`, `epochs=500`, or training code.
- Predicting only one chemical species as the headline benchmark.
- Broad history-length sweeps.

## Decision Summary

- Use a trajectory-level dynamic dataset with grouped HDF5 trajectories shaped like `####/data: (T,128,128,2)` after staging verifies the schema.
- Set the primary local benchmark contract to `history_len=2`: input `(4,128,128)` predicts next state target `(2,128,128)`.
- Stable primary shape shorthand: `input=(4,128,128)`, `target=(2,128,128)`.
- Include `history_len=1` as the required low-complexity temporal-context ablation if budget permits.
- Treat `history_len=4` or `history_len=10` as optional later rows only if `K=1/2` results are stable and runtime/memory permit.
- Preserve both physical channels in the target. "Single-channel output" is not the primary PDEBench task unless a later plan explicitly creates a diagnostic species-only ablation.
- Split at trajectory level before window expansion; full benchmark rows must train on all training trajectories and all eligible windows for the selected history length.

## Decision Records

### ADR-001: Primary Temporal Context Is `history_len=2`

- Status: proposed
- Decision: Use two successive prior states as the primary input window: `x = [u(t-2), u(t-1)] -> y = u(t)`.
- Context: PDEBench's official config uses `initial_step=10`, but the user correctly flagged that as too high for the first Hybrid ResNet comparison. A 20-channel input would increase adapter and memory risk before the benchmark proves useful.
- Rationale: `K=2` provides a minimal finite-difference-like temporal cue while keeping all model input sizes small. It is close to the existing SWE one-step contract and easy to explain.
- Alternatives considered:
  - `K=1` primary - simplest, but it removes almost all temporal-context information and may make the result less competitive.
  - `K=10` primary - closer to official PDEBench examples, but too high for this compute-constrained local comparison and harder to justify before a simpler baseline exists.
  - Full autoregressive training with pushforward - potentially stronger, but too much first-tranche complexity.
- Consequences: Model builders must accept `in_channels=4`, `out_channels=2` for the primary profile.
- Evidence required before implementation: Synthetic HDF5 grouping tests that prove the input channels are ordered and shaped as specified.
- Follow-up required if this decision changes: Update this design, the suite plan, run-budget validation, model-profile metadata, and result-summary labels before training.

### ADR-002: `history_len=1` Is The First Ablation

- Status: proposed
- Decision: Add a `K=1` ablation when feasible: `x = u(t-1) -> y = u(t)`, with input `(2,128,128)` and target `(2,128,128)`.
- Context: The paper needs focused ablations that isolate architectural and data-contract choices without broad sweeps.
- Rationale: `K=1` tests whether temporal context beyond the current state matters. It is cheap, interpretable, and keeps the same model/data stack.
- Alternatives considered:
  - `K=4` ablation - useful later, but less crisp and more expensive.
  - `K=10` ablation - useful only for official-protocol comparison, not the first local evidence pass.
- Consequences: Result summaries must label `K=1` and `K=2` rows distinctly.
- Evidence required before implementation: Run config validation that prevents mixing history lengths in the same comparison row.
- Follow-up required if this decision changes: Update benchmark tables and summary caveats so history length is not hidden.

### ADR-003: Target Remains Both Physical Channels

- Status: proposed
- Decision: Predict the full two-channel future state, not a single species, for the primary benchmark.
- Context: The user asked whether multi-channel input and single-channel output might be strong if complexity stays low. For this dataset, the two channels are the two physical state variables of the reaction-diffusion system.
- Rationale: A single-species target is a different and easier problem. The local benchmark should stay aligned with the PDEBench two-field forward-propagator framing.
- Alternatives considered:
  - Species-only target - rejected for primary evidence; allowed only as a later diagnostic ablation if clearly labeled.
  - Predicting a transformed scalar such as magnitude or difference field - rejected because it breaks comparability with FNO/U-Net and PDEBench metrics.
- Consequences: Primary model profiles use `out_channels=2`; metrics report aggregate and per-channel values.
- Evidence required before implementation: Tests that verify target shape `(2,H,W)` and channel metadata.
- Follow-up required if this decision changes: Rename the task row and prohibit comparison to full-state baselines.

### ADR-004: Trajectory-Level Split Before Window Expansion

- Status: proposed
- Decision: Split trajectories first, then expand each split into eligible history windows.
- Context: Window-level random splits would leak future or neighboring states from the same trajectory across train/validation/test.
- Rationale: Trajectory-level splits make the benchmark cleaner and match the suite's dynamic-task policy.
- Alternatives considered:
  - Window-level split - rejected because it risks trajectory leakage.
  - Parameter-level split - scientifically interesting for OOD tests, but not the first local same-protocol row.
- Consequences: If the staged file has `1000` trajectories, the default split should be `800/100/100` trajectories using seed `20260420`. With `T=101` and `K=2`, the full training set has `800 * 99 = 79,200` one-step windows.
- Evidence required before implementation: Split manifest tests for disjoint trajectory IDs and expected window counts.
- Follow-up required if this decision changes: All summaries must explain the new split unit and why it remains comparable.

### ADR-005: Train-Only Per-Physical-Channel Normalization

- Status: proposed
- Decision: Fit two train-only normalization records, one per physical state channel, and apply those same records to every history slot and target channel.
- Context: The input channels are repeated time slices of the same physical variables, not distinct physical quantities.
- Rationale: Separate stats per time slot would leak representation choices into the normalization contract and make `K=1`/`K=2` rows harder to compare.
- Alternatives considered:
  - One scalar global mean/std across both channels - rejected because it can hide channel-scale differences.
  - Separate stats for each flattened input channel - rejected because it treats time slots as different physical variables.
- Consequences: Normalization artifacts must record `field_names`, `history_len`, train trajectory IDs, and sample/time counts used for stats.
- Evidence required before implementation: Tests proving stats are fitted only from train trajectories and reused across history slots.
- Follow-up required if this decision changes: Rerun all affected rows; do not compare old and new rows without caveats.

## Proposed Design

### Implementation Shape

Existing code or workflow to reuse:

- HDF5 recursive inspection and grouped trajectory detection concepts from `scripts/studies/pdebench_swe/manifest.py`.
- Dynamic one-step dataset and split concepts from `scripts/studies/pdebench_swe/data.py` and `scripts/studies/pdebench_swe/splits.py`.
- Shared supervised model adapter concepts from `scripts/studies/pdebench_swe/models.py` and `scripts/studies/pdebench_image128/models.py`.
- Baseline recipe from `docs/model_baselines.md`.

New files or artifacts likely needed in the later implementation plan:

- `scripts/studies/pdebench_image128/reacdiff.py` or a generalized dynamic-task module.
- Tests under `tests/studies/` for grouped HDF5 loading, history-window construction, splits, normalization, metrics, run-config validation, and smoke/readiness artifacts.
- Durable summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_reacdiff_summary.md`.
- Run artifacts under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-reacdiff-temporal-context/`.

Files or APIs likely touched:

- `scripts/studies/pdebench_image128/task_specs.py`
- `scripts/studies/pdebench_image128/preflight.py`
- `scripts/studies/pdebench_image128/data.py`
- `scripts/studies/pdebench_image128/splits.py`
- `scripts/studies/pdebench_image128/normalization.py`
- `scripts/studies/pdebench_image128/metrics.py`
- `scripts/studies/pdebench_image128/models.py`
- `scripts/studies/pdebench_image128/run_config.py`
- `scripts/studies/run_pdebench_image128_suite.py`

Files or APIs that must not be touched:

- Stable CDI physics/model code unless a later plan explicitly authorizes it.
- `/home/ollie/Documents/neurips/` during this design/tranche.
- Dataset files inside git.

Design-plan-implement boundary:

- Decisions this design makes: history length, target channel contract, split unit, normalization unit, claim boundaries.
- Details deferred to implementation plan: exact CLI flags, artifact filenames, dataloader worker settings, batch size, epoch budget, and whether diffusion-reaction shares a generic dynamic-task runner with SWE immediately or starts as a thin task-specific module.

### Core Contracts and Invariants

- Contract: Primary samples are `K=2` history windows predicting the next full two-channel state.
  - Invariant: `input.shape == (4,128,128)` and `target.shape == (2,128,128)` for the primary profile.
  - Failure mode if violated: The comparison no longer tests the agreed temporal-context contract.
  - How to prove it: Dataset unit tests and model-profile metadata.

- Contract: `K=1` is an ablation, not a replacement primary unless a later plan changes the design.
  - Invariant: Results tables carry `history_len`.
  - Failure mode if violated: Rows with different information content are compared as if same-protocol.
  - How to prove it: Run-budget validation and comparison-summary schema tests.

- Contract: Full benchmark rows use all training trajectories and all eligible windows for the selected `K`.
  - Invariant: No capped trajectory, capped pair, or smoke setting is labeled as benchmark performance.
  - Failure mode if violated: The paper claim becomes misleading.
  - How to prove it: Split manifests, dataset counts, and evidence-scope fields.

- Contract: Target predictions are denormalized before metrics.
  - Invariant: Metrics operate in target physical-field space.
  - Failure mode if violated: nRMSE/RMSE values are not comparable.
  - How to prove it: Metric tests with nontrivial train-only stats.

### Data Flow / Control Flow

```text
2D_diff-react_NA_NA.h5
  -> schema preflight confirms grouped trajectories and `*/data` shape
  -> trajectory split manifest using seed 20260420
  -> train-only per-field normalization stats
  -> history-window dataset expands each split
  -> model profiles consume channel-first tensors
  -> denormalized one-step metrics and optional rollout diagnostics
  -> readiness/pilot/full-run summaries with evidence-scope labels
```

History-window construction:

```text
raw trajectory data:      data[t, x, y, channel] where channel = A,B
state u[t]:              (2, 128, 128)

primary K=2 sample:
  input  = concat(u[t-2], u[t-1]) along channel axis -> (4, 128, 128)
  target = u[t]                                      -> (2, 128, 128)
```

Eligible target times:

```text
K=1: t = 1..T-1, windows per trajectory = T-1
K=2: t = 2..T-1, windows per trajectory = T-2
```

If the staged file verifies `T=101` and `1000` trajectories:

```text
K=1 full train windows: 800 * 100 = 80,000
K=2 full train windows: 800 * 99  = 79,200
```

### Model and Baseline Contract

Primary profiles for diffusion-reaction must receive the same `history_len`, channel order, splits, normalization, and metrics:

| Row | `history_len` | Input channels | Output channels | Purpose |
| --- | ---: | ---: | ---: | --- |
| Primary | 2 | 4 | 2 | Main local same-protocol comparison |
| Ablation | 1 | 2 | 2 | Tests temporal-context value |
| Optional later | 4 or 10 | 8 or 20 | 2 | Only if primary results are stable and budget permits |

The Hybrid ResNet profile inherits the current project recipe with the PDE-specific scheduler floor: `fno_modes=12`, hidden width `32`, `fno_blocks=4`, Hybrid ResNet downsample/local depth `2/6`, MAE loss, Adam `lr=2e-4`, and `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5, threshold=0.0)`. Across PDE studies, the scheduler floor must be no higher than `1e-5` for benchmark-performance rows.

FNO and U-Net baselines must use the same input/output tensor contract. A tiny U-Net may be used for readiness only; benchmark interpretation requires a non-toy U-Net profile with recorded parameter count.

## Data, Dependency, and Provenance Decisions

### Data and Artifact Identity

Required input:

- `/home/ollie/Documents/pdebench-data/2d_reacdiff/2D_diff-react_NA_NA.h5`, or the same file under an approved external data root.

Current status:

- Missing locally as of the current preflight.
- Listed size in the suite plan: `13 GB`.
- Current root free space in the preflight: about `22.98 GB`; staging may be tight once run artifacts are included.

Required manifests after implementation:

- Data source, DOI/DaRUS ID or URL, license/access note.
- File identity: path, size, mtime, and checksum where practical.
- HDF5 schema: group count, selected dataset path pattern, shape, dtype, axis order, channel count, time count.
- Split manifest: trajectory IDs and expanded window counts for each split.
- Normalization stats: train-only per physical channel, with counts.
- Run budget: `history_len`, model profiles, seed, optimizer/scheduler, epoch budget, batch size, device, evidence scope.

Freshness/cache policy:

- Smoke/readiness output roots must record invocation and run ID.
- Full benchmark rows must not reuse stale metrics from a different `history_len`, split, seed, or normalization artifact.

### Dependency Discovery

No new runtime dependency is required for the local adapter beyond PyTorch, h5py, NumPy, and optional `neuralop` for FNO. The implementation may use official PDEBench code only as reference context; it should not require importing PDEBench training code unless a later plan explicitly chooses an official-protocol reproduction.

### Provenance and Reproducibility

- Seed: `20260420` for trajectory splits and training unless a later plan records an override.
- Capture command, cwd, git commit/dirty summary, Python executable/version, package versions, CUDA/GPU info where available, data file identity, split/normalization artifact paths, and model-profile parameters.
- Store bulky run outputs under `.artifacts/NEURIPS-HYBRID-RESNET-2026/`.

## Claim, Behavior, Or API Boundaries

Allowed claim after successful implementation and full-training runs:

- Same-protocol local one-step comparison of Hybrid ResNet, FNO, and U-Net on PDEBench 2D diffusion-reaction using the stated `history_len`, splits, normalization, and metric contract.
- Focused evidence about whether minimal temporal context (`K=2` vs `K=1`) changes performance.

Disallowed claim:

- Any model ranking from smoke/readiness metrics.
- Any benchmark-performance claim from capped trajectories or capped windows.
- Same-protocol reproduction of PDEBench's official `initial_step=10` rows unless `K=10`, official-like training, and matching protocol are explicitly run and documented.
- A claim about single-species prediction unless a later design defines that diagnostic task separately.
- Long-horizon rollout quality unless rollout evaluation is implemented and labeled separately from one-step metrics.

Required caveat:

- Published PDEBench baseline values and official configuration settings are protocol context, not automatically same-protocol local comparators.

## Pivot Criteria and Stop Conditions

Pivot to smaller scope if:

- The official file cannot be staged within local/external storage constraints.
- The staged schema differs from the expected grouped trajectory layout and cannot be adapted without overreach.
- `K=2` primary profile exceeds memory/runtime budget even after batch-size adjustment.

Stop before reviewer-facing claims if:

- Any primary model uses a different split, normalization, history length, or output-channel contract.
- FNO or strong U-Net is missing and no blocker is recorded.
- Only smoke or capped-pilot runs exist.

Treat as exploratory only if:

- Training uses fewer than all train trajectories or fewer than all eligible train windows.
- Only `K=1` or species-only diagnostics are run.
- Rollout metrics are produced from a model trained only for one-step prediction and without a rollout-specific claim.

Escalate for human decision if:

- External storage is required.
- Full-training runs are estimated to exceed the remaining RTX 3090 budget.
- Strong local baselines fail to build or are clearly underpowered.

## Required Final Assets

If implemented:

- Data/schema preflight update showing `2d_reacdiff` ready or blocked.
- Implementation plan for the diffusion-reaction tranche.
- Tests for grouped trajectory loading, `K=1/2` window construction, splits, normalization, metrics, run-budget validation, and readiness artifacts.
- Readiness run artifacts labeled `smoke_feasibility_only`.
- Full-run budget JSON before any long training launch.
- Summary under `docs/plans/NEURIPS-HYBRID-RESNET-2026/` separating readiness, pilot, and full benchmark evidence.

If not implemented:

- Durable blocker entry explaining data, storage, schema, dependency, or runtime reason.

## Open Questions

- What is the exact HDF5 schema and group naming after the official file is staged?
- Does the file include `T=101` states or another time count after local download?
- Should a later official-protocol comparison add `K=10`, or is the local `K=2` contract sufficient for the NeurIPS evidence package?
- Should rollout evaluation be added after one-step rows are stable, and if so what horizon should fit the compute budget?

## Verification Targets For The Later Plan

Focused checks should include:

```bash
python -m pytest --collect-only tests/studies/test_pdebench_2d_reacdiff_data.py tests/studies/test_pdebench_2d_reacdiff_runner.py -q
python -m pytest tests/studies/test_pdebench_2d_reacdiff_data.py tests/studies/test_pdebench_2d_reacdiff_metrics.py tests/studies/test_pdebench_2d_reacdiff_runner.py -q
```

Readiness runs must use small caps only for plumbing and emit `performance_assessment_complete: false`.
