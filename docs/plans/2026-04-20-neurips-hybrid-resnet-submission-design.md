# NeurIPS Hybrid ResNet Submission Design

## Design Metadata

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Title: NeurIPS 2026 Hybrid ResNet Submission Brief
- Status: approved
- Date: 2026-04-20
- Source brief / issue: user-approved brainstorming in current Codex session
- Related roadmap: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- Manuscript artifact root: `/home/ollie/Documents/neurips/`
- Experiment root: `/home/ollie/Documents/PtychoPINN/`

## Consumed Inputs and Authority

- Docs index: `docs/index.md`
- Agent guidance: `CLAUDE.md`
- Current recommended Hybrid ResNet baseline: `docs/model_baselines.md`
- Studies/runbook map: `docs/studies/index.md`
- PyTorch workflow context: `docs/workflows/pytorch.md`
- Command recipes: `docs/COMMANDS_REFERENCE.md`
- Known findings and policies: `docs/findings.md`
- Architecture precedent note: `docs/litsurvey.md`
- Hybrid ResNet implementation: `ptycho_torch/generators/hybrid_resnet.py`
- Hybrid ResNet components: `ptycho_torch/generators/resnet_components.py`
- Non-ML CDI benchmark design context: `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-design-seed.md`
- Official deadline references:
  - `https://neurips.cc/Conferences/2026/Dates`
  - `https://neurips.cc/Conferences/2026/CallForPapers`

Authority order for this design:

1. User-approved scope decisions in the current session.
2. PtychoPINN repo guidance and current model/study docs.
3. Existing literature and benchmark context.

## Problem and Scope

The project needs a roadmap-ready brief for a NeurIPS 2026 main-conference submission centered on the Hybrid ResNet architecture. The brief must define the paper shape, required empirical pillars, evidence standards, artifact layout, and triage gates tightly enough that a concrete roadmap can be executed without reopening core scope decisions.

In scope:

- A two-pillar submission strategy: CDI reconstruction plus a compact native `128x128` PDEBench image-suite forward-modeling benchmark.
- Regeneration-first CDI evidence planning for the `128x128` grid-lines Hybrid ResNet anchor, because the most relevant prior local runs are no longer available.
- A neutral selection phase for the required PDE/inverse/forward-modeling benchmark, followed by a scoped PDEBench image-suite amendment covering SWE, Darcy Flow, and 2D Compressible Navier-Stokes.
- A scoped `256x256` CDI scaling branch where higher FNO mode counts are strongly considered.
- Future paper-facing artifact layout under `/home/ollie/Documents/neurips/`.

Out of scope:

- Manuscript prose.
- Writing generated table, figure, or benchmark artifacts into `/home/ollie/Documents/neurips/` during this design step.
- Broad architecture sweeps beyond the compute-constrained evidence package.
- Creating worktrees.

Non-goals:

- Proving Hybrid ResNet is best across a broad PDEBench or operator-learning suite beyond the scoped three-task native `128x128` image suite.
- Reproducing expensive PDE SOTA locally when published numbers are adequate and protocol differences are clearly labeled.

## Decision Summary

- Treat NeurIPS 2026 as a tight, triage-driven campaign. Abstracts are due 2026-05-04 AOE and full papers are due 2026-05-06 AOE.
- Regenerate the `128x128` grid-lines Hybrid ResNet CDI anchor from the known Torch/grid-lines path, then verify and package it after reducing PDE risk.
- Make the PDE/forward-modeling contribution required, with Phase 2 now scoped to a compact native `128x128` PDEBench image suite: SWE (`swe`), Darcy Flow (`darcy`), and 2D Compressible Navier-Stokes (`2d_cfd_cns`).
- Treat OpenFWI FlatVel-A as an optional fallback or adjacent inverse-wave extension, not as the immediate next performance path while the PDEBench image suite is viable.
- Permit external benchmark dependencies and datasets.
- Permit published SOTA comparisons when local reproduction is prohibitive, provided local reasonable baselines are still run and protocol caveats are explicit.
- Require PDE/forward-modeling Hybrid ResNet runs that inform competitiveness to inherit the current grid-lines Hybrid ResNet recipe from `docs/model_baselines.md` unless a later plan records a justified override. At minimum this includes `fno_modes=12`, `fno_width`/hidden width `32`, `fno_blocks=4`, MAE training loss, Adam with `lr=2e-4`, and `ReduceLROnPlateau` with factor `0.5`, patience `2`, min LR no higher than `1e-5` (default `1e-5` for PDE studies), and threshold `0.0`.
- Require any meaningful PDE benchmark-performance row to train on the full available training split for the selected official file after validation/test holdout; capped, subsampled, smoke, or pilot runs are decision-support only.
- Use `/home/ollie/Documents/neurips/index.md` as the eventual top-level evidence map for all submission-relevant artifacts, not just tables. This path is a planned Phase 5 output and is not expected to exist during earlier design, planning, preflight, or benchmark-execution tranches.

## Decision Records

### ADR-001: Two Required Empirical Pillars

- Status: accepted
- Decision: The submission must have a CDI reconstruction pillar and one required PDE/forward-modeling pillar.
- Context: CDI evidence is strongest locally, but the user requires a second contribution exploring PDE/forward-modeling use cases.
- Rationale: NeurIPS positioning improves if the architecture is framed as a spectral/spatial physics-modeling architecture rather than a CDI-only engineering result.
- Alternatives considered:
  - CDI-only submission - rejected because the user explicitly requires the PDE contribution.
  - Broad multi-benchmark operator-learning survey - rejected as infeasible with one RTX 3090 for several days.
- Consequences: The roadmap must prioritize PDE benchmark selection and execution before CDI polish.
- Evidence required before implementation: Candidate benchmark screen, selected benchmark rationale, and fallback benchmark.
- Follow-up required if this decision changes: Revise roadmap priority order and manuscript claim boundaries.

### ADR-002: Native `128x128` PDEBench Image Suite

- Status: amended on 2026-04-20
- Decision: Phase 2 will target a compact native `128x128` PDEBench image suite with SWE (`swe`), Darcy Flow (`darcy`), and 2D Compressible Navier-Stokes (`2d_cfd_cns`), rather than continuing the earlier single-SWE-primary then OpenFWI-fallback path.
- Context: The earlier "one deep benchmark plus fallback" decision avoided shallow breadth under the one-RTX-3090 budget. After the user asked to replace 2D diffusion-reaction with the harder PDEBench 2D CNS benchmark, the better scoped unit remains one reusable PDEBench image-suite contribution, with CNS as the discriminating time-dependent member.
- Rationale: These tasks share the native image-grid resolution, PDEBench source family, metrics/reporting style, and reusable adapter surface. The suite can test the spectral/local hypothesis across wave/flow-like dynamics, elliptic operator mapping, and four-field compressible fluid dynamics without opening a broad PDEBench survey. For CNS, high-frequency Fourier-space RMSE (`fRMSE_high`) is required as the shock/small-scale-structure diagnostic alongside denormalized nRMSE/RMSE.
- Alternatives considered:
  - Continue with SWE only and OpenFWI next - superseded because OpenFWI is `70x70` and does not answer the user's native `128x128` PDEBench image question.
  - Treat all of PDEBench as in scope - rejected as infeasible with one RTX 3090 for several days.
  - Run three completely separate adapters - rejected unless the shared image-suite adapter proves impractical.
- Consequences: The roadmap and selector should make the next Phase 2 plan a shared PDEBench `128x128` image-suite preflight and adapter plan. OpenFWI is deferred unless the suite is infeasible or explicitly requested as an adjacent inverse-wave result.
- Evidence required before implementation: Exact file/schema preflight for Darcy and the selected official `128x128` 2D CNS file, reuse plan for the existing SWE adapter, shared split/normalization/metric contracts including CNS `fRMSE_low/mid/high`, and a compute budget that distinguishes smoke readiness from benchmark-performance runs.
- Follow-up required if this decision changes: Record why the suite failed and whether OpenFWI, SWE-only, or a CDI-only fallback is now the least weak paper option.

### ADR-003: Neutral Benchmark Selection

- Status: accepted
- Decision: Select the PDE benchmark through a neutral screen over PDEBench/PDEArena-style fluids/waves and inverse-scattering or wave-propagation candidates.
- Context: The architecture may suit fluids with local instabilities plus large-scale structure, but inverse/wave/scattering tasks may better align with CDI.
- Rationale: A neutral screen avoids prematurely committing to a benchmark that is thematically appealing but not feasible or strong.
- Alternatives considered:
  - Default to PDEBench fluids - rejected as too assumption-heavy.
  - Default to inverse scattering - rejected until benchmark maturity and baselines are verified.
- Consequences: The first roadmap phase must be evidence discovery and benchmark selection.
- Evidence required before implementation: Scorecard covering fit, maturity, data size, baselines, SOTA availability, and RTX 3090 feasibility.
- Follow-up required if this decision changes: Record rejected candidates and why the selection rule changed.

### ADR-004: Regenerate the `128x128` CDI Anchor

- Status: accepted
- Decision: Regenerate the paper-grade `128x128` grid-lines Hybrid ResNet anchor, using the known Torch/grid-lines integration path with longer training rather than relying on prior local run artifacts. Use `docs/studies/index.md` and `tests/torch/test_grid_lines_hybrid_resnet_integration.py` as runtime/runbook guidance when drafting the regeneration command and budget.
- Context: Compute is limited to one RTX 3090 for several days, and the most relevant prior `128x128` Hybrid ResNet runs have been lost.
- Rationale: The submission needs an auditable core CDI result. Since the relevant prior anchor artifacts are unavailable, paper-grade evidence must come from a fresh, provenance-captured rerun rather than from partial historical remnants.
- Alternatives considered:
  - Use surviving historical metrics directly - rejected because the relevant anchor runs are not available and would not be auditable enough.
  - Full broad CDI rerun campaign - rejected as too costly.
- Consequences: The roadmap must schedule fresh `128x128` anchor regeneration, with compact baselines and ablations scoped around that fresh run rather than around lost prior runs.
- Evidence required before implementation: A regeneration runbook or command, dataset/split identity, seed/config capture, metric contract, qualitative output plan, and runtime budget grounded in the study index and Hybrid ResNet integration-test path where possible.
- Follow-up required if this decision changes: Record which recovered run artifacts are sufficient to replace the fresh anchor and why.

### ADR-005: `256x256` Higher FNO Modes As Targeted Scaling Hypothesis

- Status: accepted
- Decision: The `256x256` branch should strongly consider higher FNO mode counts as a targeted scaling axis.
- Context: Higher resolution can preserve higher spatial frequencies than the `128x128` anchor.
- Rationale: If the spectral branch is central to the claim, the scaling branch should test whether increased spectral support improves fine structure at `256x256`.
- Alternatives considered:
  - Reuse `N=128` modes unchanged - accepted only as the inherited baseline comparator.
  - Broad mode/width/depth sweep - rejected as infeasible.
- Consequences: Run at most one or two higher-mode variants after CDI/PDE core viability is secured.
- Evidence required before implementation: Memory/runtime preflight and primary metric comparison.
- Follow-up required if this decision changes: Document why high modes were infeasible or scientifically unnecessary.

### ADR-006: Top-Level Submission Evidence Index

- Status: accepted
- Decision: Future paper-facing artifacts under `/home/ollie/Documents/neurips/` should be organized by a top-level `index.md`.
- Context: The user clarified that the index should reference basically everything important, not just tables.
- Rationale: A single evidence map reduces reviewer-response and final-paper assembly risk.
- Alternatives considered:
  - Table-specific index only - rejected as too narrow.
  - No central index - rejected as too hard to audit under deadline pressure.
- Consequences: When artifacts are eventually created, `index.md` should link CDI/PDE tables, figure manifests, provenance, metric contracts, baselines, ablations, datasets, published-SOTA notes, and failed/pivoted experiments.
- Evidence required before implementation: Artifact layout proposal in the roadmap.
- Follow-up required if this decision changes: Update artifact writers and docs references.

## Proposed Design

### Implementation Shape

- Existing code or workflow to reuse for regeneration and comparison:
  - `scripts/studies/grid_lines_compare_wrapper.py`
  - `scripts/studies/grid_lines_torch_runner.py`
  - `scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py`
  - `scripts/studies/runbooks/run_nersc_scan807_cameraman_study*.py`
  - `scripts/reconstruction/hio_cdi_benchmark.py`
  - `ptycho_torch/generators/hybrid_resnet.py`
- New files or artifacts likely needed later:
  - `/home/ollie/Documents/neurips/index.md` (created during the roadmap evidence-bundle phase)
  - `/home/ollie/Documents/neurips/benchmarks/...`
  - `/home/ollie/Documents/neurips/tables/...`
  - `/home/ollie/Documents/neurips/figure_manifests/...`
  - `/home/ollie/Documents/neurips/provenance/...`
- Files or APIs likely touched during roadmap execution:
  - New study scripts or runbooks under `scripts/studies/`
  - New tests under `tests/studies/` or `tests/torch/`
  - Documentation under `docs/plans/` and possibly `docs/studies/`
- Files or APIs that must not be touched without a narrower approved plan:
  - `ptycho/model.py`
  - `ptycho/diffsim.py`
  - `ptycho/tf_helper.py`
- One-off versus reusable boundary:
  - Submission artifact assembly can be one-off.
  - Benchmark adapters and runbooks should be reusable if they introduce external dependencies or data contracts.
- Design-plan-implement boundary:
  - This design fixes scope, priorities, claims, and artifact surfaces.
  - The roadmap fixes phases, gates, and expected artifacts.
  - Implementation plans should be written for any code-changing benchmark adapter, artifact writer, or workflow.

### Core Contracts and Invariants

- Contract: PDE benchmark contribution is required.
  - Invariant: The roadmap carries the PDEBench `128x128` image-suite path and keeps OpenFWI as a documented optional fallback or extension.
  - Failure mode if violated: The submission becomes CDI-only against user intent.
  - Proof: Benchmark selection amendment, image-suite plan, and execution gate artifacts.
- Contract: PDE/forward-modeling competitiveness runs inherit the Hybrid ResNet baseline recipe.
  - Invariant: A PDE/OpenFWI result cannot be promoted as Hybrid ResNet competitiveness evidence if it used a one-epoch feasibility budget, reduced spectral modes, a different learning rate, a scheduler minimum LR higher than `1e-5`, or no plateau schedule without an explicit documented override and rationale.
  - Failure mode if violated: The paper rejects a potentially viable architecture because the benchmark was run under a smoke or underconfigured recipe.
  - Proof: Run budget JSON, profile config, invocation artifacts, and summary text that lists the inherited or intentionally overridden recipe.
- Contract: Non-ptychography benchmarks use a supervised real-channel Hybrid ResNet adapter, not the CDI `PtychoPINN_Lightning` physics wrapper.
  - Current capability: the supervised SWE and OpenFWI adapters keep the full Hybrid ResNet encoder-bottleneck-decoder body: `SpatialLifter`, Hybrid ResNet encoder blocks, downsampling, ResNet bottleneck, CycleGAN-style upsampling, and a final real-channel projection. They are task-specific wrappers around the full body, not partial feature extractors. `scripts/studies/pdebench_swe/models.py` builds `HybridResnetSweModel(in_channels, out_channels, ...)`, and `scripts/studies/openfwi_flatvel_a/models.py` builds `HybridResnetSmoke(in_channels, out_channels, ...)`.
  - CDI wrapper boundary: the registered `architecture="hybrid_resnet"` path is not single-channel in the naive sense; it uses `data_config.C` as the ptychographic channel/group count, so the lifter sees `C` input channels when `in_channels=1`. The boundary is semantic and output-side: those `C` channels are treated as ptychographic object-patch channels, and the generator emits `C` complex object channels through the real/imag output adapter for the CDI forward model.
  - Modification needed for multichannel non-ptychography data: keep the Hybrid ResNet encoder-bottleneck-decoder body intact and remove only the CDI-specific boundary contract: the `PtychoPINN_Lightning` physics wrapper, probe/scan-position inputs, ptychographic `C` grouping, and real/imag-to-complex output conversion. The supervised adapter sets the first `SpatialLifter` to the dataset input channel count and the final `Conv2d` to the dataset target channel count. Add pad/crop around the model when downsampling requires spatial divisibility.
  - Failure mode if violated: a PDE/OpenFWI run may appear to execute with "Hybrid ResNet" naming while silently using a complex ptychographic output contract that is wrong for real multichannel fields.
  - Proof: adapter code, model profile config, input/output tensor shape assertions, and summary text that labels the path as a supervised real-channel Hybrid ResNet adapter.
- Contract: Smoke gates are feasibility gates, not benchmark-performance gates.
  - Invariant: Smoke metrics may confirm data loading, model execution, metric plumbing, and provenance, but cannot rank models, trigger a performance pivot, support a paper-facing claim, or satisfy the PDE competitiveness gate.
  - Failure mode if violated: A tiny, unstable subset is mistaken for benchmark evidence and either overclaims a weak result or rejects a viable model.
  - Proof: Smoke summary fields such as `evidence_scope: smoke_feasibility_only`, `metric_interpretation: sanity_only_not_benchmark_performance`, and `performance_assessment_complete: false`, plus a separate longer-run plan for any benchmark-performance decision.
- Contract: Meaningful PDE benchmark rows use the full available training split.
  - Invariant: After the selected official file and validation/test split are fixed, benchmark-performance runs train on every remaining available training sample. If 10,000 total samples exist and 2,000 are held out for validation/test, the meaningful benchmark training set is 8,000 samples.
  - Failure mode if violated: A capped or convenience subset is presented as a benchmark and underestimates or misranks models.
  - Proof: Split manifest, run budget, dataloader counts, and summary text that distinguish full-training benchmark rows from capped smoke or pilot rows.
- Contract: CDI headline is `128x128`.
  - Invariant: `256x256` is secondary scaling/comparison evidence unless later explicitly promoted.
  - Failure mode if violated: Scope expands beyond compute budget.
  - Proof: Roadmap phase order and table plan labels.
- Contract: Published SOTA comparisons are allowed only when labeled.
  - Invariant: Local baselines and published SOTA are not presented as same-protocol reproduction.
  - Failure mode if violated: Reviewer-facing comparison becomes misleading.
  - Proof: Benchmark summary now; `/home/ollie/Documents/neurips/index.md` notes after Phase 5 creates that manuscript index.
- Contract: Paper-facing artifacts eventually live under `/home/ollie/Documents/neurips/`.
  - Invariant: PtychoPINN remains the experiment/code root; `/home/ollie/Documents/neurips/` is the planned evidence/manuscript root and need not exist before the evidence-bundle phase.
  - Failure mode if violated: Evidence becomes difficult to audit.
  - Proof: Artifact index and provenance links.

### Data Flow / Control Flow

```text
approved design
  -> roadmap phases and gates
  -> evidence inventory
  -> PDE benchmark scorecard
  -> PDEBench 128x128 image-suite amendment
  -> shared PDEBench image-suite plan
  -> PDE suite execution artifacts
  -> CDI anchor verification and compact ablations
  -> optional 256x256 scaling branch
  -> create /home/ollie/Documents/neurips/index.md and the paper-facing evidence bundle
```

## Data, Dependency, and Provenance Decisions

### Data and Artifact Identity

- Required inputs:
  - Surviving CDI run directories, metrics, invocation logs, and runbooks.
  - Fresh `128x128` Hybrid ResNet regeneration outputs after Phase 3.
  - Existing N128/N256 runbooks and dataset paths.
  - External PDE benchmark candidate metadata.
- Required outputs later:
  - Benchmark scorecard.
  - PDEBench `128x128` image-suite manifest.
  - CDI evidence inventory and fresh `128x128` anchor regeneration summary.
  - Paper-facing evidence index.
- Checksum / manifest fields:
  - Dataset path, size, checksum where practical, split identity, command, git commit, model config, seed, runtime environment, metric contract.
- Freshness or cache policy:
  - Historical artifacts require provenance verification before paper use.
  - Fresh reruns require invocation and config capture.
- Reuse policy:
  - Reuse is acceptable for CDI support evidence and baseline context when provenance is sufficient.
  - The `128x128` Hybrid ResNet anchor itself requires a fresh regenerated run unless a later recovery note proves an existing artifact is complete and auditable.
  - PDE benchmark data can be newly downloaded or installed, subject to license and feasibility checks.

### Dependency Discovery

- Discovery scope:
  - Search current PtychoPINN repo and environment for reusable code and dependencies.
  - Search official or primary benchmark sources for PDEBench/PDEArena/inverse-scattering candidates.
  - Verify installation and dataset size before full runs.
- Candidate acceptance criteria:
  - Public benchmark with clear metrics.
  - Data and models feasible on one RTX 3090.
  - At least two reasonable local baselines or one strong local baseline plus published SOTA.
  - Strong rationale for spectral/spatial hybridization.
- Candidate rejection criteria:
  - Excessive data size or training time.
  - Unclear metric protocol.
  - Weak thematic connection.
  - Unavailable or irreproducible data.
- Installation policy:
  - Add external dependencies only behind a documented benchmark adapter or study runbook.
- Fallback:
  - If the native `128x128` PDEBench image suite fails feasibility or competitiveness gates, pivot only after a plan-level decision to SWE-only, OpenFWI FlatVel-A, diffusion-reaction as a deferred replacement, or a narrowed PDE claim.

### Provenance and Reproducibility

- Required command capture:
  - `invocation.json` and/or `invocation.sh` for local runs where existing infrastructure supports it.
- Required environment capture:
  - Python, PyTorch, CUDA, dependency versions, git commit, and GPU model.
- Required random seeds:
  - Fixed seeds for local baseline and Hybrid ResNet runs.
  - Multi-seed only where compute permits or where stochastic instability threatens the claim.
- Required evidence logs:
  - Metrics JSON/CSV, stdout/stderr logs for long runs, run summaries, and explicit failed/pivoted attempt notes.

## Claim Boundaries

Allowed claims after successful roadmap execution:

- Hybrid ResNet improves `128x128` CDI reconstruction over the selected local baselines under the recorded metric contract.
- Hybrid ResNet transfers to the scoped native `128x128` PDEBench image-suite tasks with locally run baseline comparisons, if the suite execution completes under the recorded metric contracts and full available training-split rule.
- `256x256` results support scaling only if the branch produces clean evidence.
- Higher FNO modes are a tested scaling hypothesis at `256x256` if the branch runs.

Disallowed claims:

- Broad SOTA across PDEBench or operator learning outside the scoped image suite.
- Same-protocol comparison against published SOTA when local reproduction was not performed.
- General `256x256` superiority if only limited or exploratory results exist.
- CDI results without provenance as paper-grade evidence.

Required caveats:

- Compute-constrained scope.
- Published SOTA protocol differences.
- Any capped or subsampled PDE runs are pilot/triage evidence, not meaningful benchmark rows.
- Failed or excluded benchmark candidates and failed suite tasks that affect interpretation.

## Pivot Criteria and Stop Conditions

- Pivot away from the PDEBench image suite if Darcy or the selected 2D CNS file cannot be staged, the shared adapter cannot support the required schemas without overreach, the suite cannot fit the GPU/storage budget, or Hybrid ResNet underperforms simple local baselines after full-training benchmark runs that are explicitly outside the smoke-gate and capped-pilot scope.
- Stop before reviewer-facing PDE claims if only published SOTA exists and no local baseline can be run.
- Treat `256x256` results as exploratory if higher modes are infeasible or only one unstable run exists.
- Escalate for human decision before promoting OpenFWI, SWE-only, or CDI-only as a replacement story.

## Required Final Assets

If the roadmap succeeds:

- Design and roadmap docs under `docs/plans/`.
- CDI evidence inventory.
- PDE benchmark selection scorecard.
- PDEBench `128x128` image-suite plan and execution summary.
- CDI anchor table/figure source artifacts under `/home/ollie/Documents/neurips/` after the evidence-bundle phase creates that root.
- Top-level `/home/ollie/Documents/neurips/index.md` created by the evidence-bundle phase.
- Provenance notes linking back to PtychoPINN run directories.

If the roadmap pivots or fails:

- Rejected-candidate summary.
- Failed-attempt summary.
- Narrowed claim notes for the evidence index.

## Verification Plan

- Markdown/link inspection for design and roadmap docs.
- Deterministic checks for any generated manifests or JSON files in later phases.
- Unit/integration tests for any new benchmark adapters or artifact writers.
- Smoke runs before full runs for PDE benchmark adapters.
- Artifact existence and freshness checks before any completion claim.

## Open Questions

| ID | Question | Resolution needed by | Status |
|---|---|---|---|
| Q1 | Which exact existing `128x128` run is the CDI anchor? | Evidence inventory | open |
| Q2 | Which PDE benchmark wins the neutral selection phase? | Roadmap Phase 1 | resolved: PDEBench SWE selected, then amended to the PDEBench `128x128` image suite |
| Q3 | Which fallback PDE benchmark is viable? | Roadmap Phase 1/2 | resolved for readiness only: OpenFWI FlatVel-A smoke access is available, but it is deferred while the image suite is viable |
| Q4 | Which exact higher FNO mode values are feasible at `256x256`? | Scaling branch preflight | open |
| Q5 | Which exact Darcy and 2D CNS HDF5 schemas, split files, and checksums are available locally or through an external data root? | PDEBench image-suite preflight | open |

## Planning Handoff Checklist

- [x] Design status is approved by user in current session.
- [x] `docs/index.md` was used to select relevant docs.
- [x] The PDE contribution is required.
- [x] The CDI headline is `128x128`.
- [x] `256x256` higher FNO modes are included as a targeted scaling hypothesis.
- [x] `/home/ollie/Documents/neurips/index.md` is the future evidence map and is not expected to exist until Phase 5 creates it.
- [x] No manuscript prose or `/home/ollie/Documents/neurips/` artifacts are written by this design.
