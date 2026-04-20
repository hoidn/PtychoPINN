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

- A two-pillar submission strategy: CDI reconstruction plus one required PDE/forward-modeling benchmark.
- Regeneration-first CDI evidence planning for the `128x128` grid-lines Hybrid ResNet anchor, because the most relevant prior local runs are no longer available.
- A neutral selection phase for the required PDE/inverse/forward-modeling benchmark.
- A scoped `256x256` CDI scaling branch where higher FNO mode counts are strongly considered.
- Future paper-facing artifact layout under `/home/ollie/Documents/neurips/`.

Out of scope:

- Manuscript prose.
- Writing generated table, figure, or benchmark artifacts into `/home/ollie/Documents/neurips/` during this design step.
- Broad architecture sweeps beyond the compute-constrained evidence package.
- Creating worktrees.

Non-goals:

- Proving Hybrid ResNet is best across a broad PDE benchmark suite.
- Reproducing expensive PDE SOTA locally when published numbers are adequate and protocol differences are clearly labeled.

## Decision Summary

- Treat NeurIPS 2026 as a tight, triage-driven campaign. Abstracts are due 2026-05-04 AOE and full papers are due 2026-05-06 AOE.
- Regenerate the `128x128` grid-lines Hybrid ResNet CDI anchor from the known Torch/grid-lines path, then verify and package it after reducing PDE risk.
- Make the PDE/forward-modeling contribution required, but select exactly one deep benchmark through a neutral screen.
- Permit external benchmark dependencies and datasets.
- Permit published SOTA comparisons when local reproduction is prohibitive, provided local reasonable baselines are still run and protocol caveats are explicit.
- Use `/home/ollie/Documents/neurips/index.md` as the eventual top-level evidence map for all submission-relevant artifacts, not just tables.

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

### ADR-002: One Deep PDE Benchmark

- Status: accepted
- Decision: The roadmap will target one deep PDE/forward-modeling benchmark plus one fallback, not two shallow benchmarks.
- Context: The user chose "one deep" after confirming the PDE contribution is required.
- Rationale: One careful benchmark with strong baselines, clear metrics, and focused ablations is more credible than multiple shallow comparisons under the deadline and compute budget.
- Alternatives considered:
  - Two smaller representative benchmarks - rejected as likely too shallow.
  - Pure literature-only PDE argument - rejected because the PDE contribution is required empirically.
- Consequences: Selection criteria and pivot gates matter more than benchmark breadth.
- Evidence required before implementation: Shortlist scorecard and selected benchmark feasibility proof.
- Follow-up required if this decision changes: Split the PDE phase and revise compute allocation.

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
  - `/home/ollie/Documents/neurips/index.md`
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
  - Invariant: The roadmap carries both primary and fallback benchmark paths.
  - Failure mode if violated: The submission becomes CDI-only against user intent.
  - Proof: Benchmark selection scorecard and execution gate artifacts.
- Contract: CDI headline is `128x128`.
  - Invariant: `256x256` is secondary scaling/comparison evidence unless later explicitly promoted.
  - Failure mode if violated: Scope expands beyond compute budget.
  - Proof: Roadmap phase order and table plan labels.
- Contract: Published SOTA comparisons are allowed only when labeled.
  - Invariant: Local baselines and published SOTA are not presented as same-protocol reproduction.
  - Failure mode if violated: Reviewer-facing comparison becomes misleading.
  - Proof: Benchmark summary and `/home/ollie/Documents/neurips/index.md` notes.
- Contract: Paper-facing artifacts eventually live under `/home/ollie/Documents/neurips/`.
  - Invariant: PtychoPINN remains the experiment/code root; `/home/ollie/Documents/neurips/` is the evidence/manuscript root.
  - Failure mode if violated: Evidence becomes difficult to audit.
  - Proof: Artifact index and provenance links.

### Data Flow / Control Flow

```text
approved design
  -> roadmap phases and gates
  -> evidence inventory
  -> PDE benchmark scorecard
  -> primary/fallback benchmark decision
  -> PDE execution artifacts
  -> CDI anchor verification and compact ablations
  -> optional 256x256 scaling branch
  -> /home/ollie/Documents/neurips/index.md and paper-facing evidence bundle
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
  - Selected PDE benchmark manifest.
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
  - If the primary benchmark fails feasibility or competitiveness gates, pivot quickly to the preselected fallback benchmark.

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
- Hybrid ResNet transfers to one selected PDE/forward-modeling benchmark with locally run baseline comparisons.
- `256x256` results support scaling only if the branch produces clean evidence.
- Higher FNO modes are a tested scaling hypothesis at `256x256` if the branch runs.

Disallowed claims:

- Broad SOTA across PDEBench or operator learning.
- Same-protocol comparison against published SOTA when local reproduction was not performed.
- General `256x256` superiority if only limited or exploratory results exist.
- CDI results without provenance as paper-grade evidence.

Required caveats:

- Compute-constrained scope.
- Published SOTA protocol differences.
- Failed or excluded benchmark candidates that affect interpretation.

## Pivot Criteria and Stop Conditions

- Pivot to fallback PDE benchmark if the primary candidate cannot be installed, cannot fit the GPU budget, lacks usable metrics, or shows Hybrid ResNet underperforming simple local baselines after reasonable configuration checks.
- Stop before reviewer-facing PDE claims if only published SOTA exists and no local baseline can be run.
- Treat `256x256` results as exploratory if higher modes are infeasible or only one unstable run exists.
- Escalate for human decision if both PDE primary and fallback benchmarks fail.

## Required Final Assets

If the roadmap succeeds:

- Design and roadmap docs under `docs/plans/`.
- CDI evidence inventory.
- PDE benchmark selection scorecard.
- PDE benchmark execution summary.
- CDI anchor table/figure source artifacts under `/home/ollie/Documents/neurips/`.
- Top-level `/home/ollie/Documents/neurips/index.md`.
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
| Q2 | Which PDE benchmark wins the neutral selection phase? | Roadmap Phase 1 | open |
| Q3 | Which fallback PDE benchmark is viable? | Roadmap Phase 1 | open |
| Q4 | Which exact higher FNO mode values are feasible at `256x256`? | Scaling branch preflight | open |

## Planning Handoff Checklist

- [x] Design status is approved by user in current session.
- [x] `docs/index.md` was used to select relevant docs.
- [x] The PDE contribution is required.
- [x] The CDI headline is `128x128`.
- [x] `256x256` higher FNO modes are included as a targeted scaling hypothesis.
- [x] `/home/ollie/Documents/neurips/index.md` is the future evidence map.
- [x] No manuscript prose or `/home/ollie/Documents/neurips/` artifacts are written by this design.
