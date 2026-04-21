# PDEBench 2D CFD CNS Physics Regularization Design

## Design Metadata

- ID: `NEURIPS-HYBRID-RESNET-2026-pdebench-2d-cfd-cns-physics-regularization`
- Title: Reusable Physics-Loss Framework With CNS Backend
- Status: approved
- Date: 2026-04-21
- Source brief / issue: add a reusable physics-regularization framework for PDEBench image-suite training, with a first backend for `2d_cfd_cns`
- Related plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-cfd-cns-physics-regularization/execution_plan.md`
- Related checklist / backlog item: N/A

Approval source:

- User-approved in-session on 2026-04-21 after section-by-section design review.

## Consumed Inputs and Authority

- Docs index: `docs/index.md`
- Knowledge base: `docs/findings.md`
- Suite plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- CNS design: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_design.md`
- CNS implementation summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Current CNS runner: `scripts/studies/pdebench_image128/cfd_cns.py`
- Current CNS data contract: `scripts/studies/pdebench_image128/data.py`
- Current normalization helpers: `scripts/studies/pdebench_image128/normalization.py`
- Current CNS HDF5 file metadata: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse/hdf5_metadata.json`
- Official PDEBench compressible-fluid generator family, accessed 2026-04-21:
  - `CFD_Hydra.py`
  - `CFD_multi_Hydra.py`
  - compressible-fluid Hydra config files with `gamma = 5/3`

Authority order for this design:

1. Existing local CNS data/training contract in the repository
2. NeurIPS PDEBench suite and CNS design docs
3. Official PDEBench generator family for PDE semantics and field interpretation

## Problem and Scope

- Problem:
  The current `2d_cfd_cns` training path is purely supervised on normalized next-state targets. It does not enforce basic physical admissibility or mass conservation, even though the dataset is a periodic compressible-flow benchmark with enough metadata to support lightweight physics regularization.
- User / reviewer / system need:
  Add a reusable physics-loss framework that can be wired into PDEBench image-suite training without entangling the loss logic with Hybrid ResNet internals or the legacy ptychographic physics code.
- In scope:
  - A reusable physics-regularizer interface in the PDEBench image-suite study code
  - A first concrete backend for `2d_cfd_cns`
  - Three v1 CNS regularizers: positivity, continuity residual, global mass consistency
  - CLI/config/plumbing so the same framework can be enabled for Hybrid, FNO, or U-Net profiles on CNS
  - Metadata/provenance/logging/test support for the new loss path
- Out of scope:
  - Momentum residuals
  - Energy residuals
  - Darcy or SWE physics-loss implementations
  - Architectural changes to Hybrid, FNO, or U-Net
  - Any use of ptychographic physics modules
- Non-goals:
  - Claim PDE-consistent training in a strong scientific sense
  - Change the default CNS benchmark recipe away from pure supervised loss
  - Introduce silent no-op physics hooks for unsupported tasks

## Decision Summary

- Decision:
  Build a generic physics-loss framework under `scripts/studies/pdebench_image128/`, but ship only a `2d_cfd_cns` backend in v1.
- Rationale:
  This preserves a clean boundary between training-loop wiring and task-specific PDE logic, while avoiding a broad multi-task abstraction before there is a second real consumer.
- Expected implementation shape:
  Add a regularizer builder + result object, periodic derivative helpers, CNS metadata plumbing for `dx/dy/dt/eta/zeta`, CLI flags, logging, and focused tests.
- Claim or behavior limits:
  The framework is a training regularization facility. It does not by itself justify claims of physical correctness or improved scientific fidelity.
- Pivot or abandonment condition:
  If periodic derivative scaling or continuity-term stability cannot be validated in focused tests and a small CNS smoke run, ship only the generic interface plus positivity/global-mass terms and leave continuity behind a blocked note.

## Decision Records

### ADR-001: Use a generic framework with a CNS-only backend

- Status: accepted
- Decision:
  Implement a small reusable physics-regularizer interface, but register only `2d_cfd_cns` in v1.
- Context:
  The user asked for a reusable framework, but only the CNS task currently has a justified v1 physics-loss bundle and the necessary periodic-grid metadata.
- Rationale:
  This keeps the training-loop integration reusable while avoiding empty abstractions or unsupported silent no-ops for Darcy/SWE.
- Alternatives considered:
  - Inline CNS-only logic in `cfd_cns.py` - rejected because it welds PDE logic into one runner and makes later reuse harder.
  - Multi-task physics framework with SWE/Darcy placeholders - rejected because no-op backends are misleading and widen the surface without real need.
- Consequences:
  - New task registry/builder surface is introduced.
  - Unsupported tasks must fail closed when physics regularization is requested.
- Evidence required before implementation:
  - Existing CNS runner/location of loss hook confirmed
  - HDF5 metadata confirms periodic-grid coordinates and separate primitive fields
- Follow-up required if this decision changes:
  - Update the interface contract and provenance fields to distinguish task-local and shared loss settings.

### ADR-002: Limit v1 to positivity, continuity residual, and global mass

- Status: accepted
- Decision:
  The first CNS backend will implement only positivity, continuity residual, and global mass consistency.
- Context:
  Momentum and energy terms are plausible follow-ons, but they require more PDE-specific assumptions and are more brittle under mismatched discretization.
- Rationale:
  The selected bundle gives one hard admissibility term, one local conservation term, and one global conservation term with minimal additional PDE assumptions.
- Alternatives considered:
  - Add momentum residual now - rejected for v1 because it increases implementation and stability risk.
  - Add energy residual now - rejected for v1 because the PDEBench viscous treatment is simplified and the exact training-time residual would be easy to mis-specify.
- Consequences:
  - v1 is intentionally narrow and interpretable.
  - The design remains extensible for later momentum/energy additions.
- Evidence required before implementation:
  - Periodic finite-difference helpers validated on synthetic tensors
  - Small CNS smoke run completes with the new bundle enabled
- Follow-up required if this decision changes:
  - Extend provenance and tests to capture the added loss terms explicitly.

### ADR-003: Compute all physics terms in denormalized physical units

- Status: accepted
- Decision:
  Physics losses will denormalize the latest-history state and predicted next state before computing positivity, continuity, and mass terms.
- Context:
  The current CNS task normalizes fields with train-split per-field z-scores. Physics equations in normalized coordinates would be scale-dependent and harder to interpret.
- Rationale:
  Physical-unit losses are easier to reason about, easier to debug, and less sensitive to changes in normalization policy.
- Alternatives considered:
  - Compute regularization in normalized space - rejected because weights would be tied to normalization statistics and less physically meaningful.
- Consequences:
  - The regularizer depends on `state_stats`.
  - `dx/dy/dt` and field order must be explicit and stable.
- Evidence required before implementation:
  - Current normalization/denormalization helpers verified
  - Field order and metadata verified from current HDF5 contract
- Follow-up required if this decision changes:
  - Re-tune weights and update the scientific caveat in docs/metrics payloads.

## Proposed Design

### Implementation Shape

- Existing code or workflow to reuse:
  - Current CNS training loop in `scripts/studies/pdebench_image128/cfd_cns.py`
  - Existing normalization helpers in `scripts/studies/pdebench_image128/normalization.py`
  - Existing HDF5 inspection and history-window dataset in `scripts/studies/pdebench_image128/data.py`
- New files or artifacts likely needed:
  - `scripts/studies/pdebench_image128/physics_losses.py`
  - `tests/studies/test_pdebench_physics_losses.py`
- Files or APIs likely touched:
  - `scripts/studies/pdebench_image128/cfd_cns.py`
  - `scripts/studies/pdebench_image128/data.py`
  - `tests/studies/test_pdebench_cfd_cns_data.py`
  - `tests/studies/test_pdebench_image128_runner.py`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
  - `docs/index.md`
- Files or APIs that must not be touched:
  - `ptycho/model.py`
  - `ptycho/diffsim.py`
  - `ptycho/tf_helper.py`
  - model definitions in `scripts/studies/pdebench_image128/models.py` unless required only for provenance/logging
- One-off versus reusable boundary:
  - Reusable: loss interface, config surface, periodic derivative helpers
  - One-off/task-local: CNS term definitions and metadata interpretation
- Design-plan-implement boundary:
  - Decisions this design must make:
    - backend shape
    - v1 term bundle
    - physical-unit computation
    - fail-closed unsupported-task behavior
  - Details intentionally deferred to the implementation plan:
    - exact helper function names
    - test selector ordering
    - final CLI spelling if a narrower parser shape proves cleaner

### Core Contracts and Invariants

- Contract:
  Physics regularization is disabled by default and must preserve current CNS behavior when off.
- Invariant:
  Enabling physics regularization for `2d_cfd_cns` must work identically across Hybrid, FNO, and U-Net because it is applied in the shared runner, not the model classes.
- Failure mode if violated:
  Architecture-specific behavior would make comparisons ambiguous and invalidate the “shared training recipe variant” framing.
- How the implementation should prove it:
  Runner tests must show that the same config surface applies to multiple CNS profiles without model-code edits.

- Contract:
  Unsupported tasks must reject requested physics regularization explicitly.
- Invariant:
  `darcy` and `swe` cannot silently accept `--physics-regularization on`.
- Failure mode if violated:
  The framework would appear broader than it is, and runs could record misleading provenance.
- How the implementation should prove it:
  Add a focused failure-path test or explicit guard in the builder.

- Contract:
  Physics terms are computed on denormalized physical fields with fixed field order `[density, Vx, Vy, pressure]`.
- Invariant:
  No physics term may operate on normalized tensors or reordered channels.
- Failure mode if violated:
  Loss weights become normalization-dependent and physically uninterpretable.
- How the implementation should prove it:
  Synthetic tests plus provenance fields documenting field order and physical spacings.

### Data Flow / Control Flow

```text
HDF5 metadata + normalization stats + CLI config
  -> build_physics_regularizer(task_id="2d_cfd_cns", ...)
  -> training batch:
       x_norm, y_norm
       -> model(x_norm) = pred_norm
       -> denormalize latest history state and pred state
       -> positivity / continuity / mass terms
       -> PhysicsLossResult(total, per_term)
       -> supervised loss + weighted physics loss
  -> epoch logs + per-profile metrics payload
  -> comparison summary remains performance-focused
```

## Data, Dependency, and Provenance Decisions

### Data and Artifact Identity

- Required inputs:
  - official CNS HDF5 file
  - `state_stats`
  - fixed field order `[density, Vx, Vy, pressure]`
- Required outputs:
  - per-profile metrics JSON with physics-loss provenance
  - updated CNS summary doc
- Checksum / manifest fields:
  - no new dataset identity fields beyond existing HDF5 metadata; extend metadata with `dx/dy/dt/eta/zeta/boundary`
- Freshness or cache policy:
  - reuse current staged file and normalization artifacts from the run root
- Reuse policy for historical artifacts:
  - earlier MAE CNS pilots remain historical only; new physics-loss runs must carry explicit training-loss and regularization provenance

### Dependency Discovery

- Discovery scope:
  - Search current repo and environment only
  - No new external package is expected
- Installation policy:
  - No production dependency additions for v1
- Fallback if no acceptable dependency is found:
  - Use only `torch` tensor ops (`torch.roll`) for periodic finite differences

### Provenance and Reproducibility

- Required command capture:
  - CLI must record whether physics regularization is on and the weights used
- Required environment capture:
  - existing invocation/provenance capture is sufficient
- Required random seeds or determinism policy:
  - unchanged from current CNS runner
- Required artifact manifest fields:
  - `physics_regularization_enabled`
  - `physics_loss_terms`
  - `physics_loss_weights`
  - `dx`, `dy`, `dt`
  - `boundary_condition`
- Required evidence logs:
  - per-epoch logging of supervised loss and active physics terms

## Claim, Behavior, or API Boundaries

- Allowed claim / behavior:
  - The PDEBench image-suite runner can optionally add reusable physics regularization for the CNS task.
- Disallowed claim / behavior:
  - “Physics-informed” or “PDE-consistent” benchmark claims without explicit evidence
  - implying SWE/Darcy support exists in v1
- Required caveat or limitation:
  - Only the CNS backend is implemented in v1, and only positivity/continuity/global-mass terms are active.
- Conditions that narrow the claim:
  - If only one profile is trained with the new loss, it must be labeled as a method variant rather than a pure architecture comparison.

## Pivot Criteria and Stop Conditions

- Pivot to smaller scope if:
  - continuity residual scaling proves unstable in focused tests or the smoke run
- Stop before user-facing / reviewer-facing claims if:
  - metadata does not expose `dx/dy/dt` cleanly, or the smoke run fails with the framework enabled
- Treat as exploratory only if:
  - the framework lands but only positivity/global-mass terms are stable
- Escalate for human decision if:
  - the user wants momentum or energy terms folded into the same tranche

## Required Final Assets

- Code or scripts:
  - reusable physics-loss module
  - CNS runner/data plumbing updates
- Tests:
  - synthetic loss/derivative tests
  - CNS runner/provenance tests
- Machine-readable outputs:
  - per-profile metrics JSON fields for physics regularization
- Figures / tables / docs:
  - updated CNS summary and docs index entry
- Manifests / logs:
  - CNS HDF5 metadata extended with spacing/boundary fields

If the design is abandoned or pivots:

- Required artifact note:
  - record the blocked term or unstable residual in the CNS summary
- Required rejected-candidate or failed-attempt summary:
  - document which term(s) were deferred and why
- Required docs / checklist updates:
  - update the CNS summary and relevant roadmap state

## Verification Plan

- Unit or integration tests:
  - periodic derivative helper tests
  - positivity/continuity/global-mass unit tests
  - runner provenance/config tests
- Artifact inspections:
  - inspect emitted CNS metadata for `dx/dy/dt/eta/zeta`
  - inspect per-profile metrics JSON for physics-loss fields
- Manifest/schema checks:
  - `python -m compileall -q scripts/studies/pdebench_image128`
  - focused `pytest` selectors for new and touched tests
- Reproducibility checks:
  - small readiness smoke with physics regularization enabled
- Manual inspection:
  - confirm epoch logs print supervised and physics losses separately
- Paper or docs build checks, if relevant:
  - N/A

## Open Questions

| ID | Question | Owner | Resolution needed by | Status |
|---|---|---|---|---|
| Q1 | Should `--physics-loss-weights` use a single packed argument or three explicit flags? | implementation plan | before code | open |
| Q2 | Should continuity residual default weight be tuned conservatively for the 512-trajectory capped smoke, or stay at a placeholder `1.0`? | implementation plan | before smoke run | open |

## Planning Handoff Checklist

- [x] Design status is `approved` with approval source recorded.
- [x] `docs/index.md` was used and selected docs/findings are listed.
- [x] Scope, non-goals, and claim boundaries are explicit.
- [x] Reusable-vs-task-local boundaries are explicit.
- [x] Verification requirements are concrete enough for an implementation plan.
