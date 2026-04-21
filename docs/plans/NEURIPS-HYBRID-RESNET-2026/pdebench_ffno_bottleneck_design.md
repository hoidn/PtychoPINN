# PDEBench FFNO-Close Bottleneck Variant Design

## Design Metadata

- ID: `NEURIPS-HYBRID-RESNET-2026-pdebench-ffno-bottleneck`
- Title: FFNO-Close Bottleneck Variant For The PDEBench 128x128 Image Suite
- Status: draft
- Date: 2026-04-21
- Source brief / issue: define a second optional PDEBench bottleneck family that is closer to an FFNO-style bottleneck than the existing spectral-ResNet bottleneck, then compare the local, spectral-ResNet, and FFNO-close bottlenecks fairly under the current canonical CNS shell.
- Related plan: ``
- Related checklist / backlog item: `docs/backlog/paused/2026-04-20-pdebench-ffno-bottleneck-variant.md`

## Consumed Inputs and Authority

- Primary source: user request in this session
- Docs index: `docs/index.md` read first? `yes`
- Project docs:
  - `AGENTS.md`
  - `docs/findings.md`
  - `docs/INITIATIVE_WORKFLOW_GUIDE.md`
  - `docs/model_baselines.md`
- Architecture docs:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_design.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_resnet_bottleneck_design.md`
- Workflow guides / runbooks:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- Prior artifacts or reports:
  - `docs/backlog/paused/2026-04-20-pdebench-ffno-bottleneck-variant.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/spectral_resnet_bottleneck_n128_integration_summary.md`
- Code surfaces that constrain the design:
  - `scripts/studies/pdebench_image128/models.py`
  - `scripts/studies/pdebench_image128/run_config.py`
  - `ptycho_torch/generators/spectral_resnet_bottleneck.py`
  - `ptycho_torch/generators/fno.py`
- Known findings / policies:
  - obey `PYTHON-ENV-001` from `docs/DEVELOPER_GUIDE.md`
  - do not promote capped readiness evidence into benchmark claims

Authority order for this design:

1. `AGENTS.md`
2. `docs/index.md` and the selected PDEBench/CNS design docs it points to
3. the existing spectral bottleneck design and paused FFNO backlog note

## Problem and Scope

- Problem:
  The repo has one implemented non-local PDEBench bottleneck variant, `spectral_resnet_bottleneck_net`, but that variant is explicitly narrower than an FFNO-style bottleneck. The repo also has a paused note describing an FFNO-style bottleneck idea, but not an active design for implementing it fairly against the current canonical CNS shell.
- User / reviewer / system need:
  Define a bottleneck-only FFNO-close variant that can be compared against both the current local Hybrid bottleneck and the current spectral-ResNet bottleneck without conflating the result with skip-policy or shell differences.
- In scope:
  - a new optional bottleneck family outside `hybrid_resnet_*`
  - a supervised PDEBench wrapper that reuses the current canonical CNS shell
  - a fairness contract for comparing local, spectral-ResNet, and FFNO-close bottlenecks
  - required tests, profile plumbing, and comparison artifacts
- Out of scope:
  - a full FFNO encoder-decoder rewrite
  - changing PDEBench task data contracts, splits, or metrics
  - making this new row part of required benchmark bundles before it earns that role
- Non-goals:
  - claiming equivalence to the original F-FNO paper architecture
  - matching published FFNO hyperparameters before the bottleneck-only comparison is stabilized
  - replacing the current canonical `hybrid_resnet_cns` row

## Decision Summary

- Decision:
  Add a new optional family, `ffno_bottleneck_net`, that swaps only the bottleneck inside the current canonical CNS shell.
- Rationale:
  The only defensible way to answer whether an FFNO-close bottleneck helps is to keep the shell fixed and change only the bottleneck. The current spectral bottleneck row is useful, but shell mismatches would otherwise confound the comparison.
- Expected implementation shape:
  Refactor the supervised PDEBench shell so that the local, spectral-ResNet, and FFNO-close bottlenecks can all run under the same skip-add CNS shell.
- Claim or behavior limits:
  The variant is a bottleneck-only FFNO-close adaptation, not a paper-faithful full-resolution FFNO baseline.
- Pivot or abandonment condition:
  Stop if the shell cannot be equalized across rows or if the FFNO-close row requires enough extra architecture changes that the result no longer isolates the bottleneck.

## Decision Records

### ADR-001: Compare bottlenecks under the canonical CNS shell

- Status: proposed
- Decision:
  Use the current `hybrid_resnet_cns` shell as the fairness anchor for all bottleneck comparisons.
- Context:
  The repo has already promoted skip-add into the canonical CNS Hybrid row. Any bottleneck comparison that changes skip policy, encoder tap usage, decoder path, or upsampler at the same time will not isolate the bottleneck.
- Rationale:
  A fair answer requires the same:
  - lifter
  - encoder blocks
  - downsample schedule
  - skip-add policy
  - decoder / upsampler
  - output head
  - loss, scheduler, split, and metrics
  Only the bottleneck should vary.
- Alternatives considered:
  - Compare all rows with skips disabled - simpler, but weaker scientifically because it diverges from the current real CNS baseline.
  - Let the spectral and FFNO-close rows keep separate shells - rejected because shell differences would dominate interpretation.
- Consequences:
  The current spectral bottleneck wrapper likely needs to be lifted into the same common shell before any fair multi-row comparison is reported.
- Evidence required before implementation:
  Confirm that the current shell logic in `scripts/studies/pdebench_image128/models.py` can be factored without changing the existing `hybrid_resnet_cns` output contract.
- Follow-up required if this decision changes:
  Rewrite the comparison framing and benchmark caveats.

### ADR-002: Keep the FFNO-close family outside `hybrid_resnet_*`

- Status: proposed
- Decision:
  Name the new family `ffno_bottleneck_net` and the first PDEBench profile `ffno_bottleneck_base`.
- Context:
  The existing spectral bottleneck design already established that materially different bottleneck families should not be hidden under `hybrid_resnet_*`.
- Rationale:
  This keeps benchmark tables readable and avoids implying that the new row is just another recommended Hybrid baseline.
- Alternatives considered:
  - `hybrid_resnet_ffno` - rejected because it obscures the architectural difference.
  - `hybrid_resnet_bottleneck_ffno` - rejected for the same reason.
- Consequences:
  The new family stays manual opt-in until it proves competitive and stable.
- Evidence required before implementation:
  None beyond consistency with the existing naming policy.
- Follow-up required if this decision changes:
  Update docs, profile IDs, and comparison summaries together.

### ADR-003: Implement an FFNO-close bottleneck, not a full FFNO rewrite

- Status: proposed
- Decision:
  Implement a shape-preserving bottleneck stack with factorized spectral mixing plus channel feedforward sublayers, but do not rewrite the encoder or decoder into a full FFNO architecture.
- Context:
  The user asked for something close to the FFNO bottleneck. The current campaign already has a working PDEBench shell and a separate spectral-ResNet bottleneck family. A full rewrite would broaden the scope too far and destroy fairness.
- Rationale:
  The missing architectural ingredient today is not a factorized spectral operator; the repo already has one. The missing ingredient is a bottleneck block with FFNO-close spectral-plus-feedforward semantics.
- Alternatives considered:
  - Keep the local `3x3` branch inside the FFNO-close bottleneck - rejected as the default because it blurs the contrast against `spectral_resnet_bottleneck_net`.
  - Full FFNO encoder-decoder rewrite - rejected as out of scope.
- Consequences:
  The new bottleneck will still require explicit caveats when compared against published FFNO CNS numbers.
- Evidence required before implementation:
  Define the bottleneck block contract precisely enough that the planner can write unit tests first.
- Follow-up required if this decision changes:
  Re-evaluate parameter-count fairness and comparison scope.

## Proposed Design

### Implementation Shape

- Existing code or workflow to reuse:
  - `FactorizedSpectralConv2d` from `ptycho_torch/generators/spectral_resnet_bottleneck.py`
  - current PDEBench supervised shell logic from `scripts/studies/pdebench_image128/models.py`
  - current profile and reporting machinery from `scripts/studies/pdebench_image128/run_config.py` and reporting helpers
- New files or artifacts likely needed:
  - `ptycho_torch/generators/ffno_bottleneck.py`
  - `tests/torch/test_ffno_bottleneck.py`
  - this design document
- Files or APIs likely touched:
  - `scripts/studies/pdebench_image128/models.py`
  - `scripts/studies/pdebench_image128/run_config.py`
  - `tests/studies/test_pdebench_image128_models.py`
  - possibly runner/reporting tests if required-primary profile logic or profile descriptions change
- Files or APIs that must not be touched:
  - `ptycho/model.py`
  - `ptycho/diffsim.py`
  - `ptycho/tf_helper.py`
- One-off versus reusable boundary:
  The FFNO-close bottleneck family should be reusable across PDEBench tasks, but the fairness anchor for this design is specifically the current CNS shell.
- Design-plan-implement boundary:
  - Decisions this design must make:
    - fairness anchor shell
    - naming and family exposure
    - bottleneck block semantics
    - comparison protocol and claim boundaries
  - Details intentionally deferred to the implementation plan:
    - exact normalization choice if more than one is plausible
    - exact MLP ratio default if the initial candidate needs tuning
    - line-by-line file edit sequencing

### Core Contracts and Invariants

- Contract:
  All three comparison rows must share the same supervised shell contract and differ only in bottleneck implementation.
- Invariant:
  `hybrid_resnet_cns`, `spectral_resnet_bottleneck_base`, and `ffno_bottleneck_base` must produce the same task-level tensor shape for the same PDEBench task.
- Failure mode if violated:
  Any comparison result becomes shell-confounded rather than bottleneck-isolated.
- How the implementation should prove it:
  Builder tests, parameter-count capture before first forward, shared-shell shape tests, and deterministic capped CNS comparison runs.

- Contract:
  The FFNO-close bottleneck remains shape-preserving: `(B, C, H, W) -> (B, C, H, W)`.
- Invariant:
  No spatial-size or channel-count change inside the bottleneck stack.
- Failure mode if violated:
  The common shell becomes harder to share and the comparison loses scope control.
- How the implementation should prove it:
  Unit tests on representative latent shapes such as `(2, 128, 32, 32)`.

- Contract:
  The new row is an optional ablation family, not a default benchmark row.
- Invariant:
  Required benchmark bundles remain unchanged until future evidence explicitly changes policy.
- Failure mode if violated:
  The suite would silently redefine the baseline policy before the variant has earned it.
- How the implementation should prove it:
  Profile-set tests and docs updates.

### Data Flow / Control Flow

```text
PDEBench CNS history-window sample
  -> shared supervised shell (same as canonical `hybrid_resnet_cns`)
  -> selected bottleneck:
       local ResNet
       or spectral-ResNet
       or FFNO-close
  -> same decoder / skip-add fusion / output head
  -> standard CNS metrics and sample PNG rendering
  -> per-profile metrics JSON + comparison summary + galleries
```

## Data, Dependency, and Provenance Decisions

### Data and Artifact Identity

- Required inputs:
  - the existing PDEBench CNS dataset contract already used by the suite
  - deterministic split manifest for capped comparison runs
- Required outputs:
  - per-profile metrics JSON
  - comparison summary JSON
  - sample prediction PNGs and error PNGs
  - model metadata including parameter count
- Checksum / manifest fields:
  Reuse current suite manifests; no new dataset identity scheme is required.
- Freshness or cache policy:
  The first comparison should use fresh capped runs under a new artifact root.
- Reuse policy for historical artifacts:
  Prior spectral or Hybrid runs are context only and must not be mixed into a new same-shell bottleneck comparison unless the split and shell are proven identical.

### Dependency Discovery

- Discovery scope:
  - Search current repo and environment:
    - reuse existing factorized spectral pieces first
  - Search external package sources:
    - only if the repo lacks a needed shape-preserving feedforward or factorized spectral primitive
  - Candidate acceptance criteria:
    - preserves PyTorch-only PDEBench path
    - shape-preserving bottleneck semantics
    - no new runtime service dependency
  - Candidate rejection criteria:
    - rewrites the PDEBench shell
    - imposes a full FFNO training stack in the first tranche
    - muddies fairness against the current CNS row
- Installation policy:
  Prefer repo-local implementation using existing PyTorch and `neuraloperator` dependencies already present in the project.
- Fallback if no acceptable dependency is found:
  Implement the FFNO-close bottleneck with repo-local PyTorch modules and the existing factorized spectral operator.

### Provenance and Reproducibility

- Required command capture:
  Reuse the current PDEBench suite command capture and artifact manifests.
- Required environment capture:
  Reuse current suite metadata capture.
- Required random seeds or determinism policy:
  Use deterministic split manifests and explicit seed capture for capped comparison runs.
- Required artifact manifest fields:
  - profile ID
  - base model family
  - bottleneck kind
  - parameter count
  - split manifest path
  - training loss trace
- Required evidence logs:
  - targeted pytest logs
  - compile check log if touched modules require it
  - capped comparison run logs

## Claim, Behavior, or API Boundaries

- Allowed claim / behavior:
  - the repo contains a bottleneck-only FFNO-close PDEBench family
  - capped CNS comparisons isolate the bottleneck under the same shell
  - the new row is optional/manual unless future policy changes promote it
- Disallowed claim / behavior:
  - this is a paper-faithful full FFNO baseline
  - capped runs prove benchmark superiority
  - gains, if any, are attributable to shell changes
- Required caveat or limitation:
  The row is a bottleneck-only FFNO-close adaptation inside the existing Hybrid-style shell.
- Conditions that narrow the claim:
  If the spectral or FFNO-close rows do not fully match the canonical CNS shell, results are exploratory only.

## Pivot Criteria and Stop Conditions

- Pivot to smaller scope if:
  the common-shell refactor becomes larger than the bottleneck implementation itself; in that case, first land the shared-shell refactor without claiming any bottleneck result.
- Stop before user-facing / reviewer-facing claims if:
  any comparison row still differs in skip policy, decoder path, or loss contract.
- Treat as exploratory only if:
  the first evidence is limited to capped CNS pilots or if one row needs ad hoc shell changes to run.
- Escalate for human decision if:
  parameter-count blow-up or training instability makes the FFNO-close row incomparable to the other rows without additional fairness policy.

## Required Final Assets

- Design document:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_design.md`
- Future implementation plan:
  - to be written separately after design approval
- Required future evidence bundle if implemented:
  - unit tests for the FFNO-close bottleneck
  - PDEBench model-builder tests
  - capped same-shell CNS comparison artifacts
