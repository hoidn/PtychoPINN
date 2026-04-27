# Current Objective

- Keep the NeurIPS Hybrid ResNet effort moving by selecting the next backlog item
  that most directly advances the current paper evidence and comparison story.

## Ordered Near-Term Priorities

- Preserve the approved NeurIPS roadmap gates and phase order.
- Prefer work that removes active paper blockers or strengthens core comparison
  evidence.
- Avoid spending execution budget on optional follow-ups while required
  prerequisite evidence is still missing.
- For the current PDEBench CNS external-baseline queue, run the author-FFNO
  equal-footing compare before the paper-default GNOT rerun unless FFNO is
  explicitly blocked.
- For external PDEBench CNS baselines, treat the paper-default GNOT rerun and
  the author-FFNO baseline as both valid comparison candidates. Choose ordering
  pragmatically based on expected baseline strength, setup cost, and training
  cost on the local equal-footing contract.

## Required Comparison Standards

- Keep equal-footing comparisons explicit.
- Preserve the metric, data-split, and protocol boundaries recorded in the
  approved NeurIPS design and roadmap unless a reviewed roadmap update says
  otherwise.

## Fairness / Apples-to-Apples Constraints

- Do not silently relax fairness constraints to make a backlog item easier.
- If a planned compare cannot be kept on equal footing, record the incompatibly
  constrained outcome instead of drifting the protocol.

## Known Blockers / Deferred Work

- Items blocked by missing upstream evidence, unavailable data, or unresolved
  roadmap prerequisites should remain blocked until the blocker is cleared.

## Non-Goals / Not Now

- Do not rewrite the NeurIPS roadmap every cycle.
- Do not treat the steering document as a mutable queue manifest.

## Initiative Guidance

- Use steering as strategic intent, the roadmap as ordered execution authority,
  backlog items as candidate work units, and fresh approved plans as the
  implementation authority for a selected item.
- Treat author FFNO as a real external CNS baseline, not just an optional
  follow-up. GNOT and FFNO can both be justified baseline work, but for the
  current queue FFNO should be attempted first and GNOT should follow only
  after FFNO completes or is recorded blocked.
