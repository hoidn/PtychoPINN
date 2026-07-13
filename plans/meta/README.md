# Historical Meta-Level Process Documents

This directory preserves earlier process proposals, workflow analyses, and
initiative worksheets for provenance. It is not the current planning entry
point and its documents do not impose prerequisites on new work.

## Current authority boundary

Start at [`docs/index.md`](../../docs/index.md) and the repository instructions
to identify current policy. Use [`plans/templates/`](../templates/) for
canonical planning templates and `docs/plans/` for new active plans.

In particular, the Phase -1/Phase 0 constraint worksheets and mandatory test
strategy proposals preserved here are historical. They may help explain past
initiatives, but they are not required before planning or implementation and
must not override the proportional current templates. Bounded work with clear
requirements may start directly; unresolved durable behavior, architecture,
public or scientific contract, migration, or material-alternative decisions
route through the current design template.

## Preserved contents

- [`PROCESS_OPTIMIZATIONS.md`](PROCESS_OPTIMIZATIONS.md),
  [`INFRASTRUCTURE_OPTIMIZATIONS.md`](INFRASTRUCTURE_OPTIMIZATIONS.md), and
  [`CLAUDE_MD_UPDATES.md`](CLAUDE_MD_UPDATES.md) record proposals from the
  earlier development loop. Their imperative wording is proposal/history, not
  current repository policy.
- [`UNIVERSAL_PATTERN.md`](UNIVERSAL_PATTERN.md) preserves the Phase -1 analysis
  rationale and explicitly marks its historical body as non-normative.
- [`templates/`](templates/) contains the retired constraint-analysis,
  cross-cutting-concerns, and test-strategy worksheets. Do not copy them as a
  default starting workflow.

Use these materials only for historical analysis or when an active,
higher-authority plan cites a specific item for context. New reusable planning
templates belong in [`plans/templates/`](../templates/), and initiative-specific
plans belong in `docs/plans/` (normally one
`docs/plans/YYYY-MM-DD-<initiative>.md` file).

## Related current documentation

- [`docs/index.md`](../../docs/index.md) — documentation and authority routing
- [`plans/README.md`](../README.md) — current planning locations and selection
- [`docs/DEVELOPER_GUIDE.md`](../../docs/DEVELOPER_GUIDE.md) — development guidance
- [`docs/workflows/agent_orchestration_backlog_loop.md`](../../docs/workflows/agent_orchestration_backlog_loop.md) — backlog-loop operations

**Status:** Historical / informative
