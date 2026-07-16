# Plans Directory Organization

`plans/templates/` is the canonical template catalog. New active plans live in
`docs/plans/`; `plans/meta/` and older plan trees preserve historical process
and initiative material and do not define current prerequisites.

---

## Refactoring / debt-paydown initiative (2026-07)

Codebase-wide debt paydown, framed by a 6-analysis audit. The
[`2026-07-07-refactoring-roadmap.md`](../docs/plans/2026-07-07-refactoring-roadmap.md) roadmap
is the initiative's sole live status and sequencing authority. The phase plans
below preserve historical execution detail and design context; their embedded
status notes and task-local coordination procedures do not independently govern
current eligibility.

- [`2026-07-07-refactoring-roadmap.md`](../docs/plans/2026-07-07-refactoring-roadmap.md) — live strategic frame and status authority for the audited root risks and seven vertical boundary slices.
- [`2026-07-07-refactor-phase-0-cleanup.md`](../docs/plans/2026-07-07-refactor-phase-0-cleanup.md) — historical Phase 0 subtractive-cleanup execution record (RG3).
- [`2026-07-07-refactor-phase-1-safety-net.md`](../docs/plans/2026-07-07-refactor-phase-1-safety-net.md) — historical Phase 1 safety-net execution plan (RG2/RG4).
- [`2026-07-07-refactor-phase-2-consolidate.md`](../docs/plans/2026-07-07-refactor-phase-2-consolidate.md) — historical Phase 2 consolidation plan (RG3/RG5).
- [`2026-07-07-refactor-phase-3-core-extraction.md`](../docs/plans/2026-07-07-refactor-phase-3-core-extraction.md) — historical Phase 3 backend-neutral-core planning record (RG1/RG5).
- [`2026-07-06-pipeline-consolidation.md`](../docs/plans/2026-07-06-pipeline-consolidation.md) and [`2026-07-06-pipeline-consolidation-tiers-0-2.md`](../docs/plans/2026-07-06-pipeline-consolidation-tiers-0-2.md) — historical vertical-slice context for reassembly, inference, and solver work.

---

## Current planning locations

| Location | Current role |
| --- | --- |
| `plans/templates/` | Canonical design, implementation, debug, workflow-contract, and test-strategy templates |
| `docs/plans/` | Active plans and durable initiative-specific planning records |
| `plans/meta/` | Historical process proposals, analyses, and worksheets; informative lineage only |
| `plans/active/` and `plans/*.md` | Older initiative plans retained for reference; not the default destination for new work |

## Choosing a planning path

Bounded work with clear requirements and no unresolved durable decision may
start directly. When an execution plan is useful, copy
[`templates/implementation_plan.md`](templates/implementation_plan.md) to the
single-file default `docs/plans/YYYY-MM-DD-<initiative>.md` and keep only the
sections proportional to the work.

Use [`templates/design_template.md`](templates/design_template.md) first only
when the work still needs a durable decision about behavior, architecture, a
public or scientific contract, migration policy, or material alternatives.
Once accepted, link that design from the implementation plan rather than
reopening its decisions there.

Specialized templates are optional and apply only to their named concern:

- [`templates/debug_task_plan.md`](templates/debug_task_plan.md) for a bounded,
  hypothesis-driven investigation; use the implementation template for a fix.
- [`templates/workflow_contract_plan.md`](templates/workflow_contract_plan.md)
  as a supplement when data crosses orchestrated step or stage boundaries.
- [`templates/test_strategy_template.md`](templates/test_strategy_template.md)
  as a supplement when test architecture, CI tiers, expensive scientific
  validation, or an environment/dependency capability contract changes.
- `templates/implementation_plan_simple.md` is a compatibility pointer to the
  canonical implementation template, not a second planning process.

Use an initiative folder under `docs/plans/` only when genuinely
multi-document work needs a shared durable home. Historical worksheets in
`plans/meta/templates/` are not required starting points, gates, or
prerequisites.

## Completing work

Keep the plan's status and verification evidence accurate within the plan's
declared scope. Add or update the relevant `docs/index.md` entry when the plan
or its durable outcome needs discoverability. There is no general requirement
to create archived log trees, move plans between lifecycle folders, or extract
new process documents after ordinary completion.

---

## See Also

- `meta/README.md` - Historical process-document guide and current authority boundary
- `docs/workflows/agent_orchestration_backlog_loop.md` - How to run initiatives
- `docs/DEVELOPER_GUIDE.md` - Development best practices
- `CLAUDE.md` - Core project directives

---

**Last Updated:** 2026-07-15
**Maintainer:** Project leads
