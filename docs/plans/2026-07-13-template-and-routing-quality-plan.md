# Plan And Design Template Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give PtychoPINN one clearly routed, proportionate design-and-planning
template system with optional scientific rigor and no mandatory procedural
worksheets for ordinary work.

**Architecture:** `plans/templates/` remains the canonical template directory
and continues to be reachable through the existing `docs/plans/templates`
symlink. One general design template and one general implementation-plan
template own ordinary work; debug, workflow-contract, and standalone
test-strategy templates remain narrow optional extensions. Older duplicate
paths become compatibility pointers, while legacy Phase -1 material is labeled
historical rather than silently deleted.

**Tech Stack:** Markdown, repository-relative links, Git symlinks, `rg`, shell
link/path validation.

**Status:** Complete (2026-07-13)

---

## Authority and scope

- `docs/index.md` is the discoverability authority.
- `plans/templates/design_template.md` owns general designs.
- `plans/templates/implementation_plan.md` owns general implementation plans.
- `plans/templates/debug_task_plan.md`,
  `plans/templates/workflow_contract_plan.md`, and
  `plans/templates/test_strategy_template.md` are optional specialized aids;
  they do not replace the general design or plan.
- `docs/plans/templates` remains a compatibility symlink to
  `plans/templates`.
- Existing completed plans and run artifacts are out of scope.
- Ordinary bounded work may start directly from an implementation plan. A
  separate design is required only when behavior, architecture, public or
  scientific contracts, migration policy, or material alternatives need a
  durable decision.
- Single-file plans under `docs/plans/YYYY-MM-DD-<initiative>.md` are the
  default. Initiative folders are used only when multiple durable documents
  genuinely need a shared home.
- Backlog-item, blocker-report, and workflow-idea schemas are specialized
  runtime/queue contracts and are out of scope unless routing incorrectly
  presents them as general design or plan templates.

## Task 1: Consolidate the canonical design surface

**Files:**

- Modify: `plans/templates/design_template.md`
- Modify: `docs/templates/design_template.md`

- [x] Rewrite the canonical design template around durable decisions:
  authority, problem, goals/non-goals, decision, design details, contracts,
  dependencies, invariants/failures, compatibility, verification, declarative
  acceptance, success/stop criteria, docs impact, and implementation handoff.
- [x] Preserve optional PtychoPINN-specific sections for scientific claims,
  data identity, provenance, reproducibility, and pivot criteria.
- [x] State that sections should be omitted when irrelevant; do not require an
  ADR, dependency search, workflow contract, or asset inventory for every
  design.
- [x] Replace `docs/templates/design_template.md` with a short compatibility
  pointer to the canonical template so two bodies cannot drift.

## Task 2: Consolidate ordinary and specialized plans

**Files:**

- Modify: `plans/templates/implementation_plan.md`
- Modify: `plans/templates/implementation_plan_simple.md`
- Modify: `plans/templates/debug_task_plan.md`
- Modify: `plans/templates/workflow_contract_plan.md`
- Modify: `plans/templates/test_strategy_template.md`

- [x] Rewrite the general implementation plan so tasks name exact files,
  observable RED/GREEN evidence where behavior changes, exact commands,
  expected results, bounded commit/review checkpoints, completion criteria,
  and stop conditions.
- [x] Make single-file `docs/plans/YYYY-MM-DD-<initiative>.md` authoring the
  default in canonical and specialized template examples; describe an
  initiative folder as an optional escalation for multi-document work.
- [x] Remove mandatory compliance, findings, fix-plan, report-directory, and
  context-inventory boilerplate. Retain authority and scientific/provenance
  fields only when relevant to the selected work.
- [x] Replace `implementation_plan_simple.md` with a compatibility pointer to
  the single canonical general plan.
- [x] Keep the debug template hypothesis-driven but remove mandatory archived
  report trees; evidence paths are required only when the investigation
  produces durable evidence.
- [x] Replace the workflow template's hard two-artifact ceiling with a
  minimal-sufficient contract rule. Require producer, consumer, type,
  validation, authority, freshness, and failure behavior for every promoted
  workflow-level artifact.
- [x] Reduce the standalone test-strategy template to an optional aid for work
  that changes test architecture, CI tiers, expensive scientific validation,
  or environment capability. Ordinary test steps stay in the implementation
  plan.

## Task 3: Demote legacy process worksheets

**Files:**

- Modify: `plans/meta/templates/constraint_analysis_template.md`
- Modify: `plans/meta/templates/cross_cutting_concerns_template.md`
- Modify: `plans/meta/templates/test_strategy_template.md`
- Modify: `plans/meta/UNIVERSAL_PATTERN.md`

- [x] Replace the three active-looking meta templates with concise historical
  compatibility notices pointing to current canonical templates and optional
  sections.
- [x] Add a clear historical/superseded banner to `UNIVERSAL_PATTERN.md`:
  Phase -1 analysis is not required for new work, and relevant concerns are
  selected proportionately through the current templates.
- [x] Preserve the historical body for provenance; do not rewrite old claims
  as current policy.

## Task 4: Repair discoverability and routing

**Files:**

- Modify: `docs/index.md`
- Modify: `plans/README.md`
- Modify: `plans/meta/README.md`
- Modify: `docs/workflows/orchestration_start_here.md`

- [x] Make `docs/index.md` route bounded ordinary work directly to the general
  implementation plan, and route through the general design template first
  only when a durable design decision is needed. Select specialized templates
  only when their scope applies.
- [x] Ensure every index description matches the actual template and remove
  references to absent Do-Now or plan-update sections.
- [x] Rewrite the template portions of `plans/README.md` to identify
  `plans/templates/` as canonical, `docs/plans/` as the location for active
  plans, and `plans/meta/` as historical process material rather than a
  mandatory starting point.
- [x] Update `plans/meta/README.md` so legacy Phase -1 worksheets are historical
  references, not current starting points or mandatory prerequisites.
- [x] Update the orchestration guide to use the minimal-sufficient workflow
  contract rule and explain that the workflow-contract template supplements,
  rather than replaces, the general implementation plan.

## Task 5: Validate the authority chain

- [x] Run `git diff --check` over every touched path.
- [x] Confirm `docs/plans/templates` still resolves to `plans/templates` and
  each routed template exists.
- [x] Search active routing/templates for stale obligations:
  `Compliance Matrix (Mandatory)`, mandatory report roots, archived pytest
  logs, universal Phase -1, and the hard `< 3` artifact rule.
- [x] Check all Markdown links in touched files resolve to a file or anchored
  repository document where locally verifiable.
- [x] Review the complete diff and confirm no completed plan, run artifact,
  queue item, scientific result, or unrelated dirty file changed.

## Completion criteria

- One canonical general design template and one canonical general
  implementation-plan template are discoverable from `docs/index.md`, with
  proportional selection rules and single-file plans as the default.
- Specialized templates have explicit selection criteria and do not impose
  themselves on ordinary work.
- Scientific provenance, claim boundaries, and pivot/stop criteria remain
  available without being mandatory boilerplate.
- Historical Phase -1 materials cannot be mistaken for current policy.
- Existing compatibility paths continue to resolve.
- Routing text and template contents agree, and validation is clean.

## Execution evidence

- The implementation plan received an independent review, was revised for
  proportional design selection, meta-README routing, and single-file plan
  defaults, then was approved.
- Tasks 1-4 each received separate specification and quality reviews. Task 2's
  quality loop corrected copied-file link stability and removed duplicated
  command/result ownership from the optional test-strategy supplement.
- Exact-path `git diff --check` passed for all 16 changed planning and routing
  files.
- `docs/plans/templates` still resolves to `../../plans/templates`; every
  routed canonical/compatibility path exists.
- The active-template stale-obligation search returned no matches.
- Local Markdown-link validation passed across all 16 changed files.
- Existing completed plans, run artifacts, queue items, scientific results,
  and unrelated dirty paths were not modified.
