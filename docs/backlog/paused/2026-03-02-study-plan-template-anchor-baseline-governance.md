# Backlog: Harden Study-Plan Templates for Anchor/Baseline Governance

**Created:** 2026-03-02
**Status:** Paused (Inactive)
**Priority:** High
**Related:** `docs/bugs/2026-03-02-hybrid-resnet-stage-anchor-and-baseline-governance.md`, `docs/backlog/templates/backlog_item_workflow.md`, `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md`, `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-stage-*.md`

## Summary
Improve project guidance/templates for study-type plans so canonical stage progression, baseline coverage, tie-break ordering, and execution-time validity gates are explicit, consistent, and audit-friendly by default.

## Problem
Recent study plans encode valid control/ablation intent but leave critical governance behavior under-specified or conflicting:

- canonical stage progression can use a control anchor instead of prior-stage champion anchor,
- default baseline runs are required as artifacts but not guaranteed for every active `(dataset_profile, N)` promotion context,
- tie-break policies are deterministic but not consistently aligned with declared primary objectives.

Also, plans lack a generic execution-time blocker policy when evaluation quality is invalid/non-discriminative during runs (for example uniformly poor metrics across all model variants). Without a stop-and-revise contract, workflows can continue canonical promotion on bad evidence.

These failures should be prevented by default plan/template guidance instead of ad hoc reviewer correction.

## Scope
- In scope:
  - Add/upgrade reusable study-plan guidance/template sections for:
    - canonical progression lane vs control-comparison lane,
    - baseline registry requirements per active `(dataset_profile, N)`,
    - deterministic tie-break chain aligned with promotion objectives,
    - execution-time evaluation-validity gates and unblock policy.
  - Add a concise checklist section reviewers can apply mechanically to any staged study plan.
  - Add required artifact-path conventions for discoverable baseline metadata.
- Out of scope:
  - Re-running historical studies.
  - Changing model code or runbook behavior in this item.
  - Retrofitting every old plan immediately (follow-up migration item can handle that).

## Proposed Direction
1. Introduce a **Study Plan Governance Contract** snippet reusable across new study plans.
2. Require plans to declare two explicit lanes:
   - canonical promotion lane (next-stage source must be previous-stage robust feasible champion),
   - default-control lane (attribution/apples-to-apples only; non-canonical for stage transitions).
3. Add a template-level **Baseline Coverage Matrix** requiring one default baseline record per active `(dataset_profile, N)` pair used in promotion claims.
4. Standardize tie-break guidance order to objective-consistent precedence:
   - primary score direction,
   - runtime objective,
   - params,
   - inference (within SLA),
   - stable lexical fallback.
5. Add a mandatory "promotion-source summary contract" line in stage execution templates to prevent control-anchor misuse.
6. Add a mandatory **Execution-Time Evaluation Validity Gate** section:
   - define generic trigger conditions for invalid/non-discriminative evaluation evidence,
   - require canonical promotion freeze when triggered,
   - require plan revision (including profile classification: canonical vs diagnostic-only) before unblocking.
7. Add a mandatory **Rebuild After Revision** contract:
   - regenerate promotion artifacts and baseline registry for affected stage(s),
   - rerun downstream stages that consumed invalid evidence,
   - persist revision/rerun provenance paths in stage artifacts.

## Acceptance Criteria
1. A reusable guidance section exists in planning docs/templates for staged study plans covering anchor lane separation, baseline coverage, and tie-break order.
2. New study plans must include an explicit baseline registry path and coverage matrix for all active `(dataset_profile, N)` pairs.
3. Stage execution plan templates/checklists require canonical `--promotion-source-summary` to reference prior-stage champion source, not control-anchor source.
4. Tie-break guidance is objective-consistent and identical across hub/design/stage templates.
5. Reviewer checklist includes fail-fast checks for:
   - control-vs-canonical lane confusion,
   - missing baseline rows for active promotion contexts,
   - inconsistent tie-break ordering across docs.
6. Templates include an execution-time validity gate that blocks canonical promotion when evaluation evidence is invalid/non-discriminative.
7. Templates include an unblock workflow requiring plan revision, artifact regeneration, and downstream reruns before canonical promotion resumes.
