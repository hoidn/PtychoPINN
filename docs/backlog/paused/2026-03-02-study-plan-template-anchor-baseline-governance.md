# Backlog: Harden Study-Plan Templates for Anchor/Baseline Governance

**Created:** 2026-03-02
**Status:** Paused (Inactive)
**Priority:** High
**Related:** `docs/bugs/2026-03-02-hybrid-resnet-stage-anchor-and-baseline-governance.md`, `docs/backlog/templates/backlog_item_workflow.md`, `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md`, `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-stage-*.md`

## Summary
Improve project guidance/templates for study-type plans so canonical progression, baseline coverage, tie-break ordering, and execution-time validity gates are explicit, consistent, and audit-friendly by default.

## Problem
Recent study plans encode valid control/ablation intent but leave critical governance behavior under-specified or conflicting:

- canonical progression can use a control anchor instead of prior-step champion anchor,
- default baseline runs are required as artifacts but not guaranteed for every active `(dataset_profile, N)` promotion context,
- tie-break policies are deterministic but not consistently aligned with declared primary objectives,
- promotion-source usage is not fail-closed (missing cardinality/provenance checks allow wrong or ambiguous upstream source artifacts to drive canonical transitions).

Also, plans lack a generic execution-time blocker policy when evaluation quality is invalid/non-discriminative during runs (for example uniformly poor metrics across all model variants). Without a stop-and-revise contract, workflows can continue canonical promotion on bad evidence.

Finally, plans lack a meta-level contract for when execution reveals the plan itself is wrong or incomplete (not just when run output is poor). Without this, teams patch symptoms while leaving invalid plan assumptions in place.

These failures should be prevented by default plan/template guidance instead of ad hoc reviewer correction.

## Scope
- In scope:
  - Add/upgrade reusable study-plan guidance/template sections for:
    - canonical progression lane vs control-comparison lane,
    - baseline registry requirements per active `(dataset_profile, N)`,
    - deterministic tie-break chain aligned with promotion objectives,
    - fail-closed promotion-source contract for canonical transitions,
    - execution-time evaluation-validity gates and unblock policy.
  - Add a concise checklist section reviewers can apply mechanically to any study plan that uses iterative promotion decisions.
  - Add required artifact-path conventions for discoverable baseline metadata.
- Out of scope:
  - Re-running historical studies.
  - Changing model code or runbook behavior in this item.
  - Retrofitting every old plan immediately (follow-up migration item can handle that).

## Proposed Direction
1. Introduce a **Study Plan Governance Contract** snippet reusable across new study plans.
2. Require plans to declare two explicit lanes:
   - canonical promotion lane (next-step source must be previous-step robust feasible champion),
   - default-control lane (attribution/apples-to-apples only; non-canonical for progression transitions).
3. Add a template-level **Baseline Coverage Matrix** requiring one default baseline record per active `(dataset_profile, N)` pair used in promotion claims.
4. Standardize tie-break guidance order to objective-consistent precedence:
   - primary score direction,
   - runtime objective,
   - params,
   - inference (within SLA),
   - stable lexical fallback.
5. Add a mandatory **Fail-Closed Promotion-Source Contract** in execution templates:
   - canonical progression transitions must consume exactly one champion-source artifact (for example `promotion/champion_anchor_summary.csv`), not a control-lane artifact and not a multi-row summary,
   - champion-source artifact must carry deterministic selection provenance (source summary path + deterministic tie-break chain used),
   - canonical progression must hard-fail when source is missing, empty, multi-row, control-lane, or missing selection provenance.
6. Add a mandatory **Execution-Time Evaluation Validity Gate** section:
   - define generic trigger conditions for invalid/non-discriminative evaluation evidence,
   - require canonical promotion freeze when triggered,
   - require plan revision (including profile classification: canonical vs diagnostic-only) before unblocking.
7. Add a mandatory **Rebuild After Revision** contract:
   - regenerate promotion artifacts and baseline registry for affected execution scopes,
   - rerun dependent scopes that consumed invalid evidence,
   - persist revision/rerun provenance paths in execution artifacts.
8. Add a mandatory **Plan-Defect Escalation Loop** section:
   - when execution uncovers a plan-level defect, classify it explicitly as `plan_defect` (separate from `execution_defect`),
   - freeze canonical progression immediately,
   - require a plan update that states root cause, corrected contract text, and affected evidence scope,
   - require explicit re-entry criteria before canonical progression resumes.

## Meta Governance Requirement
- Study-plan templates must include a "Plan Defect Discovered During Execution" subsection that defines:
  - detection signal examples (conflicting plan contracts, non-actionable acceptance criteria, invalid promotion-source assumptions, invalid dataset-profile assumptions),
  - ownership (who updates plan text vs who reruns execution),
  - unblock sequence (revise plan -> regenerate evidence -> rerun dependent scopes -> resume progression),
  - provenance update requirements (link from execution log to revised plan section and rerun evidence paths).

## Acceptance Criteria
1. A reusable guidance section exists in planning docs/templates for study plans that use iterative promotion decisions, covering anchor lane separation, baseline coverage, and tie-break order.
2. New study plans must include an explicit baseline registry path and coverage matrix for all active `(dataset_profile, N)` pairs.
3. Execution plan templates/checklists require canonical `--promotion-source-summary` to reference prior-step champion source, not control-anchor source, and fail-closed rules enforce:
   - exactly one-row champion-source artifact for canonical transition input,
   - deterministic selection provenance recorded on that artifact,
   - hard failure for missing/ambiguous/control-lane source artifacts.
4. Tie-break guidance is objective-consistent and identical across hub/design/execution templates.
5. Reviewer checklist includes fail-fast checks for:
   - control-vs-canonical lane confusion,
   - missing baseline rows for active promotion contexts,
   - inconsistent tie-break ordering across docs.
6. Templates include an execution-time validity gate that blocks canonical promotion when evaluation evidence is invalid/non-discriminative.
7. Templates include an unblock workflow requiring plan revision, artifact regeneration, and downstream reruns before canonical promotion resumes.
8. Templates include an explicit meta-level plan-defect escalation loop, and reviewer checklists can distinguish `plan_defect` from `execution_defect` with different unblock requirements.
