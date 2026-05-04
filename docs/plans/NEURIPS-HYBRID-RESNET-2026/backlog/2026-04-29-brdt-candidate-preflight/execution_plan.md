# BRDT Candidate Lane Umbrella Closeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. This item is administrative closeout only. Do not create worktrees, do not relaunch BRDT operator/dataset/adapter/preflight runs, do not add new BRDT functionality, and do not promote BRDT into manuscript tables, figures, or `/home/ollie/Documents/neurips/` outputs from this item. Keep any normal verification command under implementation ownership until it exits successfully or reaches a documented recoverable failure; reserve `BLOCKED` for missing required artifacts, unavailable external dependencies outside current authority, or an unrecoverable contradiction after one narrow fix attempt.

**Goal:** Close backlog item `2026-04-29-brdt-candidate-preflight` by auditing the already-completed BRDT child items, confirming that the durable summary and recommendation are discoverable, and making only the minimum discoverability updates required by that audit.

**Architecture:** Treat this as a documentation-and-state audit over completed BRDT outputs, not as another BRDT implementation pass. First confirm the child-item completion chain and the authoritative summary/recommendation surfaces. Then audit whether later readers can discover those outputs through the intended indexes without reopening raw artifact trees. Only if that discoverability audit fails should implementation edit an index or roadmap note; otherwise the correct outcome is a no-op repo closeout plus an execution report that records why no further change was needed.

**Tech Stack:** Markdown, JSON state files, backlog/item state under `state/NEURIPS-HYBRID-RESNET-2026/`, durable docs under `docs/plans/NEURIPS-HYBRID-RESNET-2026/`, PATH `python` for deterministic checks.

---

## Selected Objective

- Implement backlog item `2026-04-29-brdt-candidate-preflight`.
- Confirm the completed BRDT child items produced:
  - operator validation;
  - dataset preflight;
  - task adapters;
  - four-row decision-support preflight;
  - durable summary plus promotion/deferral/rejection recommendation.
- Confirm the durable summary and recommendation are discoverable through the intended project indexes.
- Update indexes or roadmap notes only if the audit finds a real discoverability gap.

## Scope

- Consume the BRDT candidate-lane design as background authority for claim boundaries:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md`
- Consume the completed BRDT child-item authorities:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_operator_validation_report.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_dataset_preflight.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_task_adapters.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_preflight_summary.md`
- Consume the discoverability surfaces that may need audit or conditional repair:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/index.md`
  - `docs/backlog/index.md`
- Consume the implementation-state, implementation-review, lineage-path, and roadmap-state evidence for the split BRDT items under `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/` plus `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`.
- Produce only the minimum repo edits needed to make the completed BRDT lane discoverable and unambiguous.

## Explicit Non-Goals

- Do not implement or modify BRDT physics, datasets, adapters, metrics, training, or orchestration.
- Do not rerun BRDT operator validation, dataset generation, adapter sanity passes, or the four-row preflight to satisfy this closeout.
- Do not widen BRDT into a manuscript pillar, evidence-amendment plan, or new benchmark lane.
- Do not rewrite the NeurIPS roadmap, steering document, or BRDT design unless a narrow discoverability correction absolutely requires a note and no smaller index fix exists.
- Do not create a vague “BRDT follow-up” scope here. If substantial BRDT work remains, implementation should leave it to a new concrete backlog item rather than expanding this umbrella item.

## Steering, Roadmap, and Prerequisite Constraints

- Steering is binding: CDI `lines128` and PDEBench CNS remain the required manuscript pillars, and BRDT stays additive candidate work only.
- The approved design, roadmap, inverse-wave rationale, and paper-evidence package all treat BRDT as an optional candidate lane that requires a later reviewed amendment before any paper-facing promotion.
- The progress ledger matters here as a concrete non-promotion check: implementation must verify from `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` that BRDT is not recorded as a completed roadmap tranche, blocked roadmap tranche, or paper-pillar promotion update. That keeps this closeout administrative and low-cost; it cannot silently reopen BRDT execution just because the child items were successful.
- Child-item prerequisite status is already satisfied and should be treated as locked input, not reopened work:
  - `2026-04-29-brdt-operator-validation`: `COMPLETED`, final implementation review `APPROVE`
  - `2026-04-29-brdt-dataset-preflight`: `COMPLETED`, final implementation review `APPROVE`
  - `2026-04-29-brdt-task-adapters`: `COMPLETED`, final implementation review `APPROVE`
  - `2026-04-29-brdt-four-row-preflight`: `COMPLETED`, final implementation review `APPROVE`
  - `2026-04-29-brdt-preflight-summary-promotion-decision`: `COMPLETED`, final implementation review `APPROVE`
- The durable BRDT recommendation is already locked in `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_preflight_summary.md` as `defer_after_preflight`; this umbrella item must not relitigate that recommendation unless the checked-in summary is missing or internally inconsistent.

## Implementation Architecture

1. **Completion Audit**
   - Owns prerequisite-state confirmation across the five BRDT child items and their checked-in authorities.
2. **Discoverability Audit**
   - Owns the check that the durable summary and recommendation can be found through the intended index surfaces without reopening raw artifacts.
3. **Conditional Repair**
   - Owns any minimal index or roadmap-note edit, but only if the discoverability audit proves one is necessary.

## File and Artifact Targets

### Mandatory Source Inputs To Audit

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_operator_validation_report.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_dataset_preflight.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_task_adapters.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_preflight_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- `docs/index.md`
- `docs/backlog/index.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-brdt-operator-validation/implementation-phase/final_implementation_state.txt`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-brdt-operator-validation/implementation-phase/final_implementation_review_decision.txt`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-brdt-operator-validation/implementation-phase/final_execution_report_path.txt`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-brdt-operator-validation/implementation-phase/final_checks_report_path.txt`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-brdt-dataset-preflight/implementation-phase/final_implementation_state.txt`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-brdt-dataset-preflight/implementation-phase/final_implementation_review_decision.txt`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-brdt-dataset-preflight/implementation-phase/final_execution_report_path.txt`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-brdt-dataset-preflight/implementation-phase/final_checks_report_path.txt`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-29-brdt-task-adapters/implementation-phase/final_implementation_state.txt`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-29-brdt-task-adapters/implementation-phase/final_implementation_review_decision.txt`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-29-brdt-task-adapters/implementation-phase/final_execution_report_path.txt`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-29-brdt-task-adapters/implementation-phase/final_checks_report_path.txt`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/2/items/2026-04-29-brdt-four-row-preflight/implementation-phase/final_implementation_state.txt`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/2/items/2026-04-29-brdt-four-row-preflight/implementation-phase/final_implementation_review_decision.txt`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/2/items/2026-04-29-brdt-four-row-preflight/implementation-phase/final_execution_report_path.txt`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/2/items/2026-04-29-brdt-four-row-preflight/implementation-phase/final_checks_report_path.txt`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/3/items/2026-04-29-brdt-preflight-summary-promotion-decision/implementation-phase/final_implementation_state.txt`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/3/items/2026-04-29-brdt-preflight-summary-promotion-decision/implementation-phase/final_implementation_review_decision.txt`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/3/items/2026-04-29-brdt-preflight-summary-promotion-decision/implementation-phase/final_execution_report_path.txt`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/3/items/2026-04-29-brdt-preflight-summary-promotion-decision/implementation-phase/final_checks_report_path.txt`

### Conditional Repo Targets

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - edit only if the BRDT four-row row no longer points readers to `brdt_preflight_summary.md`
- `docs/index.md`
  - edit only if the audit shows the candidate-lane summary is not discoverable through the intended BRDT-specific surfaces
- `docs/backlog/index.md`
  - edit only if the umbrella item or BRDT chain description is stale enough to hide the durable summary/recommendation
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
  - edit only as a last resort if a discoverability gap cannot be repaired on the index surfaces above without misleading future readers

### Preferred Packaging

- Verification notes and any no-op discoverability rationale should live in the workflow-owned execution report under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-candidate-preflight/`.
- If the audit finds no discoverability gap, prefer leaving repo docs unchanged and recording the explicit “no update required” outcome in the execution report.

## Closeout Rule

Use this item to confirm and package discoverability, not to create new BRDT authority.

- If all five child items are complete and the summary/recommendation are already discoverable through the intended indexes, the correct repo outcome is **no documentation change**.
- If the child items are complete but one or more intended discoverability surfaces are stale or missing, make the smallest edit that restores discoverability while preserving BRDT’s candidate-only boundary.
- If a required child-item authority is missing, unreadable, or contradicts the checked-in summary after one narrow diagnosis, stop and surface that concrete blocker. Do not regenerate BRDT artifacts from this umbrella closeout.

## Execution Tranches

### Tranche 1: Confirm The Child-Item Completion Chain

**Purpose:** Prove that the umbrella closeout is acting on completed BRDT work rather than guessing.

- [ ] Read the BRDT candidate-lane design only for scope and non-goal boundaries.
- [ ] Read the four child-item summaries plus `brdt_preflight_summary.md`.
- [ ] Confirm from the backlog-drain state files that each split BRDT item ended `COMPLETED` with final implementation review `APPROVE`.
- [ ] Confirm from `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` that BRDT is not recorded as a completed roadmap tranche, blocked roadmap tranche, or paper-pillar promotion update.
- [ ] Confirm the summary authority records one explicit recommendation token and that the current locked recommendation is still `defer_after_preflight`.
- [ ] Resolve the final execution-report and final checks-report paths for each child item and copy those resolved lineage pointers into the umbrella execution report.
- [ ] If any child-item authority is missing or incomplete, stop after one narrow path/state diagnosis and record the exact blocker rather than widening scope.

**Verification for Tranche 1**

- [ ] **Blocking:** prerequisite state files, final implementation-review decision files, checked-in summary docs, and the progress-ledger non-promotion check agree on the child-item completion chain and candidate-only scope.
- [ ] **Supporting:** capture the resolved final execution-report and final checks-report paths for each child item in the execution report so later readers can trace the lineage without reopening the workflow state tree manually.

### Tranche 2: Audit Discoverability Surfaces

**Purpose:** Determine whether the durable summary and recommendation are already discoverable enough to close the lane.

- [ ] Audit `paper_evidence_index.md` for the BRDT four-row row and confirm it points to `brdt_preflight_summary.md` with the candidate-only, decision-support boundary intact.
- [ ] Audit `docs/backlog/index.md` and confirm the BRDT chain and umbrella-item description still match the split-item workflow and closeout purpose.
- [ ] Audit `docs/index.md` only to verify whether the current BRDT-specific entries are sufficient; do not assume the top-level docs hub must list the candidate-lane summary if `paper_evidence_index.md` is already the intended discovery surface.
- [ ] Decide whether any repo doc edit is actually required:
  - if yes, record the exact stale or missing surface;
  - if no, record that the summary is already discoverable and no repo change is needed.

**Verification for Tranche 2**

- [ ] **Blocking:** the audit must explicitly answer where a later reader should discover the BRDT summary/recommendation.
- [ ] **Blocking:** any proposed edit must preserve the rule that BRDT remains additive candidate work and not a replacement manuscript pillar.
- [ ] **Supporting:** prefer `paper_evidence_index.md` as the summary discovery surface unless the audit finds that surface insufficient or stale.

### Tranche 3: Apply Only Minimal Discoverability Repairs

**Purpose:** Fix genuine discoverability gaps without reopening BRDT scope.

- [ ] If Tranche 2 found no gap, make no repo doc edits and move directly to final verification.
- [ ] If Tranche 2 found a gap, update only the smallest necessary surface:
  - `paper_evidence_index.md` first;
  - `docs/backlog/index.md` second if the chain/umbrella description is stale;
  - `docs/index.md` only if the candidate summary truly needs direct top-level discovery.
- [ ] Do not add new BRDT design content, recommendation logic, or follow-up backlog scope while repairing discoverability.
- [ ] Do not edit the roadmap unless every smaller repair would still leave later readers unable to discover the durable summary/recommendation.

**Verification for Tranche 3**

- [ ] **Blocking:** any repo edit is limited to discoverability wording/indexing and does not change BRDT scientific scope, recommendation outcome, or claim boundary.
- [ ] **Supporting:** if a no-op outcome is selected, state that explicitly in the execution report so the umbrella item still leaves an auditable closeout record.

### Tranche 4: Run Deterministic Closeout Checks

**Purpose:** Prove the umbrella item closed the lane cleanly.

- [ ] Run the backlog-required deterministic check from the selected item context.
- [ ] Run a stronger closeout audit that verifies:
  - the five child items remain `COMPLETED`;
  - the five child items retain final implementation review `APPROVE`;
  - `progress_ledger.json` does not elevate BRDT into a completed roadmap tranche, blocked roadmap tranche, or paper-pillar promotion update;
  - `brdt_preflight_summary.md` exists;
  - exactly one approved recommendation token appears in that summary;
  - `paper_evidence_index.md` references `brdt_preflight_summary.md`;
  - `docs/backlog/index.md` still contains the BRDT umbrella row.
- [ ] If implementation changed only docs, stop after the deterministic doc/state checks.
- [ ] Do not run BRDT pytest selectors or compile checks unless implementation had to touch executable code, which should be unnecessary for this item.

**Verification for Tranche 4**

- [ ] **Blocking:** required backlog check command:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing BRDT candidate design: {missing}")
print("brdt candidate design present")
PY
```

- [ ] **Blocking:** stronger closeout consistency check:

```bash
python - <<'PY'
from pathlib import Path

summary = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_preflight_summary.md")
paper_index = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md")
backlog_index = Path("docs/backlog/index.md")
progress_ledger = Path("state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json")

required_state_files = [
    Path("state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-brdt-operator-validation/implementation-phase/final_implementation_state.txt"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-brdt-dataset-preflight/implementation-phase/final_implementation_state.txt"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-29-brdt-task-adapters/implementation-phase/final_implementation_state.txt"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/2/items/2026-04-29-brdt-four-row-preflight/implementation-phase/final_implementation_state.txt"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/3/items/2026-04-29-brdt-preflight-summary-promotion-decision/implementation-phase/final_implementation_state.txt"),
]
required_review_files = [
    Path("state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-brdt-operator-validation/implementation-phase/final_implementation_review_decision.txt"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-brdt-dataset-preflight/implementation-phase/final_implementation_review_decision.txt"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-29-brdt-task-adapters/implementation-phase/final_implementation_review_decision.txt"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/2/items/2026-04-29-brdt-four-row-preflight/implementation-phase/final_implementation_review_decision.txt"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/3/items/2026-04-29-brdt-preflight-summary-promotion-decision/implementation-phase/final_implementation_review_decision.txt"),
]
required_lineage_path_files = [
    Path("state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-brdt-operator-validation/implementation-phase/final_execution_report_path.txt"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-brdt-operator-validation/implementation-phase/final_checks_report_path.txt"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-brdt-dataset-preflight/implementation-phase/final_execution_report_path.txt"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-brdt-dataset-preflight/implementation-phase/final_checks_report_path.txt"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-29-brdt-task-adapters/implementation-phase/final_execution_report_path.txt"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-29-brdt-task-adapters/implementation-phase/final_checks_report_path.txt"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/2/items/2026-04-29-brdt-four-row-preflight/implementation-phase/final_execution_report_path.txt"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/2/items/2026-04-29-brdt-four-row-preflight/implementation-phase/final_checks_report_path.txt"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/3/items/2026-04-29-brdt-preflight-summary-promotion-decision/implementation-phase/final_execution_report_path.txt"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/3/items/2026-04-29-brdt-preflight-summary-promotion-decision/implementation-phase/final_checks_report_path.txt"),
]

missing = [
    str(p)
    for p in [summary, paper_index, backlog_index, progress_ledger, *required_state_files, *required_review_files, *required_lineage_path_files]
    if not p.exists()
]
if missing:
    raise SystemExit(f"missing closeout inputs: {missing}")

bad_states = [str(p) for p in required_state_files if p.read_text().strip() != "COMPLETED"]
if bad_states:
    raise SystemExit(f"non-completed BRDT prerequisite states: {bad_states}")

bad_reviews = [str(p) for p in required_review_files if p.read_text().strip() != "APPROVE"]
if bad_reviews:
    raise SystemExit(f"non-approved BRDT prerequisite implementation reviews: {bad_reviews}")

lineage_targets = [Path(p.read_text().strip()) for p in required_lineage_path_files]
missing_lineage_targets = [str(p) for p in lineage_targets if not p.exists()]
if missing_lineage_targets:
    raise SystemExit(f"missing child-item lineage targets: {missing_lineage_targets}")

summary_text = summary.read_text()
tokens = [
    "promote_to_evidence_amendment_plan",
    "defer_after_preflight",
    "reject_for_current_manuscript",
]
hits = [token for token in tokens if token in summary_text]
if len(hits) != 1:
    raise SystemExit(f"expected exactly one recommendation token, found {hits}")

paper_index_text = paper_index.read_text()
if "brdt_preflight_summary.md" not in paper_index_text:
    raise SystemExit("paper_evidence_index.md does not reference brdt_preflight_summary.md")

backlog_index_text = backlog_index.read_text()
if "2026-04-29-brdt-candidate-preflight" not in backlog_index_text:
    raise SystemExit("docs/backlog/index.md is missing the BRDT umbrella item row")

ledger_text = progress_ledger.read_text().lower()
if any(term in ledger_text for term in ["brdt", "born_rytov", "candidate lane", "candidate_lane"]):
    raise SystemExit("progress_ledger.json unexpectedly contains BRDT promotion state; inspect before closing the umbrella item")

print("brdt umbrella closeout inputs and discoverability surfaces are consistent")
PY
```

## Completion Criteria

- The five split BRDT backlog items are confirmed complete from checked-in state files and durable summaries.
- Those five child items are also confirmed to retain final implementation review `APPROVE`, and their resolved execution/check lineage is copied into the umbrella execution report.
- The umbrella item explicitly confirms where later readers discover the BRDT durable summary and recommendation.
- Any repo edit, if needed, is limited to discoverability repair only.
- If no discoverability gap exists, the execution report states that explicitly and repo docs remain unchanged.
- The backlog-required deterministic check passes.
- The stronger closeout consistency check passes, including the progress-ledger non-promotion audit.
- No BRDT functionality, reruns, roadmap promotion, or manuscript-evidence expansion was performed from this item.
