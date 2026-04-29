---
priority: 126
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md
check_commands:
  - |
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
prerequisites:
  - 2026-04-29-brdt-operator-validation
  - 2026-04-29-brdt-dataset-preflight
  - 2026-04-29-brdt-task-adapters
  - 2026-04-29-brdt-four-row-preflight
  - 2026-04-29-brdt-preflight-summary-promotion-decision
related_roadmap_phases:
  - candidate-brdt-preflight
signals_for_selection:
  - This umbrella item closes the BRDT candidate lane after the concrete operator, dataset, adapter, four-row, and summary items have completed.
  - It keeps BRDT additional to CDI/CNS and prevents the broad design from being selected as a vague implementation scope.
  - It should only run after the split BRDT preflight chain has produced a durable promotion, deferral, or rejection recommendation.
---

# Backlog Item: BRDT Candidate Lane Umbrella Closeout

## Objective

- Close the BRDT candidate lane after the split preflight items complete, and
  confirm that the durable summary and promotion decision are discoverable.

## Scope

- Consume
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md`.
- Confirm the completed child items produced:
  - operator validation;
  - dataset preflight;
  - task adapters;
  - four-row decision-support preflight;
  - durable summary and promotion/deferral/rejection recommendation.
- Update indexes or roadmap notes only if the completed child items created new
  durable artifacts that are not discoverable.
- Do not implement new BRDT functionality in this umbrella closeout.

## Notes for Reviewer

- Keep BRDT additional to CDI `lines128` and PDEBench CNS. It is not a
  replacement pillar.
- Treat this as administrative closeout only. If substantial BRDT work remains,
  add a new concrete backlog item instead of expanding this umbrella.
- Paper-table promotion still requires a later checked-in roadmap or
  evidence-package amendment.
