---
priority: 14
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/execution_plan.md
check_commands:
  - pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
  - python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
signals_for_selection:
  - Current CNS evidence is mostly capped decision-support; the paper needs an explicit decision before spending full-training compute or packaging bounded claims.
  - This item prevents the queue from mixing capped rows, full-training rows, and claim language without an auditable contract decision.
---

# Backlog Item: Decide CNS Paper Evidence Contract

## Objective

- Decide whether the paper will use full-training CNS benchmark rows or a
  bounded capped CNS table, and record the exact contract before any paper
  result claims are drafted.

## Scope

- Audit completed CNS rows and identify the strongest current Hybrid-family,
  FNO, and U-Net/CNN-style core candidates, plus authored FFNO if it is
  available under the same local CNS contract.
- Estimate the compute needed for same-contract full-training rows on the
  official `2d_cfd_cns` file under the current split/history/normalization
  contract.
- Choose one of two explicit paths:
  `full_training_paper_benchmark` or `bounded_capped_decision_support`.
- Set an authored-FFNO inclusion cutoff before row locking. The decision must
  say whether authored FFNO is available under the same local CNS contract by
  that cutoff, or whether it will be recorded as `blocked` /
  `not_protocol_compatible` with CNS claims limited accordingly.
- If full-training is selected, write the required row list, training split,
  validation/test split, epoch/budget policy, model profiles, metric schema,
  runtime/provenance requirements, and stop/failure criteria.
- If bounded capped evidence is selected, write the exact cap, frozen row
  manifests, claim limitations, and missing full-training caveat.
- Write the decision as a durable document under
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/` and update downstream backlog items
  if the decision changes their row requirements.

## Notes for Reviewer

- Do not let a capped row satisfy a full-training benchmark claim.
- Do not choose full-training by implication. The decision must include an
  explicit compute and deadline rationale.
- Do not leave FFNO in an ambiguous "when available" state. The cutoff and claim
  impact must be explicit before the row-lock item runs.
- Do not reopen broad architecture search in this item; use existing capped
  studies only to choose which rows deserve the locked paper contract.
