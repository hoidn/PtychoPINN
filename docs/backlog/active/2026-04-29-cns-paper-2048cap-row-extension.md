---
priority: 65
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md
check_commands:
  - pytest -q tests/studies/test_pdebench_cfd_cns_metrics.py tests/studies/test_pdebench_image128_runner.py
  - python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
prerequisites:
  - 2026-04-29-paper-evidence-package-audit
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
signals_for_selection:
  - This is a later, long-running CNS evidence extension, not a blocker for the current 1024-cap CNS table/figure bundle.
  - Select only after the main CNS/CDI evidence package audit is complete or explicitly reprioritized.
  - The goal is same-cap 2048 CNS row promotion for comparators, not mixed-cap headline reporting.
---

# Backlog Item: Extend CNS Paper Rows To 2048 Cap

## Objective

- Promote the CNS comparator bundle to a same-contract `2048 / 256 / 256`
  capped row set as a later evidence-strengthening pass, after the current
  `1024 / 128 / 128` CNS table/figure bundle and paper evidence audit are no
  longer blocked.

## Scope

- Reuse the completed `2048 / 256 / 256` spectral-family scaling rows where
  they satisfy the locked CNS local contract.
- Recover or run same-cap `2048 / 256 / 256`, `history_len=2`, `40`-epoch,
  `max_windows_per_trajectory=8` rows for:
  - `author_ffno_cns_base`
  - `fno_base`
  - `unet_strong`
- Keep the current paper-bundle `1024 / 128 / 128` row set as the immediate
  manuscript-supporting evidence unless this later item finishes and passes the
  same provenance and row-status checks.
- Do not mix `2048` rows with `1024` or `512` rows in one headline CNS table.
- Publish a separate 2048-cap extension summary and, only if all required rows
  complete under the same contract, a replacement CNS table payload.

## Notes for Reviewer

- This item is intentionally lower priority because the `2048 / 256 / 256`
  comparator rows are expected to be long-running.
- Do not let this item delay the current 1024-cap CNS table/figure bundle,
  minimum CDI paper table, or paper evidence audit.
- If any comparator is missing or incompatible at 2048, record a row-level
  blocker and keep the 1024-cap bundle as the current paper package.
