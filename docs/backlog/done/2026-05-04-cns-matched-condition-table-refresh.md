---
priority: 31
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cns-matched-condition-table-refresh/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/backlog/done/2026-04-29-cns-paper-2048cap-row-extension.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_2048cap_extension_summary.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history5_comparator_gap_fill_summary.md"),
        Path("scripts/studies/paper_results_refresh.py"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing CNS matched-table inputs: {missing}")
    print("CNS matched-table inputs present")
    PY
  - pytest -q tests/studies/test_paper_results_refresh.py tests/studies/test_pdebench_cfd_cns_metrics.py
  - python -m compileall -q scripts/studies
prerequisites:
  - 2026-05-04-cns-history5-comparator-gap-fill
  - 2026-04-29-cns-paper-2048cap-row-extension
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
signals_for_selection:
  - Reviewer-style feedback flagged the current best-observed CNS table as not a fair model comparison.
  - The h5 comparator gap-fill creates enough evidence to choose a same-history CNS table instead of mixing rows.
  - This item controls manuscript CNS table authority and should run before further optional CNS context is promoted.
---

# Backlog Item: CNS Matched-Condition Table Refresh

## Objective

- Refresh the PDEBench CNS manuscript table so the headline model ranking uses
  one matched condition instead of best-observed rows from different history
  lengths, caps, or variants.

## Scope

- Consume the completed `history_len=5` comparator gap-fill evidence and the
  current `history_len=2`, `2048 / 256 / 256`, `40`-epoch capped CNS authority.
- Prefer an all-row `history_len=5` headline table only if
  `author_ffno_cns_base`, `spectral_resnet_bottleneck_base` / SRU-Net*,
  `fno_base`, and `unet_strong` all have complete same-condition rows.
- Otherwise keep the locked `history_len=2`, `2048 / 256 / 256` authority as
  the headline CNS table.
- Preserve best-observed or mixed-condition rows as appendix/context evidence,
  not as the headline model-ranking table.
- Do not rerun CNS training in this item.

## Required Interpretation

- The headline CNS table must compare models under one history length, cap,
  epoch budget, normalization, and metric schema.
- Best-observed rows are useful context, but they must not be formatted as a
  fair head-to-head ranking unless the conditions match.
- Keep the existing bounded capped claim boundary unless a later roadmap update
  explicitly reopens full-training CNS evidence.

## Outputs

- A durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md`
- Refreshed CNS JSON, CSV, TeX, and manuscript table payloads for the selected
  matched condition.
- Updated paper evidence index, evidence matrix, model variant index, and study
  index entries.

## Notes For Reviewer

- Reject plans that choose the numerically best row per model for the headline
  table when the row conditions differ.
- Reject plans that rerun CNS rows instead of using the completed h5 and h2
  authorities already available.
- If the h5 set is incomplete or internally inconsistent, require the h2 2048
  same-condition authority to remain the headline table and record h5 as
  adjacent context.
