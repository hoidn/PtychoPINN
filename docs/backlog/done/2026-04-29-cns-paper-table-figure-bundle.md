---
priority: 22
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/execution_plan.md
check_commands:
  - pytest -q tests/studies/test_pdebench_cfd_cns_metrics.py tests/studies/test_pdebench_image128_runner.py
  - python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
prerequisites:
  - 2026-04-29-cns-paper-benchmark-rows
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
signals_for_selection:
  - The CNS table should upgrade to a `1024 / 128 / 128` capped row bundle if the missing same-cap rows can be recovered or run inside this item.
  - The paper needs a single CNS table and visual comparison bundle with shared sample IDs, field scales, error scales, and row-status labels.
  - Metrics alone are insufficient; the bundle must include regenerable source arrays and figures for density, velocity, and pressure fields.
---

# Backlog Item: Build CNS Paper Table And Figure Bundle

## Objective

- Upgrade the locked CNS evidence to a same-cap `1024 / 128 / 128` bounded
  bundle when possible, then merge the accepted CNS rows into paper-ready metric
  tables, visual comparison figures, source-array manifests, and a durable
  claim-boundary summary.

## Scope

- Consume only rows accepted by the CNS row-lock summary.
- Prefer a same-contract `1024 / 128 / 128`, `history_len=2`, `40`-epoch,
  `max_windows_per_trajectory=8` capped bundle over the earlier `512 / 64 / 64`
  row lock.
- Reuse the existing `1024 / 128 / 128` spectral-family row when it satisfies
  the same local CNS contract:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z`.
- Before table assembly, inventory and either recover or run same-cap rows for:
  - `author_ffno_cns_base`
  - `fno_base`
  - `unet_strong`
- If any required `1024 / 128 / 128` row cannot be recovered or run, write a
  row-level blocker and fall back to the earlier complete `512 / 64 / 64`
  bounded row lock rather than mixing caps in one headline table.
- Produce table-ready JSON, CSV, and TeX for `err_nRMSE`, `err_RMSE`,
  `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`, parameter count,
  runtime, hardware, split/cap/full-training status, and row status.
- Generate fixed-sample visual comparisons for `density`, `Vx`, `Vy`, and
  `pressure`: ground truth, prediction, and error panels for every accepted
  row.
- Use shared sample IDs, shared color scales per field, shared error scales per
  quantity, and source arrays sufficient to regenerate every figure.
- Write a durable CNS paper-bundle summary under
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/` and update `docs/studies/index.md`
  if new reusable study artifacts are produced.

## Notes for Reviewer

- Do not silently promote capped rows to benchmark-performance claims.
- Do not mix `1024 / 128 / 128` spectral rows with `512 / 64 / 64`
  FFNO/FNO/U-Net rows in one headline table.
- Do not publish a merged table if required metric fields are missing without
  explicit `benchmark_incomplete` or row-level missing-field labels.
- Do not let visual collation pick different sample IDs or scales per model.
