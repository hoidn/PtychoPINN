---
priority: 13
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-22-pdebench-gnot-paper-default-cns-compare/execution_plan.md
check_commands:
  - python -m pytest tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py -q
  - python -m compileall -q scripts/studies/pdebench_image128
prerequisites: []
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
---

# Backlog Item: Add Paper-Default GNOT CNS Compare

## Objective
- Rerun `gnot_cns_base` on the existing capped PDEBench `2d_cfd_cns` contract
  using the patched paper-style GNOT recipe, then compare it directly against
  the current spectral anchor.

## Scope
- Use the validated `ptycho311_2` CUDA+DGL environment rather than the default
  repo env.
- Keep the local CNS compare contract fixed: `128x128`, `history_len=2`,
  `512/64/64` trajectories, `8` windows per trajectory, batch size `4`.
- Run the paper-default `gnot_cns_base` row for `40` epochs and update the
  durable summary/findings if the result changes the current interpretation.

## Notes for Reviewer
- This item is specifically about the already integrated official GNOT source,
  not a new external-model integration.
- Do not reuse the earlier local fairness-probe recipe; the point of this item
  is to answer whether the patched paper-style recipe changes the result.
- Keep output roots timestamped and fresh, and pin exact anchor artifact roots
  in the summary if any non-GNOT rows are compared numerically.
