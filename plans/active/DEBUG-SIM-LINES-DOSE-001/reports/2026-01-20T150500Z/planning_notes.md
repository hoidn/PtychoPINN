# 2026-01-20T150500Z Planning Notes
- Checked off Phase A0/A1b/A2 and C4d in implementation.md using artifacts from 2026-01-16 and 2026-01-20 hubs.
- Documented rationale for waiving A1b (legacy run redundant) and summarized the C4d analyzer results (gs1 best scalar≈1.878).
- Scoped C4e: add amplitude-rescale hook (runner + sim_lines pipeline) plus analyzer reruns; guard with pytest selector `tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke`.
