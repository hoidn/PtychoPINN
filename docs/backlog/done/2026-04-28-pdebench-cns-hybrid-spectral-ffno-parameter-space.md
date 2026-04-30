---
priority: 21
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-28-pdebench-cns-hybrid-spectral-ffno-parameter-space/execution_plan.md
check_commands:
  - pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
  - python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
prerequisites:
  - 2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation
  - 2026-04-27-pdebench-ffno-convolutional-features-cns
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
---

# Backlog Item: Explore CNS Hybrid-Spectral To FFNO Parameter Space

## Objective

- Run the PDEBench CNS half of the Hybrid-spectral to FFNO architecture-space
  study under the capped `2d_cfd_cns` decision-support contract.

## Scope

- Compare selected intermediate architecture points between the current
  Hybrid-spectral finalist family and FFNO-like variants on CNS only.
- Include one-axis-at-a-time encoder/downsampling, decoder, and bottleneck
  variants where each row has an attributable architecture-family change.
- Reuse the verified CNS `history_len=2` MSE anchor and the current capped
  image-suite contract; do not treat these runs as full-training benchmark
  evidence.

## Notes for Reviewer

- This is the Phase 2 portion of the former mixed CNS/CDI parameter-space item.
- Do not run CDI/ptycho experiments from this backlog item.
- Keep the final summary local to CNS and explicitly mark any conclusion as
  capped decision-support evidence unless a later full-training item reruns the
  selected rows on the full available training split.
