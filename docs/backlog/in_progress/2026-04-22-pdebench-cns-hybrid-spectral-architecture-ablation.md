---
priority: 17
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation/execution_plan.md
check_commands:
  - pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
  - python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
prerequisites: []
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
---

# Backlog Item: Add CNS Hybrid-Spectral Architecture Ablation

## Objective

- Run a focused PDEBench `2d_cfd_cns` architecture ablation for the
  Hybrid-spectral family under the fixed canonical CNS shell so the repo can
  separate spectral-family architectural effects from already-settled shell
  choices.

## Scope

- Keep the current CNS shell fixed to:
  - `hybrid_skip_connections=True`
  - `hybrid_skip_style=add`
  - `hybrid_upsampler=pixelshuffle`
- Keep the current capped CNS contract fixed:
  - `512 / 64 / 64` trajectories
  - `8` windows per trajectory
  - `history_len=2`
  - `mse`
  - batch size `4`
- Cover Hybrid-spectral axes only:
  - spectral weight sharing
  - spectral bottleneck depth
- Use the larger-cap `1024 / 128 / 128` confirmation pass only for finalist
  rows.

## Notes for Reviewer

- This is **CNS only**. Do not widen it into CDI/ptycho.
- This is **Hybrid-spectral only**. Do not turn it into a local-vs-spectral
  family compare in the first tranche.
- Skip connections and upsampling mode are fixed shell controls here, not
  primary ablation axes.
- Keep Markov `history_len=1`, physics regularization, and higher-mode spectral
  runs as separate ablation families.
