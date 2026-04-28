# Hybrid-Spectral CNS 2048-Cap Scaling Seed Plan

## Status

Seed plan for backlog selection. The NeurIPS backlog workflow may replace this
with a fresh reviewed execution plan before implementation.

## Backlog Item

- ID: `2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap`
- Related roadmap phase: `phase-2-pdebench-128x128-image-suite`
- Prerequisite: `2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation`

## Goal

Check whether the hybrid-spectral CNS finalists keep separating as the capped
training set grows from `512` to `1024` to `2048` trajectories.

## Scope

- Task: PDEBench `2d_cfd_cns`
- Dataset: official `128x128` CNS file
- Split cap: `2048 / 256 / 256`
- `history_len=2`
- `max_windows_per_trajectory=8`
- loss: `mse`
- batch size: `4`
- epochs: `40`
- rows:
  - `spectral_resnet_bottleneck_base`
  - `spectral_resnet_bottleneck_shared_blocks10`

Do not rerun the existing `512 / 64 / 64` or `1024 / 128 / 128` rows. Treat
those as frozen references for trend comparison only.

## Execution Notes

- Use the existing PDEBench image128 runner and CNS reporting path.
- Preserve the same metric family used by the completed architecture ablation:
  `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, and
  `fRMSE_high`.
- Report absolute metrics and deltas for `512 -> 1024` and `1024 -> 2048`.
- Keep the conclusion bounded to capped scaling evidence. Do not promote either
  profile to a paper/default claim from this item alone.

## Checks

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```
