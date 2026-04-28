# CNS Spectral Modes 24 Convergence Seed Plan

## Status

Seed plan for backlog selection. The NeurIPS backlog workflow may replace this
with a fresh reviewed execution plan before implementation.

## Backlog Item

- ID: `2026-04-28-pdebench-cns-spectral-modes24-convergence-compare`
- Related roadmap phase: `phase-2-pdebench-128x128-image-suite`
- Prerequisite: `2026-04-21-pdebench-cns-spectral-modes32-compare`

## Goal

Revisit CNS spectral mode count under a convergence-oriented budget so the
workflow can separate mode-count value from under-training.

## Scope

- Task: PDEBench `2d_cfd_cns`
- Dataset: official `128x128` CNS file
- Split cap: `1024 / 128 / 128`
- `history_len=2`
- `max_windows_per_trajectory=8`
- loss: `mse`
- batch size: `16`, or the largest smaller identical batch size that lets both
  rows run if `16` has a concrete runtime blocker
- rows:
  - baseline: `fno_modes=12`, `spectral_bottleneck_modes=12`
  - candidate: `fno_modes=24`, `spectral_bottleneck_modes=24`

Do not widen this into a full mode sweep, a `2048` scaling item, FFNO/CDI work,
history-length changes, or physics regularization.

## Execution Notes

- Define the convergence stop standard before launching long runs.
- Report train loss trajectory, late-epoch slope or last-window delta, final
  eval metrics, and whether either row is still materially improving at stop.
- If both rows are still materially improving, mark the result inconclusive
  instead of promoting either profile.
- Keep completed modes-32 results as context only; the primary comparison is
  fresh converged-budget `12/12` versus `24/24`.

## Checks

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```
